#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spider_ocr_pipeline_v2.py (UPDATED)

Mejoras incluidas en esta versión:
- Detecta y evita "stalls" en páginas que requieren autenticación (HTTP 401/403/407)
  o contienen formularios de login (heurística), marcándolas como "auth_required" y
  continuando con la cola de URLs.
- Soporta que el argumento --seeds sea un archivo (seeds.txt) o un directorio que contenga
  múltiples ficheros .txt con seeds; si se pasa un directorio, concatena todos los seeds
  sin duplicados y los encola.
- Añadido flag CLI --skip-auth (boolean) para forzar comportamiento de saltado de páginas
  protegidas; por defecto activo.
- Mejor manejo de timeouts y detección rápida de páginas que esperan interacción (JS/login).
- Correcciones menores y robustez en el enqueue y en la escritura de ficheros.

Mantiene el resto de funcionalidades definidas anteriormente (OCR, PDF text extraction,
DB dedupe, normalization, etc.)
"""
from __future__ import annotations
import os
import sys
import re
import time
import json
import math
import uuid
import queue
import hashlib
import shutil
import logging
import argparse
import asyncio
import aiosqlite
import aiofiles
import aiohttp
import ftfy
import unicodedata
import concurrent.futures
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any, Set
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
from tqdm.asyncio import tqdm

# Optional sync imports used in process pool (import safe)
try:
    from pdf2image import convert_from_path
except Exception:
    convert_from_path = None
try:
    import pytesseract
except Exception:
    pytesseract = None
try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None
try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None

# Language detection
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
except Exception:
    detect = None

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("spider_ocr_v2")

# ---------------------------
# Default configuration
# ---------------------------
DEFAULT_USER_AGENT = "MiniLLM-Spider/2.0 (+https://example.com)"
DEFAULT_POPPLER_PATH = os.environ.get("POPPLER_PATH", None)  # set on Windows if needed
DEFAULT_TESSERACT_LANG = "spa"
DEFAULT_DB = "spider_meta.db"
DEFAULT_OUTDIR = "downloads"
DEFAULT_MAX_PDF_PAGES = 300

# ---------------------------
# Utilities
# ---------------------------
def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()

def safe_filename(s: str, maxlen: int = 200) -> str:
    s2 = re.sub(r"[^\w\-_\. ]+", "_", s)
    return s2[:maxlen]

# ---------------------------
# Text normalization & QC
# ---------------------------
SPANISH_WORD_RE = re.compile(r"^[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+(?:[-'][A-Za-zÁÉÍÓÚÜÑáéíóúüñ]+)*$")

def normalize_text(text: str) -> str:
    """Fix encoding, normalize unicode, remove control chars, collapse spaces."""
    if not text:
        return text
    s = ftfy.fix_text(text)
    s = unicodedata.normalize("NFKC", s)
    # Remove unusual control chars but keep \n \t
    s = "".join(ch for ch in s if (ch == "\n" or ch == "\t" or ord(ch) >= 32))
    # Collapse multiple spaces and excessive newlines
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = "\n".join(line.strip() for line in s.splitlines())
    return s.strip()

def detect_ocr_issues_sync(text: str) -> Dict[str, Any]:
    """Heuristics to detect OCR quality issues (synchronous version for process pool)."""
    stats = {}
    if not text or len(text) < 50:
        stats.update({"ok": False, "reason": "empty_or_too_short"})
        return stats

    chars = len(text)
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    num_words = len(words)
    stats["chars"] = chars
    stats["words"] = num_words

    # mojibake detection
    mojibake_patterns = re.compile(r"[ÂÃ�¿�]|Ã[¡-º]|â|Ã©|Ã³|Ã±", flags=re.UNICODE)
    mojibake_count = len(mojibake_patterns.findall(text))
    stats["mojibake_count"] = mojibake_count
    stats["mojibake_ratio"] = mojibake_count / max(1, chars)

    # fraction of Spanish-like words
    word_tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ'-]+", text)
    if word_tokens:
        spanish_like = sum(1 for w in word_tokens if SPANISH_WORD_RE.match(w))
        frac_spanish = spanish_like / len(word_tokens)
    else:
        frac_spanish = 0.0
    stats["frac_spanish_words"] = frac_spanish

    # single-letter tokens
    tokens = re.findall(r"\S+", text)
    single_letter_frac = sum(1 for t in tokens if len(t) == 1) / max(1, len(tokens))
    stats["single_letter_frac"] = single_letter_frac

    issues = []
    if frac_spanish < 0.6:
        issues.append("low_spanish_fraction")
    if stats["mojibake_ratio"] > 0.002:
        issues.append("mojibake_present")
    if single_letter_frac > 0.05:
        issues.append("many_single_letters")
    if len(tokens) < 20:
        issues.append("too_few_tokens")

    stats["issues"] = issues
    stats["ok"] = len(issues) == 0
    return stats

# ---------------------------
# Sync extraction functions (run in executor)
# ---------------------------
def extract_text_from_pdf_sync(path: str, max_pages: Optional[int]=None) -> Tuple[str, List[str]]:
    """Extract text layer via PyPDF2; returns (full_text, per_page_texts)."""
    if PdfReader is None:
        raise RuntimeError("PyPDF2 is required for PDF text extraction.")
    pages_texts = []
    try:
        reader = PdfReader(path)
        n = len(reader.pages)
        if max_pages:
            n = min(n, max_pages)
        for i in range(n):
            try:
                p = reader.pages[i]
                txt = p.extract_text() or ""
                pages_texts.append(txt)
            except Exception:
                pages_texts.append("")
    except Exception as e:
        logger.debug(f"Pdf read error {path}: {e}")
        return "", []
    full = "\n\n".join(pages_texts)
    return full, pages_texts

def ocr_pdf_pages_sync(path: str, poppler_path: Optional[str], dpi: int, max_pages: Optional[int], lang: str) -> Tuple[str, List[str]]:
    """Convert PDF to images + pytesseract OCR. Returns (full_text, per_page_texts)."""
    if convert_from_path is None or pytesseract is None:
        raise RuntimeError("pdf2image and pytesseract are required for OCRing PDFs.")
    poppler_kw = {"poppler_path": poppler_path} if poppler_path else {}
    try:
        images = convert_from_path(path, dpi=dpi, fmt="jpeg", thread_count=1, **poppler_kw)
    except Exception as e:
        logger.error(f"pdf2image conversion failed: {e}")
        return "", []
    texts = []
    if max_pages:
        images = images[:max_pages]
    for img in images:
        try:
            txt = pytesseract.image_to_string(img, lang=lang)
            texts.append(txt)
        except Exception as e:
            logger.warning(f"pytesseract error on page image: {e}")
            texts.append("")
    return "\n\n".join(texts), texts

def extract_text_from_docx_sync(path: str) -> str:
    """Extract text from .docx using python-docx"""
    if DocxDocument is None:
        raise RuntimeError("python-docx is required to extract docx files.")
    try:
        doc = DocxDocument(path)
        paras = [p.text for p in doc.paragraphs]
        return "\n".join(paras)
    except Exception as e:
        logger.error(f"docx read error {path}: {e}")
        return ""

def extract_text_from_image_sync(path: str, lang: str) -> str:
    """OCR on image file path (jpg/png)."""
    if pytesseract is None:
        raise RuntimeError("pytesseract required for OCR on images.")
    from PIL import Image
    try:
        img = Image.open(path)
        txt = pytesseract.image_to_string(img, lang=lang)
        return txt
    except Exception as e:
        logger.error(f"OCR image failed {path}: {e}")
        return ""

# ---------------------------
# Async crawling & processing
# ---------------------------
class SpiderConfig:
    def __init__(self,
                 outdir: str = DEFAULT_OUTDIR,
                 db_path: str = DEFAULT_DB,
                 user_agent: str = DEFAULT_USER_AGENT,
                 concurrency: int = 6,
                 max_depth: int = 2,
                 politeness: float = 0.5,
                 max_pages: int = DEFAULT_MAX_PDF_PAGES,
                 poppler_path: Optional[str] = DEFAULT_POPPLER_PATH,
                 ocr_lang: str = DEFAULT_TESSERACT_LANG,
                 min_words: int = 50,
                 timeout: int = 30,
                 allowed_domains: Optional[List[str]] = None,
                 skip_auth: bool = True):
        self.outdir = Path(outdir)
        self.db_path = Path(db_path)
        self.user_agent = user_agent
        self.concurrency = concurrency
        self.max_depth = max_depth
        self.politeness = politeness
        self.max_pdf_pages = max_pages
        self.poppler_path = poppler_path
        self.ocr_lang = ocr_lang
        self.min_words = min_words
        self.timeout = timeout
        self.allowed_domains = allowed_domains or []
        self.skip_auth = skip_auth

class Spider:
    def __init__(self, config: SpiderConfig, search_terms: List[str], expand_terms: bool = True):
        self.config = config
        self.search_terms = search_terms
        self.expand_terms = expand_terms
        self.session: Optional[aiohttp.ClientSession] = None
        self.db: Optional[aiosqlite.Connection] = None
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max(1, os.cpu_count()-1))
        self.loop = asyncio.get_event_loop()
        self.rate_limit_cache: Dict[str, float] = {}  # last request per domain
        self.seen_url_cache: Set[str] = set()  # in-memory fast check
        self.term_variants = self._expand_terms(search_terms) if expand_terms else set(search_terms)
        # Prepare output directories
        self.config.outdir.mkdir(parents=True, exist_ok=True)
        (self.config.outdir / "raw").mkdir(exist_ok=True)
        (self.config.outdir / "txt").mkdir(exist_ok=True)
        (self.config.outdir / "meta").mkdir(exist_ok=True)

    async def init(self):
        # Create aiohttp session and aiosqlite DB
        timeout = aiohttp.ClientTimeout(total=self.config.timeout)
        headers = {"User-Agent": self.config.user_agent}
        conn = aiohttp.TCPConnector(limit_per_host=self.config.concurrency, ssl=False)
        self.session = aiohttp.ClientSession(timeout=timeout, headers=headers, connector=conn)
        self.db = await aiosqlite.connect(str(self.config.db_path))
        await self._init_db()

    async def close(self):
        if self.session:
            await self.session.close()
        if self.db:
            await self.db.close()
        if self.executor:
            self.executor.shutdown(wait=False)

    async def _init_db(self):
        """Initialize metadata DB for tracking downloads and deduplication."""
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS urls (
                url TEXT PRIMARY KEY,
                domain TEXT,
                status TEXT,
                depth INTEGER,
                content_path TEXT,
                text_path TEXT,
                sha256 TEXT,
                words INTEGER,
                lang TEXT,
                ocr_used INTEGER DEFAULT 0,
                added_at REAL,
                last_seen REAL
            );
        """)
        await self.db.execute("""
            CREATE TABLE IF NOT EXISTS content_hashes (
                sha256 TEXT PRIMARY KEY,
                saved_text_path TEXT,
                added_at REAL
            );
        """)
        await self.db.execute("""
            CREATE INDEX IF NOT EXISTS idx_urls_domain ON urls(domain);
        """)
        await self.db.commit()

    def _expand_terms(self, terms: List[str]) -> Set[str]:
        """Produce simple morphological/alias expansions for search terms."""
        out = set()
        for t in terms:
            t0 = t.strip()
            if not t0:
                continue
            out.add(t0)
            out.add(t0.lower())
            out.add(t0.title())
            # Without accents
            t_noacc = self._strip_accents(t0)
            out.add(t_noacc)
            out.add(t_noacc.lower())
            # plural heuristic
            if not t0.endswith('s'):
                out.add(t0 + "s")
            # small stemming heuristics (remove 'ción' -> 'cion' already accent stripped)
            if t0.endswith("ción"):
                out.add(t0[:-4] + "cion")
        return out

    @staticmethod
    def _strip_accents(s: str) -> str:
        return ''.join(ch for ch in unicodedata.normalize('NFD', s) if unicodedata.category(ch) != 'Mn')

    async def seed_urls(self, seeds: List[str]):
        """Insert seeds into DB with depth=0. Accepts file or dir input handling upstream."""
        now = time.time()
        async with self.db.execute("BEGIN"):
            for url in seeds:
                url = url.strip()
                if not url:
                    continue
                parsed = urlparse(url)
                if not parsed.scheme:
                    # try to normalize
                    url = "http://" + url
                domain = urlparse(url).netloc
                try:
                    await self.db.execute(
                        "INSERT OR IGNORE INTO urls(url, domain, status, depth, added_at, last_seen) VALUES (?, ?, ?, ?, ?, ?)",
                        (url, domain, 'queued', 0, now, now)
                    )
                    self.seen_url_cache.add(url)
                except Exception as e:
                    logger.debug(f"DB insert seed failed: {e}")
            await self.db.commit()

    async def crawl(self, max_pages: Optional[int] = None, continuous: bool = False):
        """
        Main crawl loop.
        If continuous==True, it will keep polling the DB for new queued URLs.
        """
        sem = asyncio.Semaphore(self.config.concurrency)
        tasks = []
        processed = 0
        pbar = None
        if max_pages:
            pbar = tqdm(total=max_pages, desc="Pages")
        while True:
            # fetch next url to process
            row = await self._get_next_queued_url()
            if row is None:
                if continuous:
                    await asyncio.sleep(2.0)
                    continue
                else:
                    break
            url, domain, depth = row
            # enforce allowed domains
            if self.config.allowed_domains and not any(domain.endswith(d) for d in self.config.allowed_domains):
                logger.debug(f"Skipping {url} (domain not allowed).")
                await self._update_url_status(url, "skipped_domain")
                if pbar:
                    pbar.update(1)
                continue

            async with sem:
                task = asyncio.create_task(self._process_url_task(url, depth))
                tasks.append(task)
            processed += 1
            if pbar:
                pbar.update(1)
            if max_pages and processed >= max_pages:
                break

        if tasks:
            await asyncio.gather(*tasks)
        if pbar:
            pbar.close()

    async def _get_next_queued_url(self) -> Optional[Tuple[str, str, int]]:
        """Pop one queued URL (FIFO) from DB."""
        async with self.db.execute("SELECT url, domain, depth FROM urls WHERE status='queued' ORDER BY added_at LIMIT 1") as cur:
            row = await cur.fetchone()
            if not row:
                return None
            url, domain, depth = row
            # mark it as in-progress
            await self.db.execute("UPDATE urls SET status=?, last_seen=? WHERE url=?", ("in_progress", time.time(), url))
            await self.db.commit()
            return url, domain, depth

    async def _update_url_status(self, url: str, status: str, **kwargs):
        """Update URL row with status and optional fields."""
        fields = []
        values = []
        for k, v in kwargs.items():
            fields.append(f"{k} = ?")
            values.append(v)
        fields_sql = ", ".join(fields) + ", status = ?, last_seen = ?"
        values.extend([status, time.time(), url])
        sql = f"UPDATE urls SET {fields_sql} WHERE url = ?"
        try:
            await self.db.execute(sql, values)
            await self.db.commit()
        except Exception as e:
            logger.debug(f"DB update failed: {e}")

    async def _process_url_task(self, url: str, depth: int):
        """Wrapper for processing a single URL: download, extract, normalize, save, enqueue links."""
        try:
            domain = urlparse(url).netloc
            # rate-limit per domain
            await self._respect_rate_limit(domain)
            logger.info(f"[depth {depth}] Fetching: {url}")
            headers = {"User-Agent": self.config.user_agent}
            try:
                async with self.session.get(url, headers=headers) as resp:
                    status = resp.status
                    content_type = resp.headers.get("Content-Type", "").lower()
                    # If auth required and skip_auth configured -> mark and return
                    if self.config.skip_auth and status in (401, 403, 407):
                        logger.info(f"Skipping {url} due to HTTP status {status} (auth required).")
                        await self._update_url_status(url, "auth_required", last_seen=time.time())
                        return

                    body = await resp.read()
            except asyncio.TimeoutError:
                logger.warning(f"Timeout fetching {url}")
                await self._update_url_status(url, "timeout")
                return
            except Exception as e:
                logger.warning(f"HTTP GET failed {url}: {e}")
                await self._update_url_status(url, "failed_http")
                return

            # Save raw bytes
            parsed = urlparse(url)
            domain_safe = parsed.netloc.replace(":", "_")
            safe = safe_filename(parsed.path or parsed.netloc)
            raw_path = self.config.outdir / "raw" / domain_safe
            raw_path.mkdir(parents=True, exist_ok=True)
            file_ext = ""
            if "pdf" in content_type or url.lower().endswith(".pdf"):
                file_ext = ".pdf"
            elif any(url.lower().endswith(ext) for ext in (".jpg", ".jpeg", ".png")) or "image" in content_type:
                file_ext = ".img"
            elif url.lower().endswith(".docx") or "officedocument.wordprocessingml" in content_type:
                file_ext = ".docx"
            elif url.lower().endswith(".txt") or "text/plain" in content_type:
                file_ext = ".txt"
            else:
                # Unknown: treat as html
                file_ext = ".html"

            filename = f"{safe or 'index'}_{int(time.time())}{file_ext}"
            file_path = raw_path / filename
            try:
                async with aiofiles.open(file_path, "wb") as f:
                    await f.write(body)
            except Exception as e:
                logger.error(f"Failed to write raw file {file_path}: {e}")
                await self._update_url_status(url, "failed_write")
                return

            # Process depending on type
            text = ""
            per_page_texts = []
            ocr_used = False

            # If HTML or unknown treat as HTML extraction
            if file_ext == ".html":
                try:
                    html = body.decode('utf-8', errors='replace')
                    # Quick heuristic: detect login form or interactive login pages
                    if self.config.skip_auth and looks_like_login_page(html, url):
                        logger.info(f"Detected login/interactive page for {url}. Marking as auth_required and skipping.")
                        await self._update_url_status(url, "auth_required", content_path=str(file_path))
                        return
                    text = extract_text_from_html(html)
                except Exception as e:
                    logger.debug(f"HTML text extraction failed: {e}")
                    text = ""
            elif file_ext == ".txt":
                try:
                    text = body.decode('utf-8', errors='replace')
                except Exception:
                    text = body.decode('latin1', errors='replace')
            elif file_ext == ".docx":
                # run docx extraction in executor
                text = await self.loop.run_in_executor(None, extract_text_from_docx_sync, str(file_path))
            elif file_ext == ".pdf":
                # try text layer first
                text, per_page_texts = await self.loop.run_in_executor(
                    self.executor, extract_text_from_pdf_sync, str(file_path), self.config.max_pdf_pages
                )
                if (not text) or sum(len(p.strip()) for p in per_page_texts) < 200:
                    # run OCR in process pool
                    logger.info("No text layer or too small -> OCR PDF")
                    ocr_text, ocr_pages = await self.loop.run_in_executor(
                        self.executor, ocr_pdf_pages_sync,
                        str(file_path), self.config.poppler_path, 300, self.config.max_pdf_pages, self.config.ocr_lang
                    )
                    if ocr_text and len(ocr_text) > len(text):
                        text = ocr_text
                        per_page_texts = ocr_pages
                        ocr_used = True
            elif file_ext == ".img":
                text = await self.loop.run_in_executor(self.executor, extract_text_from_image_sync, str(file_path), self.config.ocr_lang)
                ocr_used = True
            else:
                # fallback: HTML extraction
                try:
                    html = body.decode('utf-8', errors='replace')
                    if self.config.skip_auth and looks_like_login_page(html, url):
                        logger.info(f"Detected login/interactive page for {url}. Marking as auth_required and skipping.")
                        await self._update_url_status(url, "auth_required", content_path=str(file_path))
                        return
                    text = extract_text_from_html(html)
                except Exception:
                    text = ""

            # Normalize and QC
            norm_text = normalize_text(text)
            # Optionally remove repeating headers
            if per_page_texts:
                cleaned_pages, removed = await self.loop.run_in_executor(self.executor, remove_repeating_headers, per_page_texts, 0.4)
                norm_text = normalize_text("\n\n".join(cleaned_pages))

            # basic statistics & heuristics
            words = len(re.findall(r"\w+", norm_text, flags=re.UNICODE))
            lang = None
            lang_ok = True
            if detect is not None and norm_text:
                try:
                    lang = await self.loop.run_in_executor(None, detect, (norm_text[:2000] if len(norm_text) > 2000 else norm_text))
                except Exception:
                    lang = None
            if lang and lang != "es":
                lang_ok = False

            qc = await self.loop.run_in_executor(self.executor, detect_ocr_issues_sync, norm_text)

            # Filter by quality / min words
            if words < self.config.min_words:
                logger.info(f"Filtered (too short) {url} words={words}")
                await self._update_url_status(url, "filtered_short", content_path=str(file_path), text_path=None, sha256=None, words=words, lang=lang, ocr_used=int(ocr_used))
                return

            if not qc.get("ok", True):
                logger.info(f"QC warnings for {url}: {qc.get('issues')}")
                # still save but mark as qc_failed
                status_after_qc = "qc_issues"
            else:
                status_after_qc = "processed"

            # Deduplication: hash normalized text
            text_hash = sha256_text(norm_text)
            async with self.db.execute("SELECT sha256 FROM content_hashes WHERE sha256 = ?", (text_hash,)) as cur:
                found = await cur.fetchone()
            if found:
                # Already have this content
                existing = found[0]
                logger.info(f"Duplicate content detected for {url} (sha256={text_hash})")
                await self._update_url_status(url, "duplicate", content_path=str(file_path), text_path=None, sha256=text_hash, words=words, lang=lang, ocr_used=int(ocr_used))
                return
            # Save normalized text to file
            txt_dir = self.config.outdir / "txt" / domain_safe
            txt_dir.mkdir(parents=True, exist_ok=True)
            txt_filename = f"{safe_filename(parsed.path or parsed.netloc)[:120]}_{int(time.time())}.txt"
            txt_path = txt_dir / txt_filename
            try:
                async with aiofiles.open(txt_path, "w", encoding="utf-8") as f:
                    await f.write(norm_text)
            except Exception as e:
                logger.error(f"Failed to write txt file {txt_path}: {e}")
                await self._update_url_status(url, "failed_write_txt")
                return

            # Insert content hash
            await self.db.execute(
                "INSERT OR IGNORE INTO content_hashes(sha256, saved_text_path, added_at) VALUES (?, ?, ?)",
                (text_hash, str(txt_path), time.time())
            )
            await self.db.execute(
                "UPDATE urls SET status=?, content_path=?, text_path=?, sha256=?, words=?, lang=?, ocr_used=? WHERE url=?",
                (status_after_qc, str(file_path), str(txt_path), text_hash, words, lang, int(ocr_used), url)
            )
            await self.db.commit()

            # Extract links and enqueue new candidates if depth < max_depth
            if file_ext == ".html" and depth < self.config.max_depth:
                try:
                    html = (body.decode('utf-8', errors='replace'))
                    links = get_links_from_html(url, html)
                    await self._enqueue_links(links, depth + 1)
                except Exception as e:
                    logger.debug(f"Extract links failed {url}: {e}")

            logger.info(f"Processed and saved text for {url} -> {txt_path} (words={words})")
        except Exception as e:
            logger.exception(f"Unexpected error processing {url}: {e}")
            await self._update_url_status(url, "error")
            return

    async def _enqueue_links(self, links: Set[str], depth: int):
        """Add new links to DB queue if not already present and if they match term variants."""
        now = time.time()
        for link in links:
            # Basic normalization: remove fragments, whitespace
            link = link.split("#")[0].strip()
            if not link or len(link) > 2000:
                continue
            if link in self.seen_url_cache:
                continue
            parsed = urlparse(link)
            if not parsed.scheme.startswith("http"):
                continue
            domain = parsed.netloc
            # optionally filter by allowed domains
            if self.config.allowed_domains and not any(domain.endswith(d) for d in self.config.allowed_domains):
                continue
            # Filter by search terms: ensure page content likely relevant (cheap heuristic: check url/path)
            path_lower = parsed.path.lower()
            if self.term_variants:
                if not any(tv.lower() in path_lower for tv in self.term_variants):
                    # we still allow enqueueing but less priority; skip to keep crawl focused
                    continue
            # Insert into DB
            try:
                await self.db.execute(
                    "INSERT OR IGNORE INTO urls(url, domain, status, depth, added_at, last_seen) VALUES (?, ?, ?, ?, ?, ?)",
                    (link, domain, 'queued', depth, now, now)
                )
                await self.db.commit()
                self.seen_url_cache.add(link)
            except Exception as e:
                logger.debug(f"DB enqueue failed for {link}: {e}")

    async def _respect_rate_limit(self, domain: str):
        """Ensure politeness per domain using simple timestamp check."""
        now = time.time()
        last = self.rate_limit_cache.get(domain, 0)
        elapsed = now - last
        if elapsed < self.config.politeness:
            await asyncio.sleep(self.config.politeness - elapsed)
        self.rate_limit_cache[domain] = time.time()

# ---------------------------
# Small helper functions reused
# ---------------------------
def extract_text_from_html(html: str) -> str:
    """Synchronous HTML extraction; fast, used inline."""
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript", "header", "footer", "nav", "svg"]):
        s.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines)

def get_links_from_html(url: str, text: str) -> Set[str]:
    soup = BeautifulSoup(text, "html.parser")
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("javascript:") or href.startswith("mailto:"):
            continue
        full = urljoin(url, href)
        links.add(full)
    return links

def remove_repeating_headers(pages_texts: List[str], min_pages_fraction: float = 0.4) -> Tuple[List[str], List[str]]:
    """Heuristic header/footer removal: return cleaned pages and list of removed lines."""
    lines_counter = {}
    pages_lines = []
    for page in pages_texts:
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        pages_lines.append(lines)
        unique = set(lines)
        for ln in unique:
            lines_counter.setdefault(ln, 0)
            lines_counter[ln] += 1
    n_pages = max(1, len(pages_texts))
    remove_set = {ln for ln, cnt in lines_counter.items() if cnt >= max(2, math.ceil(min_pages_fraction * n_pages)) and len(ln) > 3}
    cleaned_pages = []
    for lines in pages_lines:
        cleaned_lines = [ln for ln in lines if ln not in remove_set]
        cleaned_pages.append("\n".join(cleaned_lines))
    return cleaned_pages, list(remove_set)

def looks_like_login_page(html: str, url: str) -> bool:
    """
    Heurística para detectar páginas que piden autenticación o interacción (login forms).
    - Busca <input type="password">, formulario con id/class 'login', 'signin', 'auth'
    - Busca palabras claves en título o body: 'iniciar sesión', 'acceder', 'login', 'sign in'
    - URLs que contienen '/login', '/signin', '/auth', '/account' suelen ser login
    """
    if not html:
        return False
    low = html.lower()
    # quick url heuristic
    if any(tok in url.lower() for tok in ["/login", "/signin", "/auth", "/account", "/iniciar-sesion", "/acceder"]):
        return True
    if "<input" in low and "password" in low:
        return True
    if re.search(r'form[^>]*\b(class|id)\b[^>]*(login|signin|auth|account)', low):
        return True
    if re.search(r'(iniciar sesi[oó]n|inicia sesi[oó]n|acceder|identificarse|sign in|log in|sign-in)', low):
        return True
    return False

# ---------------------------
# CLI entrypoint (enhanced seeds handling)
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Advanced spider + OCR ETL pipeline for historical documents")
    p.add_argument("--seeds", required=True, help="Text file with seed URLs (one per line) or a directory containing many seed files (.txt)")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR, help="Output directory")
    p.add_argument("--db", default=DEFAULT_DB, help="SQLite DB path for metadata")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--max-pages", type=int, default=1000, help="Max number of pages to process (overall)")
    p.add_argument("--poppler-path", default=DEFAULT_POPPLER_PATH, help="poppler binaries path (if needed on Windows)")
    p.add_argument("--ocr-lang", default=DEFAULT_TESSERACT_LANG, help="tesseract OCR language code (e.g., spa)")
    p.add_argument("--min-words", type=int, default=50)
    p.add_argument("--politeness", type=float, default=0.5, help="Seconds between requests per domain")
    p.add_argument("--allowed-domains", nargs="*", default=None, help="Domains to restrict crawling to (optional)")
    p.add_argument("--continuous", action="store_true", dest="continuous", help="Keep crawling continuously (poll DB)")
    p.add_argument("--expand-terms", action="store_true", help="Expand search terms heuristically")
    p.add_argument("--terms", nargs="*", default=[], help="Optional search terms to prioritize (space separated)")
    p.add_argument("--skip-auth", action="store_true", default=True, help="Skip pages requiring authentication (HTTP 401/403/407 or login forms). Default: enabled")
    return p.parse_args()

async def collect_seeds(seeds_path: Path) -> List[str]:
    """
    If seeds_path is file -> read lines.
    If seeds_path is dir -> read all .txt files inside (non-recursive) and aggregate unique URLs.
    """
    if seeds_path.is_file():
        text = seeds_path.read_text(encoding="utf-8", errors="ignore")
        seeds = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        return seeds
    elif seeds_path.is_dir():
        seeds_set = set()
        for p in sorted(seeds_path.glob("*.txt")):
            try:
                text = p.read_text(encoding="utf-8", errors="ignore")
                for ln in text.splitlines():
                    ln = ln.strip()
                    if not ln or ln.startswith("#"):
                        continue
                    seeds_set.add(ln)
            except Exception as e:
                logger.debug(f"Failed to read seeds file {p}: {e}")
        return sorted(seeds_set)
    else:
        raise FileNotFoundError(f"Seeds path not found: {seeds_path}")

async def main_async():
    args = parse_args()
    seeds_path = Path(args.seeds)
    if not seeds_path.exists():
        logger.error(f"Seeds path not found: {seeds_path}")
        sys.exit(1)

    seeds = await collect_seeds(seeds_path)
    if not seeds:
        logger.error("No seeds found in provided seeds file/directory.")
        sys.exit(1)

    logger.info(f"Loaded {len(seeds)} seed URLs from {seeds_path}")

    config = SpiderConfig(
        outdir=args.outdir,
        db_path=args.db,
        user_agent=DEFAULT_USER_AGENT,
        concurrency=args.concurrency,
        max_depth=args.max_depth,
        politeness=args.politeness,
        max_pages=args.max_pages,
        poppler_path=args.poppler_path,
        ocr_lang=args.ocr_lang,
        min_words=args.min_words,
        timeout=30,
        allowed_domains=args.allowed_domains,
        skip_auth=args.skip_auth
    )
    spider = Spider(config=config, search_terms=args.terms or [], expand_terms=args.expand_terms)
    await spider.init()
    await spider.seed_urls(seeds)
    try:
        await spider.crawl(max_pages=args.max_pages, continuous=args.continuous)
    finally:
        await spider.close()

def main():
    try:
        asyncio.run(main_async())
    except KeyboardInterrupt:
        logger.info("Interrupted by user, shutting down.")

if __name__ == "__main__":
    main()
