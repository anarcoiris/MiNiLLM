#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
spider_ocr_pipeline_v2.py (PDF-prioritizing variant)

Mejoras principales sobre la versión anterior:
- Prioriza enlaces a PDFs (y otros recursos "documentales") en la cola mediante un campo `priority`.
- Usa HTML principalmente para *mapear* el sitio y descubrir PDFs; evita guardar/almacenar HTML que no contenga contenido útil
  (p.ej. menús, login, páginas de índice sin PDFs). Solo guarda HTML si contiene texto útil (>= min_words_html) o si contiene
  enlaces a PDFs (para trazabilidad).
- Añade campo `priority` en la tabla `urls` y selecciona URLs a procesar ordenando por `(priority DESC, added_at ASC)`.
- Argumento CLI `--prefer-pdf` para activar el comportamiento; por defecto True.
- Clasifica y separa claramente outputs: `raw/html/`, `raw/pdf/`, `txt/`, `meta/`.
- Conserva OCR, deduplicación, QC y todo el pipeline previo.
- Documentación inline en español.

Uso típico:
    python spider_ocr_pipeline_v2.py --seeds seeds.txt --outdir downloads --prefer-pdf --max-depth 2 --max-pages 1000

Nota: basado en la versión previa; modifica la lógica de enqueue y selección para priorizar PDFs.
"""
from __future__ import annotations
import os
import sys
import re
import time
import json
import math
import uuid
import hashlib
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

# Funciones opcionales usadas en process pool
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

# language detect optional
try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
except Exception:
    detect = None

# Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("spider_ocr_v2")

# ---------------------------
# Defaults
# ---------------------------
DEFAULT_USER_AGENT = "MiniLLM-Spider/2.0 (+https://example.com)"
DEFAULT_POPPLER_PATH = os.environ.get("POPPLER_PATH", None)
DEFAULT_TESSERACT_LANG = "spa"
DEFAULT_DB = "spider_meta.db"
DEFAULT_OUTDIR = "downloads"
DEFAULT_MAX_PDF_PAGES = 1000

# ---------------------------
# Utilities
# ---------------------------
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
    if not text:
        return text
    s = ftfy.fix_text(text)
    s = unicodedata.normalize("NFKC", s)
    s = "".join(ch for ch in s if (ch == "\n" or ch == "\t" or ord(ch) >= 32))
    s = re.sub(r"[ \t]{2,}", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    s = "\n".join(line.strip() for line in s.splitlines())
    return s.strip()

def detect_ocr_issues_sync(text: str) -> Dict[str, Any]:
    stats = {}
    if not text or len(text) < 50:
        stats.update({"ok": False, "reason": "empty_or_too_short"})
        return stats
    chars = len(text)
    words = re.findall(r"\w+", text, flags=re.UNICODE)
    num_words = len(words)
    stats["chars"] = chars
    stats["words"] = num_words
    mojibake_patterns = re.compile(r"[ÂÃ�¿�]|Ã[¡-º]|â|Ã©|Ã³|Ã±", flags=re.UNICODE)
    mojibake_count = len(mojibake_patterns.findall(text))
    stats["mojibake_count"] = mojibake_count
    stats["mojibake_ratio"] = mojibake_count / max(1, chars)
    word_tokens = re.findall(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ'-]+", text)
    if word_tokens:
        spanish_like = sum(1 for w in word_tokens if SPANISH_WORD_RE.match(w))
        frac_spanish = spanish_like / len(word_tokens)
    else:
        frac_spanish = 0.0
    stats["frac_spanish_words"] = frac_spanish
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
# Sync extraction helpers (run in executor)
# ---------------------------
def extract_text_from_pdf_sync(path: str, max_pages: Optional[int]=None) -> Tuple[str, List[str]]:
    if PdfReader is None:
        raise RuntimeError("PyPDF2 required")
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
    if convert_from_path is None or pytesseract is None:
        raise RuntimeError("pdf2image and pytesseract required")
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
    if DocxDocument is None:
        raise RuntimeError("python-docx required")
    try:
        doc = DocxDocument(path)
        paras = [p.text for p in doc.paragraphs]
        return "\n".join(paras)
    except Exception as e:
        logger.error(f"docx read error {path}: {e}")
        return ""

def extract_text_from_image_sync(path: str, lang: str) -> str:
    if pytesseract is None:
        raise RuntimeError("pytesseract required")
    from PIL import Image
    try:
        img = Image.open(path)
        txt = pytesseract.image_to_string(img, lang=lang)
        return txt
    except Exception as e:
        logger.error(f"OCR image failed {path}: {e}")
        return ""

# ---------------------------
# HTML helpers
# ---------------------------
def extract_text_from_html(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for s in soup(["script", "style", "noscript", "header", "footer", "nav", "svg", "form"]):
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

def looks_like_login_page(html: str, url: str) -> bool:
    if not html:
        return False
    low = html.lower()
    if any(tok in url.lower() for tok in ["/login", "/signin", "/auth", "/account", "/iniciar-sesion", "/acceder"]):
        return True
    if "<input" in low and "password" in low:
        return True
    if re.search(r'form[^>]*\b(class|id)\b[^>]*(login|signin|auth|account)', low):
        return True
    if re.search(r'(iniciar sesi[oó]n|acceder|identificarse|sign in|log in|sign-in)', low):
        return True
    return False

def remove_repeating_headers(pages_texts: List[str], min_pages_fraction: float = 0.4) -> Tuple[List[str], List[str]]:
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

# ---------------------------
# Spider config & core
# ---------------------------
class SpiderConfig:
    def __init__(self,
                 outdir: str = DEFAULT_OUTDIR,
                 db_path: str = DEFAULT_DB,
                 user_agent: str = DEFAULT_USER_AGENT,
                 concurrency: int = 6,
                 max_depth: int = 2,
                 politeness: float = 0.5,
                 max_pdf_pages: int = DEFAULT_MAX_PDF_PAGES,
                 poppler_path: Optional[str] = DEFAULT_POPPLER_PATH,
                 ocr_lang: str = DEFAULT_TESSERACT_LANG,
                 min_words: int = 50,
                 min_words_html: int = 200,
                 timeout: int = 30,
                 allowed_domains: Optional[List[str]] = None,
                 skip_auth: bool = True,
                 prefer_pdf: bool = True):
        self.outdir = Path(outdir)
        self.db_path = Path(db_path)
        self.user_agent = user_agent
        self.concurrency = concurrency
        self.max_depth = max_depth
        self.politeness = politeness
        self.max_pdf_pages = max_pdf_pages
        self.poppler_path = poppler_path
        self.ocr_lang = ocr_lang
        self.min_words = min_words
        self.min_words_html = min_words_html
        self.timeout = timeout
        self.allowed_domains = allowed_domains or []
        self.skip_auth = skip_auth
        self.prefer_pdf = prefer_pdf

class Spider:
    def __init__(self, config: SpiderConfig, search_terms: List[str], expand_terms: bool = True):
        self.config = config
        self.search_terms = search_terms or []
        self.expand_terms = expand_terms
        self.session: Optional[aiohttp.ClientSession] = None
        self.db: Optional[aiosqlite.Connection] = None
        self.executor = concurrent.futures.ProcessPoolExecutor(max_workers=max(1, os.cpu_count()-1))
        self.loop = asyncio.get_event_loop()
        self.rate_limit_cache: Dict[str, float] = {}
        self.seen_url_cache: Set[str] = set()
        self.term_variants = self._expand_terms(self.search_terms) if expand_terms else set(self.search_terms)
        self.config.outdir.mkdir(parents=True, exist_ok=True)
        (self.config.outdir / "raw" / "html").mkdir(parents=True, exist_ok=True)
        (self.config.outdir / "raw" / "pdf").mkdir(parents=True, exist_ok=True)
        (self.config.outdir / "txt").mkdir(parents=True, exist_ok=True)
        (self.config.outdir / "meta").mkdir(parents=True, exist_ok=True)

    async def init(self):
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
                last_seen REAL,
                priority INTEGER DEFAULT 0
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
            CREATE INDEX IF NOT EXISTS idx_urls_domain_priority ON urls(domain, priority DESC);
        """)
        await self.db.commit()

    def _expand_terms(self, terms: List[str]) -> Set[str]:
        out = set()
        for t in terms:
            t0 = t.strip()
            if not t0:
                continue
            out.add(t0)
            out.add(t0.lower())
            out.add(t0.title())
            t_noacc = ''.join(ch for ch in unicodedata.normalize('NFD', t0) if unicodedata.category(ch) != 'Mn')
            out.add(t_noacc)
            out.add(t_noacc.lower())
            if not t0.endswith('s'):
                out.add(t0 + "s")
        return out

    async def seed_urls(self, seeds: List[str]):
        now = time.time()
        async with self.db.execute("BEGIN"):
            for url in seeds:
                url = url.strip()
                if not url:
                    continue
                parsed = urlparse(url)
                if not parsed.scheme:
                    url = "http://" + url
                domain = urlparse(url).netloc
                priority = 10 if self._looks_like_pdf_link(url) else 0
                try:
                    await self.db.execute(
                        "INSERT OR IGNORE INTO urls(url, domain, status, depth, added_at, last_seen, priority) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (url, domain, 'queued', 0, now, now, priority)
                    )
                    self.seen_url_cache.add(url)
                except Exception as e:
                    logger.debug(f"DB insert seed failed: {e}")
            await self.db.commit()

    def _looks_like_pdf_link(self, url: str) -> bool:
        url_l = url.lower()
        if url_l.endswith(".pdf"):
            return True
        if "pdf" in url_l and any(tok in url_l for tok in ("/download", "/view", "document")):
            return True
        return False

    async def crawl(self, max_pages: Optional[int] = None, continuous: bool = False):
        sem = asyncio.Semaphore(self.config.concurrency)
        tasks = []
        processed = 0
        while True:
            row = await self._get_next_queued_url()
            if row is None:
                if continuous:
                    await asyncio.sleep(2.0)
                    continue
                else:
                    break
            url, domain, depth = row
            if self.config.allowed_domains and not any(domain.endswith(d) for d in self.config.allowed_domains):
                await self._update_url_status(url, "skipped_domain")
                continue
            async with sem:
                task = asyncio.create_task(self._process_url_task(url, depth))
                tasks.append(task)
            processed += 1
            if max_pages and processed >= max_pages:
                break
        if tasks:
            await asyncio.gather(*tasks)

    async def _get_next_queued_url(self) -> Optional[Tuple[str, str, int]]:
        # Select queued URL ordering by priority desc then added_at asc
        async with self.db.execute("SELECT url, domain, depth FROM urls WHERE status='queued' ORDER BY priority DESC, added_at ASC LIMIT 1") as cur:
            row = await cur.fetchone()
            if not row:
                return None
            url, domain, depth = row
            await self.db.execute("UPDATE urls SET status=?, last_seen=? WHERE url=?", ("in_progress", time.time(), url))
            await self.db.commit()
            return url, domain, depth

    async def _update_url_status(self, url: str, status: str, **kwargs):
        fields = []
        values = []
        for k, v in kwargs.items():
            fields.append(f"{k} = ?")
            values.append(v)
        fields_sql = ", ".join(fields) + (", status = ?, last_seen = ?" if fields else "status = ?, last_seen = ?")
        values.extend([status, time.time(), url])
        sql = f"UPDATE urls SET {fields_sql} WHERE url = ?"
        try:
            await self.db.execute(sql, values)
            await self.db.commit()
        except Exception as e:
            logger.debug(f"DB update failed: {e}")

    async def _process_url_task(self, url: str, depth: int):
        try:
            parsed = urlparse(url)
            domain = parsed.netloc
            await self._respect_rate_limit(domain)
            logger.info(f"[depth {depth}] Fetching: {url}")
            headers = {"User-Agent": self.config.user_agent}
            try:
                async with self.session.get(url, headers=headers) as resp:
                    status = resp.status
                    content_type = resp.headers.get("Content-Type", "").lower()
                    if self.config.skip_auth and status in (401, 403, 407):
                        logger.info(f"Skipping {url} due to HTTP status {status} (auth required).")
                        await self._update_url_status(url, "auth_required")
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

            # Decide resource type
            is_pdf = ("application/pdf" in content_type) or url.lower().endswith(".pdf") or self._looks_like_pdf_link(url)
            is_image = any(ext in url.lower() for ext in (".jpg", ".jpeg", ".png")) or "image" in content_type
            is_docx = url.lower().endswith(".docx") or "officedocument.wordprocessingml" in content_type
            is_text = "text/plain" in content_type or url.lower().endswith(".txt")
            is_html = ("text/html" in content_type) or (not (is_pdf or is_image or is_docx or is_text) and url.lower().endswith((".php", ".aspx", "/")) )

            domain_safe = domain.replace(":", "_")
            timestamp = int(time.time())

            # If PDF: save to raw/pdf and process (preferential path)
            if is_pdf:
                raw_dir = self.config.outdir / "raw" / "pdf" / domain_safe
                raw_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{safe_filename(parsed.path or parsed.netloc)[:120]}_{timestamp}.pdf"
                file_path = raw_dir / filename
                try:
                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(body)
                except Exception as e:
                    logger.error(f"Failed to write PDF {file_path}: {e}")
                    await self._update_url_status(url, "failed_write")
                    return

                # Try to extract text layer
                text = ""
                per_page_texts = []
                ocr_used = False
                try:
                    text, per_page_texts = await self.loop.run_in_executor(self.executor, extract_text_from_pdf_sync, str(file_path), self.config.max_pdf_pages)
                except Exception as e:
                    logger.debug(f"Pdf text extraction failed sync: {e}")
                    text = ""
                    per_page_texts = []

                # If no/low text, run OCR
                total_page_text_len = sum(len(p.strip()) for p in per_page_texts)
                if not text or total_page_text_len < 200:
                    if convert_from_path and pytesseract:
                        logger.info(f"OCRing PDF {file_path} (no text layer or too small)")
                        ocr_text, ocr_pages = await self.loop.run_in_executor(
                            self.executor, ocr_pdf_pages_sync, str(file_path), self.config.poppler_path, 300, self.config.max_pdf_pages, self.config.ocr_lang
                        )
                        if ocr_text and len(ocr_text) > len(text):
                            text = ocr_text
                            per_page_texts = ocr_pages
                            ocr_used = True

                norm_text = normalize_text(text)
                words = len(re.findall(r"\w+", norm_text, flags=re.UNICODE))
                qc = await self.loop.run_in_executor(self.executor, detect_ocr_issues_sync, norm_text)

                if words < self.config.min_words:
                    logger.info(f"Filtered PDF (too short) {url} words={words}")
                    await self._update_url_status(url, "filtered_short", content_path=str(file_path), words=words, lang=None, ocr_used=int(ocr_used))
                    return

                text_hash = sha256_text(norm_text)
                async with self.db.execute("SELECT sha256 FROM content_hashes WHERE sha256 = ?", (text_hash,)) as cur:
                    found = await cur.fetchone()
                if found:
                    logger.info(f"Duplicate PDF content detected for {url} (sha256={text_hash})")
                    await self._update_url_status(url, "duplicate", content_path=str(file_path), sha256=text_hash, words=words, lang=None, ocr_used=int(ocr_used))
                    return

                txt_dir = self.config.outdir / "txt" / domain_safe
                txt_dir.mkdir(parents=True, exist_ok=True)
                txt_filename = f"{safe_filename(parsed.path or parsed.netloc)[:120]}_{timestamp}.txt"
                txt_path = txt_dir / txt_filename
                try:
                    async with aiofiles.open(txt_path, "w", encoding="utf-8") as f:
                        await f.write(norm_text)
                except Exception as e:
                    logger.error(f"Failed to write txt {txt_path}: {e}")
                    await self._update_url_status(url, "failed_write_txt")
                    return

                await self.db.execute(
                    "INSERT OR IGNORE INTO content_hashes(sha256, saved_text_path, added_at) VALUES (?, ?, ?)",
                    (text_hash, str(txt_path), time.time())
                )
                await self.db.execute(
                    "UPDATE urls SET status=?, content_path=?, text_path=?, sha256=?, words=?, ocr_used=? WHERE url=?",
                    ("processed", str(file_path), str(txt_path), text_hash, words, int(ocr_used), url)
                )
                await self.db.commit()

                # do not extract links from pdf for enqueueing generally, but you may inspect metadata (skipped)
                logger.info(f"Processed PDF {url} -> {txt_path} (words={words})")
                return

            # If image or docx or text: handle similarly (save raw and extract text)
            if is_image:
                raw_dir = self.config.outdir / "raw" / "images" / domain_safe
                raw_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{safe_filename(parsed.path or parsed.netloc)[:120]}_{timestamp}.img"
                file_path = raw_dir / filename
                try:
                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(body)
                except Exception as e:
                    logger.error(f"Failed to write image {file_path}: {e}")
                    await self._update_url_status(url, "failed_write")
                    return
                text = await self.loop.run_in_executor(self.executor, extract_text_from_image_sync, str(file_path), self.config.ocr_lang)
                norm_text = normalize_text(text)
                words = len(re.findall(r"\w+", norm_text, flags=re.UNICODE))
                if words < self.config.min_words:
                    await self._update_url_status(url, "filtered_short", content_path=str(file_path), words=words)
                    return
                txt_dir = self.config.outdir / "txt" / domain_safe
                txt_dir.mkdir(parents=True, exist_ok=True)
                txt_filename = f"{safe_filename(parsed.path or parsed.netloc)[:120]}_{timestamp}.txt"
                txt_path = txt_dir / txt_filename
                try:
                    async with aiofiles.open(txt_path, "w", encoding="utf-8") as f:
                        await f.write(norm_text)
                except Exception as e:
                    logger.error(f"Failed to write txt {txt_path}: {e}")
                    await self._update_url_status(url, "failed_write_txt")
                    return
                text_hash = sha256_text(norm_text)
                await self.db.execute(
                    "INSERT OR IGNORE INTO content_hashes(sha256, saved_text_path, added_at) VALUES (?, ?, ?)",
                    (text_hash, str(txt_path), time.time())
                )
                await self.db.execute(
                    "UPDATE urls SET status=?, content_path=?, text_path=?, sha256=?, words=? WHERE url=?",
                    ("processed", str(file_path), str(txt_path), text_hash, words, url)
                )
                await self.db.commit()
                logger.info(f"Processed image {url} -> {txt_path} (words={words})")
                return

            # If docx
            if is_docx:
                raw_dir = self.config.outdir / "raw" / "docx" / domain_safe
                raw_dir.mkdir(parents=True, exist_ok=True)
                filename = f"{safe_filename(parsed.path or parsed.netloc)[:120]}_{timestamp}.docx"
                file_path = raw_dir / filename
                try:
                    async with aiofiles.open(file_path, "wb") as f:
                        await f.write(body)
                except Exception as e:
                    logger.error(f"Failed to write docx {file_path}: {e}")
                    await self._update_url_status(url, "failed_write")
                    return
                text = await self.loop.run_in_executor(self.executor, extract_text_from_docx_sync, str(file_path))
                norm_text = normalize_text(text)
                words = len(re.findall(r"\w+", norm_text, flags=re.UNICODE))
                if words < self.config.min_words:
                    await self._update_url_status(url, "filtered_short", content_path=str(file_path), words=words)
                    return
                txt_dir = self.config.outdir / "txt" / domain_safe
                txt_dir.mkdir(parents=True, exist_ok=True)
                txt_filename = f"{safe_filename(parsed.path or parsed.netloc)[:120]}_{timestamp}.txt"
                txt_path = txt_dir / txt_filename
                try:
                    async with aiofiles.open(txt_path, "w", encoding="utf-8") as f:
                        await f.write(norm_text)
                except Exception as e:
                    logger.error(f"Failed to write txt {txt_path}: {e}")
                    await self._update_url_status(url, "failed_write_txt")
                    return
                text_hash = sha256_text(norm_text)
                await self.db.execute(
                    "INSERT OR IGNORE INTO content_hashes(sha256, saved_text_path, added_at) VALUES (?, ?, ?)",
                    (text_hash, str(txt_path), time.time())
                )
                await self.db.execute(
                    "UPDATE urls SET status=?, content_path=?, text_path=?, sha256=?, words=? WHERE url=?",
                    ("processed", str(file_path), str(txt_path), text_hash, words, url)
                )
                await self.db.commit()
                logger.info(f"Processed docx {url} -> {txt_path} (words={words})")
                return

            # HTML handling: use HTML primarily to discover links and PDFs.
            if is_html:
                try:
                    html = body.decode('utf-8', errors='replace')
                except Exception:
                    html = body.decode('latin1', errors='replace')

                if self.config.skip_auth and looks_like_login_page(html, url):
                    await self._update_url_status(url, "auth_required", content_path=None)
                    return

                # Extract links early
                links = get_links_from_html(url, html)

                # Identify pdf links among extracted links
                pdf_links = set(l for l in links if self._looks_like_pdf_link(l) or l.lower().endswith(".pdf"))

                # Determine whether HTML contains substantial textual content
                extracted_text = extract_text_from_html(html)
                norm_html_text = normalize_text(extracted_text)
                words_html = len(re.findall(r"\w+", norm_html_text, flags=re.UNICODE))

                # If prefer_pdf: prioritize adding PDF links with high priority
                now = time.time()
                for l in pdf_links:
                    if l in self.seen_url_cache:
                        continue
                    parsed_l = urlparse(l)
                    domain_l = parsed_l.netloc
                    priority = 50  # high priority for discovered PDFs
                    await self.db.execute(
                        "INSERT OR IGNORE INTO urls(url, domain, status, depth, added_at, last_seen, priority) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (l, domain_l, 'queued', depth + 1, now, now, priority)
                    )
                    self.seen_url_cache.add(l)
                await self.db.commit()

                # Enqueue other links (lower priority) but only if they match term variants OR we are under depth
                for l in links:
                    if l in self.seen_url_cache:
                        continue
                    parsed_l = urlparse(l)
                    if not parsed_l.scheme.startswith("http"):
                        continue
                    if self.config.allowed_domains and not any(parsed_l.netloc.endswith(d) for d in self.config.allowed_domains):
                        continue
                    # cheap relevance check: if term_variants in path
                    path_lower = parsed_l.path.lower()
                    if self.term_variants:
                        if not any(tv.lower() in path_lower for tv in self.term_variants):
                            # skip general pages unless they're PDF links (handled) or depth allows mapping
                            if depth + 1 <= self.config.max_depth and words_html >= self.config.min_words_html:
                                # still allow mapping pages that have sufficient text (sitemap-like)
                                priority = 1
                            else:
                                continue
                        else:
                            priority = 5
                    else:
                        priority = 1 if words_html >= self.config.min_words_html else 0

                    await self.db.execute(
                        "INSERT OR IGNORE INTO urls(url, domain, status, depth, added_at, last_seen, priority) VALUES (?, ?, ?, ?, ?, ?, ?)",
                        (l, parsed_l.netloc, 'queued', depth + 1, now, now, priority)
                    )
                    self.seen_url_cache.add(l)
                await self.db.commit()

                # Decide whether to store the HTML raw and/or extract text
                # We save HTML raw/text only if:
                #  - contains PDF links (for traceability), OR
                #  - contains useful text (words_html >= min_words_html)
                save_html_raw = bool(pdf_links) or (words_html >= self.config.min_words_html)
                if save_html_raw:
                    raw_dir = self.config.outdir / "raw" / "html" / domain_safe
                    raw_dir.mkdir(parents=True, exist_ok=True)
                    filename = f"{safe_filename(parsed.path or parsed.netloc)[:120]}_{timestamp}.html"
                    file_path = raw_dir / filename
                    try:
                        async with aiofiles.open(file_path, "w", encoding="utf-8") as f:
                            await f.write(html)
                    except Exception as e:
                        logger.error(f"Failed to write html raw {file_path}: {e}")
                        await self._update_url_status(url, "failed_write")
                        return

                    # If HTML has substantial text, store it as processed text (optional)
                    if words_html >= self.config.min_words_html:
                        norm_text = norm_html_text
                        words = words_html
                        text_hash = sha256_text(norm_text)
                        async with self.db.execute("SELECT sha256 FROM content_hashes WHERE sha256 = ?", (text_hash,)) as cur:
                            found = await cur.fetchone()
                        if found:
                            await self._update_url_status(url, "duplicate_html", content_path=str(file_path), sha256=text_hash, words=words)
                            return
                        txt_dir = self.config.outdir / "txt" / domain_safe
                        txt_dir.mkdir(parents=True, exist_ok=True)
                        txt_filename = f"{safe_filename(parsed.path or parsed.netloc)[:120]}_{timestamp}.txt"
                        txt_path = txt_dir / txt_filename
                        try:
                            async with aiofiles.open(txt_path, "w", encoding="utf-8") as f:
                                await f.write(norm_text)
                        except Exception as e:
                            logger.error(f"Failed to write txt from html {txt_path}: {e}")
                            await self._update_url_status(url, "failed_write_txt")
                            return
                        await self.db.execute(
                            "INSERT OR IGNORE INTO content_hashes(sha256, saved_text_path, added_at) VALUES (?, ?, ?)",
                            (text_hash, str(txt_path), time.time())
                        )
                        await self.db.execute(
                            "UPDATE urls SET status=?, content_path=?, text_path=?, sha256=?, words=? WHERE url=?",
                            ("processed_html", str(file_path), str(txt_path), text_hash, words, url)
                        )
                        await self.db.commit()
                        logger.info(f"Saved HTML page content {url} -> {txt_path} (words={words})")
                    else:
                        # Save only raw for traceability (PDF links etc.)
                        await self._update_url_status(url, "mapped", content_path=str(file_path))
                        logger.info(f"Mapped HTML (saved raw only) {url} (pdf_links={len(pdf_links)}, words={words_html})")
                else:
                    # Do not save raw HTML (we used it only to map/discover)
                    await self._update_url_status(url, "mapped_no_save")
                    logger.info(f"Mapped HTML (not saved) {url} (pdf_links={len(pdf_links)}, words={words_html})")
                return

            # fallback: unknown resource type - attempt html processing as above
            # (reuse the HTML branch by decoding)
            try:
                html = body.decode('utf-8', errors='replace')
            except Exception:
                html = body.decode('latin1', errors='replace')
            # reuse HTML logic by recursive call-like behavior:
            # temporary insert as 'html' by setting is_html True and reusing code path:
            # For simplicity, we will handle as HTML (same as above).
            # (Duplicated minimal logic)
            links = get_links_from_html(url, html)
            pdf_links = set(l for l in links if self._looks_like_pdf_link(l) or l.lower().endswith(".pdf"))
            now = time.time()
            for l in pdf_links:
                if l in self.seen_url_cache:
                    continue
                parsed_l = urlparse(l)
                domain_l = parsed_l.netloc
                priority = 50
                await self.db.execute(
                    "INSERT OR IGNORE INTO urls(url, domain, status, depth, added_at, last_seen, priority) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (l, domain_l, 'queued', depth + 1, now, now, priority)
                )
                self.seen_url_cache.add(l)
            await self.db.commit()
            await self._update_url_status(url, "mapped_no_save")
            logger.info(f"Mapped (fallback) {url}")
            return

        except Exception as e:
            logger.exception(f"Unexpected error processing {url}: {e}")
            await self._update_url_status(url, "error")
            return

    async def _enqueue_links(self, links: Set[str], depth: int):
        # Legacy function kept for compatibility; primary enqueueing now done in HTML branch
        now = time.time()
        for link in links:
            link = link.split("#")[0].strip()
            if not link or len(link) > 2000:
                continue
            if link in self.seen_url_cache:
                continue
            parsed = urlparse(link)
            if not parsed.scheme.startswith("http"):
                continue
            domain = parsed.netloc
            if self.config.allowed_domains and not any(domain.endswith(d) for d in self.config.allowed_domains):
                continue
            priority = 10 if self._looks_like_pdf_link(link) else 0
            try:
                await self.db.execute(
                    "INSERT OR IGNORE INTO urls(url, domain, status, depth, added_at, last_seen, priority) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (link, domain, 'queued', depth, now, now, priority)
                )
                await self.db.commit()
                self.seen_url_cache.add(link)
            except Exception as e:
                logger.debug(f"DB enqueue failed for {link}: {e}")

    async def _respect_rate_limit(self, domain: str):
        now = time.time()
        last = self.rate_limit_cache.get(domain, 0)
        elapsed = now - last
        if elapsed < self.config.politeness:
            await asyncio.sleep(self.config.politeness - elapsed)
        self.rate_limit_cache[domain] = time.time()

# ---------------------------
# CLI and seeds handling
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Spider + OCR ETL pipeline (PDF-prioritizing)")
    p.add_argument("--seeds", required=True, help="Text file with seed URLs (one per line) or a directory containing many seed files (.txt)")
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--max-depth", type=int, default=2)
    p.add_argument("--max-pages", type=int, default=1000)
    p.add_argument("--poppler-path", default=DEFAULT_POPPLER_PATH)
    p.add_argument("--ocr-lang", default=DEFAULT_TESSERACT_LANG)
    p.add_argument("--min-words", type=int, default=50)
    p.add_argument("--min-words-html", type=int, default=200)
    p.add_argument("--politeness", type=float, default=0.5)
    p.add_argument("--allowed-domains", nargs="*", default=None)
    p.add_argument("--continuous", action="store_true")
    p.add_argument("--expand-terms", action="store_true")
    p.add_argument("--terms", nargs="*", default=[])
    p.add_argument("--skip-auth", action="store_true", default=True)
    p.add_argument("--prefer-pdf", action="store_true", default=True, help="Prioritize PDFs discovered during crawling")
    return p.parse_args()

async def collect_seeds(seeds_path: Path) -> List[str]:
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
        concurrency=args.concurrency,
        max_depth=args.max_depth,
        politeness=args.politeness,
        max_pdf_pages=DEFAULT_MAX_PDF_PAGES,
        poppler_path=args.poppler_path,
        ocr_lang=args.ocr_lang,
        min_words=args.min_words,
        min_words_html=args.min_words_html,
        timeout=30,
        allowed_domains=args.allowed_domains,
        skip_auth=args.skip_auth,
        prefer_pdf=args.prefer_pdf
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
