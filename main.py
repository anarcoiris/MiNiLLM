"""
mini_llm_cli.py

Revisión mejorada de tu ejemplo: CLI guiada (subcomandos), tokenizer BPE (huggingface tokenizers),
Dataset con ventanas aleatorias o por documento, modelo GPT-mini (bloques MHA + FFN, máscara causal),
entrenamiento con checkpoints, mixed precision opcional, generación con temperature/top-k/top-p,
visualización CLI con rich + tqdm.

Requirements (pip):
  pip install torch tokenizers rich tqdm sentencepiece

Uso (ejemplos):
  # entrenar tokenizer BPE desde corpus
  python mini_llm_cli.py init-tokenizer --files data1.txt data2.txt --vocab-size 16000 --out tokenizer.json

  # entrenar modelo
  python mini_llm_cli.py train --tokenizer tokenizer.json --data data.txt --outdir runs/run1 --epochs 10 --batch 64

  # generar texto
  python mini_llm_cli.py generate --tokenizer tokenizer.json --ckpt runs/run1/ckpt_last.pt --prompt "Hola mundo" --max-tokens 100

Abre este archivo en el panel para ver el código completo.
"""

from __future__ import annotations
import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# Tokenizer: HuggingFace `tokenizers`
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import ByteLevel

# CLI visuals
from rich.console import Console
from rich.table import Table
from rich.progress import track
from tqdm.auto import tqdm

console = Console()

# ----------------------------- Utilities -----------------------------

def save_json(path, obj):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

# ----------------------------- Tokenizer helpers -----------------------------

def train_bpe_tokenizer(files: List[str], vocab_size: int = 20000, out: str = 'tokenizer.json') -> Tokenizer:
    """Train a byte-level BPE tokenizer on given files and save to `out`."""
    console.print(f"Training BPE tokenizer on [bold]{len(files)}[/] files (vocab={vocab_size})")
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=['[PAD]', '[UNK]', '[CLS]', '[SEP]'])
    tokenizer.train(files, trainer)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
    tokenizer.save(out)
    console.print(f"Saved tokenizer to {out}")
    return Tokenizer.from_file(out)

def load_tokenizer(path: str) -> Tokenizer:
    return Tokenizer.from_file(path)

# small helper to encode/decode with tokenizers Tokenizer
class HFTokenizerWrapper:
    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self.vocab_size = tok.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tok.decode(ids)

# ----------------------------- Dataset -----------------------------

@dataclass
class DatasetConfig:
    block_size: int = 32
    stride: int = 1
    randomize: bool = True

class CausalTextDataset(Dataset):
    """Dataset that yields sequences of length block_size (x) and targets shifted by one (y).

    If randomize=True, samples random starting indices. Otherwise yields sequential.
    """
    def __init__(self, ids: List[int], block_size: int, randomize: bool = True):
        self.ids = ids
        self.block_size = block_size
        self.randomize = randomize
        self.max_start = max(0, len(ids) - (block_size + 1))

    def __len__(self):
        return max(1, self.max_start)

    def __getitem__(self, idx):
        if self.randomize:
            i = random.randint(0, self.max_start)
        else:
            i = idx
        x = torch.tensor(self.ids[i:i + self.block_size], dtype=torch.long)
        y = torch.tensor(self.ids[i + 1:i + 1 + self.block_size], dtype=torch.long)
        return x, y

# ----------------------------- Model: Tiny GPT-like -----------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, attn_dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.out = nn.Linear(embed_dim, embed_dim)
        self.attn_dropout = nn.Dropout(attn_dropout)

    def forward(self, x, mask=None):
        # x: (B, T, E)
        B, T, E = x.size()
        qkv = self.qkv(x)  # (B, T, 3E)
        q, k, v = qkv.chunk(3, dim=-1)
        # reshape -> (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        # scaled dot-product
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # causal mask
        causal = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn = attn.masked_fill(causal.unsqueeze(0).unsqueeze(0), float('-inf'))
        if mask is not None:
            attn = attn + mask
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        out = torch.matmul(attn, v)  # (B, nh, T, hd)
        out = out.transpose(1, 2).contiguous().view(B, T, E)
        out = self.out(out)
        return out

class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio=4.0, drop=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads, attn_dropout=drop)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size=32, n_embd=128, n_layer=4, n_head=4, drop=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.token_emb = nn.Embedding(vocab_size, n_embd)
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([TransformerBlock(n_embd, n_head, drop=drop) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.block_size
        tok = self.token_emb(idx)
        pos = self.pos_emb(torch.arange(T, device=idx.device))[None, :, :]
        x = self.drop(tok + pos)
        for b in self.blocks:
            x = b(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

# ----------------------------- Generation -----------------------------

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0):
    # logits: (V,)
    logits = logits.clone()
    V = logits.size(0)
    if top_k > 0:
        values, _ = torch.topk(logits, top_k)
        min_value = values[-1]
        logits[logits < min_value] = -float('Inf')
    if top_p > 0.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum_probs = torch.cumsum(probs, dim=-1)
        # remove tokens with cumulative prob above top_p
        sorted_idx_to_remove = cum_probs > top_p
        # keep at least one
        if sorted_idx_to_remove[0]:
            sorted_idx_to_remove[0] = False
        indices_to_remove = sorted_idx[sorted_idx_to_remove]
        logits[indices_to_remove] = -float('Inf')
    return logits

@torch.no_grad()
def generate_text(model: nn.Module, tokenizer_wrapper: HFTokenizerWrapper, prompt: str, max_new_tokens=50, temperature=1.0, top_k=50, top_p=0.0, device='cpu'):
    model.eval()
    ids = tokenizer_wrapper.encode(prompt)
    if len(ids) == 0:
        ids = [0]
    for _ in range(max_new_tokens):
        context = ids[-model.block_size:]
        x = torch.tensor([context], dtype=torch.long).to(device)
        logits = model(x)
        logits = logits[0, -1, :] / max(1e-8, temperature)
        logits = top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        ids.append(next_id)
    return tokenizer_wrapper.decode(ids)

# ----------------------------- Training loop -----------------------------

def train_loop(
    model: nn.Module,
    tokenizer_wrapper: HFTokenizerWrapper,
    train_loader: DataLoader,
    val_loader: DataLoader,
    out_dir: str,
    epochs: int = 5,
    lr: float = 3e-4,
    device: str = 'cpu',
    save_every_steps: int = 1000,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    model.to(device)
    global_step = 0
    best_val = float('inf')
    scaler = torch.cuda.amp.GradScaler() if device.startswith('cuda') else None

    for ep in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {ep+1}/{epochs}")
        running_loss = 0.0
        for x, y in pbar:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            if scaler:
                with torch.cuda.amp.autocast():
                    logits = model(x)
                    B, L, V = logits.shape
                    loss = criterion(logits.view(B*L, V), y.view(B*L))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                logits = model(x)
                B, L, V = logits.shape
                loss = criterion(logits.view(B*L, V), y.view(B*L))
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            global_step += 1
            if global_step % 10 == 0:
                pbar.set_postfix({'loss': running_loss / global_step})
            if global_step % save_every_steps == 0:
                ckpt = {
                    'model': model.state_dict(),
                    'optim': optimizer.state_dict(),
                    'step': global_step,
                }
                Path(out_dir).mkdir(parents=True, exist_ok=True)
                torch.save(ckpt, Path(out_dir) / f'ckpt_step_{global_step}.pt')
        # validation
        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                logits = model(x)
                B, L, V = logits.shape
                loss = criterion(logits.view(B*L, V), y.view(B*L))
                val_loss += loss.item()
        val_loss = val_loss / max(1, len(val_loader))
        console.print(f"Epoch {ep+1} validation loss: {val_loss:.4f}")
        # save best
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(), 'step': global_step}, Path(out_dir) / 'ckpt_last.pt')
        if val_loss < best_val:
            best_val = val_loss
            torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(), 'step': global_step}, Path(out_dir) / 'ckpt_best.pt')

# ----------------------------- CLI -----------------------------

def cmd_init_tokenizer(args):
    tok = train_bpe_tokenizer(args.files, vocab_size=args.vocab_size, out=args.out)
    console.print("Tokenizer trained and saved.")


def cmd_train(args):
    # load tokenizer
    tok = load_tokenizer(args.tokenizer)
    tw = HFTokenizerWrapper(tok)
    # load data
    text = Path(args.data).read_text(encoding='utf-8')
    ids = tw.encode(text)
    console.print(f"Corpus tokens: {len(ids)} (vocab_size={tw.vocab_size})")
    # train/val split
    split = int(len(ids) * (1 - args.val_ratio))
    train_ids = ids[:split]
    val_ids = ids[split:]

    train_ds = CausalTextDataset(train_ids, block_size=args.block_size, randomize=True)
    val_ds = CausalTextDataset(val_ids, block_size=args.block_size, randomize=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TinyGPT(vocab_size=tw.vocab_size, block_size=args.block_size, n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head, drop=args.drop)
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    train_loop(model, tw, train_loader, val_loader, out_dir=args.outdir, epochs=args.epochs, lr=args.lr, device=device, save_every_steps=args.save_every_steps)
    console.print("Training finished.")


def cmd_generate(args):
    tok = load_tokenizer(args.tokenizer)
    tw = HFTokenizerWrapper(tok)
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    # build model
    # We need vocab_size and block_size: try to read from args or infer
    model = TinyGPT(vocab_size=tok.get_vocab_size(), block_size=args.block_size, n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head, drop=args.drop)
    # load ckpt
    ckpt = torch.load(args.ckpt, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    model.to(device)
    out = generate_text(model, tw, args.prompt, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, device=device)
    console.print("\n[bold green]Generated text:[/]\n")
    console.print(out)


def build_parser():
    p = argparse.ArgumentParser(description='Mini-LLM CLI (train, tokenizer, generate)')
    sub = p.add_subparsers(dest='cmd')

    p_tok = sub.add_parser('init-tokenizer', help='Train a BPE tokenizer from text files')
    p_tok.add_argument('--files', nargs='+', required=True)
    p_tok.add_argument('--vocab-size', type=int, default=16000)
    p_tok.add_argument('--out', type=str, default='tokenizer.json')

    p_train = sub.add_parser('train', help='Train a tiny GPT model')
    p_train.add_argument('--tokenizer', type=str, required=True)
    p_train.add_argument('--data', type=str, required=True)
    p_train.add_argument('--outdir', type=str, default='runs/exp')
    p_train.add_argument('--block-size', type=int, default=32)
    p_train.add_argument('--batch-size', type=int, default=64)
    p_train.add_argument('--epochs', type=int, default=5)
    p_train.add_argument('--lr', type=float, default=3e-4)
    p_train.add_argument('--n-embd', type=int, default=128)
    p_train.add_argument('--n-layer', type=int, default=4)
    p_train.add_argument('--n-head', type=int, default=4)
    p_train.add_argument('--drop', type=float, default=0.1)
    p_train.add_argument('--val-ratio', type=float, default=0.1)
    p_train.add_argument('--device', type=str, default='')
    p_train.add_argument('--save-every-steps', type=int, default=1000)

    p_gen = sub.add_parser('generate', help='Generate from a checkpoint')
    p_gen.add_argument('--tokenizer', type=str, required=True)
    p_gen.add_argument('--ckpt', type=str, required=True)
    p_gen.add_argument('--prompt', type=str, default='')
    p_gen.add_argument('--max-tokens', type=int, default=50)
    p_gen.add_argument('--temperature', type=float, default=1.0)
    p_gen.add_argument('--top-k', type=int, default=50)
    p_gen.add_argument('--top-p', type=float, default=0.0)
    p_gen.add_argument('--block-size', type=int, default=32)
    p_gen.add_argument('--n-embd', type=int, default=128)
    p_gen.add_argument('--n-layer', type=int, default=4)
    p_gen.add_argument('--n-head', type=int, default=4)
    p_gen.add_argument('--drop', type=float, default=0.1)
    p_gen.add_argument('--device', type=str, default='')

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()
    if not args.cmd:
        parser.print_help()
        return
    if args.cmd == 'init-tokenizer':
        cmd_init_tokenizer(args)
    elif args.cmd == 'train':
        cmd_train(args)
    elif args.cmd == 'generate':
        cmd_generate(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()
