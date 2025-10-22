#!/usr/bin/env python3
"""
main.py - VERSIÓN CORREGIDA

CLI mejorado para Mini-LLM con validación robusta de inputs.

CORRECCIONES:
- Validación de archivos antes de cargar
- Manejo de errores comprehensivo
- Mensajes de error claros
- Protección contra corpus vacíos/corruptos
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
import traceback

# Al inicio del archivo, ANTES de imports de torch:
def ensure_torch():
    """Verifica que torch esté disponible antes de continuar"""
    try:
        import torch
        return True
    except ImportError:
        console.print("[red]❌ PyTorch no está instalado[/]")
        console.print("\n[yellow]💡 Instala con:[/]")
        console.print("   pip install torch --index-url https://download.pytorch.org/whl/cu118")
        return False

# Luego, en cada función que use torch:
def cmd_train(args):
    if not ensure_torch():
        sys.exit(1)
    
    import torch  # Import local
    from torch.utils.data import DataLoader
    # ... resto del código

import torch
from torch.utils.data import DataLoader

# Imports locales
from model import TinyGPTv2, create_small_model, create_medium_model, create_large_model
from dataset import (
    DocumentAwareDataset, CausalTextDataset,
    prepare_datasets, analyze_corpus_structure
)
from training import TrainingConfig, train, load_checkpoint
from generation import GenerationConfig, generate, generate_batch, compute_perplexity, print_generation_stats

# ======================== PyTorch 2.6+ Compatibility ========================
# SAFE_GLOBALS_REGISTERED: Registra clases custom para torch.load
# Esto permite cargar checkpoints con TrainingHistory de forma segura

try:
    import torch
    from training import TrainingHistory
    
    # Registra TrainingHistory como clase segura
    # Esto es la solución RECOMENDADA por PyTorch 2.6+
    if hasattr(torch.serialization, 'add_safe_globals'):
        torch.serialization.add_safe_globals([TrainingHistory])
        # También registra dataclasses si es necesario
        import dataclasses
        torch.serialization.add_safe_globals([dataclasses.dataclass])
except ImportError:
    # PyTorch < 2.6 o training no disponible aún
    pass
except Exception as e:
    # No crítico si falla, usaremos weights_only=False como fallback
    pass

# ============================================================================


# Tokenizer
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.normalizers import NFKC

# CLI visuals
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import track

console = Console()


# ======================== Validación de Inputs ========================

def normalize_path(path_str: str) -> Path:
    """Normaliza paths para Windows/Unix"""
    path = Path(path_str)
    # Resuelve paths relativos
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve()

def validate_file_exists(filepath: str, file_type: str = "archivo") -> Path:
    """Valida que un archivo exista."""
    path = Path(filepath)
    if not path.exists():
        console.print(f"[red]❌ Error: {file_type} no encontrado: {filepath}[/]")
        sys.exit(1)
    if not path.is_file():
        console.print(f"[red]❌ Error: {filepath} no es un archivo válido[/]")
        sys.exit(1)
    return path


def validate_corpus(filepath: Path) -> str:
    """Valida y carga un corpus de texto."""
    try:
        text = filepath.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        try:
            text = filepath.read_text(encoding='latin-1')
            console.print("[yellow]⚠️  Corpus decodificado como latin-1 (no UTF-8)[/]")
        except Exception as e:
            console.print(f"[red]❌ Error al leer corpus: {e}[/]")
            sys.exit(1)

    if len(text.strip()) == 0:
        console.print(f"[red]❌ Error: El corpus está vacío: {filepath}[/]")
        sys.exit(1)

    if len(text) < 1000:
        console.print(f"[yellow]⚠️  Advertencia: Corpus muy pequeño ({len(text)} chars). "
                     "Se recomienda al menos 100KB para resultados razonables.[/]")

    return text


def validate_tokenizer(filepath: Path) -> Tokenizer:
    """Valida y carga un tokenizer."""
    try:
        tokenizer = Tokenizer.from_file(str(filepath))
        return tokenizer
    except Exception as e:
        console.print(f"[red]❌ Error al cargar tokenizer: {e}[/]")
        console.print(f"[yellow]💡 Tip: Asegúrate de que el archivo sea un tokenizer válido (.json)[/]")
        sys.exit(1)


def validate_checkpoint(filepath: Path) -> dict:
    """Valida y carga un checkpoint."""
    try:
        ckpt = torch.load(filepath, map_location='cpu', weights_only=False)  # TORCH_LOAD_FIX_APPLIED

        # Verifica claves necesarias
        required_keys = ['model_state_dict', 'model_config']
        missing_keys = [k for k in required_keys if k not in ckpt]

        if missing_keys:
            console.print(f"[red]❌ Checkpoint corrupto. Faltan claves: {missing_keys}[/]")
            sys.exit(1)

        return ckpt
    except Exception as e:
        console.print(f"[red]❌ Error al cargar checkpoint: {e}[/]")
        sys.exit(1)


def safe_create_directory(path: Path):
    """Crea directorio de forma segura."""
    try:
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        console.print(f"[red]❌ Error al crear directorio {path}: {e}[/]")
        sys.exit(1)


# ======================== Tokenizer ========================

class HFTokenizerWrapper:
    """Wrapper para tokenizer de HuggingFace."""
    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self.vocab_size = tok.get_vocab_size()

    def encode(self, text: str) -> List[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tok.decode(ids)

    def get_special_token_id(self, token: str) -> Optional[int]:
        """Obtiene ID de un token especial."""
        vocab = self.tok.get_vocab()
        return vocab.get(token)


def train_bpe_tokenizer(
    files: List[str],
    vocab_size: int = 32000,
    special_tokens: List[str] = None,
    out: str = 'tokenizer.json'
) -> Tokenizer:
    """Entrena tokenizer BPE."""
    if special_tokens is None:
        special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '<|doc|>', '<|endoc|>']

    # Valida archivos
    validated_files = []
    for f in files:
        path = validate_file_exists(f, "archivo de entrenamiento")
        validated_files.append(str(path))

    console.print(f"\n🔤 Entrenando tokenizer BPE...")
    console.print(f"   Archivos: {len(validated_files)}")
    console.print(f"   Vocab size: {vocab_size:,}")
    console.print(f"   Tokens especiales: {special_tokens}")

    try:
        tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
        tokenizer.normalizer = NFKC()
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
            show_progress=True
        )

        tokenizer.train(validated_files, trainer)
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.post_processor = processors.ByteLevel(trim_offsets=True)
        tokenizer.save(out)

        console.print(f"✅ Tokenizer guardado en [bold]{out}[/]")
        return tokenizer
    except Exception as e:
        console.print(f"[red]❌ Error al entrenar tokenizer: {e}[/]")
        console.print(f"[yellow]💡 Verifica que los archivos contengan texto válido[/]")
        sys.exit(1)


# ======================== Comandos CLI ========================

def cmd_init_tokenizer(args):
    """Comando: init-tokenizer"""
    tok = train_bpe_tokenizer(
        files=args.files,
        vocab_size=args.vocab_size,
        out=args.out
    )

    # Test del tokenizer
    test_texts = [
        "La fotosíntesis es un proceso bioquímico fundamental.",
        "Durante el Imperio Romano, la tecnología avanzó significativamente.",
        "El teorema de Pitágoras establece que a² + b² = c²."
    ]

    console.print("\n🔍 Ejemplos de tokenización:")
    for text in test_texts[:2]:
        ids = tok.encode(text).ids
        tokens = [tok.id_to_token(i) for i in ids[:15]]
        console.print(f"  Texto: {text[:50]}...")
        console.print(f"  Tokens: {tokens}")
        console.print(f"  Ratio: {len(text) / len(ids):.2f} chars/token\n")


def cmd_prepare_corpus(args):
    """Comando: prepare-corpus (limpieza de texto)"""
    try:
        import ftfy
        import unicodedata
        import re
    except ImportError:
        console.print("[red]❌ Error: ftfy no instalado. Ejecuta: pip install ftfy[/]")
        sys.exit(1)

    # Valida archivo de entrada
    input_path = validate_file_exists(args.input, "corpus de entrada")

    console.print(f"\n🧹 Preparando corpus...")
    console.print(f"   Input: {args.input}")
    console.print(f"   Output: {args.output}")

    # Lee corpus
    try:
        text = input_path.read_text(encoding='utf-8-sig')
    except UnicodeDecodeError:
        try:
            text = input_path.read_text(encoding='latin-1')
            console.print("[yellow]⚠️  Decodificado como latin-1[/]")
        except Exception as e:
            console.print(f"[red]❌ Error al leer archivo: {e}[/]")
            sys.exit(1)

    console.print(f"   Tamaño original: {len(text):,} chars")

    if len(text.strip()) == 0:
        console.print("[red]❌ Error: El archivo de entrada está vacío[/]")
        sys.exit(1)

    # Limpieza
    text = ftfy.fix_text(text)
    text = unicodedata.normalize('NFKC', text)

    # Reemplazos comunes
    text = text.replace('\u00A0', ' ')
    text = text.replace('\u200B', '')
    text = text.replace('\u2013', '-')
    text = text.replace('\u2014', '-')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    text = text.replace('\u201C', '"').replace('\u201D', '"')
    text = re.sub(r'\s+', ' ', text)

    if not args.preserve_case:
        text = text.lower()

    if args.remove_control:
        text = ''.join(
            ch for ch in text
            if unicodedata.category(ch)[0] != "C" or ch in ("\n", "\t")
        )

    text = text.strip()

    if len(text) < 100:
        console.print("[yellow]⚠️  Advertencia: Corpus muy pequeño después de limpieza[/]")

    # Guarda
    try:
        Path(args.output).write_text(text, encoding='utf-8')
    except Exception as e:
        console.print(f"[red]❌ Error al guardar archivo: {e}[/]")
        sys.exit(1)

    console.print(f"   Tamaño final: {len(text):,} chars")
    original_len = len(input_path.read_text(encoding='utf-8-sig', errors='ignore'))
    if original_len > 0:
        console.print(f"   Reducción: {(1 - len(text) / original_len)*100:.1f}%")
    console.print(f"✅ Corpus limpio guardado en [bold]{args.output}[/]")


def cmd_analyze_corpus(args):
    """Comando: analyze-corpus"""
    corpus_path = validate_file_exists(args.corpus, "corpus")

    console.print(f"\n🔍 Analizando corpus: {args.corpus}")

    # Lee corpus
    text = validate_corpus(corpus_path)

    # Mock tokenizer simple (split por espacios)
    mock_ids = list(range(len(text.split())))

    console.print(f"   Chars: {len(text):,}")
    console.print(f"   Palabras (aprox): {len(text.split()):,}")

    # Si hay separador, analiza estructura
    if args.separator:
        # Crea IDs con separadores
        docs = text.split(args.separator)

        if len(docs) < 2:
            console.print(f"[yellow]⚠️  Separador '{args.separator}' no encontrado en el corpus[/]")
            console.print("   Se tratará como un solo documento")
        else:
            ids_with_sep = []
            sep_id = 999999

            for doc in docs:
                doc_words = doc.split()
                ids_with_sep.extend(range(len(doc_words)))
                ids_with_sep.append(sep_id)

            analyze_corpus_structure(ids_with_sep, separator_id=sep_id)
    else:
        console.print("\n💡 No se especificó separador. Usa --separator para análisis detallado.")
        console.print(f"   Sugerencias:")
        console.print(f"   - Block size: 256-512 (para corpus general)")
        console.print(f"   - Vocab size: 16000-32000")


def cmd_train(args):
    """Comando: train"""
    console.print(Panel.fit(
        "[bold cyan]🚀 ENTRENAMIENTO DE MODELO[/]",
        border_style="cyan"
    ))

    # Valida inputs
    tokenizer_path = validate_file_exists(args.tokenizer, "tokenizer")
    corpus_path = validate_file_exists(args.corpus, "corpus")

    # Carga tokenizer
    console.print(f"\n📚 Cargando tokenizer: {args.tokenizer}")
    tok = validate_tokenizer(tokenizer_path)
    tw = HFTokenizerWrapper(tok)
    console.print(f"   Vocab size: {tw.vocab_size:,}")

    # Carga corpus
    console.print(f"\n📖 Cargando corpus: {args.corpus}")
    text = validate_corpus(corpus_path)

    try:
        ids = tw.encode(text)
    except Exception as e:
        console.print(f"[red]❌ Error al tokenizar corpus: {e}[/]")
        sys.exit(1)

    if len(ids) == 0:
        console.print("[red]❌ Error: El corpus tokenizado está vacío[/]")
        sys.exit(1)

    if len(ids) < args.block_size * 10:
        console.print(f"[yellow]⚠️  Advertencia: Corpus muy pequeño ({len(ids)} tokens). "
                     f"Se recomiendan al menos {args.block_size * 100} tokens.[/]")

    console.print(f"   Tokens: {len(ids):,}")

    # Obtiene ID del separador si existe
    doc_sep_id = None
    if args.doc_separator:
        doc_sep_id = tw.get_special_token_id(args.doc_separator)
        if doc_sep_id is None:
            console.print(f"[yellow]⚠️  Separador '{args.doc_separator}' no encontrado en vocabulario[/]")

    # Prepara datasets
    try:
        train_ds, val_ds = prepare_datasets(
            ids=ids,
            block_size=args.block_size,
            val_ratio=args.val_ratio,
            doc_separator_id=doc_sep_id,
            stride=args.stride,
            use_document_aware=not args.simple_dataset
        )
    except Exception as e:
        console.print(f"[red]❌ Error al preparar datasets: {e}[/]")
        console.print(f"[yellow]💡 Tip: Intenta reducir block_size o aumentar el tamaño del corpus[/]")
        sys.exit(1)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0 if sys.platform == 'win32' else args.num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    # Check win num_workers
    if sys.platform == 'win32' and args.num_workers > 0:
        console.print("[yellow]⚠️  En Windows, num_workers forzado a 0[/]")
        args.num_workers = 0

    # Crea modelo
    console.print(f"\n🗂️ Creando modelo...")

    try:
        if args.config == 'small':
            model = create_small_model(tw.vocab_size, args.block_size)
        elif args.config == 'medium':
            model = create_medium_model(tw.vocab_size, args.block_size)
        elif args.config == 'large':
            model = create_large_model(tw.vocab_size, args.block_size)
        else:
            # Configuración custom
            model = TinyGPTv2(
                vocab_size=tw.vocab_size,
                block_size=args.block_size,
                n_embd=args.n_embd,
                n_layer=args.n_layer,
                n_head=args.n_head,
                drop=args.drop,
                use_rope=not args.no_rope,
                use_checkpoint=args.gradient_checkpointing
            )
    except Exception as e:
        console.print(f"[red]❌ Error al crear modelo: {e}[/]")
        sys.exit(1)

    num_params = model.get_num_params()
    console.print(f"   Parámetros: {num_params:,}")
    console.print(f"   Arquitectura: {args.config if args.config != 'custom' else 'custom'}")

    # Configuración de entrenamiento
    train_config = TrainingConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        grad_clip=args.grad_clip,
        accumulation_steps=args.accumulation_steps,
        epochs=args.epochs,
        eval_every=args.eval_every,
        save_every=args.save_every,
        patience=args.patience,
        use_amp=not args.no_amp,
        device=args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    )

    # Entrena
    out_dir = Path(args.outdir)
    safe_create_directory(out_dir)

    resume_from = None
    if args.resume:
        resume_path = validate_file_exists(args.resume, "checkpoint")
        validate_checkpoint(resume_path)
        resume_from = resume_path

    try:
        history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=train_config,
            out_dir=out_dir,
            resume_from=resume_from
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Entrenamiento interrumpido por usuario[/]")
        sys.exit(0)
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            console.print(f"\n[red]❌ Error: Out of memory (OOM)[/]")
            console.print("[yellow]💡 Intenta:")
            console.print("   - Reducir batch_size")
            console.print("   - Reducir block_size")
            console.print("   - Usar gradient checkpointing (--gradient-checkpointing)")
            console.print("   - Usar un modelo más pequeño")
        else:
            console.print(f"\n[red]❌ Error durante entrenamiento: {e}[/]")
            traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]❌ Error inesperado: {e}[/]")
        traceback.print_exc()
        sys.exit(1)

    console.print(f"\n✅ Entrenamiento completado")
    console.print(f"   Mejor val_loss: {history.best_val_loss:.4f}")
    console.print(f"   Checkpoints en: {out_dir}")


def cmd_generate(args):
    """Comando: generate"""
    console.print(Panel.fit(
        "[bold green]✨ GENERACIÓN DE TEXTO[/]",
        border_style="green"
    ))

    # Valida inputs
    tokenizer_path = validate_file_exists(args.tokenizer, "tokenizer")
    ckpt_path = validate_file_exists(args.ckpt, "checkpoint")

    # Carga tokenizer
    console.print(f"\n📚 Cargando tokenizer: {args.tokenizer}")
    tok = validate_tokenizer(tokenizer_path)
    tw = HFTokenizerWrapper(tok)

    # Carga modelo
    console.print(f"\n🤖 Cargando modelo: {args.ckpt}")
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')

    # Lee config del checkpoint
    ckpt = validate_checkpoint(ckpt_path)
    model_config = ckpt.get('model_config', {})

    try:
        model = TinyGPTv2(
            vocab_size=model_config.get('vocab_size', tw.vocab_size),
            block_size=model_config.get('block_size', args.block_size),
            n_embd=model_config.get('n_embd', args.n_embd),
            n_layer=model_config.get('n_layer', args.n_layer),
            n_head=model_config.get('n_head', args.n_head),
            use_rope=model_config.get('use_rope', True)
        )

        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()
    except Exception as e:
        console.print(f"[red]❌ Error al cargar modelo: {e}[/]")
        sys.exit(1)

    console.print(f"   Device: {device}")
    console.print(f"   Parámetros: {model.get_num_params():,}")

    # Configuración de generación
    gen_config = GenerationConfig(
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        do_sample=not args.greedy
    )

    # Lee prompts
    if args.prompt_file:
        prompt_file_path = validate_file_exists(args.prompt_file, "archivo de prompts")
        try:
            prompts = prompt_file_path.read_text(encoding='utf-8').strip().split('\n')
            prompts = [p for p in prompts if p.strip()]  # Filtra líneas vacías
        except Exception as e:
            console.print(f"[red]❌ Error al leer prompts: {e}[/]")
            sys.exit(1)
    else:
        prompts = [args.prompt] if args.prompt else [""]

    if not prompts or all(not p.strip() for p in prompts):
        console.print("[yellow]⚠️  Advertencia: No se proporcionaron prompts. Generando desde vacío.[/]")
        prompts = [""]

    console.print(f"\n🎨 Generando {len(prompts)} texto(s)...\n")

    # Genera
    try:
        if len(prompts) == 1:
            # Generación única con streaming
            prompt = prompts[0]
            console.print(f"[bold cyan]Prompt:[/] {prompt}\n")
            console.print("[bold green]Generación:[/]")

            if args.stream:
                # Streaming
                def stream_callback(token_text):
                    console.print(token_text, end='')

                generated = generate(model, tw, prompt, gen_config, device, stream_callback)
                console.print()  # Nueva línea
            else:
                # Sin streaming
                generated = generate(model, tw, prompt, gen_config, device)
                # Imprime solo la parte generada (sin prompt)
                new_text = generated[len(prompt):]
                console.print(new_text)

            if args.stats:
                print_generation_stats(prompt, generated, tw)

            # Guarda si se especifica
            if args.output:
                try:
                    Path(args.output).write_text(generated, encoding='utf-8')
                    console.print(f"\n💾 Guardado en: {args.output}")
                except Exception as e:
                    console.print(f"[red]❌ Error al guardar: {e}[/]")

        else:
            # Generación batch
            if args.batch_generate:
                generated_texts = generate_batch(model, tw, prompts, gen_config, device)
            else:
                generated_texts = [
                    generate(model, tw, prompt, gen_config, device)
                    for prompt in track(prompts, description="Generando...")
                ]

            # Muestra resultados
            table = Table(title="Generaciones", show_lines=True)
            table.add_column("Prompt", style="cyan", width=40)
            table.add_column("Generación", style="green", width=60)

            for prompt, generated in zip(prompts, generated_texts):
                new_text = generated[len(prompt):].strip()[:200]
                table.add_row(prompt[:40], new_text + "...")

            console.print(table)

            # Guarda si se especifica
            if args.output:
                try:
                    output_text = "\n\n" + "="*70 + "\n\n".join(generated_texts)
                    Path(args.output).write_text(output_text, encoding='utf-8')
                    console.print(f"\n💾 Guardado en: {args.output}")
                except Exception as e:
                    console.print(f"[red]❌ Error al guardar: {e}[/]")

    except Exception as e:
        console.print(f"\n[red]❌ Error durante generación: {e}[/]")
        traceback.print_exc()
        sys.exit(1)


def cmd_evaluate(args):
    """Comando: evaluate (perplexity en test set)"""
    console.print(Panel.fit(
        "[bold magenta]📊 EVALUACIÓN DE MODELO[/]",
        border_style="magenta"
    ))

    # Valida inputs
    tokenizer_path = validate_file_exists(args.tokenizer, "tokenizer")
    ckpt_path = validate_file_exists(args.ckpt, "checkpoint")

    # Carga tokenizer
    tok = validate_tokenizer(tokenizer_path)
    tw = HFTokenizerWrapper(tok)

    # Carga modelo
    device = args.device if args.device else ('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = validate_checkpoint(ckpt_path)
    model_config = ckpt.get('model_config', {})

    try:
        model = TinyGPTv2(
            vocab_size=model_config.get('vocab_size', tw.vocab_size),
            block_size=model_config.get('block_size', 256),
            n_embd=model_config.get('n_embd', 384),
            n_layer=model_config.get('n_layer', 8),
            n_head=model_config.get('n_head', 6),
            use_rope=model_config.get('use_rope', True)
        )

        model.load_state_dict(ckpt['model_state_dict'])
        model.to(device)
        model.eval()
    except Exception as e:
        console.print(f"[red]❌ Error al cargar modelo: {e}[/]")
        sys.exit(1)

    console.print(f"✅ Modelo cargado")

    # Lee test texts
    if args.test_file:
        test_file_path = validate_file_exists(args.test_file, "archivo de test")
        try:
            text = test_file_path.read_text(encoding='utf-8')
            test_texts = text.strip().split('\n\n')  # Divide por párrafos
            test_texts = [t for t in test_texts if t.strip()]
        except Exception as e:
            console.print(f"[red]❌ Error al leer test file: {e}[/]")
            sys.exit(1)
    else:
        test_texts = [args.test_text] if args.test_text else []

    if not test_texts:
        console.print("[red]❌ Error: Especifica --test-file o --test-text[/]")
        sys.exit(1)

    console.print(f"\n📊 Calculando perplexity en {len(test_texts)} texto(s)...")

    # Calcula perplexity
    try:
        perplexities = compute_perplexity(model, tw, test_texts, device)
    except Exception as e:
        console.print(f"[red]❌ Error al calcular perplexity: {e}[/]")
        sys.exit(1)

    # Muestra resultados
    table = Table(title="Resultados de Evaluación")
    table.add_column("Texto", style="cyan", width=50)
    table.add_column("Perplexity", style="green", justify="right")
    table.add_column("Tokens", style="yellow", justify="right")

    for text, ppl in zip(test_texts, perplexities):
        tokens = len(tw.encode(text))
        table.add_row(
            text[:50] + "..." if len(text) > 50 else text,
            f"{ppl:.2f}",
            f"{tokens:,}"
        )

    console.print(table)

    # Estadísticas generales
    valid_ppls = [p for p in perplexities if p != float('inf')]
    if valid_ppls:
        avg_ppl = sum(valid_ppls) / len(valid_ppls)
        console.print(f"\n📈 Perplexity promedio: {avg_ppl:.2f}")


# ======================== Argument Parser ========================

def build_parser():
    """Construye parser de argumentos."""
    parser = argparse.ArgumentParser(
        description='Mini-LLM v2: CLI mejorado para entrenar modelos de lenguaje',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(dest='command', help='Comando a ejecutar')

    # ===== init-tokenizer =====
    p_tok = subparsers.add_parser('init-tokenizer', help='Entrena tokenizer BPE')
    p_tok.add_argument('--files', nargs='+', required=True, help='Archivos de texto para entrenar')
    p_tok.add_argument('--vocab-size', type=int, default=32000, help='Tamaño del vocabulario')
    p_tok.add_argument('--out', type=str, default='tokenizer.json', help='Archivo de salida')

    # ===== prepare-corpus =====
    p_prep = subparsers.add_parser('prepare-corpus', help='Limpia y prepara corpus')
    p_prep.add_argument('--input', type=str, required=True, help='Archivo de entrada')
    p_prep.add_argument('--output', type=str, required=True, help='Archivo de salida')
    p_prep.add_argument('--preserve-case', action='store_true', help='Mantener mayúsculas')
    p_prep.add_argument('--remove-control', action='store_true', default=True, help='Eliminar caracteres de control')

    # ===== analyze-corpus =====
    p_analyze = subparsers.add_parser('analyze-corpus', help='Analiza estructura del corpus')
    p_analyze.add_argument('--corpus', type=str, required=True, help='Archivo del corpus')
    p_analyze.add_argument('--separator', type=str, help='Token separador de documentos (e.g., <|doc|>)')

    # ===== train =====
    p_train = subparsers.add_parser('train', help='Entrena modelo')
    p_train.add_argument('--tokenizer', type=str, required=True, help='Tokenizer (.json)')
    p_train.add_argument('--corpus', type=str, required=True, help='Corpus de entrenamiento')
    p_train.add_argument('--outdir', type=str, default='runs/exp', help='Directorio de salida')

    # Model architecture
    p_train.add_argument('--config', type=str, default='small', choices=['small', 'medium', 'large', 'custom'],
                         help='Configuración predefinida del modelo')
    p_train.add_argument('--block-size', type=int, default=256, help='Longitud de contexto')
    p_train.add_argument('--n-embd', type=int, default=384, help='Dimensión de embeddings')
    p_train.add_argument('--n-layer', type=int, default=8, help='Número de capas')
    p_train.add_argument('--n-head', type=int, default=6, help='Número de attention heads')
    p_train.add_argument('--drop', type=float, default=0.1, help='Dropout rate')
    p_train.add_argument('--no-rope', action='store_true', help='No usar RoPE (usar pos embeddings aprendidos)')
    p_train.add_argument('--gradient-checkpointing', action='store_true', help='Activar gradient checkpointing')

    # Dataset
    p_train.add_argument('--doc-separator', type=str, default='<|doc|>', help='Token separador de documentos')
    p_train.add_argument('--stride', type=int, help='Stride para overlapping windows (default: block_size/2)')
    p_train.add_argument('--simple-dataset', action='store_true', help='Usar dataset simple (no document-aware)')
    p_train.add_argument('--val-ratio', type=float, default=0.1, help='Ratio de datos para validación')

    # Training
    p_train.add_argument('--batch-size', type=int, default=32, help='Batch size')
    p_train.add_argument('--epochs', type=int, default=20, help='Número de épocas')
    p_train.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    p_train.add_argument('--weight-decay', type=float, default=0.01, help='Weight decay')
    p_train.add_argument('--warmup-steps', type=int, default=500, help='Pasos de warmup')
    p_train.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping')
    p_train.add_argument('--accumulation-steps', type=int, default=1, help='Gradient accumulation steps')
    p_train.add_argument('--patience', type=int, default=5, help='Early stopping patience')
    p_train.add_argument('--no-amp', action='store_true', help='Desactivar mixed precision')

    # Logging & checkpointing
    p_train.add_argument('--eval-every', type=int, default=500, help='Evaluar cada N pasos')
    p_train.add_argument('--save-every', type=int, default=2000, help='Guardar checkpoint cada N pasos')
    p_train.add_argument('--num-workers', type=int, default=0, help='DataLoader workers')

    # Resume
    p_train.add_argument('--resume', type=str, help='Checkpoint para continuar entrenamiento')
    p_train.add_argument('--device', type=str, help='Device (cuda/cpu)')

    # ===== generate =====
    p_gen = subparsers.add_parser('generate', help='Genera texto')
    p_gen.add_argument('--tokenizer', type=str, required=True, help='Tokenizer (.json)')
    p_gen.add_argument('--ckpt', type=str, required=True, help='Checkpoint del modelo')
    p_gen.add_argument('--prompt', type=str, default='', help='Prompt para generación')
    p_gen.add_argument('--prompt-file', type=str, help='Archivo con múltiples prompts (uno por línea)')
    p_gen.add_argument('--output', type=str, help='Guardar resultado en archivo')

    # Generation params
    p_gen.add_argument('--max-tokens', type=int, default=100, help='Tokens máximos a generar')
    p_gen.add_argument('--temperature', type=float, default=0.8, help='Temperature')
    p_gen.add_argument('--top-k', type=int, default=50, help='Top-k sampling')
    p_gen.add_argument('--top-p', type=float, default=0.9, help='Top-p (nucleus) sampling')
    p_gen.add_argument('--repetition-penalty', type=float, default=1.0, help='Repetition penalty')
    p_gen.add_argument('--greedy', action='store_true', help='Usar greedy decoding (argmax)')
    p_gen.add_argument('--stream', action='store_true', help='Streaming de tokens')
    p_gen.add_argument('--batch-generate', action='store_true', help='Generación batch (más rápido)')
    p_gen.add_argument('--stats', action='store_true', help='Mostrar estadísticas')

    # Model config (para cargar sin metadata)
    p_gen.add_argument('--block-size', type=int, default=256)
    p_gen.add_argument('--n-embd', type=int, default=384)
    p_gen.add_argument('--n-layer', type=int, default=8)
    p_gen.add_argument('--n-head', type=int, default=6)
    p_gen.add_argument('--device', type=str, help='Device (cuda/cpu)')

    # ===== evaluate =====
    p_eval = subparsers.add_parser('evaluate', help='Evalúa modelo (perplexity)')
    p_eval.add_argument('--tokenizer', type=str, required=True, help='Tokenizer (.json)')
    p_eval.add_argument('--ckpt', type=str, required=True, help='Checkpoint del modelo')
    p_eval.add_argument('--test-file', type=str, help='Archivo con textos de test')
    p_eval.add_argument('--test-text', type=str, help='Texto único para evaluar')
    p_eval.add_argument('--device', type=str, help='Device (cuda/cpu)')

    return parser


# ======================== Main ========================

def main():
    """Punto de entrada principal."""
    parser = build_parser()
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Ejecuta comando con manejo robusto de errores
    try:
        if args.command == 'init-tokenizer':
            cmd_init_tokenizer(args)
        elif args.command == 'prepare-corpus':
            cmd_prepare_corpus(args)
        elif args.command == 'analyze-corpus':
            cmd_analyze_corpus(args)
        elif args.command == 'train':
            cmd_train(args)
        elif args.command == 'generate':
            cmd_generate(args)
        elif args.command == 'evaluate':
            cmd_evaluate(args)
        else:
            parser.print_help()

    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Operación cancelada por usuario[/]")
        sys.exit(0)

    except Exception as e:
        console.print(f"\n[bold red]❌ Error inesperado: {e}[/]")
        console.print("\n[yellow]Stack trace:[/]")
        traceback.print_exc()
        console.print("\n[cyan]💡 Tips para debugging:[/]")
        console.print("   1. Verifica que todos los archivos existan")
        console.print("   2. Asegúrate de tener las dependencias instaladas")
        console.print("   3. Revisa que el corpus no esté corrupto")
        console.print("   4. Si es OOM, reduce batch_size o block_size")
        sys.exit(1)


if __name__ == '__main__':
    main()
