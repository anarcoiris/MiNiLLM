#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
patches_critical.py - Script para aplicar parches críticos a Mini-LLM v2

Uso:
    python patches_critical.py --check    # Solo verifica problemas
    python patches_critical.py --apply    # Aplica correcciones
    python patches_critical.py --backup   # Crea backup antes de aplicar

Problemas corregidos (lista de parches incluidos en este script):
1. Validación de stride en dataset.py
2. Corrección de lógica last_start en dataset.py
3. Mismo fix para modo memory-efficient en dataset.py
4. Limpieza de GPU memory entre epochs (training.py)
5. Validación de vocab_size extremos (main.py)
6. Eliminación de variable mock_ids sin uso (main.py)
7. Documentación/warning sobre memory_efficient en dataset.py

Nota: este script evita caracteres no ASCII "mojibake" en los mensajes y cadenas
para prevenir errores como "SyntaxError: invalid decimal literal" al ejecutarlo
en entornos con archivos parcheados o patches diffs mal codificados.
"""
from __future__ import annotations

import sys
import shutil
from pathlib import Path
from typing import List, Tuple
import re


class Patch:
    """Representa un parche a aplicar."""
    def __init__(self, file: str, description: str, search: str, replace: str, line_hint: int = None):
        self.file = file
        self.description = description
        self.search = search
        self.replace = replace
        self.line_hint = line_hint

    def apply(self, content: str) -> Tuple[str, bool]:
        """Aplica el parche. Returns (nuevo_contenido, exito)."""
        # Intentar reemplazo literal primero
        if self.search in content:
            new_content = content.replace(self.search, self.replace, 1)
            return new_content, True

        # Si no se encuentra, intentar una búsqueda más permisiva (espacios / indentación)
        # Convierte multiples espacios en regex y busca primer match
        try:
            # Escape search to regex but allow flexible indentation
            pattern = re.escape(self.search)
            # Replace runs of whitespace in escaped pattern with '\s+' to be permissive
            pattern = re.sub(r'\\\s+', r'\\s+', pattern)
            m = re.search(pattern, content, flags=re.MULTILINE)
            if m:
                start, end = m.span()
                new_content = content[:start] + self.replace + content[end:]
                return new_content, True
        except re.error:
            pass

        return content, False


# ============================================================================
# PARCHES CRÍTICOS
# ============================================================================

PATCHES = [
    # PATCH 1: Validación de stride en dataset.py
    Patch(
        file="dataset.py",
        description="[CRITICO] Añadir validación stride <= block_size",
        search=(
            "        # Stride por defecto: 50% overlap\n"
            "        self.stride = stride if stride is not None else block_size // 2\n\n"
            "        # Longitud mínima por defecto: al menos 2 tokens más que block_size\n"
            "        self.min_doc_length = min_doc_length if min_doc_length is not None else block_size + 2"
        ),
        replace=(
            "        # Stride por defecto: 50% overlap\n"
            "        self.stride = stride if stride is not None else block_size // 2\n\n"
            "        # Validacion: stride no puede ser mayor que block_size\n"
            "        if self.stride > block_size:\n"
            "            raise ValueError(\n"
            "                f\"stride ({self.stride}) no puede ser mayor que block_size ({block_size}). \"\n"
            "                \"Esto causaria saltos en la cobertura del corpus.\"\n"
            "            )\n\n"
            "        if self.stride <= 0:\n"
            "            raise ValueError(f\"stride debe ser > 0, recibido: {self.stride}\")\n\n"
            "        # Longitud minima por defecto: al menos 2 tokens mas que block_size\n"
            "        self.min_doc_length = min_doc_length if min_doc_length is not None else block_size + 2"
        ),
        line_hint=66
    ),

    # PATCH 2: Correccion de logica last_start en dataset.py
    Patch(
        file="dataset.py",
        description="[CRITICO] Corregir logica de ultima ventana en documentos",
        search=(
            "                # Añade última ventana si no está ya incluida\n"
            "                last_start = doc_len - block_size - 1\n"
            "                if last_start >= 0 and (not self.samples or self.samples[-1][1] < last_start):\n"
            "                    self.samples.append((doc_idx, last_start))\n"
            "                    for pos in range(last_start, doc_len):\n"
            "                        unique_tokens_covered.add((doc_idx, pos))"
        ),
        replace=(
            "                # Añade ultima ventana si no esta ya incluida\n"
            "                # Asegura que cubrimos hasta el final del documento\n"
            "                last_possible_start = doc_len - block_size - 1\n"
            "                if last_possible_start >= 0:\n"
            "                    # Verifica si la ultima sample cubre el final del documento\n"
            "                    if not self.samples or (self.samples[-1][1] + block_size < doc_len):\n"
            "                        # Necesitamos otra ventana para cubrir el final\n"
            "                        self.samples.append((doc_idx, last_possible_start))\n"
            "                        for pos in range(last_possible_start, min(last_possible_start + block_size + 1, doc_len)):\n"
            "                            unique_tokens_covered.add((doc_idx, pos))"
        ),
        line_hint=136
    ),

    # PATCH 3: Mismo fix para memory-efficient mode
    Patch(
        file="dataset.py",
        description="[CRITICO] Corregir logica de ultima ventana (memory-efficient)",
        search=(
            "                last_start = doc_len - block_size - 1\n"
            "                if last_start >= 0 and (not self.samples or self.samples[-1][1] < last_start):\n"
            "                    self.samples.append((doc_idx, last_start))\n"
            "                    for pos in range(last_start, doc_len):\n"
            "                        unique_tokens_covered.add((doc_idx, pos))"
        ),
        replace=(
            "                last_possible_start = doc_len - block_size - 1\n"
            "                if last_possible_start >= 0:\n"
            "                    if not self.samples or (self.samples[-1][1] + block_size < doc_len):\n"
            "                        self.samples.append((doc_idx, last_possible_start))\n"
            "                        for pos in range(last_possible_start, min(last_possible_start + block_size + 1, doc_len)):\n"
            "                            unique_tokens_covered.add((doc_idx, pos))"
        ),
        line_hint=156
    ),

    # PATCH 4: Limpieza de GPU memory entre epochs (training.py)
    Patch(
        file="training.py",
        description="[MEDIO] Añadir limpieza de GPU memory entre epochs",
        search=(
            "        # End of epoch evaluation\n"
            "        epoch_train_loss = epoch_loss / epoch_tokens\n"
            "        val_loss = evaluate(model, val_loader, device)\n\n"
            "        print(f\"\\n{'='*70}\")\n"
            "        print(f\"Epoch {epoch + 1} Summary:\")"
        ),
        replace=(
            "        # End of epoch evaluation\n"
            "        epoch_train_loss = epoch_loss / epoch_tokens\n"
            "        val_loss = evaluate(model, val_loader, device)\n\n"
            "        # Limpia cache de GPU para liberar memoria fragmentada\n"
            "        try:\n"
            "            import torch\n"
            "            if torch.cuda.is_available():\n"
            "                torch.cuda.empty_cache()\n"
            "        except Exception:\n"
            "            pass\n\n"
            "        print(f\"\\n{'='*70}\")\n"
            "        print(f\"Epoch {epoch + 1} Summary:\")"
        ),
        line_hint=290
    ),

    # PATCH 5: Validacion de vocab_size en main.py
    Patch(
        file="main.py",
        description="[BAJO] Añadir validación de vocab_size extremos",
        search=(
            "    # Carga tokenizer\n"
            "    console.print(f\"\\nðŸ\"š Cargando tokenizer: {args.tokenizer}\")\n"
            "    tok = validate_tokenizer(tokenizer_path)\n"
            "    tw = HFTokenizerWrapper(tok)\n"
            "    console.print(f\"   Vocab size: {tw.vocab_size:,}\")"
        ),
        replace=(
            "    # Carga tokenizer\n"
            "    console.print(f\"\\n Cargando tokenizer: {args.tokenizer}\")\n"
            "    tok = validate_tokenizer(tokenizer_path)\n"
            "    tw = HFTokenizerWrapper(tok)\n"
            "    console.print(f\"   Vocab size: {tw.vocab_size:,}\")\n\n"
            "    # Validacion de vocab_size\n"
            "    if tw.vocab_size < 1000:\n"
            "        console.print(\"[yellow]Advertencia: Vocabulario muy pequeno (<1000). Esto puede causar underfitting.[/]\")\n"
            "    elif tw.vocab_size > 100000:\n"
            "        console.print(\"[yellow]Advertencia: Vocabulario muy grande (>100K). Esto aumentara uso de memoria.[/]\")"
        ),
        line_hint=491
    ),

    # PATCH 6: Eliminar variable sin uso en main.py
    Patch(
        file="main.py",
        description="[BAJO] Eliminar variable mock_ids sin uso",
        search=(
            "    # Lee corpus\n"
            "    text = validate_corpus(corpus_path)\n\n"
            "    # Mock tokenizer simple (split por espacios)\n"
            "    mock_ids = list(range(len(text.split())))\n\n"
            "    console.print(f\"   Chars: {len(text):,}\")\n"
            "    console.print(f\"   Palabras (aprox): {len(text.split()):,}\")"
        ),
        replace=(
            "    # Lee corpus\n"
            "    text = validate_corpus(corpus_path)\n\n"
            "    console.print(f\"   Chars: {len(text):,}\")\n"
            "    console.print(f\"   Palabras (aprox): {len(text.split()):,}\")"
        ),
        line_hint=471
    ),

    # PATCH 7: Añadir warning sobre memory_efficient en dataset.py (documentacion)
    Patch(
        file="dataset.py",
        description="[BAJO] Documentar impacto de memory_efficient en performance",
        search=(
            "        memory_efficient: Si True, no guarda documentos en memoria (mas lento pero menos RAM)\n"
        ),
        replace=(
            "        memory_efficient: Si True, no guarda documentos en memoria (mas lento pero menos RAM).\n"
            "                          ADVERTENCIA: Puede reducir velocidad de training en 20-30% debido a slicing.\n"
            "                          Usalo solo si el corpus no cabe en RAM (>4GB de tokens).\n"
        ),
        line_hint=53
    ),
]


# ============================================================================
# FUNCIONES DE APLICACION
# ============================================================================

def read_file(filepath: Path) -> str:
    """Lee archivo con encoding correcto."""
    try:
        return filepath.read_text(encoding='utf-8')
    except UnicodeDecodeError:
        return filepath.read_text(encoding='latin-1')


def write_file(filepath: Path, content: str):
    """Escribe archivo con encoding UTF-8."""
    filepath.write_text(content, encoding='utf-8')


def create_backup(filepath: Path):
    """Crea backup del archivo."""
    backup_path = filepath.with_suffix(filepath.suffix + '.backup')
    shutil.copy2(filepath, backup_path)
    return backup_path


def check_patches(patches: List[Patch]) -> Tuple[List[Patch], List[Patch]]:
    """
    Verifica que parches se pueden aplicar.
    Returns: (aplicables, no_aplicables)
    """
    aplicables = []
    no_aplicables = []

    for patch in patches:
        filepath = Path(patch.file)
        if not filepath.exists():
            print(f"NOT FOUND: {patch.file}")
            no_aplicables.append(patch)
            continue

        content = read_file(filepath)
        # Usar búsqueda literal o regex flexible
        new_content, ok = patch.apply(content)
        if ok:
            aplicables.append(patch)
        else:
            no_aplicables.append(patch)

    return aplicables, no_aplicables


def apply_patches(patches: List[Patch], create_backups: bool = True):
    """Aplica los parches a los archivos."""
    results = {
        'success': [],
        'failed': [],
        'skipped': []
    }

    files_processed = set()

    for patch in patches:
        filepath = Path(patch.file)

        print("\n" + "="*70)
        print(f"Applying: {patch.description}")
        print(f"File: {patch.file}")
        if patch.line_hint:
            print(f"Approx line: {patch.line_hint}")

        if not filepath.exists():
            print(f"SKIP: File not found: {patch.file}")
            results['skipped'].append(patch)
            continue

        # Backup solo una vez por archivo
        if create_backups and filepath not in files_processed:
            backup_path = create_backup(filepath)
            print(f"Backup created: {backup_path}")
            files_processed.add(filepath)

        # Lee contenido
        content = read_file(filepath)

        # Aplica parche
        new_content, success = patch.apply(content)

        if success:
            write_file(filepath, new_content)
            print("OK: Patch applied")
            results['success'].append(patch)
        else:
            print("FAIL: Pattern not found - patch not applied")
            results['failed'].append(patch)

    return results


def print_summary(results: dict):
    """Imprime resumen de resultados."""
    print("\n" + "="*70)
    print("PATCH SUMMARY")
    print("="*70)
    print(f"Success:  {len(results['success'])}")
    print(f"Failed:   {len(results['failed'])}")
    print(f"Skipped:  {len(results['skipped'])}")
    print("="*70 + "\n")

    if results['failed']:
        print("Failed patches:")
        for patch in results['failed']:
            print(f"  - {patch.description} ({patch.file})")
        print()

    if results['skipped']:
        print("Skipped patches:")
        for patch in results['skipped']:
            print(f"  - {patch.description} ({patch.file})")
        print()


def restore_backups():
    """Restaura archivos desde backups."""
    backups = list(Path('.').glob('*.backup'))

    if not backups:
        print("No backups found")
        return

    print(f"\nFound {len(backups)} backups:")
    for backup in backups:
        original = backup.with_suffix('')
        print(f"  {backup} -> {original}")

    resp = input("\nRestore all backups? (y/n): ")
    if resp.lower() != 'y':
        print("Cancelled")
        return

    for backup in backups:
        original = backup.with_suffix('')
        shutil.copy2(backup, original)
        print(f"Restored: {original}")

    print("\nBackups restored")


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Aplica parches criticos a Mini-LLM v2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python patches_critical.py --check           # Solo verifica
  python patches_critical.py --apply           # Aplica con backup
  python patches_critical.py --apply --no-backup  # Aplica sin backup (peligroso)
  python patches_critical.py --restore         # Restaura desde backups
        """
    )

    parser.add_argument('--check', action='store_true', help='Solo verifica que parches son aplicables')
    parser.add_argument('--apply', action='store_true', help='Aplica los parches')
    parser.add_argument('--restore', action='store_true', help='Restaura archivos desde backups')
    parser.add_argument('--no-backup', action='store_true', help='No crear backups (usar con precaucion)')

    args = parser.parse_args()

    if args.restore:
        restore_backups()
        return

    # Si el usuario pide check o no especifica apply, hacemos verificacion
    if args.check or not args.apply:
        print("Checking patches available...\n")
        aplicables, no_aplicables = check_patches(PATCHES)

        print("="*70)
        print(f"Patches applicable: {len(aplicables)}/{len(PATCHES)}")
        print("="*70 + "\n")

        for patch in aplicables:
            sev = "CRITICO" if "CRITICO" in patch.description else ("MEDIO" if "MEDIO" in patch.description else "BAJO")
            print(f"[{sev}] {patch.description}")
            print(f"   File: {patch.file}")
            if patch.line_hint:
                print(f"   Approx line: {patch.line_hint}")
            print()

        if no_aplicables:
            print(f"\nPatches NOT applicable ({len(no_aplicables)}):")
            for patch in no_aplicables:
                print(f" - {patch.description} ({patch.file}) - pattern not found or file missing")
            print()

        if not args.apply:
            print("Use --apply to apply patches")
        return

    # Aplicar
    if args.apply:
        print("Applying patches...\n")
        if not args.no_backup:
            print("Backups (.backup) will be created for modified files")
        else:
            print("WARNING: No backups will be created")
            resp = input("Continue? (y/n): ")
            if resp.lower() != 'y':
                print("Cancelled")
                return

        results = apply_patches(PATCHES, create_backups=not args.no_backup)
        print_summary(results)

        if results['success']:
            print("Patches applied.")
            if not args.no_backup:
                print("Backups saved with .backup suffix.")
        if results['failed']:
            print("Some patches failed. Please inspect the files manually.")


if __name__ == '__main__':
    main()
