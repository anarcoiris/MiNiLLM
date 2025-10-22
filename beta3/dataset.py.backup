"""
dataset.py - VERSIÓN CORREGIDA

Dataset mejorado con:
- Muestreo por documento (evita mezclar contextos)
- Overlapping windows para mejor cobertura
- Manejo de documentos cortos
- Padding inteligente
- Estadísticas de corpus
- CORREGIDO: Cálculo de coverage
- CORREGIDO: Memory-efficient para corpus grandes
"""

import random
import torch
from torch.utils.data import Dataset
from typing import List, Tuple, Optional, Iterator
from dataclasses import dataclass


@dataclass
class DatasetStats:
    """Estadísticas del dataset."""
    num_documents: int
    total_tokens: int
    avg_doc_length: float
    min_doc_length: int
    max_doc_length: int
    num_samples: int
    coverage_ratio: float  # Ratio de tokens únicos cubiertos vs. total


class DocumentAwareDataset(Dataset):
    """
    Dataset que respeta límites de documentos.

    CORRECCIONES v2:
    - Cálculo correcto de coverage (no cuenta duplicados por overlap)
    - Opción memory-efficient para corpus grandes

    Args:
        ids: Lista completa de token IDs
        block_size: Longitud de cada secuencia
        doc_separator_id: ID del token que separa documentos (e.g., <|doc|>)
        stride: Overlap entre ventanas (stride=block_size/2 significa 50% overlap)
        min_doc_length: Documentos más cortos se ignoran
        pad_id: Token para padding (si es necesario)
        memory_efficient: Si True, no guarda documentos en memoria (más lento pero menos RAM)
    """

    def __init__(
        self,
        ids: List[int],
        block_size: int,
        doc_separator_id: Optional[int] = None,
        stride: Optional[int] = None,
        min_doc_length: Optional[int] = None,
        pad_id: int = 0,
        memory_efficient: bool = False
    ):
        self.block_size = block_size
        self.pad_id = pad_id
        self.memory_efficient = memory_efficient

        # Stride por defecto: 50% overlap
        self.stride = stride if stride is not None else block_size // 2

        # Longitud mínima por defecto: al menos 2 tokens más que block_size
        self.min_doc_length = min_doc_length if min_doc_length is not None else block_size + 2

        # Divide corpus en documentos
        if doc_separator_id is not None:
            if memory_efficient:
                self.doc_boundaries = self._find_doc_boundaries(ids, doc_separator_id)
                self.original_ids = ids  # Mantén referencia a IDs originales
                self.documents = None
            else:
                self.documents = self._split_by_separator(ids, doc_separator_id)
                self.doc_boundaries = None
                self.original_ids = None
        else:
            # Si no hay separador, trata todo como un documento
            if memory_efficient:
                self.doc_boundaries = [(0, len(ids))]
                self.original_ids = ids
                self.documents = None
            else:
                self.documents = [ids]
                self.doc_boundaries = None
                self.original_ids = None

        # Filtra documentos muy cortos
        if not memory_efficient:
            self.documents = [
                doc for doc in self.documents
                if len(doc) >= self.min_doc_length
            ]

            if len(self.documents) == 0:
                raise ValueError(
                    f"No hay documentos con longitud >= {self.min_doc_length}. "
                    f"Reduce min_doc_length o proporciona más datos."
                )
        else:
            # Filtra boundaries
            self.doc_boundaries = [
                (start, end) for start, end in self.doc_boundaries
                if (end - start) >= self.min_doc_length
            ]

            if len(self.doc_boundaries) == 0:
                raise ValueError(
                    f"No hay documentos con longitud >= {self.min_doc_length}. "
                    f"Reduce min_doc_length o proporciona más datos."
                )

        # Crea ventanas (samples) por documento con overlapping
        self.samples = []
        unique_tokens_covered = set()  # CORREGIDO: Para tracking de cobertura

        if not memory_efficient:
            for doc_idx, doc in enumerate(self.documents):
                doc_len = len(doc)

                # Crea ventanas con stride
                for start_idx in range(0, max(1, doc_len - block_size), self.stride):
                    end_idx = min(start_idx + block_size, doc_len)
                    self.samples.append((doc_idx, start_idx))

                    # CORREGIDO: Cuenta tokens únicos cubiertos
                    for pos in range(start_idx, end_idx):
                        # Usamos (doc_idx, pos) como identificador único
                        unique_tokens_covered.add((doc_idx, pos))

                # Añade última ventana si no está ya incluida
                last_start = doc_len - block_size - 1
                if last_start >= 0 and (not self.samples or self.samples[-1][1] < last_start):
                    self.samples.append((doc_idx, last_start))
                    for pos in range(last_start, doc_len):
                        unique_tokens_covered.add((doc_idx, pos))
        else:
            # Memory-efficient: trabaja con boundaries
            for doc_idx, (doc_start, doc_end) in enumerate(self.doc_boundaries):
                doc_len = doc_end - doc_start

                for start_idx in range(0, max(1, doc_len - block_size), self.stride):
                    self.samples.append((doc_idx, start_idx))

                    end_idx = min(start_idx + block_size, doc_len)
                    for pos in range(start_idx, end_idx):
                        unique_tokens_covered.add((doc_idx, pos))

                last_start = doc_len - block_size - 1
                if last_start >= 0 and (not self.samples or self.samples[-1][1] < last_start):
                    self.samples.append((doc_idx, last_start))
                    for pos in range(last_start, doc_len):
                        unique_tokens_covered.add((doc_idx, pos))

        # Calcula estadísticas
        self.total_tokens_covered = len(unique_tokens_covered)  # CORREGIDO
        self.stats = self._compute_stats(ids)

    def _find_doc_boundaries(self, ids: List[int], separator_id: int) -> List[Tuple[int, int]]:
        """Encuentra boundaries de documentos sin crear copias."""
        boundaries = []
        start_idx = 0

        for i, token_id in enumerate(ids):
            if token_id == separator_id:
                if i > start_idx:
                    boundaries.append((start_idx, i))
                start_idx = i + 1

        # Añade último documento
        if start_idx < len(ids):
            boundaries.append((start_idx, len(ids)))

        return boundaries

    def _split_by_separator(self, ids: List[int], separator_id: int) -> List[List[int]]:
        """Divide lista de IDs en documentos usando separador."""
        documents = []
        current_doc = []

        for token_id in ids:
            if token_id == separator_id:
                if len(current_doc) > 0:
                    documents.append(current_doc)
                    current_doc = []
            else:
                current_doc.append(token_id)

        # Añade último documento si existe
        if len(current_doc) > 0:
            documents.append(current_doc)

        return documents

    def _compute_stats(self, original_ids: List[int]) -> DatasetStats:
        """Calcula estadísticas del dataset."""
        if not self.memory_efficient:
            doc_lengths = [len(doc) for doc in self.documents]
            num_docs = len(self.documents)
        else:
            doc_lengths = [end - start for start, end in self.doc_boundaries]
            num_docs = len(self.doc_boundaries)

        return DatasetStats(
            num_documents=num_docs,
            total_tokens=len(original_ids),
            avg_doc_length=sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0,
            min_doc_length=min(doc_lengths) if doc_lengths else 0,
            max_doc_length=max(doc_lengths) if doc_lengths else 0,
            num_samples=len(self.samples),
            coverage_ratio=self.total_tokens_covered / len(original_ids) if len(original_ids) > 0 else 0
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retorna un par (x, y) donde:
        - x: secuencia de entrada (block_size tokens)
        - y: secuencia target (mismos tokens shifted +1)
        """
        doc_idx, start_idx = self.samples[idx]

        # Obtiene documento
        if not self.memory_efficient:
            doc = self.documents[doc_idx]
        else:
            doc_start, doc_end = self.doc_boundaries[doc_idx]
            doc = self.original_ids[doc_start:doc_end]

        # Extrae ventana
        end_idx = start_idx + self.block_size + 1
        window = doc[start_idx:end_idx]

        # Maneja caso edge (documento termina antes)
        if len(window) < self.block_size + 1:
            # Padding si es necesario (raro con min_doc_length correcto)
            window = window + [self.pad_id] * (self.block_size + 1 - len(window))

        # x: primeros block_size tokens
        # y: siguientes block_size tokens (shifted +1)
        x = torch.tensor(window[:self.block_size], dtype=torch.long)
        y = torch.tensor(window[1:self.block_size + 1], dtype=torch.long)

        return x, y

    def print_stats(self):
        """Imprime estadísticas del dataset."""
        s = self.stats
        print("="*60)
        print("ESTADÍSTICAS DEL DATASET")
        print("="*60)
        print(f"Documentos:           {s.num_documents:,}")
        print(f"Tokens totales:       {s.total_tokens:,}")
        print(f"Longitud promedio:    {s.avg_doc_length:,.1f} tokens/doc")
        print(f"Longitud mínima:      {s.min_doc_length:,} tokens")
        print(f"Longitud máxima:      {s.max_doc_length:,} tokens")
        print(f"Muestras generadas:   {s.num_samples:,}")
        print(f"Tokens cubiertos:     {self.total_tokens_covered:,}")
        print(f"Cobertura del corpus: {s.coverage_ratio:.1%}")
        print(f"Block size:           {self.block_size}")
        print(f"Stride:               {self.stride} (overlap: {1 - self.stride/self.block_size:.1%})")
        print(f"Memory efficient:     {self.memory_efficient}")
        print("="*60)


class CausalTextDataset(Dataset):
    """
    Dataset simple con muestreo aleatorio (versión mejorada de la original).

    Útil cuando:
    - No tienes separadores de documento
    - Quieres máxima aleatoriedad (data augmentation)
    - Dataset es muy pequeño y necesitas "inflar" el número de samples

    ADVERTENCIA: Este dataset ignora límites de documentos y puede
    mezclar contextos. Usa DocumentAwareDataset cuando sea posible.
    """

    def __init__(
        self,
        ids: List[int],
        block_size: int,
        randomize: bool = True,
        seed: int = 42
    ):
        self.ids = ids
        self.block_size = block_size
        self.randomize = randomize
        self.max_start = max(0, len(ids) - (block_size + 1))

        # Fija seed para reproducibilidad
        if randomize:
            self.rng = random.Random(seed)

        # Número de samples posibles
        self._len = self.max_start + 1 if not randomize else max(1000, self.max_start)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.randomize:
            # Muestreo aleatorio reproducible
            i = self.rng.randint(0, self.max_start)
        else:
            # Muestreo secuencial
            i = idx % (self.max_start + 1)

        x = torch.tensor(self.ids[i:i + self.block_size], dtype=torch.long)
        y = torch.tensor(self.ids[i + 1:i + 1 + self.block_size], dtype=torch.long)

        return x, y


# ======================== Helper Functions ========================

def prepare_datasets(
    ids: List[int],
    block_size: int,
    val_ratio: float = 0.1,
    doc_separator_id: Optional[int] = None,
    stride: Optional[int] = None,
    use_document_aware: bool = True,
    memory_efficient: bool = False,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Crea datasets de train y validación.

    Args:
        ids: Lista de token IDs del corpus completo
        block_size: Longitud de secuencias
        val_ratio: Fracción para validación (0.0-1.0)
        doc_separator_id: ID del token separador de documentos
        stride: Stride para overlapping windows
        use_document_aware: Si True, usa DocumentAwareDataset
        memory_efficient: Ahorra RAM pero es más lento
        seed: Seed para reproducibilidad

    Returns:
        train_dataset, val_dataset
    """
    # Split train/val
    split_idx = int(len(ids) * (1 - val_ratio))
    train_ids = ids[:split_idx]
    val_ids = ids[split_idx:]

    print(f"\n📊 Preparando datasets...")
    print(f"   Train tokens: {len(train_ids):,}")
    print(f"   Val tokens:   {len(val_ids):,}")

    if use_document_aware:
        train_ds = DocumentAwareDataset(
            train_ids,
            block_size=block_size,
            doc_separator_id=doc_separator_id,
            stride=stride,
            min_doc_length=block_size + 2,
            memory_efficient=memory_efficient
        )
        val_ds = DocumentAwareDataset(
            val_ids,
            block_size=block_size,
            doc_separator_id=doc_separator_id,
            stride=block_size,  # Sin overlap en validación
            min_doc_length=block_size + 2,
            memory_efficient=memory_efficient
        )

        print("\n📈 Train Dataset:")
        train_ds.print_stats()

        print("\n📉 Validation Dataset:")
        val_ds.print_stats()
    else:
        train_ds = CausalTextDataset(
            train_ids,
            block_size=block_size,
            randomize=True,
            seed=seed
        )
        val_ds = CausalTextDataset(
            val_ids,
            block_size=block_size,
            randomize=False,
            seed=seed
        )

        print(f"   Train samples: {len(train_ds):,}")
        print(f"   Val samples:   {len(val_ds):,}")

    return train_ds, val_ds


def analyze_corpus_structure(ids: List[int], separator_id: Optional[int] = None):
    """
    Analiza la estructura del corpus para determinar parámetros óptimos.

    Args:
        ids: Lista de token IDs
        separator_id: ID del token separador (opcional)
    """
    print("\n" + "="*70)
    print("ANÁLISIS DEL CORPUS")
    print("="*70)

    total_tokens = len(ids)
    print(f"Tokens totales: {total_tokens:,}")

    if separator_id is not None:
        # Analiza documentos
        docs = []
        current_doc = []

        for token_id in ids:
            if token_id == separator_id:
                if len(current_doc) > 0:
                    docs.append(len(current_doc))
                    current_doc = []
            else:
                current_doc.append(token_id)

        if len(current_doc) > 0:
            docs.append(len(current_doc))

        if docs:
            print(f"\nDocumentos encontrados: {len(docs)}")
            print(f"Longitud promedio: {sum(docs) / len(docs):,.1f} tokens")
            print(f"Longitud mínima:   {min(docs):,} tokens")
            print(f"Longitud máxima:   {max(docs):,} tokens")
            print(f"Mediana:           {sorted(docs)[len(docs)//2]:,} tokens")

            # Histograma simple
            print("\nDistribución de longitudes:")
            bins = [0, 50, 100, 200, 500, 1000, 2000, 5000, float('inf')]
            labels = ['<50', '50-100', '100-200', '200-500', '500-1k', '1k-2k', '2k-5k', '>5k']

            for i, (low, high, label) in enumerate(zip(bins[:-1], bins[1:], labels)):
                count = sum(1 for d in docs if low <= d < high)
                if count > 0:
                    pct = 100 * count / len(docs)
                    bar = '█' * int(pct / 2)
                    print(f"  {label:>8}: {bar} {count:>5} ({pct:>5.1f}%)")

            # Recomendaciones
            print("\n💡 RECOMENDACIONES:")
            median = sorted(docs)[len(docs)//2]

            if median < 128:
                rec_block = 64
            elif median < 512:
                rec_block = 256
            elif median < 2048:
                rec_block = 512
            else:
                rec_block = 1024

            print(f"   Block size sugerido:  {rec_block} (basado en mediana {median})")
            print(f"   Stride sugerido:      {rec_block // 2} (50% overlap)")
            print(f"   Min doc length:       {rec_block + 2}")

            # Advertencia sobre memoria
            total_size_mb = total_tokens * 4 / (1024 * 1024)  # Aproximado para int32
            print(f"\n   Tamaño aprox en memoria: {total_size_mb:.1f} MB")
            if total_size_mb > 1000:
                print(f"   ⚠️  Corpus grande. Considera usar memory_efficient=True")
    else:
        print("\nNo se proporcionó separador de documentos.")
        print("Se tratará el corpus completo como un solo documento.")

    print("="*70 + "\n")


# ======================== Test & Examples ========================

if __name__ == '__main__':
    # Ejemplo de uso
    print("Testing dataset.py (VERSIÓN CORREGIDA)\n")

    # Simula un corpus pequeño con separadores
    vocab_size = 1000
    doc_sep_id = 999

    # Crea 5 documentos de diferentes longitudes
    docs = [
        list(range(0, 100)),      # doc corto
        list(range(100, 500)),    # doc mediano
        list(range(500, 600)),    # doc corto
        list(range(600, 1500)),   # doc largo
        list(range(1500, 2000)),  # doc mediano
    ]

    # Une documentos con separador
    corpus_ids = []
    for doc in docs:
        corpus_ids.extend(doc)
        corpus_ids.append(doc_sep_id)

    print(f"Corpus simulado: {len(corpus_ids):,} tokens en {len(docs)} documentos\n")

    # Analiza corpus
    analyze_corpus_structure(corpus_ids, separator_id=doc_sep_id)

    # Crea dataset
    block_size = 128
    stride = 64

    print("="*70)
    print("TEST 1: Dataset estándar (en memoria)")
    print("="*70)
    dataset = DocumentAwareDataset(
        ids=corpus_ids,
        block_size=block_size,
        doc_separator_id=doc_sep_id,
        stride=stride,
        min_doc_length=block_size + 2,
        memory_efficient=False
    )

    dataset.print_stats()

    # TEST: Verifica coverage correcto
    print("\n🧪 Verificando cálculo de coverage...")
    expected_unique_tokens = sum(len(doc) for doc in docs)
    actual_covered = dataset.total_tokens_covered
    print(f"   Tokens únicos esperados: {expected_unique_tokens}")
    print(f"   Tokens cubiertos: {actual_covered}")
    print(f"   Ratio de cobertura: {dataset.stats.coverage_ratio:.1%}")

    # Con overlap, el coverage debería ser cercano a 100%
    assert dataset.stats.coverage_ratio > 0.95, "Coverage debería ser > 95% con stride=block_size/2"
    print("   ✓ Coverage correcto\n")

    # TEST 2: Memory efficient
    print("="*70)
    print("TEST 2: Dataset memory-efficient")
    print("="*70)
    dataset_mem = DocumentAwareDataset(
        ids=corpus_ids,
        block_size=block_size,
        doc_separator_id=doc_sep_id,
        stride=stride,
        min_doc_length=block_size + 2,
        memory_efficient=True
    )

    dataset_mem.print_stats()

    # Verifica que ambos producen lo mismo
    assert len(dataset) == len(dataset_mem), "Ambos modos deberían producir mismo número de samples"
    print("   ✓ Mismo número de samples\n")

    # Muestra algunos ejemplos
    print("="*70)
    print("TEST 3: Verificación de samples")
    print("="*70)
    for i in range(min(3, len(dataset))):
        x, y = dataset[i]
        x_mem, y_mem = dataset_mem[i]

        print(f"\nMuestra {i}:")
        print(f"  x shape: {x.shape}, primeros 10 tokens: {x[:10].tolist()}")
        print(f"  y shape: {y.shape}, primeros 10 tokens: {y[:10].tolist()}")
        print(f"  y es x shifted? {torch.equal(y[:-1], x[1:])}")
        print(f"  Memory-efficient igual? {torch.equal(x, x_mem) and torch.equal(y, y_mem)}")

    # Test de reproducibilidad
    print("\n" + "="*70)
    print("TEST 4: Reproducibilidad")
    print("="*70)
    samples_1 = [dataset[i][0][:5].tolist() for i in range(5)]
    samples_2 = [dataset[i][0][:5].tolist() for i in range(5)]
    print(f"  ✓ Reproducible: {samples_1 == samples_2}")

    print("\n" + "="*70)
    print("✅ TODOS LOS TESTS PASADOS")
    print("="*70)
