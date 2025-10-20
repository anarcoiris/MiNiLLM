# Mini-LLM v2: Transformer Mejorado para Dominios Especializados

Sistema completo para entrenar modelos de lenguaje ligeros pero competentes en dominios específicos como ciencias, historia e historiografía.

## 🎯 Características Principales

### Arquitectura Mejorada
- ✅ **RoPE (Rotary Position Embeddings)**: Mejor extrapolación a secuencias largas
- ✅ **Pre-Layer Normalization**: Mayor estabilidad de entrenamiento
- ✅ **Weight Tying**: Reducción de parámetros sin pérdida de calidad
- ✅ **Gradient Checkpointing**: Entrena modelos más grandes con menos memoria
- ✅ **Flash Attention** (opcional): 2-3x más rápido

### Dataset Inteligente
- ✅ **Document-Aware Sampling**: Respeta límites de documentos
- ✅ **Overlapping Windows**: Mejor cobertura del corpus
- ✅ **Análisis Automático**: Sugiere hiperparámetros óptimos

### Entrenamiento Robusto
- ✅ **Learning Rate Warmup + Cosine Decay**: Convergencia más rápida
- ✅ **Gradient Clipping & Accumulation**: Estabilidad en GPUs pequeñas
- ✅ **Mixed Precision (AMP)**: Entrena 2x más rápido
- ✅ **Early Stopping**: Evita overfitting automáticamente
- ✅ **Perplexity Tracking**: Métrica interpretable

### Generación Avanzada
- ✅ **Top-k, Top-p, Temperature**: Control fino de creatividad
- ✅ **Repetition Penalty**: Reduce loops
- ✅ **Streaming**: Tokens en tiempo real
- ✅ **Batch Generation**: Múltiples textos eficientemente

## 📦 Instalación

```bash
# Clonar repositorio
git clone <tu-repo>
cd mini-llm-v2

# Instalar dependencias
pip install torch tokenizers rich tqdm ftfy

# Opcional: Flash Attention (requiere CUDA)
pip install flash-attn --no-build-isolation
```

## 🚀 Guía de Inicio Rápido

### 1. Preparar Corpus

```bash
# Limpia y normaliza tu corpus
python main.py prepare-corpus \
    --input datos_raw.txt \
    --output corpus_limpio.txt \
    --preserve-case  # Mantener mayúsculas (recomendado para ciencias)
```

### 2. Analizar Corpus

```bash
# Analiza estructura y obtén recomendaciones
python main.py analyze-corpus \
    --corpus corpus_limpio.txt \
    --separator "<|doc|>"  # Si tienes separadores de documento
```

Salida ejemplo:
```
ANÁLISIS DEL CORPUS
==================================================================
Tokens totales: 1,234,567
Documentos encontrados: 150
Longitud promedio: 8,230 tokens
Longitud mínima: 120 tokens
Longitud máxima: 45,000 tokens

💡 RECOMENDACIONES:
   Block size sugerido: 512
   Stride sugerido: 256 (50% overlap)
```

### 3. Entrenar Tokenizer

```bash
python main.py init-tokenizer \
    --files corpus_limpio.txt \
    --vocab-size 32000 \
    --out tokenizer.json
```

### 4. Entrenar Modelo

#### Opción A: Configuración Predefinida (Recomendado)

```bash
# Modelo pequeño (~10M parámetros, ideal para 50-100MB de datos)
python main.py train \
    --tokenizer tokenizer.json \
    --corpus corpus_limpio.txt \
    --outdir runs/small_model \
    --config small

# Modelo mediano (~30M parámetros, ideal para 200-500MB)
python main.py train \
    --tokenizer tokenizer.json \
    --corpus corpus_limpio.txt \
    --outdir runs/medium_model \
    --config medium

# Modelo grande (~70M parámetros, ideal para >500MB)
python main.py train \
    --tokenizer tokenizer.json \
    --corpus corpus_limpio.txt \
    --outdir runs/large_model \
    --config large
```

#### Opción B: Configuración Custom

```bash
python main.py train \
    --tokenizer tokenizer.json \
    --corpus corpus_limpio.txt \
    --outdir runs/custom_model \
    --config custom \
    --block-size 512 \
    --n-embd 384 \
    --n-layer 10 \
    --n-head 6 \
    --batch-size 16 \
    --epochs 30 \
    --lr 5e-4 \
    --warmup-steps 1000
```

### 5. Generar Texto

```bash
# Generación simple
python main.py generate \
    --tokenizer tokenizer.json \
    --ckpt runs/small_model/ckpt_best.pt \
    --prompt "La fotosíntesis es un proceso" \
    --max-tokens 200 \
    --temperature 0.8

# Generación con streaming
python main.py generate \
    --tokenizer tokenizer.json \
    --ckpt runs/small_model/ckpt_best.pt \
    --prompt "Durante el Imperio Romano" \
    --max-tokens 150 \
    --stream

# Múltiples prompts desde archivo
python main.py generate \
    --tokenizer tokenizer.json \
    --ckpt runs/small_model/ckpt_best.pt \
    --prompt-file prompts.txt \
    --output generaciones.txt \
    --batch-generate
```

### 6. Evaluar Modelo

```bash
python main.py evaluate \
    --tokenizer tokenizer.json \
    --ckpt runs/small_model/ckpt_best.pt \
    --test-file test_corpus.txt
```

## 📊 Configuraciones Recomendadas

### Para Dataset Pequeño (50-100MB)
```bash
--config small
--block-size 256
--batch-size 32
--epochs 30
--lr 5e-4
```
**Resultado esperado**: Perplexity < 50, ~10M parámetros

### Para Dataset Mediano (200-500MB)
```bash
--config medium
--block-size 512
--batch-size 16
--epochs 20
--lr 3e-4
```
**Resultado esperado**: Perplexity < 30, ~30M parámetros

### Para Dataset Grande (>500MB)
```bash
--config large
--block-size 1024
--batch-size 8
--epochs 15
--lr 2e-4
--gradient-checkpointing
```
**Resultado esperado**: Perplexity < 20, ~70M parámetros

## 🎛️ Parámetros Importantes

### Arquitectura
- `--block-size`: Longitud de contexto (256/512/1024)
- `--n-embd`: Dimensión de embeddings (256/384/512)
- `--n-layer`: Profundidad del modelo (6/8/12)
- `--n-head`: Número de attention heads (debe dividir n-embd)

### Entrenamiento
- `--lr`: Learning rate (3e-4 a 5e-4 típicamente)
- `--warmup-steps`: Pasos de warmup (500-2000)
- `--grad-clip`: Gradient clipping (1.0 recomendado)
- `--accumulation-steps`: Para simular batch size mayor
- `--patience`: Early stopping (3-5 épocas)

### Generación
- `--temperature`: Creatividad (0.7=conservador, 1.0=balanceado, 1.5=creativo)
- `--top-k`: Limita a k tokens más probables (50 típico)
- `--top-p`: Nucleus sampling (0.9 recomendado)
- `--repetition-penalty`: Reduce repetición (1.0-1.2)

## 📁 Estructura de Archivos

```
mini-llm-v2/
├── model.py           # Arquitectura del transformer
├── dataset.py         # Dataset con document-awareness
├── training.py        # Training loop robusto
├── generation.py      # Generación de texto
├── main.py           # CLI principal
├── standarize.py        # Limpieza de texto (legacy)
└── runs/                # Checkpoints y logs
    └── exp1/
        ├── ckpt_best.pt     # Mejor modelo
        ├── ckpt_last.pt     # Último checkpoint
        ├── history.json     # Métricas de entrenamiento
        └── config.json      # Configuración usada
```

## 🔧 Solución de Problemas

### "CUDA out of memory"
```bash
# Solución 1: Reduce batch size
--batch-size 8

# Solución 2: Usa gradient accumulation
--batch-size 8 --accumulation-steps 4  # Simula batch=32

# Solución 3: Activa gradient checkpointing
--gradient-checkpointing

# Solución 4: Reduce tamaño del modelo
--n-embd 256 --n-layer 6
```

### "Perplexity muy alto (>100)"
- ✅ Entrena más épocas
- ✅ Aumenta `--warmup-steps`
- ✅ Reduce learning rate
- ✅ Aumenta tamaño del modelo
- ✅ Verifica calidad del corpus

### "Modelo repite texto"
```bash
# Durante generación, usa:
--repetition-penalty 1.2
--top-p 0.9
--temperature 0.8
```

### "Entrenamiento muy lento"
- ✅ Activa mixed precision: (por defecto activo)
- ✅ Reduce `--eval-every` (menos evaluaciones)
- ✅ Usa GPU con `--device cuda`
- ✅ Aumenta `--num-workers` (DataLoader)

## 📈 Métricas de Éxito

### Mínimo Viable
- ✅ Perplexity < 50 en validación
- ✅ Genera 3+ oraciones coherentes
- ✅ No loops infinitos

### Bueno
- ✅ Perplexity < 30
- ✅ Mantiene contexto por 1-2 párrafos
- ✅ Gramática mayormente correcta

###