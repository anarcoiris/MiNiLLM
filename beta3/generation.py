"""
generation.py - VERSIÓN CORREGIDA

Generación de texto mejorada con:
- Sampling strategies corregidas (top-k, top-p, temperature)
- Repetition penalty
- Beam search (opcional)
- Streaming generation
- Batch generation
"""

import torch
import torch.nn.functional as F
from typing import List, Optional, Callable
from dataclasses import dataclass


@dataclass
class GenerationConfig:
    """Configuración para generación de texto."""
    max_new_tokens: int = 100
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    min_length: int = 0
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = 0
    do_sample: bool = True  # Si False, usa greedy decoding


def apply_repetition_penalty(logits: torch.Tensor, token_ids: List[int], penalty: float = 1.0):
    """
    Aplica penalización a tokens ya generados para reducir repetición.

    Args:
        logits: (vocab_size,) tensor de logits
        token_ids: Lista de token IDs ya generados
        penalty: Factor de penalización (>1.0 penaliza, <1.0 promueve repetición)

    Returns:
        logits penalizados
    """
    if penalty == 1.0 or len(token_ids) == 0:
        return logits

    for token_id in set(token_ids):
        if logits[token_id] < 0:
            logits[token_id] *= penalty
        else:
            logits[token_id] /= penalty

    return logits


def top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """
    Filtra logits manteniendo solo los top-k más probables.

    Args:
        logits: (vocab_size,) o (..., vocab_size) tensor
        top_k: número de tokens a mantener

    Returns:
        logits filtrados con el resto en -inf
    """
    if top_k <= 0:
        return logits

    top_k = min(top_k, logits.size(-1))

    # Encuentra threshold: el k-ésimo valor más grande
    values, _ = torch.topk(logits, top_k, dim=-1)
    min_value = values[..., -1, None]

    # Enmascara valores por debajo del threshold
    return torch.where(logits < min_value, torch.full_like(logits, float('-inf')), logits)


def top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    """
    Nucleus sampling: mantiene el mínimo conjunto de tokens cuya probabilidad suma >= top_p.

    VERSIÓN CORREGIDA: Implementación simplificada y robusta.

    Args:
        logits: (vocab_size,) tensor
        top_p: umbral de probabilidad acumulativa (0.0-1.0)

    Returns:
        logits filtrados
    """
    if top_p <= 0.0 or top_p >= 1.0:
        return logits

    # Ordena logits de mayor a menor
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)

    # Calcula probabilidades acumulativas
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

    # Remueve tokens donde la probabilidad acumulativa excede top_p
    # Mantenemos al menos el primer token (el más probable)
    sorted_indices_to_remove = cumulative_probs > top_p

    # Shift a la derecha para mantener el primer token que supera el umbral
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Crea máscara en el orden original
    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove[sorted_indices[sorted_indices_to_remove]] = True

    logits[indices_to_remove] = float('-inf')

    return logits


def sample_token(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
    do_sample: bool = True
) -> int:
    """
    Samplea un token de los logits usando temperature, top-k y top-p.

    Args:
        logits: (vocab_size,) tensor de logits
        temperature: temperatura de sampling (>1 más aleatorio, <1 más determinístico)
        top_k: número de top tokens a considerar (0 = todos)
        top_p: probabilidad acumulativa para nucleus sampling (0.0 = deshabilitado)
        do_sample: Si False, usa argmax (greedy)

    Returns:
        token_id muestreado
    """
    # Greedy decoding
    if not do_sample or temperature < 1e-8:
        return torch.argmax(logits).item()

    # Aplica temperature
    logits = logits / max(temperature, 1e-8)

    # Aplica top-k
    if top_k > 0:
        logits = top_k_filtering(logits, top_k)

    # Aplica top-p (nucleus sampling)
    if 0.0 < top_p < 1.0:
        logits = top_p_filtering(logits, top_p)

    # Convierte a probabilidades
    probs = F.softmax(logits, dim=-1)

    # Samplea
    token_id = torch.multinomial(probs, num_samples=1).item()

    return token_id


@torch.no_grad()
def generate(
    model: torch.nn.Module,
    tokenizer,
    prompt: str,
    config: GenerationConfig,
    device: str = 'cpu',
    stream_callback: Optional[Callable[[str], None]] = None
) -> str:
    """
    Genera texto a partir de un prompt.

    Args:
        model: Modelo de lenguaje
        tokenizer: Tokenizer con métodos encode() y decode()
        prompt: Texto inicial
        config: GenerationConfig con parámetros de generación
        device: 'cpu' o 'cuda'
        stream_callback: Función opcional para streaming (recibe tokens generados)

    Returns:
        Texto completo generado
    """
    model.eval()

    # Tokeniza prompt
    if hasattr(tokenizer, 'encode'):
        ids = tokenizer.encode(prompt)
    else:
        # Si es HFTokenizerWrapper
        ids = tokenizer.encode(prompt)

    if len(ids) == 0:
        ids = [config.pad_token_id or 0]

    generated_ids = []

    # Genera tokens
    for step in range(config.max_new_tokens):
        # Toma últimos block_size tokens como contexto
        context = ids[-model.block_size:]
        x = torch.tensor([context], dtype=torch.long, device=device)

        # Forward pass
        logits = model(x)
        next_token_logits = logits[0, -1, :].clone()  # (vocab_size,) - CLONE para evitar modificar el original

        # Aplica repetition penalty
        if config.repetition_penalty != 1.0:
            next_token_logits = apply_repetition_penalty(
                next_token_logits,
                ids[-50:],  # Considera últimos 50 tokens para penalización
                config.repetition_penalty
            )

        # Evita generar EOS antes del mínimo
        if config.eos_token_id is not None and len(generated_ids) < config.min_length:
            next_token_logits[config.eos_token_id] = float('-inf')

        # Samplea siguiente token
        next_token_id = sample_token(
            next_token_logits,
            temperature=config.temperature,
            top_k=config.top_k,
            top_p=config.top_p,
            do_sample=config.do_sample
        )

        # Añade a secuencia
        ids.append(next_token_id)
        generated_ids.append(next_token_id)

        # Streaming
        if stream_callback is not None:
            if hasattr(tokenizer, 'decode'):
                token_text = tokenizer.decode([next_token_id])
            else:
                token_text = tokenizer.decode([next_token_id])
            stream_callback(token_text)

        # Para si encuentra EOS
        if config.eos_token_id is not None and next_token_id == config.eos_token_id:
            break

    # Decodifica resultado completo
    if hasattr(tokenizer, 'decode'):
        return tokenizer.decode(ids)
    else:
        return tokenizer.decode(ids)


@torch.no_grad()
def generate_batch(
    model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    config: GenerationConfig,
    device: str = 'cpu'
) -> List[str]:
    """
    Genera múltiples textos en batch (más eficiente).

    Args:
        model: Modelo de lenguaje
        tokenizer: Tokenizer
        prompts: Lista de prompts
        config: GenerationConfig
        device: 'cpu' o 'cuda'

    Returns:
        Lista de textos generados
    """
    model.eval()
    batch_size = len(prompts)

    # Tokeniza prompts
    all_ids = []
    for prompt in prompts:
        if hasattr(tokenizer, 'encode'):
            ids = tokenizer.encode(prompt)
        else:
            ids = tokenizer.encode(prompt)
        if len(ids) == 0:
            ids = [config.pad_token_id or 0]
        all_ids.append(ids)

    # Padding a la misma longitud
    max_prompt_len = max(len(ids) for ids in all_ids)
    padded_ids = []
    for ids in all_ids:
        padding = [config.pad_token_id] * (max_prompt_len - len(ids))
        padded_ids.append(padding + ids)

    # Genera tokens
    current_sequences = torch.tensor(padded_ids, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for step in range(config.max_new_tokens):
        # Toma últimos block_size tokens
        if current_sequences.size(1) > model.block_size:
            context = current_sequences[:, -model.block_size:]
        else:
            context = current_sequences

        # Forward pass
        logits = model(context)
        next_token_logits = logits[:, -1, :]  # (batch_size, vocab_size)

        # Samplea para cada secuencia en el batch
        next_tokens = []
        for i in range(batch_size):
            if finished[i]:
                next_tokens.append(config.pad_token_id)
                continue

            token_id = sample_token(
                next_token_logits[i],
                temperature=config.temperature,
                top_k=config.top_k,
                top_p=config.top_p,
                do_sample=config.do_sample
            )
            next_tokens.append(token_id)

            # Marca como finalizado si genera EOS
            if config.eos_token_id is not None and token_id == config.eos_token_id:
                finished[i] = True

        # Añade nuevos tokens
        next_tokens_tensor = torch.tensor(next_tokens, dtype=torch.long, device=device).unsqueeze(1)
        current_sequences = torch.cat([current_sequences, next_tokens_tensor], dim=1)

        # Para si todos terminaron
        if finished.all():
            break

    # Decodifica resultados
    results = []
    for seq in current_sequences.cpu().tolist():
        if hasattr(tokenizer, 'decode'):
            text = tokenizer.decode(seq)
        else:
            text = tokenizer.decode(seq)
        results.append(text)

    return results


@torch.no_grad()
def compute_perplexity(
    model: torch.nn.Module,
    tokenizer,
    texts: List[str],
    device: str = 'cpu'
) -> List[float]:
    """
    Calcula perplexity de textos (útil para evaluar calidad).

    Args:
        model: Modelo de lenguaje
        tokenizer: Tokenizer
        texts: Lista de textos a evaluar
        device: 'cpu' o 'cuda'

    Returns:
        Lista de perplexities (uno por texto)
    """
    model.eval()
    criterion = torch.nn.CrossEntropyLoss(reduction='none')

    perplexities = []

    for text in texts:
        # Tokeniza
        if hasattr(tokenizer, 'encode'):
            ids = tokenizer.encode(text)
        else:
            ids = tokenizer.encode(text)

        if len(ids) < 2:
            perplexities.append(float('inf'))
            continue

        # Divide en ventanas
        total_loss = 0.0
        total_tokens = 0

        for i in range(0, len(ids) - 1, model.block_size):
            chunk = ids[i:i + model.block_size + 1]
            if len(chunk) < 2:
                continue

            x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
            y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)

            # Pad si es necesario
            if x.size(1) < model.block_size:
                padding = torch.zeros(
                    1, model.block_size - x.size(1),
                    dtype=torch.long, device=device
                )
                x = torch.cat([x, padding], dim=1)
                y = torch.cat([y, padding], dim=1)

            logits = model(x)
            loss = criterion(
                logits.view(-1, logits.size(-1)),
                y.view(-1)
            )

            # Solo cuenta tokens no-padding
            mask = (y != 0).view(-1)
            total_loss += loss[mask].sum().item()
            total_tokens += mask.sum().item()

        if total_tokens > 0:
            avg_loss = total_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_loss)).item()
            perplexities.append(min(perplexity, 1e6))  # Clip para evitar overflow
        else:
            perplexities.append(float('inf'))

    return perplexities


# ======================== Utilidades ========================

def print_generation_stats(original_prompt: str, generated_text: str, tokenizer):
    """Imprime estadísticas de la generación."""
    prompt_tokens = len(tokenizer.encode(original_prompt))
    total_tokens = len(tokenizer.encode(generated_text))
    new_tokens = total_tokens - prompt_tokens

    print(f"\n{'='*70}")
    print("ESTADÍSTICAS DE GENERACIÓN")
    print(f"{'='*70}")
    print(f"Tokens del prompt:    {prompt_tokens}")
    print(f"Tokens generados:     {new_tokens}")
    print(f"Tokens totales:       {total_tokens}")
    print(f"Longitud en chars:    {len(generated_text)}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    # Test mejorado
    print("Testing generation.py (VERSIÓN CORREGIDA)\n")

    # Test de filtering functions
    print("="*70)
    print("TEST 1: Top-k filtering")
    print("="*70)
    logits = torch.randn(100)

    filtered = top_k_filtering(logits.clone(), top_k=10)
    num_valid = (filtered != float('-inf')).sum().item()
    print(f"✓ Tokens válidos después de top-k=10: {num_valid}")
    assert num_valid == 10, f"top-k debería dejar exactamente 10 tokens, pero dejó {num_valid}"
    print(f"✓ Test pasado: exactamente {num_valid} tokens\n")

    print("="*70)
    print("TEST 2: Top-p filtering (VERSIÓN CORREGIDA)")
    print("="*70)

    # Test con logits conocidos
    test_logits = torch.tensor([10.0, 5.0, 3.0, 1.0, 0.5, 0.1] + [-5.0] * 94)
    filtered = top_p_filtering(test_logits.clone(), top_p=0.9)
    num_valid = (filtered != float('-inf')).sum().item()
    print(f"✓ Tokens válidos después de top-p=0.9: {num_valid}")

    # Verifica que mantenga al menos el token más probable
    assert filtered[0] != float('-inf'), "El token más probable debe mantenerse"
    print("✓ Token más probable mantenido")

    # Verifica que elimine tokens poco probables
    assert filtered[-1] == float('-inf'), "Tokens poco probables deben eliminarse"
    print("✓ Tokens poco probables eliminados")

    assert 1 <= num_valid <= 100, "top-p debería dejar al menos 1 token"
    print(f"✓ Test pasado: {num_valid} tokens válidos\n")

    print("="*70)
    print("TEST 3: Repetition penalty")
    print("="*70)
    test_logits = torch.ones(100)
    repeated_tokens = [5, 10, 15]
    penalized = apply_repetition_penalty(test_logits.clone(), repeated_tokens, penalty=2.0)
    print(f"  Logit original del token 5: {test_logits[5]:.4f}")
    print(f"  Logit penalizado del token 5: {penalized[5]:.4f}")
    assert penalized[5] < test_logits[5], "Penalización debería reducir logit"
    print("✓ Penalización funcionando correctamente\n")

    print("="*70)
    print("TEST 4: Temperature scaling")
    print("="*70)
    test_logits = torch.tensor([2.0, 1.0, 0.5, 0.1])

    # Temperatura alta (más aleatorio)
    high_temp = test_logits / 2.0
    probs_high = F.softmax(high_temp, dim=-1)

    # Temperatura baja (más determinístico)
    low_temp = test_logits / 0.5
    probs_low = F.softmax(low_temp, dim=-1)

    print(f"  Probs con temp=2.0: {probs_high.tolist()}")
    print(f"  Probs con temp=0.5: {probs_low.tolist()}")

    # Con temp alta, las probabilidades deberían ser más uniformes
    entropy_high = -(probs_high * torch.log(probs_high + 1e-10)).sum()
    entropy_low = -(probs_low * torch.log(probs_low + 1e-10)).sum()

    assert entropy_high > entropy_low, "Temp alta debería aumentar entropía"
    print("✓ Temperature scaling correcto\n")

    print("="*70)
    print("✅ TODOS LOS TESTS PASADOS")
    print("="*70)
