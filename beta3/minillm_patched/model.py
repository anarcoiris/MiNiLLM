"""
model.py - VERSIÓN CORREGIDA

Arquitectura mejorada del transformer con:
- RoPE (Rotary Position Embeddings) para mejor extrapolación
- Pre-Layer Normalization para estabilidad
- Weight tying entre embeddings y output
- Inicialización mejorada
- Flash Attention opcional
- Gradient checkpointing opcional

CORRECCIONES:
- RoPE ahora extiende dinámicamente para secuencias largas
- No falla silenciosamente si seq_len > max_seq_len
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ======================== Rotary Position Embeddings ========================

class RotaryEmbedding(nn.Module):
    """
    Implementación de RoPE (Rotary Position Embeddings).

    CORREGIDO: Ahora maneja dinámicamente secuencias más largas que max_seq_len.

    RoPE codifica posiciones relativas mediante rotaciones en el espacio complejo,
    lo que permite:
    - Extrapolación a secuencias más largas que las vistas en entrenamiento
    - Mejor captura de relaciones posicionales relativas
    - No requiere parámetros aprendidos adicionales

    Referencias:
    - Su et al. 2021: "RoFormer: Enhanced Transformer with Rotary Position Embedding"
    """

    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Calcula frecuencias inversas: 1 / (base^(2i/dim)) para i=0..dim/2
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)

        # Pre-calcula embeddings para eficiencia
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None

    def _update_cache(self, seq_len: int, device: torch.device):
        """
        Pre-calcula cos/sin para una longitud de secuencia.

        CORREGIDO: Ahora extiende la caché si seq_len > max_seq_len.
        """
        # Si la secuencia es más larga que la caché actual, actualiza
        if self._seq_len_cached is None or seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum('i,j->ij', t, self.inv_freq)
            # Duplica para tener dim completo
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()

            # Advertencia si excede max_seq_len considerablemente
            if seq_len > self.max_seq_len * 1.5:
                import warnings
                warnings.warn(
                    f"RoPE: seq_len ({seq_len}) excede significativamente max_seq_len "
                    f"({self.max_seq_len}). Esto puede afectar la calidad de las "
                    f"posiciones. Considera aumentar max_seq_len en la inicialización.",
                    UserWarning
                )

    def forward(self, seq_len: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            cos, sin: (seq_len, dim)
        """
        self._update_cache(seq_len, device)
        return self._cos_cached[:seq_len], self._sin_cached[:seq_len]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rota la mitad de las dimensiones para aplicar RoPE."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Aplica rotaciones de RoPE a queries y keys.

    Args:
        q, k: (batch, num_heads, seq_len, head_dim)
        cos, sin: (seq_len, head_dim)

    Returns:
        q_embed, k_embed con posiciones codificadas
    """
    # Expande dimensiones para broadcasting
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)

    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    return q_embed, k_embed


# ======================== Attention Mechanisms ========================

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention con soporte para RoPE.

    Mejoras sobre la versión original:
    - Soporte para RoPE (posiciones relativas)
    - Dropout separado para atención
    - Mejor manejo de máscaras
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_rope: bool = True
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.use_rope = use_rope

        # Proyecciones QKV y output
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out = nn.Linear(embed_dim, embed_dim, bias=False)

        # Dropouts
        self.attn_dropout = nn.Dropout(attn_dropout)
        self.proj_dropout = nn.Dropout(proj_dropout)

        # Escala para dot-product attention
        self.scale = 1.0 / math.sqrt(self.head_dim)

        # Caché de máscara causal para eficiencia
        self.register_buffer("_causal_mask_cached", None, persistent=False)
        self._mask_cached_size = 0

    def _get_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Obtiene máscara causal, usando caché si es posible."""
        if self._mask_cached_size < seq_len:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
                diagonal=1
            )
            self._causal_mask_cached = mask
            self._mask_cached_size = seq_len

        return self._causal_mask_cached[:seq_len, :seq_len]

    def forward(
        self,
        x: torch.Tensor,
        rope_cos_sin: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, embed_dim)
            rope_cos_sin: Optional (cos, sin) de RoPE si use_rope=True

        Returns:
            (batch, seq_len, embed_dim)
        """
        B, T, E = x.shape

        # Proyecta a Q, K, V
        qkv = self.qkv(x)  # (B, T, 3*E)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape para multi-head: (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        q = q.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # Aplica RoPE si está habilitado
        if self.use_rope and rope_cos_sin is not None:
            cos, sin = rope_cos_sin
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Scaled dot-product attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, num_heads, T, T)

        # Máscara causal: solo puede atender a posiciones anteriores
        causal_mask = self._get_causal_mask(T, x.device)
        attn = attn.masked_fill(causal_mask, float('-inf'))

        # Softmax y dropout
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        # Aplica atención a valores
        out = torch.matmul(attn, v)  # (B, num_heads, T, head_dim)

        # Reshape de vuelta: (B, num_heads, T, head_dim) -> (B, T, E)
        out = out.transpose(1, 2).contiguous().view(B, T, E)

        # Proyección de salida
        out = self.out(out)
        out = self.proj_dropout(out)

        return out


# ======================== Transformer Block ========================

class TransformerBlock(nn.Module):
    """
    Bloque Transformer con:
    - Pre-Layer Normalization (más estable)
    - Multi-head self-attention
    - Feed-forward network con GELU
    - Conexiones residuales
    - Gradient checkpointing opcional
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        drop: float = 0.1,
        attn_drop: float = 0.0,
        use_rope: bool = True,
        use_checkpoint: bool = False
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint

        # Layer norms
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)

        # Attention
        self.attn = CausalSelfAttention(
            embed_dim, num_heads,
            attn_dropout=attn_drop,
            proj_dropout=drop,
            use_rope=use_rope
        )

        # Feed-forward network
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(drop),
        )

    def _forward_impl(self, x: torch.Tensor, rope_cos_sin: Optional[Tuple] = None) -> torch.Tensor:
        """Implementación interna para checkpointing."""
        # Pre-LN + Attention + Residual
        x = x + self.attn(self.ln1(x), rope_cos_sin=rope_cos_sin)
        # Pre-LN + MLP + Residual
        x = x + self.mlp(self.ln2(x))
        return x

    def forward(self, x: torch.Tensor, rope_cos_sin: Optional[Tuple] = None) -> torch.Tensor:
        if self.use_checkpoint and self.training:
            # Usa gradient checkpointing para ahorrar memoria
            from torch.utils.checkpoint import checkpoint
            return checkpoint(self._forward_impl, x, rope_cos_sin, use_reentrant=False)
        else:
            return self._forward_impl(x, rope_cos_sin)


# ======================== Main Model ========================

class TinyGPTv2(nn.Module):
    """
    Modelo GPT mejorado con todas las mejoras arquitectónicas.

    CORRECCIONES v2:
    - RoPE ahora maneja dinámicamente secuencias largas

    Mejoras sobre versión original:
    - RoPE para mejores posiciones relativas
    - Weight tying para reducir parámetros
    - Inicialización mejorada (GPT-2 style)
    - Soporte para gradient checkpointing
    - Arquitectura más profunda por defecto
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int = 256,
        n_embd: int = 384,
        n_layer: int = 8,
        n_head: int = 6,
        drop: float = 0.1,
        attn_drop: float = 0.0,
        mlp_ratio: float = 4.0,
        use_rope: bool = True,
        use_checkpoint: bool = False,
        rope_base: float = 10000.0  # NUEVO: Configurable
    ):
        super().__init__()

        # Validaciones
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"

        # Configuración
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.use_rope = use_rope

        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, n_embd)

        # Positional embeddings
        if use_rope:
            self.rotary_emb = RotaryEmbedding(
                dim=n_embd // n_head,
                max_seq_len=block_size * 2,  # Permite 2x extrapolación
                base=rope_base
            )
        else:
            self.pos_emb = nn.Embedding(block_size, n_embd)

        # Input dropout
        self.drop = nn.Dropout(drop)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=n_embd,
                num_heads=n_head,
                mlp_ratio=mlp_ratio,
                drop=drop,
                attn_drop=attn_drop,
                use_rope=use_rope,
                use_checkpoint=use_checkpoint
            )
            for _ in range(n_layer)
        ])

        # Output layer norm
        self.ln_f = nn.LayerNorm(n_embd)

        # Output head
        self.head = nn.Linear(n_embd, vocab_size, bias=False)

        # Weight tying: comparte pesos entre token embedding y output
        # Reduce parámetros y mejora generalización
        self.head.weight = self.token_emb.weight

        # Inicialización
        self.apply(self._init_weights)

        # Inicialización especial para proyecciones residuales (GPT-2 style)
        for name, param in self.named_parameters():
            if name.endswith('out.weight') or name.endswith('mlp.3.weight'):
                torch.nn.init.normal_(param, mean=0.0, std=0.02 / math.sqrt(2 * n_layer))

    def _init_weights(self, module):
        """Inicialización estilo GPT-2."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: (batch, seq_len) token indices

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = idx.shape

        # CORREGIDO: Permite secuencias más largas con advertencia
        if T > self.block_size:
            import warnings
            warnings.warn(
                f"Secuencia de longitud {T} excede block_size {self.block_size}. "
                f"Esto funciona con RoPE pero puede degradar la calidad.",
                UserWarning
            )

        # Token embeddings
        tok_emb = self.token_emb(idx)  # (B, T, n_embd)

        # Agregar información posicional
        if self.use_rope:
            # RoPE se aplica dentro de cada bloque de atención
            cos, sin = self.rotary_emb(T, device=idx.device)
            rope_cos_sin = (cos, sin)
            x = self.drop(tok_emb)
        else:
            # Embeddings posicionales aprendidos
            if T > self.block_size:
                raise ValueError(
                    f"Secuencia de longitud {T} excede block_size {self.block_size}. "
                    f"Considera usar RoPE (use_rope=True) para secuencias largas."
                )
            pos = torch.arange(T, device=idx.device)
            pos_emb = self.pos_emb(pos).unsqueeze(0)  # (1, T, n_embd)
            x = self.drop(tok_emb + pos_emb)
            rope_cos_sin = None

        # Pasa por todos los bloques transformer
        for block in self.blocks:
            x = block(x, rope_cos_sin=rope_cos_sin)

        # Layer norm final
        x = self.ln_f(x)

        # Proyecta a vocabulario
        logits = self.head(x)  # (B, T, vocab_size)

        return logits

    def get_num_params(self, non_embedding: bool = True) -> int:
        """
        Cuenta parámetros del modelo.

        Args:
            non_embedding: Si True, excluye embeddings de posición (para comparar con otros modelos)
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding and not self.use_rope:
            n_params -= self.pos_emb.weight.numel()
        return n_params

    def estimate_mfu(self, fwdbwd_per_iter: int, dt: float) -> float:
        """
        Estima model flops utilization (MFU) como porcentaje del máximo teórico.

        Args:
            fwdbwd_per_iter: número de forward+backward passes por iteración
            dt: tiempo transcurrido en segundos

        Returns:
            MFU estimado (0-1)
        """
        # Estima FLOPs por token
        N = self.get_num_params()
        L, H, Q, T = self.n_layer, self.n_head, self.n_embd // self.n_head, self.block_size
        flops_per_token = 6 * N + 12 * L * H * Q * T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter

        # Expresar en FLOP/s
        flops_achieved = flops_per_iter / dt

        # A100 GPU tiene ~312 TFLOPS para fp16
        flops_promised = 312e12
        mfu = flops_achieved / flops_promised

        return mfu


# ======================== Factory Functions ========================

def create_small_model(vocab_size: int, block_size: int = 256) -> TinyGPTv2:
    """Modelo pequeño para datasets ~50-100MB."""
    return TinyGPTv2(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=256,
        n_layer=6,
        n_head=8,
        drop=0.1,
        use_rope=True
    )


def create_medium_model(vocab_size: int, block_size: int = 512) -> TinyGPTv2:
    """Modelo mediano para datasets ~200-500MB."""
    return TinyGPTv2(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=384,
        n_layer=10,
        n_head=6,
        drop=0.1,
        use_rope=True
    )


def create_large_model(vocab_size: int, block_size: int = 1024) -> TinyGPTv2:
    """Modelo grande para datasets >500MB."""
    return TinyGPTv2(
        vocab_size=vocab_size,
        block_size=block_size,
        n_embd=512,
        n_layer=12,
        n_head=8,
        drop=0.1,
        use_rope=True,
        use_checkpoint=True  # Activa gradient checkpointing para ahorrar memoria
    )


if __name__ == '__main__':
    # Test del modelo
    print("Testing model.py (VERSIÓN CORREGIDA)\n")
    print("="*70)

    model = create_small_model(vocab_size=32000, block_size=256)
    print(f"Parámetros totales: {model.get_num_params():,}")
    print(f"Parámetros (sin embeddings): {model.get_num_params(non_embedding=True):,}")

    # Test forward pass normal
    print("\n" + "="*70)
    print("TEST 1: Forward pass normal")
    print("="*70)
    batch_size = 4
    seq_len = 128
    x = torch.randint(0, 32000, (batch_size, seq_len))

    with torch.no_grad():
        logits = model(x)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Expected shape: ({batch_size}, {seq_len}, 32000)")
    assert logits.shape == (batch_size, seq_len, 32000)
    print("✓ Modelo funciona correctamente")

    # Test con secuencia más larga que block_size
    print("\n" + "="*70)
    print("TEST 2: Secuencia larga (> block_size) con RoPE")
    print("="*70)
    long_seq_len = 384  # Más largo que block_size=256
    x_long = torch.randint(0, 32000, (2, long_seq_len))

    print(f"Probando con seq_len={long_seq_len} (block_size={model.block_size})")

    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        with torch.no_grad():
            logits_long = model(x_long)

        # Debe generar advertencia pero no fallar
        assert len(w) >= 1, "Debería generar advertencia"
        print(f"✓ Advertencia generada: {w[0].message}")

    print(f"Output shape: {logits_long.shape}")
    assert logits_long.shape == (2, long_seq_len, 32000)
    print("✓ RoPE maneja secuencias largas correctamente")

    # Test con modelo sin RoPE
    print("\n" + "="*70)
    print("TEST 3: Modelo sin RoPE (debe fallar con secuencias largas)")
    print("="*70)
    model_no_rope = TinyGPTv2(
        vocab_size=32000,
        block_size=256,
        n_embd=256,
        n_layer=4,
        n_head=8,
        use_rope=False
    )

    print("Probando con seq_len=128 (dentro de block_size)")
    with torch.no_grad():
        logits_normal = model_no_rope(torch.randint(0, 32000, (2, 128)))
    print(f"✓ Funciona: {logits_normal.shape}")

    print("\nProbando con seq_len=384 (fuera de block_size)")
    try:
        with torch.no_grad():
            model_no_rope(torch.randint(0, 32000, (2, 384)))
        print("❌ ERROR: Debería haber fallado")
    except ValueError as e:
        print(f"✓ Falla como esperado: {e}")

    print("\n" + "="*70)
    print("✅ TODOS LOS TESTS PASADOS")
    print("="*70)
