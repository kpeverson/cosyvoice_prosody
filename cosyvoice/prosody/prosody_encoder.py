"""
Standalone PyTorch port of the fairseq-based ProsodyvecModel (prosodyvec).

Architecture (layer_norm_first=False, post-norm):
  ConvFeatureExtractor  → LayerNorm(512) → Linear(512→768)
  → TransformerEncoder:
       pos_conv (Conv1d weight-normed 128-tap, groups=16) + GELU
       → LayerNorm(768) [applied BEFORE transformer layers]
       → 12 × TransformerEncoderLayer (post-norm, GELU FFN)
  → 768-dim features at 62.5 Hz (stride 256 @ 16 kHz)

Usage:
    enc = ProsodyEncoder.from_checkpoint("path/to/checkpoint_best.pt")
    enc.eval().to(device)
    with torch.no_grad():
        feats, pad_mask = enc(audio_16khz, padding_mask)   # feats: [B, T', 768]
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Conv feature extractor
# ---------------------------------------------------------------------------

class ConvFeatureExtractor(nn.Module):
    """
    7-layer conv feature extractor.
    Config: [(512,10,4)] + [(512,3,2)] * 4 + [(512,2,2)] * 2
    First layer: Conv → GroupNorm(512,512)
    Other layers: Conv → GELU
    Input: [B, T], Output: [B, T', 512]
    """

    CONV_CFG = [(512, 10, 4)] + [(512, 3, 2)] * 4 + [(512, 2, 2)] * 2

    def __init__(self):
        super().__init__()
        layers = []
        in_d = 1
        for i, (n, k, s) in enumerate(self.CONV_CFG):
            conv = nn.Conv1d(in_d, n, k, stride=s, bias=False)
            if i == 0:
                # GroupNorm(num_groups=n, num_channels=n) ≡ LayerNorm over channels
                block = nn.Sequential(conv, nn.Dropout(0.0), nn.GroupNorm(n, n, affine=True))
            else:
                block = nn.Sequential(conv, nn.Dropout(0.0), nn.GELU())
            layers.append(block)
            in_d = n
        self.conv_layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)           # [B, 1, T]
        for block in self.conv_layers:
            x = block(x)             # [B, 512, T']
        return x.transpose(1, 2)     # [B, T', 512]


# ---------------------------------------------------------------------------
# Self-attention with separate q/k/v projections (matching fairseq keys)
# ---------------------------------------------------------------------------

class _MultiheadSelfAttention(nn.Module):
    """
    Multi-head self-attention using separate q/k/v projection modules so that
    state-dict keys match the fairseq checkpoint exactly:
      self_attn.q_proj.{weight,bias}
      self_attn.k_proj.{weight,bias}
      self_attn.v_proj.{weight,bias}
      self_attn.out_proj.{weight,bias}
    """

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout = dropout

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:                [B, T, D]
            key_padding_mask: [B, T] bool, True = pad position
        Returns:
            [B, T, D]
        """
        B, T, D = x.shape
        H, d = self.num_heads, self.head_dim

        q = self.q_proj(x).view(B, T, H, d).transpose(1, 2)   # [B,H,T,d]
        k = self.k_proj(x).view(B, T, H, d).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, d).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,H,T,T]

        if key_padding_mask is not None:
            attn = attn.masked_fill(
                key_padding_mask.unsqueeze(1).unsqueeze(2), float("-inf")
            )

        attn = F.softmax(attn, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.matmul(attn, v)               # [B,H,T,d]
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# Transformer encoder layer  (post-norm, layer_norm_first=False)
# ---------------------------------------------------------------------------

class TransformerEncoderLayer(nn.Module):
    """
    fairseq TransformerSentenceEncoderLayer with layer_norm_first=False (post-norm).
    State-dict key layout:
      self_attn.{q,k,v,out}_proj.{weight,bias}
      self_attn_layer_norm.{weight,bias}
      fc1.{weight,bias}, fc2.{weight,bias}
      final_layer_norm.{weight,bias}
    """

    def __init__(
        self,
        embed_dim: int = 768,
        ffn_dim: int = 3072,
        num_heads: int = 12,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
    ):
        super().__init__()
        self.self_attn = _MultiheadSelfAttention(embed_dim, num_heads, attention_dropout)
        self.self_attn_layer_norm = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, ffn_dim)
        self.fc2 = nn.Linear(ffn_dim, embed_dim)
        self.final_layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = dropout
        self.activation_dropout = activation_dropout

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Post-norm: sublayer → dropout → residual → layer_norm
        residual = x
        x_attn = self.self_attn(x, key_padding_mask=key_padding_mask)
        x = residual + F.dropout(x_attn, p=self.dropout, training=self.training)
        x = self.self_attn_layer_norm(x)

        residual = x
        x = F.gelu(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = residual + F.dropout(x, p=self.dropout, training=self.training)
        x = self.final_layer_norm(x)
        return x


# ---------------------------------------------------------------------------
# Transformer encoder (pos_conv + layer_norm + N layers)
# ---------------------------------------------------------------------------

class _TransformerEncoder(nn.Module):
    """
    fairseq TransformerEncoder (layer_norm_first=False):
      pos_conv → encoder.layer_norm (BEFORE layers) → 12 × TransformerEncoderLayer
    State-dict keys live under the 'encoder.' prefix in the full checkpoint.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        conv_pos: int = 128,
        conv_pos_groups: int = 16,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
    ):
        super().__init__()
        # pos_conv: stored as plain Conv1d here; weights loaded from weight-norm form
        self.pos_conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size=conv_pos,
            padding=conv_pos // 2,
            groups=conv_pos_groups,
        )
        self._conv_pos = conv_pos

        # Applied BEFORE transformer layers when layer_norm_first=False
        self.layer_norm = nn.LayerNorm(embed_dim)

        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                embed_dim=embed_dim,
                ffn_dim=ffn_dim,
                num_heads=num_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                activation_dropout=activation_dropout,
            )
            for _ in range(num_layers)
        ])
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:            [B, T, D]
            padding_mask: [B, T] bool, True = pad position
        Returns:
            [B, T, D]
        """
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0.0)

        # Positional conv
        x_conv = self.pos_conv(x.transpose(1, 2))  # [B, D, T+1]
        x_conv = x_conv[:, :, :-1]                  # trim to [B, D, T]
        x_conv = F.gelu(x_conv)
        x = x + x_conv.transpose(1, 2)

        # Input normalisation (layer_norm_first=False → norm is before layers)
        x = self.layer_norm(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        for layer in self.layers:
            x = layer(x, key_padding_mask=padding_mask)

        return x


# ---------------------------------------------------------------------------
# Full prosody encoder
# ---------------------------------------------------------------------------

class ProsodyEncoder(nn.Module):
    """
    Standalone port of ProsodyvecModel for inference only.

    forward() input:  raw audio at src_sample_rate (will be resampled to 16 kHz
                      BEFORE calling this module — caller is responsible).
    forward() output: (features [B, T', 768], padding_mask [B, T'] bool)
                      at ~62.5 Hz (conv stride 256 @ 16 kHz).
    """

    def __init__(
        self,
        embed_dim: int = 768,
        conv_embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 12,
        ffn_dim: int = 3072,
        conv_pos: int = 128,
        conv_pos_groups: int = 16,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.0,
    ):
        super().__init__()
        self.feature_extractor = ConvFeatureExtractor()
        self.layer_norm = nn.LayerNorm(conv_embed_dim)
        self.post_extract_proj = nn.Linear(conv_embed_dim, embed_dim)
        self.encoder = _TransformerEncoder(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ffn_dim=ffn_dim,
            conv_pos=conv_pos,
            conv_pos_groups=conv_pos_groups,
            dropout=dropout,
            attention_dropout=attention_dropout,
            activation_dropout=activation_dropout,
        )
        self._embed_dim = embed_dim

    def forward(
        self,
        source: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            source:       [B, T] float32, 16 kHz glottal-source audio.
            padding_mask: [B, T] bool, True = pad sample (optional).
        Returns:
            features:     [B, T', embed_dim] at ~62.5 Hz
            out_mask:     [B, T'] bool padding mask for the output frames
        """
        features = self.feature_extractor(source)       # [B, T', 512]
        features = self.layer_norm(features)
        features = self.post_extract_proj(features)     # [B, T', 768]

        # Propagate padding mask through the conv strides
        if padding_mask is not None:
            out_mask = self._propagate_padding_mask(padding_mask, source.shape[1], features.shape[1])
        else:
            out_mask = torch.zeros(
                features.shape[:2], dtype=torch.bool, device=features.device
            )

        features = self.encoder(features, padding_mask=out_mask)
        return features, out_mask

    @staticmethod
    def _propagate_padding_mask(
        padding_mask: torch.Tensor, src_len: int, tgt_len: int
    ) -> torch.Tensor:
        """Down-sample a [B, src_len] bool mask to [B, tgt_len]."""
        if padding_mask.shape[1] == tgt_len:
            return padding_mask
        # Simple nearest-neighbour: sample uniformly spaced positions
        indices = torch.linspace(0, src_len - 1, tgt_len, device=padding_mask.device).long()
        return padding_mask[:, indices]

    # ------------------------------------------------------------------
    # Checkpoint loading
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(cls, path: str, device: str = "cpu") -> "ProsodyEncoder":
        """
        Load weights from either:
          - a fairseq prosodyvec checkpoint  (contains 'model' key + fairseq cfg)
          - a pre-extracted weights-only file (plain state dict produced by
            ``torch.save(ckpt['model'], path)`` in the fairseq environment)

        Training-only modules (mask_emb, label_embs_concat, final_proj,
        spkr_class_layer, etc.) are silently skipped.
        The pos_conv weight is reconstructed from weight_g / weight_v.
        """
        raw = torch.load(path, map_location=device, weights_only=True)
        # Detect whether this is a full fairseq checkpoint or a plain state dict
        if isinstance(raw, dict) and "model" in raw and isinstance(raw["model"], dict):
            sd = raw["model"]
        else:
            sd = raw  # already a state dict

        model = cls()

        # ---- Reconstruct pos_conv weight from weight-norm parameters ----
        weight_g = sd.pop("encoder.pos_conv.0.weight_g")   # [1, 1, K]
        weight_v = sd.pop("encoder.pos_conv.0.weight_v")   # [D, D//G, K]
        bias     = sd.pop("encoder.pos_conv.0.bias")       # [D]
        norm_v   = weight_v.norm(dim=(0, 1), keepdim=True)
        weight   = weight_g * weight_v / norm_v            # [D, D//G, K]
        sd["encoder.pos_conv.weight"] = weight
        sd["encoder.pos_conv.bias"]   = bias

        # ---- Strip training-only keys ----
        skip_prefixes = (
            "mask_emb",
            "label_embs_concat",
            "left_boundary_emb",
            "right_boundary_emb",
            "left_rel_pos_emb",
            "right_rel_pos_emb",
            "label_embs_concat_span",
            "final_proj",
            "final_span_proj",
            "spkr_class_layer",
        )
        sd = {k: v for k, v in sd.items() if not k.startswith(skip_prefixes)}

        missing, unexpected = model.load_state_dict(sd, strict=False)

        # Any remaining unexpected keys are a sign of trouble
        if unexpected:
            raise RuntimeError(
                f"ProsodyEncoder.from_checkpoint: unexpected keys in checkpoint:\n"
                + "\n".join(f"  {k}" for k in unexpected)
            )
        # Missing keys that are NOT in skip_prefixes would indicate a bug
        non_skip_missing = [k for k in missing if not any(k.startswith(p) for p in skip_prefixes)]
        if non_skip_missing:
            import warnings
            warnings.warn(
                f"ProsodyEncoder.from_checkpoint: missing keys:\n"
                + "\n".join(f"  {k}" for k in non_skip_missing)
            )

        return model
