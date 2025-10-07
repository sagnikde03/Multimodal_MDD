"""
================================================================
  SliceTransformer-Based Multimodal EEG-Audio Classification
 ------------------------------------------------------------
  This implementation demonstrates end-to-end data loading,
  preprocessing, and training for the proposed SliceTransformer
  architecture. It simulates a realistic experimental pipeline
  for Major Depressive Disorder (MDD) detection using:
    • EEG Superlet Wavelet Transformed TF images
    • Audio Mel Spectrograms

  Each TF image (EEG or Audio) is treated as a 2D matrix (N×C):
    - Rows (N): Temporal frames
    - Columns (C): Frequency bins

  The model processes row-wise spectral slices as sequential tokens,
  adds learnable positional embeddings, and encodes them via:
    - Parallel Encoder (PE): Two MHSA + MLP branches in parallel
    - Class Encoder (CE): Transformer block for global class-level fusion

  The final EEG and Audio class tokens are concatenated and passed
  through a small MLP head for binary classification (MDD vs Control).

  Author: Sagnik De, Anurag Singh, A.K. Bhandari
  Framework: PyTorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ParallelTransformerEncoderBlock(nn.Module):
    """
    Parallel Encoder (PE) block:
    Implements the PE equations:
        X'_k+1 = MHSA1(LN(X_k)) + MHSA2(LN(X_k)) + X_k
        X_k+1  = MLP1(LN(X'_k+1)) + MLP2(LN(X'_k+1)) + X'_k+1

    - Pre-norm: LayerNorm before attention and before MLPs (matches your LN placement).
    - Uses two independent MultiHeadAttention modules and two independent MLPs,
      their outputs are summed (parallel paths), then residuals applied.
    """
    def __init__(self, dim: int, num_heads: int, mlp_hidden_dim: int,
                 attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        self.dim = dim

        # LayerNorm before attention (pre-norm)
        self.norm1 = nn.LayerNorm(dim)

        # Two parallel MHSA modules (they internally create W_q, W_k, W_v, W_o)
        # batch_first=True so input is (batch, seq_len, dim)
        self.mhsa1 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                           dropout=attn_dropout, batch_first=True)
        self.mhsa2 = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                           dropout=attn_dropout, batch_first=True)

        # small dropout after attention outputs (projection dropout)
        self.attn_dropout = nn.Dropout(proj_dropout)

        # LayerNorm before MLP (pre-norm)
        self.norm2 = nn.LayerNorm(dim)

        # Two independent MLPs with GeLU nonlinearity and output projection back to dim
        self.mlp1 = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(proj_dropout)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(proj_dropout)
        )

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim) -- sequence of spectral slices (tokens)
            attn_mask: optional attention mask (see torch.nn.MultiheadAttention docs)
            key_padding_mask: optional padding mask for variable-length sequences
        Returns:
            x_next: (batch, seq_len, dim)
        """
        # Pre-attention normalization
        x_norm = self.norm1(x)

        # MHSA path 1
        attn_out1, _ = self.mhsa1(x_norm, x_norm, x_norm,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask)

        # MHSA path 2
        attn_out2, _ = self.mhsa2(x_norm, x_norm, x_norm,
                                  attn_mask=attn_mask,
                                  key_padding_mask=key_padding_mask)

        # Sum parallel attention outputs, apply dropout and residual add
        attn_sum = attn_out1 + attn_out2
        attn_sum = self.attn_dropout(attn_sum)
        x_prime = x + attn_sum  # X'_{k+1} = X_k + (MHSA1 + MHSA2)

        # Pre-MLP normalization
        x_prime_norm = self.norm2(x_prime)

        # Parallel MLPs
        mlp_out1 = self.mlp1(x_prime_norm)
        mlp_out2 = self.mlp2(x_prime_norm)

        # Sum parallel MLP outputs and residual
        mlp_sum = mlp_out1 + mlp_out2
        x_next = x_prime + mlp_sum  # X_{k+1}

        return x_next


class ClassEncoderBlock(nn.Module):
    """
    Class Encoder (CE) block:
    Standard transformer-like block operating on the sequence including the class token:
        X' = X + MHSA(LN(X))
        X_out = X' + MLP(LN(X'))
    """
    def __init__(self, dim: int, num_heads: int, mlp_hidden_dim: int,
                 attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mhsa = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                          dropout=attn_dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(proj_dropout)

        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(proj_dropout)
        )

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (batch, seq_len+1, dim) where last token is class token
        """
        x_norm = self.norm1(x)
        attn_out, _ = self.mhsa(x_norm, x_norm, x_norm,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask)
        x_prime = x + self.attn_dropout(attn_out)

        x_prime_norm = self.norm2(x_prime)
        mlp_out = self.mlp(x_prime_norm)
        x_out = x_prime + mlp_out
        return x_out


class SliceTransformer(nn.Module):
    """
    SliceTransformer (formerly SpectralTransformer):
    - Input: (batch, N, C) : N temporal frames (tokens), C frequency bins (embedding dim)
    - Adds learnable positional embeddings (shape: 1 x N x C) added element-wise to each slice
    - Applies a stack of Parallel Encoder (PE) blocks
    - Appends a learnable class token (1 token) to the sequence
    - Applies Class Encoder (CE) stack on sequence+class token
    - Returns class token embedding (batch, C)
    """

    def __init__(self, seq_len: int, dim: int,
                 num_heads: int = 4, mlp_hidden_dim: Optional[int] = None,
                 num_pe_blocks: int = 2, num_ce_blocks: int = 1,
                 attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        """
        Args:
            seq_len: N, number of temporal frames (tokens)
            dim: C, embedding dimension (frequency bins)
            num_heads: H, number of attention heads
            mlp_hidden_dim: D_h, MLP hidden dim (if None, default = 4 * dim)
            num_pe_blocks: number of sequential Parallel Encoder blocks
            num_ce_blocks: number of Class Encoder blocks
        """
        super().__init__()
        self.seq_len = seq_len
        self.dim = dim
        if mlp_hidden_dim is None:
            mlp_hidden_dim = 4 * dim

        # Learnable positional embeddings per time frame: p_i in R^{1 x C} for each i in 1..N
        # shape: (1, seq_len, dim) so it can be broadcast and added element-wise
        self.pos_embed = nn.Parameter(torch.zeros(1, seq_len, dim))

        # Learnable class token (1 x 1 x dim), appended after PE
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        # Initialize positional and class embeddings
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Stack of Parallel Encoder blocks (PE)
        self.pe_blocks = nn.ModuleList([
            ParallelTransformerEncoderBlock(dim=dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim,
                                            attn_dropout=attn_dropout, proj_dropout=proj_dropout)
            for _ in range(num_pe_blocks)
        ])

        # Stack of Class Encoder blocks (CE) to process sequence including class token
        self.ce_blocks = nn.ModuleList([
            ClassEncoderBlock(dim=dim, num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim,
                              attn_dropout=attn_dropout, proj_dropout=proj_dropout)
            for _ in range(num_ce_blocks)
        ])

        # Final LayerNorm applied to the extracted class token (good practice)
        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor,
                attn_mask: Optional[torch.Tensor] = None,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, dim) exactly matching seq_len and dim provided at init
            attn_mask: optional attention mask (LxS or appropriate shape for nn.MultiheadAttention)
            key_padding_mask: optional padding mask (batch, seq_len) with True values in positions that should be masked
        Returns:
            cls_embedding: (batch, dim) final class token embedding after CE and final_norm
        """
        batch_size, seq_len, dim = x.shape
        assert seq_len == self.seq_len, f"Expected seq_len {self.seq_len}, got {seq_len}"
        assert dim == self.dim, f"Expected dim {self.dim}, got {dim}"

        # 1) Add learnable positional embeddings element-wise to each spectral slice (token)
        # pos_embed shape (1, seq_len, dim) broadcasts over batch dimension
        x = x + self.pos_embed  # preserves temporal ordering via learned pos embeddings

        # 2) Pass through stacked Parallel Encoder (PE) blocks
        for pe in self.pe_blocks:
            x = pe(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        # 3) Append learnable class token (replicate across batch)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, dim)
        x_with_cls = torch.cat([x, cls_tokens], dim=1)  # (batch, seq_len+1, dim)

        # 4) Process with Class Encoder (CE) blocks
        for ce in self.ce_blocks:
            x_with_cls = ce(x_with_cls, attn_mask=attn_mask, key_padding_mask=key_padding_mask)

        # 5) Extract class token (last token) and apply final layernorm
        cls_embedding = x_with_cls[:, -1, :]  # (batch, dim)
        cls_embedding = self.final_norm(cls_embedding)

        return cls_embedding  # (batch, dim)


class MultimodalSliceClassifier(nn.Module):
    """
    Full multimodal model composed of:
    - EEG stream: SliceTransformer
    - Audio stream: SliceTransformer
    - Fusion: concatenate class token embeddings from both streams
    - Classification head: Linear( fused_dim -> 64 ) -> ReLU -> Dropout(0.3) -> Linear(64 -> 1)
    - Training uses BCEWithLogitsLoss (so classifier returns logits)
    """
    def __init__(self,
                 seq_len_eeg: int, seq_len_audio: int,
                 dim_eeg: int, dim_audio: int,
                 num_heads: int = 4,
                 mlp_hidden_dim: Optional[int] = None,
                 num_pe_blocks: int = 2,
                 num_ce_blocks: int = 1,
                 attn_dropout: float = 0.0,
                 proj_dropout: float = 0.0,
                 classifier_hidden: int = 64,
                 dropout_prob: float = 0.3):
        super().__init__()

        # EEG and audio SliceTransformers (identical architecture but independent weights)
        self.eeg_transformer = SliceTransformer(seq_len=seq_len_eeg, dim=dim_eeg,
                                                num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim,
                                                num_pe_blocks=num_pe_blocks, num_ce_blocks=num_ce_blocks,
                                                attn_dropout=attn_dropout, proj_dropout=proj_dropout)

        self.audio_transformer = SliceTransformer(seq_len=seq_len_audio, dim=dim_audio,
                                                  num_heads=num_heads, mlp_hidden_dim=mlp_hidden_dim,
                                                  num_pe_blocks=num_pe_blocks, num_ce_blocks=num_ce_blocks,
                                                  attn_dropout=attn_dropout, proj_dropout=proj_dropout)

        # Fusion head: concatenation then small MLP head
        fused_dim = dim_eeg + dim_audio
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, classifier_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_prob),
            nn.Linear(classifier_hidden, 1)  # final logit for BCEWithLogitsLoss
        )

    def forward(self, eeg_x: torch.Tensor, audio_x: torch.Tensor,
                eeg_key_padding_mask: Optional[torch.Tensor] = None,
                audio_key_padding_mask: Optional[torch.Tensor] = None) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            eeg_x: (batch, seq_len_eeg, dim_eeg)
            audio_x: (batch, seq_len_audio, dim_audio)
            eeg_key_padding_mask: optional (batch, seq_len_eeg) mask with True for padded positions
            audio_key_padding_mask: optional (batch, seq_len_audio)
        Returns:
            logits: (batch, 1) raw logits suitable for BCEWithLogitsLoss
            probs: (batch, 1) sigmoid(logits) probabilities for interpretation
        """
        # Get class token embeddings from each stream
        cls_eeg = self.eeg_transformer(eeg_x, key_padding_mask=eeg_key_padding_mask)      # (batch, dim_eeg)
        cls_audio = self.audio_transformer(audio_x, key_padding_mask=audio_key_padding_mask)  # (batch, dim_audio)

        # Concatenate fused representation
        fused = torch.cat([cls_eeg, cls_audio], dim=1)  # (batch, fused_dim)

        # Classification head -> logits (no sigmoid here)
        logits = self.classifier(fused)  # (batch, 1)
        probs = torch.sigmoid(logits)    # (batch, 1) for interpretability

        return logits, probs

