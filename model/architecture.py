import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple



class C3RecModel(nn.Module):
    """C3Rec: Continuous Correctness-aware Course Recommendation"""

    def __init__(
        self,
        num_questions: int,
        num_concepts: int,
        num_lectures: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
        seq_len: int = 200,
    ):
        super().__init__()

        self.d_model = d_model
        self.seq_len = seq_len

        # --- Discrete embeddings ---
        self.q_emb = nn.Embedding(num_questions, d_model, padding_idx=0)
        self.c_emb = nn.Embedding(num_concepts, d_model, padding_idx=0)
        self.l_emb = nn.Embedding(num_lectures, d_model, padding_idx=0)

        # --- Continuous feature projections ---
        self.elapsed_proj = nn.Linear(1, d_model)
        self.time_proj = nn.Linear(1, d_model)
        self.correct_proj = nn.Linear(1, d_model)

        # --- Positional embedding ---
        self.pos_emb = nn.Embedding(seq_len, d_model)

        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)

        # --- Heads ---
        self.correct_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
        )

        self.nextq_head = nn.Linear(d_model, num_questions)
        self.nextl_head = nn.Linear(d_model, num_lectures)

        self.ability_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        batch keys:
          - question, concept, lecture: [B, S_full] long
          - elapsed, time, correct, mask: [B, S_full] float
        The model is trained in next-step mode:
          - input: positions 0..S_full-2
          - targets: positions 1..S_full-1
        """
        q_full = batch["question"]
        c_full = batch["concept"]
        l_full = batch["lecture"]
        e_full = batch["elapsed"]
        t_full = batch["time"]
        r_full = batch["correct"]
        m_full = batch["mask"]

        # Shift for next-step predictions
        q = q_full[:, :-1]
        c = c_full[:, :-1]
        l = l_full[:, :-1]
        e = e_full[:, :-1]
        t = t_full[:, :-1]
        r = r_full[:, :-1]
        m = m_full[:, :-1]

        B, S = q.shape

        # --- Embedding + feature encoding ---
        q_emb = self.q_emb(q)
        c_emb = self.c_emb(c)
        l_emb = self.l_emb(l)
        e_emb = self.elapsed_proj(e.unsqueeze(-1))
        t_emb = self.time_proj(t.unsqueeze(-1))
        r_emb = self.correct_proj(r.unsqueeze(-1))

        seq = q_emb + c_emb + l_emb + e_emb + t_emb + r_emb

        # Positional encoding
        pos_idx = torch.arange(S, device=seq.device).unsqueeze(0).expand(B, -1)
        seq = seq + self.pos_emb(pos_idx)

        # Padding mask
        pad_mask = (m == 0)

        # --- Transformer encoder ---
        h = self.encoder(seq, src_key_padding_mask=pad_mask)  # [B, S-1, d_model]

        # --- Heads ---
        correct_next_logits = self.correct_head(h).squeeze(-1)     # [B, S-1]
        nextq_logits = self.nextq_head(h)                          # [B, S-1, Q]
        nextl_logits = self.nextl_head(h)                          # [B, S-1, L]

        # Ability: pooled over time with mask
        mask_expanded = m.unsqueeze(-1)                            # [B, S-1, 1]
        h_masked = h * mask_expanded                               # [B, S-1, d]
        seq_lengths = mask_expanded.sum(dim=1).clamp_min(1.0)      # [B, 1]
        pooled = h_masked.sum(dim=1) / seq_lengths                 # [B, d]
        ability = self.ability_mlp(pooled).squeeze(-1)             # [B]

        return {
            "correct_next_logits": correct_next_logits,
            "nextq_logits": nextq_logits,
            "nextl_logits": nextl_logits,
            "ability": ability,
            "hidden_states": h,     # [B, S-1, d_model]
            "mask_prefix": m,       # [B, S-1]
        }
