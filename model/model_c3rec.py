
import torch
import torch.nn as nn
import torch.nn.functional as F


class C3RecModel(nn.Module):
    """Simplified C3Rec-inspired model.

    - Embeds items and concepts
    - Encodes interaction sequence with a GRU
    - Estimates a concept mastery vector
    - Distills mastery and sequence representation into a shared latent
    - Scores all items for next-item prediction
    """

    def __init__(
        self,
        num_users: int,
        num_items: int,
        num_concepts: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.num_items = num_items
        self.num_concepts = num_concepts

        self.user_emb = nn.Embedding(num_users, embed_dim)
        self.item_emb = nn.Embedding(num_items, embed_dim)
        self.concept_emb = nn.Embedding(num_concepts, embed_dim)

        # GRU over (item_emb, concept_emb, answer)
        self.gru = nn.GRU(
            input_size=embed_dim * 2 + 1,
            hidden_size=hidden_dim,
            batch_first=True,
        )

        # cognitive mastery head (concept-level)
        self.mastery_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts),
            nn.Sigmoid(),
        )

        # distillation: combine mastery and sequence state
        self.distill = nn.Sequential(
            nn.Linear(hidden_dim + num_concepts, hidden_dim),
            nn.ReLU(),
        )

        # final scorer over items
        self.item_scorer = nn.Linear(hidden_dim, num_items)

    def forward(self, user_ids, seq_items, seq_concepts, seq_answers):
        """Forward pass.

        user_ids:   (B,)
        seq_items:  (B, T)
        seq_concepts:(B, T)
        seq_answers:(B, T)
        """
        batch_size, T = seq_items.shape

        item_e = self.item_emb(seq_items)         # (B, T, D)
        concept_e = self.concept_emb(seq_concepts) # (B, T, D)
        answers = seq_answers.unsqueeze(-1)       # (B, T, 1)

        seq_input = torch.cat([item_e, concept_e, answers], dim=-1)  # (B, T, 2D+1)
        _, h_last = self.gru(seq_input)          # h_last: (1, B, H)
        h_last = h_last.squeeze(0)               # (B, H)

        mastery = self.mastery_head(h_last)      # (B, num_concepts)

        distill_input = torch.cat([h_last, mastery], dim=-1)  # (B, H + num_concepts)
        z = self.distill(distill_input)                        # (B, H)

        logits = self.item_scorer(z)             # (B, num_items)
        return logits, mastery

    def predict_topk(self, user_ids, seq_items, seq_concepts, seq_answers, k: int = 10):
        """Return top-k item indices and scores for each batch element."""
        self.eval()
        with torch.no_grad():
            logits, mastery = self.forward(user_ids, seq_items, seq_concepts, seq_answers)
            scores = F.softmax(logits, dim=-1)
            topk_scores, topk_indices = torch.topk(scores, k, dim=-1)
        return topk_indices, topk_scores, mastery
