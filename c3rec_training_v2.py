import os
import time
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# =============================
# CONFIGURATION
# =============================

CONFIG = {
    # Data
    "dataset_path": "c3rec_dataset_seq.pt",
    "subset_ratio": 1,  # 0.01=1%, 0.1=10%, 0.5=50%, 1.0=full
    "train_split": 0.8,
    
    # Model architecture
    "d_model": 128,
    "n_heads": 4,
    "n_layers": 2,
    "dropout": 0.1,
    
    # Training
    "batch_size": 64,
    "learning_rate": 5e-5,
    "epochs": 20,
    "grad_clip": 1.0,  # Changed from 0.5 to more standard value
    
    # Loss weights
    "weight_correct": 1.0,
    "weight_nextq": 0.1,
    "weight_nextl": 0.05,
    "weight_ability": 0.05,
    
    # Evaluation
    "ndcg_k": 10,
    
    # Output
    "model_version": "v10",  # Consistent versioning
    "save_best_only": True,
}

# =============================
# STEP 1 — Load Dataset
# =============================

print("=" * 60)
print("LOADING DATASET")
print("=" * 60)

DATASET_PATH = CONFIG["dataset_path"]

if not os.path.exists(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

data = torch.load(DATASET_PATH, map_location="cpu")

print(f"✓ Loaded keys: {list(data.keys())}")
print(f"✓ Total users: {len(data['user_ids'])}")
print(f"✓ Sequence length: {data['seq_len']}")

SEQ_LEN = data["seq_len"]

# =============================
# STEP 2 — Dataset & DataLoader
# =============================

# =============================
# STEP 2 — Dataset & DataLoader
# =============================

class C3RecDataset(Dataset):
    """
    C3Rec Dataset for EdNet-KT1 style data.
    
    Computes normalization statistics ONLY on the provided data split
    (not the full dataset) to avoid train/test leakage.
    
    Notes:
    - ID 0 is reserved for padding/invalid entries
    - Mask=1 indicates valid data, Mask=0 indicates padding
    """
    
    def __init__(self, data, compute_stats=True):
        self.user_ids      = data["user_ids"]
        self.X_question    = data["X_question"]
        self.X_lecture     = data["X_lecture"]
        self.X_concept     = data["X_concept"]
        self.X_elapsed     = data["X_elapsed"]
        self.X_time        = data["X_time"]
        self.y_correct     = data["y_correct"]
        self.mask          = data["mask"]

        self.question_vocab = data["question_vocab"]
        self.lecture_vocab  = data["lecture_vocab"]
        self.concept_vocab  = data["concept_vocab"]

        # Initialize stats to None
        self.elapsed_mean = None
        self.elapsed_std = None
        self.time_mean = None
        self.time_std = None

    def set_normalization_stats(self, elapsed_mean, elapsed_std, time_mean, time_std):
        """Set normalization stats from training set (for val/test)"""
        self.elapsed_mean = elapsed_mean
        self.elapsed_std = elapsed_std
        self.time_mean = time_mean
        self.time_std = time_std

    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, idx):
        q = self.X_question[idx].long()
        l = self.X_lecture[idx].long()
        c = self.X_concept[idx].long()

        # Only normalize if stats are available
        if self.elapsed_mean is not None:
            elapsed = (self.X_elapsed[idx].float() - self.elapsed_mean) / self.elapsed_std
            t = (self.X_time[idx].float() - self.time_mean) / self.time_std
        else:
            # Return raw values if stats not set yet
            elapsed = self.X_elapsed[idx].float()
            t = self.X_time[idx].float()

        correct = self.y_correct[idx].float()
        mask = self.mask[idx].float()

        return {
            "user_id": self.user_ids[idx],
            "question": q,
            "lecture": l,
            "concept": c,
            "elapsed": elapsed,
            "time": t,
            "correct": correct,
            "mask": mask,
        }


print("\n" + "=" * 60)
print("PREPARING DATA SPLITS")
print("=" * 60)

# Create full dataset
dataset_full = C3RecDataset(data, compute_stats=False)  # Don't compute stats yet

# Create subset if needed
subset_ratio = CONFIG["subset_ratio"]
if subset_ratio < 1.0:
    subset_size = max(1, int(len(dataset_full) * subset_ratio))
    indices = torch.randperm(len(dataset_full))[:subset_size]
    dataset = Subset(dataset_full, indices)
    print(f"⚠️  Using {subset_size}/{len(dataset_full)} samples (~{subset_ratio*100:.1f}%)")
else:
    dataset = dataset_full
    print(f"✓ Using full dataset: {len(dataset)} samples")

# Split into train/val
train_size = int(CONFIG["train_split"] * len(dataset))
val_size = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

print(f"✓ Train samples: {len(train_ds)}")
print(f"✓ Val samples: {len(val_ds)}")

# NOW compute normalization stats on TRAIN split only
print("\n" + "=" * 60)
print("COMPUTING NORMALIZATION STATISTICS (TRAIN ONLY)")
print("=" * 60)

# First, temporarily set stats to dummy values to allow data access
dataset_full.set_normalization_stats(
    torch.tensor(0.0), torch.tensor(1.0),
    torch.tensor(0.0), torch.tensor(1.0)
)

train_indices = train_ds.indices
train_elapsed_list = []
train_time_list = []

for idx in train_indices:
    # Access the underlying dataset
    if isinstance(dataset, Subset):
        actual_idx = dataset.indices[idx]
    else:
        actual_idx = idx
    
    # Get raw data directly (before normalization)
    mask = dataset_full.mask[actual_idx]
    valid = mask.bool()
    
    if valid.any():
        train_elapsed_list.append(dataset_full.X_elapsed[actual_idx][valid])
        train_time_list.append(dataset_full.X_time[actual_idx][valid])

train_elapsed_all = torch.cat(train_elapsed_list).float()
train_time_all = torch.cat(train_time_list).float()

elapsed_mean = train_elapsed_all.mean()
elapsed_std = train_elapsed_all.std().clamp_min(1e-6)
time_mean = train_time_all.mean()
time_std = train_time_all.std().clamp_min(1e-6)

# NOW set the correct stats
dataset_full.set_normalization_stats(elapsed_mean, elapsed_std, time_mean, time_std)

print(f"✓ Elapsed — mean: {elapsed_mean.item():.2f}, std: {elapsed_std.item():.2f}")
print(f"✓ Time — mean: {time_mean.item():.2f}, std: {time_std.item():.2f}")

# Create DataLoaders
train_loader = DataLoader(train_ds, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=0)
# =============================
# STEP 3 — C3Rec Model
# =============================

class C3RecModel(nn.Module):
    """
    C3Rec: Continuous Correctness-aware Course Recommendation
    
    Multi-task transformer model for knowledge tracing with:
    
    INPUTS (for each timestep t in prefix [0..T-2]):
        - question_id[t]: Discrete ID
        - concept_id[t]: Discrete ID  
        - lecture_id[t]: Discrete ID
        - elapsed_time[t]: Continuous, normalized
        - time_feature[t]: Continuous, normalized
        - correctness[t]: Binary (0/1), used as INPUT signal
    
    OUTPUTS (predicting t+1 from history up to t):
        - P(correct[t+1]): Binary classification
        - next_question[t+1]: Multi-class classification
        - next_lecture[t+1]: Multi-class classification
        - ability_score: Scalar per user (pooled representation)
    
    Key Design:
    - Non-leaky: Uses only h_t to predict t+1
    - Correctness as input: Past performance informs future predictions
    - Multi-task learning: Joint optimization of multiple objectives
    """
    
    def __init__(self,
                 num_questions,
                 num_concepts,
                 num_lectures,
                 d_model=128,
                 n_heads=4,
                 n_layers=2,
                 dropout=0.1,
                 seq_len=None):
        super().__init__()
        
        self.d_model = d_model
        self.seq_len = seq_len
        
        # Embeddings for discrete IDs (0 is padding)
        self.q_emb = nn.Embedding(num_questions, d_model, padding_idx=0)
        self.c_emb = nn.Embedding(num_concepts, d_model, padding_idx=0)
        self.l_emb = nn.Embedding(num_lectures, d_model, padding_idx=0)
        
        # Projections for continuous features
        self.elapsed_proj = nn.Linear(1, d_model)
        self.time_proj = nn.Linear(1, d_model)
        self.correct_proj = nn.Linear(1, d_model)
        
        # Positional encoding
        self.pos_emb = nn.Embedding(seq_len, d_model)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=False
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers)
        
        # Task-specific heads
        self.correct_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
        self.nextq_head = nn.Linear(d_model, num_questions)
        self.nextl_head = nn.Linear(d_model, num_lectures)
        
        # Ability prediction (from pooled sequence representation)
        self.ability_mlp = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier/Kaiming"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0, std=0.02)
                if m.padding_idx is not None:
                    m.weight.data[m.padding_idx].zero_()

    def forward(self, batch):
        """
        Forward pass with prefix-based prediction (no leakage).
        
        Given sequence [0, 1, 2, ..., T-1]:
        - Use prefix [0..T-2] as input
        - Predict targets at [1..T-1]
        
        Returns:
            dict with predictions and masks
        """
        # Full sequences
        q_full = batch["question"]  # (B, S)
        c_full = batch["concept"]
        l_full = batch["lecture"]
        e_full = batch["elapsed"]
        t_full = batch["time"]
        r_full = batch["correct"]
        m_full = batch["mask"]
        
        # Use prefix [0..S-2] to predict [1..S-1]
        q = q_full[:, :-1]  # (B, S-1)
        c = c_full[:, :-1]
        l = l_full[:, :-1]
        e = e_full[:, :-1]
        t = t_full[:, :-1]
        r = r_full[:, :-1]
        m = m_full[:, :-1]  # (B, S-1)
        
        B, S1 = q.shape
        
        # Embed all features
        q_emb = self.q_emb(q)  # (B, S-1, d)
        c_emb = self.c_emb(c)
        l_emb = self.l_emb(l)
        
        e_emb = self.elapsed_proj(e.unsqueeze(-1))  # (B, S-1, d)
        t_emb = self.time_proj(t.unsqueeze(-1))
        r_emb = self.correct_proj(r.unsqueeze(-1))
        
        # Combine embeddings (additive fusion)
        seq = q_emb + c_emb + l_emb + e_emb + t_emb + r_emb  # (B, S-1, d)
        
        # Add positional encoding
        if S1 > self.seq_len:
            raise ValueError(f"Sequence length {S1} exceeds model capacity {self.seq_len}")
        
        pos_idx = torch.arange(S1, device=seq.device).unsqueeze(0).expand(B, -1)  # (B, S-1)
        seq = seq + self.pos_emb(pos_idx)
        
        # Create padding mask: True where padding (mask==0)
        pad_mask = (m == 0)  # (B, S-1)
        
        # Encode sequence
        h = self.encoder(seq, src_key_padding_mask=pad_mask)  # (B, S-1, d)
        
        # Task predictions (each h_t predicts t+1)
        correct_next_logits = self.correct_head(h).squeeze(-1)  # (B, S-1)
        nextq_logits = self.nextq_head(h)  # (B, S-1, num_questions)
        nextl_logits = self.nextl_head(h)  # (B, S-1, num_lectures)
        
        # Pooled representation for ability
        mask_expanded = m.unsqueeze(-1)  # (B, S-1, 1)
        h_masked = h * mask_expanded
        seq_lengths = mask_expanded.sum(dim=1).clamp_min(1.0)  # (B, 1)
        pooled = h_masked.sum(dim=1) / seq_lengths  # (B, d)
        
        ability = self.ability_mlp(pooled).squeeze(-1)  # (B,)
        
        return {
            "correct_next_logits": correct_next_logits,
            "nextq_logits": nextq_logits,
            "nextl_logits": nextl_logits,
            "ability": ability,
            "mask_prefix": m,
            "hidden_states": h  # For potential analysis
        }

# =============================
# STEP 4 — Loss Functions
# =============================

def masked_bce_with_logits(logits, target, mask):
    """
    Masked binary cross-entropy loss.
    
    Args:
        logits: (B, S) raw logits
        target: (B, S) binary targets {0, 1}
        mask: (B, S) validity mask {0, 1}
    
    Returns:
        Scalar loss
    """
    loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    loss = loss * mask
    denom = mask.sum()
    if denom < 1:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    return loss.sum() / denom


def masked_cross_entropy(logits, targets, mask, ignore_index=0):
    """
    Masked cross-entropy for next-item prediction.
    
    Args:
        logits: (B, S, V) logits over vocabulary
        targets: (B, S) target IDs
        mask: (B, S) validity mask
        ignore_index: ID to ignore (padding, typically 0)
    
    Returns:
        Scalar loss
    """
    B, S, V = logits.shape
    
    logits_flat = logits.reshape(-1, V)  # (B*S, V)
    targets_flat = targets.reshape(-1)  # (B*S,)
    mask_flat = mask.reshape(-1)  # (B*S,)
    
    # Valid positions: mask=1 AND target != ignore_index
    valid = (mask_flat > 0) & (targets_flat != ignore_index)
    valid_idx = valid.nonzero(as_tuple=False).squeeze(-1)
    
    if valid_idx.numel() == 0:
        return torch.tensor(0.0, device=logits.device, requires_grad=True)
    
    logits_valid = logits_flat[valid_idx]
    targets_valid = targets_flat[valid_idx]
    
    loss = F.cross_entropy(logits_valid, targets_valid, reduction="mean")
    return loss


def compute_ability_target(correct_history, mask):
    """
    Compute ability target as mean correctness over valid history.
    
    Args:
        correct_history: (B, S) correctness values
        mask: (B, S) validity mask
    
    Returns:
        (B,) ability targets
    """
    masked_correct = correct_history * mask
    num_valid = mask.sum(dim=1).clamp_min(1.0)
    ability_target = masked_correct.sum(dim=1) / num_valid
    return ability_target

# =============================
# STEP 5 — Training Loop
# =============================

print("\n" + "=" * 60)
print("INITIALIZING MODEL")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"✓ Device: {device}")

model = C3RecModel(
    num_questions=len(data["question_vocab"]),
    num_concepts=len(data["concept_vocab"]),
    num_lectures=len(data["lecture_vocab"]),
    d_model=CONFIG["d_model"],
    n_heads=CONFIG["n_heads"],
    n_layers=CONFIG["n_layers"],
    dropout=CONFIG["dropout"],
    seq_len=SEQ_LEN
).to(device)

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"✓ Trainable parameters: {num_params:,}")

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=CONFIG["learning_rate"],
    weight_decay=1e-5
)

# Learning rate scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2,
    verbose=True
)

# Training tracking
train_losses = []
val_losses = []
best_val_loss = float('inf')
best_epoch = 0

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)
print(f"Epochs: {CONFIG['epochs']}")
print(f"Batch size: {CONFIG['batch_size']}")
print(f"Learning rate: {CONFIG['learning_rate']}")
print(f"Loss weights: correct={CONFIG['weight_correct']}, nextq={CONFIG['weight_nextq']}, "
      f"nextl={CONFIG['weight_nextl']}, ability={CONFIG['weight_ability']}")
print("=" * 60 + "\n")

global_start = time.time()

for epoch in range(1, CONFIG["epochs"] + 1):
    epoch_start = time.time()
    
    # ========== TRAINING ==========
    model.train()
    train_loss_total = 0.0
    train_loss_correct = 0.0
    train_loss_nextq = 0.0
    train_loss_nextl = 0.0
    train_loss_ability = 0.0
    train_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        for k in batch:
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device)
        
        # Forward pass
        out = model(batch)
        
        # Extract predictions
        correct_next_logits = out["correct_next_logits"]  # (B, S-1)
        nextq_logits = out["nextq_logits"]  # (B, S-1, num_questions)
        nextl_logits = out["nextl_logits"]  # (B, S-1, num_lectures)
        ability_pred = out["ability"]  # (B,)
        m_prefix = out["mask_prefix"]  # (B, S-1)
        
        # Extract targets (shift by 1)
        target_correct = batch["correct"][:, 1:]  # (B, S-1)
        target_nextq = batch["question"][:, 1:]  # (B, S-1)
        target_nextl = batch["lecture"][:, 1:]  # (B, S-1)
        
        # Compute losses
        loss_correct = masked_bce_with_logits(correct_next_logits, target_correct, m_prefix)
        loss_nextq = masked_cross_entropy(nextq_logits, target_nextq, m_prefix, ignore_index=0)
        loss_nextl = masked_cross_entropy(nextl_logits, target_nextl, m_prefix, ignore_index=0)
        
        # Ability loss (target = historical mean correctness)
        ability_target = compute_ability_target(batch["correct"][:, :-1], m_prefix)
        loss_ability = F.mse_loss(ability_pred, ability_target)
        
        # Combined loss
        loss = (
            CONFIG["weight_correct"] * loss_correct +
            CONFIG["weight_nextq"] * loss_nextq +
            CONFIG["weight_nextl"] * loss_nextl +
            CONFIG["weight_ability"] * loss_ability
        )
        
        # Check for NaN/Inf
        if not torch.isfinite(loss):
            print(f"⚠️  Warning: Non-finite loss at batch {batch_idx}, skipping...")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG["grad_clip"])
        optimizer.step()
        
        # Track losses
        train_loss_total += loss.item()
        train_loss_correct += loss_correct.item()
        train_loss_nextq += loss_nextq.item()
        train_loss_nextl += loss_nextl.item()
        train_loss_ability += loss_ability.item()
        train_batches += 1
    
    # Average training losses
    avg_train_loss = train_loss_total / max(1, train_batches)
    avg_train_correct = train_loss_correct / max(1, train_batches)
    avg_train_nextq = train_loss_nextq / max(1, train_batches)
    avg_train_nextl = train_loss_nextl / max(1, train_batches)
    avg_train_ability = train_loss_ability / max(1, train_batches)
    train_losses.append(avg_train_loss)
    
    # ========== VALIDATION ==========
    model.eval()
    val_loss_total = 0.0
    val_loss_correct = 0.0
    val_loss_nextq = 0.0
    val_loss_nextl = 0.0
    val_loss_ability = 0.0
    val_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            # Move to device
            for k in batch:
                if torch.is_tensor(batch[k]):
                    batch[k] = batch[k].to(device)
            
            # Forward pass
            out = model(batch)
            
            correct_next_logits = out["correct_next_logits"]
            nextq_logits = out["nextq_logits"]
            nextl_logits = out["nextl_logits"]
            ability_pred = out["ability"]
            m_prefix = out["mask_prefix"]
            
            # Targets
            target_correct = batch["correct"][:, 1:]
            target_nextq = batch["question"][:, 1:]
            target_nextl = batch["lecture"][:, 1:]
            
            # Losses
            loss_correct = masked_bce_with_logits(correct_next_logits, target_correct, m_prefix)
            loss_nextq = masked_cross_entropy(nextq_logits, target_nextq, m_prefix, ignore_index=0)
            loss_nextl = masked_cross_entropy(nextl_logits, target_nextl, m_prefix, ignore_index=0)
            
            ability_target = compute_ability_target(batch["correct"][:, :-1], m_prefix)
            loss_ability = F.mse_loss(ability_pred, ability_target)
            
            loss = (
                CONFIG["weight_correct"] * loss_correct +
                CONFIG["weight_nextq"] * loss_nextq +
                CONFIG["weight_nextl"] * loss_nextl +
                CONFIG["weight_ability"] * loss_ability
            )
            
            if not torch.isfinite(loss):
                print(f"⚠️  Warning: Non-finite validation loss, skipping batch...")
                continue
            
            val_loss_total += loss.item()
            val_loss_correct += loss_correct.item()
            val_loss_nextq += loss_nextq.item()
            val_loss_nextl += loss_nextl.item()
            val_loss_ability += loss_ability.item()
            val_batches += 1
    
    # Average validation losses
    avg_val_loss = val_loss_total / max(1, val_batches)
    avg_val_correct = val_loss_correct / max(1, val_batches)
    avg_val_nextq = val_loss_nextq / max(1, val_batches)
    avg_val_nextl = val_loss_nextl / max(1, val_batches)
    avg_val_ability = val_loss_ability / max(1, val_batches)
    val_losses.append(avg_val_loss)
    
    # Update learning rate
    scheduler.step(avg_val_loss)
    
    # Track best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_epoch = epoch
        if CONFIG["save_best_only"]:
            torch.save(model.state_dict(), f"c3rec_model_{CONFIG['model_version']}_best.pt")
    
    # Epoch summary
    epoch_time = time.time() - epoch_start
    print(f"Epoch {epoch}/{CONFIG['epochs']} ({epoch_time:.1f}s)")
    print(f"  Train: {avg_train_loss:.6f} [correct={avg_train_correct:.6f}, "
          f"nextq={avg_train_nextq:.6f}, nextl={avg_train_nextl:.6f}, ability={avg_train_ability:.6f}]")
    print(f"  Val:   {avg_val_loss:.6f} [correct={avg_val_correct:.6f}, "
          f"nextq={avg_val_nextq:.6f}, nextl={avg_val_nextl:.6f}, ability={avg_val_ability:.6f}]")
    if epoch == best_epoch:
        print(f"  ✓ New best validation loss!")
    print()

total_time = time.time() - global_start
print("=" * 60)
print(f"✓ Training completed in {total_time:.1f}s ({total_time/60:.1f}m)")
print(f"✓ Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
print("=" * 60)

# =============================
# STEP 6 — Evaluation: NDCG@K
# =============================
# =============================
# STEP 6 — Evaluation: NDCG@K
# =============================

print("\n" + "=" * 60)
print("EVALUATING: NDCG@K FOR NEXT-QUESTION PREDICTION")
print("=" * 60)

def compute_ndcg_at_k(logits, true_id, k=10):
    """
    Compute NDCG@K for a single prediction.
    
    Args:
        logits: (V,) logits over vocabulary
        true_id: scalar, true next item ID
        k: top-k cutoff
    
    Returns:
        NDCG score (0 to 1)
    """
    topk_ids = torch.topk(logits, min(k, logits.size(0))).indices.tolist()
    true_id = int(true_id)
    
    if true_id in topk_ids:
        rank = topk_ids.index(true_id)
        return 1.0 / torch.log2(torch.tensor(rank + 2.0, device=logits.device))
    return torch.tensor(0.0, device=logits.device)


model.eval()
ndcg_scores = []

with torch.no_grad():
    for batch in val_loader:
        for k in batch:
            if torch.is_tensor(batch[k]):
                batch[k] = batch[k].to(device)
        
        out = model(batch)
        nextq_logits = out["nextq_logits"]  # (B, S-1, num_questions)
        m_prefix = out["mask_prefix"]  # (B, S-1)
        q_full = batch["question"]  # (B, S)
        
        B, S1, num_questions = nextq_logits.shape
        
        for i in range(B):
            # Find last valid position in prefix
            seq_len = int(m_prefix[i].sum().item())
            
            if seq_len <= 0:
                continue
            
            last_idx = seq_len - 1
            
            # Check bounds
            if last_idx >= S1:
                continue
            
            # Get prediction at last position
            logits_last = nextq_logits[i, last_idx]  # (num_questions,)
            true_next = q_full[i, last_idx + 1]  # Next question ID
            
            # Skip padding
            if int(true_next.item()) == 0:
                continue
            
            # Compute NDCG
            score = compute_ndcg_at_k(logits_last, true_next, k=CONFIG["ndcg_k"])
            ndcg_scores.append(score.item())

ndcg_mean = sum(ndcg_scores) / max(1, len(ndcg_scores))
print(f"✓ NDCG@{CONFIG['ndcg_k']} = {ndcg_mean:.4f} ({len(ndcg_scores)} samples)")
print("=" * 60)

# =============================
# STEP 7 — Save Model & Metadata
# =============================

print("\n" + "=" * 60)
print("SAVING MODEL AND METADATA")
print("=" * 60)

# Prepare metadata
metadata = {
    # Vocabularies
    "question_vocab": data["question_vocab"],
    "lecture_vocab": data["lecture_vocab"],
    "concept_vocab": data["concept_vocab"],
    
    # Normalization stats (from TRAIN split)
    "elapsed_mean": elapsed_mean.item(),
    "elapsed_std": elapsed_std.item(),
    "time_mean": time_mean.item(),
    "time_std": time_std.item(),
    
    # Model configuration
    "seq_len": SEQ_LEN,
    "model_config": {
        "d_model": CONFIG["d_model"],
        "n_heads": CONFIG["n_heads"],
        "n_layers": CONFIG["n_layers"],
        "dropout": CONFIG["dropout"],
        "num_questions": len(data["question_vocab"]),
        "num_concepts": len(data["concept_vocab"]),
        "num_lectures": len(data["lecture_vocab"]),
    },
    
    # Training configuration
    "training_config": {
        "batch_size": CONFIG["batch_size"],
        "learning_rate": CONFIG["learning_rate"],
        "epochs": CONFIG["epochs"],
        "grad_clip": CONFIG["grad_clip"],
        "weight_correct": CONFIG["weight_correct"],
        "weight_nextq": CONFIG["weight_nextq"],
        "weight_nextl": CONFIG["weight_nextl"],
        "weight_ability": CONFIG["weight_ability"],
    },
    
    # Task heads
    "heads": {
        "next_correct": True,
        "next_question": True,
        "next_lecture": True,
        "ability": True,
    },
    
    # Training results
    "best_epoch": best_epoch,
    "best_val_loss": best_val_loss,
    "final_train_loss": train_losses[-1] if train_losses else None,
    "final_val_loss": val_losses[-1] if val_losses else None,
    "ndcg_at_k": {
        "k": CONFIG["ndcg_k"],
        "score": ndcg_mean,
        "num_samples": len(ndcg_scores)
    },
    
    # Version and timestamp
    "version": CONFIG["model_version"],
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
}

# Save metadata
metadata_path = f"c3rec_metadata_{CONFIG['model_version']}.pt"
torch.save(metadata, metadata_path)
print(f"✓ Metadata saved to: {metadata_path}")

# Save final model (if not using best-only)
if not CONFIG["save_best_only"]:
    model_path = f"c3rec_model_{CONFIG['model_version']}.pt"
    torch.save(model.state_dict(), model_path)
    print(f"✓ Final model saved to: {model_path}")
else:
    print(f"✓ Best model saved to: c3rec_model_{CONFIG['model_version']}_best.pt")

print("=" * 60)

# =============================
# STEP 8 — Visualize Training
# =============================

print("\n" + "=" * 60)
print("PLOTTING TRAINING CURVES")
print("=" * 60)

plt.figure(figsize=(10, 6))
epochs_range = range(1, len(train_losses) + 1)

plt.plot(epochs_range, train_losses, 'b-o', label='Train Loss', linewidth=2, markersize=6)
plt.plot(epochs_range, val_losses, 'r-s', label='Val Loss', linewidth=2, markersize=6)

# Mark best epoch
if best_epoch > 0:
    plt.axvline(x=best_epoch, color='g', linestyle='--', linewidth=1.5, 
                label=f'Best Epoch ({best_epoch})')

plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss', fontsize=12)
plt.title('C3Rec Training Curves (Multi-Task Learning)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()

plot_path = f"c3rec_training_{CONFIG['model_version']}.png"
plt.savefig(plot_path, dpi=150)
print(f"✓ Training plot saved to: {plot_path}")
plt.show()

print("=" * 60)
print("✅ ALL DONE!")
print("=" * 60)