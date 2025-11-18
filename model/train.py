
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter

from data_loader import EdNetPreprocessor, EdNetSequenceDataset, collate_fn
from model_c3rec import C3RecModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 64
LR = 1e-3
EPOCHS = 5
MAX_SEQ_LEN = 100


def ndcg_at_k(preds, truth, k=10):
    import numpy as np
    ndcgs = []
    for p, t in zip(preds, truth):
        topk = p[:k]
        if t in topk:
            rank = topk.index(t)
            ndcg = 1.0 / np.log2(rank + 2)
        else:
            ndcg = 0.0
        ndcgs.append(ndcg)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def recall_at_k(preds, truth, k=20):
    recalls = []
    for p, t in zip(preds, truth):
        topk = p[:k]
        recalls.append(1.0 if t in topk else 0.0)
    return float(sum(recalls) / len(recalls)) if recalls else 0.0


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for batch in loader:
        user_ids = batch["user_ids"].to(DEVICE)
        seq_items = batch["seq_items"].to(DEVICE)
        seq_concepts = batch["seq_concepts"].to(DEVICE)
        seq_answers = batch["seq_answers"].to(DEVICE)
        targets = batch["target_target_item"] if "target_target_item" in batch else batch["target_item"]
        targets = targets.to(DEVICE)

        optimizer.zero_grad()
        logits, mastery = model(user_ids, seq_items, seq_concepts, seq_answers)
        loss = criterion(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def validate(model, loader, k_ndcg=10, k_recall=20):
    model.eval()
    all_preds, all_truth = [], []

    with torch.no_grad():
        for batch in loader:
            user_ids = batch["user_ids"].to(DEVICE)
            seq_items = batch["seq_items"].to(DEVICE)
            seq_concepts = batch["seq_concepts"].to(DEVICE)
            seq_answers = batch["seq_answers"].to(DEVICE)
            targets = batch["target_item"].to(DEVICE)

            topk_idx, _, _ = model.predict_topk(
                user_ids, seq_items, seq_concepts, seq_answers, k=k_recall
            )

            preds = topk_idx.cpu().tolist()
            truth = targets.cpu().tolist()
            all_preds.extend(preds)
            all_truth.extend(truth)

    val_ndcg = ndcg_at_k(all_preds, all_truth, k=k_ndcg)
    val_recall = recall_at_k(all_preds, all_truth, k=k_recall)
    return val_ndcg, val_recall


def main():
    writer = SummaryWriter(log_dir="logs")

    preproc = EdNetPreprocessor()
    df = preproc.load_and_encode()
    num_users, num_items, num_concepts = preproc.get_num_entities(df)

    dataset = EdNetSequenceDataset(df, max_seq_len=MAX_SEQ_LEN)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    model = C3RecModel(
        num_users=num_users,
        num_items=num_items,
        num_concepts=num_concepts,
        embed_dim=64,
        hidden_dim=128,
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_recall = 0.0
    ckpt_path = os.path.join(os.path.dirname(__file__), "model_best.pt")

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        ndcg, recall = validate(model, val_loader)

        writer.add_scalar("loss/train", train_loss, epoch)
        writer.add_scalar("metrics/ndcg@10", ndcg, epoch)
        writer.add_scalar("metrics/recall@20", recall, epoch)

        print(f"Epoch {epoch}: loss={train_loss:.4f}, NDCG@10={ndcg:.4f}, Recall@20={recall:.4f}")

        if recall > best_recall:
            best_recall = recall
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "num_users": num_users,
                    "num_items": num_items,
                    "num_concepts": num_concepts,
                },
                ckpt_path,
            )
            print(f"Saved new best model to {ckpt_path} (Recall@20={best_recall:.4f})")

    writer.close()


if __name__ == "__main__":
    main()
