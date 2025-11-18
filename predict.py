
import os
import numpy as np
import torch

from data_loader import EdNetPreprocessor
from model_c3rec import C3RecModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_SEQ_LEN = 100


def load_model_and_preproc(csv_path: str = "data/ednet_clean.csv"):
    """Load trained model checkpoint and a fresh preprocessor fit on the same CSV."""
    ckpt_path = os.path.join(os.path.dirname(__file__), "model_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at {ckpt_path}. Train the model first.")

    preproc = EdNetPreprocessor(csv_path=csv_path)
    df = preproc.load_and_encode()
    num_users, num_items, num_concepts = preproc.get_num_entities(df)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model = C3RecModel(
        num_users=ckpt["num_users"],
        num_items=ckpt["num_items"],
        num_concepts=ckpt["num_concepts"],
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, preproc, df


def recommend_for_user(user_id: str, k: int = 5, csv_path: str = "data/ednet_clean.csv"):
    """Generate top-k item recommendations for a given user_id."""
    model, preproc, df = load_model_and_preproc(csv_path=csv_path)

    if user_id not in df["user_id"].unique():
        raise ValueError(f"user_id '{user_id}' not found in dataset.")

    user_df = df[df["user_id"] == user_id].sort_values("Timestamp").tail(MAX_SEQ_LEN)
    if len(user_df) == 0:
        raise ValueError(f"No interactions for user_id '{user_id}'.")

    user_idx = preproc.user_encoder.transform([user_id])[0]
    seq_items = user_df["item_idx"].values
    seq_concepts = user_df["concept_idx"].values
    seq_answers = user_df["answerCode"].astype(float).values

    pad_len = MAX_SEQ_LEN - len(seq_items)
    if pad_len > 0:
        seq_items = np.pad(seq_items, (pad_len, 0), constant_values=0)
        seq_concepts = np.pad(seq_concepts, (pad_len, 0), constant_values=0)
        seq_answers = np.pad(seq_answers, (pad_len, 0), constant_values=0.0)

    user_ids = torch.tensor([user_idx], dtype=torch.long, device=DEVICE)
    seq_items_t = torch.tensor([seq_items], dtype=torch.long, device=DEVICE)
    seq_concepts_t = torch.tensor([seq_concepts], dtype=torch.long, device=DEVICE)
    seq_answers_t = torch.tensor([seq_answers], dtype=torch.float32, device=DEVICE)

    topk_idx, topk_scores, _ = model.predict_topk(
        user_ids, seq_items_t, seq_concepts_t, seq_answers_t, k=k
    )

    item_indices = topk_idx[0].cpu().tolist()
    scores = topk_scores[0].cpu().tolist()

    # map indices back to real assessmentItemID
    item_ids = preproc.item_encoder.inverse_transform(item_indices)
    return [
        {"item_idx": int(idx), "assessmentItemID": str(item_id), "score": float(score)}
        for idx, item_id, score in zip(item_indices, item_ids, scores)
    ]


if __name__ == "__main__":
    # quick manual test (requires you to set a valid user id)
    test_user = "u0001"
    try:
        recs = recommend_for_user(test_user, k=5)
        from pprint import pprint
        pprint(recs)
    except Exception as e:
        print("Error:", e)
