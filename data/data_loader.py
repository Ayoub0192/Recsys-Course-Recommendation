
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder

MAX_SEQ_LEN = 100  # maximum history length used for each training sample


class EdNetPreprocessor:
    """Loads and encodes a cleaned EdNet-like CSV.

    Expected columns in the CSV:
    - user_id
    - assessmentItemID
    - KnowledgeTag
    - answerCode
    - Timestamp
    """

    def __init__(self, csv_path: str = "data/ednet_clean.csv"):
        self.csv_path = csv_path
        self.user_encoder = LabelEncoder()
        self.item_encoder = LabelEncoder()
        self.concept_encoder = LabelEncoder()

    def load_and_encode(self) -> pd.DataFrame:
        df = pd.read_csv(self.csv_path)

        required = ["user_id", "assessmentItemID", "KnowledgeTag", "answerCode", "Timestamp"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in CSV: {missing}")

        # order interactions
        df = df.sort_values(["user_id", "Timestamp"])

        df["user_id"] = df["user_id"].astype(str)
        df["assessmentItemID"] = df["assessmentItemID"].astype(str)
        df["KnowledgeTag"] = df["KnowledgeTag"].astype(str)

        df["user_idx"] = self.user_encoder.fit_transform(df["user_id"])
        df["item_idx"] = self.item_encoder.fit_transform(df["assessmentItemID"])
        df["concept_idx"] = self.concept_encoder.fit_transform(df["KnowledgeTag"])

        return df

    def get_num_entities(self, df: pd.DataFrame):
        num_users = int(df["user_idx"].max()) + 1
        num_items = int(df["item_idx"].max()) + 1
        num_concepts = int(df["concept_idx"].max()) + 1
        return num_users, num_items, num_concepts


class EdNetSequenceDataset(Dataset):
    """PyTorch Dataset for next-item prediction using EdNet.

    Each sample:
    - seq_items:   (T,) previous item indices
    - seq_concepts:(T,) previous concept indices
    - seq_answers: (T,) previous correctness (0/1)
    - target_item: scalar item index to predict
    - user_ids:    scalar user index
    """

    def __init__(self, df: pd.DataFrame, max_seq_len: int = MAX_SEQ_LEN):
        self.max_seq_len = max_seq_len
        self.samples = []

        for user_idx, group in df.groupby("user_idx"):
            items = group["item_idx"].tolist()
            concepts = group["concept_idx"].tolist()
            answers = group["answerCode"].astype(float).tolist()

            if len(items) < 2:
                continue

            for t in range(1, len(items)):
                start = max(0, t - max_seq_len)
                seq_items = items[start:t]
                seq_concepts = concepts[start:t]
                seq_answers = answers[start:t]
                target_item = items[t]

                self.samples.append(
                    (
                        np.array(seq_items, dtype=np.int64),
                        np.array(seq_concepts, dtype=np.int64),
                        np.array(seq_answers, dtype=np.float32),
                        np.int64(target_item),
                        int(user_idx),
                    )
                )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        seq_items, seq_concepts, seq_answers, target_item, user_id = self.samples[idx]
        return {
            "user_ids": user_id,
            "seq_items": seq_items,
            "seq_concepts": seq_concepts,
            "seq_answers": seq_answers,
            "target_item": target_item,
        }


def collate_fn(batch, pad_idx: int = 0):
    """Pads sequences in a batch to the same length."""
    import torch

    max_len = max(len(x["seq_items"]) for x in batch)

    seq_items_batch = []
    seq_concepts_batch = []
    seq_answers_batch = []
    targets = []
    user_ids = []

    for x in batch:
        L = len(x["seq_items"])
        pad_len = max_len - L

        seq_items = np.pad(x["seq_items"], (pad_len, 0), constant_values=pad_idx)
        seq_concepts = np.pad(x["seq_concepts"], (pad_len, 0), constant_values=pad_idx)
        seq_answers = np.pad(x["seq_answers"], (pad_len, 0), constant_values=0.0)

        seq_items_batch.append(seq_items)
        seq_concepts_batch.append(seq_concepts)
        seq_answers_batch.append(seq_answers)
        targets.append(x["target_item"])
        user_ids.append(x["user_ids"])

    return {
        "user_ids": torch.tensor(user_ids, dtype=torch.long),
        "seq_items": torch.tensor(seq_items_batch, dtype=torch.long),
        "seq_concepts": torch.tensor(seq_concepts_batch, dtype=torch.long),
        "seq_answers": torch.tensor(seq_answers_batch, dtype=torch.float32),
        "target_item": torch.tensor(targets, dtype=torch.long),
    }
