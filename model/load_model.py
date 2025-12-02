import os
import torch
from typing import Tuple
from .architecture import C3RecModel


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "c3rec_model_v10_best.pt")
DEFAULT_META_PATH  = os.path.join(BASE_DIR, "c3rec_metadata_v10.pt")


def load_c3rec_model(
    model_path: str = DEFAULT_MODEL_PATH,
    metadata_path: str = DEFAULT_META_PATH,
    device: str = "cpu",
):
    """
    Load C3Rec v10 model + metadata + vocab.
    Returns: model, vocab, metadata
    """

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found: {model_path}")

    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"❌ Metadata file not found: {metadata_path}")

    print("📥 Loading metadata...")
    metadata = torch.load(metadata_path, map_location=device)

    config = metadata["model_config"]

    print("📥 Rebuilding C3RecModel v10 architecture...")
    model = C3RecModel(
        num_questions=config["num_questions"],
        num_concepts=config["num_concepts"],
        num_lectures=config["num_lectures"],
        d_model=config["d_model"],
        n_heads=config["n_heads"],
        n_layers=config["n_layers"],
        dropout=config["dropout"],
        seq_len=metadata["seq_len"],
    ).to(device)

    print("📥 Loading model weights...")
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    print("✅ C3Rec v10 model loaded successfully!")

    vocab = {
        "question": metadata["question_vocab"],
        "concept":  metadata["concept_vocab"],
        "lecture":  metadata["lecture_vocab"],
    }

    return model, vocab, metadata
