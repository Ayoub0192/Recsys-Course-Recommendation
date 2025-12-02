import torch
from typing import Dict, List, Tuple
from model.architecture import C3RecModel   # type: ignore

def tensorize_user_history(
    req: dict,
    metadata: dict,
    device: str = "cpu",
) -> dict:
    """
    Convert user history JSON (frontend payload) into tensors compatible with v10.
    Expected req:
      {
        "user_id": str,
        "question": [int],
        "concept": [int],
        "lecture": [int],
        "elapsed": [float],
        "time": [float],p
        "correct": [0/1 float or int],
      }
    """

    q = torch.tensor(req["question"], dtype=torch.long, device=device)
    c = torch.tensor(req["concept"], dtype=torch.long, device=device)
    l = torch.tensor(req["lecture"], dtype=torch.long, device=device)

    elapsed = torch.tensor(req["elapsed"], dtype=torch.float, device=device)
    time = torch.tensor(req["time"], dtype=torch.float, device=device)
    correct = torch.tensor(req["correct"], dtype=torch.float, device=device)

    mask = torch.ones_like(q, dtype=torch.float, device=device)

    # Training stats for normalization
    elapsed_mean = float(metadata["elapsed_mean"])
    elapsed_std = float(metadata["elapsed_std"])
    time_mean = float(metadata["time_mean"])
    time_std = float(metadata["time_std"])

    elapsed = (elapsed - elapsed_mean) / max(elapsed_std, 1e-8)
    time = (time - time_mean) / max(time_std, 1e-8)

    return {
        "user_id": req["user_id"],
        "question": q,
        "concept": c,
        "lecture": l,
        "elapsed": elapsed,
        "time": time,
        "correct": correct,
        "mask": mask,
        "history_length": len(q),
    }


# ============================================================================
# 📌 4. VOCAB HELPERS
# ============================================================================
def vocab_inv(v: dict) -> dict:
    """
    Invert a vocab mapping: token -> ID  ⟶  ID -> token
    Example:
      {"Q_1": 1, "Q_2": 2} ⟶ {1: "Q_1", 2: "Q_2"}
    """
    return {idx: key for key, idx in v.items()}


# ============================================================================
# 📌 5. PREDICT NEXT STEP (correctness + ranked items)
# ============================================================================
def _temperature_scaled_sigmoid(logit: float, temperature: float = 8.0) -> float:
    """Apply temperature scaling to a scalar logit then sigmoid + clamp."""
    scaled_logit = logit / max(temperature, 1e-6)
    prob = torch.sigmoid(torch.tensor(scaled_logit)).item()
    return float(max(1e-6, min(0.999999, prob)))


def predict_next_step(
    model: C3RecModel,
    user: dict,
    top_k: int = 10,
    temperature: float = 8.0,
) -> dict:
    """
    High-level prediction:
      - next_correct_prob
      - top-k next questions (IDs + scores)
      - top-k next lectures (IDs + scores)
      - ability score

    Returns a dict similar to your original _predict_single output.
    """
    device = next(model.parameters()).device

    batch = {
        k: v.unsqueeze(0).to(device)
        for k, v in user.items()
        if isinstance(v, torch.Tensor)
    }

    with torch.no_grad():
        out = model(batch)

    # Sequence length in model output (S-1)
    logits_correct = out["correct_next_logits"][0]        # [T]
    T = logits_correct.size(0)
    last_idx = T - 1

    raw_logit = logits_correct[last_idx].item()
    next_correct_prob = _temperature_scaled_sigmoid(raw_logit, temperature)

    nextq_logits = out["nextq_logits"][0, last_idx]       # [num_questions]
    nextl_logits = out["nextl_logits"][0, last_idx]       # [num_lectures]

    topq = torch.topk(nextq_logits, k=top_k)
    topl = torch.topk(nextl_logits, k=top_k)

    return {
        "user_id": user["user_id"],
        "history_length": user["history_length"],
        "ability_score": float(out["ability"][0].item()),
        "next_correct_prob": next_correct_prob,
        "top_questions": {
            "ids": topq.indices.cpu().tolist(),
            "scores": topq.values.cpu().tolist(),
        },
        "top_lectures": {
            "ids": topl.indices.cpu().tolist(),
            "scores": topl.values.cpu().tolist(),
        },
    }


# ============================================================================
# 📌 6. RECOMMENDATIONS (semantic IDs using vocab)
# ============================================================================
def recommend_next_question(
    model: C3RecModel,
    vocab: dict,
    user: dict,
    top_k: int = 5,
) -> List[str]:
    """
    Return top-k *external question tokens* (not raw IDs),
    using vocab["question"] mapping token -> ID.
    """
    device = next(model.parameters()).device
    batch = {
        k: v.unsqueeze(0).to(device)
        for k, v in user.items()
        if isinstance(v, torch.Tensor)
    }

    with torch.no_grad():
        out = model(batch)

    last_idx = out["nextq_logits"].shape[1] - 1
    logits = out["nextq_logits"][0, last_idx]

    top = torch.topk(logits, k=top_k)
    inv_q = vocab_inv(vocab["question"])

    return [inv_q[int(idx)] for idx in top.indices.tolist()]


def recommend_lessons(
    model: C3RecModel,
    vocab: dict,
    user: dict,
    top_k: int = 5,
) -> List[str]:
    """
    Return top-k *external lecture tokens* (not raw IDs),
    using vocab["lecture"] mapping token -> ID.
    """
    device = next(model.parameters()).device
    batch = {
        k: v.unsqueeze(0).to(device)
        for k, v in user.items()
        if isinstance(v, torch.Tensor)
    }

    with torch.no_grad():
        out = model(batch)

    last_idx = out["nextl_logits"].shape[1] - 1
    logits = out["nextl_logits"][0, last_idx]

    top = torch.topk(logits, k=top_k)
    inv_l = vocab_inv(vocab["lecture"])

    return [inv_l[int(idx)] for idx in top.indices.tolist()]


# ============================================================================
# 📌 7. ABILITY SCORE
# ============================================================================
def predict_ability(
    model: C3RecModel,
    user: dict,
) -> float:
    device = next(model.parameters()).device
    batch = {
        k: v.unsqueeze(0).to(device)
        for k, v in user.items()
        if isinstance(v, torch.Tensor)
    }

    with torch.no_grad():
        out = model(batch)

    return float(out["ability"][0].item())


# ============================================================================
# 📌 8. REAL CONCEPT MASTERY GRAPH
# ============================================================================
def get_mastery_graph(
    model: C3RecModel,
    vocab: dict,
    user: dict,
) -> List[dict]:
    """
    Compute a concept mastery score in [0,1] for each concept, based on the model.

    For each concept c:
      - Align concept sequence with model's hidden states (length S-1).
      - For each timestep t where concept[t] == c and mask[t] == 1:
            use h_t and the model's correctness head to get a correctness probability.
      - Mastery(c) = mean(probabilities over all those timesteps).
      - If a concept is never seen in the prefix => mastery 0.0.

    Returns: list of dicts
      [
        { "concept": <external_concept_token>, "mastery": float, "count": int },
        ...
      ]
    """
    device = next(model.parameters()).device
    batch = {
        k: v.unsqueeze(0).to(device)
        for k, v in user.items()
        if isinstance(v, torch.Tensor)
    }

    with torch.no_grad():
        out = model(batch)

    # hidden states: [1, T, d]  -> [T, d]
    h = out["hidden_states"][0]                  # [T, d_model]
    T = h.size(0)

    # Align concepts + mask to T (which is S_full-1)
    concepts_full = user["concept"].to(device)   # [S_full]
    mask_full = user["mask"].to(device)          # [S_full]

    concepts = concepts_full[:T]                 # [T]
    mask = mask_full[:T]                         # [T]

    inv_c = vocab_inv(vocab["concept"])

    mastery_list = []

    for cid, cname in inv_c.items():
        # positions where this concept is active and not masked
        idx = ((concepts == cid) & (mask > 0.5)).nonzero(as_tuple=True)[0]

        if idx.numel() == 0:
            mastery_list.append({
                "concept": cname,
                "mastery": 0.0,
                "count": 0,
            })
            continue

        # hidden states for that concept
        h_c = h[idx]                             # [Nc, d]
        # use model's correctness head directly
        logits_c = model.correct_head(h_c).squeeze(-1)   # [Nc]
        probs_c = torch.sigmoid(logits_c)                # [Nc]

        score = float(probs_c.mean().item())

        mastery_list.append({
            "concept": cname,
            "mastery": score,
            "count": int(idx.numel()),
        })

    # optionally sort by mastery descending
    mastery_list.sort(key=lambda x: x["mastery"], reverse=True)
    return mastery_list


# ============================================================================
# 📌 9. AUTOREGRESSIVE COURSE PATH GENERATION (v10-consistent)
# ============================================================================
def generate_course_path(
    model: C3RecModel,
    vocab: dict,
    metadata: dict,
    user: dict,
    steps: int = 10,
    top_k: int = 1,
    temperature_correct: float = 8.0,
) -> List[dict]:
    """
    Autoregressively generate a personalized course path using the v10 logits.

    - At each step, we:
        1) Run the model on the current (normalized) history.
        2) Take the last index predictions (nextq_logits, nextl_logits, correct_next_logits).
        3) Pick top-1 question + lecture (or top_k if you want sampling/variety).
        4) Use the model's correctness logit (with temperature scaling) as expected correctness.
        5) Append this "simulated" interaction to the history and repeat.

    NOTE: This uses the same off-by-one convention as training/inference
          (model uses prefix to predict next step).
    """
    device = next(model.parameters()).device

    # clone history so we can mutate it in-place safely
    u = {k: (v.clone().to(device) if isinstance(v, torch.Tensor) else v)
         for k, v in user.items()}

    inv_q = vocab_inv(vocab["question"])
    inv_l = vocab_inv(vocab["lecture"])
    inv_c = vocab_inv(vocab["concept"])

    elapsed_mean = float(metadata["elapsed_mean"])
    elapsed_std = float(metadata["elapsed_std"])
    time_mean = float(metadata["time_mean"])
    time_std = float(metadata["time_std"])

    def normalize(e: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        e = (e - elapsed_mean) / max(elapsed_std, 1e-8)
        t = (t - time_mean) / max(time_std, 1e-8)
        return e, t

    path = []

    for step in range(steps):
        # Build batch
        batch = {
            k: v.unsqueeze(0).to(device)
            for k, v in u.items()
            if isinstance(v, torch.Tensor)
        }

        with torch.no_grad():
            out = model(batch)

        T = out["nextq_logits"].shape[1]
        last_idx = T - 1

        # ---- Predict correctness probability for "next" item ----
        raw = out["correct_next_logits"][0, last_idx].item()
        prob = _temperature_scaled_sigmoid(raw, temperature_correct)

        # ---- Predict next question + lecture IDs ----
        topq = torch.topk(out["nextq_logits"][0, last_idx], k=top_k)
        topl = torch.topk(out["nextl_logits"][0, last_idx], k=top_k)

        next_q_idx = int(topq.indices[0].item())
        next_l_idx = int(topl.indices[0].item())
        # Simple heuristic: keep current concept as the last seen one
        next_c_idx = int(u["concept"][-1].item())

        # Append step info (external tokens)
        path.append({
            "step": step + 1,
            "question_id": next_q_idx,
            "question": inv_q.get(next_q_idx, next_q_idx),
            "lecture_id": next_l_idx,
            "lecture": inv_l.get(next_l_idx, next_l_idx),
            "concept_id": next_c_idx,
            "concept": inv_c.get(next_c_idx, next_c_idx),
            "predicted_correct_probability": prob,
        })

        # ---- Autoregressive history update ----
        # Here we simulate the student doing this item with expected correctness = prob.
        # For elapsed/time, we use training means and a simple time drift heuristic.

        # raw (unnormalized) elapsed/time for the new point
        e_raw = torch.tensor([elapsed_mean], dtype=torch.float, device=device)
        t_raw = torch.tensor(
            [time_mean + (u["history_length"] + step) * 10000.0],
            dtype=torch.float,
            device=device,
        )

        e_norm, t_norm = normalize(e_raw, t_raw)

        u["question"] = torch.cat([u["question"], torch.tensor([next_q_idx], device=device)])
        u["concept"] = torch.cat([u["concept"], torch.tensor([next_c_idx], device=device)])
        u["lecture"] = torch.cat([u["lecture"], torch.tensor([next_l_idx], device=device)])
        u["elapsed"] = torch.cat([u["elapsed"], e_norm])
        u["time"] = torch.cat([u["time"], t_norm])
        u["correct"] = torch.cat([u["correct"], torch.tensor([prob], device=device)])
        u["mask"] = torch.cat([u["mask"], torch.tensor([1.0], device=device)])

        u["history_length"] += 1

    return path