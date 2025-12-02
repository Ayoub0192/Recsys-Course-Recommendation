from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import FastAPI
from pydantic import BaseModel

import torch

from model.load_model import load_c3rec_model
from utils.inference import (
    tensorize_user_history,
    recommend_next_question,
    recommend_lessons,
    predict_ability,
    generate_course_path,
    get_mastery_graph
)

###############################################
# INIT FASTAPI
###############################################
app = FastAPI(title="C3Rec API", version="1.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

###############################################
# LOAD MODEL + METADATA + VOCAB
###############################################
print("🚀 Initializing C3Rec model...")
device = "cuda" if torch.cuda.is_available() else "cpu"

model, vocab, metadata = load_c3rec_model(device=device)

# Attach metadata inside model (used by course_path)
model.metadata = metadata

print("✅ Model loaded!")


###############################################
# REQUEST BODY FORMAT
###############################################
class HistoryRequest(BaseModel):
    user_id: int
    question: list[int]
    concept: list[int]
    lecture: list[int]
    elapsed: list[float]
    time: list[float]
    correct: list[float]
    topk: int = 5
    steps: int = 10


###############################################
# ROUTES
###############################################
@app.get("/")
def home():
    return {"status": "C3Rec API running", "version": "1.0"}


@app.post("/recommend_questions")
def api_recommend_questions(req: HistoryRequest):
    user = tensorize_user_history(req.dict(), metadata, device)
    rec = recommend_next_question(model, vocab, user, req.topk)
    return {"user_id": req.user_id, "recommended_questions": rec}


@app.post("/recommend_lessons")
def api_recommend_lessons(req: HistoryRequest):
    user = tensorize_user_history(req.dict(), metadata, device)
    rec = recommend_lessons(model, vocab, user, req.topk)
    return {"user_id": req.user_id, "recommended_lessons": rec}


@app.post("/course_path")
def api_course_path(req: HistoryRequest):
    user = tensorize_user_history(req.dict(), metadata, device)
    path = generate_course_path(model, vocab, metadata, user, req.steps, 1, 8.0)
    return {"user_id": req.user_id, "course_path": path}


@app.post("/mastery_graph")
def api_mastery_graph(req: HistoryRequest):
    user = tensorize_user_history(req.dict(), metadata, device)
    mg = get_mastery_graph(model, vocab, user)
    return {"user_id": req.user_id, "mastery_graph": mg}
