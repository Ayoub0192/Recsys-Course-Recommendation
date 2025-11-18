
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from predict import recommend_for_user

app = FastAPI(title="LearnWise AI API")


class InteractionLog(BaseModel):
    user_id: str
    lesson_id: str
    correct: bool
    timestamp: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/recommend_next_lesson")
def recommend_next_lesson(user_id: str, k: int = 5):
    try:
        recs = recommend_for_user(user_id=user_id, k=k)
        return {"user_id": user_id, "recommendations": recs}
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/log_interaction")
def log_interaction(log: InteractionLog):
    # Stub for now – you can extend this to write to PostgreSQL.
    return {"status": "logged", "data": log.dict()}
