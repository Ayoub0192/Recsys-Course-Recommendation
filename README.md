
# LearnWise AI (Cohesive Version)

This repository contains a cohesive end-to-end MVP for an adaptive course recommendation system
built on a simplified EdNet-like dataset.

## Project Overview

- **Goal**: Recommend the next best lesson (assessment item) for a given learner.
- **Model**: C3Rec-inspired sequence model with:
  - Item and concept embeddings
  - GRU sequence encoder
  - Cognitive mastery head
  - Distillation layer
  - Item scoring head (next-item prediction)
- **Stack**:
  - Python, PyTorch
  - FastAPI backend
  - Streamlit frontend

## Expected Data

Place a CSV at:
\`\`\`
data/ednet_clean.csv
\`\`\`

With at least the following columns:

- \`user_id\`
- \`assessmentItemID\`
- \`KnowledgeTag\`
- \`answerCode\` (0 or 1)
- \`Timestamp\`

## Installation

1. Create a virtual environment and install dependencies:
   \`\`\`bash
   pip install torch sklearn fastapi uvicorn streamlit pandas numpy
   \`\`\`

2. Ensure your data file exists at \`data/ednet_clean.csv\`.

## Training

Run:
\`\`\`bash
python train.py
\`\`\`

This will:
- Load and encode the dataset
- Train the C3Rec model
- Save the best checkpoint to \`model_best.pt\` in the repo root

## API

Start the FastAPI server:
\`\`\`bash
uvicorn api_main:app --reload
\`\`\`

Health check:
\`\`\`bash
curl http://localhost:8000/health
\`\`\`

Get recommendations:
\`\`\`bash
curl "http://localhost:8000/recommend_next_lesson?user_id=u0001&k=5"
\`\`\`

## Frontend (Streamlit)

In another terminal:
\`\`\`bash
streamlit run ui_app.py
\`\`\`

Then open the URL shown in the console (usually \`http://localhost:8501\`).

## Notes

- This is an MVP skeleton: you can extend it with PostgreSQL logging, richer metadata for lessons,
  improved model architectures, and better evaluation.
- Make sure to adapt \`data/ednet_clean.csv\` preprocessing to your real EdNet KT1 format if needed.
