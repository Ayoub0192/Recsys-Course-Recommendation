 C3Rec — AI-Powered Learning Recommendation System

    C3Rec is a full end-to-end intelligent learning recommendation platform combining:

    A FastAPI backend for model inference

    A Transformer-based recommendation model (C3Rec v10, PyTorch)

    A modern interactive dashboard built with Next.js and TailwindCSS

    Full inference utilities (recommendations, mastery, learning paths)

    A complete training pipeline and testing suite

This repository contains the entire system: model, backend, frontend, utilities, and training tools.


Project Structure:
c3rec_project/
├── api/                     # FastAPI backend
│   └── main.py
│
├── model/                   # Model architecture, weights, metadata
│   ├── architecture.py
│   ├── load_model.py
│   ├── c3rec_model_v10_best.pt
│   ├── c3rec_metadata_v10.pt
│   └── c3rec_dataset_seq.pt
│
├── utils/
│   └── inference.py         # Core inference logic
│
├── training/
│   ├── c3rec_training_v2.py # Training pipeline
│   └── tester.py            # Synthetic profile tester
│
├── c3rec_frontend/          # Next.js dashboard
│   ├── app/
│   ├── public/
│   ├── tailwind.config.ts
│   └── README.md
│
├── requirements.txt         # Backend Python dependencies
├── C3recdata.parquet        # Training dataset
└── README.md                # Project documentation


Backend Overview (FastAPI)

The backend loads the C3Rec v10 model and exposes several endpoints used by the frontend.

Start the backend

source venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload


Endpoints:

    GET / — API health

    POST /recommend_questions — Top-K question recommendations

    POST /recommend_lessons — Top-K lesson recommendations

    POST /course_path — Auto-generated learning trajectory

    POST /mastery_graph — Concept mastery estimation


Swagger documentation is available at:

http://127.0.0.1:8000/docs



Frontend Overview (Next.js + Tailwind)

A full interactive dashboard is located in:

c3rec_frontend/


Features:

    Dashboard to simulate a student profile

    Interactive recommendations

    Course path visualization (timeline and graphs)

    Concept mastery charts (Radar, Bar, Donut)

    Auto-update with backend predictions

    Clean professional UI using TailwindCSS

    Data visualization powered by Recharts

Run the frontend

cd c3rec_frontend
npm install
npm run dev


Frontend URL:

http://localhost:3000


Inference Module (utils/inference.py)

This module contains the core logic for:

Preprocessing

tensorize_user_history converts raw user history from JSON to model-ready tensors.

Predictions

    predict_next_step computes:

        next correctness probability

        ranked questions and lessons

        ability score

Recommendations

    recommend_next_question — Top-K next questions

    recommend_lessons — Top-K lectures


Concept Mastery

get_mastery_graph computes mastery per concept using model hidden states.


Course Path Generation

generate_course_path implements auto-regressive prediction to generate:

    multi-step learning path

    expected question sequence

    expected lecture sequence

    predicted correctness probability per step


Model (C3Rec v10)

The model is defined in:

model/architecture.py


Key components: 

    Embeddings for questions, concepts, lectures

    Continuous feature projections (elapsed time, timestamp, correctness)

    Positional embeddings

    Transformer encoder

    Three prediction heads:

        next correctness

        next question

        next lecture

    Ability regression head

    Hidden states used for mastery scoring
    

Weights and metadata are stored in:

c3rec_model_v10_best.pt
c3rec_metadata_v10.pt



Training Pipeline

The full training script:

training/c3rec_training_v2.py


Includes:

    Dataset loading from Parquet

    Collation and batching

    Training loop

    Checkpoint saving

    Metadata creation (vocabularies, stats, config)


Synthetic tester:

training/tester.py


Simulates several user profiles and prints:

    recommendations

    predicted ability

    predicted correctness

    top questions/lectures


Example API Request:
POST /recommend_questions
{
  "user_id": 123,
  "question": [101, 102, 103],
  "concept": [5, 5, 6],
  "lecture": [10, 10, 11],
  "elapsed": [30, 40, 50],
  "time": [1000, 2000, 3000],
  "correct": [1, 0, 1],
  "topk": 5
}


Dependencies
Backend

Defined in requirements.txt:

    fastapi

    uvicorn

    torch

    pandas

    numpy

    pydantic

Frontend

Installed via npm install:

    next

    react

    tailwindcss

    recharts

    axios

    framer-motion