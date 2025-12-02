# LearWise Inc — AI-Powered Learning Recommendation System

A deep learning-based educational recommendation system that provides personalized course recommendations, question suggestions, and learning path generation based on student interaction history.

## 🎯 Overview

C3Rec leverages transformer-based neural networks to:
- **Recommend next questions** based on student performance history
- **Suggest relevant lessons** tailored to individual learning needs
- **Generate personalized course paths** with autoregressive prediction
- **Track concept mastery** through detailed analytics and visualizations
- **Predict student ability** scores from interaction patterns

## 🏗️ Architecture

The system consists of three main components:

1. **Backend API** (`api/`) - FastAPI server with PyTorch Transformer Encoder model (C3Rec v10)
2. **Frontend Dashboard** (`c3rec_frontend/`) - Next.js 14 dashboard with interactive visualizations
3. **Model & Training** (`model/`, `Model training and validation/`) - Transformer-based multi-task learning model

## 📁 Project Structure

```
c3rec_project/
├── api/                           # FastAPI backend
│   └── main.py                    # API endpoints and model loading
├── model/                         # PyTorch model + metadata
│   ├── architecture.py            # C3RecModel definition
│   ├── load_model.py              # Model loading utilities
│   ├── c3rec_model_v10_best.pt    # Trained model weights
│   └── c3rec_metadata_v10.pt      # Metadata and vocabularies
├── utils/                         # Inference utilities
│   └── inference.py               # Recommendation and prediction functions
├── DataPrep/                      # Data preprocessing scripts
│   ├── Merge_Interactions.py
│   ├── Merge_Content_Interactions.py
│   └── Fix_timestamps_seq.py
├── Model training and validation/
│   └── Main.py                    # Training and validation script
├── c3rec_frontend/                 # Next.js dashboard UI
│   ├── app/
│   │   ├── page.tsx               # Main dashboard component
│   │   ├── layout.tsx
│   │   └── globals.css
│   └── package.json
├── requirements.txt               # Python dependencies
└── README.md
```

## 🚀 Installation & Setup

### Prerequisites

- **Python** 3.10+ (with virtual environment support)
- **Node.js** 18+ and npm
- **CUDA-capable GPU** (optional, for faster inference)

### Backend Installation (FastAPI + PyTorch)

#### 1. Create and activate virtual environment

```bash
cd c3rec_project
python3 -m venv venv

# macOS / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

#### 2. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 3. Verify model files

Ensure these files exist in `model/`:
- `c3rec_model_v10_best.pt`
- `c3rec_metadata_v10.pt`
- `load_model.py`

If files are located elsewhere, adjust paths in `api/main.py`.

#### 4. Start the FastAPI server

```bash
uvicorn api.main:app --reload
```

Backend runs at: **http://127.0.0.1:8000**

### Frontend Installation (Next.js 14 + Tailwind)

#### 1. Verify Node.js installation

```bash
node -v
npm -v
```

Install Node.js LTS from https://nodejs.org if needed.

#### 2. Install dependencies

```bash
cd c3rec_frontend
npm install
```

#### 3. Configure API endpoint

In `c3rec_frontend/app/page.tsx`, verify:

```typescript
const API_BASE = "http://127.0.0.1:8000";
```

Update if backend runs on a different host/port.

#### 4. Start development server

```bash
npm run dev
```

Frontend runs at: **http://localhost:3000**

### Running Both Services

**Terminal 1 (Backend):**
```bash
cd c3rec_project
source venv/bin/activate  # or venv\Scripts\activate on Windows
uvicorn api.main:app --reload
```

**Terminal 2 (Frontend):**
```bash
cd c3rec_project/c3rec_frontend
npm run dev
```

## 📊 Model Architecture

The C3Rec model uses a **Transformer Encoder** with:

- **Embeddings**: Question, Concept, and Lecture embeddings
- **Continuous Features**: Projected elapsed time, timestamp, and correctness
- **Positional Encoding**: Learned positional embeddings
- **Multi-task Heads**:
  - Correctness prediction (sigmoid)
  - Next question prediction (linear)
  - Next lesson prediction (linear)
  - Ability score MLP (pooled over sequence)

### Model Configuration (v10)

- `d_model`: 128
- `n_heads`: 4
- `n_layers`: 2
- `dropout`: 0.1
- `seq_len`: 200

### Input Format

```json
{
  "user_id": 123,
  "question": [101, 102, 103, ...],
  "concept": [5, 5, 6, ...],
  "lecture": [10, 10, 11, ...],
  "elapsed": [30.0, 40.0, 35.0, ...],
  "time": [1000.0, 2000.0, 3000.0, ...],
  "correct": [1.0, 1.0, 0.0, ...],
  "topk": 5,
  "steps": 10
}
```

All arrays must have equal length and represent chronological interaction history.

## 🔧 API Endpoints

### `POST /recommend_questions`

Returns top-k recommended question IDs.

**Response:**
```json
{
  "user_id": 123,
  "recommended_questions": ["Q_101", "Q_102", ...]
}
```

### `POST /recommend_lessons`

Returns top-k recommended lesson IDs.

**Response:**
```json
{
  "user_id": 123,
  "recommended_lessons": ["L_10", "L_11", ...]
}
```

### `POST /course_path`

Generates autoregressive course path with predicted correctness probabilities.

**Response:**
```json
{
  "user_id": 123,
  "course_path": [
    {
      "step": 1,
      "question": "Q_201",
      "lecture": "L_15",
      "concept": "C_7",
      "predicted_correct_probability": 0.75
    }
  ]
}
```

### `POST /mastery_graph`

Computes mastery scores for each concept.

**Response:**
```json
{
  "user_id": 123,
  "mastery_graph": [
    {
      "concept": "C_5",
      "mastery": 0.85,
      "count": 5
    }
  ]
}
```

## 🧪 Training the Model

```bash
cd "Model training and validation"
python Main.py
```

The training script:
1. Loads preprocessed dataset
2. Splits into train/validation sets
3. Trains with multi-task loss
4. Evaluates using NDCG@k metrics
5. Saves best model checkpoint

Edit `CONFIG` in `Model training and validation/Main.py` to adjust hyperparameters.

## 📝 Data Preprocessing

Scripts in `DataPrep/`:
- `Merge_Interactions.py` - Merges multiple interaction CSV files
- `Merge_Content_Interactions.py` - Merges content and interaction data
- `Fix_timestamps_seq.py` - Fixes timestamp sequences

## 🛠️ Dependencies

### Python (`requirements.txt`)
- `fastapi`, `uvicorn[standard]` - Web framework and server
- `torch` - Deep learning framework
- `pandas`, `numpy` - Data manipulation
- `pydantic` - Data validation
- `scikit-learn` - ML utilities
- `pyarrow` - Data serialization

### Frontend (`c3rec_frontend/package.json`)
- `next`, `react`, `react-dom` - Framework and UI
- `tailwindcss` - CSS framework
- `recharts`, `reactflow` - Visualizations
- `framer-motion` - Animations
- `axios` - HTTP client

## 📈 Features

- ✅ Personalized AI-powered recommendations
- ✅ Autoregressive course path generation
- ✅ Concept-level mastery tracking
- ✅ Real-time inference with GPU support
- ✅ Interactive dashboard with visualizations
- ✅ Multi-task learning architecture

## 🔍 Model Version

**Current:** v10

- `model/c3rec_model_v10_best.pt` - Trained weights
- `model/c3rec_metadata_v10.pt` - Metadata and vocabularies

