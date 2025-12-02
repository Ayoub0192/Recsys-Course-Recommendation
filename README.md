# LearnWise Inc - Powered Learning Recommendation System

A deep learning-based educational recommendation system that provides personalized course recommendations, question suggestions, and learning path generation based on student interaction history.

## 🎯 Overview

C3Rec is an intelligent educational platform that leverages transformer-based neural networks to:
- **Recommend next questions** based on student performance history
- **Suggest relevant lessons** tailored to individual learning needs
- **Generate personalized course paths** with autoregressive prediction
- **Track concept mastery** through detailed analytics and visualizations
- **Predict student ability** scores from interaction patterns

## 🏗️ Architecture

The project consists of three main components:

### 1. **Backend API** (`api/`)
- **Framework**: FastAPI
- **Model**: PyTorch-based Transformer Encoder (C3Rec v10)
- **Endpoints**:
  - `/recommend_questions` - Get top-k question recommendations
  - `/recommend_lessons` - Get top-k lesson recommendations
  - `/course_path` - Generate personalized learning path
  - `/mastery_graph` - Compute concept mastery scores

### 2. **Frontend Dashboard** (`c3rec_frontend/`)
- **Framework**: Next.js 14 with React 18
- **UI Libraries**: Tailwind CSS, Framer Motion, Recharts, ReactFlow
- **Features**:
  - Interactive dashboard with multiple visualization tabs
  - Real-time API integration
  - CSV-style input for student history
  - Mastery radar charts and course path visualization

### 3. **Model & Training** (`model/`, `Model training and validation/`)
- **Architecture**: Transformer Encoder with multi-task learning
- **Input Features**:
  - Discrete: Question IDs, Concept IDs, Lecture IDs
  - Continuous: Elapsed time, Timestamp, Correctness (0/1)
- **Outputs**: Next question prediction, next lesson prediction, correctness probability, ability score

## 📁 Project Structure

```
.
├── api/                    # FastAPI backend server
│   └── main.py            # API endpoints and model loading
├── c3rec_frontend/        # Next.js frontend application
│   ├── app/
│   │   ├── page.tsx       # Main dashboard component
│   │   ├── layout.tsx     # App layout
│   │   └── globals.css    # Global styles
│   └── package.json       # Frontend dependencies
├── model/                 # Model architecture and loading
│   ├── architecture.py    # C3RecModel definition
│   ├── load_model.py      # Model loading utilities
│   ├── c3rec_model_v10_best.pt      # Trained model weights
│   └── c3rec_metadata_v10.pt        # Model metadata and vocabularies
├── utils/                 # Inference utilities
│   └── inference.py       # Recommendation and prediction functions
├── DataPrep/              # Data preprocessing scripts
│   ├── Merge_Interactions.py
│   ├── Merge_Content_Interactions.py
│   └── Fix_timestamps_seq.py
├── Model training and validation/
│   └── Main.py            # Training and validation script
├── requirements.txt       # Python dependencies
└── env/                   # Python virtual environment
```

## 🚀 Installation & Environment Setup

### Prerequisites

- **Python** 3.10+ (with virtual environment support)
- **Node.js** 18+ and npm
- **CUDA-capable GPU** (optional, for faster inference)

### Backend Installation (FastAPI + PyTorch)

#### 1.1. Create and activate a virtual environment

```bash
# Navigate to project root
cd c3rec_project

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# macOS / Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

#### 1.2. Install Python dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 1.3. Confirm that the model files exist

Inside the folder `model/`, confirm the following files:

```
model/
├── c3rec_model_v10_best.pt
├── c3rec_metadata_v10.pt
└── load_model.py
```

If your files are located elsewhere, adjust the paths inside `api/main.py`.

#### 1.4. Run the FastAPI server

```bash
uvicorn api.main:app --reload
```

Your backend is now running on:

**http://127.0.0.1:8000**

### Frontend Installation (Next.js 14 + Tailwind)

#### 2.1. Install Node.js (if not installed)

Verify installation:

```bash
node -v
npm -v
```

If missing, install Node.js LTS from: https://nodejs.org

#### 2.2. Move to the frontend folder

```bash
cd c3rec_frontend
```

#### 2.3. Install dependencies

```bash
npm install
```

This installs Next.js, TailwindCSS, Recharts, and all UI dependencies.

#### 2.4. Start the development server

```bash
npm run dev
```

The frontend runs at:

**http://localhost:3000**

### Connecting Frontend & Backend

In the file `c3rec_frontend/app/page.tsx`, confirm this line points to your backend:

```typescript
const API_BASE = "http://127.0.0.1:8000";
```

If the backend runs on another machine/server, update this value accordingly.

### Recommended Project Structure

```
c3rec_project/
├── api/                    # FastAPI backend
│   └── main.py
├── model/                  # PyTorch model + metadata
│   ├── c3rec_model_v10_best.pt
│   ├── c3rec_metadata_v10.pt
│   └── load_model.py
├── utils/                  # Inference preprocessing & prediction logic
│   └── inference.py
├── training/               # Training scripts (optional)
├── c3rec_frontend/         # Next.js dashboard UI
│   ├── app/
│   │   └── page.tsx
│   └── package.json
├── requirements.txt
└── README.md
```

### Starting Both Services

Open **two terminals**:

**Terminal 1 → Backend**

```bash
cd c3rec_project
source venv/bin/activate  # or venv\Scripts\activate on Windows
uvicorn api.main:app --reload
```

**Terminal 2 → Frontend**

```bash
cd c3rec_project/c3rec_frontend
npm run dev
```

The system is now fully operational! 🎉

## 📊 Model Details

### Architecture

The C3Rec model uses a **Transformer Encoder** architecture with:

- **Embeddings**: Question, Concept, and Lecture embeddings
- **Continuous Features**: Projected elapsed time, timestamp, and correctness
- **Positional Encoding**: Learned positional embeddings
- **Multi-task Heads**:
  - Correctness prediction head (sigmoid)
  - Next question prediction head (linear)
  - Next lesson prediction head (linear)
  - Ability score MLP (pooled over sequence)

### Model Configuration (v10)

- `d_model`: 128
- `n_heads`: 4
- `n_layers`: 2
- `dropout`: 0.1
- `seq_len`: 200

### Input Format

The API expects a JSON payload with the following structure:

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

All arrays must have the same length and represent the student's interaction history in chronological order.

## 🔧 API Endpoints

### `POST /recommend_questions`

Returns top-k recommended question IDs based on student history.

**Response:**
```json
{
  "user_id": 123,
  "recommended_questions": ["Q_101", "Q_102", ...]
}
```

### `POST /recommend_lessons`

Returns top-k recommended lesson IDs based on student history.

**Response:**
```json
{
  "user_id": 123,
  "recommended_lessons": ["L_10", "L_11", ...]
}
```

### `POST /course_path`

Generates an autoregressive course path with predicted correctness probabilities.

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
    },
    ...
  ]
}
```

### `POST /mastery_graph`

Computes mastery scores for each concept based on interaction history.

**Response:**
```json
{
  "user_id": 123,
  "mastery_graph": [
    {
      "concept": "C_5",
      "mastery": 0.85,
      "count": 5
    },
    ...
  ]
}
```

## 🧪 Training the Model

To train or retrain the C3Rec model:

```bash
cd "Model training and validation"
python Main.py
```

The training script will:
1. Load the preprocessed dataset
2. Split into train/validation sets
3. Train the model with multi-task loss
4. Evaluate using NDCG@k metrics
5. Save the best model checkpoint

### Training Configuration

Edit `CONFIG` in `Model training and validation/Main.py` to adjust:
- Model architecture parameters
- Training hyperparameters (learning rate, batch size, epochs)
- Loss function weights
- Evaluation metrics

## 📝 Data Preprocessing

The `DataPrep/` directory contains scripts for preprocessing educational interaction data:

- **Merge_Interactions.py**: Merges multiple interaction CSV files
- **Merge_Content_Interactions.py**: Merges content and interaction data
- **Fix_timestamps_seq.py**: Fixes timestamp sequences in the dataset

## 🛠️ Dependencies

### Python Dependencies (`requirements.txt`)
- `fastapi` - Web framework
- `uvicorn[standard]` - ASGI server
- `torch` - Deep learning framework
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `pydantic` - Data validation
- `scikit-learn` - Machine learning utilities
- `pyarrow` - Data serialization

### Frontend Dependencies (`c3rec_frontend/package.json`)
- `next` - React framework
- `react` & `react-dom` - UI library
- `tailwindcss` - CSS framework
- `recharts` - Charting library
- `reactflow` - Graph visualization
- `framer-motion` - Animations
- `axios` - HTTP client

## 📈 Features

- ✅ **Personalized Recommendations**: AI-powered question and lesson suggestions
- ✅ **Course Path Generation**: Autoregressive learning path prediction
- ✅ **Mastery Tracking**: Concept-level mastery visualization
- ✅ **Real-time Inference**: Fast API responses with GPU acceleration support
- ✅ **Interactive Dashboard**: Modern, responsive web interface
- ✅ **Multi-task Learning**: Joint prediction of correctness, questions, and lessons



