# Recsys-Course-Recommendation

This repository contains the current working MVP of a sequential course recommendation system built for the RecSys Startup Sprint.

## ✔ What Is Implemented So Far

### 1. **Data Pipeline**
- `data_loader.py` fully implemented
- Handles:
  - Loading cleaned EdNet-style CSV
  - Label encoding (users, items, concepts)
  - Sorting by timestamp
  - Sequence generation per user
  - Padding via custom collate function

### 2. **Model (C3Rec-inspired)**
- `model_c3rec.py` implemented with:
  - Item, concept, and user embeddings
  - GRU-based sequence encoder
  - Cognitive mastery head
  - Distillation layer
  - Item scorer for next-item prediction
  - `predict_topk()` working

### 3. **Training Loop**
- `train.py` implemented:
  - CrossEntropyLoss
  - Adam optimizer
  - Gradient clipping
  - Validation using NDCG@10, Recall@20
  - Checkpointing (`model_best.pt`)
  - TensorBoard logging

### 4. **Inference**
- `predict.py`:
  - Loads model + encoders
  - Extracts last MAX_SEQ_LEN interactions
  - Runs top-k prediction
  - Returns real item IDs + scores

### 5. **API**
- `api_main.py`:
  - FastAPI backend
  - `/health`
  - `/recommend_next_lesson`
  - `/log_interaction` (stub)

### 6. **UI**
- `ui_app.py`:
  - Streamlit frontend
  - Calls backend API
  - Displays recommendations table

### 7. **Database Schema**
- `schema.sql` created for:
  - users
  - lessons
  - concepts
  - interactions

## 📦 How To Run

### 1. Install Dependencies
```
pip install -r requirements.txt
```

### 2. Train the Model
```
python train.py
```

### 3. Start API Backend
```
uvicorn api_main:app --reload --port 8000
```

### 4. Start UI
```
streamlit run ui_app.py
```

## 📁 Current Status
This repository is a **fully functional MVP**:
- Data → Model → Training → API → UI all connected
- Only small fixes left depending on real KT1 dataset formatting
