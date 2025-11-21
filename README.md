✅ Project Components (Completed)
1. Data Loader (data_loader.py)

Loads a cleaned EdNet-style CSV file.

Encodes:

user_id

assessmentItemID

KnowledgeTag

answerCode

Builds per-user sequences.

Converts sequences into PyTorch dataset samples (seq → next item).

Provides a batch collate_fn with padding.

2. Model (model_c3rec.py)

A simplified C3Rec-inspired architecture including:

Item embeddings

Concept embeddings

GRU sequence encoder

Cognitive mastery estimation

Distillation layer combining mastery & hidden state

Final scoring head producing logits for all items

Top-k prediction function

Fully implemented and ready for training & inference.

3. Training Pipeline (train.py)

Includes:

Train loop (Adam optimizer, gradient clipping)

Validation loop

NDCG@10 & Recall@20 calculation

Dataset split into train/val

Saves best model to model_best.pt

TensorBoard support (optional)

This script can fully train the model from scratch using the prepared dataset.

4. Inference / Recommendation (predict.py)

Implements:

Loading the saved checkpoint

Rebuilding label encoders

Preparing user sequence inputs

Generating top-k lesson recommendations

Mapping predicted indices back to original assessmentItemID

Returns a clear list of recommended items with scores.

5. FastAPI Backend (api_main.py)

Contains:

/health endpoint

/recommend_next_lesson which returns the model’s output

/log_interaction placeholder (future-ready)

This backend serves recommendations to the UI or external applications.

6. Streamlit UI (ui_app.py)

A simple user interface supporting:

Input of user_id

Selection of number of recommendations

Displaying model outputs in a table

Communicates directly with the FastAPI backend.

7. SQL Schema (schema.sql)

Database tables defined (not yet fully used):

users

concepts

lessons

interactions

Prepared for future logging or metadata integration.
