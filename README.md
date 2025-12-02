 C3Rec — AI-Powered Learning Recommendation System

    Installation & Environment Setup:

1. Backend Installation (FastAPI + PyTorch)
1.1. Create and activate a virtual environment
cd c3rec_project
python3 -m venv venv
source venv/bin/activate     # macOS / Linux
venv\Scripts\activate        # Windows

1.2. Install Python dependencies
pip install --upgrade pip
pip install -r requirements.txt

1.3. Confirm that the model files exist

Inside the folder model/, confirm the following files:

model/
 ├── c3rec_model_v10_best.pt
 ├── c3rec_metadata_v10.pt
 └── load_model.py


If your files are located elsewhere, adjust the paths inside api/main.py.

1.4. Run the FastAPI server
uvicorn api.main:app --reload


Your backend is now running on:

http://127.0.0.1:8000

2. Frontend Installation (Next.js 14 + Tailwind)
2.1. Install Node.js (if not installed)

Verify:

node -v
npm -v


If missing, install Node.js LTS from:
https://nodejs.org

2.2. Move to the frontend folder
cd c3rec_frontend

2.3. Install dependencies
npm install


This installs Next.js, TailwindCSS, Recharts, and all UI dependencies.

2.4. Start the development server
npm run dev


The frontend runs at:

http://localhost:3000

3. Connecting Frontend & Backend

In the file:

c3rec_frontend/app/page.tsx


Confirm this line points to your backend:

const API_BASE = "http://127.0.0.1:8000";


If the backend runs on another machine/server, update this value.

4. Recommended Project Structure
c3rec_project/
 ├── api/                  # FastAPI backend
 │    └── main.py
 ├── model/                # PyTorch model + metadata
 ├── utils/                # inference preprocessing & prediction logic
 ├── training/             # training scripts (optional)
 ├── c3rec_frontend/       # Next.js dashboard UI
 ├── requirements.txt
 └── README.md

5. Starting Both Services

Open two terminals:

Terminal 1 → Backend

cd c3rec_project
source venv/bin/activate
uvicorn api.main:app --reload


Terminal 2 → Frontend

cd c3rec_project/c3rec_frontend
npm run dev


The system is now fully operational.