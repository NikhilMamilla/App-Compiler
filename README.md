# AppCompiler 🚀

**Natural Language → Validated App Architecture (UI + API + DB)**

AppCompiler is an autonomous pipeline that converts simple descriptions into structured, cross-validated application blueprints. It uses a 4-stage reasoning process to ensure that your UI pages, API endpoints, and Database tables are perfectly synced and ready for code generation.

## 🛠️ The 4-Stage Pipeline

1.  **Intent Extraction**: Parses the prompt into entities, roles, and core features.
2.  **System Design**: Architectures the app (pages, API groups, permissions).
3.  **Schema Generation**: Generates detailed JSON schemas for UI, API, and DB in parallel.
4.  **Refinement & Repair**: Cross-validates all schemas and applies deterministic or LLM-based repairs to fix inconsistencies.

## 📦 Project Structure

- `api/`: FastAPI server providing the `/generate` endpoint.
- `pipeline/`: Core logic for all 4 stages and the repair engine.
- `frontend/`: Modern React/TypeScript frontend with a dual-theme dashboard.
- `eval/`: Automated evaluation framework to run test cases and generate reports.

## 🚀 Getting Started

### 1. Backend Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Configure environment
# Add GROQ_API_KEY_1...10 to your .env file
uvicorn api.main:app --reload --port 8000
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 🌐 Deployment (Vercel)

This project is configured for easy deployment on Vercel:

1.  **Push to GitHub**: Push your code to a GitHub repository.
2.  **Import to Vercel**: Import the repository into Vercel.
3.  **Environment Variables**: Add your `GROQ_API_KEY_1...10` in the Vercel Project Settings.
4.  **Done**: Vercel will automatically detect the `vercel.json` and deploy both the frontend and backend.

### 3. Run Evaluations
```bash
python eval/run_evals.py --report eval_report.txt
```

## 🔐 Environment Variables
Create a `.env` file in the root with your Groq API keys:
```env
GROQ_API_KEY_1=your_key_here
GROQ_API_KEY_2=your_key_here
...
```

## 🎯 Key Features
- **Round-robin API Rotation**: Automatically switches between 10 keys to bypass rate limits.
- **Deterministic Repair**: Rule-based engine that fixes 90% of schema mismatches without extra LLM costs.
- **Cross-Layer Validation**: Ensures API endpoints match the UI requirements and DB columns.
- **Dual-Theme UI**: High-fidelity dashboard for real-time generation monitoring.
