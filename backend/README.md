
# AI Credit Scoring App

This is a full-stack AI-powered credit scoring system built with:
- **FastAPI** (backend)
- **OpenAI GPT-4** for LLM risk summarization
- **React** (frontend with TailwindCSS and lucide-react)
- **MongoDB** (recommended for storage and optional vector indexing)

## Features

- 🧠 AI-generated credit score and breakdown
- 📊 Visual sliders and tabbed UI
- 💬 LLM-generated risk summary and suggestions
- 📁 Modular FastAPI backend

## Architecture

```
User Input → React UI → FastAPI (/score) → Rule-based + GPT scoring → JSON → UI rendering
```

## Running Locally

### Backend
```bash
cd ai-credit-scoring-app/backend
cp .env.example .env
# Add your OpenAI key in .env
# Configure AWS services via environment variables
# MODEL_SERVICE=frauddetector|sagemaker-runtime
# FRAUD_DETECTOR_NAME=your-detector
# FRAUD_DETECTOR_EVENT_TYPE=your-event-type
# SAGEMAKER_ENDPOINT_NAME=your-endpoint
pip install -r requirements.txt
uvicorn main:app --reload
```

### Frontend
```bash
cd ai-credit-scoring-app/frontend
npm install
npm start
```

Visit: http://localhost:3000

