# TruthLens 🔍

**Text-first misinformation and bias detection app** — paste any text, get a verdict with evidence and explainability.

> **Version 0.1 MVP** — Focuses on text input only. No image, video, or deepfake detection yet.

---

## Features

- 🧠 **LLM-powered claim extraction** — Uses Google Gemini to extract atomic factual claims from any text
- 📚 **Evidence retrieval** — Matches claims against a curated trusted fact corpus
- 🏷️ **Verdict classification** — Each claim is classified as Supported, Contradicted, Mixed, or Unknown
- 📊 **Confidence scoring** — Transparent confidence levels for every claim
- ⚠️ **Uncertainty notes** — Always shows limitations and caveats (never a blunt "true/fake")
- 🔄 **Heuristic fallback** — If the LLM is unavailable, falls back to sentence-level heuristics

---

## Tech Stack

| Layer    | Technology     |
| -------- | -------------- |
| Frontend | React + Vite   |
| Backend  | FastAPI        |
| LLM      | Google Gemini  |
| Language | Python 3.10+   |

---

## Project Structure

```
TruthLens/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point
│   │   ├── api/routes.py        # POST /api/v1/analyze
│   │   ├── core/config.py       # Environment settings
│   │   ├── models/schemas.py    # Pydantic request/response models
│   │   └── services/
│   │       ├── analyzer.py      # Orchestrator pipeline
│   │       ├── llm_client.py    # Gemini LLM interface (swappable)
│   │       ├── corpus.py        # Trusted fact corpus
│   │       └── fallback.py      # Heuristic claim extractor
│   ├── tests/test_api.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── components/
│   │       ├── Header.jsx
│   │       ├── ClaimCard.jsx
│   │       └── ResultsView.jsx
│   ├── index.html
│   └── package.json
└── README.md
```

---

## Setup & Run

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Google Gemini API key** — get one free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### 1. Backend

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate
# Activate (macOS/Linux)
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
copy .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Run the server
uvicorn app.main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### 2. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run the dev server
npm run dev
```

The UI will be available at `http://localhost:5173`.

---

## API Reference

### `POST /api/v1/analyze`

**Request:**
```json
{
  "text": "The earth is flat and vaccines contain microchips."
}
```

**Response:**
```json
{
  "verdict": "Contradicted",
  "confidence_score": 0.95,
  "uncertainty_note": "Analysis is based on a limited trusted corpus and LLM reasoning...",
  "explanation": "All claims in the text are contradicted by trusted evidence.",
  "claims": [
    {
      "claim": "The earth is flat.",
      "status": "Contradicted",
      "evidence": "The Earth is an oblate spheroid... (Source: NASA, ESA)",
      "confidence": 0.98
    }
  ]
}
```

### `GET /api/v1/health`

Returns `{"status": "ok"}`.

---

## Sample Test Requests

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Analyze text
curl -X POST http://localhost:8000/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"text": "The Earth is flat and the moon landing was faked by NASA."}'
```

---

## Running Tests

```bash
cd backend
pytest tests/ -v
```

---

## Architecture Notes

- **Modular LLM layer**: `llm_client.py` defines an abstract `BaseLLMClient`. Swap Gemini for OpenAI/Anthropic by implementing the interface.
- **Graceful degradation**: If the LLM is unavailable or errors, the system falls back to heuristic sentence splitting + corpus-only matching.
- **No absolute truth**: Every response includes an `uncertainty_note` — by design, TruthLens never returns a blunt "true/fake" without context.

---

## License

MIT
