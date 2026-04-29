# TruthLens 🔍

**Text-first misinformation and bias detection app** — paste any text, get a structured verdict with evidence, calibrated confidence, and explainability.

> **Version 0.1 MVP** — Focuses on text input only. No image, video, or deepfake detection yet.

---

## Features

- 🧠 **LLM-powered claim extraction** — Uses Groq (Llama 3.x) to extract atomic factual claims from any text
- 📚 **Evidence retrieval** — Matches claims against a curated trusted fact corpus (12 topics, expandable)
- 🏷️ **Verdict classification** — Each claim is classified as `Supported`, `Contradicted`, `Mixed`, or `Unknown`
- 📊 **Multi-factor confidence calibration** — Confidence is never just the raw LLM score; it's adjusted based on evidence strength, source count, keyword match quality, claim clarity, and LLM/corpus agreement
- 💬 **Confidence reasoning** — Every claim includes a plain-English explanation of *why* that confidence level was assigned
- ⚠️ **Uncertainty notes** — Always shows limitations and caveats (never a blunt "true/fake")
- 🔄 **Heuristic fallback** — If the LLM is unavailable, falls back to sentence-level heuristics + corpus matching
- 🔌 **Swappable LLM backend** — Abstract `BaseLLMClient` interface makes it easy to swap Groq for OpenAI, Anthropic, or Gemini

---

## Tech Stack

| Layer     | Technology                   |
| --------- | ---------------------------- |
| Frontend  | React + Vite                 |
| Backend   | FastAPI (Python 3.10+)       |
| LLM       | Groq — `llama-3.3-70b-versatile` |
| Validation| Pydantic v2                  |
| Styling   | Vanilla CSS (dark theme)     |

---

## Project Structure

```
TruthLens/
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI entry point + CORS config
│   │   ├── api/routes.py        # POST /api/v1/analyze, GET /api/v1/health
│   │   ├── core/config.py       # Environment settings (pydantic-settings)
│   │   ├── models/schemas.py    # Pydantic request/response models
│   │   └── services/
│   │       ├── analyzer.py      # Orchestrator pipeline
│   │       ├── llm_client.py    # BaseLLMClient + GroqLLMClient
│   │       ├── corpus.py        # Trusted fact corpus + EvidenceMatch
│   │       ├── confidence.py    # Multi-factor confidence calibration engine
│   │       └── fallback.py      # Heuristic claim extractor (no-LLM fallback)
│   ├── tests/test_api.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── index.css
│   │   └── components/
│   │       ├── Header.jsx
│   │       ├── ClaimCard.jsx    # Shows claim, verdict, evidence, confidence reason
│   │       └── ResultsView.jsx
│   ├── vite.config.js           # Dev proxy: /api/* → localhost:8000
│   ├── index.html
│   └── package.json
└── README.md
```

---

## Setup & Run

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Groq API key** — get one free at [console.groq.com/keys](https://console.groq.com/keys)

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
# Edit .env and set your GROQ_API_KEY

# Run the server
uvicorn app.main:app --reload --port 8000
```

The API is available at `http://localhost:8000`.  
Interactive Swagger docs at `http://localhost:8000/docs`.

### 2. Frontend

```bash
cd frontend

# Install dependencies
npm install

# Run the dev server
npm run dev
```

The UI is available at `http://localhost:5173`.

> **Note:** The Vite dev server proxies all `/api/*` requests to `localhost:8000` — no CORS issues, no manual URL changes needed.

---

## Environment Variables

Create `backend/.env` from the example:

```bash
# backend/.env
GROQ_API_KEY=gsk_your_key_here
```

| Variable     | Required | Default                    | Description                        |
|--------------|----------|----------------------------|------------------------------------|
| `GROQ_API_KEY` | Yes    | `""`                       | Your Groq API key                  |
| `GROQ_MODEL` | No       | `llama-3.3-70b-versatile`  | Groq model to use                  |
| `FRONTEND_ORIGIN` | No  | `http://localhost:5173`    | CORS allowed origin                |

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
  "confidence_score": 0.84,
  "uncertainty_note": "Analysis is based on a limited trusted corpus and LLM reasoning. Results should be treated as indicative, not definitive.",
  "explanation": "All claims in the text are contradicted by trusted evidence.",
  "claims": [
    {
      "claim": "The earth is flat.",
      "status": "Contradicted",
      "evidence": "The Earth is an oblate spheroid. Confirmed by satellite imagery, physics, and centuries of observation. (Source: NASA, ESA, and scientific consensus)",
      "confidence": 0.88,
      "confidence_reason": "High confidence due to 5 independent sources and strong corpus evidence; dampened from raw 0.95 to avoid overstatement."
    },
    {
      "claim": "Vaccines contain microchips.",
      "status": "Contradicted",
      "evidence": "Vaccines do not contain microchips or tracking devices. (Source: WHO, FDA, peer-reviewed immunology research)",
      "confidence": 0.85,
      "confidence_reason": "Strong direct corpus evidence from 3 major health organizations; no LLM/corpus disagreement detected."
    }
  ]
}
```

### `GET /api/v1/health`

```json
{ "status": "ok", "service": "TruthLens API" }
```

---

## Confidence Calibration System

Raw LLM confidence scores tend to be overconfident (e.g. `0.99`). TruthLens applies a multi-factor calibration:

| Signal | Effect |
|--------|--------|
| Corpus evidence found | +bonus scaled to evidence strength |
| Independent source count (≥4) | +0.06 |
| Strong keyword match (≥50%) | +0.04 |
| LLM and corpus agree | +0.05 |
| LLM and corpus disagree | −0.08 |
| No corpus evidence | −0.15 |
| Very short or very long claim | −0.03 to −0.05 |

Typical output range: **0.50 – 0.90**. Scores above 0.92 are reserved for incontrovertible, multi-source scientific consensus.

---

## Running Tests

```bash
cd backend
pytest tests/ -v
```

All 5 integration tests should pass (no API key required — uses heuristic fallback).

---

## Architecture Notes

- **Modular LLM layer**: `llm_client.py` defines an abstract `BaseLLMClient`. Add a new provider by implementing `extract_claims()` and `classify_claim()`, then swap the singleton in `analyzer.py`.
- **Calibrated confidence**: `confidence.py` post-processes raw LLM confidence using corpus metadata (`evidence_strength`, `source_count`) and match quality — producing consistent, trustworthy scores.
- **Vite proxy**: All `/api` requests from the frontend are transparently forwarded to the FastAPI backend by Vite's dev server, eliminating CORS friction in development.
- **Graceful degradation**: If the LLM is unavailable or errors, the system falls back to heuristic sentence splitting + corpus-only matching. The app remains functional.
- **No absolute truth**: Every response includes an `uncertainty_note` — TruthLens is designed to inform, not to replace critical thinking.

---

## Roadmap

- [ ] Vector search (Chroma/FAISS) for semantic evidence retrieval
- [ ] URL input — fetch and analyze web articles directly
- [ ] Multi-modal support — image and deepfake detection
- [ ] Result caching (Redis) to reduce LLM costs on repeat queries
- [ ] Export results as PDF or JSON

---

## License

MIT
