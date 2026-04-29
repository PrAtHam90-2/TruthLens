/**
 * TruthLens — Main Application
 *
 * Text-first misinformation detection UI.
 * Sends text to the FastAPI backend and displays structured results.
 */
import { useState } from 'react';
import Header from './components/Header';
import ResultsView from './components/ResultsView';
import './index.css';

const API_URL = '/api/v1/analyze';
const MAX_CHARS = 5000;

export default function App() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const canAnalyze = text.trim().length >= 10 && !loading;

  async function handleAnalyze() {
    if (!canAnalyze) return;

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim() }),
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => null);
        throw new Error(
          errData?.detail || `Server error (${response.status})`
        );
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err.message || 'Failed to connect to the analysis server.');
    } finally {
      setLoading(false);
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
      handleAnalyze();
    }
  }

  return (
    <>
      <Header />

      <main className="main-content">
        {/* --- Text Input --- */}
        <section className="input-section">
          <label className="input-label" htmlFor="text-input">
            Paste text to analyze
          </label>

          <div className="textarea-wrapper">
            <textarea
              id="text-input"
              className="text-input"
              placeholder="Paste a news article, social media post, or any claim you'd like to fact-check…"
              value={text}
              onChange={(e) => setText(e.target.value.slice(0, MAX_CHARS))}
              onKeyDown={handleKeyDown}
              disabled={loading}
            />
          </div>

          <div className="input-footer">
            <span className="char-count">
              {text.length} / {MAX_CHARS}
            </span>
            <button
              className="analyze-btn"
              onClick={handleAnalyze}
              disabled={!canAnalyze}
              title="Ctrl+Enter to analyze"
            >
              {loading ? (
                <>
                  <span className="btn-icon">⏳</span> Analyzing…
                </>
              ) : (
                <>
                  <span className="btn-icon">🔍</span> Analyze
                </>
              )}
            </button>
          </div>
        </section>

        {/* --- Error --- */}
        {error && (
          <div className="error-banner">
            <span className="error-icon">⚠️</span>
            <p className="error-text">{error}</p>
          </div>
        )}

        {/* --- Loading --- */}
        {loading && (
          <div className="loading-container">
            <div className="loading-spinner" />
            <p className="loading-text">
              Extracting claims and checking evidence…
            </p>
          </div>
        )}

        {/* --- Results --- */}
        {result && !loading && <ResultsView data={result} />}
      </main>

      <footer className="footer">
        <p className="footer-text">
          TruthLens v0.1 — Results are indicative, not definitive. Always verify
          critical information with trusted sources.
        </p>
      </footer>
    </>
  );
}
