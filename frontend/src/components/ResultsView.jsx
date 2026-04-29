/**
 * ResultsView — displays the overall verdict card and the list of
 * individual claim analyses.
 */
import ClaimCard from './ClaimCard';

const VERDICT_ICONS = {
  Supported: '✅',
  Contradicted: '❌',
  Mixed: '⚠️',
  Unknown: '❓',
};

export default function ResultsView({ data }) {
  const verdictClass = data.verdict.toLowerCase();

  return (
    <div className="results-section">
      {/* --- Overall Verdict Card --- */}
      <div className={`verdict-card status-${verdictClass}`}>
        <div className="verdict-header">
          <span className={`verdict-badge ${verdictClass}`}>
            {VERDICT_ICONS[data.verdict] || '❓'} {data.verdict}
          </span>

          <div className="confidence-display">
            <div className="confidence-bar-track">
              <div
                className={`confidence-bar-fill ${verdictClass}`}
                style={{ width: `${Math.round(data.confidence_score * 100)}%` }}
              />
            </div>
            <span className="confidence-value">
              {Math.round(data.confidence_score * 100)}%
            </span>
          </div>
        </div>

        <p className="verdict-explanation">{data.explanation}</p>

        <div className="uncertainty-note">
          <span className="uncertainty-icon">ℹ️</span>
          <p className="uncertainty-text">{data.uncertainty_note}</p>
        </div>
      </div>

      {/* --- Individual Claims --- */}
      <div className="claims-header">
        Extracted Claims
        <span className="claims-count">{data.claims.length}</span>
      </div>

      <div className="claims-list">
        {data.claims.map((claim, i) => (
          <ClaimCard key={i} claim={claim} index={i} />
        ))}
      </div>
    </div>
  );
}
