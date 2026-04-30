/**
 * ClaimCard — displays a single extracted claim with its verdict,
 * confidence, claim type, and evidence.
 */

const STATUS_ICONS = {
  Supported: '✅',
  Contradicted: '❌',
  Mixed: '⚠️',
  Unknown: '❓',
  Unverifiable: '🔮',
};

const TYPE_LABELS = {
  Factual: null, // Don't show a badge for factual (it's the default)
  Opinion: '💭 Opinion',
  Unverifiable: '🔮 Unverifiable',
};

export default function ClaimCard({ claim, index }) {
  const statusClass = claim.status.toLowerCase();
  const typeBadge = claim.claim_type ? TYPE_LABELS[claim.claim_type] : null;

  return (
    <div
      className={`claim-card status-${statusClass}`}
      style={{ animationDelay: `${index * 80}ms` }}
    >
      <div className="claim-card-header">
        <div className="claim-badges">
          <span className={`claim-status-badge ${statusClass}`}>
            {STATUS_ICONS[claim.status] || '❓'} {claim.status}
          </span>
          {typeBadge && (
            <span className="claim-type-badge">{typeBadge}</span>
          )}
        </div>
        <span className="claim-confidence">
          {Math.round(claim.confidence * 100)}% confidence
        </span>
      </div>

      <p className="claim-text">"{claim.claim}"</p>

      <div className="claim-evidence">
        <div className="evidence-label">Evidence</div>
        <p className="evidence-text">{claim.evidence}</p>
      </div>

      {claim.confidence_reason && (
        <div className="claim-confidence-reason">
          <div className="evidence-label">Confidence Reasoning</div>
          <p className="confidence-reason-text">{claim.confidence_reason}</p>
        </div>
      )}
    </div>
  );
}
