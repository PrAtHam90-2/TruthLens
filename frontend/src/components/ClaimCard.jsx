/**
 * ClaimCard — displays a single extracted claim with its verdict,
 * confidence, claim type, evidence source, and evidence.
 *
 * Now includes an expandable Evidence Panel showing individual
 * evidence items from multi-source fusion.
 */

import { useState } from 'react';

const STATUS_ICONS = {
  Supported: '✅',
  Contradicted: '❌',
  Mixed: '⚠️',
  Unknown: '❓',
  Unverifiable: '🔮',
};

const TYPE_LABELS = {
  Factual: null,
  Opinion: '💭 Opinion',
  Unverifiable: '🔮 Unverifiable',
};

const EVIDENCE_SOURCE_LABELS = {
  corpus: { label: '📚 Trusted Source', className: 'evidence-source-corpus' },
  semantic: { label: '🔍 Semantic Match', className: 'evidence-source-semantic' },
  dynamic: { label: '🌐 Wikipedia', className: 'evidence-source-dynamic' },
  weak_corpus: { label: '📄 Partial Match', className: 'evidence-source-weak' },
  llm_only: { label: '🤖 LLM Only', className: 'evidence-source-llm' },
  none: { label: '⚪ No Evidence', className: 'evidence-source-none' },
};

const SOURCE_TYPE_ICONS = {
  corpus: '📚',
  semantic: '🔍',
  dynamic: '🌐',
  weak_corpus: '📄',
  llm_only: '🤖',
  none: '⚪',
};

const ROLE_CONFIG = {
  supporting: { icon: '✅', label: 'Supporting', className: 'role-supporting' },
  conflicting: { icon: '❌', label: 'Conflicting', className: 'role-conflicting' },
  neutral: { icon: '➖', label: 'Neutral', className: 'role-neutral' },
};

export default function ClaimCard({ claim, index }) {
  const [panelOpen, setPanelOpen] = useState(false);
  const statusClass = claim.status.toLowerCase();
  const typeBadge = claim.claim_type ? TYPE_LABELS[claim.claim_type] : null;
  const evidenceSource = claim.evidence_source
    ? EVIDENCE_SOURCE_LABELS[claim.evidence_source]
    : null;

  const hasItems = claim.evidence_items && claim.evidence_items.length > 0;

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
          {evidenceSource && (
            <span className={`evidence-source-badge ${evidenceSource.className}`}>
              {evidenceSource.label}
            </span>
          )}
          {claim.source_count > 0 && (
            <span className="evidence-source-badge evidence-source-count">
              📊 {claim.source_count} source{claim.source_count !== 1 ? 's' : ''}
            </span>
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

      {/* Evidence Panel — expandable */}
      {hasItems && (
        <div className="evidence-panel-wrapper">
          <button
            className="evidence-panel-toggle"
            onClick={() => setPanelOpen(!panelOpen)}
          >
            {panelOpen ? '▾' : '▸'} {panelOpen ? 'Hide' : 'Show'} {claim.evidence_items.length} evidence source{claim.evidence_items.length !== 1 ? 's' : ''}
          </button>
          {panelOpen && (
            <div className="evidence-panel">
              {claim.evidence_items.map((item, i) => {
                const roleConfig = ROLE_CONFIG[item.role] || ROLE_CONFIG.neutral;
                const typeIcon = SOURCE_TYPE_ICONS[item.source_type] || '📄';
                return (
                  <div className="evidence-item" key={i}>
                    <div className="evidence-item-header">
                      <span className="evidence-item-type">
                        {typeIcon} {item.source_name}
                      </span>
                      <span className={`role-badge ${roleConfig.className}`}>
                        {roleConfig.icon} {roleConfig.label}
                      </span>
                    </div>
                    <p className="evidence-item-text">{item.text}</p>
                    {item.source_url && (
                      <a
                        className="evidence-item-url"
                        href={item.source_url}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        {item.source_url}
                      </a>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </div>
      )}

      {claim.confidence_reason && (
        <div className="claim-confidence-reason">
          <div className="evidence-label">Confidence Reasoning</div>
          <p className="confidence-reason-text">{claim.confidence_reason}</p>
        </div>
      )}
    </div>
  );
}
