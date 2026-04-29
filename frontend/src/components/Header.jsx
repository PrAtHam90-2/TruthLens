/**
 * Header component for TruthLens.
 */
export default function Header() {
  return (
    <header className="header">
      <div className="header-inner">
        <div className="header-brand">
          <div className="header-logo">TL</div>
          <div>
            <div className="header-title">TruthLens</div>
            <div className="header-subtitle">Misinformation Detector</div>
          </div>
        </div>
        <div className="header-subtitle" style={{ fontSize: '0.7rem' }}>
          v0.1 MVP
        </div>
      </div>
    </header>
  );
}
