type SectionLoaderProps = {
  rows?: number;
  compact?: boolean;
  size?: "sm" | "md" | "lg";
  fullPage?: boolean;
  className?: string;
};

export function AppLoadingScreen() {
  return (
    <div className="app-loading-screen">
      <div className="app-loading-card">
        <div className="app-loading-spinner" aria-hidden />
        <div className="app-loading-title" style={{ color: "#ffffff" }}>Loading your workspace</div>
        <div className="app-loading-subtitle">Fetching your profile, roles, and dashboard data.</div>
      </div>
    </div>
  );
}

export function SectionLoader({ rows, compact = false, size, fullPage = false, className = "" }: SectionLoaderProps) {
  let finalSize = size;
  if (!finalSize) {
    if (compact) finalSize = "sm";
    else if (rows && rows <= 2) finalSize = "sm";
    else if (rows && rows >= 6) finalSize = "lg";
    else if (fullPage) finalSize = "lg";
    else finalSize = "md";
  }

  const loaderContent = (
    <div className={`section-loader section-loader--${finalSize} ${className}`.trim()}>
      <div className={`section-loader__spinner section-loader__spinner--${finalSize}`} aria-hidden />
      <div className="section-loader__text">Loading data...</div>
    </div>
  );

  if (fullPage) {
    return (
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', minHeight: '60vh', width: '100%' }}>
        {loaderContent}
      </div>
    );
  }

  return loaderContent;
}
