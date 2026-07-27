import type { CSSProperties, ReactNode } from "react";

interface TableToolbarProps {
  search: string;
  onSearchChange: (v: string) => void;
  placeholder?: string;
  showClear: boolean;
  onClear: () => void;
  /** Extra controls rendered to the left of the search box. */
  leftControls?: ReactNode;
  /** Extra controls rendered to the right of the search box (e.g. Add button). */
  rightControls?: ReactNode;
  /** Show count badge "X of Y". */
  count?: { shown: number; total: number };
  style?: CSSProperties;
}

export default function TableToolbar({
  search,
  onSearchChange,
  placeholder = "Search...",
  showClear,
  onClear,
  leftControls,
  rightControls,
  count,
  style,
}: TableToolbarProps) {
  return (
    <div
      style={{
        display: "flex",
        gap: "0.75rem",
        marginBottom: "1rem",
        flexWrap: "wrap",
        alignItems: "center",
        ...style,
      }}
    >
      {leftControls}
      <div
        style={{
          position: "relative",
          minWidth: 240,
          flex: "1 1 240px",
          maxWidth: 380,
        }}
      >
        <input
          type="search"
          value={search}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder={placeholder}
          style={{
            width: "100%",
            padding: "0.55rem 2.25rem 0.55rem 0.9rem",
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: 8,
            color: "inherit",
            fontSize: "0.92rem",
            outline: "none",
          }}
        />
        <span
          aria-hidden
          style={{
            position: "absolute",
            right: 10,
            top: "50%",
            transform: "translateY(-50%)",
            opacity: 0.7,
            pointerEvents: "none",
            display: "inline-flex",
            alignItems: "center",
            color: "currentColor",
          }}
        >
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="16"
            height="16"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <circle cx="11" cy="11" r="7" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
        </span>
      </div>
      {count && (
        <span style={{ opacity: 0.6, fontSize: "0.85rem", whiteSpace: "nowrap" }}>
          {count.shown} of {count.total}
        </span>
      )}
      {showClear && (
        <button
          type="button"
          className="btn btn-secondary btn-sm"
          onClick={onClear}
          title="Clear search, sort and column filters"
          style={{ height: 38 }}
        >
          Clear filters
        </button>
      )}
      <div style={{ marginLeft: "auto", display: "flex", gap: "0.75rem", alignItems: "center" }}>
        {rightControls}
      </div>
    </div>
  );
}
