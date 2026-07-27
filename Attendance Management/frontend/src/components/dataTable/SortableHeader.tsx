import type { CSSProperties, ReactNode } from "react";
import type { SortState } from "./useTableControls";

interface SortableHeaderProps {
  label: ReactNode;
  columnKey: string;
  sort: SortState;
  onToggle: (key: string) => void;
  align?: "left" | "center" | "right";
  className?: string;
  style?: CSSProperties;
  /** Hide the sort affordance (use for action columns). */
  notSortable?: boolean;
  title?: string;
}

export default function SortableHeader({
  label,
  columnKey,
  sort,
  onToggle,
  align = "left",
  className,
  style,
  notSortable,
  title,
}: SortableHeaderProps) {
  const active = sort.key === columnKey;
  const dir = active ? sort.direction : null;
  const arrow = !active ? "↕" : dir === "asc" ? "↑" : "↓";

  if (notSortable) {
    return (
      <th
        className={className}
        style={{ textAlign: align, ...style }}
        title={title}
      >
        {label}
      </th>
    );
  }

  return (
    <th
      className={className}
      style={{
        textAlign: align,
        cursor: "pointer",
        userSelect: "none",
        ...style,
      }}
      onClick={() => onToggle(columnKey)}
      title={title ?? `Sort by ${typeof label === "string" ? label : "column"}`}
    >
      <span
        style={{
          display: "inline-flex",
          alignItems: "center",
          gap: 6,
          width: "100%",
          justifyContent: align === "center" ? "center" : align === "right" ? "flex-end" : "flex-start",
        }}
      >
        <span>{label}</span>
        <span
          aria-hidden
          style={{
            fontSize: "0.8em",
            opacity: active ? 1 : 0.35,
            color: active ? "#60a5fa" : "inherit",
            transition: "opacity 120ms ease, color 120ms ease",
          }}
        >
          {arrow}
        </span>
      </span>
    </th>
  );
}
