/**
 * Feed layout selector — show 1, 2 or 4 camera feeds at once.
 *
 * WHY THIS CAPS AT 4, AND WHY 1 IS THE DEFAULT
 * --------------------------------------------
 * Each live feed is an MJPEG stream: a `multipart/x-mixed-replace` response
 * that never completes. Two consequences drove the original "one camera at a
 * time" rule in DvrCameraDashboard, and both still apply:
 *
 *   1. BROWSER CONNECTION LIMIT. HTTP/1.1 allows ~6 concurrent connections per
 *      host. Every open feed holds one for as long as it is on screen. At 4
 *      feeds the status poll and every other API call share the remaining two;
 *      go beyond that and they queue behind streams that never finish, which is
 *      what used to stall the whole page.
 *
 *   2. SERVER COST. Each feed makes the backend encode a full-frame JPEG per
 *      frame, on a 4-core box whose CPU is already the binding constraint on
 *      detection latency. Four feeds is four encoders competing with the
 *      recognition threads.
 *
 * So this is a real trade, not a free layout preference: more tiles means a
 * slower pipeline behind them. 1 stays the default so the cost is opt-in, and 4
 * is the ceiling because at 5+ the page starves itself.
 */
export type FeedLayout = 1 | 2 | 4;

export const FEED_LAYOUTS: FeedLayout[] = [1, 2, 4];

/** Grid template for a layout. 2 and 4 both use two columns; 4 wraps to two rows. */
export function feedGridColumns(layout: FeedLayout): string {
  return layout === 1 ? "1fr" : "repeat(2, minmax(0, 1fr))";
}

/**
 * Height for each tile, chosen so the whole grid fits one screen without
 * scrolling. A single feed gets the tall box the page has always used; two
 * sit side by side at the same height; four halve it to fit two rows.
 */
export function feedTileHeight(layout: FeedLayout): string {
  if (layout === 1) return "78vh";
  if (layout === 2) return "62vh";
  return "38vh";
}

export default function FeedLayoutPicker({
  value,
  onChange,
  disabled = false,
}: {
  value: FeedLayout;
  onChange: (next: FeedLayout) => void;
  disabled?: boolean;
}) {
  return (
    <div
      role="group"
      aria-label="Number of camera feeds to show at once"
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 2,
        padding: 3,
        borderRadius: 10,
        background: "rgba(255,255,255,0.04)",
        border: "1px solid rgba(255,255,255,0.09)",
        opacity: disabled ? 0.5 : 1,
      }}
    >
      <span
        style={{
          fontSize: "0.72rem",
          fontWeight: 600,
          letterSpacing: "0.04em",
          textTransform: "uppercase",
          color: "rgba(255,255,255,0.5)",
          padding: "0 8px 0 6px",
        }}
      >
        Feeds
      </span>
      {FEED_LAYOUTS.map((n) => {
        const active = value === n;
        return (
          <button
            key={n}
            type="button"
            disabled={disabled}
            aria-pressed={active}
            onClick={() => onChange(n)}
            title={
              n === 1
                ? "Show one feed — lowest CPU and network cost"
                : `Show ${n} feeds at once — ${n} simultaneous streams, which costs ${n}x the JPEG encoding on the server`
            }
            style={{
              minWidth: 38,
              padding: "0.34rem 0.6rem",
              borderRadius: 7,
              border: "none",
              cursor: disabled ? "not-allowed" : "pointer",
              fontSize: "0.82rem",
              fontWeight: 700,
              fontVariantNumeric: "tabular-nums",
              transition: "background 160ms ease, color 160ms ease",
              background: active ? "rgba(122,162,255,0.22)" : "transparent",
              color: active ? "#cfe0ff" : "rgba(255,255,255,0.62)",
              boxShadow: active ? "inset 0 0 0 1px rgba(122,162,255,0.38)" : "none",
            }}
          >
            {n}
          </button>
        );
      })}
    </div>
  );
}
