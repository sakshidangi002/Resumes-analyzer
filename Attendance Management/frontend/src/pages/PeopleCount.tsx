import { useCallback, useEffect, useRef, useState } from "react";
import { peopleCount, type CameraPeopleRow, type PeopleCountReport } from "../api/client";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import { SectionLoader } from "../components/LoadingState";

/**
 * Live people count — how many bodies each camera can see right now.
 *
 * This counts BODY tracks, not faces. A person walking away from the lens has
 * no visible face, so the face pipeline reports nothing for them — which is
 * indistinguishable from an empty room. Counting bodies separately is the only
 * way to answer "is anyone there?" independently of "who is it?".
 *
 * Two things this screen refuses to fake:
 *
 *  1. The total is NOT occupancy. These cameras overlap, so somebody standing
 *     where two views meet is counted twice. Deduplicating would need
 *     cross-camera Re-ID, which the pipeline does not attempt, so the number is
 *     labelled for what it actually is.
 *  2. A count is only as fresh as the last completed analysis pass, and a pass
 *     costs seconds on this hardware. Stale counts are greyed and stamped with
 *     their age instead of being shown as current — presenting a several-second
 *     old zero as fact is exactly how a plainly visible person came to read as
 *     "People: 0".
 */

const POLL_MS = 2000;

// Past this, the number on screen is old enough that a person could have walked
// in and out since it was measured, so it is shown as stale rather than live.
const STALE_AFTER_SEC = 5;

function ageLabel(sec: number | null): string {
  if (sec == null) return "not yet analysed";
  if (sec < 1) return "just now";
  if (sec < 60) return `${sec.toFixed(0)}s ago`;
  return `${Math.floor(sec / 60)}m ago`;
}

function CameraCard({ row }: { row: CameraPeopleRow }) {
  const stale = row.analysis_age_sec == null || row.analysis_age_sec > STALE_AFTER_SEC;
  const offline = row.status !== "running";
  const untracked = !row.body_tracking;

  let accent = "#22c55e";
  if (offline) accent = "#ef4444";
  else if (untracked || stale) accent = "#f59e0b";

  return (
    <div
      className="card"
      style={{ padding: "16px 18px", borderLeft: `3px solid ${accent}`, display: "grid", gap: 8 }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 10 }}>
        <strong title={row.name}>{row.name}</strong>
        <span className="text-muted" style={{ fontSize: ".72rem" }}>{row.purpose}</span>
      </div>

      <div style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
        <span
          style={{
            fontSize: "2.4rem",
            fontWeight: 700,
            lineHeight: 1,
            // A stale or untracked figure must not read as a confident live one.
            color: untracked || offline ? "#6b7280" : stale ? "#9ca3af" : accent,
          }}
        >
          {untracked || offline ? "—" : row.people}
        </span>
        <span className="text-muted" style={{ fontSize: ".8rem" }}>
          {untracked || offline ? "" : row.people === 1 ? "person" : "people"}
        </span>
      </div>

      <span className="text-muted" style={{ fontSize: ".74rem" }}>
        {offline
          ? `camera ${row.status}`
          : untracked
          ? "body tracking off — faces only, so people are not counted here"
          : stale
          ? `stale · analysed ${ageLabel(row.analysis_age_sec)}`
          : `live · ${ageLabel(row.analysis_age_sec)}`}
      </span>
    </div>
  );
}

export default function PeopleCount() {
  const [report, setReport] = useState<PeopleCountReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const timer = useRef<number | null>(null);

  const load = useCallback(async () => {
    try {
      const r = await peopleCount.get();
      setReport(r.data);
      setError("");
    } catch {
      setError("Could not read camera stats.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
    timer.current = window.setInterval(load, POLL_MS);
    return () => {
      if (timer.current) window.clearInterval(timer.current);
    };
  }, [load]);

  const rows = report?.people_by_camera ?? [];
  const tracked = rows.filter((r) => r.body_tracking && r.status === "running");
  const anyStale = tracked.some(
    (r) => r.analysis_age_sec == null || r.analysis_age_sec > STALE_AFTER_SEC
  );

  return (
    <div className="eds">
      <header className="eds-topbar">
        <div>
          <h1 className="eds-title">People Count</h1>
          <p className="eds-subtitle">
            Bodies detected on each camera, whether or not a face is visible.
          </p>
        </div>
        <GlobalHeaderControls />
      </header>

      <div className="eds-page">
        {error && <div className="alert alert-error">{error}</div>}

        {loading ? (
          <SectionLoader />
        ) : !report ? null : (
          <>
            <div className="card" style={{ padding: "18px 22px", marginBottom: 18 }}>
              <div style={{ display: "flex", alignItems: "baseline", gap: 14, flexWrap: "wrap" }}>
                <span style={{ fontSize: "3rem", fontWeight: 700, lineHeight: 1 }}>
                  {report.people_detected}
                </span>
                <div style={{ display: "grid", gap: 2 }}>
                  <span style={{ fontWeight: 600 }}>detected across {tracked.length} camera(s)</span>
                  <span className="text-muted" style={{ fontSize: ".76rem" }}>
                    Not a headcount — the cameras overlap, so a person visible in
                    two views is counted twice.
                  </span>
                </div>
              </div>
              {anyStale && (
                <p
                  className="text-muted"
                  style={{ fontSize: ".76rem", marginTop: 12, marginBottom: 0, color: "#f59e0b" }}
                >
                  Some cameras have not completed an analysis pass in the last{" "}
                  {STALE_AFTER_SEC}s — those counts are behind the live picture.
                </p>
              )}
            </div>

            {rows.length === 0 ? (
              <p className="text-muted">No cameras are running.</p>
            ) : (
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(auto-fill, minmax(230px, 1fr))",
                  gap: 14,
                }}
              >
                {rows.map((r) => (
                  <CameraCard key={String(r.camera_id)} row={r} />
                ))}
              </div>
            )}

            <p className="text-muted" style={{ fontSize: ".74rem", marginTop: 20 }}>
              Refreshes every {POLL_MS / 1000}s. Counts come from body detection
              (YOLO + ByteTrack), so they are independent of face recognition —
              an unrecognised or unenrolled person still counts.
            </p>
          </>
        )}
      </div>
    </div>
  );
}
