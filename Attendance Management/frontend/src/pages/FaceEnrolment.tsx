import { useCallback, useEffect, useState } from "react";
import { faceEnrolment, type CoverageEntry, type CoverageReport } from "../api/client";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import { SectionLoader } from "../components/LoadingState";

/**
 * Face Enrolment — what the cameras actually have to work with.
 *
 * Two things were invisible before this page:
 *
 *  1. The enrolment photos. `register_employee_face` wrote them to disk and
 *     derived an embedding, but nothing ever served them back, so nobody could
 *     see which employees had a face on file or what it looked like.
 *  2. Who is actually recognisable. An employee drops out of the matcher's
 *     gallery silently — a recognition-model change orphans their embeddings
 *     and the query simply stops selecting them. No error, no flag, and the
 *     enrolment screen still reports them as registered.
 *
 * The photo is a COVER image, not the gallery. Enrolment accepts up to 10
 * images and keeps an embedding for every one, but stores only the first as a
 * file — so "5 vectors · 1 photo stored" is expected, not a bug.
 */

type Status = "ok" | "warn";

const REASON_LABEL: Record<string, string> = {
  stale_model: "Wrong model",
  no_enrolment: "No face enrolled",
  no_stack: "Needs rebuild",
};

function initials(name: string): string {
  return name
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0]?.toUpperCase() ?? "")
    .join("");
}

function FaceCard({ entry, status }: { entry: CoverageEntry; status: Status }) {
  // `has_photo` only means a file existed when the report was built; the image
  // request can still 404. Fall back to initials rather than a broken image.
  const [broken, setBroken] = useState(false);
  const showPhoto = entry.has_photo && !broken;
  const accent = status === "ok" ? "#22c55e" : "#f59e0b";

  return (
    <div
      className="card"
      style={{
        padding: 0,
        overflow: "hidden",
        display: "flex",
        flexDirection: "column",
        borderTop: `3px solid ${accent}`,
      }}
    >
      <div
        style={{
          aspectRatio: "1 / 1",
          background: "#1f2937",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        {showPhoto ? (
          <img
            src={faceEnrolment.photoUrl(entry.employee_id)}
            alt={`Enrolment photo for ${entry.name}`}
            onError={() => setBroken(true)}
            style={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        ) : (
          <span style={{ fontSize: "2rem", fontWeight: 600, color: "#9ca3af" }}>
            {initials(entry.name) || "?"}
          </span>
        )}
      </div>

      <div style={{ padding: "12px 14px", display: "grid", gap: 6 }}>
        <strong style={{ lineHeight: 1.2 }} title={entry.name}>
          {entry.name}
        </strong>
        <span className="text-muted" style={{ fontSize: ".8rem" }}>
          {entry.employee_code ? `#${entry.employee_code}` : `ID ${entry.employee_id}`}
        </span>

        <span
          style={{
            display: "inline-block",
            width: "fit-content",
            padding: "2px 8px",
            borderRadius: 999,
            fontSize: ".75rem",
            fontWeight: 600,
            color: accent,
            border: `1px solid ${accent}`,
          }}
        >
          {status === "ok"
            ? `Recognisable · ${entry.model_matched} vector${entry.model_matched === 1 ? "" : "s"}`
            : REASON_LABEL[entry.reason ?? ""] ?? entry.reason ?? "Excluded"}
        </span>

        {status === "warn" && entry.detail && (
          <span className="text-muted" style={{ fontSize: ".75rem", lineHeight: 1.35 }}>
            {entry.detail}
          </span>
        )}

        {status === "ok" && (
          <span className="text-muted" style={{ fontSize: ".75rem" }}>
            {entry.has_photo ? "1 photo stored" : "no photo stored"}
          </span>
        )}
      </div>
    </div>
  );
}

function Stat({ label, value, tone }: { label: string; value: string; tone?: string }) {
  return (
    <div className="card" style={{ padding: "14px 18px", minWidth: 150 }}>
      <div style={{ fontSize: "1.6rem", fontWeight: 700, color: tone }}>{value}</div>
      <div className="text-muted" style={{ fontSize: ".8rem" }}>
        {label}
      </div>
    </div>
  );
}

const GRID: React.CSSProperties = {
  display: "grid",
  gridTemplateColumns: "repeat(auto-fill, minmax(190px, 1fr))",
  gap: 14,
};

export default function FaceEnrolment() {
  const [report, setReport] = useState<CoverageReport | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = useCallback(() => {
    setLoading(true);
    setError("");
    faceEnrolment
      .coverage()
      .then((r) => setReport(r.data))
      .catch(() => setError("Could not load recognition coverage."))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const excluded = report?.excluded ?? [];
  const recognisable = report?.recognisable ?? [];

  return (
    <div className="eds">
      <header className="eds-topbar">
        <div>
          <h1 className="eds-title">Face Enrolment</h1>
          <p className="eds-subtitle">
            Enrolled photos, and which employees the cameras can actually recognise.
          </p>
        </div>
        <GlobalHeaderControls />
      </header>

      <div className="eds-page">
        <div className="eds-actionbar">
          <button type="button" className="eds-action eds-action--info" onClick={load}>
            Refresh
          </button>
        </div>

        {error && <div className="alert alert-error">{error}</div>}

        {loading ? (
          <SectionLoader />
        ) : !report ? null : (
          <>
            <div style={{ display: "flex", gap: 12, flexWrap: "wrap", marginBottom: 20 }}>
              <Stat
                label="Recognisable"
                value={`${report.in_gallery} / ${report.active_employees}`}
                tone={report.in_gallery === report.active_employees ? "#22c55e" : "#f59e0b"}
              />
              <Stat label="Coverage" value={`${report.coverage_pct}%`} />
              <Stat label="Gallery vectors" value={String(report.gallery_vectors)} />
            </div>

            <p className="text-muted" style={{ fontSize: ".8rem", marginTop: 0 }}>
              Recognition model: <code>{report.model_version}</code>. Only employees
              enrolled under this model can be matched — vectors produced by a
              different model are not comparable and are ignored.
            </p>

            {excluded.length > 0 && (
              <>
                <h2 style={{ fontSize: "1rem", marginBottom: 10 }}>
                  Needs attention ({excluded.length})
                </h2>
                <div style={{ ...GRID, marginBottom: 28 }}>
                  {excluded.map((e) => (
                    <FaceCard key={e.employee_id} entry={e} status="warn" />
                  ))}
                </div>
              </>
            )}

            <h2 style={{ fontSize: "1rem", marginBottom: 10 }}>
              Recognisable ({recognisable.length})
            </h2>
            {recognisable.length === 0 ? (
              <p className="text-muted">
                No employee is currently recognisable — the cameras cannot identify anyone.
              </p>
            ) : (
              <div style={GRID}>
                {recognisable.map((e) => (
                  <FaceCard key={e.employee_id} entry={e} status="ok" />
                ))}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
