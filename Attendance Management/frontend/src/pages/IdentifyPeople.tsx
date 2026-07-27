import { useCallback, useEffect, useMemo, useState } from "react";
import {
  liveIdentify as liveApi,
  cameras as camerasApi,
  employees as employeesApi,
  type LiveTrackRow,
  type AnalysisStatusRow,
} from "../api/client";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import CustomSelect from "../components/CustomSelect";
import { SectionLoader } from "../components/LoadingState";

/**
 * Identify People — click an "Unknown" box and say who it is.
 *
 * A ceiling camera resolves no faces, so Body Re-ID never gets the face match it
 * normally learns from and has nothing to compare against. Naming a person once
 * supplies that missing bootstrap: their current appearance is enrolled, and
 * Re-ID then carries the name on their track (and onto the other room camera)
 * for the rest of the day.
 *
 * Clothing changes daily, so this is a once-a-day action per person.
 * Labelling only — this never marks attendance.
 */

type CameraRow = { id: number; name: string; camera_purpose?: string };
type EmployeeRow = { id: number; full_name?: string; first_name?: string; last_name?: string };

const empName = (e: EmployeeRow) =>
  e.full_name || [e.first_name, e.last_name].filter(Boolean).join(" ") || `#${e.id}`;

export default function IdentifyPeople() {
  const [cameras, setCameras] = useState<CameraRow[]>([]);
  const [cameraId, setCameraId] = useState<string>("");
  const [emps, setEmps] = useState<EmployeeRow[]>([]);
  const [tracks, setTracks] = useState<LiveTrackRow[]>([]);
  const [imgUrl, setImgUrl] = useState("");
  const [msg, setMsg] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  const [picked, setPicked] = useState<LiveTrackRow | null>(null);
  const [pickedEmp, setPickedEmp] = useState("");
  const [saving, setSaving] = useState(false);
  const [analysis, setAnalysis] = useState<AnalysisStatusRow[]>([]);

  const flash = (m: string) => { setMsg(m); setTimeout(() => setMsg(""), 4000); };

  useEffect(() => {
    Promise.allSettled([camerasApi.list(), employeesApi.list({ status: "Active" })])
      .then(([c, e]) => {
        if (c.status === "fulfilled") {
          const list: CameraRow[] = c.value.data || [];
          const monitors = list.filter((x) => (x.camera_purpose || "").toUpperCase() === "MONITOR");
          const use = monitors.length ? monitors : list;
          setCameras(use);
          if (use[0]) setCameraId(String(use[0].id));
        }
        if (e.status === "fulfilled") setEmps(e.value.data || []);
      })
      .finally(() => setLoading(false));
  }, []);

  // Live picture + the boxes currently on it.
  const refresh = useCallback(() => {
    if (!cameraId) return;
    liveApi.snapshot(cameraId)
      .then((r) => setImgUrl((prev) => {
        if (prev) URL.revokeObjectURL(prev);
        return URL.createObjectURL(r.data as Blob);
      }))
      .catch(() => setError("This camera isn't sending a picture. Start it in Camera Manager."));
    liveApi.tracks(cameraId)
      .then((r) => { setTracks(r.data); setError(""); })
      .catch((e: any) =>
        setError(e?.response?.data?.detail || "Could not read the people on this camera."));
  }, [cameraId]);

  useEffect(() => { refresh(); }, [refresh, tick]);
  useEffect(() => {
    liveApi.analysisStatus().then((r) => setAnalysis(r.data)).catch(() => setAnalysis([]));
  }, [tick]);

  const togglePause = (cam: AnalysisStatusRow) => {
    liveApi.pauseAnalysis(cam.camera_id, !cam.paused)
      .then((r) => { flash(r.data.message); setTick((n) => n + 1); })
      .catch((e: any) => setError(e?.response?.data?.detail || "Could not change that camera"));
  };
  useEffect(() => {
    const t = setInterval(() => setTick((n) => n + 1), 4000);
    return () => clearInterval(t);
  }, []);

  const taken = useMemo(
    () => new Set(tracks.filter((t) => t.named && t.employee_id).map((t) => t.employee_id as number)),
    [tracks]
  );
  const empOptions = useMemo(
    () => emps.filter((e) => !taken.has(e.id) || String(e.id) === pickedEmp)
      .map((e) => ({ value: String(e.id), label: empName(e) })),
    [emps, taken, pickedEmp]
  );

  const save = () => {
    if (!picked || !pickedEmp) { setError("Pick a person and an employee."); return; }
    setSaving(true); setError("");
    liveApi.nameTrack({
      camera_id: Number(cameraId), track_id: picked.track_id, employee_id: Number(pickedEmp),
    })
      .then((r) => { flash(r.data.message); setPicked(null); setPickedEmp(""); setTick((n) => n + 1); })
      .catch((e: any) => setError(e?.response?.data?.detail || "Could not identify that person"))
      .finally(() => setSaving(false));
  };

  const unknownCount = tracks.filter((t) => !t.named).length;

  if (loading) return <div style={{ padding: "3rem 0" }}><SectionLoader size="md" /></div>;

  return (
    <div>
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="page-title">Identify People</h1>
          <div className="page-subtitle">Click an unknown person and say who they are — the camera remembers them for the day.</div>
        </div>
        <GlobalHeaderControls />
      </div>

      {msg && <div className="alert alert-success">{msg}</div>}
      {error && <div className="alert alert-error">{error}</div>}

      <div className="card" style={{ marginBottom: "1rem", padding: "1rem", display: "flex", gap: "1rem", alignItems: "flex-end", flexWrap: "wrap" }}>
        <div style={{ minWidth: 220 }}>
          <label style={{ display: "block", marginBottom: 4, fontSize: "0.7rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.04em", color: "#64748b" }}>
            Room camera
          </label>
          <CustomSelect
            value={cameraId}
            onChange={(v) => { setCameraId(String(v)); setPicked(null); }}
            options={cameras.map((c) => ({ value: String(c.id), label: `${c.name} (#${c.id})` }))}
            placeholder="Select camera"
          />
        </div>
        <button type="button" className="btn btn-secondary" onClick={() => setTick((n) => n + 1)}>Refresh</button>
        <div style={{ fontSize: "0.82rem", color: "rgba(255,255,255,0.6)" }}>
          {tracks.length} detected · <strong style={{ color: unknownCount ? "#f87171" : "#4ade80" }}>{unknownCount} unknown</strong>
        </div>
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "minmax(0,2fr) minmax(260px,1fr)", gap: "1rem", alignItems: "start" }}>
        <div className="card" style={{ padding: "0.75rem" }}>
          <div style={{ position: "relative", lineHeight: 0, background: "#0b0f19", borderRadius: 8, overflow: "hidden" }}>
            {imgUrl
              ? <img src={imgUrl} alt="camera" style={{ width: "100%", display: "block" }} draggable={false} />
              : <div style={{ width: "100%", paddingTop: "56%" }} />}

            {tracks.map((t) => {
              const [x1, y1, x2, y2] = t.box;
              const pc = (v: number, total: number) => `${(v / total) * 100}%`;
              const active = picked?.track_id === t.track_id;
              const colour = t.named ? "#4ade80" : (active ? "#60a5fa" : "#f87171");
              return (
                <div
                  key={t.track_id}
                  onClick={() => { if (!t.named) { setPicked(t); setPickedEmp(""); } }}
                  title={t.named ? t.label : "Click to identify this person"}
                  style={{
                    position: "absolute",
                    left: pc(x1, t.frame_w), top: pc(y1, t.frame_h),
                    width: pc(x2 - x1, t.frame_w), height: pc(y2 - y1, t.frame_h),
                    border: `2px solid ${colour}`,
                    background: active ? "rgba(96,165,250,0.20)" : "transparent",
                    cursor: t.named ? "default" : "pointer",
                    borderRadius: 3,
                  }}
                >
                  <span style={{
                    position: "absolute", top: -1, left: -1, background: colour,
                    color: "#08121f", fontSize: 10, fontWeight: 800, padding: "1px 5px",
                    borderRadius: "3px 0 3px 0", whiteSpace: "nowrap", lineHeight: 1.5,
                  }}>
                    {t.label}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>
          {picked ? (
            <div className="card" style={{ padding: "1rem" }}>
              <div style={{ fontWeight: 700, marginBottom: "0.75rem" }}>Who is {picked.label}?</div>
              <div className="form-group">
                <label>Employee</label>
                <CustomSelect
                  value={pickedEmp}
                  onChange={(v) => setPickedEmp(String(v))}
                  options={empOptions}
                  placeholder="Select employee"
                />
              </div>
              <div style={{ display: "flex", gap: "0.5rem" }}>
                <button type="button" className="btn btn-primary" onClick={save} disabled={saving}>
                  {saving ? "Saving…" : "Identify"}
                </button>
                <button type="button" className="btn btn-cancel-alt" onClick={() => { setPicked(null); setPickedEmp(""); }}>
                  Cancel
                </button>
              </div>
            </div>
          ) : (
            <div className="card" style={{ padding: "1rem", fontSize: "0.85rem", color: "rgba(255,255,255,0.6)" }}>
              Click a <span style={{ color: "#f87171", fontWeight: 700 }}>red</span> box on the picture to identify that person.
            </div>
          )}

          <div className="card" style={{ padding: "1rem" }}>
            <div style={{ fontWeight: 700, marginBottom: "0.5rem" }}>People on camera</div>
            {tracks.length === 0 ? (
              <p className="text-muted" style={{ fontSize: "0.85rem", margin: 0 }}>Nobody detected right now.</p>
            ) : (
              <div style={{ display: "flex", flexDirection: "column", gap: "0.35rem" }}>
                {tracks.map((t) => (
                  <div key={t.track_id} style={{
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                    padding: "0.45rem 0.6rem", borderRadius: 8, background: "rgba(255,255,255,0.04)",
                  }}>
                    <span style={{ fontSize: "0.85rem", fontWeight: 600, color: t.named ? "#4ade80" : "#f87171" }}>
                      {t.label}
                    </span>
                    {!t.named && (
                      <button type="button" className="btn btn-secondary btn-sm"
                        onClick={() => { setPicked(t); setPickedEmp(""); }}>
                        Identify
                      </button>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="card" style={{ padding: "1rem" }}>
            <div style={{ fontWeight: 700, marginBottom: "0.25rem" }}>Camera analysis (CPU)</div>
            <div style={{ fontSize: "0.72rem", color: "rgba(255,255,255,0.5)", marginBottom: "0.6rem", lineHeight: 1.5 }}>
              Each analysing camera costs ~2.3s of CPU per frame. Pausing a room camera frees CPU
              immediately — the video keeps playing, only the AI stops.
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: "0.35rem" }}>
              {analysis.map((c) => {
                const isAttendance = ["IN", "OUT"].includes((c.camera_purpose || "").toUpperCase());
                return (
                  <div key={c.camera_id} style={{
                    display: "flex", justifyContent: "space-between", alignItems: "center",
                    padding: "0.45rem 0.6rem", borderRadius: 8, background: "rgba(255,255,255,0.04)",
                  }}>
                    <div style={{ minWidth: 0 }}>
                      <div style={{ fontSize: "0.82rem", fontWeight: 600 }}>{c.name}</div>
                      <div style={{ fontSize: "0.68rem", color: c.paused ? "#fbbf24" : "#4ade80" }}>
                        {c.paused ? "Paused — no CPU used" : "Analysing"}
                        {isAttendance && " · attendance camera"}
                      </div>
                    </div>
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={() => {
                        if (!c.paused && isAttendance &&
                          !window.confirm(`${c.name} marks attendance. Pausing it will STOP attendance being recorded. Continue?`)) return;
                        togglePause(c);
                      }}
                    >
                      {c.paused ? "Resume" : "Pause"}
                    </button>
                  </div>
                );
              })}
            </div>
          </div>

          <div className="card" style={{ padding: "0.85rem", fontSize: "0.75rem", color: "rgba(255,255,255,0.55)", lineHeight: 1.6 }}>
            Identifying someone teaches the camera what they look like <strong>today</strong>. The
            name then follows them around the room — and onto the other room camera — even from
            behind.
            <br /><br />
            Because it learns their <em>appearance</em>, it must be repeated each day once
            they change clothes. This never marks attendance.
          </div>
        </div>
      </div>
    </div>
  );
}
