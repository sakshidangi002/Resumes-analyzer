import { useEffect, useRef, useState } from "react";
import { NavLink } from "react-router-dom";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import { recognition as recognitionApi, cameras as camerasApi } from "../api/client";

type FacePayload = {
  box: number[];
  confidence?: number;
  employee_id: number | null;
  employee_code?: string | null;
  employee_name: string;
  matched: boolean;
  score?: number;
  runner_up_score?: number;
  margin?: number;
  state: string;
};

type AttendanceSummary = {
  employee_id: number;
  employee_code?: string | null;
  employee_name: string;
  date: string;
  sign_in_time: string | null;
  sign_out_time: string | null;
  total_work_hours?: number | null;
  total_break_hours?: number | null;
  status: string;
  event_type?: string | null;
};

type RecognitionResult = {
  status: boolean;
  message: string;
  source: string;
  faces: FacePayload[];
  attendance?: AttendanceSummary | null;
};

function formatTime(value: string | null): string {
  if (!value) return "-";
  const parts = value.split(":");
  if (parts.length < 2) return value;
  const h = Number(parts[0]);
  const m = parts[1];
  const ampm = h >= 12 ? "PM" : "AM";
  const h12 = h % 12 || 12;
  return String(h12) + ":" + m + " " + ampm;
}

function formatHours(hours: number | null | undefined): string {
  if (hours == null) return "-";
  const totalMinutes = Math.round(Number(hours) * 60);
  const h = Math.floor(totalMinutes / 60);
  const m = totalMinutes % 60;
  if (h <= 0) return String(m) + "m";
  if (m <= 0) return String(h) + "h";
  return String(h) + "h " + String(m) + "m";
}

function formatEmployeeLabel(face: FacePayload): string {
  if (!face.matched) return "Unknown Person";
  const idLabel = face.employee_code || (face.employee_id == null ? "N/A" : String(face.employee_id));
  return face.employee_name + " | " + idLabel;
}

function eventLabel(eventType: string | null | undefined): string {
  switch ((eventType || "").toUpperCase()) {
    case "IN":
    case "CHECK_IN":
      return "Check-in";
    case "OUT":
    case "CHECK_OUT":
      return "Check-out";
    case "BREAK_OUT":
      return "Break out";
    case "BREAK_IN":
      return "Break in";
    default:
      return eventType || "Unknown";
  }
}

function eventTone(eventType: string | null | undefined): string {
  switch ((eventType || "").toUpperCase()) {
    case "OUT":
    case "CHECK_OUT":
    case "BREAK_OUT":
      return "#f87171";
    case "BREAK_IN":
    case "IN":
    case "CHECK_IN":
      return "#22c55e";
    default:
      return "#94a3b8";
  }
}

function rgba(hex: string, alpha: number): string {
  const clean = hex.replace("#", "");
  const value = clean.length === 3
    ? clean.split("").map((ch) => ch + ch).join("")
    : clean.padEnd(6, "0").slice(0, 6);
  const r = Number.parseInt(value.slice(0, 2), 16);
  const g = Number.parseInt(value.slice(2, 4), 16);
  const b = Number.parseInt(value.slice(4, 6), 16);
  return "rgba(" + r + ", " + g + ", " + b + ", " + alpha + ")";
}

function statusColor(status: string): string {
  switch (status.toUpperCase()) {
    case "PRESENT": return "#22c55e";
    case "HALF_DAY": return "#f59e0b";
    case "SHORT": return "#f59e0b";
    case "ABSENT": return "#ef4444";
    case "WEEKLY_OFF": return "#6366f1";
    case "HOLIDAY": return "#06b6d4";
    default: return "#94a3b8";
  }
}

const panelStyle: React.CSSProperties = {
  padding: "1rem",
  borderRadius: 18,
  background: "rgba(255,255,255,0.06)",
  border: "1px solid rgba(255,255,255,0.08)",
};

export default function CctvAttendance() {
  const scanTimerRef = useRef<number | null>(null);
  const busyRef = useRef(false);
  const [streamUrl, setStreamUrl] = useState("");
  const [cameraId, setCameraId] = useState("gate-1");
  const [cameraType, setCameraType] = useState("IN");
  const [threshold, setThreshold] = useState("0.45");
  const [scanInterval, setScanInterval] = useState("10");
  const [autoScan, setAutoScan] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [scanStatus, setScanStatus] = useState("Enter a CCTV stream URL and run a scan.");
  const [attendanceStatus, setAttendanceStatus] = useState("No attendance recorded yet.");
  const [lastScanAt, setLastScanAt] = useState("-");
  const [latestFaces, setLatestFaces] = useState<FacePayload[]>([]);
  const [lastEmployee, setLastEmployee] = useState({
    name: "",
    id: null as number | null,
    code: "",
    firstIn: null as string | null,
    lastOut: null as string | null,
    workHours: null as number | null,
    breakHours: null as number | null,
    status: "",
    lastEventType: null as string | null,
    lastEventTime: "",
  });

  // ── DB camera list ──────────────────────────────────────────────────────
  type DbCamera = {
    id: number;
    name: string;
    location: string | null;
    stream_url: string;  // API returns stream_url (mapped from source_url)
    camera_purpose: string;
    threshold: number;
    interval_sec: number;
    enabled: boolean;
    live: { status: string; fps: number } | null;
  };
  const [dbCameras, setDbCameras] = useState<DbCamera[]>([]);
  const [selectedCamId, setSelectedCamId] = useState<number | "">("");
  const [previewTs, setPreviewTs] = useState(Date.now());
  const previewTimer = useRef<number | null>(null);

  useEffect(() => {
    void camerasApi.list().then((r) => {
      const payload = r.data as { cameras?: DbCamera[] } | DbCamera[];
      const list = (Array.isArray(payload) ? payload : payload.cameras ?? []).filter((c) => c.enabled);
      setDbCameras(list);
      if (list.length > 0 && selectedCamId === "") {
        const first = list[0];
        setSelectedCamId(first.id);
        setStreamUrl(first.stream_url);
        setCameraId(String(first.id));
        setCameraType(first.camera_purpose);
        setThreshold(String(first.threshold));
      }
    }).catch((err) => {
      console.error("Failed to load CCTV cameras", err);
      /* cameras API optional */
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // refresh preview every 2s when a camera is selected
  useEffect(() => {
    if (selectedCamId !== "") {
      previewTimer.current = window.setInterval(() => setPreviewTs(Date.now()), 2000);
    } else {
      if (previewTimer.current) window.clearInterval(previewTimer.current);
    }
    return () => { if (previewTimer.current) window.clearInterval(previewTimer.current); };
  }, [selectedCamId]);

  const handleCamSelect = (id: number | "") => {
    setSelectedCamId(id);
    if (id === "") { setStreamUrl(""); setCameraId(""); return; }
    const cam = dbCameras.find((c) => c.id === id);
    if (!cam) return;
    setStreamUrl(cam.stream_url);
    setCameraId(String(cam.id));
    setCameraType(cam.camera_purpose);
    setThreshold(String(cam.threshold));
    setScanInterval(String(cam.interval_sec));
  };

  const stopScanLoop = () => {
    if (scanTimerRef.current != null) {
      window.clearInterval(scanTimerRef.current);
      scanTimerRef.current = null;
    }
  };

  const mergeAttendance = (att: AttendanceSummary) => {
    setLastEmployee((prev) => ({
      name: att.employee_name,
      id: att.employee_id,
      code: att.employee_code || String(att.employee_id),
      firstIn: prev.firstIn && att.sign_in_time ? prev.firstIn : (att.sign_in_time ?? prev.firstIn),
      lastOut: att.sign_out_time ?? prev.lastOut,
      workHours: att.total_work_hours ?? prev.workHours,
      breakHours: att.total_break_hours ?? prev.breakHours,
      status: att.status ?? prev.status,
      lastEventType: att.event_type ?? prev.lastEventType,
      lastEventTime: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }),
    }));
  };

  const runScan = async () => {
    if (busyRef.current) return;
    if (!streamUrl.trim()) {
      setCameraError("Enter a CCTV stream URL first.");
      return;
    }

    busyRef.current = true;
    setCameraError("");

    try {
      const response = await recognitionApi.recognizeCctvFrame({
        stream_url: streamUrl.trim(),
        threshold: Number(threshold),
        camera_id: cameraId.trim() || null,
        camera_type: cameraType,
      });
      const data = response.data as RecognitionResult;
      const faces = data.faces || [];
      setLatestFaces(faces);
      setScanStatus(data.message || "Scan complete.");
      setLastScanAt(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));

      const attendance = data.attendance;
      const knownFaces = faces.filter((face) => face.matched);
      if (attendance) {
        mergeAttendance(attendance);
        setAttendanceStatus(eventLabel(attendance.event_type) + " recorded for " + attendance.employee_name + ".");
      } else if (knownFaces.length > 0) {
        setAttendanceStatus(knownFaces[0].employee_name + " recognized. No attendance row was returned.");
      } else if (faces.length === 0) {
        setAttendanceStatus("No face detected in the CCTV frame.");
      } else {
        setAttendanceStatus("Unknown person detected.");
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "CCTV recognition failed.";
      setCameraError(message);
      setScanStatus("Scan failed.");
    } finally {
      busyRef.current = false;
    }
  };

  useEffect(() => {
    if (!autoScan) {
      stopScanLoop();
      return;
    }

    const intervalSeconds = Number(scanInterval) || 10;
    const intervalMs = Math.max(5, intervalSeconds) * 1000;
    stopScanLoop();
    scanTimerRef.current = window.setInterval(() => {
      void runScan();
    }, intervalMs);
    void runScan();

    return () => {
      stopScanLoop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoScan, scanInterval, streamUrl, threshold, cameraId, cameraType]);

  useEffect(() => {
    return () => {
      stopScanLoop();
    };
  }, []);

  const knownCount = latestFaces.filter((face) => face.matched).length;
  const unknownCount = latestFaces.length - knownCount;
  const bestConfidence = latestFaces.reduce((max, face) => {
    const score = typeof face.score === "number" ? face.score : 0;
    return score > max ? score : max;
  }, 0);
  const hasAttendanceSummary = lastEmployee.name && (lastEmployee.firstIn || lastEmployee.lastOut);

  return (
    <div className="page-stack">
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "1rem", flexWrap: "wrap" }}>
        <div>
          <h1 className="page-title">CCTV Attendance</h1>
          <div className="page-subtitle">Read one frame from a CCTV or IP camera stream and mark attendance automatically.</div>
        </div>
        <GlobalHeaderControls />
      </div>

      <section className="card" style={{ marginTop: "1rem", padding: "1.4rem", border: "none", background: "linear-gradient(135deg, rgba(9,14,31,0.98), rgba(15,23,42,0.94))", color: "#fff" }}>
        <div style={{ display: "grid", gap: "1rem", gridTemplateColumns: "minmax(0, 1.4fr) minmax(280px, 0.6fr)" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.65rem", flexWrap: "wrap" }}>
              <span style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem", padding: "0.42rem 0.75rem", borderRadius: 999, background: "rgba(34,197,94,0.12)", border: "1px solid rgba(34,197,94,0.22)", color: "#d3f9d8", fontSize: "0.82rem", fontWeight: 700 }}>Backend camera capture</span>
              <span style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem", padding: "0.42rem 0.75rem", borderRadius: 999, background: "rgba(96,165,250,0.12)", border: "1px solid rgba(96,165,250,0.22)", color: "#dbeafe", fontSize: "0.82rem", fontWeight: 700 }}>Attendance API linked</span>
              <span style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem", padding: "0.42rem 0.75rem", borderRadius: 999, background: "rgba(245,158,11,0.12)", border: "1px solid rgba(245,158,11,0.22)", color: "#fef3c7", fontSize: "0.82rem", fontWeight: 700 }}>60 s duplicate guard</span>
            </div>

            <div style={{ marginTop: "1.1rem", display: "grid", gap: "0.9rem", gridTemplateColumns: "repeat(auto-fit, minmax(210px, 1fr))" }}>
              <label style={{ display: "grid", gap: "0.35rem", color: "rgba(255,255,255,0.72)" }}>
                Camera
                {dbCameras.length > 0 ? (
                  <select
                    value={selectedCamId}
                    onChange={(e) => handleCamSelect(e.target.value === "" ? "" : Number(e.target.value))}
                    style={{ width: "100%", borderRadius: 10, border: "1px solid rgba(255,255,255,0.15)", padding: "0.8rem", background: "#0b1220", color: "#fff" }}
                  >
                    <option value="">— Select camera —</option>
                    {dbCameras.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.name} ({c.camera_purpose === "IN" ? "Check-In" : "Check-Out"})
                        {c.live?.status === "running" ? " ✓" : " ⚠"}
                      </option>
                    ))}
                  </select>
                ) : (
                  <input type="text" value={streamUrl} onChange={(e) => setStreamUrl(e.target.value)} placeholder="rtsp://user:pass@192.168.1.20:554/Streaming/Channels/101" style={{ width: "100%", borderRadius: 10, border: "1px solid rgba(255,255,255,0.15)", padding: "0.8rem", background: "#0b1220", color: "#fff" }} />
                )}
              </label>
              <label style={{ display: "grid", gap: "0.35rem", color: "rgba(255,255,255,0.72)" }}>
                Camera ID
                <input type="text" value={cameraId} onChange={(e) => setCameraId(e.target.value)} placeholder="gate-1" style={{ width: "100%", borderRadius: 10, border: "1px solid rgba(255,255,255,0.15)", padding: "0.8rem", background: "#0b1220", color: "#fff" }} />
              </label>
              <label style={{ display: "grid", gap: "0.35rem", color: "rgba(255,255,255,0.72)" }}>
                Camera Purpose
                <select value={cameraType} onChange={(e) => setCameraType(e.target.value)} style={{ width: "100%", borderRadius: 10, border: "1px solid rgba(255,255,255,0.15)", padding: "0.8rem", background: "#0b1220", color: "#fff" }}>
                  <option value="IN">Check-in (IN)</option>
                  <option value="OUT">Check-out (OUT)</option>
                  <option value="BREAK_OUT">Break-out (BREAK_OUT)</option>
                  <option value="BREAK_IN">Break-in (BREAK_IN)</option>
                </select>
              </label>
              <label style={{ display: "grid", gap: "0.35rem", color: "rgba(255,255,255,0.72)" }}>
                Match threshold
                <input type="number" step="0.01" min="0" max="1" value={threshold} onChange={(e) => setThreshold(e.target.value)} style={{ width: "100%", borderRadius: 10, border: "1px solid rgba(255,255,255,0.15)", padding: "0.8rem", background: "#0b1220", color: "#fff" }} />
              </label>
              <label style={{ display: "grid", gap: "0.35rem", color: "rgba(255,255,255,0.72)" }}>
                Auto-scan interval (seconds)
                <input type="number" min="5" step="1" value={scanInterval} onChange={(e) => setScanInterval(e.target.value)} style={{ width: "100%", borderRadius: 10, border: "1px solid rgba(255,255,255,0.15)", padding: "0.8rem", background: "#0b1220", color: "#fff" }} />
              </label>
            </div>

            <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", marginTop: "1.1rem" }}>
              <button className="btn btn-primary" type="button" onClick={() => void runScan()}>Scan once</button>
              <button className="btn btn-secondary" type="button" onClick={() => setAutoScan(true)}>Start auto-scan</button>
              <button className="btn btn-secondary" type="button" onClick={() => setAutoScan(false)}>Stop auto-scan</button>
              <NavLink className="btn btn-secondary" to="/attendance">Review Attendance</NavLink>
              <NavLink className="btn btn-secondary" to="/face-detection">Webcam Mode</NavLink>
              <NavLink className="btn btn-secondary" to="/cctv-cameras">📷 Camera Manager</NavLink>
            </div>
          </div>

          <div>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.85rem" }}>
              <div style={panelStyle}>
                <div style={{ fontSize: "0.8rem", color: "rgba(255,255,255,0.65)" }}>Scan state</div>
                <div style={{ fontSize: "1.05rem", fontWeight: 800, marginTop: 4 }}>{autoScan ? "Auto-scan" : "Idle"}</div>
              </div>
              <div style={panelStyle}>
                <div style={{ fontSize: "0.8rem", color: "rgba(255,255,255,0.65)" }}>Known faces</div>
                <div style={{ fontSize: "1.05rem", fontWeight: 800, marginTop: 4 }}>{knownCount}</div>
              </div>
              <div style={panelStyle}>
                <div style={{ fontSize: "0.8rem", color: "rgba(255,255,255,0.65)" }}>Unknown faces</div>
                <div style={{ fontSize: "1.05rem", fontWeight: 800, marginTop: 4 }}>{unknownCount}</div>
              </div>
              <div style={panelStyle}>
                <div style={{ fontSize: "0.8rem", color: "rgba(255,255,255,0.65)" }}>Last scan</div>
                <div style={{ fontSize: "1.05rem", fontWeight: 800, marginTop: 4 }}>{lastScanAt}</div>
              </div>
            </div>

            <div style={{ marginTop: "1rem", ...panelStyle }}>
              <div style={{ fontSize: "0.82rem", color: "rgba(255,255,255,0.65)", marginBottom: "0.35rem" }}>Live preview</div>
              {selectedCamId !== "" ? (
                <img
                  src={`${camerasApi.previewUrl(selectedCamId as number)}?t=${previewTs}`}
                  alt="Camera preview"
                  style={{ width: "100%", borderRadius: 10, maxHeight: 180, objectFit: "cover", background: "#000" }}
                  onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                />
              ) : (
                <div style={{ color: "rgba(255,255,255,0.4)", fontSize: "0.85rem" }}>Select a camera to see live preview.</div>
              )}
            </div>
          </div>
        </div>
      </section>

      {cameraError ? (
        <div style={{ marginTop: "1rem", padding: "0.9rem 1rem", borderRadius: 16, background: "rgba(239,68,68,0.14)", border: "1px solid rgba(239,68,68,0.3)", color: "#fecaca" }}>
          {cameraError}
        </div>
      ) : null}

      <section className="card" style={{ marginTop: "1rem", padding: "1.1rem", borderRadius: 22, background: "linear-gradient(180deg, #111827 0%, #0b1220 100%)", color: "#fff", border: "1px solid rgba(255,255,255,0.08)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap", alignItems: "center" }}>
          <div>
            <div style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem", padding: "0.42rem 0.75rem", borderRadius: 999, background: "rgba(122,162,255,0.12)", border: "1px solid rgba(122,162,255,0.18)", color: "#cfe0ff", fontSize: "0.82rem", fontWeight: 700 }}>Recognition output</div>
            <h3 style={{ margin: "0.8rem 0 0", fontSize: "1.45rem" }}>Latest CCTV scan</h3>
          </div>
          <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", alignItems: "center", color: "rgba(255,255,255,0.72)" }}>
            <span>Stream: {streamUrl ? "configured" : "not set"}</span>
            <span>Camera: {dbCameras.find((c) => c.id === selectedCamId)?.name || cameraId || "-"}</span>
            <span>Type: {eventLabel(cameraType)}</span>
          </div>
        </div>

        <div style={{ display: "grid", gap: "1rem", gridTemplateColumns: "minmax(0, 1.35fr) minmax(280px, 0.65fr)", marginTop: "1rem" }}>
          <div style={{ display: "grid", gap: "0.85rem" }}>
            <div style={panelStyle}>
              <div style={{ fontSize: "0.82rem", color: "rgba(255,255,255,0.65)", marginBottom: "0.35rem" }}>Scan status</div>
              <div style={{ fontWeight: 800, color: "#fff", lineHeight: 1.5 }}>{scanStatus}</div>
            </div>

            <div style={panelStyle}>
              <div style={{ fontSize: "0.82rem", color: "rgba(255,255,255,0.65)", marginBottom: "0.35rem" }}>Attendance status</div>
              <div style={{ color: "rgba(255,255,255,0.78)", lineHeight: 1.6 }}>{attendanceStatus}</div>
            </div>

            {hasAttendanceSummary ? (
              <div style={{ padding: "1rem", borderRadius: 18, background: "rgba(34,197,94,0.07)", border: "1px solid rgba(34,197,94,0.2)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.75rem" }}>
                  <div>
                    <div style={{ fontWeight: 800, color: "#fff", fontSize: "1rem" }}>{lastEmployee.name}</div>
                    <div style={{ color: "rgba(255,255,255,0.55)", fontSize: "0.78rem", marginTop: 2 }}>ID: {lastEmployee.code || (lastEmployee.id ?? "-")}</div>
                  </div>
                  {lastEmployee.status ? (
                    <div style={{
                      padding: "0.25rem 0.65rem",
                      borderRadius: 999,
                      fontSize: "0.72rem",
                      fontWeight: 800,
                      background: rgba(statusColor(lastEmployee.status), 0.18),
                      border: "1px solid " + rgba(statusColor(lastEmployee.status), 0.4),
                      color: statusColor(lastEmployee.status),
                    }}>
                      {lastEmployee.status}
                    </div>
                  ) : null}
                </div>

                {lastEmployee.lastEventType ? (
                  <div style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "0.35rem",
                    padding: "0.3rem 0.7rem",
                    borderRadius: 999,
                    marginBottom: "0.85rem",
                    fontSize: "0.8rem",
                    fontWeight: 700,
                    background: rgba(eventTone(lastEmployee.lastEventType), 0.15),
                    border: "1px solid " + rgba(eventTone(lastEmployee.lastEventType), 0.3),
                    color: eventTone(lastEmployee.lastEventType),
                  }}>
                    {eventLabel(lastEmployee.lastEventType)}
                    {lastEmployee.lastEventTime ? " at " + lastEmployee.lastEventTime : ""}
                  </div>
                ) : null}

                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.6rem" }}>
                  <div style={{ padding: "0.6rem 0.75rem", borderRadius: 12, background: "rgba(255,255,255,0.05)" }}>
                    <div style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.5)", marginBottom: 3, textTransform: "uppercase", letterSpacing: "0.04em" }}>First In</div>
                    <div style={{ fontWeight: 800, color: "#22c55e", fontSize: "1rem" }}>{formatTime(lastEmployee.firstIn)}</div>
                  </div>
                  <div style={{ padding: "0.6rem 0.75rem", borderRadius: 12, background: "rgba(255,255,255,0.05)" }}>
                    <div style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.5)", marginBottom: 3, textTransform: "uppercase", letterSpacing: "0.04em" }}>Last Out</div>
                    <div style={{ fontWeight: 800, color: lastEmployee.lastOut ? "#f87171" : "rgba(255,255,255,0.35)", fontSize: "1rem" }}>{formatTime(lastEmployee.lastOut)}</div>
                  </div>
                  <div style={{ padding: "0.6rem 0.75rem", borderRadius: 12, background: "rgba(255,255,255,0.05)" }}>
                    <div style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.5)", marginBottom: 3, textTransform: "uppercase", letterSpacing: "0.04em" }}>Working Hours</div>
                    <div style={{ fontWeight: 800, color: "#60a5fa", fontSize: "1rem" }}>{formatHours(lastEmployee.workHours)}</div>
                  </div>
                  <div style={{ padding: "0.6rem 0.75rem", borderRadius: 12, background: "rgba(255,255,255,0.05)" }}>
                    <div style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.5)", marginBottom: 3, textTransform: "uppercase", letterSpacing: "0.04em" }}>Break Time</div>
                    <div style={{ fontWeight: 800, color: "#a78bfa", fontSize: "1rem" }}>{formatHours(lastEmployee.breakHours)}</div>
                  </div>
                </div>
              </div>
            ) : (
              <div style={{ padding: "1rem", borderRadius: 18, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)" }}>
                <div style={{ fontSize: "0.82rem", color: "rgba(255,255,255,0.65)", marginBottom: "0.35rem" }}>Last recognized employee</div>
                <div style={{ fontWeight: 800, color: "#fff" }}>{lastEmployee.name || "None yet"}</div>
              </div>
            )}
          </div>

          <div style={{ display: "grid", gap: "0.9rem" }}>
            <div style={panelStyle}>
              <div style={{ fontSize: "0.82rem", color: "rgba(255,255,255,0.65)", marginBottom: "0.35rem" }}>Quick guidance</div>
              <div style={{ color: "rgba(255,255,255,0.78)", lineHeight: 1.6 }}>
                Use the RTSP or HTTP snapshot URL that the backend server can reach. A single scan will mark the next attendance event, and auto-scan repeats the same flow at your chosen interval.
              </div>
            </div>

            <div style={panelStyle}>
              <div style={{ fontSize: "0.82rem", color: "rgba(255,255,255,0.65)", marginBottom: "0.35rem" }}>Recognition result</div>
            <div style={{ fontSize: "0.8rem", color: "rgba(255,255,255,0.7)", marginBottom: "0.7rem" }}>
              Best confidence: {bestConfidence > 0 ? bestConfidence.toFixed(4) : "n/a"}
            </div>
              {latestFaces.length ? (
                <div style={{ display: "grid", gap: "0.7rem" }}>
                  {latestFaces.map((face, index) => (
                    <div key={String(face.employee_id ?? index) + "-" + String(index)} style={{ padding: "0.8rem", borderRadius: 14, background: face.matched ? rgba("#22c55e", 0.1) : rgba("#ef4444", 0.1), border: "1px solid " + (face.matched ? rgba("#22c55e", 0.22) : rgba("#ef4444", 0.22)) }}>
                      <div style={{ fontWeight: 800, color: "#fff" }}>{formatEmployeeLabel(face)}</div>
                      <div style={{ color: "rgba(255,255,255,0.72)", marginTop: 4 }}>State: {face.state}</div>
                      <div style={{ color: "rgba(255,255,255,0.72)", marginTop: 4 }}>Score: {typeof face.score === "number" ? face.score.toFixed(4) : "n/a"}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ color: "rgba(255,255,255,0.72)" }}>No recognition result yet.</div>
              )}
            </div>
          </div>
        </div>
      </section>

      <section style={{ marginTop: "1rem", display: "grid", gap: "1rem", gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))" }}>
        <NavLink to="/attendance" style={{ textDecoration: "none" }}>
          <div className="card" style={{ height: "100%", padding: "1.1rem", borderRadius: 20, background: "linear-gradient(180deg, #111827 0%, #0b1220 100%)", color: "#fff", border: "1px solid rgba(255,255,255,0.08)" }}>
            <div style={{ fontWeight: 800, fontSize: "1.02rem" }}>Attendance Review</div>
            <div style={{ marginTop: "0.45rem", color: "rgba(255,255,255,0.72)", lineHeight: 1.6 }}>Open the daily and monthly attendance grid.</div>
            <div style={{ marginTop: "0.9rem", color: "#7aa2ff", fontWeight: 800 }}>Open Attendance</div>
          </div>
        </NavLink>
        <NavLink to="/cctv-cameras" style={{ textDecoration: "none" }}>
          <div className="card" style={{ height: "100%", padding: "1.1rem", borderRadius: 20, background: "linear-gradient(180deg, #111827 0%, #0b1220 100%)", color: "#fff", border: "1px solid rgba(99,102,241,0.3)" }}>
            <div style={{ fontWeight: 800, fontSize: "1.02rem" }}>📷 Camera Manager</div>
            <div style={{ marginTop: "0.45rem", color: "rgba(255,255,255,0.72)", lineHeight: 1.6 }}>Add, configure, and monitor Hikvision DVR cameras.</div>
            <div style={{ marginTop: "0.9rem", color: "#7aa2ff", fontWeight: 800 }}>Manage Cameras</div>
          </div>
        </NavLink>
        <NavLink to="/face-detection" style={{ textDecoration: "none" }}>
          <div className="card" style={{ height: "100%", padding: "1.1rem", borderRadius: 20, background: "linear-gradient(180deg, #111827 0%, #0b1220 100%)", color: "#fff", border: "1px solid rgba(255,255,255,0.08)" }}>
            <div style={{ fontWeight: 800, fontSize: "1.02rem" }}>Webcam Mode</div>
            <div style={{ marginTop: "0.45rem", color: "rgba(255,255,255,0.72)", lineHeight: 1.6 }}>Switch back to browser webcam recognition.</div>
            <div style={{ marginTop: "0.9rem", color: "#7aa2ff", fontWeight: 800 }}>Open Webcam UI</div>
          </div>
        </NavLink>
      </section>
    </div>
  );
}