import { useEffect, useRef, useState } from "react";
import { NavLink } from "react-router-dom";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import { recognition as recognitionApi } from "../api/client";

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
  pose?: { yaw?: number; pitch?: number; roll?: number };
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

// Per-employee attendance summary kept in-memory for the session.
// Mirrors what the backend returns but ensures First In is never overwritten.
type EmployeeAttendanceCache = {
  firstIn: string | null;    // earliest IN time seen today – NEVER overwritten
  lastOut: string | null;    // latest OUT time – always updated on new OUT
  workHours: number | null;
  breakHours: number | null;
  status: string;
  lastEventType: string | null;
  lastEventTime: number;     // ms timestamp of last event (for 60s UI dedup)
};

const cards = [
  {
    title: "Employee onboarding",
    subtitle: "Register people and store face samples before recognition.",
    to: "/employees",
    action: "Open Employees",
  },
  {
    title: "Attendance review",
    subtitle: "Review check-ins created automatically by face recognition.",
    to: "/attendance",
    action: "Open Attendance",
  },
  {
    title: "Monthly audit",
    subtitle: "Verify who was marked and when from the HR attendance grid.",
    to: "/attendance",
    action: "Review Records",
  },
];

const SCAN_INTERVAL_MS = 1200;

function formatEmployeeLabel(face: FacePayload): string {
  if (!face.matched) return "Unknown Person";
  const idLabel = face.employee_code || (face.employee_id == null ? "N/A" : String(face.employee_id));
  return `${face.employee_name} | ${idLabel}`;
}

function rgba(hex: string, alpha: number): string {
  const clean = hex.replace("#", "");
  const value = clean.length === 3
    ? clean.split("").map((ch) => ch + ch).join("")
    : clean.padEnd(6, "0").slice(0, 6);
  const r = Number.parseInt(value.slice(0, 2), 16);
  const g = Number.parseInt(value.slice(2, 4), 16);
  const b = Number.parseInt(value.slice(4, 6), 16);
  return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function formatTime(timeStr: string | null): string {
  if (!timeStr) return "–";
  // Handle "HH:MM:SS" format from backend
  const parts = timeStr.split(":");
  if (parts.length >= 2) {
    const h = parseInt(parts[0], 10);
    const m = parts[1];
    const ampm = h >= 12 ? "PM" : "AM";
    const h12 = h % 12 || 12;
    return `${h12}:${m} ${ampm}`;
  }
  return timeStr;
}

function formatHours(hours: number | null | undefined): string {
  if (hours == null) return "–";
  const h = Math.floor(hours);
  const m = Math.round((hours - h) * 60);
  if (h === 0) return `${m}m`;
  if (m === 0) return `${h}h`;
  return `${h}h ${m}m`;
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

export default function FaceDetection() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const scanTimerRef = useRef<number | null>(null);
  const busyRef = useRef(false);
  const lastFacesRef = useRef<FacePayload[]>([]);

  // Per-employee attendance cache: employee_id → EmployeeAttendanceCache
  // This ensures First In is preserved across multiple detections.
  const attendanceCacheRef = useRef<Map<number, EmployeeAttendanceCache>>(new Map());

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [scanStatus, setScanStatus] = useState("Start the webcam to begin live detection.");
  const [attendanceStatus, setAttendanceStatus] = useState("Recognized employees are sent to the HR attendance API automatically.");
  const [threshold, setThreshold] = useState("0.45");
  const [latestFaces, setLatestFaces] = useState<FacePayload[]>([]);

  // Last recognized employee summary for display
  const [lastEmployee, setLastEmployee] = useState<{
    name: string;
    id: number | null;
    code: string;
    firstIn: string | null;
    lastOut: string | null;
    workHours: number | null;
    breakHours: number | null;
    status: string;
    lastEventType: string | null;
    lastEventTime: string;
  }>({
    name: "",
    id: null,
    code: "",
    firstIn: null,
    lastOut: null,
    workHours: null,
    breakHours: null,
    status: "",
    lastEventType: null,
    lastEventTime: "",
  });

  const stopRecognitionLoop = () => {
    if (scanTimerRef.current != null) {
      window.clearInterval(scanTimerRef.current);
      scanTimerRef.current = null;
    }
  };

  const clearOverlay = () => {
    const canvas = overlayRef.current;
    const ctx = canvas?.getContext("2d");
    if (!canvas || !ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  };

  const syncOverlaySize = () => {
    const canvas = overlayRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
  };

  const drawFaces = (faces: FacePayload[]) => {
    const canvas = overlayRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const rect = canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(rect.width * dpr);
    canvas.height = Math.round(rect.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);

    const srcW = video.videoWidth || rect.width;
    const srcH = video.videoHeight || rect.height;
    const scale = Math.max(rect.width / srcW, rect.height / srcH);
    const drawnW = srcW * scale;
    const drawnH = srcH * scale;
    const offsetX = (rect.width - drawnW) / 2;
    const offsetY = (rect.height - drawnH) / 2;

    ctx.lineWidth = 3;
    ctx.textBaseline = "top";
    ctx.font = "600 14px Inter, Segoe UI, sans-serif";

    faces.forEach((face) => {
      if (!Array.isArray(face.box) || face.box.length < 4) return;

      const [x1, y1, x2, y2] = face.box;
      const x = x1 * scale + offsetX;
      const y = y1 * scale + offsetY;
      const w = (x2 - x1) * scale;
      const h = (y2 - y1) * scale;
      const color = face.matched ? "#22c55e" : "#ef4444";
      const fill = face.matched ? rgba(color, 0.12) : rgba(color, 0.14);

      ctx.strokeStyle = color;
      ctx.fillStyle = fill;
      ctx.fillRect(x, y, w, h);
      ctx.strokeRect(x, y, w, h);

      const lines = face.matched
        ? [face.employee_name, `ID: ${(face.employee_code || face.employee_id || "N/A")} | Known Employee`]
        : ["Unknown Person"];

      const widths = lines.map((line) => ctx.measureText(line).width);
      const labelWidth = Math.max(...widths, 0) + 20;
      const labelHeight = lines.length * 18 + 12;
      const labelY = Math.max(8, y - labelHeight - 10);
      const labelX = Math.max(8, x);

      ctx.fillStyle = color;
      ctx.fillRect(labelX, labelY, labelWidth, labelHeight);
      ctx.fillStyle = "#ffffff";

      lines.forEach((line, index) => {
        ctx.fillText(line, labelX + 10, labelY + 6 + index * 18);
      });
    });
  };

  const getUserMedia = async (constraints: MediaStreamConstraints): Promise<MediaStream> => {
    if (navigator.mediaDevices?.getUserMedia) {
      return navigator.mediaDevices.getUserMedia(constraints);
    }
    const legacyGetUserMedia = (navigator as any).getUserMedia || (navigator as any).webkitGetUserMedia || (navigator as any).mozGetUserMedia || (navigator as any).msGetUserMedia;
    if (!legacyGetUserMedia) {
      throw new Error("This browser does not support webcam access.");
    }
    return new Promise((resolve, reject) => {
      legacyGetUserMedia.call(navigator, constraints, resolve, reject);
    });
  };

  const stopCamera = () => {
    stopRecognitionLoop();
    const stream = streamRef.current;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    const video = videoRef.current;
    if (video) {
      video.srcObject = null;
    }
    setCameraActive(false);
    clearOverlay();
    setLatestFaces([]);
    lastFacesRef.current = [];
    attendanceCacheRef.current.clear();
    setLastEmployee({
      name: "", id: null, code: "",
      firstIn: null, lastOut: null,
      workHours: null, breakHours: null,
      status: "", lastEventType: null, lastEventTime: "",
    });
    setScanStatus("Camera stopped.");
    setAttendanceStatus("Recognition paused.");
  };

  const captureFrameBlob = async (): Promise<Blob> => {
    const video = videoRef.current;
    if (!video) {
      throw new Error("Webcam video element is not available.");
    }

    const width = video.videoWidth || 1280;
    const height = video.videoHeight || 720;
    const canvas = captureCanvasRef.current || document.createElement("canvas");
    captureCanvasRef.current = canvas;
    canvas.width = width;
    canvas.height = height;

    const ctx = canvas.getContext("2d");
    if (!ctx) {
      throw new Error("Could not prepare the frame capture canvas.");
    }
    ctx.drawImage(video, 0, 0, width, height);

    const blob = await new Promise<Blob | null>((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.9));
    if (!blob) {
      throw new Error("Could not capture the current video frame.");
    }
    return blob;
  };

  /**
   * Merge the backend attendance response into the local per-employee cache.
   *
   * Rules:
   *  - First In is set once (when cache entry is new or firstIn is null) and NEVER overwritten.
   *  - Last Out is always updated when the backend reports a sign_out_time.
   *  - work_hours and break_hours are always taken from the latest backend response.
   */
  const mergeAttendanceCache = (
    employeeId: number,
    att: AttendanceSummary,
  ): EmployeeAttendanceCache => {
    const now = Date.now();
    const existing = attendanceCacheRef.current.get(employeeId);

    // Determine First In: once set, never overwrite.
    let firstIn: string | null;
    if (existing?.firstIn) {
      // Keep the original first-in time always
      firstIn = existing.firstIn;
    } else {
      // First time seeing this employee today
      firstIn = att.sign_in_time ?? null;
    }

    // Determine Last Out: always take the latest value from backend.
    // The backend recalculates this from all events so it's always correct.
    const lastOut = att.sign_out_time ?? (existing?.lastOut ?? null);

    const updated: EmployeeAttendanceCache = {
      firstIn,
      lastOut,
      workHours: att.total_work_hours ?? existing?.workHours ?? null,
      breakHours: att.total_break_hours ?? existing?.breakHours ?? null,
      status: att.status ?? existing?.status ?? "",
      lastEventType: att.event_type ?? existing?.lastEventType ?? null,
      lastEventTime: now,
    };

    attendanceCacheRef.current.set(employeeId, updated);
    return updated;
  };

  const handleAttendanceResult = (_face: FacePayload, result: RecognitionResult) => {
    const att = result.attendance;
    if (!att || att.employee_id == null) return;

    const empId = att.employee_id;

    // Merge into cache (preserves First In, updates Last Out)
    const cached = mergeAttendanceCache(empId, att);

    // Determine human-readable event label
    const eventLabel =
      att.event_type === "OUT" || att.event_type === "CHECK_OUT" ? "✓ Check-Out"
      : att.event_type === "IN" || att.event_type === "CHECK_IN" ? "✓ Check-In"
      : "Attendance recorded";

    setAttendanceStatus(`${eventLabel} recorded for ${att.employee_name}.`);

    const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

    setLastEmployee({
      name: att.employee_name,
      id: empId,
      code: att.employee_code || String(empId),
      firstIn: cached.firstIn,
      lastOut: cached.lastOut,
      workHours: cached.workHours,
      breakHours: cached.breakHours,
      status: cached.status,
      lastEventType: cached.lastEventType,
      lastEventTime: timeStr,
    });
  };

  const runRecognition = async () => {
    if (!cameraActive || busyRef.current) return;
    busyRef.current = true;
    setCameraError("");

    try {
      const blob = await captureFrameBlob();
      const response = await recognitionApi.recognizeFrame(blob, Number(threshold));
      const data = response.data as RecognitionResult;
      const faces = data.faces || [];

      lastFacesRef.current = faces;
      setLatestFaces(faces);
      drawFaces(faces);

      const recognizedFaces = faces.filter((face) => face.matched);
      const unknownFaces = faces.length - recognizedFaces.length;

      if (!faces.length) {
        setScanStatus(data.message || "No face detected.");
        setAttendanceStatus("Waiting for a face to enter the frame.");
        return;
      }

      // Update recognized-name display for first matched face
      if (recognizedFaces.length) {
        const first = recognizedFaces[0];
        // Update lastEmployee name/id/code without overwriting attendance summary
        // (full update happens in handleAttendanceResult when attendance is returned)
        if (first.employee_id != null && !attendanceCacheRef.current.has(first.employee_id)) {
          // Show name even before first attendance event
          setLastEmployee((prev) => ({
            ...prev,
            name: first.employee_name,
            id: first.employee_id,
            code: first.employee_code || String(first.employee_id ?? ""),
          }));
        }
      }

      // The face state from backend determines what happened:
      // "in" / "out"        → attendance event recorded (IN or OUT)
      // "cooldown"          → duplicate ignored by backend (60 s)
      // "already_marked"    → same as cooldown
      // "marked_in"         → legacy: first mark of the day
      // "recognized"        → photo/non-webcam source, no attendance
      const markedFace = faces.find((f) =>
        f.state === "in" || f.state === "out" || f.state === "marked_in" || f.state === "check_in" || f.state === "check_out"
      );
      const cooldownFace = faces.find((f) =>
        f.state === "cooldown" || f.state === "already_marked"
      );

      if (data.attendance?.employee_id != null) {
        // A real attendance event was recorded
        const actingFace = markedFace ?? recognizedFaces[0];
        if (actingFace) {
          handleAttendanceResult(actingFace, data);
        }
      } else if (cooldownFace) {
        // Backend rejected — within 60 s cooldown
        setAttendanceStatus(
          `${cooldownFace.employee_name} – duplicate scan ignored (within 60 s cooldown).`
        );
      } else if (recognizedFaces.length && !data.attendance) {
        // Recognized but no attendance action taken (e.g. non-webcam source)
        setAttendanceStatus(`${recognizedFaces[0].employee_name} recognized. Attendance tracked.`);
      } else if (unknownFaces > 0 && recognizedFaces.length === 0) {
        setAttendanceStatus("Unknown person detected. No attendance action taken.");
      }

      if (recognizedFaces.length && unknownFaces === 0) {
        setScanStatus(`${recognizedFaces.length} known employee${recognizedFaces.length > 1 ? "s" : ""} detected.`);
      } else if (recognizedFaces.length && unknownFaces > 0) {
        setScanStatus(`${recognizedFaces.length} known employee${recognizedFaces.length > 1 ? "s" : ""} and ${unknownFaces} unknown face${unknownFaces > 1 ? "s" : ""} detected.`);
      } else {
        setScanStatus("Unknown face detected.");
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Recognition request failed.";
      setCameraError(message);
      setScanStatus("Recognition stopped because of an error.");
    } finally {
      busyRef.current = false;
    }
  };

  const startRecognitionLoop = () => {
    stopRecognitionLoop();
    scanTimerRef.current = window.setInterval(() => {
      void runRecognition();
    }, SCAN_INTERVAL_MS);
    void runRecognition();
  };

  const startCamera = async () => {
    setCameraError("");
    try {
      const stream = await getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      const video = videoRef.current;
      if (video) {
        video.srcObject = stream;
        await video.play();
      }
      setCameraActive(true);
      setScanStatus("Camera live. Live recognition is running.");
      setAttendanceStatus("Waiting for a recognized employee.");
    } catch (err) {
      setCameraError(err instanceof Error ? err.message : "Could not start webcam.");
      setCameraActive(false);
    }
  };

  useEffect(() => {
    if (!cameraActive) return;
    startRecognitionLoop();
    return () => {
      stopRecognitionLoop();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cameraActive, threshold]);

  useEffect(() => {
    const handleResize = () => {
      syncOverlaySize();
      drawFaces(lastFacesRef.current);
    };
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, []);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    const onLoadedMetadata = () => {
      syncOverlaySize();
      drawFaces(lastFacesRef.current);
    };

    video.addEventListener("loadedmetadata", onLoadedMetadata);
    return () => {
      video.removeEventListener("loadedmetadata", onLoadedMetadata);
    };
  }, []);

  useEffect(() => {
    return () => {
      stopRecognitionLoop();
      const stream = streamRef.current;
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
      }
      clearOverlay();
    };
  }, []);

  const knownCount = latestFaces.filter((face) => face.matched).length;
  const unknownCount = latestFaces.length - knownCount;
  const markedCount = latestFaces.filter((face) =>
    face.state === "in" || face.state === "out" || face.state === "marked_in"
  ).length;

  const hasAttendanceSummary = lastEmployee.name && (lastEmployee.firstIn || lastEmployee.lastOut);

  return (
    <div className="page-stack">
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "1rem", flexWrap: "wrap" }}>
        <div>
          <h1 className="page-title">Face Detection</h1>
          <div className="page-subtitle">Live webcam scanning, face recognition, and automatic HR attendance marking</div>
        </div>
        <GlobalHeaderControls />
      </div>

      <section className="card" style={{ marginTop: "1rem", padding: "1.4rem", border: "none", background: "linear-gradient(135deg, rgba(9,14,31,0.98), rgba(15,23,42,0.94))", color: "#fff", overflow: "hidden" }}>
        <div style={{ display: "grid", gap: "1rem", gridTemplateColumns: "minmax(0, 1.4fr) minmax(280px, 0.6fr)" }}>
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: "0.65rem", flexWrap: "wrap" }}>
              <span style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem", padding: "0.42rem 0.75rem", borderRadius: 999, background: "rgba(34,197,94,0.12)", border: "1px solid rgba(34,197,94,0.22)", color: "#d3f9d8", fontSize: "0.82rem", fontWeight: 700 }}>Real-time scanner</span>
              <span style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem", padding: "0.42rem 0.75rem", borderRadius: 999, background: "rgba(96,165,250,0.12)", border: "1px solid rgba(96,165,250,0.22)", color: "#dbeafe", fontSize: "0.82rem", fontWeight: 700 }}>Attendance API linked</span>
              <span style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem", padding: "0.42rem 0.75rem", borderRadius: 999, background: "rgba(245,158,11,0.12)", border: "1px solid rgba(245,158,11,0.22)", color: "#fef3c7", fontSize: "0.82rem", fontWeight: 700 }}>60 s duplicate guard</span>
            </div>

            <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", marginTop: "1.2rem" }}>
              <button className="btn btn-primary" type="button" onClick={startCamera}>Start Live Webcam</button>
              <button className="btn btn-secondary" type="button" onClick={stopCamera}>Stop Camera</button>
              <NavLink className="btn btn-secondary" to="/employees">Register Employees</NavLink>
              <NavLink className="btn btn-secondary" to="/attendance">Review Attendance</NavLink>
            </div>

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: "0.85rem", marginTop: "1.4rem" }}>
              <div style={{ padding: "0.95rem 1rem", borderRadius: 18, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)" }}>
                <div style={{ fontSize: "0.8rem", color: "rgba(255,255,255,0.65)" }}>Camera</div>
                <div style={{ fontSize: "1.05rem", fontWeight: 800, marginTop: 4 }}>{cameraActive ? "Live" : "Idle"}</div>
              </div>
              <div style={{ padding: "0.95rem 1rem", borderRadius: 18, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)" }}>
                <div style={{ fontSize: "0.8rem", color: "rgba(255,255,255,0.65)" }}>Known faces</div>
                <div style={{ fontSize: "1.05rem", fontWeight: 800, marginTop: 4 }}>{knownCount}</div>
              </div>
              <div style={{ padding: "0.95rem 1rem", borderRadius: 18, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)" }}>
                <div style={{ fontSize: "0.8rem", color: "rgba(255,255,255,0.65)" }}>Unknown faces</div>
                <div style={{ fontSize: "1.05rem", fontWeight: 800, marginTop: 4 }}>{unknownCount}</div>
              </div>
              <div style={{ padding: "0.95rem 1rem", borderRadius: 18, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)" }}>
                <div style={{ fontSize: "0.8rem", color: "rgba(255,255,255,0.65)" }}>Attendance marks</div>
                <div style={{ fontSize: "1.05rem", fontWeight: 800, marginTop: 4 }}>{markedCount}</div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="card" style={{ marginTop: "1rem", padding: "1.1rem", borderRadius: 22, background: "linear-gradient(180deg, #111827 0%, #0b1220 100%)", color: "#fff", border: "1px solid rgba(255,255,255,0.08)" }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap", alignItems: "center" }}>
          <div>
            <div style={{ display: "inline-flex", alignItems: "center", gap: "0.4rem", padding: "0.42rem 0.75rem", borderRadius: 999, background: "rgba(122,162,255,0.12)", border: "1px solid rgba(122,162,255,0.18)", color: "#cfe0ff", fontSize: "0.82rem", fontWeight: 700 }}>Live webcam</div>
            <h3 style={{ margin: "0.8rem 0 0", fontSize: "1.45rem" }}>Real-time detection window</h3>
          </div>
          <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap", alignItems: "center" }}>
            <label style={{ display: "grid", gap: "0.35rem", color: "rgba(255,255,255,0.72)", minWidth: 190 }}>
              Match threshold
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={threshold}
                onChange={(e) => setThreshold(e.target.value)}
                style={{ width: "100%", borderRadius: 10, border: "1px solid rgba(255,255,255,0.15)", padding: "0.75rem", background: "#0b1220", color: "#fff" }}
              />
            </label>
            <button className="btn btn-primary" type="button" onClick={startCamera}>Start Camera</button>
            <button className="btn btn-secondary" type="button" onClick={stopCamera}>Stop</button>
          </div>
        </div>

        {cameraError ? (
          <div style={{ marginTop: "1rem", padding: "0.9rem 1rem", borderRadius: 16, background: "rgba(239,68,68,0.14)", border: "1px solid rgba(239,68,68,0.3)", color: "#fecaca" }}>
            {cameraError}
          </div>
        ) : null}

        <div style={{ display: "grid", gap: "1rem", gridTemplateColumns: "minmax(0, 1.35fr) minmax(280px, 0.65fr)", marginTop: "1rem" }}>
          <div style={{ borderRadius: 22, minHeight: 420, background: "radial-gradient(circle at 50% 20%, rgba(122,162,255,0.22), transparent 30%), linear-gradient(180deg, rgba(16,24,40,0.95), rgba(7,10,18,0.98))", border: "1px solid rgba(255,255,255,0.08)", overflow: "hidden", position: "relative" }}>
            <video
              ref={videoRef}
              id="face-webcam-video"
              autoPlay
              playsInline
              muted
              onLoadedMetadata={() => {
                syncOverlaySize();
                drawFaces(lastFacesRef.current);
              }}
              style={{ width: "100%", height: "100%", minHeight: 420, objectFit: "cover", display: "block", background: "#050816" }}
            />
            <canvas
              ref={overlayRef}
              style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
            />
            {!cameraActive ? (
              <div style={{ position: "absolute", inset: 0, display: "grid", placeItems: "center", textAlign: "center", padding: "1rem", background: "linear-gradient(180deg, rgba(3,7,18,0.15), rgba(3,7,18,0.55))" }}>
                <div>
                  <div style={{ fontSize: "3rem", lineHeight: 1 }}>📷</div>
                  <div style={{ marginTop: "0.75rem", fontSize: "1.1rem", fontWeight: 800 }}>Webcam preview</div>
                  <div style={{ marginTop: "0.45rem", color: "rgba(255,255,255,0.68)", lineHeight: 1.6 }}>Click Start Camera to show the live feed and begin recognition.</div>
                </div>
              </div>
            ) : null}
          </div>

          <div style={{ display: "grid", gap: "0.9rem" }}>
            {/* Scan status */}
            <div style={{ padding: "1rem", borderRadius: 18, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)" }}>
              <div style={{ fontSize: "0.82rem", color: "rgba(255,255,255,0.65)", marginBottom: "0.35rem" }}>Scan status</div>
              <div style={{ fontWeight: 800, color: "#fff", lineHeight: 1.5 }}>{scanStatus}</div>
            </div>

            {/* Attendance status */}
            <div style={{ padding: "1rem", borderRadius: 18, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)" }}>
              <div style={{ fontSize: "0.82rem", color: "rgba(255,255,255,0.65)", marginBottom: "0.35rem" }}>Attendance status</div>
              <div style={{ color: "rgba(255,255,255,0.78)", lineHeight: 1.6 }}>{attendanceStatus}</div>
            </div>

            {/* Attendance summary card — shown once an employee is recognized */}
            {hasAttendanceSummary ? (
              <div style={{ padding: "1rem", borderRadius: 18, background: "rgba(34,197,94,0.07)", border: "1px solid rgba(34,197,94,0.2)" }}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.75rem" }}>
                  <div>
                    <div style={{ fontWeight: 800, color: "#fff", fontSize: "1rem" }}>{lastEmployee.name}</div>
                    <div style={{ color: "rgba(255,255,255,0.55)", fontSize: "0.78rem", marginTop: 2 }}>
                      ID: {lastEmployee.code || (lastEmployee.id ?? "–")}
                    </div>
                  </div>
                  {lastEmployee.status && (
                    <div style={{
                      padding: "0.25rem 0.65rem",
                      borderRadius: 999,
                      fontSize: "0.72rem",
                      fontWeight: 800,
                      background: rgba(statusColor(lastEmployee.status), 0.18),
                      border: `1px solid ${rgba(statusColor(lastEmployee.status), 0.4)}`,
                      color: statusColor(lastEmployee.status),
                    }}>
                      {lastEmployee.status}
                    </div>
                  )}
                </div>

                {/* Last event badge */}
                {lastEmployee.lastEventType && (
                  <div style={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: "0.35rem",
                    padding: "0.3rem 0.7rem",
                    borderRadius: 999,
                    marginBottom: "0.85rem",
                    fontSize: "0.8rem",
                    fontWeight: 700,
                    background: lastEmployee.lastEventType === "OUT" || lastEmployee.lastEventType === "CHECK_OUT"
                      ? "rgba(239,68,68,0.15)"
                      : "rgba(34,197,94,0.15)",
                    border: `1px solid ${lastEmployee.lastEventType === "OUT" || lastEmployee.lastEventType === "CHECK_OUT" ? "rgba(239,68,68,0.3)" : "rgba(34,197,94,0.3)"}`,
                    color: lastEmployee.lastEventType === "OUT" || lastEmployee.lastEventType === "CHECK_OUT" ? "#fca5a5" : "#86efac",
                  }}>
                    {lastEmployee.lastEventType === "OUT" || lastEmployee.lastEventType === "CHECK_OUT" ? "⬆ Checked Out" : "⬇ Checked In"}
                    {lastEmployee.lastEventTime ? ` at ${lastEmployee.lastEventTime}` : ""}
                  </div>
                )}

                {/* Time summary grid */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.6rem" }}>
                  <div style={{ padding: "0.6rem 0.75rem", borderRadius: 12, background: "rgba(255,255,255,0.05)" }}>
                    <div style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.5)", marginBottom: 3, textTransform: "uppercase", letterSpacing: "0.04em" }}>First In</div>
                    <div style={{ fontWeight: 800, color: "#22c55e", fontSize: "1rem" }}>
                      {formatTime(lastEmployee.firstIn)}
                    </div>
                  </div>
                  <div style={{ padding: "0.6rem 0.75rem", borderRadius: 12, background: "rgba(255,255,255,0.05)" }}>
                    <div style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.5)", marginBottom: 3, textTransform: "uppercase", letterSpacing: "0.04em" }}>Last Out</div>
                    <div style={{ fontWeight: 800, color: lastEmployee.lastOut ? "#f87171" : "rgba(255,255,255,0.35)", fontSize: "1rem" }}>
                      {formatTime(lastEmployee.lastOut)}
                    </div>
                  </div>
                  <div style={{ padding: "0.6rem 0.75rem", borderRadius: 12, background: "rgba(255,255,255,0.05)" }}>
                    <div style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.5)", marginBottom: 3, textTransform: "uppercase", letterSpacing: "0.04em" }}>Working Hours</div>
                    <div style={{ fontWeight: 800, color: "#60a5fa", fontSize: "1rem" }}>
                      {formatHours(lastEmployee.workHours)}
                    </div>
                  </div>
                  <div style={{ padding: "0.6rem 0.75rem", borderRadius: 12, background: "rgba(255,255,255,0.05)" }}>
                    <div style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.5)", marginBottom: 3, textTransform: "uppercase", letterSpacing: "0.04em" }}>Break Time</div>
                    <div style={{ fontWeight: 800, color: "#a78bfa", fontSize: "1rem" }}>
                      {formatHours(lastEmployee.breakHours)}
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              /* Placeholder when no employee is recognized yet */
              <div style={{ padding: "1rem", borderRadius: 18, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)" }}>
                <div style={{ fontSize: "0.82rem", color: "rgba(255,255,255,0.65)", marginBottom: "0.35rem" }}>Last recognized employee</div>
                <div style={{ fontWeight: 800, color: "#fff" }}>{lastEmployee.name || "None yet"}</div>
                {lastEmployee.id && (
                  <>
                    <div style={{ color: "rgba(255,255,255,0.72)", marginTop: 6 }}>ID: {lastEmployee.code || (lastEmployee.id ?? "-")}</div>
                    <div style={{ color: "rgba(255,255,255,0.5)", marginTop: 4, fontSize: "0.82rem" }}>Database row: {lastEmployee.id ?? "-"}</div>
                  </>
                )}
              </div>
            )}

            {/* Recognition result panel */}
            <div style={{ padding: "1rem", borderRadius: 18, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)" }}>
              <div style={{ fontSize: "0.82rem", color: "rgba(255,255,255,0.65)", marginBottom: "0.35rem" }}>Recognition result</div>
              {latestFaces.length ? (
                <div style={{ display: "grid", gap: "0.7rem" }}>
                  {latestFaces.map((face, index) => (
                    <div key={`${face.employee_id ?? index}-${index}`} style={{ padding: "0.8rem", borderRadius: 14, background: face.matched ? rgba("#22c55e", 0.1) : rgba("#ef4444", 0.1), border: `1px solid ${face.matched ? rgba("#22c55e", 0.22) : rgba("#ef4444", 0.22)}` }}>
                      <div style={{ fontWeight: 800, color: "#fff" }}>{formatEmployeeLabel(face)}</div>
                      <div style={{ color: "rgba(255,255,255,0.72)", marginTop: 4 }}>State: {face.state}</div>
                      <div style={{ color: "rgba(255,255,255,0.72)", marginTop: 4 }}>Score: {typeof face.score === "number" ? face.score.toFixed(4) : "n/a"}</div>
                    </div>
                  ))}
                </div>
              ) : (
                <div style={{ color: "rgba(255,255,255,0.72)" }}>No recognition result yet. Start the camera and keep a face in frame.</div>
              )}
            </div>
          </div>
        </div>
      </section>

      <section style={{ marginTop: "1rem", display: "grid", gap: "1rem", gridTemplateColumns: "repeat(auto-fit, minmax(230px, 1fr))" }}>
        {cards.map((card) => (
          <NavLink key={card.title} to={card.to} style={{ textDecoration: "none" }}>
            <div className="card" style={{ height: "100%", padding: "1.1rem", borderRadius: 20, background: "linear-gradient(180deg, #111827 0%, #0b1220 100%)", color: "#fff", border: "1px solid rgba(255,255,255,0.08)" }}>
              <div style={{ fontWeight: 800, fontSize: "1.02rem" }}>{card.title}</div>
              <div style={{ marginTop: "0.45rem", color: "rgba(255,255,255,0.72)", lineHeight: 1.6 }}>{card.subtitle}</div>
              <div style={{ marginTop: "0.9rem", color: "#7aa2ff", fontWeight: 800 }}>{card.action}</div>
            </div>
          </NavLink>
        ))}
      </section>
    </div>
  );
}
