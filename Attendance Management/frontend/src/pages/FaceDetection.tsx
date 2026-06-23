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

type RecognitionResult = {
  status: boolean;
  message: string;
  source: string;
  faces: FacePayload[];
  attendance?: {
    employee_id: number;
    employee_code?: string | null;
    employee_name: string;
    date: string;
    sign_in_time: string | null;
    sign_out_time: string | null;
    status: string;
  } | null;
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
const ATTENDANCE_COOLDOWN_MS = 8000;

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

export default function FaceDetection() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const overlayRef = useRef<HTMLCanvasElement | null>(null);
  const captureCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const scanTimerRef = useRef<number | null>(null);
  const busyRef = useRef(false);
  const lastFacesRef = useRef<FacePayload[]>([]);
  const lastAttendanceAnnounceRef = useRef<Map<number, number>>(new Map());

  const [cameraActive, setCameraActive] = useState(false);
  const [cameraError, setCameraError] = useState("");
  const [scanStatus, setScanStatus] = useState("Start the webcam to begin live detection.");
  const [attendanceStatus, setAttendanceStatus] = useState("Recognized employees are sent to the HR attendance API automatically.");
  const [threshold, setThreshold] = useState("0.45");
  const [latestFaces, setLatestFaces] = useState<FacePayload[]>([]);
  const [lastRecognizedName, setLastRecognizedName] = useState<string>("");
  const [lastRecognizedId, setLastRecognizedId] = useState<number | null>(null);
  const [lastRecognizedCode, setLastRecognizedCode] = useState<string>("");
  const [lastMarkedAt, setLastMarkedAt] = useState<string>("");

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
    setLastRecognizedName("");
    setLastRecognizedId(null);
    setLastMarkedAt("");
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

  const announceAttendance = (face: FacePayload, result: RecognitionResult) => {
    const now = Date.now();
    if (face.employee_id != null) {
      const last = lastAttendanceAnnounceRef.current.get(face.employee_id);
      if (last && now - last < ATTENDANCE_COOLDOWN_MS && face.state !== "marked_in") {
        return;
      }
      lastAttendanceAnnounceRef.current.set(face.employee_id, now);
    }

    if (result.attendance?.employee_id != null && result.attendance.employee_name) {
      setLastRecognizedName(result.attendance.employee_name);
      setLastRecognizedId(result.attendance.employee_id);
      setLastRecognizedCode(result.attendance.employee_code || String(result.attendance.employee_id));
      setLastMarkedAt(result.attendance.date + " " + (result.attendance.sign_in_time || ""));
      setAttendanceStatus(`Attendance marked for ${result.attendance.employee_name}.`);
      return;
    }

    if (face.state === "already_marked") {
      setAttendanceStatus(`${face.employee_name} is already recorded for today.`);
      return;
    }

    if (face.state === "cooldown") {
      setAttendanceStatus(`${face.employee_name} is in the cooldown window. Attendance not re-triggered.`);
    }
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
      const markedFace = faces.find((face) => face.state === "marked_in");
      const alreadyMarkedFace = faces.find((face) => face.state === "already_marked");

      if (!faces.length) {
        setScanStatus(data.message || "No face detected.");
        setAttendanceStatus("Waiting for a face to enter the frame.");
        return;
      }

      if (recognizedFaces.length) {
        const firstRecognized = recognizedFaces[0];
        setLastRecognizedName(firstRecognized.employee_name);
        setLastRecognizedId(firstRecognized.employee_id);
        setLastRecognizedCode(firstRecognized.employee_code || String(firstRecognized.employee_id));
      }

      if (markedFace) {
        announceAttendance(markedFace, data);
      } else if (alreadyMarkedFace) {
        announceAttendance(alreadyMarkedFace, data);
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
  const markedCount = latestFaces.filter((face) => face.state === "marked_in").length;

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
            </div>
            {/* <h2 style={{ margin: "0.9rem 0 0", fontSize: "2rem", lineHeight: 1.1 }}>Live webcam feed with automatic HRM recognition</h2> */}



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
                  <div style={{ fontSize: "3rem", lineHeight: 1 }}>?</div>
                  <div style={{ marginTop: "0.75rem", fontSize: "1.1rem", fontWeight: 800 }}>Webcam preview</div>
                  <div style={{ marginTop: "0.45rem", color: "rgba(255,255,255,0.68)", lineHeight: 1.6 }}>Click Start Camera to show the live feed and begin recognition.</div>
                </div>
              </div>
            ) : null}
          </div>

          <div style={{ display: "grid", gap: "0.9rem" }}>
            <div style={{ padding: "1rem", borderRadius: 18, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)" }}>
              <div style={{ fontSize: "0.82rem", color: "rgba(255,255,255,0.65)", marginBottom: "0.35rem" }}>Scan status</div>
              <div style={{ fontWeight: 800, color: "#fff", lineHeight: 1.5 }}>{scanStatus}</div>
            </div>
            <div style={{ padding: "1rem", borderRadius: 18, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)" }}>
              <div style={{ fontSize: "0.82rem", color: "rgba(255,255,255,0.65)", marginBottom: "0.35rem" }}>Attendance status</div>
              <div style={{ color: "rgba(255,255,255,0.78)", lineHeight: 1.6 }}>{attendanceStatus}</div>
            </div>
            <div style={{ padding: "1rem", borderRadius: 18, background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)" }}>
              <div style={{ fontSize: "0.82rem", color: "rgba(255,255,255,0.65)", marginBottom: "0.35rem" }}>Last recognized employee</div>
              <div style={{ fontWeight: 800, color: "#fff" }}>{lastRecognizedName || "None yet"}</div>
              <div style={{ color: "rgba(255,255,255,0.72)", marginTop: 6 }}>ID: {lastRecognizedCode || (lastRecognizedId ?? "-")}</div>
              <div style={{ color: "rgba(255,255,255,0.5)", marginTop: 4, fontSize: "0.82rem" }}>Database row: {lastRecognizedId ?? "-"}</div>
              <div style={{ color: "rgba(255,255,255,0.72)", marginTop: 6 }}>Last mark: {lastMarkedAt || "-"}</div>
            </div>
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
