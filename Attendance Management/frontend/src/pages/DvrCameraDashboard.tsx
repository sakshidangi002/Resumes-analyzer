import { useCallback, useEffect, useRef, useState } from "react";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import { dvr } from "../api/client";
import { useMediaToken } from "../hooks/useMediaToken";

type LiveCamera = {
  channel_id: number;
  name: string;
  status: string;
  recognition_enabled: boolean;
  last_frame_time: number;
  error_message: string;
  worker_status?: {
    is_alive: boolean;
    last_error: string | null;
    fps: number;
    total_frames: number;
  };
};

type DVRStatus = {
  connected: boolean;
  connection_info?: {
    ip: string;
    port: number;
    username: string;
    connected: boolean;
    device_info?: {
      model: string;
      serial: string;
      total_channels: number;
    };
    cameras_count: number;
  };
  cameras: LiveCamera[];
};

export default function DvrCameraDashboard() {
  const [dvrForm, setDvrForm] = useState({
    ip: "",
    port: "8000",
    username: "",
    password: "",
  });
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState("");
  const [dvrStatus, setDvrStatus] = useState<DVRStatus | null>(null);

  // Only ONE channel streams at a time. Each MJPEG feed is an open
  // multipart/x-mixed-replace connection that never ends, and the browser only
  // allows ~6 connections per host: with every online camera streaming, the 5 s
  // status poll and every other API call queued behind them and the whole page
  // stalled. Server-side each feed also costs a full-frame JPEG encode per
  // frame, which on this 4-core box is what makes the picture drift behind.
  const [liveChannel, setLiveChannel] = useState<number | null>(null);
  const liveImgRef = useRef<HTMLImageElement | null>(null);
  // Short-lived token for the <img>-rendered DVR feed (see useMediaToken).
  const { mediaToken } = useMediaToken();

  // Removing the <img> from the DOM usually aborts its request, but not
  // reliably for a stream that never completes. Clearing src first guarantees
  // the browser drops the connection and the server stops encoding for us.
  useEffect(() => {
    const img = liveImgRef.current;
    return () => {
      if (img) img.src = "";
    };
  }, [liveChannel]);

  const fetchStatus = useCallback(async () => {
    try {
      const res = await dvr.status();
      setDvrStatus(res.data);
    } catch (err) {
      console.error("Failed to fetch DVR status", err);
    }
  }, []);

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 5000);
    return () => clearInterval(interval);
  }, [fetchStatus]);

  // Auto-open the first running camera so the page isn't blank on arrival, and
  // drop the selection if that camera stops.
  useEffect(() => {
    const running = (dvrStatus?.cameras ?? []).filter((c) => c.worker_status?.is_alive);
    if (!running.length) {
      if (liveChannel !== null) setLiveChannel(null);
      return;
    }
    if (liveChannel === null || !running.some((c) => c.channel_id === liveChannel)) {
      setLiveChannel(running[0].channel_id);
    }
  }, [dvrStatus, liveChannel]);

  const handleConnect = async () => {
    if (!dvrForm.ip.trim() || !dvrForm.username.trim() || !dvrForm.password.trim()) {
      setError("Please fill in all DVR credentials.");
      return;
    }
    setConnecting(true);
    setError("");
    try {
      const res = await dvr.connect({
        ip: dvrForm.ip.trim(),
        port: parseInt(dvrForm.port) || 8000,
        username: dvrForm.username.trim(),
        password: dvrForm.password.trim(),
      });
      const data = res.data as { success: boolean; message: string; device_info?: any; cameras?: any[] };
      if (data.success) {
        await fetchStatus();
      } else {
        setError(data.message || "Connection failed");
      }
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail ?? "Connection failed. Check your DVR credentials and network.");
    } finally {
      setConnecting(false);
    }
  };

  const handleDisconnect = async () => {
    try {
      await dvr.disconnect();
      setDvrStatus(null);
    } catch (err) {
      console.error("Failed to disconnect", err);
    }
  };

  const handleStartCamera = async (channelId: number) => {
    try {
      await dvr.startCamera(channelId);
      await fetchStatus();
    } catch (err) {
      console.error("Failed to start camera", err);
    }
  };

  const handleStopCamera = async (channelId: number) => {
    try {
      await dvr.stopCamera(channelId);
      await fetchStatus();
    } catch (err) {
      console.error("Failed to stop camera", err);
    }
  };

  const handleToggleRecognition = async (channelId: number, enabled: boolean) => {
    try {
      await dvr.setRecognition(channelId, enabled);
      await fetchStatus();
    } catch (err) {
      console.error("Failed to toggle recognition", err);
    }
  };

  const handleStartAll = async () => {
    try {
      await dvr.startAll();
      await fetchStatus();
    } catch (err) {
      console.error("Failed to start all cameras", err);
    }
  };

  const handleStopAll = async () => {
    try {
      await dvr.stopAll();
      await fetchStatus();
    } catch (err) {
      console.error("Failed to stop all cameras", err);
    }
  };

  // Styles
  const cardStyle: React.CSSProperties = {
    background: "var(--eds-card)",
    border: "1px solid var(--eds-border)",
    borderRadius: 14,
    padding: "1.1rem 1.25rem",
  };

  const inputStyle: React.CSSProperties = {
    background: "rgba(255,255,255,0.06)",
    border: "1px solid rgba(255,255,255,0.09)",
    borderRadius: 8,
    padding: "0.75rem 1rem",
    color: "white",
    fontSize: "0.9rem",
    width: "100%",
    boxSizing: "border-box",
  };

  const btnStyle = (variant: "primary" | "secondary" | "danger" | "ghost"): React.CSSProperties => {
    const base = {
      display: "inline-flex",
      alignItems: "center",
      justifyContent: "center",
      gap: "0.4rem",
      padding: "0.6rem 1.2rem",
      borderRadius: 8,
      fontSize: "0.85rem",
      fontWeight: 600,
      cursor: "pointer",
      border: "none",
      transition: "all 0.2s",
    };
    const variants = {
      primary: { background: "rgba(96,165,250,0.14)", color: "var(--eds-sky)", border: "1px solid rgba(96,165,250,0.3)" },
      secondary: { background: "rgba(255,255,255,0.04)", color: "var(--eds-text)", border: "1px solid var(--eds-border)" },
      danger: { background: "rgba(251,113,133,0.12)", color: "var(--eds-rose)", border: "1px solid rgba(251,113,133,0.28)" },
      ghost: { background: "transparent", color: "rgba(255,255,255,0.7)" },
    };
    return { ...base, ...variants[variant] };
  };

  const labelStyle: React.CSSProperties = {
    display: "block",
    marginBottom: "0.4rem",
    fontSize: "0.8rem",
    fontWeight: 600,
    color: "rgba(255,255,255,0.7)",
  };

  return (
    <div className="eds">
      {/* Header */}
      <header className="eds-topbar">
        <div>
          <h1 className="page-title">📷 DVR Camera Dashboard</h1>
          <div className="page-subtitle">Automatic Hikvision DVR camera discovery and live streaming</div>
        </div>
        <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
          <GlobalHeaderControls />
        </div>
      </header>

      <div className="eds-page dvr-dashboard-page">

      {/* Error */}
      {error && (
        <div style={{ background: "rgba(239,68,68,0.12)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 12, padding: "0.9rem 1.2rem", color: "#fca5a5" }}>
          {error}
        </div>
      )}

      {!dvrStatus?.connected ? (
        /* DVR Login Form */
        <div style={{ maxWidth: 500, margin: "2rem auto" }}>
          <div style={cardStyle}>
            <h2 style={{ margin: "0 0 1.5rem", fontSize: "1.1rem" }}>Connect to DVR</h2>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
              <div style={{ gridColumn: "1/-1" }}>
                <label style={labelStyle}>DVR IP Address *</label>
                <input style={inputStyle} value={dvrForm.ip}
                  onChange={(e) => setDvrForm((f) => ({ ...f, ip: e.target.value }))}
                  placeholder="192.168.29.181" />
              </div>
              <div>
                <label style={labelStyle}>DVR Port</label>
                <input style={inputStyle} value={dvrForm.port}
                  onChange={(e) => setDvrForm((f) => ({ ...f, port: e.target.value }))}
                  placeholder="8000" />
              </div>
              <div style={{ gridColumn: "1/-1" }}></div>
              <div>
                <label style={labelStyle}>Username *</label>
                <input style={inputStyle} value={dvrForm.username}
                  onChange={(e) => setDvrForm((f) => ({ ...f, username: e.target.value }))}
                  placeholder="admin" />
              </div>
              <div>
                <label style={labelStyle}>Password *</label>
                <input style={inputStyle} type="password" value={dvrForm.password}
                  onChange={(e) => setDvrForm((f) => ({ ...f, password: e.target.value }))}
                  placeholder="••••••••" />
              </div>
            </div>
            <div style={{ marginTop: "1.5rem" }}>
              <button type="button" style={{ ...btnStyle("primary"), width: "100%" }} onClick={handleConnect} disabled={connecting}>
                {connecting ? "Connecting..." : "🔌 Connect DVR"}
              </button>
            </div>
          </div>
        </div>
      ) : (
        /* Camera Dashboard */
        <>
          {/* DVR Info Bar */}
          <div style={{ ...cardStyle, marginBottom: "1.5rem", display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem" }}>
            <div>
              <div style={{ fontSize: "0.9rem", fontWeight: 600, marginBottom: "0.3rem" }}>
                ✓ Connected to {dvrStatus.connection_info?.ip}:{dvrStatus.connection_info?.port}
              </div>
              <div style={{ fontSize: "0.8rem", color: "rgba(255,255,255,0.6)" }}>
                {dvrStatus.connection_info?.device_info?.model} | {dvrStatus.connection_info?.cameras_count} cameras
              </div>
            </div>
            <div style={{ display: "flex", gap: "0.5rem" }}>
              <button type="button" style={btnStyle("primary")} onClick={handleStartAll}>▶ Start All</button>
              <button type="button" style={btnStyle("danger")} onClick={handleStopAll}>⏹ Stop All</button>
              <button type="button" style={btnStyle("ghost")} onClick={handleDisconnect}>🔌 Disconnect</button>
            </div>
          </div>

          {/* Camera Grid */}
          <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: "1rem" }}>
            {dvrStatus.cameras.map((camera) => (
              <div key={camera.channel_id} style={cardStyle}>
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1rem" }}>
                  <div>
                    <div style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "0.3rem" }}>{camera.name}</div>
                    <div style={{ fontSize: "0.8rem", color: "rgba(255,255,255,0.6)" }}>
                      Channel {camera.channel_id}
                    </div>
                  </div>
                  <div style={{
                    padding: "0.3rem 0.6rem",
                    borderRadius: 6,
                    fontSize: "0.75rem",
                    fontWeight: 600,
                    background: camera.status === "online" ? "rgba(34,197,94,0.2)" : "rgba(239,68,68,0.2)",
                    color: camera.status === "online" ? "#86efac" : "#fca5a5",
                  }}>
                    {camera.status}
                  </div>
                </div>

                {/* Live Preview */}
                <div
                  className="dvr-feed"
                  style={{
                    background: "#000",
                    borderRadius: 8,
                    width: "100%",
                    // Fixed viewport-height box; the feed fits inside it
                    // (object-fit: contain) — whole frame, as big as fits, no
                    // scroll. Thin black bars fill any leftover space.
                    height: "82vh",
                    marginBottom: "1rem",
                    border: "1px solid rgba(255,255,255,0.1)",
                    overflow: "hidden",
                    position: "relative",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  {camera.worker_status?.is_alive && camera.channel_id !== liveChannel ? (
                    // Not the selected channel: render NO <img> at all, so no
                    // MJPEG connection is opened and the server does no JPEG
                    // encoding for it.
                    <button
                      type="button"
                      onClick={() => setLiveChannel(camera.channel_id)}
                      style={{
                        display: "flex", flexDirection: "column", alignItems: "center",
                        gap: "0.6rem", background: "transparent", border: 0,
                        color: "rgba(255,255,255,0.75)", cursor: "pointer", fontSize: "0.95rem",
                      }}
                    >
                      <span style={{ fontSize: "2.4rem", lineHeight: 1 }}>▶</span>
                      <span>View live</span>
                      <span style={{ fontSize: "0.78rem", color: "rgba(255,255,255,0.45)" }}>
                        One camera streams at a time to keep the feed real-time
                      </span>
                    </button>
                  ) : camera.worker_status?.is_alive && mediaToken ? (
                    <>
                    <img
                      // Keyed by channel so switching cameras unmounts the old
                      // <img> and closes its never-ending HTTP connection.
                      key={`live-${camera.channel_id}`}
                      ref={(el) => {
                        if (el) liveImgRef.current = el;
                      }}
                      src={dvr.streamUrl(camera.channel_id, mediaToken)}
                      alt={camera.name}
                      style={{
                        // Fit the whole frame inside the box (like a video
                        // player): as large as possible, no cropping, no scroll.
                        width: "100%",
                        height: "100%",
                        objectFit: "contain",
                        display: "block",
                      }}
                      onError={(e) => {
                        // Clear src too: leaving it set makes a failing stream
                        // retry forever behind display:none.
                        const img = e.target as HTMLImageElement;
                        img.src = "";
                        img.style.display = "none";
                      }}
                    />
                    <button
                      type="button"
                      onClick={(e) => {
                        const box = (e.currentTarget.closest(".dvr-feed") as HTMLElement | null);
                        if (document.fullscreenElement) void document.exitFullscreen();
                        else void box?.requestFullscreen?.();
                      }}
                      style={{
                        position: "absolute", top: 10, right: 10, zIndex: 2,
                        padding: "0.35rem 0.7rem", borderRadius: 8, cursor: "pointer",
                        background: "rgba(0,0,0,0.6)", border: "1px solid rgba(255,255,255,0.3)",
                        color: "#fff", fontSize: "0.8rem", fontWeight: 700,
                      }}
                    >
                      ⛶ Fullscreen
                    </button>
                    </>
                  ) : (
                    <div style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      height: "100%",
                      color: "rgba(255,255,255,0.4)",
                      fontSize: "0.85rem",
                    }}>
                      📷 No Signal
                    </div>
                  )}
                </div>

                {/* Controls */}
                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.8rem" }}>
                  <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.5)" }}>
                    {camera.worker_status?.fps ? `${camera.worker_status.fps} FPS` : "0 FPS"}
                  </div>
                  <div style={{ display: "flex", gap: "0.5rem" }}>
                    {camera.worker_status?.is_alive ? (
                      <button type="button" style={btnStyle("danger")} onClick={() => handleStopCamera(camera.channel_id)}>
                        ⏹ Stop
                      </button>
                    ) : (
                      <button type="button" style={btnStyle("primary")} onClick={() => handleStartCamera(camera.channel_id)}>
                        ▶ Start
                      </button>
                    )}
                  </div>
                </div>

                {/* Recognition Toggle */}
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", paddingTop: "0.8rem", borderTop: "1px solid rgba(255,255,255,0.1)" }}>
                  <span style={{ fontSize: "0.85rem", color: "rgba(255,255,255,0.8)" }}>Face Recognition</span>
                  <button
                    type="button"
                    onClick={() => handleToggleRecognition(camera.channel_id, !camera.recognition_enabled)}
                    style={{
                      padding: "0.4rem 0.8rem",
                      borderRadius: 6,
                      fontSize: "0.8rem",
                      fontWeight: 600,
                      cursor: "pointer",
                      border: "none",
                      background: camera.recognition_enabled ? "rgba(34,197,94,0.2)" : "rgba(255,255,255,0.1)",
                      color: camera.recognition_enabled ? "#86efac" : "rgba(255,255,255,0.7)",
                    }}
                  >
                    {camera.recognition_enabled ? "☑ Enabled" : "☐ Disabled"}
                  </button>
                </div>
              </div>
            ))}
          </div>
        </>
      )}
      </div>
    </div>
  );
}
