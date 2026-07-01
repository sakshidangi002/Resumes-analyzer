import { useCallback, useEffect, useState } from "react";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import { dvr } from "../api/client";

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
    ip: "192.168.29.181",
    port: "8000",
    username: "anilchanna",
    password: "test@123",
  });
  const [connecting, setConnecting] = useState(false);
  const [error, setError] = useState("");
  const [dvrStatus, setDvrStatus] = useState<DVRStatus | null>(null);

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
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 16,
    padding: "0.8rem 1rem",
  };

  const inputStyle: React.CSSProperties = {
    background: "rgba(255,255,255,0.06)",
    border: "1px solid rgba(255,255,255,0.1)",
    borderRadius: 8,
    padding: "0.75rem 1rem",
    color: "white",
    fontSize: "0.9rem",
    width: "100%",
    boxSizing: "border-box",
  };

  const btnStyle = (variant: "primary" | "secondary" | "danger" | "ghost"): React.CSSProperties => {
   const base = {
      padding: "0.6rem 1.2rem",
      borderRadius: 8,
      fontSize: "0.85rem",
      fontWeight: 600,
      cursor: "pointer",
      border: "none",
      transition: "all 0.2s",
    };
    const variants = {
      primary: { background: "#3b82f6", color: "white" },
      secondary: { background: "rgba(255,255,255,0.1)", color: "white" },
      danger: { background: "#ef4444", color: "white" },
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
    <div className="page-stack">
      {/* Header */}
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem" }}>
        <div>
          <h1 className="page-title">📷 DVR Camera Dashboard</h1>
          <div className="page-subtitle">Automatic Hikvision DVR camera discovery and live streaming</div>
        </div>
        <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
          <GlobalHeaderControls />
        </div>
      </div>

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
                <div style={{
                  background: "rgba(0,0,0,0.3)",
                  borderRadius: 8,
                  height: "80vh",
                  marginBottom: "1rem",
                  border: "1px solid rgba(255,255,255,0.1)",
                  overflow: "hidden",
                  position: "relative",
                }}>
                  {camera.worker_status?.is_alive ? (
                    <img
                      src={dvr.streamUrl(camera.channel_id)}
                      alt={camera.name}
                      style={{
                        width: "100%",
                        height: "100%",
                        objectFit: "contain",
                      }}
                      onError={(e) => {
                        (e.target as HTMLImageElement).style.display = "none";
                      }}
                    />
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
  );
}
