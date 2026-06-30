import { useCallback, useEffect, useRef, useState } from "react";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import { cameras as camerasApi } from "../api/client";

// ─── DVR Discovery Types ─────────────────────────────────────────────────────
type DiscoveredChannel = {
  id: number;
  name: string;
  status: string;
  channel_type: string;
  resolution: string | null;
};

type DiscoveredDevice = {
  model: string;
  firmware: string;
  serial: string;
  total_channels: number;
  analog_channels: number;
  ip_channels: number;
  channels: DiscoveredChannel[];
};

// ─── Types ───────────────────────────────────────────────────────────────────
type LiveStatus = {
  status: string;
  fps: number;
  total_frames: number;
  reconnect_count: number;
  last_frame_time: number;
  last_error: string | null;
  last_result_faces: number;
};

type Camera = {
  id: number;
  name: string;
  location: string | null;
  stream_url: string;
  source_type: string;
  camera_purpose: string;
  threshold: number;
  interval_sec: number;
  enabled: boolean;
  created_at: string | null;
  live: LiveStatus | null;
};

type FormState = {
  name: string;
  location: string;
  source_type: string;
  camera_purpose: string;
  camera_ip: string;
  rtsp_port: string;
  rtsp_username: string;
  rtsp_password: string;
  rtsp_channel: string;
  stream_url: string;
  threshold: string;
  interval_sec: string;
  enabled: boolean;
};

const BLANK_FORM: FormState = {
  name: "",
  location: "",
  source_type: "rtsp",
  camera_purpose: "IN",
  camera_ip: "",
  rtsp_port: "554",
  rtsp_username: "",
  rtsp_password: "",
  rtsp_channel: "1",
  stream_url: "",
  threshold: "0.45",
  interval_sec: "2",
  enabled: false,
};

// ─── Helpers ──────────────────────────────────────────────────────────────────
function statusColor(s: string): string {
  if (s === "running") return "#22c55e";
  if (s === "connecting" || s === "reconnecting") return "#f59e0b";
  if (s === "error") return "#ef4444";
  return "#64748b";
}

function purposeBadge(p: string) {
  const isIn = p === "IN";
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: "0.3rem",
      padding: "0.2rem 0.7rem", borderRadius: 999, fontSize: "0.78rem", fontWeight: 700,
      background: isIn ? "rgba(34,197,94,0.15)" : "rgba(239,68,68,0.15)",
      border: `1px solid ${isIn ? "rgba(34,197,94,0.4)" : "rgba(239,68,68,0.4)"}`,
      color: isIn ? "#86efac" : "#fca5a5",
    }}>
      {isIn ? "⬆ Check-In" : "⬇ Check-Out"}
    </span>
  );
}

function statusBadge(s: string) {
  const color = statusColor(s);
  const label = s.charAt(0).toUpperCase() + s.slice(1);
  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: "0.35rem",
      padding: "0.2rem 0.7rem", borderRadius: 999, fontSize: "0.78rem", fontWeight: 700,
      background: color + "22", border: `1px solid ${color}55`, color,
    }}>
      <span style={{ width: 7, height: 7, borderRadius: "50%", background: color, display: "inline-block" }} />
      {label}
    </span>
  );
}

function elapsedSince(ts: number): string {
  if (!ts) return "-";
  const s = Math.round(Date.now() / 1000 - ts);
  if (s < 60) return `${s}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  return `${Math.floor(s / 3600)}h ago`;
}

function parseRtspUrl(rtspUrl: string) {
  try {
    const parsed = new URL(rtspUrl);
    if (parsed.protocol !== "rtsp:") return null;
    const match = parsed.pathname.match(/\/Streaming\/Channels\/([1-9])01$/i);
    return {
      host: parsed.hostname,
      port: parsed.port || "554",
      username: decodeURIComponent(parsed.username || ""),
      password: decodeURIComponent(parsed.password || ""),
      channel: match?.[1] ?? "",
    };
  } catch {
    return null;
  }
}

function buildRtspUrl(form: FormState): string {
  const host = form.camera_ip.trim();
  const port = form.rtsp_port.trim() || "554";
  const channel = form.rtsp_channel.trim();
  if (!host) throw new Error("Camera IP address is required.");
  if (!/^[0-9]+$/.test(port)) throw new Error("RTSP port must be numeric.");
  if (!/^[1-5]$/.test(channel)) throw new Error("Channel must be a number from 1 to 5.");
  const rtspPath = "/Streaming/Channels/" + channel + "01";

  const auth = form.rtsp_username.trim()
    ? encodeURIComponent(form.rtsp_username.trim()) + (form.rtsp_password ? ":" + encodeURIComponent(form.rtsp_password) : "") + "@"
    : "";
  return "rtsp://" + auth + host + ":" + port + rtspPath;
}

// ─── Component ───────────────────────────────────────────────────────────────
export default function CctvCameraManager() {
  const [cameraList, setCameraList] = useState<Camera[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  // Add/Edit modal
  const [showModal, setShowModal] = useState(false);
  const [editId, setEditId] = useState<number | null>(null);
  const [form, setForm] = useState<FormState>(BLANK_FORM);
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState("");

  // Test connection
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ ok: boolean; msg: string } | null>(null);

  // Preview modal
  const [previewId, setPreviewId] = useState<number | null>(null);
  const [previewTs, setPreviewTs] = useState(Date.now());
  const previewTimerRef = useRef<number | null>(null);

  // Delete confirm
  const [deleteId, setDeleteId] = useState<number | null>(null);
  const [deleting, setDeleting] = useState(false);

  // DVR Discovery
  const [showDiscoveryModal, setShowDiscoveryModal] = useState(false);
  const [discovering, setDiscovering] = useState(false);
  const [discoveryError, setDiscoveryError] = useState("");
  const [discoveredDevice, setDiscoveredDevice] = useState<DiscoveredDevice | null>(null);
  const [selectedChannels, setSelectedChannels] = useState<Set<number>>(new Set());
  const [dvrForm, setDvrForm] = useState({
    ip: "",
    port: "8000",
    username: "",
    password: "",
  });

  // ── Data fetching ──────────────────────────────────────────────────────────
  const fetchCameras = useCallback(async () => {
    try {
      const res = await camerasApi.list();
      const payload = res.data as { cameras?: Camera[] } | Camera[];
      const cameras = Array.isArray(payload) ? payload : payload.cameras ?? [];
      setCameraList(cameras);
      setError("");
    } catch (err) {
      console.error("Failed to load cameras", err);
      setError("Failed to load cameras.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchCameras();
    const timer = window.setInterval(() => void fetchCameras(), 5000);
    return () => window.clearInterval(timer);
  }, [fetchCameras]);

  // ── Preview polling ────────────────────────────────────────────────────────
  useEffect(() => {
    if (previewId != null) {
      previewTimerRef.current = window.setInterval(
        () => setPreviewTs(Date.now()),
        1500,
      );
    } else {
      if (previewTimerRef.current) window.clearInterval(previewTimerRef.current);
    }
    return () => {
      if (previewTimerRef.current) window.clearInterval(previewTimerRef.current);
    };
  }, [previewId]);

  // ── Handlers ──────────────────────────────────────────────────────────────
  const openAddModal = () => {
    setEditId(null);
    setForm(BLANK_FORM);
    setFormError("");
    setTestResult(null);
    setShowModal(true);
  };

  const openEditModal = (cam: Camera) => {
    const parsed = cam.source_type === "rtsp" ? parseRtspUrl(cam.stream_url) : null;
    setEditId(cam.id);
    setForm({
      name: cam.name,
      location: cam.location ?? "",
      source_type: cam.source_type,
      camera_purpose: cam.camera_purpose,
      camera_ip: parsed?.host ?? "",
      rtsp_port: parsed?.port ?? "554",
      rtsp_username: parsed?.username ?? "",
      rtsp_password: parsed?.password ?? "",
      rtsp_channel: parsed?.channel ?? "1",
      stream_url: cam.stream_url,
      threshold: String(cam.threshold),
      interval_sec: String(cam.interval_sec),
      enabled: cam.enabled,
    });
    setFormError(parsed ? "" : (cam.source_type === "rtsp" ? "This camera URL is not in the expected RTSP format. You can still test or replace it below." : ""));
    setTestResult(null);
    setShowModal(true);
  };

  const handleTestConnection = async () => {
    try {
      const streamUrl = form.source_type === "rtsp" ? buildRtspUrl(form) : form.stream_url.trim();
      if (!streamUrl) {
        setTestResult({ ok: false, msg: "Enter a stream URL or fill in the RTSP fields first." });
        return;
      }
      setTesting(true);
      setTestResult(null);
      const res = await camerasApi.testConnection({
        source_url: streamUrl,
        source_type: form.source_type,
      });
      const d = res.data as { ok?: boolean; frame_size?: { width: number; height: number }; fps?: number; message?: string; details?: string };
      if (d.ok === false) {
        setTestResult({ ok: false, msg: d.details ?? d.message ?? "Connection failed." });
      } else {
        const width = d.frame_size?.width ?? "?";
        const height = d.frame_size?.height ?? "?";
        const fps = typeof d.fps === "number" ? d.fps.toFixed(1) : "?";
        setTestResult({
          ok: true,
          msg: `✓ Connected! ${width}×${height} @ ${fps}fps`,
        });
      }
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      console.error("Camera connection test failed", err);
      setTestResult({ ok: false, msg: detail ?? "Connection failed." });
    } finally {
      setTesting(false);
    }
  };

  const handleSave = async () => {
    if (!form.name.trim()) { setFormError("Camera name is required."); return; }
    setSaving(true);
    setFormError("");
    try {
      const streamUrl = form.source_type === "rtsp" ? buildRtspUrl(form) : form.stream_url.trim();
      if (!streamUrl) {
        setFormError("Stream URL is required.");
        return;
      }
      const payload = {
        name: form.name.trim(),
        location: form.location.trim() || undefined,
        source_url: streamUrl,
        source_type: form.source_type,
        camera_purpose: form.camera_purpose,
        threshold: Number.parseFloat(form.threshold) || 0.45,
        interval_sec: Number.parseFloat(form.interval_sec) || 2,
        enabled: form.enabled,
      };
      if (editId != null) {
        await camerasApi.update(editId, payload);
      } else {
        await camerasApi.create(payload);
      }
      setShowModal(false);
      await fetchCameras();
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      console.error("Failed to save camera", err);
      setFormError(detail ?? (err instanceof Error ? err.message : "Save failed."));
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = async () => {
    if (deleteId == null) return;
    setDeleting(true);
    try {
      await camerasApi.remove(deleteId);
      setDeleteId(null);
      await fetchCameras();
    } catch (err) {
      console.error("Failed to delete camera", err);
      setDeleteId(null);
      setError("Failed to delete camera.");
    } finally {
      setDeleting(false);
    }
  };

  const handleStart = async (id: number) => {
    try {
      await camerasApi.start(id);
      await fetchCameras();
    } catch (err) {
      console.error("Failed to start camera", err);
      setError("Failed to start camera.");
    }
  };

  const handleStop = async (id: number) => {
    try {
      await camerasApi.stop(id);
      await fetchCameras();
    } catch (err) {
      console.error("Failed to stop camera", err);
      setError("Failed to stop camera.");
    }
  };

  const handleRestart = async (id: number) => {
    try {
      await camerasApi.restart(id);
      await fetchCameras();
    } catch (err) {
      console.error("Failed to restart camera", err);
      setError("Failed to restart camera.");
    }
  };

  // ── DVR Discovery Handlers ──────────────────────────────────────────────────
  const handleDiscoverDVR = async () => {
    if (!dvrForm.ip.trim() || !dvrForm.username.trim() || !dvrForm.password.trim()) {
      setDiscoveryError("Please fill in all DVR credentials.");
      return;
    }
    setDiscovering(true);
    setDiscoveryError("");
    setDiscoveredDevice(null);
    setSelectedChannels(new Set());
    try {
      const res = await camerasApi.discoverDVR({
        ip: dvrForm.ip.trim(),
        port: parseInt(dvrForm.port) || 8000,
        username: dvrForm.username.trim(),
        password: dvrForm.password.trim(),
      });
      const data = res.data as { success: boolean; device?: DiscoveredDevice; error?: string };
      if (data.success && data.device) {
        setDiscoveredDevice(data.device);
      } else {
        setDiscoveryError(data.error || "Discovery failed.");
      }
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      console.error("DVR discovery failed", err);
      setDiscoveryError(detail ?? "Discovery failed. Check your DVR credentials and network connection.");
    } finally {
      setDiscovering(false);
    }
  };

  const handleToggleChannel = (channelId: number) => {
    setSelectedChannels((prev) => {
      const newSet = new Set(prev);
      if (newSet.has(channelId)) {
        newSet.delete(channelId);
      } else {
        newSet.add(channelId);
      }
      return newSet;
    });
  };

  const handleAddDiscoveredCameras = async () => {
    if (!discoveredDevice || selectedChannels.size === 0) return;
    setSaving(true);
    setFormError("");
    try {
      for (const channelId of selectedChannels) {
        const channel = discoveredDevice.channels.find((ch) => ch.id === channelId);
        if (!channel) continue;
        
        const sourceUrl = `hcnetsdk://${dvrForm.ip}:${dvrForm.port}@${dvrForm.username}:${dvrForm.password}?channel=${channelId}`;
        await camerasApi.create({
          name: channel.name,
          location: discoveredDevice.serial.substring(0, 20),
          source_url: sourceUrl,
          source_type: "hcnetsdk",
          camera_purpose: "IN",
          threshold: 0.45,
          interval_sec: 2,
          enabled: false,
        });
      }
      setShowDiscoveryModal(false);
      setDiscoveredDevice(null);
      setSelectedChannels(new Set());
      await fetchCameras();
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      console.error("Failed to add cameras", err);
      setDiscoveryError(detail ?? (err instanceof Error ? err.message : "Failed to add cameras."));
    } finally {
      setSaving(false);
    }
  };

  // ── Styles ─────────────────────────────────────────────────────────────────
  const cardStyle: React.CSSProperties = {
    background: "rgba(255,255,255,0.04)",
    border: "1px solid rgba(255,255,255,0.08)",
    borderRadius: 16,
    padding: "1.2rem 1.4rem",
    transition: "border-color 0.2s",
  };

  const inputStyle: React.CSSProperties = {
    width: "100%", borderRadius: 10,
    border: "1px solid rgba(255,255,255,0.15)",
    padding: "0.75rem 0.9rem",
    background: "#0b1220", color: "#fff",
    fontSize: "0.9rem", outline: "none",
    boxSizing: "border-box",
  };

  const labelStyle: React.CSSProperties = {
    fontSize: "0.8rem", color: "rgba(255,255,255,0.55)",
    marginBottom: "0.35rem", display: "block", fontWeight: 600,
  };

  const btnStyle = (variant: "primary" | "secondary" | "danger" | "ghost"): React.CSSProperties => ({
    padding: "0.45rem 0.9rem",
    borderRadius: 8,
    border: "1px solid",
    cursor: "pointer",
    fontSize: "0.82rem",
    fontWeight: 600,
    transition: "all 0.15s",
    borderColor: variant === "primary" ? "#6366f1"
      : variant === "danger" ? "rgba(239,68,68,0.5)"
      : variant === "ghost" ? "rgba(255,255,255,0.12)"
      : "rgba(255,255,255,0.2)",
    background: variant === "primary" ? "#6366f1"
      : variant === "danger" ? "rgba(239,68,68,0.12)"
      : "rgba(255,255,255,0.05)",
    color: variant === "danger" ? "#fca5a5"
      : variant === "primary" ? "#fff" : "rgba(255,255,255,0.8)",
  });

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="page-stack">
      {/* Header */}
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center", flexWrap: "wrap", gap: "1rem" }}>
        <div>
          <h1 className="page-title">📷 Camera Manager</h1>
          <div className="page-subtitle">Manage Hikvision DVR channels for automatic attendance</div>
        </div>
        <div style={{ display: "flex", gap: "0.75rem", alignItems: "center" }}>
          <GlobalHeaderControls />
          <button type="button" style={btnStyle("secondary")} onClick={() => setShowDiscoveryModal(true)}>🔍 Discover DVR</button>
          <button type="button" style={btnStyle("primary")} onClick={openAddModal}>+ Add Camera</button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{ background: "rgba(239,68,68,0.12)", border: "1px solid rgba(239,68,68,0.3)", borderRadius: 12, padding: "0.9rem 1.2rem", color: "#fca5a5" }}>
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div style={{ color: "rgba(255,255,255,0.5)", padding: "3rem", textAlign: "center" }}>
          Loading cameras...
        </div>
      )}

      {/* Empty state */}
      {!loading && cameraList.length === 0 && !error && (
        <div style={{ ...cardStyle, textAlign: "center", padding: "3rem", color: "rgba(255,255,255,0.4)" }}>
          <div style={{ fontSize: "3rem", marginBottom: "1rem" }}>📷</div>
          <div style={{ fontSize: "1.1rem", fontWeight: 600, marginBottom: "0.5rem" }}>No cameras configured</div>
          <div style={{ fontSize: "0.9rem", marginBottom: "1.5rem" }}>Add your Hikvision DVR channels to get started</div>
          <button type="button" style={btnStyle("primary")} onClick={openAddModal}>Add First Camera</button>
        </div>
      )}

      {/* Camera grid */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(360px, 1fr))", gap: "1.25rem" }}>
        {cameraList.map((cam) => {
          const live = cam.live;
          return (
            <div key={cam.id} style={{
              ...cardStyle,
              borderColor: live?.status === "running"
                ? "rgba(34,197,94,0.25)"
                : live?.status === "error" ? "rgba(239,68,68,0.25)"
                : "rgba(255,255,255,0.08)",
            }}>
              {/* Top row */}
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "0.9rem" }}>
                <div>
                  <div style={{ fontWeight: 700, fontSize: "1rem", marginBottom: "0.25rem" }}>{cam.name}</div>
                  {cam.location && (
                    <div style={{ fontSize: "0.8rem", color: "rgba(255,255,255,0.45)" }}>📍 {cam.location}</div>
                  )}
                </div>
                <div style={{ display: "flex", gap: "0.4rem", flexShrink: 0 }}>
                  {purposeBadge(cam.camera_purpose)}
                </div>
              </div>

              {/* Status */}
              <div style={{ marginBottom: "0.9rem" }}>
                {live ? statusBadge(live.status) : statusBadge("stopped")}
                {live?.fps != null && live.fps > 0 && (
                  <span style={{ marginLeft: "0.6rem", fontSize: "0.8rem", color: "rgba(255,255,255,0.5)" }}>
                    {live.fps} FPS
                  </span>
                )}
                {live?.total_frames != null && (
                  <span style={{ marginLeft: "0.5rem", fontSize: "0.78rem", color: "rgba(255,255,255,0.35)" }}>
                    · {live.total_frames} frames
                  </span>
                )}
              </div>

              {/* Error */}
              {live?.last_error && (
                <div style={{ fontSize: "0.78rem", color: "#fca5a5", marginBottom: "0.8rem", wordBreak: "break-word" }}>
                  ⚠ {live.last_error}
                </div>
              )}

              {/* Metrics */}
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.5rem", marginBottom: "1rem" }}>
                {[
                  ["Last Frame", live?.last_frame_time ? elapsedSince(live.last_frame_time) : "-"],
                  ["Reconnects", String(live?.reconnect_count ?? "-")],
                  ["Threshold", String(cam.threshold)],
                  ["Interval", cam.interval_sec + "s"],
                ].map(([k, v]) => (
                  <div key={k} style={{ background: "rgba(255,255,255,0.04)", borderRadius: 8, padding: "0.45rem 0.7rem" }}>
                    <div style={{ fontSize: "0.72rem", color: "rgba(255,255,255,0.4)", marginBottom: "0.1rem" }}>{k}</div>
                    <div style={{ fontSize: "0.9rem", fontWeight: 600 }}>{v}</div>
                  </div>
                ))}
              </div>

              {/* Stream URL */}
              <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.35)", marginBottom: "1rem", wordBreak: "break-all" }}>
                {cam.stream_url}
              </div>

              {/* Actions */}
              <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                {cam.enabled && live?.status !== "running" && live?.status !== "connecting" ? (
                  <button style={btnStyle("secondary")} onClick={() => void handleRestart(cam.id)}>🔄 Reconnect</button>
                ) : cam.enabled ? (
                  <button style={btnStyle("secondary")} onClick={() => void handleRestart(cam.id)}>🔄 Restart</button>
                ) : null}

                {cam.enabled ? (
                  <button style={btnStyle("secondary")} onClick={() => void handleStop(cam.id)}>⏹ Stop</button>
                ) : (
                  <button style={{ ...btnStyle("secondary"), borderColor: "rgba(34,197,94,0.4)", color: "#86efac" }}
                    onClick={() => void handleStart(cam.id)}>▶ Start</button>
                )}

                <button style={btnStyle("ghost")} onClick={() => setPreviewId(cam.id)}>🖥 Preview</button>
                <button style={btnStyle("ghost")} onClick={() => openEditModal(cam)}>✏ Edit</button>
                <button style={btnStyle("danger")} onClick={() => setDeleteId(cam.id)}>🗑</button>
              </div>
            </div>
          );
        })}
      </div>

      {/* ── Add/Edit Modal ── */}
      {showModal && (
        <div style={{
          position: "fixed", inset: 0, background: "rgba(0,0,0,0.75)", zIndex: 1000,
          display: "flex", alignItems: "center", justifyContent: "center", padding: "1rem",
        }}>
          <div style={{
            background: "#0f1629", border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: 20, padding: "2rem", width: "100%", maxWidth: 560,
            maxHeight: "90vh", overflowY: "auto",
          }}>
            <h2 style={{ margin: "0 0 1.5rem", fontSize: "1.2rem" }}>
              {editId != null ? "Edit Camera" : "Add Camera"}
            </h2>

            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
              <div style={{ gridColumn: "1/-1" }}>
                <label style={labelStyle}>Camera Name *</label>
                <input style={inputStyle} value={form.name}
                  onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
                  placeholder="e.g. Main Entrance Check-In" />
              </div>
              <div style={{ gridColumn: "1/-1" }}>
                <label style={labelStyle}>Location</label>
                <input style={inputStyle} value={form.location}
                  onChange={(e) => setForm((f) => ({ ...f, location: e.target.value }))}
                  placeholder="e.g. Ground Floor Lobby" />
              </div>
              <div>
                <label style={labelStyle}>Camera Purpose *</label>
                <select style={inputStyle} value={form.camera_purpose}
                  onChange={(e) => setForm((f) => ({ ...f, camera_purpose: e.target.value }))}>
                  <option value="IN">⬆ Check-In (Entry Camera)</option>
                  <option value="OUT">⬇ Check-Out (Exit Camera)</option>
                </select>
              </div>
              <div>
                <label style={labelStyle}>Source Type</label>
                <select style={inputStyle} value={form.source_type}
                  onChange={(e) => setForm((f) => ({ ...f, source_type: e.target.value }))}>
                  <option value="rtsp">RTSP (Hikvision DVR)</option>
                  <option value="usb">USB Webcam</option>
                  <option value="http">HTTP MJPEG</option>
                </select>
              </div>
              {form.source_type === "rtsp" ? (
                <>
                  <div>
                    <label style={labelStyle}>Camera IP Address *</label>
                    <input style={inputStyle} value={form.camera_ip}
                      onChange={(e) => setForm((f) => ({ ...f, camera_ip: e.target.value }))}
                      placeholder="192.168.29.181" />
                  </div>
                  <div>
                    <label style={labelStyle}>RTSP Port *</label>
                    <input style={inputStyle} value={form.rtsp_port}
                      onChange={(e) => setForm((f) => ({ ...f, rtsp_port: e.target.value }))}
                      placeholder="554" />
                  </div>
                  <div>
                    <label style={labelStyle}>Username</label>
                    <input style={inputStyle} value={form.rtsp_username}
                      onChange={(e) => setForm((f) => ({ ...f, rtsp_username: e.target.value }))}
                      placeholder="admin" />
                  </div>
                  <div>
                    <label style={labelStyle}>Password</label>
                    <input style={inputStyle} type="password" value={form.rtsp_password}
                      onChange={(e) => setForm((f) => ({ ...f, rtsp_password: e.target.value }))}
                      placeholder="••••••••" />
                  </div>
                  <div style={{ gridColumn: "1/-1" }}>
                    <label style={labelStyle}>Channel Number *</label>
                    <input style={inputStyle} value={form.rtsp_channel}
                      onChange={(e) => setForm((f) => ({ ...f, rtsp_channel: e.target.value }))}
                      placeholder="1" />
                  </div>
                  <div style={{ gridColumn: "1/-1" }}>
                    <label style={labelStyle}>Generated RTSP URL</label>
                    <input
                      style={{ ...inputStyle, opacity: 0.85 }}
                      value={(() => {
                        try {
                          return buildRtspUrl(form);
                        } catch {
                          return "Enter the RTSP fields above to generate the URL.";
                        }
                      })()}
                      readOnly
                    />
                  </div>
                </>
              ) : (
                <div style={{ gridColumn: "1/-1" }}>
                  <label style={labelStyle}>Stream URL *</label>
                  <input style={inputStyle} value={form.stream_url}
                    onChange={(e) => setForm((f) => ({ ...f, stream_url: e.target.value }))}
                    placeholder="http://192.168.29.181:8080/video or another supported stream URL" />
                </div>
              )}
              <div>
                <label style={labelStyle}>Recognition Threshold</label>
                <input style={inputStyle} type="number" min="0.1" max="0.99" step="0.05"
                  value={form.threshold}
                  onChange={(e) => setForm((f) => ({ ...f, threshold: e.target.value }))} />
              </div>
              <div>
                <label style={labelStyle}>Scan Interval (sec)</label>
                <input style={inputStyle} type="number" min="0.5" max="60" step="0.5"
                  value={form.interval_sec}
                  onChange={(e) => setForm((f) => ({ ...f, interval_sec: e.target.value }))} />
              </div>
              <div style={{ gridColumn: "1/-1", display: "flex", alignItems: "center", gap: "0.6rem" }}>
                <input id="cam-enabled" type="checkbox" checked={form.enabled}
                  onChange={(e) => setForm((f) => ({ ...f, enabled: e.target.checked }))} />
                <label htmlFor="cam-enabled" style={{ fontSize: "0.9rem", cursor: "pointer" }}>
                  Enable camera immediately after saving
                </label>
              </div>
            </div>

            {/* Test connection */}
            <div style={{ marginTop: "1.2rem" }}>
              <button style={{ ...btnStyle("secondary"), marginRight: "0.75rem" }}
                onClick={() => void handleTestConnection()} disabled={testing}>
                {testing ? "Testing..." : "🔌 Test Connection"}
              </button>
              {testResult && (
                <span style={{
                  fontSize: "0.82rem",
                  color: testResult.ok ? "#86efac" : "#fca5a5",
                }}>
                  {testResult.msg}
                </span>
              )}
            </div>

            {formError && (
              <div style={{ marginTop: "1rem", color: "#fca5a5", fontSize: "0.85rem" }}>
                {formError}
              </div>
            )}

            <div style={{ marginTop: "1.5rem", display: "flex", gap: "0.75rem", justifyContent: "flex-end" }}>
              <button type="button" style={btnStyle("ghost")} onClick={() => setShowModal(false)}>Cancel</button>
              <button type="button" style={btnStyle("primary")} onClick={() => void handleSave()} disabled={saving}>
                {saving ? "Saving..." : editId != null ? "Update Camera" : "Add Camera"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── Live Preview Modal ── */}
      {previewId != null && (
        <div style={{
          position: "fixed", inset: 0, background: "rgba(0,0,0,0.85)", zIndex: 1000,
          display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: "1rem",
        }}>
          <div style={{ color: "rgba(255,255,255,0.7)", fontSize: "0.85rem" }}>
            Camera #{previewId} – Live Preview (refreshes every 1.5s)
          </div>
          <img
            src={`${camerasApi.previewUrl(previewId)}?t=${previewTs}`}
            alt="Camera preview"
            style={{
              maxWidth: "90vw", maxHeight: "75vh", borderRadius: 12,
              border: "1px solid rgba(255,255,255,0.15)",
              objectFit: "contain", background: "#000",
            }}
            onError={(e) => {
              (e.target as HTMLImageElement).style.display = "none";
            }}
          />
          <button style={btnStyle("secondary")} onClick={() => setPreviewId(null)}>✕ Close Preview</button>
        </div>
      )}

      {/* ── Delete confirm ── */}
      {deleteId != null && (
        <div style={{
          position: "fixed", inset: 0, background: "rgba(0,0,0,0.75)", zIndex: 1000,
          display: "flex", alignItems: "center", justifyContent: "center",
        }}>
          <div style={{
            background: "#0f1629", border: "1px solid rgba(239,68,68,0.3)",
            borderRadius: 16, padding: "2rem", maxWidth: 380, width: "100%", textAlign: "center",
          }}>
            <div style={{ fontSize: "2rem", marginBottom: "0.75rem" }}>⚠️</div>
            <div style={{ fontSize: "1rem", fontWeight: 700, marginBottom: "0.5rem" }}>Delete Camera?</div>
            <div style={{ fontSize: "0.85rem", color: "rgba(255,255,255,0.5)", marginBottom: "1.5rem" }}>
              This will stop the stream and remove all configuration. Attendance events already recorded will be kept.
            </div>
            <div style={{ display: "flex", gap: "0.75rem", justifyContent: "center" }}>
              <button type="button" style={btnStyle("ghost")} onClick={() => setDeleteId(null)}>Cancel</button>
              <button type="button" style={btnStyle("danger")} onClick={() => void handleDelete()} disabled={deleting}>
                {deleting ? "Deleting..." : "Delete Camera"}
              </button>
            </div>
          </div>
        </div>
      )}

      {/* ── DVR Discovery Modal ── */}
      {showDiscoveryModal && (
        <div style={{
          position: "fixed", inset: 0, background: "rgba(0,0,0,0.75)", zIndex: 1000,
          display: "flex", alignItems: "center", justifyContent: "center", padding: "1rem",
        }}>
          <div style={{
            background: "#0f1629", border: "1px solid rgba(255,255,255,0.12)",
            borderRadius: 20, padding: "2rem", width: "100%", maxWidth: 700,
            maxHeight: "90vh", overflowY: "auto",
          }}>
            <h2 style={{ margin: "0 0 1.5rem", fontSize: "1.2rem" }}>
              🔍 Discover Hikvision DVR Cameras
            </h2>

            {!discoveredDevice ? (
              <>
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

                {discoveryError && (
                  <div style={{ marginTop: "1rem", color: "#fca5a5", fontSize: "0.85rem" }}>
                    {discoveryError}
                  </div>
                )}

                <div style={{ marginTop: "1.5rem", display: "flex", gap: "0.75rem", justifyContent: "flex-end" }}>
                  <button type="button" style={btnStyle("ghost")} onClick={() => setShowDiscoveryModal(false)}>Cancel</button>
                  <button type="button" style={btnStyle("primary")} onClick={() => void handleDiscoverDVR()} disabled={discovering}>
                    {discovering ? "Discovering..." : "🔍 Discover Cameras"}
                  </button>
                </div>
              </>
            ) : (
              <>
                <div style={{ marginBottom: "1.5rem", padding: "1rem", background: "rgba(255,255,255,0.04)", borderRadius: 12 }}>
                  <div style={{ fontSize: "0.9rem", fontWeight: 600, marginBottom: "0.5rem" }}>Device Information</div>
                  <div style={{ fontSize: "0.85rem", color: "rgba(255,255,255,0.7)" }}>
                    <div>Model: {discoveredDevice.model}</div>
                    <div>Firmware: {discoveredDevice.firmware}</div>
                    <div>Serial: {discoveredDevice.serial}</div>
                    <div>Total Channels: {discoveredDevice.total_channels}</div>
                  </div>
                </div>

                <div style={{ marginBottom: "1rem", fontSize: "0.9rem", fontWeight: 600 }}>
                  Select cameras to add ({selectedChannels.size} selected):
                </div>

                <div style={{ maxHeight: "300px", overflowY: "auto", marginBottom: "1rem" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse" }}>
                    <thead>
                      <tr style={{ borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
                        <th style={{ padding: "0.75rem", textAlign: "left", fontSize: "0.85rem", color: "rgba(255,255,255,0.6)" }}>Select</th>
                        <th style={{ padding: "0.75rem", textAlign: "left", fontSize: "0.85rem", color: "rgba(255,255,255,0.6)" }}>Channel</th>
                        <th style={{ padding: "0.75rem", textAlign: "left", fontSize: "0.85rem", color: "rgba(255,255,255,0.6)" }}>Name</th>
                        <th style={{ padding: "0.75rem", textAlign: "left", fontSize: "0.85rem", color: "rgba(255,255,255,0.6)" }}>Status</th>
                        <th style={{ padding: "0.75rem", textAlign: "left", fontSize: "0.85rem", color: "rgba(255,255,255,0.6)" }}>Type</th>
                      </tr>
                    </thead>
                    <tbody>
                      {discoveredDevice.channels.map((channel) => (
                        <tr key={channel.id} style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                          <td style={{ padding: "0.75rem" }}>
                            <input
                              type="checkbox"
                              checked={selectedChannels.has(channel.id)}
                              onChange={() => handleToggleChannel(channel.id)}
                              style={{ cursor: "pointer" }}
                            />
                          </td>
                          <td style={{ padding: "0.75rem", fontSize: "0.9rem" }}>{channel.id}</td>
                          <td style={{ padding: "0.75rem", fontSize: "0.9rem" }}>{channel.name}</td>
                          <td style={{ padding: "0.75rem", fontSize: "0.85rem", color: channel.status === "online" ? "#86efac" : "#fca5a5" }}>
                            {channel.status}
                          </td>
                          <td style={{ padding: "0.75rem", fontSize: "0.85rem", color: "rgba(255,255,255,0.6)" }}>
                            {channel.channel_type}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {discoveryError && (
                  <div style={{ marginBottom: "1rem", color: "#fca5a5", fontSize: "0.85rem" }}>
                    {discoveryError}
                  </div>
                )}

                <div style={{ display: "flex", gap: "0.75rem", justifyContent: "flex-end" }}>
                  <button type="button" style={btnStyle("ghost")} onClick={() => {
                    setDiscoveredDevice(null);
                    setSelectedChannels(new Set());
                  }}>Back</button>
                  <button type="button" style={btnStyle("primary")} onClick={() => void handleAddDiscoveredCameras()} disabled={saving || selectedChannels.size === 0}>
                    {saving ? "Adding..." : `Add ${selectedChannels.size} Camera${selectedChannels.size !== 1 ? "s" : ""}`}
                  </button>
                </div>
              </>
            )}
          </div>
        </div>
      )}
    </div>
  );
}


