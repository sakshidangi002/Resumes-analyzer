/**
 * Real-time notification WebSocket client (singleton).
 *
 * Behaviour:
 *   - One open socket per browser tab (multi-tab is fine — backend tracks
 *     one set of sockets per user_id).
 *   - JWT is appended as ?token=… on the handshake URL because browsers
 *     can't send custom headers on the WebSocket upgrade.
 *   - Exponential backoff reconnect: 1 s → 2 s → 4 s → … capped at 30 s,
 *     jittered, until the auth token disappears or the page unloads.
 *   - Incoming frames are re-broadcast on the global window as DOM
 *     `CustomEvent`s so any component (bell badge, toast host, inbox page)
 *     can subscribe without us depending on a state library.
 *
 * Events dispatched on `window`:
 *   - `softwiz:notif-realtime`  — detail = raw notification row (id, kind,
 *                                 title, body, link_path, created_at, ...)
 *                                 Use to show a toast and bump the badge.
 *   - `softwiz:notif-refresh`   — detail = undefined. Reuses the existing
 *                                 event name the bell already listens to,
 *                                 so legacy code keeps working.
 *   - `softwiz:notif-ws-state`  — detail = { state: "connecting" | "open"
 *                                  | "closed" | "auth-failed" }
 */

let socket: WebSocket | null = null;
let manuallyClosed = true;
let retryAttempt = 0;
let reconnectTimer: number | null = null;
let pingTimer: number | null = null;

export type NotifRow = {
  id: number;
  user_id: number;
  title: string;
  body: string | null;
  kind: string;
  link_path: string | null;
  read_at: string | null;
  created_at: string;
};

type WsState = "connecting" | "open" | "closed" | "auth-failed";

function wsUrl(token: string): string {
  // VITE_WS_BASE_URL is optional — most deployments serve the WS on the
  // same origin as the SPA, so we derive ws://host or wss://host from the
  // current window location.
  const explicit = (import.meta.env.VITE_WS_BASE_URL || "").toString().trim();
  let base: string;
  if (explicit) {
    base = explicit.replace(/\/$/, "");
  } else {
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    base = `${protocol}//${window.location.host}`;
  }
  const qsToken = encodeURIComponent(token);
  return `${base}/ws/notifications?token=${qsToken}`;
}

function emit(name: string, detail?: unknown): void {
  try {
    window.dispatchEvent(new CustomEvent(name, { detail }));
  } catch {
    /* ignore */
  }
}

function emitState(state: WsState): void {
  emit("softwiz:notif-ws-state", { state });
}

function clearTimers(): void {
  if (reconnectTimer != null) {
    window.clearTimeout(reconnectTimer);
    reconnectTimer = null;
  }
  if (pingTimer != null) {
    window.clearInterval(pingTimer);
    pingTimer = null;
  }
}

function scheduleReconnect(): void {
  if (manuallyClosed) return;
  // 1 s, 2 s, 4 s, 8 s, 16 s, 30 s (capped), with ±20% jitter.
  const base = Math.min(30_000, 1000 * 2 ** Math.min(retryAttempt, 5));
  const jitter = base * 0.2 * (Math.random() * 2 - 1);
  const delay = Math.max(500, Math.round(base + jitter));
  retryAttempt += 1;
  if (reconnectTimer != null) window.clearTimeout(reconnectTimer);
  reconnectTimer = window.setTimeout(() => {
    reconnectTimer = null;
    connect();
  }, delay);
}

function connect(): void {
  const token = localStorage.getItem("token");
  if (!token) {
    // Not logged in yet — nothing to do; the next login call to `start()`
    // will retry.
    return;
  }
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) {
    return;
  }
  emitState("connecting");
  try {
    socket = new WebSocket(wsUrl(token));
  } catch {
    emitState("closed");
    scheduleReconnect();
    return;
  }

  socket.onopen = () => {
    retryAttempt = 0;
    emitState("open");
    // App-level keepalive every 25 s. Backend also pings idle sockets every
    // 45 s; this keeps the connection alive through aggressive proxies (IIS
    // ARR's WebSocket idle timeout defaults to 60 s).
    if (pingTimer != null) window.clearInterval(pingTimer);
    pingTimer = window.setInterval(() => {
      try {
        socket?.send(JSON.stringify({ type: "ping" }));
      } catch {
        /* ignore — close handler will reconnect */
      }
    }, 25_000);
  };

  socket.onmessage = (ev: MessageEvent) => {
    let msg: { type?: string; data?: NotifRow } | null = null;
    try {
      msg = JSON.parse(ev.data);
    } catch {
      return;
    }
    if (!msg || typeof msg !== "object") return;
    switch (msg.type) {
      case "notification":
        if (msg.data) {
          emit("softwiz:notif-realtime", msg.data);
          // Legacy event so the bell badge re-fetches the unread count
          // even if it hasn't subscribed to -realtime yet.
          emit("softwiz:notif-refresh");
        }
        break;
      case "refresh":
        emit("softwiz:notif-refresh");
        break;
      case "hello":
      case "ping":
      case "pong":
      default:
        // Server keepalive / handshake — nothing for the UI to do.
        break;
    }
  };

  socket.onclose = (ev: CloseEvent) => {
    clearTimers();
    socket = null;
    if (ev.code === 1008 || ev.code === 4401) {
      // Auth rejected — don't burn the CPU reconnecting; the user will
      // either log in again or refresh.
      emitState("auth-failed");
      manuallyClosed = true;
      return;
    }
    emitState("closed");
    scheduleReconnect();
  };

  socket.onerror = () => {
    // Errors always precede a close — let `onclose` handle reconnect.
    try {
      socket?.close();
    } catch {
      /* ignore */
    }
  };
}

/**
 * Call once after a successful login (and once on app boot if a token is
 * already in localStorage). Idempotent.
 */
export function startNotificationSocket(): void {
  manuallyClosed = false;
  retryAttempt = 0;
  connect();
}

/** Call on logout — closes the socket and stops reconnect attempts. */
export function stopNotificationSocket(): void {
  manuallyClosed = true;
  clearTimers();
  try {
    socket?.close(1000, "logout");
  } catch {
    /* ignore */
  }
  socket = null;
}
