import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";

import type { NotifRow } from "../lib/notificationsSocket";

/**
 * Fixed top-right stack of slide-in toasts. Subscribes to the global
 * `softwiz:notif-realtime` event dispatched by the notification WebSocket
 * client, so any component that calls `create_notification` on the backend
 * automatically produces a toast in every open tab — no extra wiring at the
 * call site.
 *
 * Mounted once near the root of the authenticated app (see
 * `layouts/DashboardLayout.tsx`).
 */

type Toast = {
  id: number;
  title: string;
  body: string | null;
  kind: string;
  link_path: string | null;
};

const AUTO_DISMISS_MS = 6000;
const MAX_VISIBLE = 4;

const KIND_ACCENT: Record<string, string> = {
  DSR_REMINDER: "#f59e0b",
  DSR_SUBMITTED: "#10b981",
  LEAVE: "#3b82f6",
  LETTER: "#8b5cf6",
  ONBOARDING: "#22d3ee",
  TASK_HUB: "#22d3ee",
  SYSTEM: "#64748b",
  GENERAL: "#64748b",
};

function accentFor(kind: string): string {
  return KIND_ACCENT[kind] || "rgb(var(--brand-rgb, 79 70 229))";
}

export default function ToastHost() {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const navigate = useNavigate();

  useEffect(() => {
    const onRealtime = (ev: Event) => {
      const detail = (ev as CustomEvent<NotifRow>).detail;
      if (!detail || typeof detail !== "object") return;
      setToasts((prev) => {
        const next: Toast = {
          id: detail.id,
          title: detail.title || "Notification",
          body: detail.body,
          kind: (detail.kind || "GENERAL").toUpperCase(),
          link_path: detail.link_path,
        };
        const deduped = prev.filter((t) => t.id !== next.id);
        return [next, ...deduped].slice(0, MAX_VISIBLE);
      });
    };
    window.addEventListener("softwiz:notif-realtime", onRealtime as EventListener);
    return () => {
      window.removeEventListener(
        "softwiz:notif-realtime",
        onRealtime as EventListener
      );
    };
  }, []);

  useEffect(() => {
    if (toasts.length === 0) return;
    const timers = toasts.map((t) =>
      window.setTimeout(() => {
        setToasts((prev) => prev.filter((x) => x.id !== t.id));
      }, AUTO_DISMISS_MS)
    );
    return () => {
      timers.forEach((tid) => window.clearTimeout(tid));
    };
  }, [toasts]);

  const dismiss = (id: number) =>
    setToasts((prev) => prev.filter((t) => t.id !== id));

  const openLink = (t: Toast) => {
    dismiss(t.id);
    if (t.link_path) navigate(t.link_path);
  };

  if (toasts.length === 0) return null;

  return (
    <div
      aria-live="polite"
      aria-atomic="false"
      style={{
        position: "fixed",
        top: "1rem",
        right: "1rem",
        display: "flex",
        flexDirection: "column",
        gap: "0.6rem",
        zIndex: 12000,
        pointerEvents: "none",
        maxWidth: 380,
      }}
    >
      {toasts.map((t) => (
        <div
          key={t.id}
          role="status"
          onClick={() => openLink(t)}
          style={{
            pointerEvents: "auto",
            background: "rgba(15, 23, 42, 0.95)",
            color: "#f1f5f9",
            border: "1px solid rgba(255,255,255,0.08)",
            borderLeft: `4px solid ${accentFor(t.kind)}`,
            borderRadius: 10,
            padding: "0.75rem 0.85rem",
            cursor: t.link_path ? "pointer" : "default",
            boxShadow: "0 16px 36px -16px rgba(0,0,0,0.55)",
            backdropFilter: "blur(8px)",
            animation: "softwiz-toast-in 220ms ease-out",
          }}
        >
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              gap: "0.75rem",
              alignItems: "flex-start",
            }}
          >
            <div style={{ minWidth: 0, flex: 1 }}>
              <div
                style={{
                  fontWeight: 800,
                  fontSize: "0.85rem",
                  marginBottom: t.body ? 4 : 0,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                }}
                title={t.title}
              >
                {t.title}
              </div>
              {t.body && (
                <div
                  style={{
                    fontSize: "0.78rem",
                    color: "rgba(241,245,249,0.78)",
                    lineHeight: 1.4,
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-word",
                    display: "-webkit-box",
                    WebkitLineClamp: 3,
                    WebkitBoxOrient: "vertical",
                    overflow: "hidden",
                  }}
                >
                  {t.body}
                </div>
              )}
              {t.link_path && (
                <div
                  style={{
                    marginTop: 6,
                    fontSize: "0.72rem",
                    fontWeight: 700,
                    color: accentFor(t.kind),
                  }}
                >
                  Open →
                </div>
              )}
            </div>
            <button
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                dismiss(t.id);
              }}
              aria-label="Dismiss notification"
              style={{
                background: "transparent",
                border: "none",
                color: "rgba(241,245,249,0.5)",
                fontSize: "1rem",
                lineHeight: 1,
                cursor: "pointer",
                padding: 2,
              }}
            >
              ×
            </button>
          </div>
        </div>
      ))}
      <style>{`
        @keyframes softwiz-toast-in {
          from { transform: translateX(24px); opacity: 0; }
          to   { transform: translateX(0);    opacity: 1; }
        }
      `}</style>
    </div>
  );
}
