import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { activity, type AppNotificationRow } from "../api/client";
import ConfirmModal from "../components/ConfirmModal";
import { SectionLoader } from "../components/LoadingState";
import { formatDate } from "../utils/dateFormatter";

function formatKind(kind: string) {
  if (kind === "ONBOARDING" || kind === "TASK_HUB") return "Task";
  return kind
    .toLowerCase()
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

function formatTitle(n: AppNotificationRow) {
  if (n.kind === "ONBOARDING" || n.kind === "TASK_HUB") {
    return n.title.replace(/^onboarding/i, "Task").replace(/^task hub/i, "Task");
  }
  return n.title;
}


import GlobalHeaderControls from "../components/GlobalHeaderControls";

export default function Inbox() {
  const navigate = useNavigate();
  const [items, setItems] = useState<AppNotificationRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [confirmDelete, setConfirmDelete] = useState<number | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    activity
      .notifications({ limit: 100 })
      .then((r) => setItems(r.data))
      .catch(() => setItems([]))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const openItem = async (n: AppNotificationRow) => {
    if (!n.read_at) {
      try {
        await activity.markRead(n.id);
      } catch {
        /* ignore */
      }
    }
    if (n.link_path) {
      navigate(n.link_path);
    } else {
      load();
    }
  };

  const markAll = async () => {
    try {
      await activity.markAllRead();
      load();
    } catch {
      /* ignore */
    }
  };

  const handleDelete = (id: number) => {
    setConfirmDelete(id);
  };

  const confirmActualDelete = async () => {
    if (!confirmDelete) return;
    const deletedId = confirmDelete;
    try {
      await activity.delete(deletedId);
      setConfirmDelete(null);
      setItems((prev) => prev.filter((n) => n.id !== deletedId));
    } catch {
      /* ignore */
    }
  };

  return (
    <div>
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="page-title">Inbox</h1>
          <div className="page-subtitle">Leave decisions, new letters, and task hub updates appear here.</div>
        </div>
        <GlobalHeaderControls />
      </div>

      {items.length > 0 && (<div style={{ display: "flex", gap: "0.5rem", justifyContent: "flex-end", marginBottom: "1rem" }}>
        <button type="button" className="btn btn-secondary btn-sm" onClick={load} title="Refresh Notification List" style={{ backgroundColor: "var(--brand-500)" }}>
          Refresh
        </button>
        <button type="button" className="btn btn-secondary btn-sm" onClick={markAll} title="Mark All Notifications as Read" style={{ backgroundColor: "var(--brand-500)" }}>
          Mark all read
        </button>
      </div>)}
      {/* <p className="text-muted" style={{ marginTop: 0, marginBottom: "1rem", fontSize: "0.9rem" }}>
        Leave decisions, new letters, and task hub updates appear here.
      </p> */}

      {loading ? (
        <div style={{ padding: "3rem 0" }}><SectionLoader size="md" /></div>
      ) : items.length === 0 ? (
        <div className="card" style={{ color: "rgba(255, 255, 255, 0.92)" }}>You have no notifications yet.</div>
      ) : (
        <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
          {items.map((n) => (
            <li
              key={n.id}
              className="card"
              style={{
                marginBottom: "0.4rem",
                cursor: "pointer",
                border: n.read_at ? undefined : "1px solid rgb(var(--brand-rgb) / 0.4)",
                background: n.read_at ? undefined : "rgba(255, 255, 255, 0.06)",
              }}
              onClick={() => openItem(n)}
            >
              <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", alignItems: "flex-start", }}>
                <div style={{ minWidth: 0 }}>
                  <div style={{ fontWeight: 900, color: n.read_at ? "rgba(255, 255, 255, 0.78)" : "rgba(255, 255, 255, 0.96)" }}>{formatTitle(n)}</div>
                  <div style={{ fontSize: "0.75rem", color: "rgba(255, 255, 255, 0.65)", marginTop: 8 }}>
                    {formatDate(n.created_at)} {new Date(n.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    {n.kind && (
                      <span style={{ marginLeft: 8 }}>
                        · {formatKind(n.kind)}
                      </span>
                    )}
                    {!n.read_at && (
                      <span style={{ marginLeft: 8, color: "var(--brand-400)", fontWeight: 800 }}>Unread</span>
                    )}
                  </div>
                </div>
                <div style={{ display: "flex", alignItems: "center", gap: "1rem", flexShrink: 0, marginTop: "auto", marginBottom: "auto" }}>
                  {n.link_path && (
                    <span style={{ fontSize: "0.8rem", color: "var(--brand-400)", fontWeight: 800 }}>Open →</span>
                  )}
                  <button
                    type="button"
                    className="btn btn-secondary btn-icon btn-sm"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(n.id);
                    }}
                    title="Delete Notification"
                    style={{ border: "none", background: "transparent", color: "rgba(255, 255, 255, 0.4)", padding: "4px" }}
                  >
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="3 6 5 6 21 6"></polyline>
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
                      <line x1="10" y1="11" x2="10" y2="17"></line>
                      <line x1="14" y1="11" x2="14" y2="17"></line>
                    </svg>
                  </button>
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}

      <ConfirmModal
        isOpen={!!confirmDelete}
        onClose={() => setConfirmDelete(null)}
        onConfirm={confirmActualDelete}
        title="Delete Notification"
        message="Are you sure you want to delete this notification? This action cannot be undone."
        confirmText="Yes, Delete"
      />
    </div>
  );
}
