import { useCallback, useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { activity, type AppNotificationRow } from "../api/client";
import ConfirmModal from "../components/ConfirmModal";
import QueriesPanel from "../components/QueriesPanel";
import { SectionLoader } from "../components/LoadingState";
import { formatDate, formatTimeIST } from "../utils/dateFormatter";

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

/* 24-box strokes, round caps, currentColor. Size comes from the control that
   holds them (.eds-action 13px, .eds-iconbtn 15px, .eds-empty-tile 20px). */
const Icons = {
  Refresh: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20.5 12a8.5 8.5 0 1 1-2.5-6" />
      <polyline points="20.5 4 20.5 9.5 15 9.5" />
    </svg>
  ),
  CheckAll: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="2 13 7 18 15 8" />
      <polyline points="12 15 15 18 22 9" />
    </svg>
  ),
  ArrowRight: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="4" y1="12" x2="19" y2="12" />
      <polyline points="13 6 19 12 13 18" />
    </svg>
  ),
  Trash: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 21 6" />
      <path d="M8 6V4h8v2" />
      <path d="M6 6l1 14h10l1-14" />
    </svg>
  ),
  Bell: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <path d="M18 8A6 6 0 0 0 6 8c0 7-3 9-3 9h18s-3-2-3-9" />
      <path d="M13.73 21a2 2 0 0 1-3.46 0" />
    </svg>
  ),
};

export default function Inbox() {
  const navigate = useNavigate();
  const [tab, setTab] = useState<"notifications" | "queries">("notifications");
  const [items, setItems] = useState<AppNotificationRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [confirmDelete, setConfirmDelete] = useState<number | null>(null);
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);

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

  const totalPages = Math.max(1, Math.ceil(items.length / pageSize));
  const pageItems = items.slice((page - 1) * pageSize, page * pageSize);

  useEffect(() => {
    if (page > totalPages) setPage(totalPages);
  }, [page, totalPages]);

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

  const unreadCount = items.filter((n) => !n.read_at).length;

  return (
    <div className="eds">
      <header className="eds-topbar">
        <div>
          <h1 className="eds-title">Inbox</h1>
          <p className="eds-subtitle">Leave decisions, new letters, and task hub updates appear here.</p>
        </div>
        <GlobalHeaderControls />
      </header>

      <div className="eds-page">
      {/* Tabs: Notifications (existing feed) + Queries (employee ↔ HR) */}
      <div className="eds-utabs">
        {(["notifications", "queries"] as const).map((t) => (
          <button
            key={t}
            type="button"
            onClick={() => setTab(t)}
            className={`eds-utab${tab === t ? " is-active" : ""}`}
          >
            {t === "notifications" ? "Notifications" : "Queries"}
          </button>
        ))}
      </div>

      {tab === "queries" ? (
        <QueriesPanel />
      ) : (
      <>
      {items.length > 0 && (
        <div className="eds-controls">
          <span className="eds-tally"><b>{unreadCount}</b> unread</span>
          <div className="eds-controls-end">
            <button type="button" className="eds-action" onClick={load} title="Refresh Notification List">
              <Icons.Refresh />
              Refresh
            </button>
            <button type="button" className="eds-action eds-action--go" onClick={markAll} title="Mark All Notifications as Read">
              <Icons.CheckAll />
              Mark all read
            </button>
          </div>
        </div>
      )}

      {loading ? (
        <div style={{ padding: "3rem 0" }}><SectionLoader size="md" /></div>
      ) : items.length === 0 ? (
        <section className="eds-card">
          <div className="eds-empty--card">
            <span className="eds-empty-tile"><Icons.Bell /></span>
            <span>You have no notifications yet.</span>
          </div>
        </section>
      ) : (
        <section className="eds-card">
          <div className="eds-feed">
            {pageItems.map((n) => (
              <div
                key={n.id}
                className={`eds-feed-row${n.read_at ? "" : " is-unread"}`}
                onClick={() => openItem(n)}
                style={{ cursor: "pointer" }}
              >
                <span className="eds-feed-dot"></span>
                <div className="eds-feed-body">
                  <span className="eds-feed-title">{formatTitle(n)}</span>
                  <span className="eds-feed-meta">
                    {formatDate(n.created_at)} {formatTimeIST(n.created_at)} IST
                    {n.kind && <> · {formatKind(n.kind)}</>}
                    {!n.read_at && <> <span className="is-unread-flag">Unread</span></>}
                  </span>
                </div>
                <div className="eds-feed-actions">
                  {n.link_path && (
                    <span className="eds-linkbtn">Open <Icons.ArrowRight /></span>
                  )}
                  <button
                    type="button"
                    className="eds-iconbtn eds-iconbtn--del"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleDelete(n.id);
                    }}
                    title="Delete Notification"
                  >
                    <Icons.Trash />
                  </button>
                </div>
              </div>
            ))}
          </div>
          {totalPages > 1 && (
            <div className="eds-controls" style={{ margin: "1rem 0 0", paddingTop: "1rem", borderTop: "1px solid var(--eds-divider)" }}>
              <span className="eds-tally">
                Showing {(page - 1) * pageSize + 1}-{Math.min(page * pageSize, items.length)} of {items.length}
              </span>
              <div className="eds-pager">
                <label className="eds-pager-rows">
                  Rows
                  <select value={pageSize} onChange={(e) => { setPageSize(Number(e.target.value)); setPage(1); }}>
                    <option value={10}>10</option>
                    <option value={20}>20</option>
                    <option value={50}>50</option>
                  </select>
                </label>
                <button type="button" className="eds-action" onClick={() => setPage(1)} disabled={page === 1}>First</button>
                <button type="button" className="eds-action" onClick={() => setPage((current) => Math.max(1, current - 1))} disabled={page === 1}>Prev</button>
                <span className="eds-pager-rows">Page {page} of {totalPages}</span>
                <button type="button" className="eds-action" onClick={() => setPage((current) => Math.min(totalPages, current + 1))} disabled={page === totalPages}>Next</button>
                <button type="button" className="eds-action" onClick={() => setPage(totalPages)} disabled={page === totalPages}>Last</button>
              </div>
            </div>
          )}
        </section>
      )}
      </>
      )}
      </div>

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
