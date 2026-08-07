import { useCallback, useEffect, useState } from "react";
import { useAuth } from "../auth/AuthContext";
import { queries, type HRQueryRow } from "../api/client";
import CustomSelect from "./CustomSelect";
import { SectionLoader } from "./LoadingState";
import { formatDate, formatTimeIST } from "../utils/dateFormatter";

const CATEGORIES = [
  "Attendance correction",
  "Document update request",
  "Salary query",
  "Leave query",
  "General question",
];

/** Open needs attention (amber), pending is in flight (sky), resolved is
 *  settled (emerald). */
const STATUS_TONES: Record<string, string> = {
  OPEN: " eds-status--warn",
  PENDING: " eds-status--info",
  RESOLVED: " eds-status--present",
};

function StatusBadge({ status }: { status: string }) {
  return (
    <span className={`eds-status${STATUS_TONES[status] || ""}`}>
      <i></i>
      {status.charAt(0) + status.slice(1).toLowerCase()}
    </span>
  );
}

const ChevronDown = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <polyline points="6 9 12 15 18 9" />
  </svg>
);

const InboxIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 8l9 6 9-6" />
    <rect x="3" y="5" width="18" height="14" rx="2.5" />
  </svg>
);

const AVATAR_TINTS = ["eds-avatar--blue", "eds-avatar--green", "eds-avatar--purple", "eds-avatar--rose", ""];

function initialsOf(name: string): string {
  const parts = (name || "").trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "?";
  return parts.slice(0, 2).map((p) => p[0]).join("").toUpperCase();
}

export default function QueriesPanel() {
  const { hasRole } = useAuth();
  const isHR = hasRole("Admin") || hasRole("HR");

  const [items, setItems] = useState<HRQueryRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [statusFilter, setStatusFilter] = useState("");
  const [expanded, setExpanded] = useState<number | null>(null);
  const [replyText, setReplyText] = useState<Record<number, string>>({});
  const [showNew, setShowNew] = useState(false);
  const [newQuery, setNewQuery] = useState({ subject: "", message: "", category: "" });
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const load = useCallback(() => {
    setLoading(true);
    queries
      .list(statusFilter || undefined)
      .then((r) => setItems(r.data))
      .catch(() => setItems([]))
      .finally(() => setLoading(false));
  }, [statusFilter]);

  useEffect(() => {
    load();
  }, [load]);

  const submitNew = (e: React.FormEvent) => {
    e.preventDefault();
    if (!newQuery.subject.trim() || !newQuery.message.trim()) {
      setError("Subject and message are required");
      return;
    }
    setBusy(true);
    setError("");
    queries
      .create({
        subject: newQuery.subject.trim(),
        message: newQuery.message.trim(),
        category: newQuery.category || null,
      })
      .then(() => {
        setNewQuery({ subject: "", message: "", category: "" });
        setShowNew(false);
        load();
      })
      .catch((e: any) => setError(e?.response?.data?.detail || "Failed to send query"))
      .finally(() => setBusy(false));
  };

  const sendReply = (id: number) => {
    const msg = (replyText[id] || "").trim();
    if (!msg) return;
    setBusy(true);
    queries
      .reply(id, msg)
      .then(() => {
        setReplyText((prev) => ({ ...prev, [id]: "" }));
        load();
      })
      .catch((e: any) => setError(e?.response?.data?.detail || "Failed to send reply"))
      .finally(() => setBusy(false));
  };

  const changeStatus = (id: number, status: string) => {
    setBusy(true);
    queries
      .setStatus(id, status)
      .then(load)
      .catch((e: any) => setError(e?.response?.data?.detail || "Failed to update status"))
      .finally(() => setBusy(false));
  };

  return (
    <div>
      <div className="eds-controls" style={{ marginBottom: "1rem" }}>
        <span className="eds-filter-label">Filter</span>
        <CustomSelect
          className="eds-cselect eds-cselect--filter"
          value={statusFilter}
          onChange={(v) => setStatusFilter(String(v))}
          options={[
            { value: "", label: "All statuses" },
            { value: "OPEN", label: "Open" },
            { value: "PENDING", label: "Pending" },
            { value: "RESOLVED", label: "Resolved" },
          ]}
        />
        <span className="eds-tally" style={{ marginLeft: "auto" }}>
          <b>{items.length}</b> {items.length === 1 ? "query" : "queries"}
        </span>
        {!isHR && (
          <button type="button" className="eds-action eds-action--go" onClick={() => { setError(""); setShowNew((s) => !s); }}>
            {showNew ? "Cancel" : "New Query"}
          </button>
        )}
      </div>

      {error && <div className="alert alert-error">{error}</div>}

      {!isHR && showNew && (
        <form className="card" onSubmit={submitNew} style={{ marginBottom: "1rem" }}>
          <div className="form-group">
            <label>Category</label>
            <CustomSelect
              value={newQuery.category}
              onChange={(v) => setNewQuery({ ...newQuery, category: String(v) })}
              placeholder="Select a category"
              options={[{ value: "", label: "General question" }, ...CATEGORIES.map((c) => ({ value: c, label: c }))]}
            />
          </div>
          <div className="form-group">
            <label>Subject</label>
            <input value={newQuery.subject} onChange={(e) => setNewQuery({ ...newQuery, subject: e.target.value })} required />
          </div>
          <div className="form-group">
            <label>Message</label>
            <textarea rows={4} value={newQuery.message} onChange={(e) => setNewQuery({ ...newQuery, message: e.target.value })} required />
          </div>
          <button type="submit" className="btn btn-primary" disabled={busy}>{busy ? "Sending…" : "Send to HR"}</button>
        </form>
      )}

      {loading ? (
        <div style={{ padding: "3rem 0" }}><SectionLoader size="md" /></div>
      ) : items.length === 0 ? (
        <section className="eds-card">
          <div className="eds-empty--card">
            <span className="eds-empty-tile"><InboxIcon /></span>
            <span>{isHR ? "No employee queries yet." : "You haven't raised any queries yet."}</span>
          </div>
        </section>
      ) : (
        <section className="eds-card">
          {items.map((q) => {
            const isOpen = expanded === q.id;
            return (
              <div key={q.id} className="eds-query">
                <div
                  className="eds-feed-row"
                  style={{ cursor: "pointer" }}
                  onClick={() => setExpanded(isOpen ? null : q.id)}
                >
                  <span className={`eds-avatar eds-avatar--lg ${AVATAR_TINTS[q.id % AVATAR_TINTS.length]}`}>
                    {initialsOf(q.employee_name || q.subject)}
                  </span>
                  <div className="eds-feed-body">
                    <span className="eds-feed-title" style={{ fontWeight: 600, color: "var(--eds-text)" }}>{q.subject}</span>
                    <span className="eds-feed-meta">
                      {isHR && <span>From: {q.employee_name || "—"} · </span>}
                      {q.category ? <span>{q.category} · </span> : null}
                      {formatDate(q.created_at)} {formatTimeIST(q.created_at)} IST
                      {q.replies.length > 0 && <span> · {q.replies.length} repl{q.replies.length === 1 ? "y" : "ies"}</span>}
                    </span>
                  </div>
                  <div className="eds-feed-actions">
                    <StatusBadge status={q.status} />
                    <span
                      className="eds-iconbtn"
                      style={{ transform: isOpen ? "rotate(180deg)" : "none", transition: "transform 160ms ease" }}
                      aria-hidden
                    >
                      <ChevronDown />
                    </span>
                  </div>
                </div>

                {isOpen && (
                  <div className="eds-query-detail">
                    {/* Original message */}
                    <div className="eds-msg">
                      <div className="eds-msg-head">
                        {q.employee_name || "Employee"} · {formatDate(q.created_at)} {formatTimeIST(q.created_at)}
                      </div>
                      <div className="eds-msg-body">{q.message}</div>
                    </div>

                    {/* Threaded replies */}
                    {q.replies.map((r) => (
                      <div key={r.id} className={`eds-msg eds-msg--reply${r.author_role === "HR" ? " is-hr" : ""}`}>
                        <div className="eds-msg-head">
                          {r.author_name || "—"}{r.author_role ? ` (${r.author_role})` : ""} · {formatDate(r.created_at)} {formatTimeIST(r.created_at)}
                        </div>
                        <div className="eds-msg-body">{r.message}</div>
                      </div>
                    ))}

                    {/* Reply box (both sides can reply unless resolved) */}
                    {q.status !== "RESOLVED" && (
                      <div style={{ marginTop: "0.8rem", display: "flex", gap: "0.5rem", alignItems: "flex-end" }}>
                        <textarea
                          rows={2}
                          placeholder="Write a reply…"
                          value={replyText[q.id] || ""}
                          onChange={(e) => setReplyText((prev) => ({ ...prev, [q.id]: e.target.value }))}
                          style={{ flex: 1 }}
                        />
                        <button type="button" className="eds-action eds-action--go" disabled={busy} onClick={() => sendReply(q.id)}>
                          Reply
                        </button>
                      </div>
                    )}

                    {/* HR status controls */}
                    {isHR && (
                      <div style={{ marginTop: "0.8rem", display: "flex", gap: "0.5rem", flexWrap: "wrap" }}>
                        <button type="button" className="eds-action" disabled={busy || q.status === "PENDING"} onClick={() => changeStatus(q.id, "PENDING")}>
                          Mark Pending
                        </button>
                        <button type="button" className="eds-action" disabled={busy || q.status === "RESOLVED"} onClick={() => changeStatus(q.id, "RESOLVED")}>
                          Mark Resolved
                        </button>
                        {q.status === "RESOLVED" && (
                          <button type="button" className="eds-action" disabled={busy} onClick={() => changeStatus(q.id, "OPEN")}>
                            Reopen
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </section>
      )}
    </div>
  );
}
