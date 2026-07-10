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

const STATUS_COLORS: Record<string, string> = {
  OPEN: "#f59e0b",
  PENDING: "#3b82f6",
  RESOLVED: "#22c55e",
};

function StatusBadge({ status }: { status: string }) {
  const color = STATUS_COLORS[status] || "#9ca3af";
  return (
    <span
      style={{
        fontSize: "0.7rem",
        fontWeight: 800,
        textTransform: "uppercase",
        letterSpacing: "0.04em",
        color,
        border: `1px solid ${color}`,
        borderRadius: 999,
        padding: "2px 10px",
      }}
    >
      {status}
    </span>
  );
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
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "1rem", marginBottom: "1rem", flexWrap: "wrap" }}>
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
          <span className="text-muted" style={{ fontSize: "0.85rem" }}>Filter:</span>
          <div style={{ minWidth: 160 }}>
            <CustomSelect
              value={statusFilter}
              onChange={(v) => setStatusFilter(String(v))}
              options={[
                { value: "", label: "All statuses" },
                { value: "OPEN", label: "Open" },
                { value: "PENDING", label: "Pending" },
                { value: "RESOLVED", label: "Resolved" },
              ]}
            />
          </div>
        </div>
        {!isHR && (
          <button type="button" className="btn btn-primary btn-sm" onClick={() => { setError(""); setShowNew((s) => !s); }}>
            {showNew ? "Cancel" : "New Query"}
          </button>
        )}
      </div>

      {error && <div className="card" style={{ color: "#f87171", marginBottom: "1rem" }}>{error}</div>}

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
        <div className="card" style={{ color: "rgba(255,255,255,0.92)" }}>
          {isHR ? "No employee queries yet." : "You haven't raised any queries yet."}
        </div>
      ) : (
        <ul style={{ listStyle: "none", padding: 0, margin: 0 }}>
          {items.map((q) => {
            const isOpen = expanded === q.id;
            return (
              <li key={q.id} className="card" style={{ marginBottom: "0.5rem" }}>
                <div
                  style={{ display: "flex", justifyContent: "space-between", gap: "1rem", alignItems: "flex-start", cursor: "pointer" }}
                  onClick={() => setExpanded(isOpen ? null : q.id)}
                >
                  <div style={{ minWidth: 0 }}>
                    <div style={{ fontWeight: 800, color: "rgba(255,255,255,0.96)" }}>{q.subject}</div>
                    <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.65)", marginTop: 6 }}>
                      {isHR && <span>From: {q.employee_name || "—"} · </span>}
                      {q.category ? <span>{q.category} · </span> : null}
                      {formatDate(q.created_at)} {formatTimeIST(q.created_at)} IST
                      {q.replies.length > 0 && <span> · {q.replies.length} repl{q.replies.length === 1 ? "y" : "ies"}</span>}
                    </div>
                  </div>
                  <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", flexShrink: 0 }}>
                    <StatusBadge status={q.status} />
                    <span style={{ fontSize: "0.8rem", color: "var(--brand-400)", fontWeight: 800 }}>{isOpen ? "▲" : "▼"}</span>
                  </div>
                </div>

                {isOpen && (
                  <div style={{ marginTop: "0.9rem", borderTop: "1px solid rgba(255,255,255,0.08)", paddingTop: "0.9rem" }}>
                    {/* Original message */}
                    <div style={{ marginBottom: "0.6rem" }}>
                      <div style={{ fontSize: "0.72rem", fontWeight: 700, color: "rgba(255,255,255,0.6)" }}>
                        {q.employee_name || "Employee"} · {formatDate(q.created_at)} {formatTimeIST(q.created_at)}
                      </div>
                      <div style={{ whiteSpace: "pre-wrap", marginTop: 3 }}>{q.message}</div>
                    </div>

                    {/* Threaded replies */}
                    {q.replies.map((r) => (
                      <div
                        key={r.id}
                        style={{
                          marginTop: "0.5rem",
                          padding: "0.5rem 0.7rem",
                          borderRadius: 8,
                          background: r.author_role === "HR" ? "rgba(59,130,246,0.12)" : "rgba(255,255,255,0.05)",
                        }}
                      >
                        <div style={{ fontSize: "0.72rem", fontWeight: 700, color: "rgba(255,255,255,0.7)" }}>
                          {r.author_name || "—"}{r.author_role ? ` (${r.author_role})` : ""} · {formatDate(r.created_at)} {formatTimeIST(r.created_at)}
                        </div>
                        <div style={{ whiteSpace: "pre-wrap", marginTop: 3 }}>{r.message}</div>
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
                        <button type="button" className="btn btn-primary btn-sm" disabled={busy} onClick={() => sendReply(q.id)}>
                          Reply
                        </button>
                      </div>
                    )}

                    {/* HR status controls */}
                    {isHR && (
                      <div style={{ marginTop: "0.8rem", display: "flex", gap: "0.5rem" }}>
                        <button type="button" className="btn btn-secondary btn-sm" disabled={busy || q.status === "PENDING"} onClick={() => changeStatus(q.id, "PENDING")}>
                          Mark Pending
                        </button>
                        <button type="button" className="btn btn-secondary btn-sm" disabled={busy || q.status === "RESOLVED"} onClick={() => changeStatus(q.id, "RESOLVED")}>
                          Mark Resolved
                        </button>
                        {q.status === "RESOLVED" && (
                          <button type="button" className="btn btn-secondary btn-sm" disabled={busy} onClick={() => changeStatus(q.id, "OPEN")}>
                            Reopen
                          </button>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
