import { useEffect, useState } from "react";
import { useAuth } from "../auth/AuthContext";
import { leave as leaveApi } from "../api/client";
import ConfirmModal from "../components/ConfirmModal";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import { formatDate } from "../utils/dateFormatter";
import { SectionLoader } from "../components/LoadingState";
import CustomSelect from "../components/CustomSelect";
import { useTableControls } from "../components/dataTable";
import type { SortState } from "../components/dataTable";

interface ApprovalRow {
  id: number;
  employee_id: number;
  employee_code: string;
  employee_name: string;
  leave_type_id: number;
  leave_type_name: string;
  start_date: string;
  end_date: string;
  is_half_day: boolean;
  reason?: string | null;
  status: string;
  applied_at: string;
  requester_is_hr: boolean;
  rejection_reason?: string | null;
  response_comment?: string | null;
  // Paid-Leave split preview (Paid Leave requests only).
  pl_earned?: number | null;
  pl_used?: number | null;
  pl_remaining?: number | null;
  pl_requested?: number | null;
  pl_paid?: number | null;
  pl_unpaid?: number | null;
}

// Paid-Leave breakdown card shown while approving (Earned / Used / Remaining /
// Requested / Paid / Unpaid). Returns null for non-PL rows.
function PaidLeaveBreakdown({ r }: { r: ApprovalRow }) {
  if (r.pl_paid == null && r.pl_unpaid == null) return null;
  const fmt = (v: number | null | undefined) => (v == null ? "-" : String(Number(v)));
  const tiles: Array<[string, string, string?]> = [
    ["Earned Till Date", fmt(r.pl_earned)],
    ["Already Used", fmt(r.pl_used)],
    ["Remaining", fmt(r.pl_remaining)],
    ["Requested", fmt(r.pl_requested)],
    ["Paid Leave", fmt(r.pl_paid), "#22c55e"],
    ["Unpaid (LWP)", fmt(r.pl_unpaid), "#ef4444"],
  ];
  return (
    <div style={{ marginBottom: "1rem" }}>
      <div style={{ fontSize: "0.72rem", textTransform: "uppercase", letterSpacing: "0.04em", opacity: 0.6, marginBottom: 6 }}>
        Paid Leave calculation
      </div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.5rem" }}>
        {tiles.map(([label, value, color]) => (
          <div key={label} style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 8, padding: "0.4rem 0.55rem" }}>
            <div style={{ fontSize: "0.6rem", opacity: 0.6, textTransform: "uppercase", letterSpacing: "0.03em" }}>{label}</div>
            <div style={{ fontSize: "1rem", fontWeight: 800, marginTop: 1, color: color || "#fff" }}>{value}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

/** Identity tint for a requester avatar, stable per request. */
const AVATAR_TINTS = ["eds-avatar--blue", "eds-avatar--green", "eds-avatar--purple", "eds-avatar--rose", ""];

function initialsOf(name: string): string {
  const parts = (name || "").trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "?";
  return parts.slice(0, 2).map((x) => x[0]).join("").toUpperCase();
}

/** Approved settles emerald, rejected rose, pending amber. */
function leaveStatusTone(status: string): string {
  const s = (status || "").toUpperCase();
  if (s === "APPROVED") return " eds-status--present";
  if (s === "REJECTED") return " eds-status--absent";
  if (s === "PENDING") return " eds-status--warn";
  return "";
}

/** Column header for the approvals table. Sorting stays in useTableControls. */
function SortTh({
  label,
  columnKey,
  sort,
  onToggle,
  notSortable,
  className,
}: {
  label: string;
  columnKey: string;
  sort: SortState;
  onToggle: (key: string) => void;
  notSortable?: boolean;
  className?: string;
}) {
  if (notSortable) return <th className={`is-actions ${className || ""}`}>{label}</th>;
  const active = sort.key === columnKey;
  return (
    <th className={className}>
      <button
        type="button"
        className={`eds-sort${active ? " is-active" : ""}`}
        onClick={() => onToggle(columnKey)}
        title={`Sort by ${label}`}
      >
        {label}
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round">
          {sort.direction !== "asc" || !active ? <polyline points="7 15 12 20 17 15" /> : null}
          {sort.direction !== "desc" || !active ? <polyline points="7 9 12 4 17 9" /> : null}
        </svg>
      </button>
    </th>
  );
}

/* 24-box strokes, round caps, currentColor. Size comes from the control that
   holds them (.eds-iconbtn 15px, .eds-action 13px, .eds-search 15px). */
const Icons = {
  Search: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="7.5" />
      <line x1="21" y1="21" x2="16.7" y2="16.7" />
    </svg>
  ),
  Refresh: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20.5 12a8.5 8.5 0 1 1-2.5-6" />
      <polyline points="20.5 4 20.5 9.5 15 9.5" />
    </svg>
  ),
  Leave: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <path d="M17 20v-1.5A3.5 3.5 0 0 0 13.5 15h-6A3.5 3.5 0 0 0 4 18.5V20" />
      <circle cx="10.5" cy="8" r="3.5" />
      <polyline points="17 11 19 13 22 9" />
    </svg>
  ),
  Check: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="20 6 9 17 4 12"></polyline>
    </svg>
  ),
  X: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18"></line>
      <line x1="6" y1="6" x2="18" y2="18"></line>
    </svg>
  ),
  Delete: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6"></polyline>
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
      <line x1="10" y1="11" x2="10" y2="17"></line>
      <line x1="14" y1="11" x2="14" y2="17"></line>
    </svg>
  ),
  Eye: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
      <circle cx="12" cy="12" r="3"></circle>
    </svg>
  ),
};

export default function LeaveApprovals() {
  const { hasRole } = useAuth();
  const canView = hasRole("Admin") || hasRole("HR");
  const isAdmin = hasRole("Admin");
  const isHr = hasRole("HR");

  const [rows, setRows] = useState<ApprovalRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [statusFilter, setStatusFilter] = useState("PENDING");
  const [decision, setDecision] = useState<{ id: number; approved: boolean } | null>(null);
  const [comment, setComment] = useState("");
  const [confirmDeleteId, setConfirmDeleteId] = useState<number | null>(null);
  const [viewDetail, setViewDetail] = useState<ApprovalRow | null>(null);


  const fmtDateTime = (d: string) => {
    if (!d) return "-";
    const dt = new Date(d);
    return Number.isFinite(dt.getTime()) ? formatDate(d) + " " + dt.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) : d;
  };

  const load = () => {
    setLoading(true);
    setError("");
    leaveApi
      .approvals({ status: statusFilter })
      .then((r) => setRows(r.data))
      .catch((err) => setError(err.response?.data?.detail || "Failed to load"))
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    if (canView) load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [statusFilter]);

  const {
    displayed: displayedRows,
    search: approvalSearch,
    setSearch: setApprovalSearch,
    sort: approvalSort,
    toggleSort: toggleApprovalSort,
    clearAll: clearApprovalControls,
    hasActiveControls: approvalHasActive,
  } = useTableControls<ApprovalRow>({
    rows,
    columns: {
      employee: (r) => r.employee_name,
      type: (r) => r.leave_type_name,
      start_date: (r) => r.start_date,
      status: (r) => r.status,
      applied_at: (r) => r.applied_at,
    },
    searchableText: (r) =>
      `${r.employee_code} ${r.employee_name} ${r.leave_type_name} ${r.status} ${r.reason ?? ""}`,
  });

  const openDecision = (id: number, approved: boolean) => {
    setError("");
    setSuccess("");
    setComment("");
    setDecision({ id, approved });
  };

  const submitDecision = (e: React.FormEvent) => {
    e.preventDefault();
    if (!decision) return;
    const msg = comment.trim();
    if (!msg) {
      setError("Please enter a comment/message for the employee.");
      return;
    }
    setError("");
    setSuccess("");
    leaveApi
      .approve(decision.id, decision.approved, msg)
      .then(() => {
        setSuccess(decision.approved ? "Approved." : "Rejected.");
        // Update the row status in-place instead of reloading
        setRows((prev) =>
          prev.map((r) =>
            r.id === decision.id
              ? { ...r, status: decision.approved ? "APPROVED" : "REJECTED" }
              : r
          ).filter((r) => r.status === statusFilter || statusFilter === "")
        );
        setDecision(null);
        setComment("");
        load(); // Still reload to get accurate server state
      })
      .catch((err) => setError(err.response?.data?.detail || "Failed"));
  };

  const deleteRequest = (id: number) => {
    setConfirmDeleteId(id);
  };

  const confirmActualDelete = () => {
    if (!confirmDeleteId) return;
    const deletedId = confirmDeleteId;
    setError("");
    setSuccess("");
    leaveApi
      .deleteRequest(deletedId)
      .then(() => {
        setSuccess("Leave request deleted.");
        setRows((prev) => prev.filter((r) => r.id !== deletedId));
        setConfirmDeleteId(null);
      })
      .catch((err) => setError(err.response?.data?.detail || "Failed"));
  };

  if (!canView) {
    return (
      <div className="eds">
        <div className="eds-page">
          <section className="eds-card">
            <div className="eds-empty--card">
              <span className="eds-empty-tile"><Icons.X /></span>
              <span>Access denied. HR/Admin only.</span>
            </div>
          </section>
        </div>
      </div>
    );
  }

  return (
    <div className="eds">
      <header className="eds-topbar">
        <div>
          <h1 className="eds-title">Leave Approvals</h1>
          <p className="eds-subtitle">Approve or reject employee leave requests</p>
        </div>
        <GlobalHeaderControls />
      </header>

      <div className="eds-page">
      {success && <div className="alert alert-success">{success}</div>}
      {error && <div className="alert alert-error">{error}</div>}

      <div className="eds-controls">
        <CustomSelect
          className="eds-cselect eds-cselect--filter"
          value={statusFilter}
          onChange={(val) => setStatusFilter(val)}
          options={[
            { value: "PENDING", label: "Pending" },
            { value: "APPROVED", label: "Approved" },
            { value: "REJECTED", label: "Rejected" }
          ]}
        />
        <label className="eds-search" style={{ width: 360 }}>
          <Icons.Search />
          <input
            type="search"
            value={approvalSearch}
            onChange={(e) => setApprovalSearch(e.target.value)}
            placeholder="Search by employee, type, status, reason..."
          />
        </label>
        <span className="eds-tally">
          <b>{displayedRows.length}</b> of <b>{rows.length}</b>
        </span>
        {approvalHasActive && (
          <button type="button" className="eds-action" onClick={clearApprovalControls} title="Clear search and sort">
            Clear filters
          </button>
        )}
        <div className="eds-controls-end">
          <button type="button" className="eds-action" onClick={load}>
            <Icons.Refresh />
            Refresh
          </button>
        </div>
      </div>

      <section className="eds-card">
        {loading ? (
          <SectionLoader rows={5} />
        ) : rows.length === 0 ? (
          <div className="eds-empty--card">
            <span className="eds-empty-tile"><Icons.Leave /></span>
            <span>No requests.</span>
          </div>
        ) : (
          <div className="eds-table-wrap">
            <table className="eds-table eds-table--auto leave-approvals-table">
              <thead>
                <tr>
                  <SortTh label="Employee" columnKey="employee" sort={approvalSort} onToggle={toggleApprovalSort} />
                  <SortTh label="Type & Kind" columnKey="type" sort={approvalSort} onToggle={toggleApprovalSort} />
                  <SortTh label="Dates" columnKey="start_date" sort={approvalSort} onToggle={toggleApprovalSort} />
                  <SortTh label="Status" columnKey="status" sort={approvalSort} onToggle={toggleApprovalSort} />
                  <SortTh label="Applied" columnKey="applied_at" sort={approvalSort} onToggle={toggleApprovalSort} className="hide-xl" />
                  <SortTh label="Actions" columnKey="__actions" sort={approvalSort} onToggle={toggleApprovalSort} notSortable />
                </tr>
              </thead>
              <tbody>
                {displayedRows.length === 0 && (
                  <tr>
                    <td colSpan={6} className="eds-table-empty">
                      No requests match your search.
                    </td>
                  </tr>
                )}
                {displayedRows.map((r) => {
                  const isHrLeave = r.requester_is_hr;
                  const canApproveThis = statusFilter === "PENDING" && (
                    (isAdmin && isHrLeave) ||
                    (isHr && !isHrLeave)
                  );
                  const lowerName = r.leave_type_name.toLowerCase();
                  const isUnpaid = lowerName.includes("unpaid") || lowerName.includes("lop");
                  const isShortLeave = lowerName.includes("short");
                  const kindLabel = isShortLeave
                    ? "2 hours"
                    : isUnpaid
                      ? r.is_half_day
                        ? "Half day"
                        : "Full day"
                      : r.is_half_day
                        ? "Half day"
                        : "Full day";

                  const showKind = !lowerName.includes(kindLabel.toLowerCase());

                  return (
                    <tr key={r.id}>
                      <td data-label="Employee">
                        <div className="eds-member">
                          <span className={`eds-avatar eds-avatar--lg ${AVATAR_TINTS[r.id % AVATAR_TINTS.length]}`}>
                            {initialsOf(r.employee_name)}
                          </span>
                          <span className="eds-member-name">
                            {r.employee_name}
                            {isHrLeave && <span className="eds-auto" style={{ marginLeft: 6 }}>HR</span>}
                          </span>
                        </div>
                      </td>
                      <td data-label="Type & Kind">
                        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', whiteSpace: 'nowrap' }}>
                          <span className="eds-type eds-type--sky">{r.leave_type_name}</span>
                          {showKind && <span className="eds-cell-dim">({kindLabel})</span>}
                        </div>
                      </td>
                      <td data-label="Dates" className="eds-cell-time">
                        {formatDate(r.start_date)} to {formatDate(r.end_date)}
                      </td>
                      <td data-label="Status">
                        <span className={`eds-status${leaveStatusTone(r.status)}`}>
                          <i></i>
                          {String(r.status || "-")}
                        </span>
                      </td>
                      <td data-label="Applied" className="hide-xl eds-cell-dim" title={r.applied_at}>
                        {fmtDateTime(r.applied_at)}
                      </td>
                      <td data-label="Actions">
                        <div className="eds-rowactions">
                          <button type="button" className="eds-iconbtn eds-iconbtn--view" onClick={() => setViewDetail(r)} title="View Details">
                            <Icons.Eye />
                          </button>
                          {statusFilter === "PENDING" && canApproveThis && (
                            <>
                              <button type="button" className="eds-iconbtn eds-iconbtn--edit" onClick={() => openDecision(r.id, true)} title="Approve Leave">
                                <Icons.Check />
                              </button>
                              <button type="button" className="eds-iconbtn eds-iconbtn--del" onClick={() => openDecision(r.id, false)} title="Reject Leave">
                                <Icons.X />
                              </button>
                            </>
                          )}
                          {(statusFilter === "PENDING" && !canApproveThis) ? (
                            <span className="eds-cell-dim" style={{ fontSize: 11, whiteSpace: 'nowrap' }}>
                              {isHrLeave ? "Admin only" : "HR only"}
                            </span>
                          ) : statusFilter !== "PENDING" ? (
                            <button type="button" className="eds-iconbtn eds-iconbtn--del" onClick={() => deleteRequest(r.id)} title="Delete Request">
                              <Icons.Delete />
                            </button>
                          ) : null}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </section>
      </div>
      {decision && (
        <div className="modal-backdrop">
          <div className="modal" style={{ maxWidth: 520 }}>
            <h3 style={{ marginTop: 0, marginBottom: '0.75rem' }}>{decision.approved ? "Approve leave" : "Reject leave"}</h3>
            {(() => {
              const decisionRow = rows.find((r) => r.id === decision.id);
              // Show the paid/unpaid split only when approving a Paid Leave request.
              return decision.approved && decisionRow ? <PaidLeaveBreakdown r={decisionRow} /> : null;
            })()}
            <p className="text-muted" style={{ marginTop: 0 }}>
              Enter a professional message. This will be shown to the employee in “My Leave” as the response.
            </p>
            <form onSubmit={submitDecision}>
              <div className="form-group">
                <label>Message / Comment</label>
                <textarea
                  rows={3}
                  value={comment}
                  onChange={(e) => setComment(e.target.value)}
                  placeholder={
                    decision.approved
                      ? "Example: Approved. Please ensure proper handover and enjoy your leave."
                      : "Example: Rejected. Please reapply with correct dates / provide justification."
                  }
                  required
                />
              </div>
              <div style={{ display: "flex", gap: "0.5rem", justifyContent: "flex-end" }}>
                <button type="submit" className={decision.approved ? "btn btn-success" : "btn btn-danger"}>
                  {decision.approved ? "Approve" : "Reject"}
                </button>
                <button type="button" className="btn btn-cancel-alt" onClick={() => setDecision(null)}>
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      <ConfirmModal
        isOpen={!!confirmDeleteId}
        onClose={() => setConfirmDeleteId(null)}
        onConfirm={confirmActualDelete}
        title="Are you absolutely sure?"
        message={
          <>
            You are about to delete leave request <strong>#{confirmDeleteId}</strong>. This action cannot be undone.
          </>
        }
        confirmText="Yes, Delete Request"
      />
      {viewDetail && (
        <div className="modal-backdrop" onClick={() => setViewDetail(null)}>
          <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 500 }}>
            <h3 style={{ marginTop: 0, marginBottom: '0.75rem' }}>Leave Request Details</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div>
                <label className="text-muted" style={{ fontSize: '0.8rem', display: 'block', marginBottom: '2px' }}>Employee</label>
                <div style={{ fontWeight: 600 }}>{viewDetail.employee_name} ({viewDetail.employee_code})</div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div>
                  <label className="text-muted" style={{ fontSize: '0.8rem', display: 'block', marginBottom: '2px' }}>Leave Type</label>
                  <div>{viewDetail.leave_type_name}</div>
                </div>
                <div>
                  <label className="text-muted" style={{ fontSize: '0.8rem', display: 'block', marginBottom: '2px' }}>Kind</label>
                  <div>{viewDetail.is_half_day ? "Half Day" : "Full Day"}</div>
                </div>
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div>
                  <label className="text-muted" style={{ fontSize: '0.8rem', display: 'block', marginBottom: '2px' }}>Start Date</label>
                  <div>{formatDate(viewDetail.start_date)}</div>
                </div>
                <div>
                  <label className="text-muted" style={{ fontSize: '0.8rem', display: 'block', marginBottom: '2px' }}>End Date</label>
                  <div>{formatDate(viewDetail.end_date)}</div>
                </div>
              </div>
              <div>
                <label className="text-muted" style={{ fontSize: '0.8rem', display: 'block', marginBottom: '4px' }}>Reason</label>
                <div style={{
                  padding: '12px',
                  background: 'rgba(255,255,255,0.05)',
                  borderRadius: '8px',
                  border: '1px solid rgba(255,255,255,0.1)',
                  whiteSpace: 'pre-wrap',
                  minHeight: '80px',
                  maxHeight: '200px',
                  overflowY: 'auto',
                  wordBreak: 'break-word'
                }}>
                  {viewDetail.reason || "No reason provided."}
                </div>
              </div>
              <PaidLeaveBreakdown r={viewDetail} />
              {viewDetail.status !== "PENDING" && (
                <div>
                  <label className="text-muted" style={{ fontSize: '0.8rem', display: 'block', marginBottom: '2px' }}>Response/Comment</label>
                  <div style={{ padding: '12px', background: 'rgba(255,255,255,0.03)', borderRadius: '8px' }}>
                    {viewDetail.response_comment || viewDetail.rejection_reason || "-"}
                  </div>
                </div>
              )}
            </div>
            <div className="modal-actions" style={{ marginTop: '1.5rem' }}>
              <button type="button" className="btn btn-secondary btn-uniform" onClick={() => setViewDetail(null)}>Close</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
