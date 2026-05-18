import { useEffect, useState } from "react";
import { useAuth } from "../auth/AuthContext";
import { leave as leaveApi } from "../api/client";
import ConfirmModal from "../components/ConfirmModal";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import { formatDate } from "../utils/dateFormatter";
import { SectionLoader } from "../components/LoadingState";
import CustomSelect from "../components/CustomSelect";

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
}

// Premium SVG Icons for Actions
const Icons = {
  Check: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
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

  const statusPillClass = (s: string) => {
    const v = String(s || "").toUpperCase();
    if (v === "APPROVED") return "leave-status-pill leave-status-pill--approved";
    if (v === "REJECTED") return "leave-status-pill leave-status-pill--rejected";
    if (v === "PENDING") return "leave-status-pill leave-status-pill--pending";
    return "leave-status-pill";
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
      <div className="card">
        <p>Access denied. HR/Admin only.</p>
      </div>
    );
  }

  return (
    <>
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="page-title">Leave Approvals</h1>
          <div className="page-subtitle">Approve or reject employee leave requests</div>
        </div>
        <GlobalHeaderControls />
      </div>
      {success && <div className="alert alert-success">{success}</div>}
      {error && <div className="alert alert-error">{error}</div>}

      <div className="card">
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "flex-end", justifyContent: "space-between", gap: "1rem", marginBottom: "1rem" }}>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label>Status</label>
            <CustomSelect
              value={statusFilter}
              onChange={(val) => setStatusFilter(val)}
              style={{ width: "140px", minWidth: "140px" }}
              options={[
                { value: "PENDING", label: "Pending" },
                { value: "APPROVED", label: "Approved" },
                { value: "REJECTED", label: "Rejected" }
              ]}
            />
          </div>
          <div style={{ alignSelf: "flex-end" }}>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={load}
              style={{ width: "140px", minWidth: "140px", height: "42px", backgroundColor: "var(--brand-500)" }}
            >
              Refresh
            </button>
          </div>
        </div>

        {loading ? (
          <SectionLoader rows={5} />
        ) : rows.length === 0 ? (
          <p className="text-muted">No requests.</p>
        ) : (
          <div className="table-wrap table-wrap--dark">
            <table className="table-modern table-modern--dark leave-approvals-table">
              <thead>
                <tr>
                  <th style={{ textAlign: 'left', width: '15%', whiteSpace: 'nowrap' }}>Employee</th>
                  <th style={{ textAlign: 'left', width: '20%', whiteSpace: 'nowrap' }}>Type & Kind</th>
                  <th style={{ textAlign: 'left', width: '22%', whiteSpace: 'nowrap' }}>Dates</th>
                  <th style={{ textAlign: 'left', width: '10%', whiteSpace: 'nowrap' }}>Status</th>
                  <th className="hide-xl" style={{ textAlign: 'left', width: '18%', whiteSpace: 'nowrap' }}>Applied</th>
                  <th className="actions-center" style={{ width: '15%', whiteSpace: 'nowrap', textAlign: 'left' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((r) => {
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
                      <td data-label="Employee" style={{ textAlign: 'left', paddingLeft: '1.5rem', whiteSpace: 'nowrap' }}>
                        <div className="leave-emp-cell">
                          <div className="leave-emp-name" style={{ fontWeight: 600, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                            {r.employee_name}
                            {isHrLeave && <span className="leave-emp-hr" style={{ marginLeft: '6px' }}>HR</span>}
                          </div>
                        </div>
                      </td>
                      <td data-label="Type & Kind" style={{ textAlign: 'center', whiteSpace: 'nowrap' }}>
                        <div style={{ display: 'flex', justifyContent: 'left', alignItems: 'center', width: '100%', whiteSpace: 'nowrap', gap: '0.6rem' }}>
                          <div className="leave-type-pill" style={{ width: 'fit-content' }}>
                            {r.leave_type_name}
                          </div>
                          {showKind && (
                            <span className="text-muted" style={{ fontSize: '0.8rem', opacity: 0.8 }}>
                              ({kindLabel})
                            </span>
                          )}
                        </div>
                      </td>
                      <td data-label="Dates" style={{ whiteSpace: 'nowrap', textAlign: 'center' }}>
                        <div style={{ display: 'flex', justifyContent: 'left', width: '100%', fontWeight: 500 }}>
                          {formatDate(r.start_date)} to {formatDate(r.end_date)}
                        </div>
                      </td>
                      <td data-label="Status" style={{ textAlign: 'center', whiteSpace: 'nowrap' }}>
                        <div style={{ display: 'flex', justifyContent: 'left', width: '100%' }}>
                          <span className={statusPillClass(r.status)}>{String(r.status || "-")}</span>
                        </div>
                      </td>
                      <td data-label="Applied" className="leave-applied hide-xl" title={r.applied_at} style={{ textAlign: 'center', whiteSpace: 'nowrap' }}>
                        <div style={{ display: 'flex', justifyContent: 'left', width: '100%' }}>{fmtDateTime(r.applied_at)}</div>
                      </td>
                      <td data-label="Actions" className="actions-center" style={{ whiteSpace: 'nowrap' }}>
                        <div className="actions-stack" style={{ justifyContent: 'left', display: 'flex', gap: '0.6rem', flexWrap: 'nowrap' }}>
                          <button type="button" className="btn btn-secondary btn-icon btn-sm" onClick={() => setViewDetail(r)} title="View Details">
                            <Icons.Eye />
                          </button>
                          {statusFilter === "PENDING" && canApproveThis && (
                            <>
                              <button type="button" className="btn btn-success btn-icon btn-sm" onClick={() => openDecision(r.id, true)} title="Approve Leave">
                                <Icons.Check />
                              </button>
                              <button type="button" className="btn btn-danger btn-icon btn-sm" onClick={() => openDecision(r.id, false)} title="Reject Leave">
                                <Icons.X />
                              </button>
                            </>
                          )}
                          {(statusFilter === "PENDING" && !canApproveThis) ? (
                            <span className="text-muted" style={{ fontSize: '0.65rem', whiteSpace: 'nowrap' }}>
                              {isHrLeave ? "Admin only" : "HR only"}
                            </span>
                          ) : statusFilter !== "PENDING" ? (
                            <button type="button" className="btn btn-danger btn-icon btn-sm" onClick={() => deleteRequest(r.id)} title="Delete Request">
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
      </div>
      {decision && (
        <div className="modal-backdrop">
          <div className="modal" style={{ maxWidth: 520 }}>
            <h3 style={{ marginTop: 0, marginBottom: '0.75rem' }}>{decision.approved ? "Approve leave" : "Reject leave"}</h3>
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
    </>
  );
}
