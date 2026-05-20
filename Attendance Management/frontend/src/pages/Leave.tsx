import { useState, useEffect } from "react";
import { leave as api, company as companyApi } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { SectionLoader } from "../components/LoadingState";
import CustomSelect from "../components/CustomSelect";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import { formatDate } from "../utils/dateFormatter";
import ConfirmModal from "../components/ConfirmModal";
import { useTableControls, SortableHeader, TableToolbar } from "../components/dataTable";

type LeaveKind = "FULL_DAY" | "HALF_DAY" | "SHORT" | "UNPAID";

function diffDaysInclusive(startISO: string, endISO: string): number {
  if (!startISO || !endISO) return 0;
  const s = new Date(startISO + "T12:00:00");
  const e = new Date(endISO + "T12:00:00");
  if (Number.isNaN(s.getTime()) || Number.isNaN(e.getTime())) return 0;
  const ms = e.getTime() - s.getTime();
  const days = Math.floor(ms / (1000 * 60 * 60 * 24)) + 1;
  return days > 0 ? days : 0;
}

const Icons = {
  Eye: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
      <circle cx="12" cy="12" r="3"></circle>
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
};

export default function Leave() {
  const { hasRole, user } = useAuth();
  const canApply = hasRole("HR") || hasRole("Employee");
  const isAdminOrHr = hasRole("Admin") || hasRole("HR");
  const [currentFy, setCurrentFy] = useState<{ id: number; name?: string | null; start_date: string; end_date: string } | null>(null);
  const [fyOptions, setFyOptions] = useState<Array<{ id: number; name?: string | null; start_date: string; end_date: string; is_current: boolean }>>([]);
  const [selectedFyId, setSelectedFyId] = useState<number | null>(null);
  const [types, setTypes] = useState<Array<{ id: number; code: string; name: string }>>([]);
  const [requests, setRequests] = useState<
    Array<{
      id: number;
      start_date: string;
      end_date: string;
      status: string;
      is_half_day: boolean;
      leave_type_id: number;
      reason?: string | null;
      rejection_reason?: string | null;
      response_comment?: string | null;
    }>
  >([]);
  const [allocations, setAllocations] = useState<Array<{ leave_type_id: number; allocated_days: number; used_days: number; balance_days?: number }>>([]);
  const [loading, setLoading] = useState(true);
  const [applyStart, setApplyStart] = useState("");
  const [applyEnd, setApplyEnd] = useState("");
  const [applyKind, setApplyKind] = useState<LeaveKind>("FULL_DAY");
  const [applyReason, setApplyReason] = useState("");
  const [msg, setMsg] = useState("");
  const [showBalanceDialog, setShowBalanceDialog] = useState(false);
  const [loadWarn, setLoadWarn] = useState("");
  const [viewDetail, setViewDetail] = useState<any | null>(null);

  const fyLabel = (fy: { name?: string | null; start_date: string; end_date: string } | null) => {
    if (!fy) return "";
    if (fy.name) return fy.name;
    const sy = formatDate(fy.start_date);
    const ey = formatDate(fy.end_date);
    return `FY ${sy} - ${ey}`;
  };

  const resolveApplyType = () => {
    const paid = types.find((t) => t.code === "PL" || /paid/i.test(t.name));
    const shortLeave = types.find((t) => t.code === "SL" || /short/i.test(t.name));
    const unpaid = types.find((t) => t.code === "UL" || /unpaid|lop/i.test(t.name));
    switch (applyKind) {
      case "FULL_DAY":
        return { type: paid, is_half_day: false };
      case "HALF_DAY":
        return { type: paid, is_half_day: true };
      case "SHORT":
        return { type: shortLeave, is_half_day: true };
      case "UNPAID":
        return { type: unpaid, is_half_day: false };
      default:
        return { type: undefined, is_half_day: false };
    }
  };

  const computeRequestedUnits = () => {
    if (!applyStart || !applyEnd) return { units: 0, error: "" };
    if (applyStart > applyEnd) return { units: 0, error: "End date must be same or after start date." };
    const sameDay = applyStart === applyEnd;
    if ((applyKind === "HALF_DAY" || applyKind === "SHORT") && !sameDay) {
      return { units: 0, error: "Half day / Short leave can only be applied for a single day." };
    }
    if (applyKind === "HALF_DAY") return { units: 0.5, error: "" };
    if (applyKind === "SHORT") return { units: 1, error: "" }; // 1 short leave (2 hours)
    // FULL_DAY / UNPAID count as days
    const days = diffDaysInclusive(applyStart, applyEnd);
    return { units: days, error: "" };
  };

  useEffect(() => {
    setLoading(true);
    setLoadWarn("");

    Promise.allSettled([companyApi.financialYears(), api.types(), api.requests()])
      .then(async ([fyRes, tRes, rRes]) => {
        // Process Financial Years
        let fyIdToLoad = selectedFyId;
        if (fyRes.status === "fulfilled") {
          const fys = (fyRes.value.data || []) as any[];
          setFyOptions(fys);
          const fy = fys.find((x) => x.is_current) || fys[0] || null;
          setCurrentFy(fy ? { id: fy.id, name: fy.name, start_date: fy.start_date, end_date: fy.end_date } : null);
          if (!fyIdToLoad && fy) {
            fyIdToLoad = fy.id;
            setSelectedFyId(fy.id);
          }
        } else {
          setFyOptions([]);
          setCurrentFy(null);
        }

        // Process Types
        if (tRes.status === "fulfilled") {
          setTypes(tRes.value.data);
        } else {
          setTypes([]);
          setLoadWarn("Failed to load leave types. Please refresh.");
        }

        // Process Requests
        if (rRes.status === "fulfilled") {
          setRequests(rRes.value.data);
        } else {
          setRequests([]);
        }

        // Process Allocations
        if (fyIdToLoad) {
          try {
            const a = await api.allocations(
              isAdminOrHr && user?.employee_id != null
                ? { financial_year_id: fyIdToLoad, employee_id: user.employee_id }
                : { financial_year_id: fyIdToLoad }
            );
            setAllocations(a.data ?? []);
          } catch (err) {
            console.error("Failed to load allocations", err);
            setAllocations([]);
          }
        } else {
          setAllocations([]);
        }
      })
      .catch((err) => {
        console.error("Critical error in leave useEffect", err);
        setLoadWarn("Failed to load data. Please refresh.");
      })
      .finally(() => setLoading(false));
  }, [user?.employee_id, isAdminOrHr]);

  useEffect(() => {
    if (!selectedFyId) return;
    api
      .allocations(
        isAdminOrHr && user?.employee_id != null
          ? { financial_year_id: selectedFyId, employee_id: user.employee_id }
          : { financial_year_id: selectedFyId }
      )
      .then((r) => setAllocations(r.data ?? []))
      .catch(() => setAllocations([]));
  }, [selectedFyId, isAdminOrHr, user?.employee_id]);

  const handleApply = (e: React.FormEvent) => {
    e.preventDefault();
    if (types.length === 0) {
      setMsg("Leave types are not loaded yet. Please refresh the page and try again.");
      return;
    }
    if (!applyStart || !applyEnd) return;
    if (!applyReason.trim()) {
      setMsg("Reason is required.");
      return;
    }
    const calc = computeRequestedUnits();
    if (calc.error) {
      setMsg(calc.error);
      return;
    }

    const paid = types.find((t) => t.code === "PL" || /paid/i.test(t.name));
    const shortLeave = types.find((t) => t.code === "SL" || /short/i.test(t.name));
    const unpaid = types.find((t) => t.code === "UL" || /unpaid|lop/i.test(t.name));

    let leave_type_id: number | undefined;
    let is_half_day = false;

    switch (applyKind) {
      case "FULL_DAY":
        if (!paid) {
          setMsg("Paid leave type is not configured. Please ensure Leave Types include code 'PL' (Paid Leave).");
          return;
        }
        leave_type_id = paid.id;
        is_half_day = false;
        break;
      case "SHORT":
        if (!shortLeave) {
          setMsg("Short Leave (2 hours) type is not configured.");
          return;
        }
        leave_type_id = shortLeave.id;
        is_half_day = true;
        break;
      case "HALF_DAY":
        if (!paid) {
          setMsg("Paid leave type is not configured. Please ensure Leave Types include code 'PL' (Paid Leave).");
          return;
        }
        leave_type_id = paid.id;
        is_half_day = true;
        break;
      case "UNPAID":
        if (!unpaid) {
          setMsg("Unpaid/LOP leave type is not configured.");
          return;
        }
        leave_type_id = unpaid.id;
        is_half_day = false;
        break;
      default:
        return;
    }

    api.apply({ leave_type_id, start_date: applyStart, end_date: applyEnd, is_half_day, reason: applyReason.trim() })
      .then(() => {
        setMsg("Leave applied.");
        // Clear the form so no previous data stays in the dialog
        setApplyStart("");
        setApplyEnd("");
        setApplyKind("FULL_DAY");
        setApplyReason("");
        api.requests().then((res) => setRequests(res.data));
        api.allocations().then((res) => setAllocations(res.data));
      })
      .catch((err) => setMsg(err.response?.data?.detail || "Failed"));
  };

  const [confirmDeleteId, setConfirmDeleteId] = useState<number | null>(null);

  type LeaveRow = (typeof requests)[number];
  const typeNameFor = (id: number) => types.find((t) => t.id === id)?.name ?? String(id);

  const {
    displayed: displayedRequests,
    search: leaveSearch,
    setSearch: setLeaveSearch,
    sort: leaveSort,
    toggleSort: toggleLeaveSort,
    clearAll: clearLeaveControls,
    hasActiveControls: leaveHasActive,
  } = useTableControls<LeaveRow>({
    rows: requests,
    columns: {
      start_date: (r) => r.start_date,
      type: (r) => typeNameFor(r.leave_type_id),
      reason: (r) => r.reason ?? "",
      status: (r) => r.status,
    },
    searchableText: (r) =>
      `${typeNameFor(r.leave_type_id)} ${r.status} ${r.reason ?? ""}`,
  });

  const handleDelete = async () => {
    if (!confirmDeleteId) return;
    try {
      await api.deleteRequest(confirmDeleteId);
      setRequests(prev => prev.filter(r => r.id !== confirmDeleteId));
      setConfirmDeleteId(null);
      setMsg("Leave request deleted.");
      // Refresh allocations too
      api.allocations().then((res) => setAllocations(res.data));
    } catch (err: any) {
      setMsg(err.response?.data?.detail || "Failed to delete request");
      setConfirmDeleteId(null);
    }
  };

  return (
    <>
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="page-title">Leave Management</h1>
          <div className="page-subtitle">Apply for leave, check balance, and track status.</div>
        </div>
        <GlobalHeaderControls />
      </div>
      {loading ? (
        <div style={{ padding: "3rem 0" }}><SectionLoader size="md" /></div>
      ) : (
        <>
          {loadWarn && (
            <div className="card" style={{ borderColor: "rgba(245, 158, 11, 0.35)", background: "rgba(245, 158, 11, 0.08)" }}>
              <p style={{ margin: 0, color: "rgba(255, 255, 255, 0.92)", fontWeight: 800 }}>{loadWarn}</p>
            </div>
          )}
          <div className="card">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "1.5rem", flexWrap: "wrap", marginBottom: "1rem" }}>
              <h3 style={{ margin: 0 }}>Leave balance</h3>
              {fyOptions.length > 0 && (
                <div style={{ display: "flex", alignItems: "center", gap: "0.75rem" }}>
                  <span style={{ fontSize: "0.85rem", opacity: 0.9, color: "#ffffff" }}>Financial Year:</span>
                  <div style={{ minWidth: "320px" }}>
                    <CustomSelect
                      value={String(selectedFyId || "")}
                      onChange={(val) => setSelectedFyId(val ? Number(val) : null)}
                      options={fyOptions.map((fy) => ({
                        value: String(fy.id),
                        label: `${fyLabel(fy)} (${formatDate(fy.start_date)} to ${formatDate(fy.end_date)})${fy.is_current ? " - Current" : ""}`
                      }))}
                    />
                  </div>
                </div>
              )}
            </div>
            {allocations.length === 0 ? (
              <p className="text-muted">
                No leave allocations for the current year. Contact HR to allocate your leaves from Leave Allocations.
              </p>
            ) : (
              <>
                <div className="table-wrap table-wrap--dark" style={{ marginTop: "2.5rem" }}>
                  <table className="table-modern table-modern--dark" style={{ tableLayout: 'fixed', width: '100%' }}>
                    <thead>
                      <tr>
                        <th style={{ width: '25%', textAlign: 'left' }}>LEAVE TYPE</th>
                        <th style={{ width: '25%', textAlign: 'center' }}>
                          <div style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>TOTAL LEAVES</div>
                        </th>
                        <th style={{ width: '25%', textAlign: 'center' }}>
                          <div style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>LEAVES TAKEN</div>
                        </th>
                        <th style={{ width: '25%', textAlign: 'center' }}>
                          <div style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>LEAVES LEFT</div>
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {allocations.map((a) => {
                        const left =
                          a.balance_days != null ? Number(a.balance_days) : Number(a.allocated_days) - Number(a.used_days);
                        return (
                          <tr key={a.leave_type_id}>
                            <td style={{ textAlign: 'left' }}>{types.find((t) => t.id === a.leave_type_id)?.name ?? a.leave_type_id}</td>
                            <td style={{ textAlign: 'center' }}>
                              <div style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>
                                {Math.round(Number(a.allocated_days))}
                              </div>
                            </td>
                            <td style={{ textAlign: 'center' }}>
                              <div style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>
                                {Math.round(Number(a.used_days))}
                              </div>
                            </td>
                            <td style={{ textAlign: 'center', fontWeight: 800 }}>
                              <div style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>
                                {Math.round(left)}
                              </div>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </>
            )}
          </div>
          <div className="card">
            <h3 style={{ marginTop: 0, marginBottom: "1.5rem", fontSize: "1.3rem", fontWeight: 800 }}>Apply leave</h3>
            {canApply ? (
              <form onSubmit={handleApply} style={{ width: "100%", boxSizing: "border-box" }}>
                {/* Form Row: Type & Dates */}
                <div
                  className="leave-form-grid"
                  style={{
                    display: "grid",
                    gridTemplateColumns: "1fr 1fr 1fr",
                    width: "100%",
                    gap: "2rem",
                    boxSizing: "border-box",
                    marginBottom: "10px"
                  }}
                >
                  <div style={{ minWidth: 0 }}>
                    <label style={{ display: "block", marginBottom: "8px", fontWeight: 700, fontSize: "0.78rem", color: "rgba(255,255,255,0.82)", textTransform: "uppercase", letterSpacing: "0.03em" }}>Leave type</label>
                    {(() => {
                      const paid = types.find((t) => t.code === "PL" || /paid/i.test(t.name));
                      const shortLeave = types.find((t) => t.code === "SL" || /short/i.test(t.name));

                      const getBalance = (typeId?: number) => {
                        if (!typeId) return 0;
                        const alloc = allocations.find((a) => a.leave_type_id === typeId);
                        if (!alloc) return 0;
                        return alloc.balance_days != null ? Number(alloc.balance_days) : Number(alloc.allocated_days) - Number(alloc.used_days);
                      };

                      const paidBalance = getBalance(paid?.id);
                      const shortBalance = getBalance(shortLeave?.id);

                      if (paidBalance <= 0 && (applyKind === "FULL_DAY" || applyKind === "HALF_DAY")) {
                        setTimeout(() => setApplyKind("UNPAID"), 0);
                      }
                      if (shortBalance <= 0 && applyKind === "SHORT") {
                        setTimeout(() => setApplyKind("UNPAID"), 0);
                      }

                      return (
                        <CustomSelect
                          value={applyKind}
                          onChange={(val) => setApplyKind(val as LeaveKind)}
                          options={[
                            { value: "FULL_DAY", label: `Paid Leave (Full day) ${paidBalance <= 0 ? "(0 balance)" : ""}`, disabled: paidBalance <= 0 },
                            { value: "HALF_DAY", label: `Paid Leave (Half day) ${paidBalance <= 0 ? "(0 balance)" : ""}`, disabled: paidBalance <= 0 },
                            { value: "SHORT", label: `Short Leave ${shortBalance <= 0 ? "(0 balance)" : ""}`, disabled: shortBalance <= 0 },
                            { value: "UNPAID", label: "Unpaid Leave (LOP)" },
                          ]}
                        />
                      );
                    })()}
                  </div>

                  <div style={{ minWidth: 0 }}>
                    <label style={{ display: "block", marginBottom: "8px", fontWeight: 700, fontSize: "0.78rem", color: "rgba(255,255,255,0.82)", textTransform: "uppercase", letterSpacing: "0.03em" }}>Start date</label>
                    <input
                      type="date"
                      value={applyStart}
                      onChange={(e) => {
                        const val = e.target.value;
                        setApplyStart(val);
                        if (applyEnd && val > applyEnd) {
                          setApplyEnd(val);
                        }
                      }}
                      required
                      className="date-input-white"
                      style={{
                        width: "100%",
                        height: "48px",
                        borderRadius: "10px",
                        background: "rgba(255,255,255,0.03)",
                        border: "1px solid rgba(255,255,255,0.15)",
                        padding: "0 16px",
                        fontSize: "0.88rem"
                      }}
                    />
                  </div>

                  <div style={{ minWidth: 0 }}>
                    <label style={{ display: "block", marginBottom: "8px", fontWeight: 700, fontSize: "0.78rem", color: "rgba(255,255,255,0.82)", textTransform: "uppercase", letterSpacing: "0.03em" }}>End date</label>
                    <input
                      type="date"
                      value={applyEnd}
                      onChange={(e) => setApplyEnd(e.target.value)}
                      min={applyStart}
                      required
                      className="date-input-white"
                      style={{
                        width: "100%",
                        height: "48px",
                        borderRadius: "10px",
                        background: "rgba(255,255,255,0.03)",
                        border: "1px solid rgba(255,255,255,0.15)",
                        padding: "0 16px",
                        fontSize: "0.88rem"
                      }}
                    />
                  </div>
                </div>

                {/* Balance & Calculation Row */}
                <div style={{ marginBottom: "1.5rem", minHeight: "1.2rem", paddingLeft: "0" }}>
                  {(() => {
                    const { type } = resolveApplyType();
                    if (!type) return null;
                    const alloc = allocations.find((a) => a.leave_type_id === type.id);
                    const bal = alloc
                      ? (alloc.balance_days != null ? Number(alloc.balance_days) : Number(alloc.allocated_days) - Number(alloc.used_days))
                      : 0;
                    const calc = computeRequestedUnits();
                    return (
                      <div style={{ fontSize: "0.8rem", color: "#ffffff" }}>
                        Balance: <strong style={{ color: "#ffffff" }}>{Math.round(bal)}</strong>
                        {applyStart && applyEnd && !calc.error && (
                          <span style={{ marginLeft: "12px" }}>
                            - Will use: <strong style={{ color: "#ffffff" }}>{calc.units}</strong>
                          </span>
                        )}
                      </div>
                    );
                  })()}
                </div>

                {/* Reason Section (No Box) */}
                <div style={{ marginTop: "1.5rem", width: "100%" }}>
                  <label style={{ display: "block", marginBottom: "10px", fontWeight: 700, fontSize: "0.78rem", color: "rgba(255,255,255,0.82)", textTransform: "uppercase", letterSpacing: "0.03em" }}>Reason for leave</label>
                  <textarea
                    required
                    value={applyReason}
                    onChange={(e) => setApplyReason(e.target.value)}
                    rows={2}
                    style={{
                      width: "100%",
                      minWidth: "100%",
                      boxSizing: "border-box",
                      background: "rgba(255, 255, 255, 0.05)",
                      border: "1px solid rgba(255, 255, 255, 0.12)",
                      borderRadius: "12px",
                      padding: "1rem",
                      color: "#fff",
                      fontSize: "0.9rem",
                      lineHeight: "1.5",
                      resize: "none",
                      outline: "none"
                    }}
                    placeholder="Please provide a brief reason for your leave request..."
                  />
                </div>

                <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginTop: "1.5rem" }}>
                  <div style={{ flex: 1 }}>
                    {msg && <p className="text-muted" style={{ fontSize: "0.95rem", margin: 0, fontWeight: 600 }}>{msg}</p>}
                  </div>
                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={types.length === 0}
                    style={{
                      padding: "0.85rem 3rem",
                      borderRadius: "10px",
                      fontWeight: 700,
                      fontSize: "0.9rem",
                      boxShadow: "0 6px 20px rgb(var(--brand-rgb) / 0.3)",
                    }}
                  >
                    Apply Leave
                  </button>
                </div>
              </form>
            ) : (
              <p className="text-muted">Admins can only view leave balances and requests; applying for leave is disabled.</p>
            )}
          </div>
          <div className="card">
            <h3 style={{ marginTop: 0 }}>My leave requests</h3>
            {requests.length > 0 && (
              <div style={{ marginBottom: "0.75rem", fontSize: "0.85rem", color: "rgba(255, 255, 255, 0.78)" }}>
                {(() => {
                  const shortType = types.find((t) => t.code === "SL" || /short/i.test(t.name));
                  const unpaidType = types.find((t) => t.code === "UL" || /unpaid|lop/i.test(t.name));

                  const shortLeaves = shortType ? requests.filter((r) => r.leave_type_id === shortType.id).length : 0;
                  const paidFull = requests.filter(
                    (r) => !r.is_half_day && (!unpaidType || r.leave_type_id !== unpaidType.id)
                  ).length;
                  const paidHalf = requests.filter(
                    (r) => r.is_half_day && (!unpaidType || r.leave_type_id !== unpaidType.id) && (!shortType || r.leave_type_id !== shortType.id)
                  ).length;
                  const unpaidFull = unpaidType ? requests.filter((r) => !r.is_half_day && r.leave_type_id === unpaidType.id).length : 0;
                  const unpaidHalf = unpaidType ? requests.filter((r) => r.is_half_day && r.leave_type_id === unpaidType.id).length : 0;
                  return (
                    <>
                      <strong>Summary:</strong>{" "}
                      Paid full day: <strong>{paidFull}</strong>
                      {" "} - Paid half day: <strong>{paidHalf}</strong>
                      {" "} - Short leave: <strong>{shortLeaves}</strong>
                      {(unpaidFull + unpaidHalf) > 0 && (
                        <>
                          {" "} - Unpaid: <strong>{unpaidFull}</strong> full, <strong>{unpaidHalf}</strong> half
                        </>
                      )}
                    </>
                  );
                })()}
              </div>
            )}
            {requests.length > 0 && (
              <TableToolbar
                search={leaveSearch}
                onSearchChange={setLeaveSearch}
                placeholder="Search by type, status, reason..."
                showClear={leaveHasActive}
                onClear={clearLeaveControls}
                count={{ shown: displayedRequests.length, total: requests.length }}
              />
            )}
            <div className="table-wrap table-wrap--dark">
              {
                requests.length > 0 && <table className="table-modern table-modern--dark" style={{ tableLayout: 'fixed', width: '100%' }}>
                  <thead>
                    <tr>
                      <SortableHeader label="DATES" columnKey="start_date" sort={leaveSort} onToggle={toggleLeaveSort} style={{ width: '20%' }} />
                      <SortableHeader label="TYPE" columnKey="type" sort={leaveSort} onToggle={toggleLeaveSort} style={{ width: '20%' }} />
                      <SortableHeader label="REASON" columnKey="reason" sort={leaveSort} onToggle={toggleLeaveSort} style={{ width: '20%' }} />
                      <SortableHeader label="STATUS" columnKey="status" sort={leaveSort} onToggle={toggleLeaveSort} style={{ width: '20%' }} />
                      <SortableHeader label="ACTIONS" columnKey="__actions" sort={leaveSort} onToggle={toggleLeaveSort} notSortable align="center" style={{ width: '20%' }} />
                    </tr>
                  </thead>
                  <tbody>
                    {displayedRequests.length === 0 && (
                      <tr>
                        <td colSpan={5} style={{ textAlign: 'center', padding: '1.25rem', opacity: 0.65 }}>
                          No leave requests match your search.
                        </td>
                      </tr>
                    )}
                    {displayedRequests.map((r) => {
                      const leaveType = types.find((t) => t.id === r.leave_type_id);
                      const typeName = leaveType?.name ?? String(r.leave_type_id);
                      const typeCode = leaveType?.code ?? "";
                      const lowerName = String(typeName).toLowerCase();
                      const isUnpaid = lowerName.includes("unpaid") || lowerName.includes("lop");
                      const isShortLeave = typeCode === "SL" || lowerName.includes("short");
                      const kindLabel = isShortLeave
                        ? "Short Leave (2 hours)"
                        : isUnpaid
                          ? r.is_half_day
                            ? "Half day"
                            : "Full day"
                          : r.is_half_day
                            ? "Half day"
                            : "Full day";
                      return (
                        <tr key={r.id}>
                          <td style={{ whiteSpace: 'nowrap', textAlign: 'left' }}>
                            {formatDate(r.start_date)} - {formatDate(r.end_date)}
                          </td>
                          <td style={{ textAlign: 'left' }}>
                            <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px', alignItems: 'center', justifyContent: 'flex-start' }}>
                              <span>{typeName}</span>
                              <span style={{ opacity: 0.8, fontSize: '0.75rem' }}>({kindLabel})</span>
                            </div>
                          </td>
                          <td style={{ textAlign: 'left' }}>
                            <div className="text-truncate" style={{ width: '100%' }} title={r.reason || "-"}>
                              {r.reason || "-"}
                            </div>
                          </td>
                          <td
                            style={{
                              textAlign: 'left',
                              fontWeight: 900,
                              color:
                                r.status === "APPROVED"
                                  ? "rgba(34, 197, 94, 0.95)"
                                  : r.status === "REJECTED"
                                    ? "rgba(239, 68, 68, 0.95)"
                                    : "rgba(255, 255, 255, 0.88)",
                            }}
                          >
                            {r.status}
                          </td>
                          <td style={{ textAlign: 'center' }}>
                            <div style={{ display: 'flex', gap: '8px', justifyContent: 'center' }}>
                              <button
                                type="button"
                                className="btn btn-secondary btn-icon btn-sm"
                                onClick={() => setViewDetail({ ...r, typeName, kindLabel })}
                                title="View Details"
                              >
                                <Icons.Eye />
                              </button>
                              {r.status === "PENDING" && (
                                <button
                                  type="button"
                                  className="btn btn-danger btn-icon btn-sm"
                                  onClick={() => setConfirmDeleteId(r.id)}
                                  title="Cancel Leave Request"
                                >
                                  <Icons.Delete />
                                </button>
                              )}
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              }
            </div>
            {requests.length === 0 && <p className="text-muted">No requests.</p>}
          </div>

          <ConfirmModal
            isOpen={!!confirmDeleteId}
            onClose={() => setConfirmDeleteId(null)}
            onConfirm={handleDelete}
            title="Cancel Leave Request"
            message="Are you sure you want to cancel this leave request? This action cannot be undone."
            confirmText="Yes, Cancel Request"
          />
        </>
      )}


      {showBalanceDialog && (
        <div className="modal-backdrop" onClick={() => setShowBalanceDialog(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 720 }}>
            <h3 style={{ marginTop: 0 }}>
              My leave balance{currentFy ? ` - FY ${fyLabel(currentFy)}` : ""}
            </h3>
            {allocations.length === 0 ? (
              <p className="text-muted">No allocations.</p>
            ) : (
              <div className="table-wrap table-wrap--dark">
                <table className="table-modern table-modern--dark" style={{ tableLayout: 'fixed', width: '100%' }}>
                  <thead>
                    <tr>
                      <th style={{ width: '25%', textAlign: 'left' }}>LEAVE TYPE</th>
                      <th style={{ width: '25%', textAlign: 'center' }}>
                        <div style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>TOTAL LEAVES</div>
                      </th>
                      <th style={{ width: '25%', textAlign: 'center' }}>
                        <div style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>LEAVES TAKEN</div>
                      </th>
                      <th style={{ width: '25%', textAlign: 'center' }}>
                        <div style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>LEAVES LEFT</div>
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {allocations.map((a) => {
                      const balance =
                        a.balance_days != null ? Number(a.balance_days) : Number(a.allocated_days) - Number(a.used_days);
                      const typeName = types.find((t) => t.id === a.leave_type_id)?.name ?? String(a.leave_type_id);
                      return (
                        <tr key={a.leave_type_id} style={{ height: '50px' }}>
                          <td style={{ textAlign: 'left' }}>
                            <div className="text-truncate" style={{ width: '100%' }} title={typeName}>
                              {typeName}
                            </div>
                          </td>
                          <td style={{ textAlign: 'center' }}>
                            <div style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>
                              {Math.round(Number(a.allocated_days))}
                            </div>
                          </td>
                          <td style={{ textAlign: 'center' }}>
                            <div style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>
                              {Math.round(Number(a.used_days))}
                            </div>
                          </td>
                          <td style={{ textAlign: 'center', fontWeight: 800 }}>
                            <div style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>
                              {Math.round(balance)}
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
            <div style={{ marginTop: "1rem" }}>
              <button type="button" className="btn btn-secondary" onClick={() => setShowBalanceDialog(false)} title="Close Balance View">
                Close
              </button>
            </div>
          </div>
        </div>
      )}
      {viewDetail && (
        <div className="modal-backdrop" onClick={() => setViewDetail(null)}>
          <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 500 }}>
            <h3 style={{ marginTop: 0 }}>Leave Request Details</h3>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                <div>
                  <label className="text-muted" style={{ fontSize: '0.8rem', display: 'block', marginBottom: '2px' }}>Leave Type</label>
                  <div>{viewDetail.typeName}</div>
                </div>
                <div>
                  <label className="text-muted" style={{ fontSize: '0.8rem', display: 'block', marginBottom: '2px' }}>Kind</label>
                  <div>{viewDetail.kindLabel}</div>
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
                  <label className="text-muted" style={{ fontSize: '0.8rem', display: 'block', marginBottom: '4px' }}>Response/Comment</label>
                  <div style={{
                    padding: '12px',
                    background: 'rgba(255,255,255,0.05)',
                    borderRadius: '8px',
                    border: '1px solid rgba(255,255,255,0.1)',
                    whiteSpace: 'pre-wrap',
                    minHeight: '60px',
                    maxHeight: '150px',
                    overflowY: 'auto',
                    wordBreak: 'break-word'
                  }}>
                    {viewDetail.response_comment || viewDetail.rejection_reason || "No response comment provided."}
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
