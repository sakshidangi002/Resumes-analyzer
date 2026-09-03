import { useEffect, useState } from "react";
import { useAuth } from "../auth/AuthContext";
import { leave as leaveApi, company as companyApi, employees as employeesApi } from "../api/client";
import { SectionLoader } from "../components/LoadingState";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import { formatDate } from "../utils/dateFormatter";
import CustomSelect from "../components/CustomSelect";

interface LeaveType {
  id: number;
  code: string;
  name: string;
}

interface FinancialYear {
  id: number;
  name: string;
  start_date: string;
  end_date: string;
}

/** One date charged against an allocation. `days` is what it cost: 1 for a
 *  normal day, 0.5 for a half day, 2 for a half day against the monthly
 *  Short-Leave allowance, 0 for a day that fell to Loss-Of-Pay. */
interface UsedLeaveDay {
  date: string;
  days: number;
  source: "request" | "attendance";
  detail: string;
}

interface Allocation {
  id: number;
  employee_id: number;
  financial_year_id: number;
  leave_type_id: number;
  allocated_days: number;
  used_days: number;
  balance_days: number;
  used_dates?: UsedLeaveDay[];
}

interface EmployeeOption {
  id: number;
  employee_code: string;
  full_name: string;
  employment_status?: string;
}

// Premium SVG Icons for Actions
/** Identity tint for an employee card avatar, stable per employee. */
const AVATAR_TINTS = ["eds-avatar--blue", "eds-avatar--green", "eds-avatar--purple", "eds-avatar--rose", ""];

/* 24-box strokes, round caps, currentColor. Size comes from the control that
   holds them (.eds-iconbtn 15px, .eds-action 13px, .eds-empty-tile 20px). */
const Icons = {
  Edit: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 4H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-5"></path>
      <path d="M18.5 2.5a2.1 2.1 0 0 1 3 3L12 15l-4 1 1-4z"></path>
    </svg>
  ),
  Trash: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6M14 11v6" />
      <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
    </svg>
  ),
  Plus: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="5" x2="12" y2="19"></line><line x1="5" y1="12" x2="19" y2="12"></line>
    </svg>
  ),
  Chevron: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="6 9 12 15 18 9"></polyline>
    </svg>
  ),
  Leave: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <path d="M17 20v-1.5A3.5 3.5 0 0 0 13.5 15h-6A3.5 3.5 0 0 0 4 18.5V20"></path>
      <circle cx="10.5" cy="8" r="3.5"></circle>
      <polyline points="17 11 19 13 22 9"></polyline>
    </svg>
  ),
};

export default function LeaveAllocations() {
  const { hasRole } = useAuth();
  const [financialYears, setFinancialYears] = useState<FinancialYear[]>([]);
  const [selectedFyId, setSelectedFyId] = useState<number | null>(null);
  const [types, setTypes] = useState<LeaveType[]>([]);
  const [allocations, setAllocations] = useState<Allocation[]>([]);
  const [employees, setEmployees] = useState<EmployeeOption[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [modal, setModal] = useState<"add" | "edit" | null>(null);
  const [editRow, setEditRow] = useState<Allocation | null>(null);
  const [form, setForm] = useState({
    employee_id: "",
    leave_type_id: "",
    allocated_days: "",
  });

  // Which allocation rows have their used-dates list expanded. A full financial
  // year of dates would bury the figures if every card showed them at once.
  const [openDates, setOpenDates] = useState<Set<number>>(new Set());

  const canManage = hasRole("Admin") || hasRole("HR");

  const toggleDates = (allocId: number) =>
    setOpenDates((prev) => {
      const next = new Set(prev);
      if (next.has(allocId)) next.delete(allocId);
      else next.add(allocId);
      return next;
    });

  useEffect(() => {
    Promise.all([companyApi.financialYears(), leaveApi.types(), employeesApi.list({ status: "Active" })])
      .then(([fyRes, tRes, eRes]) => {
        setFinancialYears(fyRes.data || []);
        setTypes(tRes.data || []);
        setEmployees(
          ((eRes.data as any[]) || [])
            .filter((e: any) => (e.employment_status || "Active") === "Active")
            .map((e: any) => ({
            id: e.id,
            employee_code: e.employee_code,
            full_name: e.full_name,
            employment_status: e.employment_status,
          }))
        );
        if ((fyRes.data || []).length > 0 && !selectedFyId)
          setSelectedFyId((fyRes.data as FinancialYear[])[0].id);
      })
      .catch(() => { })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selectedFyId || !canManage) {
      setAllocations([]);
      return;
    }
    setLoading(true);
    leaveApi
      .allocations({ financial_year_id: selectedFyId })
      .then((res) =>
        setAllocations(
          (res.data || []).filter((a: Allocation) => employees.some((e) => e.id === a.employee_id))
        )
      )
      .catch(() => setAllocations([]))
      .finally(() => setLoading(false));
  }, [selectedFyId, canManage, employees]);

  const employeeLabel = (id: number) => {
    const e = employees.find((x) => x.id === id);
    return e ? `${e.employee_code} - ${e.full_name}` : `#${id}`;
  };

  const typeName = (id: number) => types.find((t) => t.id === id)?.name ?? `#${id}`;

  /** Days without a pointless ".00", but never rounding a half day away — a
   *  listed 0.5-day date beside a "Used: 1" heading reads as a bug. */
  const fmtDays = (n: number) => {
    const v = Number(n) || 0;
    return Number.isInteger(v) ? String(v) : v.toFixed(1);
  };

  const handleDelete = (a: Allocation) => {
    const label = `${typeName(a.leave_type_id)} for ${employeeLabel(a.employee_id)}`;
    if (!window.confirm(`Delete the ${label} allocation?`)) return;
    setError("");
    setSuccess("");
    leaveApi
      .deleteAllocation(a.id)
      // Drop it locally rather than refetching: the list effect is keyed on the
      // financial year, so it will not re-run just because a row went away.
      .then(() => {
        setAllocations((prev) => prev.filter((x) => x.id !== a.id));
        setSuccess(`Deleted ${label}.`);
      })
      .catch((err) =>
        setError(err.response?.data?.detail || "Failed to delete allocation.")
      );
  };

  const openAdd = () => {
    setEditRow(null);
    setForm({ employee_id: "", leave_type_id: "", allocated_days: "" });
    setModal("add");
    setError("");
    setSuccess("");
  };

  const openEdit = (row: Allocation) => {
    setEditRow(row);
    setForm({
      employee_id: String(row.employee_id),
      leave_type_id: String(row.leave_type_id),
      allocated_days: String(row.allocated_days),
    });
    setModal("edit");
    setError("");
    setSuccess("");
  };

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFyId) return;
    const allocated = Number(form.allocated_days);
    if (isNaN(allocated) || allocated < 0) {
      setError("Allocated days must be 0 or more.");
      return;
    }
    if (editRow && allocated < editRow.used_days) {
      setError(`Allocated days cannot be less than already used (${editRow.used_days}).`);
      return;
    }
    setError("");
    leaveApi
      .setAllocation({
        employee_id: Number(form.employee_id),
        leave_type_id: Number(form.leave_type_id),
        allocated_days: allocated,
        financial_year_id: selectedFyId,
      })
      .then(() => {
        setSuccess(editRow ? "Allocation updated." : "Allocation added.");
        setModal(null);
        leaveApi
          .allocations({ financial_year_id: selectedFyId })
          .then((res) => setAllocations(res.data || []));
      })
      .catch((err) => setError(err.response?.data?.detail || "Failed to save."));
  };

  if (!canManage) {
    return (
      <div className="eds">
        <div className="eds-page">
          <section className="eds-card">
            <div className="eds-empty--card">
              <span className="eds-empty-tile"><Icons.Leave /></span>
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
          <h1 className="eds-title">Leave Allocations</h1>
          <p className="eds-subtitle">
            Set or edit paid/unpaid leave days per employee per financial year.
          </p>
        </div>
        <GlobalHeaderControls />
      </header>

      <div className="eds-page">
      {success && <div className="alert alert-success">{success}</div>}
      {error && <div className="alert alert-error">{error}</div>}

      <div className="eds-controls" style={{ alignItems: "flex-end" }}>
        <div className="eds-fieldset">
          <span className="eds-fieldset-label">Financial year</span>
          <CustomSelect
            className="eds-cselect"
            value={String(selectedFyId ?? "")}
            onChange={(val) => setSelectedFyId(Number(val) || null)}
            style={{ width: "330px", maxWidth: "100%" }}
            options={financialYears.map((fy) => ({
              value: String(fy.id),
              label: `${fy.name} (${formatDate(fy.start_date)} to ${formatDate(fy.end_date)})`
            }))}
          />
        </div>
        <div className="eds-controls-end">
          <button type="button" className="eds-action eds-action--go" onClick={openAdd} title="Assign New Leave Allocation for an Employee">
            <Icons.Plus />
            Add allocation
          </button>
        </div>
      </div>

        {loading ? (
          <SectionLoader rows={4} />
        ) : allocations.length === 0 ? (
          <section className="eds-card">
            <div className="eds-empty--card">
              <span className="eds-empty-tile"><Icons.Leave /></span>
              <span>
                No leave allocations for this financial year. Click &quot;Add allocation&quot; to set days for an employee (e.g. 1 paid leave, 0, or more as per policy).
              </span>
            </div>
          </section>
        ) : (() => {
          const sortedAllocations = [...allocations].sort((a, b) => {
            const empA = employees.find(e => e.id === a.employee_id);
            const empB = employees.find(e => e.id === b.employee_id);
            const codeA = parseInt(String(empA?.employee_code || "0").replace(/\D/g, ""), 10) || 0;
            const codeB = parseInt(String(empB?.employee_code || "0").replace(/\D/g, ""), 10) || 0;
            if (codeA !== codeB) return codeA - codeB;
            return a.leave_type_id - b.leave_type_id;
          });

          const groupedMap = new Map<number, Allocation[]>();
          sortedAllocations.forEach(a => {
            if (!groupedMap.has(a.employee_id)) groupedMap.set(a.employee_id, []);
            groupedMap.get(a.employee_id)!.push(a);
          });

          return (
            <div className="eds-alloc-grid">
              {Array.from(groupedMap.entries()).map(([empId, allocs]) => {
                const label = employeeLabel(empId);
                const parts = label.split("-");
                const code = parts[0]?.trim() || "";
                const name = parts[1]?.trim() || label;
                const initial = name.charAt(0).toUpperCase();
                return (
                  <section key={empId} className="eds-card">
                    <div className="eds-card-head">
                      <span className={`eds-avatar eds-avatar--xl ${AVATAR_TINTS[empId % AVATAR_TINTS.length]}`}>
                        {initial}
                      </span>
                      <div className="eds-card-titles">
                        <h2 className="eds-card-title">{name}</h2>
                        <p className="eds-card-sub">{code}</p>
                      </div>
                    </div>
                    <div className="eds-alloc-list">
                      {allocs.map((a) => {
                        const usedDates = a.used_dates ?? [];
                        const isOpen = openDates.has(a.id);
                        return (
                        <div key={a.id} className="eds-alloc-item">
                        <div className="eds-alloc-row">
                          <div className="eds-alloc-type">
                            <span className="eds-alloc-name">{typeName(a.leave_type_id)}</span>
                            <span className="eds-alloc-meta">
                              Alloc: {fmtDays(a.allocated_days)} &middot; Used:{" "}
                              {usedDates.length > 0 ? (
                                <button
                                  type="button"
                                  className={`eds-alloc-usedbtn${isOpen ? " is-open" : ""}`}
                                  onClick={() => toggleDates(a.id)}
                                  aria-expanded={isOpen}
                                  title={`${isOpen ? "Hide" : "Show"} the ${usedDates.length} date(s) behind this figure`}
                                >
                                  {fmtDays(Number(a.used_days))}
                                  <span className="eds-alloc-usedchev"><Icons.Chevron /></span>
                                </button>
                              ) : (
                                fmtDays(Number(a.used_days))
                              )}
                            </span>
                          </div>
                          <div className="eds-alloc-end">
                            <div className="eds-alloc-balance">
                              <span className="eds-eyebrow">Balance</span>
                              <span className="eds-alloc-figure">{fmtDays(Number(a.balance_days))}</span>
                            </div>
                            <button
                              type="button"
                              className="eds-iconbtn eds-iconbtn--view"
                              onClick={() => openEdit(a)}
                              title="Edit Allocation"
                            >
                              <Icons.Edit />
                            </button>
                            {/* Disabled once any of the allocation is spent. The
                                server refuses this too -- the used figure is the
                                only record that those days were taken, so removing
                                the row would erase it while the approved requests
                                remained. Editing the allocation down is the way to
                                stop further leave. */}
                            <button
                              type="button"
                              className="eds-iconbtn eds-iconbtn--del"
                              onClick={() => handleDelete(a)}
                              disabled={Number(a.used_days) > 0}
                              title={
                                Number(a.used_days) > 0
                                  ? `Cannot delete: ${fmtDays(Number(a.used_days))} day(s) already used. Edit the allocation instead.`
                                  : "Delete Allocation"
                              }
                            >
                              <Icons.Trash />
                            </button>
                          </div>
                        </div>

                        {isOpen && (
                          <ul className="eds-alloc-dates">
                            {usedDates.map((d, i) => (
                              <li
                                key={`${d.date}-${i}`}
                                className={`eds-alloc-date eds-alloc-date--${d.source}`}
                                title={d.detail}
                              >
                                <span className="eds-alloc-date-day">{formatDate(d.date)}</span>
                                <span className="eds-alloc-date-cost">
                                  {d.days > 0 ? `${fmtDays(d.days)}d` : "LOP"}
                                </span>
                              </li>
                            ))}
                          </ul>
                        )}
                        </div>
                        );
                      })}
                    </div>
                  </section>
                );
              })}
            </div>
          );
        })()}
      </div>

      {modal && (
        <div className="modal-backdrop" onClick={() => setModal(null)}>
          <div className="modal" style={{ maxWidth: 420 }} onClick={(e) => e.stopPropagation()}>
            <h3 style={{ marginTop: 0 }}>{editRow ? "Edit allocation" : "Add allocation"}</h3>
            <form onSubmit={handleSave}>
              <div className="form-group">
                <label>Employee</label>
                <CustomSelect
                  value={form.employee_id}
                  onChange={(val) => setForm((f) => ({ ...f, employee_id: val }))}
                  options={[
                    { value: "", label: "Select employee" },
                    ...[...employees].filter((e) => (e.employment_status || "Active") === "Active").sort((a, b) => {
                      const nA = parseInt(a.employee_code.replace(/\D/g, ""), 10) || 0;
                      const nB = parseInt(b.employee_code.replace(/\D/g, ""), 10) || 0;
                      return nA - nB;
                    }).map((e) => ({
                      value: String(e.id),
                      label: `${e.employee_code} - ${e.full_name}`
                    }))
                  ]}
                />
              </div>
              <div className="form-group">
                <label>Leave type</label>
                <CustomSelect
                  value={form.leave_type_id}
                  onChange={(val) => setForm((f) => ({ ...f, leave_type_id: val }))}
                  options={[
                    { value: "", label: "Select leave type" },
                    ...types.map((t) => ({
                      value: String(t.id),
                      label: `${t.name} (${t.code})`
                    }))
                  ]}
                />
              </div>
              <div className="form-group">
                <label>Allocated days</label>
                <input
                  type="number"
                  min={editRow ? editRow.used_days : 0}
                  step="0.5"
                  required
                  value={form.allocated_days}
                  onChange={(e) => setForm((f) => ({ ...f, allocated_days: e.target.value }))}
                />
                {editRow && (
                  <p className="text-muted" style={{ fontSize: "0.8rem", marginTop: 4 }}>
                    Cannot be less than already used ({editRow.used_days}).
                  </p>
                )}
              </div>
              <div style={{ marginTop: "1rem", display: "flex", justifyContent: "flex-end", gap: "0.5rem" }}>
                <button type="submit" className="btn btn-primary btn-uniform" title="Save Allocation Changes">
                  Save
                </button>
                <button type="button" className="btn btn-cancel-alt btn-uniform" onClick={() => setModal(null)} title="Cancel Changes">
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
