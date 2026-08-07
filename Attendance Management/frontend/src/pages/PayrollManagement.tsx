import { useEffect, useState, useMemo, type CSSProperties } from "react";
import { payroll as payrollApi, employees as employeesApi, attendance as attendanceApi, type SalaryAdvanceRow } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import SalaryFormulaView from "../components/SalaryFormulaView";
import ConfirmModal from "../components/ConfirmModal";
import { SectionLoader } from "../components/LoadingState";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import CustomSelect from "../components/CustomSelect";
import { useTableControls } from "../components/dataTable";
import type { SortState } from "../components/dataTable";

interface SalaryStructure {
  id: number;
  employee_id: number;
  basic: number;
  hra: number;
  medical: number;
  travelling: number;
  miscellaneous: number;
  allowances: number;
  deductions: number;
  effective_from: string;
  effective_to?: string | null;
}

/** Monthly gross = Basic + HRA + Medical + Travelling + Miscellaneous + Allowances */
function structureMonthlyGross(s: SalaryStructure): number {
  return (
    Number(s.basic ?? 0) +
    Number(s.hra ?? 0) +
    Number(s.medical ?? 0) +
    Number(s.travelling ?? 0) +
    Number(s.miscellaneous ?? 0) +
    Number(s.allowances ?? 0)
  );
}

interface EmployeeOption {
  id: number;
  employee_code: string;
  full_name: string;
  expected_working_hours: number;
}

interface PayrollPeriod {
  id: number;
  month: number;
  year: number;
  status: string;
}

interface Payslip {
  id: number;
  employee_id: number;
  payroll_period_id: number;
  gross_salary: number;
  total_earnings: number;
  total_deductions: number;
  net_salary: number;
  paid_days: number;
  lop_days: number;
  component_breakdown?: string | null;
}

const MONTHS = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

/** Structure is effective for the given month if it covers any day of that month */
function structureEffectiveForMonth(
  s: SalaryStructure,
  month: number,
  year: number
): boolean {
  const firstDay = `${year}-${String(month).padStart(2, "0")}-01`;
  const lastDay = new Date(year, month, 0).getDate();
  const lastDayStr = `${year}-${String(month).padStart(2, "0")}-${String(lastDay).padStart(2, "0")}`;
  if (s.effective_from > lastDayStr) return false;
  if (s.effective_to != null && s.effective_to < firstDay) return false;
  return true;
}

/** For each employee, get the structure effective for the month with latest effective_from */
function structuresForMonth(
  structures: SalaryStructure[],
  month: number,
  year: number
): Map<number, SalaryStructure> {
  const filtered = structures.filter((s) => structureEffectiveForMonth(s, month, year));
  const byEmployee = new Map<number, SalaryStructure>();
  filtered
    .sort((a, b) => (b.effective_from > a.effective_from ? 1 : -1))
    .forEach((s) => {
      if (!byEmployee.has(s.employee_id)) byEmployee.set(s.employee_id, s);
    });
  return byEmployee;
}

/* 24-box strokes, round caps, currentColor. Size comes from the control that
   holds them (.eds-chip 16px, .eds-iconbtn 15px, .eds-empty-tile 20px). */
const Icons = {
  View: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <path d="M1.5 12S5 5.5 12 5.5 22.5 12 22.5 12 19 18.5 12 18.5 1.5 12 1.5 12z"></path>
      <circle cx="12" cy="12" r="3"></circle>
    </svg>
  ),
  Rupee: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="2" x2="12" y2="22" />
      <path d="M17 6H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6" />
    </svg>
  ),
  Users: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <path d="M17 20v-1.5A3.5 3.5 0 0 0 13.5 15h-6A3.5 3.5 0 0 0 4 18.5V20" />
      <circle cx="10.5" cy="8" r="3.5" />
      <path d="M20 20v-1.5a3.5 3.5 0 0 0-2.6-3.4" />
    </svg>
  ),
  Search: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="7.5" />
      <line x1="21" y1="21" x2="16.7" y2="16.7" />
    </svg>
  ),
};

/* Icons for the Salary Advances panel, which is declared above the page. */
const AdvIcons = {
  Wallet: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="6" width="18" height="13" rx="2.5" />
      <path d="M3 10h18" />
      <circle cx="17" cy="14.5" r="1.2" />
    </svg>
  ),
  Trash: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 21 6" />
      <path d="M8 6V4h8v2" />
      <path d="M6 6l1 14h10l1-14" />
    </svg>
  ),
};

/** Column header for the payroll register. `money` right-aligns the column so
 *  the header sits over the figures. Sorting stays in useTableControls. */
function PaySortTh({
  label,
  columnKey,
  sort,
  onToggle,
  notSortable,
  money,
  className,
}: {
  label: string;
  columnKey: string;
  sort: SortState;
  onToggle: (key: string) => void;
  notSortable?: boolean;
  money?: boolean;
  className?: string;
}) {
  const cls = [className, notSortable || money ? "is-actions" : ""].filter(Boolean).join(" ");
  if (notSortable) return <th className={cls || undefined}>{label}</th>;
  const active = sort.key === columnKey;
  return (
    <th className={cls || undefined}>
      <button
        type="button"
        className={`eds-sort${active ? " is-active" : ""}`}
        onClick={() => onToggle(columnKey)}
        title={`Sort by ${label}`}
      >
        {label}
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round">
          {!active || sort.direction !== "asc" ? <polyline points="7 15 12 20 17 15" /> : null}
          {!active || sort.direction !== "desc" ? <polyline points="7 9 12 4 17 9" /> : null}
        </svg>
      </button>
    </th>
  );
}

// Salary Advances (Feature 6). Self-contained: manages its own fetch/state.
// An advance is recovered in full on the next payroll run.
function SalaryAdvancesPanel({
  employees,
  canEdit,
}: {
  employees: EmployeeOption[];
  canEdit: boolean;
}) {
  const emptyForm = {
    employee_id: "",
    amount: "",
    date_taken: new Date().toISOString().slice(0, 10),
    reason: "",
  };
  const [advances, setAdvances] = useState<SalaryAdvanceRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState(emptyForm);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const empName = (id: number) => {
    const e = employees.find((x) => x.id === id);
    return e ? `${e.full_name} (${e.employee_code})` : `#${id}`;
  };
  const money = (v: number) => "₹ " + Number(v).toLocaleString("en-IN");

  const load = () => {
    setLoading(true);
    payrollApi
      .advances()
      .then((r) => setAdvances(r.data))
      .catch(() => setAdvances([]))
      .finally(() => setLoading(false));
  };
  useEffect(load, []);

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.employee_id) {
      setError("Select an employee");
      return;
    }
    const amt = Number(form.amount);
    if (!amt || amt <= 0) {
      setError("Enter a valid amount");
      return;
    }
    setBusy(true);
    setError("");
    payrollApi
      .createAdvance({
        employee_id: Number(form.employee_id),
        amount: amt,
        date_taken: form.date_taken,
        reason: form.reason || null,
      })
      .then(() => {
        setForm(emptyForm);
        setShowForm(false);
        load();
      })
      .catch((e: any) => setError(e?.response?.data?.detail || "Failed to save advance"))
      .finally(() => setBusy(false));
  };

  const remove = (id: number) => {
    payrollApi.deleteAdvance(id).then(load).catch(() => load());
  };

  return (
    <section className="eds-card">
      <div className="eds-card-head">
        <span className="eds-chip eds-chip--amber"><AdvIcons.Wallet /></span>
        <div className="eds-card-titles">
          <h2 className="eds-card-title">Salary Advances</h2>
          <p className="eds-card-sub">Recovered in full from the employee's next payroll run.</p>
        </div>
        {canEdit && (
          <button type="button" className="eds-action eds-action--go" onClick={() => { setError(""); setShowForm((s) => !s); }}>
            {showForm ? "Cancel" : "Add Advance"}
          </button>
        )}
      </div>

      {error && <div className="alert alert-error" style={{ margin: "0.75rem 20px 0" }}>{error}</div>}

      {canEdit && showForm && (
        <form onSubmit={submit} className="eds-card-body">
          <div className="eds-form-grid3" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))" }}>
            <div className="eds-fieldset">
              <span className="eds-fieldset-label">Employee</span>
              <CustomSelect
                value={form.employee_id}
                onChange={(v) => setForm({ ...form, employee_id: String(v) })}
                placeholder="Select employee"
                options={[
                  { value: "", label: "Select employee" },
                  ...employees.map((e) => ({ value: String(e.id), label: `${e.full_name} (${e.employee_code})` })),
                ]}
              />
            </div>
            <div className="eds-fieldset">
              <span className="eds-fieldset-label">Amount (₹)</span>
              <input className="eds-input" type="number" min="0" step="0.01" value={form.amount} onChange={(e) => setForm({ ...form, amount: e.target.value })} required />
            </div>
            <div className="eds-fieldset">
              <span className="eds-fieldset-label">Date Taken</span>
              <input className="eds-input" type="date" value={form.date_taken} onChange={(e) => setForm({ ...form, date_taken: e.target.value })} required />
            </div>
            <div className="eds-fieldset">
              <span className="eds-fieldset-label">Reason (optional)</span>
              <input className="eds-input" value={form.reason} onChange={(e) => setForm({ ...form, reason: e.target.value })} placeholder="e.g. Medical emergency" />
            </div>
          </div>
          <div className="eds-controls-end">
            <button type="submit" className="eds-action eds-action--go" disabled={busy}>{busy ? "Saving…" : "Save Advance"}</button>
          </div>
        </form>
      )}

      {loading ? (
        <div className="eds-empty--card">
          <span className="eds-empty-tile"><AdvIcons.Wallet /></span>
          <span>Loading advances…</span>
        </div>
      ) : advances.length === 0 ? (
        <div className="eds-empty--card">
          <span className="eds-empty-tile"><AdvIcons.Wallet /></span>
          <span>No advances recorded.</span>
        </div>
      ) : (
        <div className="eds-table-wrap">
          <table className="eds-table eds-table--auto">
            <thead>
              <tr>
                <th>Employee</th>
                <th className="is-actions">Amount</th>
                <th>Date Taken</th>
                <th>Reason</th>
                <th>Status</th>
                {canEdit && <th className="is-actions">Actions</th>}
              </tr>
            </thead>
            <tbody>
              {advances.map((a) => (
                <tr key={a.id}>
                  <td className="eds-cell-strong">{empName(a.employee_id)}</td>
                  <td className="eds-money">{money(a.amount)}</td>
                  <td className="eds-cell-mid">{a.date_taken}</td>
                  <td className="eds-cell-dim">{a.reason || "—"}</td>
                  <td>
                    <span className={`eds-status${a.status === "DEDUCTED" ? " eds-status--present" : " eds-status--warn"}`}>
                      <i></i>
                      {a.status === "DEDUCTED" ? "Recovered" : a.status === "PENDING" ? "Pending" : a.status}
                    </span>
                  </td>
                  {canEdit && (
                    <td>
                      <div className="eds-rowactions">
                        {a.status === "PENDING" && (
                          <button type="button" className="eds-iconbtn eds-iconbtn--del" onClick={() => remove(a.id)} title="Delete Advance">
                            <AdvIcons.Trash />
                          </button>
                        )}
                      </div>
                    </td>
                  )}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </section>
  );
}

export default function PayrollManagement() {
  const { hasRole } = useAuth();
  const now = new Date();
  const [selectedMonth, setSelectedMonth] = useState(now.getMonth() + 1);
  const [selectedYear, setSelectedYear] = useState(now.getFullYear());
  const [structures, setStructures] = useState<SalaryStructure[]>([]);
  const [employees, setEmployees] = useState<EmployeeOption[]>([]);
  const [periods, setPeriods] = useState<PayrollPeriod[]>([]);
  const [payslips, setPayslips] = useState<Payslip[]>([]);
  const [loading, setLoading] = useState(true);
  const [confirmDelete, setConfirmDelete] = useState<SalaryStructure | null>(null);
  const [showModal, setShowModal] = useState(false);
  const [editing, setEditing] = useState<SalaryStructure | null>(null);
  const [form, setForm] = useState({
    employee_id: "",
    basic: "",
    hra: "",
    medical: "",
    travelling: "",
    miscellaneous: "",
    allowances: "",
    deductions: "",
    effective_from: "",
  });
  const [detailDialog, setDetailDialog] = useState<{
    employee: EmployeeOption;
    structure: SalaryStructure;
    payslip: Payslip | null;
  } | null>(null);
  const [showFormulaInDetail, setShowFormulaInDetail] = useState(false);
  const [modalAttendance, setModalAttendance] = useState<{ paidDays: number; lopDays: number } | null>(null);

  const canEdit = hasRole("HR");

  const sortKeyForEmployeeCode = (code: string) => {
    const n = Number(code);
    if (!Number.isNaN(n)) return { isNumeric: true, num: n, raw: code };
    return { isNumeric: false, num: 0, raw: code.toUpperCase() };
  };

  const loadData = () => {
    setLoading(true);
    Promise.allSettled([
      payrollApi.salaryStructures(),
      employeesApi.list({ status: "Active" }),
      payrollApi.periods(),
    ])
      .then(([sRes, eRes, pRes]) => {
        if (sRes.status === "fulfilled") {
          setStructures(sRes.value.data || []);
        } else {
          setStructures([]);
          console.error("[Payroll] salary-structures failed:", sRes.reason);
        }

        if (eRes.status === "fulfilled") {
          setEmployees(
            ((eRes.value.data as any[]) || []).map((e: any) => ({
              id: e.id,
              employee_code: e.employee_code,
              full_name: e.full_name,
              expected_working_hours: e.expected_working_hours ?? 9.0,
            }))
          );
        } else {
          setEmployees([]);
          console.error("[Payroll] employees failed:", eRes.reason);
        }

        if (pRes.status === "fulfilled") {
          setPeriods(pRes.value.data || []);
        } else {
          setPeriods([]);
          console.error("[Payroll] periods failed:", pRes.reason);
        }
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadData();
  }, []);

  // Fetch attendance for selected employee + month when Add/Edit modal is open
  useEffect(() => {
    if (!showModal || !form.employee_id) {
      setModalAttendance(null);
      return;
    }
    const from = `${selectedYear}-${String(selectedMonth).padStart(2, "0")}-01`;
    const lastDay = new Date(selectedYear, selectedMonth, 0).getDate();
    const to = `${selectedYear}-${String(selectedMonth).padStart(2, "0")}-${String(lastDay).padStart(2, "0")}`;
    attendanceApi
      .list(from, to, Number(form.employee_id))
      .then((res) => {
        const records = res.data || [];
        let paid = 0;
        let lop = 0;
        records.forEach((r: { status: string }) => {
          if (r.status === "PRESENT" || r.status === "ON_LEAVE") paid += 1;
          else if (r.status === "HALF_DAY") paid += 0.5;
          else if (r.status === "ABSENT") lop += 1;
        });
        setModalAttendance({ paidDays: paid, lopDays: lop });
      })
      .catch(() => setModalAttendance(null));
  }, [showModal, form.employee_id, selectedMonth, selectedYear]);

  const periodForMonth = useMemo(
    () => periods.find((p) => p.month === selectedMonth && p.year === selectedYear),
    [periods, selectedMonth, selectedYear]
  );

  useEffect(() => {
    const periodId = periodForMonth?.id;
    if (!periodId) {
      setPayslips([]);
      return;
    }
    payrollApi
      .payslips({ period_id: periodId })
      .then((res) => setPayslips(res.data || []))
      .catch(() => setPayslips([]));
  }, [periodForMonth?.id]);

  const monthStructures = useMemo(
    () => structuresForMonth(structures, selectedMonth, selectedYear),
    [structures, selectedMonth, selectedYear]
  );

  // Index payslips by employee, so the row build below is a lookup rather than
  // a scan of every payslip for every employee (employees x payslips).
  const payslipsByEmployee = useMemo(() => {
    const byEmployee = new Map<number, (typeof payslips)[number]>();
    for (const p of payslips) byEmployee.set(p.employee_id, p);
    return byEmployee;
  }, [payslips]);

  const rows = useMemo(() => {
    return employees
      .filter((emp) => monthStructures.has(emp.id))
      .map((emp) => {
        const structure = monthStructures.get(emp.id)!;
        const payslip = payslipsByEmployee.get(emp.id) ?? null;
        const gross = structureMonthlyGross(structure);
        return { employee: emp, structure, payslip, gross };
      })
      .sort((a, b) => {
        const ca = sortKeyForEmployeeCode(a.employee.employee_code || "");
        const cb = sortKeyForEmployeeCode(b.employee.employee_code || "");
        if (ca.isNumeric && cb.isNumeric) {
          return ca.num - cb.num;
        }
        if (ca.isNumeric !== cb.isNumeric) {
          return ca.isNumeric ? -1 : 1;
        }
        return ca.raw.localeCompare(cb.raw);
      });
  }, [employees, monthStructures, payslipsByEmployee]);

  type PayrollRow = (typeof rows)[number];

  const {
    displayed: displayedRows,
    search: rowSearch,
    setSearch: setRowSearch,
    sort: rowSort,
    toggleSort: toggleRowSort,
    clearAll: clearRowControls,
    hasActiveControls: rowHasActive,
  } = useTableControls<PayrollRow>({
    rows,
    columns: {
      employee: (r) => r.employee.full_name,
      paid_days: (r) => (r.payslip ? Number(r.payslip.paid_days) : -1),
      lop_days: (r) => (r.payslip ? Number(r.payslip.lop_days) : -1),
      gross: (r) => r.gross,
      earnings: (r) => (r.payslip ? Number(r.payslip.total_earnings) : -1),
      deductions: (r) => (r.payslip ? Number(r.payslip.total_deductions) : -1),
      net: (r) => (r.payslip ? Number(r.payslip.net_salary) : -1),
      payslip_status: (r) => (r.payslip ? "Generated" : "Not run"),
    },
    searchableText: (r) =>
      `${r.employee.employee_code} ${r.employee.full_name} ${r.payslip ? "Generated" : "Not run"}`,
  });

  const openAdd = () => {
    setEditing(null);
    setForm({
      employee_id: "",
      basic: "",
      hra: "",
      medical: "",
      travelling: "",
      miscellaneous: "",
      allowances: "",
      deductions: "",
      effective_from: `${selectedYear}-${String(selectedMonth).padStart(2, "0")}-01`,
    });
    setDetailDialog(null);
    setShowModal(true);
  };

  const openEdit = (s: SalaryStructure) => {
    setEditing(s);
    setForm({
      employee_id: String(s.employee_id),
      basic: String(s.basic),
      hra: String(s.hra),
      medical: String(s.medical ?? 0),
      travelling: String(s.travelling ?? 0),
      miscellaneous: String(s.miscellaneous ?? 0),
      allowances: String(s.allowances),
      deductions: String(s.deductions),
      effective_from: s.effective_from,
    });
    setDetailDialog(null);
    setShowModal(true);
  };

  const openDetail = (row: { employee: EmployeeOption; structure: SalaryStructure; payslip: Payslip | null }) => {
    setDetailDialog({
      employee: row.employee,
      structure: row.structure,
      payslip: row.payslip,
    });
    setShowFormulaInDetail(false);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const payload = {
      employee_id: Number(form.employee_id),
      basic: Number(form.basic),
      hra: Number(form.hra),
      medical: form.medical ? Number(form.medical) : 0,
      travelling: form.travelling ? Number(form.travelling) : 0,
      miscellaneous: form.miscellaneous ? Number(form.miscellaneous) : 0,
      allowances: form.allowances ? Number(form.allowances) : 0,
      deductions: form.deductions ? Number(form.deductions) : 0,
      effective_from: form.effective_from,
    };
    const req = editing
      ? payrollApi.updateSalaryStructure(editing.id, payload)
      : payrollApi.createSalaryStructure(payload);
    req
      .then(() => {
        setShowModal(false);
        loadData();
      })
      .catch(() => { });
  };

  const handleDelete = (s: SalaryStructure) => {
    setConfirmDelete(s);
  };

  const confirmActualDelete = () => {
    if (!confirmDelete) return;
    payrollApi
      .deleteSalaryStructure(confirmDelete.id)
      .then(() => {
        loadData();
        setConfirmDelete(null);
        setDetailDialog(null);
      })
      .catch((err) => alert(err.response?.data?.detail || "Delete failed"));
  };

  const monthYearLabel = () => `${MONTHS[selectedMonth - 1]} ${selectedYear}`;

  const payrollTableShellStyle: CSSProperties = {
    overflowX: "hidden",
    overflowY: "hidden",
    paddingBottom: "0.25rem",
  };

  const payrollTableStyle: CSSProperties = {
    width: "100%",
    tableLayout: "fixed",
  };


  return (
    <div className="eds">
      <header className="eds-topbar">
        <div>
          <h1 className="eds-title">Payroll Management</h1>
          <p className="eds-subtitle">Manage salary structures and payroll processing</p>
        </div>
        <GlobalHeaderControls />
      </header>

      <div className="eds-page">
      <SalaryAdvancesPanel employees={employees} canEdit={canEdit} />

      <section className="eds-card">
        <div className="eds-card-head">
          <span className="eds-chip eds-chip--emerald"><Icons.Rupee /></span>
          <div className="eds-card-titles">
            <h2 className="eds-card-title">Monthly Payroll Summary</h2>
          </div>
          <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 9, flexShrink: 0 }}>
            <CustomSelect
              className="eds-cselect"
              value={String(selectedMonth)}
              onChange={(val) => setSelectedMonth(Number(val))}
              options={MONTHS.map((m, i) => ({ value: String(i + 1), label: m }))}
              style={{ width: "140px" }}
            />
            <input
              className="eds-input"
              type="number"
              value={selectedYear}
              onChange={(e) => setSelectedYear(Number(e.target.value))}
              min={2020}
              max={2030}
              style={{ width: "110px", height: "36px" }}
            />
          </div>
        </div>
      </section>

      <section className="eds-card">
        <div className="eds-card-head">
          <span className="eds-chip eds-chip--sky"><Icons.Users /></span>
          <div className="eds-card-titles">
            <h2 className="eds-card-title">Employees — {monthYearLabel()}</h2>
          </div>
          {canEdit && (
            <button type="button" className="eds-action eds-action--go" onClick={openAdd} title="Add New Salary Structure/Payroll">
              Add Payroll
            </button>
          )}
        </div>

        <div className="eds-tablebar">
          <label className="eds-search">
            <Icons.Search />
            <input
              type="search"
              value={rowSearch}
              onChange={(e) => setRowSearch(e.target.value)}
              placeholder="Search employee or payslip status..."
            />
          </label>
          {rowHasActive && (
            <button type="button" className="eds-action" onClick={clearRowControls} title="Clear search and sort">
              Clear filters
            </button>
          )}
          <span className="eds-showing">
            Showing <b>{displayedRows.length}</b> of <b>{rows.length}</b>
          </span>
        </div>

        {loading ? (
          <SectionLoader rows={5} />
        ) : rows.length === 0 ? (
          <div className="eds-empty--card">
            <span className="eds-empty-tile"><Icons.Rupee /></span>
            <span>No salary structure effective for {monthYearLabel()}. Add payroll with effective date covering this month.</span>
          </div>
        ) : (
          <div className="eds-table-wrap" style={payrollTableShellStyle}>
            <table className="eds-table" style={payrollTableStyle}>
              <colgroup>
                <col style={{ width: "5%" }} />
                <col style={{ width: "18%" }} />
                <col style={{ width: "8%" }} />
                <col style={{ width: "8%" }} />
                <col style={{ width: "11%" }} />
                <col style={{ width: "10%" }} />
                <col style={{ width: "10%" }} />
                <col style={{ width: "11%" }} />
                <col style={{ width: "10%" }} />
                <col style={{ width: "9%" }} />
              </colgroup>
              <thead>
                <tr>
                  <PaySortTh className="hide-md" label="S.No" columnKey="__sno" sort={rowSort} onToggle={toggleRowSort} notSortable />
                  <PaySortTh label="Employee" columnKey="employee" sort={rowSort} onToggle={toggleRowSort} />
                  <PaySortTh className="hide-sm" label="Paid days" columnKey="paid_days" sort={rowSort} onToggle={toggleRowSort} money />
                  <PaySortTh className="hide-sm" label="LOP days" columnKey="lop_days" sort={rowSort} onToggle={toggleRowSort} money />
                  <PaySortTh className="hide-md" label="Gross (month)" columnKey="gross" sort={rowSort} onToggle={toggleRowSort} money />
                  <PaySortTh className="hide-lg" label="Earnings" columnKey="earnings" sort={rowSort} onToggle={toggleRowSort} money />
                  <PaySortTh className="hide-lg" label="Deductions" columnKey="deductions" sort={rowSort} onToggle={toggleRowSort} money />
                  <PaySortTh label="Net salary" columnKey="net" sort={rowSort} onToggle={toggleRowSort} money />
                  <PaySortTh className="hide-sm" label="Payslip" columnKey="payslip_status" sort={rowSort} onToggle={toggleRowSort} />
                  <PaySortTh label="Actions" columnKey="__actions" sort={rowSort} onToggle={toggleRowSort} notSortable />
                </tr>
              </thead>
              <tbody>
                {displayedRows.length === 0 && (
                  <tr>
                    <td colSpan={10} className="eds-table-empty">
                      No rows match your search.
                    </td>
                  </tr>
                )}
                {displayedRows.map((row, idx) => {
                  const p = row.payslip;
                  const paidDays = p ? Number(p.paid_days) : 0;
                  const lopDays = p ? Number(p.lop_days) : 0;
                  const earnings = p ? Number(p.total_earnings) : 0;
                  const deductions = p ? Number(p.total_deductions) : 0;
                  const net = p ? Number(p.net_salary) : 0;
                  return (
                    <tr key={row.employee.id}>
                      <td className="hide-md eds-money eds-money--zero">{idx + 1}</td>
                      <td className="eds-cell-strong" style={{ whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                        {row.employee.full_name}
                      </td>
                      <td className="hide-sm eds-money">{p ? paidDays : "-"}</td>
                      <td className={`hide-sm eds-money${p && lopDays > 0 ? " eds-money--lop" : " eds-money--zero"}`}>
                        {p ? lopDays : "-"}
                      </td>
                      <td className="hide-md eds-money">₹ {row.gross.toFixed(2)}</td>
                      <td className="hide-lg eds-money">
                        {p ? `₹ ${earnings.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "-"}
                      </td>
                      <td className={`hide-lg eds-money${p && deductions > 0 ? " eds-money--minus" : " eds-money--zero"}`}>
                        {p ? `₹ ${deductions.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "-"}
                      </td>
                      <td className="eds-money eds-money--net">
                        {p ? `₹ ${net.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "-"}
                      </td>
                      <td className="hide-sm">
                        {p ? (
                          <span className="eds-status eds-status--present"><i></i>Generated</span>
                        ) : (
                          <span className="eds-status"><i></i>Not run</span>
                        )}
                      </td>
                      <td>
                        <div className="eds-rowactions">
                          <button
                            type="button"
                            className="eds-iconbtn eds-iconbtn--view"
                            onClick={() => openDetail(row)}
                            title="View Complete Salary and Payslip Details"
                          >
                            <Icons.View />
                          </button>
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

      {/* Add/Edit salary structure modal */}
      {showModal && (
        <div className="modal-backdrop" onClick={() => setShowModal(false)}>
          <div className="modal" style={{ maxWidth: 560 }} onClick={(e) => e.stopPropagation()}>
            <h3 style={{ marginTop: 0 }}>{editing ? "Edit Payroll" : "Add Payroll"}</h3>

            {/* Total present days & salary calculation (for selected month) */}
            <div
              style={{
                marginBottom: "1rem",
                padding: "0.75rem 1rem",
                background: "rgba(255, 255, 255, 0.06)",
                borderRadius: 8,
                border: "1px solid rgba(255, 255, 255, 0.12)",
                backdropFilter: "blur(10px)",
              }}
            >
              <div style={{ fontWeight: 600, marginBottom: "0.5rem" }}>
                Attendance for {monthYearLabel()}
              </div>
              {form.employee_id ? (
                modalAttendance !== null ? (
                  <>
                    <div style={{ display: "flex", gap: "1.5rem", flexWrap: "wrap", marginBottom: "0.5rem" }}>
                      <span>
                        <strong>Total present days:</strong>{" "}
                        <span style={{ fontSize: "1.1rem" }}>{modalAttendance.paidDays}</span>
                      </span>
                      {modalAttendance.lopDays > 0 && (
                        <span style={{ color: "#b91c1c" }}>
                          <strong>LOP days:</strong> {modalAttendance.lopDays}
                        </span>
                      )}
                    </div>
                    {(() => {
                      const basic = Number(form.basic) || 0;
                      const hra = Number(form.hra) || 0;
                      const medical = Number(form.medical) || 0;
                      const travelling = Number(form.travelling) || 0;
                      const miscellaneous = Number(form.miscellaneous) || 0;
                      const allow = Number(form.allowances) || 0;
                      const gross = basic + hra + medical + travelling + miscellaneous + allow;
                      const perDay = gross / 30;
                      const payable = perDay * modalAttendance.paidDays;
                      const ded = (Number(form.deductions) || 0) / 30 * modalAttendance.paidDays;
                      const net = payable - ded;
                      return (
                        <div style={{ fontSize: "0.9rem", color: "rgba(255, 255, 255, 0.82)", marginTop: "0.5rem", paddingTop: "0.5rem", borderTop: "1px solid rgba(255, 255, 255, 0.12)" }}>
                          <div style={{ fontWeight: 600, marginBottom: 4 }}>Salary calculated on a fixed 30-day month:</div>
                          <div>Per day = Gross ÷ 30 = ₹ {perDay.toFixed(2)}</div>
                          {(() => {
                            const emp = employees.find(e => e.id === Number(form.employee_id));
                            const expHrs = emp?.expected_working_hours || 9.0;
                            const perHour = perDay / expHrs;
                            return <div>Per hour = Per day ÷ {expHrs} hrs = <strong>₹ {perHour.toFixed(2)}</strong></div>;
                          })()}
                          <div>Salary payable = ₹ {perDay.toFixed(2)} × {modalAttendance.paidDays} days = <strong>₹ {payable.toFixed(2)}</strong></div>
                          <div>Deductions (proportional) = ₹ {ded.toFixed(2)} → <strong>Net ≈ ₹ {net.toFixed(2)}</strong></div>
                        </div>
                      );
                    })()}
                  </>
                ) : (
                  <SectionLoader size="sm" />
                )
              ) : (
                <p className="text-muted" style={{ margin: 0, fontSize: "0.9rem" }}>Select an employee to see total present days and salary calculation for this month.</p>
              )}
            </div>

            <form onSubmit={handleSubmit} className="modal-stack">
              <div className="form-grid-2" style={{ marginBottom: "1rem" }}>
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label>Employee</label>
                  <CustomSelect
                    value={String(form.employee_id || "")}
                    onChange={(val) => setForm((f) => ({ ...f, employee_id: val }))}
                    options={[
                      { value: "", label: "Select employee" },
                      ...employees.map((e) => ({
                        value: String(e.id),
                        label: `${e.employee_code} - ${e.full_name}`
                      }))
                    ]}
                    disabled={!!editing}
                  />
                </div>
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label>Effective From</label>
                  <input
                    type="date"
                    required
                    style={{ width: "100%" }}
                    value={form.effective_from}
                    onChange={(e) => setForm((f) => ({ ...f, effective_from: e.target.value }))}
                  />
                </div>
              </div>

              <div className="form-grid-2" style={{ marginBottom: "1rem" }}>
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label>Basic</label>
                  <input
                    type="number"
                    required
                    style={{ width: "100%" }}
                    value={form.basic}
                    onChange={(e) => setForm((f) => ({ ...f, basic: e.target.value }))}
                  />
                </div>
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label>HRA</label>
                  <input
                    type="number"
                    required
                    style={{ width: "100%" }}
                    value={form.hra}
                    onChange={(e) => setForm((f) => ({ ...f, hra: e.target.value }))}
                  />
                </div>
              </div>

              <div className="form-grid-2" style={{ marginBottom: "1rem" }}>
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label>Medical allowance</label>
                  <input
                    type="number"
                    step="any"
                    min={0}
                    style={{ width: "100%" }}
                    placeholder="0"
                    value={form.medical}
                    onChange={(e) => setForm((f) => ({ ...f, medical: e.target.value }))}
                  />
                </div>
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label>Travelling</label>
                  <input
                    type="number"
                    step="any"
                    min={0}
                    style={{ width: "100%" }}
                    placeholder="0"
                    value={form.travelling}
                    onChange={(e) => setForm((f) => ({ ...f, travelling: e.target.value }))}
                  />
                </div>
              </div>

              <div className="form-grid-2" style={{ marginBottom: "1rem" }}>
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label>Miscellaneous</label>
                  <input
                    type="number"
                    step="any"
                    min={0}
                    style={{ width: "100%" }}
                    placeholder="0"
                    value={form.miscellaneous}
                    onChange={(e) => setForm((f) => ({ ...f, miscellaneous: e.target.value }))}
                  />
                </div>
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label>Allowances</label>
                  <input
                    type="number"
                    style={{ width: "100%" }}
                    value={form.allowances}
                    onChange={(e) => setForm((f) => ({ ...f, allowances: e.target.value }))}
                  />
                </div>
              </div>

              <div className="form-grid-2" style={{ marginBottom: "1rem" }}>
              
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label>Deductions</label>
                  <input
                    type="number"
                    style={{ width: "100%" }}
                    value={form.deductions}
                    onChange={(e) => setForm((f) => ({ ...f, deductions: e.target.value }))}
                  />
                </div>
              </div>

              <div className="modal-actions" style={{ marginTop: "1rem" }}>
                <button type="submit" className="btn btn-primary btn-uniform" title="Save Salary Structure Changes">
                  Save
                </button>
                <button type="button" className="btn btn-cancel-alt btn-uniform" onClick={() => setShowModal(false)} title="Cancel Changes">
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Salary detail dialog for selected month */}
      {detailDialog && (
        <div className="modal-backdrop" onClick={() => setDetailDialog(null)} style={{ zIndex: 100 }}>
          <div
            className="modal"
            onClick={(e) => e.stopPropagation()}
            style={{
              maxWidth: 520,
              maxHeight: "90vh",
              overflow: "auto",
            }}
          >
            <h3 style={{ marginTop: 0, marginBottom: 4 }}>
              Salary- {detailDialog.employee.full_name}
            </h3>
            <div className="text-muted" style={{ marginBottom: "1rem", fontSize: "0.9rem" }}>
              {detailDialog.employee.employee_code} · {monthYearLabel()}
            </div>

            <section style={{ marginBottom: "1.25rem" }}>
              <div style={{ fontWeight: 600, marginBottom: "0.5rem" }}>Monthly structure</div>
              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
                <tbody>
                  <tr><td style={{ padding: "4px 8px 4px 0" }}>Basic</td><td style={{ textAlign: "right" }}>₹ {Number(detailDialog.structure.basic).toFixed(2)}</td></tr>
                  <tr><td style={{ padding: "4px 8px 4px 0" }}>HRA</td><td style={{ textAlign: "right" }}>₹ {Number(detailDialog.structure.hra).toFixed(2)}</td></tr>
                  <tr><td style={{ padding: "4px 8px 4px 0" }}>Medical</td><td style={{ textAlign: "right" }}>₹ {Number(detailDialog.structure.medical ?? 0).toFixed(2)}</td></tr>
                  <tr><td style={{ padding: "4px 8px 4px 0" }}>Travelling</td><td style={{ textAlign: "right" }}>₹ {Number(detailDialog.structure.travelling ?? 0).toFixed(2)}</td></tr>
                  <tr><td style={{ padding: "4px 8px 4px 0" }}>Miscellaneous</td><td style={{ textAlign: "right" }}>₹ {Number(detailDialog.structure.miscellaneous ?? 0).toFixed(2)}</td></tr>
                  <tr><td style={{ padding: "4px 8px 4px 0" }}>Allowances</td><td style={{ textAlign: "right" }}>₹ {Number(detailDialog.structure.allowances).toFixed(2)}</td></tr>
                  <tr style={{ borderTop: "1px solid rgba(255, 255, 255, 0.10)" }}>
                    <td style={{ padding: "6px 8px 6px 0" }}><strong>Gross</strong></td>
                    <td style={{ textAlign: "right" }}>
                      <strong>
                        ₹ {structureMonthlyGross(detailDialog.structure).toFixed(2)}
                      </strong>
                    </td>
                  </tr>
                  <tr><td style={{ padding: "4px 8px 4px 0" }}>Deductions</td><td style={{ textAlign: "right" }}>₹ {Number(detailDialog.structure.deductions).toFixed(2)}</td></tr>
                  <tr style={{ borderTop: "1px solid rgba(255, 255, 255, 0.05)" }}>
                    <td style={{ padding: "4px 8px 4px 0", color: "var(--brand-300)" }}>Per hour salary</td>
                    <td style={{ textAlign: "right", color: "var(--brand-300)" }}>
                      {(() => {
                        const gross = structureMonthlyGross(detailDialog.structure);
                        const perDay = gross / 30;
                        const expHrs = detailDialog.employee.expected_working_hours || 9.0;
                        return <strong>₹ {(perDay / expHrs).toFixed(2)}</strong>;
                      })()}
                    </td>
                  </tr>
                  <tr><td style={{ padding: "4px 8px 4px 0", fontSize: "0.8rem", color: "rgba(255, 255, 255, 0.72)" }}>Effective from</td><td style={{ textAlign: "right", fontSize: "0.8rem" }}>{detailDialog.structure.effective_from}</td></tr>
                </tbody>
              </table>
            </section>

            {detailDialog.payslip ? (
              <section style={{ marginBottom: "1.25rem" }}>
                <div style={{ fontWeight: 600, marginBottom: "0.5rem" }}>Payslip for this month</div>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.9rem" }}>
                  <tbody>
                    <tr><td style={{ padding: "4px 8px 4px 0" }}>Paid days</td><td style={{ textAlign: "right" }}>{detailDialog.payslip.paid_days}</td></tr>
                    <tr><td style={{ padding: "4px 8px 4px 0" }}>LOP days</td><td style={{ textAlign: "right" }}>{detailDialog.payslip.lop_days}</td></tr>
                    <tr><td style={{ padding: "4px 8px 4px 0" }}>Total earnings</td><td style={{ textAlign: "right" }}>₹ {Number(detailDialog.payslip.total_earnings).toFixed(2)}</td></tr>
                    <tr><td style={{ padding: "4px 8px 4px 0" }}>Total deductions</td><td style={{ textAlign: "right" }}>₹ {Number(detailDialog.payslip.total_deductions).toFixed(2)}</td></tr>
                    <tr style={{ borderTop: "1px solid rgba(255, 255, 255, 0.10)" }}>
                      <td style={{ padding: "6px 8px 6px 0" }}><strong>Net salary</strong></td>
                      <td style={{ textAlign: "right", fontWeight: 700 }}>₹ {Number(detailDialog.payslip.net_salary).toFixed(2)}</td>
                    </tr>
                  </tbody>
                </table>
                {/* Plain-language hours & deduction breakdown, day by day. */}
                {(() => {
                  let bd: any = null;
                  try {
                    bd = detailDialog.payslip.component_breakdown
                      ? JSON.parse(detailDialog.payslip.component_breakdown)
                      : null;
                  } catch { bd = null; }
                  if (!bd || !Array.isArray(bd.days) || bd.days.length === 0) {
                    return (
                      <div style={{ marginTop: "0.75rem", fontSize: "0.8rem", color: "rgba(255,255,255,0.5)" }}>
                        Re-run payroll for this month to see the day-by-day hours breakdown.
                      </div>
                    );
                  }
                  const cell: React.CSSProperties = { padding: "5px 8px", whiteSpace: "nowrap" };
                  return (
                    <div style={{ marginTop: "1rem", paddingTop: "1rem", borderTop: "1px solid rgba(255,255,255,0.10)" }}>
                      <div style={{ fontWeight: 600, marginBottom: "0.5rem" }}>Working hours &amp; deductions</div>

                      {/* One-line plain summary */}
                      <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginBottom: "0.75rem" }}>
                        <span style={{ padding: "0.35rem 0.7rem", borderRadius: 8, background: "rgba(255,255,255,0.06)", fontSize: "0.8rem" }}>
                          Worked <strong>{bd.worked_hours_total}h</strong> of <strong>{bd.expected_hours_total}h</strong>
                        </span>
                        <span style={{ padding: "0.35rem 0.7rem", borderRadius: 8, background: "rgba(34,197,94,0.14)", color: "#4ade80", fontSize: "0.8rem" }}>
                          Paid <strong>{bd.paid_days}</strong> of {bd.basis_days} days
                        </span>
                        <span style={{ padding: "0.35rem 0.7rem", borderRadius: 8, background: "rgba(239,68,68,0.14)", color: "#f87171", fontSize: "0.8rem" }}>
                          LOP <strong>{bd.lop_days}</strong> days
                        </span>
                        {Number(bd.short_leaves_used) > 0 && (
                          <span style={{ padding: "0.35rem 0.7rem", borderRadius: 8, background: "rgba(245,158,11,0.14)", color: "#fbbf24", fontSize: "0.8rem" }}>
                            {bd.short_leaves_used} free short leave used
                          </span>
                        )}
                      </div>

                      <div style={{ maxHeight: 280, overflowY: "auto", border: "1px solid rgba(255,255,255,0.08)", borderRadius: 8 }}>
                        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.78rem" }}>
                          <thead style={{ position: "sticky", top: 0, background: "#1a1a1a" }}>
                            <tr style={{ color: "rgba(255,255,255,0.6)", textAlign: "left" }}>
                              <th style={cell}>Date</th>
                              <th style={cell}>In – Out</th>
                              <th style={{ ...cell, textAlign: "right" }}>Worked</th>
                              <th style={{ ...cell, textAlign: "right" }}>Required</th>
                              <th style={{ ...cell, textAlign: "right" }}>Short by</th>
                              <th style={{ ...cell, textAlign: "right" }}>Deducted</th>
                              <th style={cell}>Reason</th>
                            </tr>
                          </thead>
                          <tbody>
                            {bd.days.map((d: any) => {
                              const off = d.expected_hours === 0;
                              const lop = Number(d.lop_days) > 0;
                              return (
                                <tr key={d.date} style={{ borderTop: "1px solid rgba(255,255,255,0.06)", opacity: off ? 0.55 : 1 }}>
                                  <td style={cell}>{d.date.slice(8)}/{d.date.slice(5, 7)} <span style={{ color: "rgba(255,255,255,0.45)" }}>{d.weekday}</span></td>
                                  <td style={{ ...cell, color: "rgba(255,255,255,0.7)" }}>
                                    {d.in_time ? `${d.in_time} – ${d.out_time || "…"}` : "—"}
                                  </td>
                                  <td style={{ ...cell, textAlign: "right" }}>{d.worked_hours != null ? `${d.worked_hours}h` : "—"}</td>
                                  <td style={{ ...cell, textAlign: "right", color: "rgba(255,255,255,0.55)" }}>{off ? "—" : `${d.expected_hours}h`}</td>
                                  <td style={{ ...cell, textAlign: "right", color: d.short_hours > 0 ? "#fbbf24" : "rgba(255,255,255,0.4)" }}>
                                    {d.short_hours > 0 ? `${d.short_hours.toFixed(2)}h` : "—"}
                                  </td>
                                  <td style={{ ...cell, textAlign: "right", fontWeight: 700, color: lop ? "#f87171" : "#4ade80" }}>
                                    {lop ? `−${Number(d.lop_days).toFixed(2)}` : "0"}
                                  </td>
                                  <td style={{ ...cell, color: "rgba(255,255,255,0.6)" }}>{d.note}</td>
                                </tr>
                              );
                            })}
                          </tbody>
                        </table>
                      </div>
                      <div style={{ fontSize: "0.72rem", color: "rgba(255,255,255,0.45)", marginTop: "0.5rem" }}>
                        Pay is based on hours worked, not arrival time. A day is only deducted when hours fall short —
                        the shortfall is charged proportionally (short hours ÷ required hours).
                      </div>
                    </div>
                  );
                })()}

                {!showFormulaInDetail ? (
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    style={{ marginTop: "0.5rem" }}
                    onClick={() => setShowFormulaInDetail(true)}
                  >
                    View calculation formula
                  </button>
                ) : (
                  <div style={{ marginTop: "1rem", paddingTop: "1rem", borderTop: "1px solid rgba(255, 255, 255, 0.10)" }}>
                    <SalaryFormulaView payslip={detailDialog.payslip} />
                  </div>
                )}
              </section>
            ) : (
              <p className="text-muted" style={{ fontSize: "0.9rem", marginBottom: "1rem" }}>
                Payslip not generated for this month. Run payroll from Payroll Periods.
              </p>
            )}

            <div style={{ display: "flex", gap: "0.5rem", flexWrap: "wrap", marginTop: "1rem", paddingTop: "1rem", borderTop: "1px solid rgba(255, 255, 255, 0.10)" }}>
              {canEdit && (
                <>
                  <button
                    type="button"
                    className="btn btn-secondary"
                    onClick={() => {
                      setDetailDialog(null);
                      openEdit(detailDialog.structure);
                    }}
                  >
                    Edit structure
                  </button>
                  <button
                    type="button"
                    className="btn btn-danger"
                    onClick={() => handleDelete(detailDialog.structure)}
                    title="Delete this Salary Structure"
                  >
                    Delete structure
                  </button>
                </>
              )}
              <button type="button" className="btn btn-secondary" onClick={() => setDetailDialog(null)} title="Close Detail View">
                Close
              </button>
            </div>
          </div>
        </div>
      )}

      <ConfirmModal
        isOpen={!!confirmDelete}
        onClose={() => setConfirmDelete(null)}
        onConfirm={confirmActualDelete}
        title="Are you absolutely sure?"
        message={
          confirmDelete ? (
            <>
              You are about to delete the salary structure for{" "}
              <strong>{employees.find((e) => e.id === confirmDelete.employee_id)?.full_name}</strong> effective from{" "}
              <strong>{confirmDelete.effective_from}</strong>.
            </>
          ) : (
            ""
          )
        }
        confirmText="Yes, Delete Structure"
      />
    </div>
  );
}


