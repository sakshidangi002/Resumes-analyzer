import { useEffect, useState, useMemo, type CSSProperties } from "react";
import { payroll as payrollApi, employees as employeesApi, attendance as attendanceApi } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import SalaryFormulaView from "../components/SalaryFormulaView";
import ConfirmModal from "../components/ConfirmModal";
import { SectionLoader } from "../components/LoadingState";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import CustomSelect from "../components/CustomSelect";
import { useTableControls, SortableHeader, TableToolbar } from "../components/dataTable";

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

// Premium SVG Icons for Actions
const Icons = {
  View: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
      <circle cx="12" cy="12" r="3"></circle>
    </svg>
  ),
};

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

  const rows = useMemo(() => {
    return employees
      .filter((emp) => monthStructures.has(emp.id))
      .map((emp) => {
        const structure = monthStructures.get(emp.id)!;
        const payslip = payslips.find((p) => p.employee_id === emp.id) ?? null;
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
  }, [employees, monthStructures, payslips]);

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
    <>
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="page-title">Payroll Management</h1>
          <div className="page-subtitle">Manage salary structures and payroll processing</div>
        </div>
        <GlobalHeaderControls />
      </div>

      <div className="card" style={{ marginBottom: "1rem" }}>
        <div className="payroll-filter-bar">
          <span className="payroll-filter-bar__label">Monthly Payroll Summary</span>
          <div className="payroll-filter-bar__controls">
            <CustomSelect
              value={String(selectedMonth)}
              onChange={(val) => setSelectedMonth(Number(val))}
              options={MONTHS.map((m, i) => ({ value: String(i + 1), label: m }))}
              style={{ width: "120px" }}
            />
            <input
              type="number"
              value={selectedYear}
              onChange={(e) => setSelectedYear(Number(e.target.value))}
              min={2020}
              max={2030}
              className="payroll-filter-bar__year"
            />
          </div>

        </div>
      </div>

      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem", flexWrap: "wrap", gap: "0.5rem" }}>
          <h3 style={{ margin: 0 }}>
            Employees - {monthYearLabel()}
          </h3>
        </div>
        <TableToolbar
          search={rowSearch}
          onSearchChange={setRowSearch}
          placeholder="Search employee or payslip status..."
          showClear={rowHasActive}
          onClear={clearRowControls}
          count={{ shown: displayedRows.length, total: rows.length }}
          rightControls={
            canEdit ? (
              <button type="button" className="btn btn-primary btn-uniform" onClick={openAdd} title="Add New Salary Structure/Payroll">
                Add Payroll
              </button>
            ) : null
          }
        />

        {loading ? (
          <SectionLoader rows={5} />
        ) : rows.length === 0 ? (
          <p className="text-muted">No salary structure effective for {monthYearLabel()}. Add payroll with effective date covering this month.</p>
        ) : (
          <div className="table-wrap table-wrap--dark" style={payrollTableShellStyle}>
            <table className="table-modern table-modern--dark" style={payrollTableStyle}>
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
                  <SortableHeader className="hide-md" label="S.NO" columnKey="__sno" sort={rowSort} onToggle={toggleRowSort} align="center" notSortable style={{ whiteSpace: "nowrap" }} />
                  <SortableHeader label="EMPLOYEE" columnKey="employee" sort={rowSort} onToggle={toggleRowSort} style={{ whiteSpace: "nowrap" }} />
                  <SortableHeader className="hide-sm" label="PAID DAYS" columnKey="paid_days" sort={rowSort} onToggle={toggleRowSort} align="center" style={{ whiteSpace: "nowrap" }} />
                  <SortableHeader className="hide-sm" label="LOP DAYS" columnKey="lop_days" sort={rowSort} onToggle={toggleRowSort} align="center" style={{ whiteSpace: "nowrap" }} />
                  <SortableHeader className="hide-md" label="GROSS (MONTH)" columnKey="gross" sort={rowSort} onToggle={toggleRowSort} align="center" style={{ whiteSpace: "nowrap" }} />
                  <SortableHeader className="hide-lg" label="EARNINGS" columnKey="earnings" sort={rowSort} onToggle={toggleRowSort} align="center" style={{ whiteSpace: "nowrap" }} />
                  <SortableHeader className="hide-lg" label="DEDUCTIONS" columnKey="deductions" sort={rowSort} onToggle={toggleRowSort} align="center" style={{ whiteSpace: "nowrap" }} />
                  <SortableHeader label="NET SALARY" columnKey="net" sort={rowSort} onToggle={toggleRowSort} align="center" style={{ whiteSpace: "nowrap" }} />
                  <SortableHeader className="hide-sm" label="PAYSLIP" columnKey="payslip_status" sort={rowSort} onToggle={toggleRowSort} align="center" style={{ whiteSpace: "nowrap" }} />
                  <SortableHeader label="ACTIONS" columnKey="__actions" sort={rowSort} onToggle={toggleRowSort} align="center" notSortable style={{ whiteSpace: "nowrap" }} />
                </tr>
              </thead>
              <tbody>
                {displayedRows.length === 0 && (
                  <tr>
                    <td colSpan={10} style={{ textAlign: "center", padding: "1.25rem", opacity: 0.65 }}>
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
                      <td className="hide-md" style={{ textAlign: "center" }}>{idx + 1}</td>
                      <td>
                        <div style={{ fontWeight: 600, textAlign: "left", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>
                          {row.employee.full_name}
                        </div>
                      </td>
                      <td className="hide-sm" style={{ whiteSpace: "nowrap", textAlign: "center" }}>
                        {p ? paidDays : "-"}
                      </td>
                      <td className="hide-sm" style={{ whiteSpace: "nowrap", textAlign: "center" }}>
                        {p ? lopDays : "-"}
                      </td>
                      <td className="hide-md" style={{ whiteSpace: "nowrap", textAlign: "center" }}>
                        ₹ {row.gross.toFixed(2)}
                      </td>
                      <td className="hide-lg" style={{ whiteSpace: "nowrap", textAlign: "center" }}>
                        {p ? `₹ ${earnings.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "-"}
                      </td>
                      <td className="hide-lg" style={{ whiteSpace: "nowrap", textAlign: "center" }}>
                        {p ? `₹ ${deductions.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "-"}
                      </td>
                      <td style={{ fontWeight: 700, whiteSpace: "nowrap", color: "#60a5fa", textAlign: "center" }}>
                        {p ? `₹ ${net.toLocaleString("en-IN", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "-"}
                      </td>
                      <td className="hide-sm" style={{ whiteSpace: "nowrap", textAlign: "center" }}>
                        {p ? (
                          <span className="payslip-pill payslip-pill--generated">Generated</span>
                        ) : (
                          <span className="payslip-pill payslip-pill--notrun">Not run</span>
                        )}
                      </td>
                      <td style={{ textAlign: "center" }}>
                        <div style={{ display: "flex", justifyContent: "center" }}>
                          <button
                            type="button"
                            className="btn btn-secondary btn-icon btn-sm"
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
    </>
  );
}


