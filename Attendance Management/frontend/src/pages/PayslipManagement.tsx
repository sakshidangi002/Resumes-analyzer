import { useEffect, useState } from "react";
import { payroll as payrollApi, employees as employeesApi } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import SalaryFormulaView from "../components/SalaryFormulaView";
import ConfirmModal from "../components/ConfirmModal";
import { SectionLoader } from "../components/LoadingState";
import CustomSelect from "../components/CustomSelect";
import { formatDate } from "../utils/dateFormatter";
import { useMemo } from "react";

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
  generated_at: string;
}

interface PayrollPeriod {
  id: number;
  month: number;
  year: number;
}

interface EmployeeOption {
  id: number;
  employee_code: string;
  full_name: string;
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
  Delete: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 21 6"></polyline>
      <path d="M8 6V4h8v2"></path>
      <path d="M6 6l1 14h10l1-14"></path>
    </svg>
  ),
  Payslip: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"></path>
      <polyline points="14 3 14 8 19 8"></polyline>
      <polyline points="9 14 11 16 15 12"></polyline>
    </svg>
  ),
};

export default function PayslipManagement() {
  const { hasRole } = useAuth();
  const [payslips, setPayslips] = useState<Payslip[]>([]);
  const [periods, setPeriods] = useState<PayrollPeriod[]>([]);
  const [employees, setEmployees] = useState<EmployeeOption[]>([]);
  const [loading, setLoading] = useState(true);
  const [formulaPayslip, setFormulaPayslip] = useState<Payslip | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<Payslip | null>(null);
  const [filterMonth, setFilterMonth] = useState<string>("");
  const [filterYear, setFilterYear] = useState<string>("");

  const canManage = hasRole("Admin") || hasRole("HR");

  const loadData = () => {
    setLoading(true);
    Promise.all([payrollApi.payslips(), payrollApi.periods(), employeesApi.list()])
      .then(([pRes, perRes, eRes]) => {
        setPayslips(pRes.data);
        setPeriods(perRes.data);
        setEmployees(
          (eRes.data as any[]).map((e: any) => ({
            id: e.id,
            employee_code: e.employee_code,
            full_name: e.full_name,
          }))
        );
      })
      .catch(() => { })
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadData();
  }, []);

  const employeeLabel = (id: number) => {
    const e = employees.find((x) => x.id === id);
    return e ? e.full_name : `#${id}`;
  };

  const periodLabel = (id: number) => {
    const p = periods.find((x) => x.id === id);
    if (!p) return `#${id}`;
    const mm = String(p.month).padStart(2, "0");
    return `${p.year}-${mm}`;
  };

  const fmtDateTime = (d: string) => {
    if (!d) return "-";
    const dt = new Date(d);
    return isFinite(dt.getTime()) ? formatDate(d) + ", " + dt.toLocaleTimeString() : d;
  };

  if (!canManage) {
    return (
      <div className="card">
        <p className="text-muted">You do not have access to payslip management.</p>
      </div>
    );
  }

  const handleDelete = (p: Payslip) => {
    setConfirmDelete(p);
  };

  const confirmActualDelete = () => {
    if (!confirmDelete) return;
    payrollApi
      .deletePayslip(confirmDelete.id)
      .then(() => {
        loadData();
        setConfirmDelete(null);
      })
      .catch((err) => alert(err.response?.data?.detail || "Delete failed"));
  };

  const sortKeyForEmployeeCode = (code: string) => {
    const n = Number(code);
    if (!Number.isNaN(n)) return { isNumeric: true, num: n, raw: code };
    return { isNumeric: false, num: 0, raw: code.toUpperCase() };
  };

  const filteredPayslips = useMemo(() => {
    return payslips.filter(p => {
      const per = periods.find(x => x.id === p.payroll_period_id);
      if (!per) return false;
      const matchMonth = !filterMonth || String(per.month) === filterMonth;
      const matchYear = !filterYear || String(per.year) === filterYear;
      return matchMonth && matchYear;
    });
  }, [payslips, filterMonth, filterYear, periods]);

  const sortedPayslips = [...filteredPayslips].sort((a, b) => {
    const ea = employees.find((e) => e.id === a.employee_id);
    const eb = employees.find((e) => e.id === b.employee_id);
    const ca = sortKeyForEmployeeCode(ea?.employee_code || "");
    const cb = sortKeyForEmployeeCode(eb?.employee_code || "");
    if (ca.isNumeric && cb.isNumeric) {
      return ca.num - cb.num;
    }
    if (ca.isNumeric !== cb.isNumeric) {
      return ca.isNumeric ? -1 : 1;
    }
    return ca.raw.localeCompare(cb.raw);
  });


  return (
    <div className="eds">
      <header className="eds-topbar">
        <div>
          <h1 className="eds-title">Payslip Registry</h1>
          <p className="eds-subtitle">Create, edit, and manage payslips</p>
        </div>
        <GlobalHeaderControls />
      </header>

      <div className="eds-page">
        <section className="eds-card">
          <div className="eds-card-head">
            <span className="eds-chip eds-chip--emerald"><Icons.Payslip /></span>
            <div className="eds-card-titles">
              <h2 className="eds-card-title">Payslips</h2>
            </div>
            <div style={{ marginLeft: "auto", display: "flex", gap: "0.6rem", flexShrink: 0 }}>
              <CustomSelect
                className="eds-cselect eds-cselect--filter"
                value={filterMonth}
                onChange={setFilterMonth}
                placeholder="All Months"
                options={[
                  { value: "", label: "All Months" },
                  ...Array.from({ length: 12 }, (_, i) => ({
                    value: String(i + 1),
                    label: new Date(2000, i).toLocaleString("default", { month: "long" })
                  }))
                ]}
              />
              <CustomSelect
                className="eds-cselect eds-cselect--year"
                value={filterYear}
                onChange={setFilterYear}
                placeholder="All Years"
                options={[
                  { value: "", label: "All Years" },
                  ...Array.from(new Set(periods.map(p => p.year)))
                    .sort((a, b) => b - a)
                    .map(y => ({ value: String(y), label: String(y) }))
                ]}
              />
            </div>
          </div>

          {loading ? (
            <div style={{ padding: "3rem 0" }}><SectionLoader size="md" /></div>
          ) : sortedPayslips.length === 0 ? (
            <div className="eds-empty--card">
              <span className="eds-empty-tile"><Icons.Payslip /></span>
              <span>No payslips found for the selected period.</span>
            </div>
          ) : (
            <div className="eds-table-wrap">
              <table className="eds-table eds-table--auto payslip-table">
                <thead>
                  <tr>
                    <th>Employee</th>
                    <th className="hide-sm">Period</th>
                    <th className="is-actions pay-center">Net Salary</th>
                    <th className="hide-md is-actions pay-center">Paid Days</th>
                    <th className="hide-md is-actions pay-center">LOP Days</th>
                    <th className="hide-sm">Generated</th>
                    <th className="is-actions">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedPayslips.map((p) => (
                    <tr key={p.id}>
                      <td className="eds-cell-strong">{employeeLabel(p.employee_id)}</td>
                      <td className="hide-sm eds-cell-time">{periodLabel(p.payroll_period_id)}</td>
                      <td className="eds-money eds-money--net pay-center">{Number(p.net_salary).toFixed(2)}</td>
                      <td className="hide-md eds-money pay-center">{p.paid_days}</td>
                      <td className={`hide-md eds-money pay-center${Number(p.lop_days) > 0 ? " eds-money--lop" : " eds-money--zero"}`}>{p.lop_days}</td>
                      <td className="hide-sm eds-cell-dim">{fmtDateTime(p.generated_at)}</td>
                      <td>
                        <div className="eds-rowactions">
                          <button
                            type="button"
                            className="eds-iconbtn eds-iconbtn--view"
                            onClick={() => setFormulaPayslip(p)}
                            title="View Calculation"
                          >
                            <Icons.View />
                          </button>
                          <button
                            type="button"
                            className="eds-iconbtn eds-iconbtn--del"
                            onClick={() => handleDelete(p)}
                            title="Delete Payslip"
                          >
                            <Icons.Delete />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>

      {formulaPayslip && (
        <div className="modal-backdrop" onClick={() => setFormulaPayslip(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 560 }}>
            <h3 style={{ marginTop: 0 }}>
              Salary calculation – {employeeLabel(formulaPayslip.employee_id)} ({periodLabel(formulaPayslip.payroll_period_id)})
            </h3>
            <SalaryFormulaView
              payslip={formulaPayslip}
            />
            <div style={{ marginTop: "1rem" }}>
              <button type="button" className="btn btn-cancel-alt" onClick={() => setFormulaPayslip(null)}>
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
              You are about to delete payslip <strong>#{confirmDelete.id}</strong> for{" "}
              <strong>{employeeLabel(confirmDelete.employee_id)}</strong>.
            </>
          ) : (
            ""
          )
        }
        confirmText="Yes, Delete Payslip"
      />
    </div>
  );
}
