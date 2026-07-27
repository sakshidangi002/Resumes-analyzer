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

// Premium SVG Icons for Actions
const Icons = {
  View: () => (
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
    <>
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="page-title">Payslip Management</h1>
          <div className="page-subtitle">Create, edit, and manage payslips</div>
        </div>
        <GlobalHeaderControls />
      </div>

      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem", flexWrap: "wrap", gap: "1rem" }}>
          <h3 style={{ margin: 0 }}>Payslips</h3>
          <div style={{ display: "flex", gap: "0.75rem" }}>
            <div style={{ minWidth: "160px" }}>
              <CustomSelect
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
            </div>
            <div style={{ minWidth: "120px" }}>
              <CustomSelect
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
        </div>

        {loading ? (
          <div style={{ padding: "3rem 0" }}><SectionLoader size="md" /></div>
        ) : sortedPayslips.length === 0 ? (
          <p className="text-muted" style={{ padding: "1.5rem", textAlign: "center", background: "rgba(255,255,255,0.02)", borderRadius: "12px", border: "1px dashed rgba(255,255,255,0.1)" }}>
            No payslips found for the selected period.
          </p>
        ) : (
          <div className="table-wrap table-wrap--dark">
            <table className="table-modern table-modern--dark">
              <thead>
                <tr>
                  <th>Employee</th>
                  <th className="hide-sm" style={{ textAlign: 'center' }}>Period</th>
                  <th style={{ textAlign: 'center' }}>Net Salary</th>
                  <th className="hide-md" style={{ textAlign: 'center' }}>Paid Days</th>
                  <th className="hide-md" style={{ textAlign: 'center' }}>LOP Days</th>
                  <th className="hide-sm" style={{ textAlign: 'center' }}>
                    <div style={{ display: 'flex', justifyContent: 'center' }}>Generated</div>
                  </th>
                  <th className="actions-center">Actions</th>
                </tr>
              </thead>
              <tbody>
                {sortedPayslips.map((p) => (
                  <tr key={p.id}>
                    <td>{employeeLabel(p.employee_id)}</td>
                    <td className="hide-sm" style={{ textAlign: "center" }}>{periodLabel(p.payroll_period_id)}</td>
                    <td style={{ textAlign: 'center' }}>{Number(p.net_salary).toFixed(2)}</td>
                    <td className="hide-md" style={{ textAlign: "center" }}>{p.paid_days}</td>
                    <td className="hide-md" style={{ textAlign: "center" }}>{p.lop_days}</td>
                    <td className="hide-sm" style={{ textAlign: 'center' }}>
                      <div style={{ display: 'flex', justifyContent: 'center' }}>{fmtDateTime(p.generated_at)}</div>
                    </td>
                    <td className="actions-center">
                      <div className="actions-stack horizontal">
                        <button
                          type="button"
                          className="btn btn-secondary btn-icon btn-sm"
                          onClick={() => setFormulaPayslip(p)}
                          title="View Calculation"
                        >
                          <Icons.View />
                        </button>
                        <button
                          type="button"
                          className="btn btn-danger btn-icon btn-sm"
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
    </>
  );
}
