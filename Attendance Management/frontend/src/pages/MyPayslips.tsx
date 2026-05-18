import { useEffect, useState } from "react";
import { payroll as payrollApi, employees as employeesApi, company as companyApi } from "../api/client";
import SalaryFormulaView from "../components/SalaryFormulaView";
import { useAuth } from "../auth/AuthContext";
import { SectionLoader } from "../components/LoadingState";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import { formatDate } from "../utils/dateFormatter";
import { jsPDF } from "jspdf";
import html2canvas from "html2canvas";
import CustomSelect from "../components/CustomSelect";
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

export default function MyPayslips() {
  const { user } = useAuth();
  const [payslips, setPayslips] = useState<Payslip[]>([]);
  const [periods, setPeriods] = useState<PayrollPeriod[]>([]);
  const [loading, setLoading] = useState(true);
  const [formulaPayslip, setFormulaPayslip] = useState<Payslip | null>(null);
  const [employeeMeta, setEmployeeMeta] = useState<{ code?: string; name?: string }>({});
  const [companyName, setCompanyName] = useState("");
  const [filterMonth, setFilterMonth] = useState<string>("");
  const [filterYear, setFilterYear] = useState<string>("");

  const fmt2 = (n: any) => Number(n || 0).toFixed(2);
  const currency = (n: any) => `₹ ${fmt2(n)}`;
  const esc = (s: any) =>
    String(s ?? "")
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#039;");

  const downloadPayslip = (s: Payslip) => {
    const per = periodLabel(s.payroll_period_id);
    let breakdown: any = null;
    if (s.component_breakdown) {
      try {
        breakdown = JSON.parse(s.component_breakdown);
      } catch {
        breakdown = null;
      }
    }
    const empLine = [employeeMeta.code, employeeMeta.name].filter(Boolean).join(" - ");
    const generated = formatDate(s.generated_at) + " " + new Date(s.generated_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });

    const html = `
<div id="payslip-render" style="font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif; padding: 40px; background: #ffffff; color: #0b1220; width: 800px; line-height: 1.5;">
  <div style="border: 1px solid #e2e8f0; border-radius: 16px; padding: 32px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);">
    <div style="display: flex; align-items: flex-start; justify-content: space-between; gap: 24px; margin-bottom: 32px;">
      <div>
        <h2 style="margin: 0 0 6px; font-size: 14px; color: #64748b; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 800;">${esc(companyName || "Softwiz Infotech")}</h2>
        <h1 style="margin: 0; font-size: 32px; font-weight: 900; letter-spacing: -0.025em; color: #0f172a;">Payslip</h1>
        <p style="margin: 8px 0 0; color: #64748b; font-size: 15px; font-weight: 500;">${esc(empLine || "Employee")} · Period ${esc(per)} · Generated ${esc(generated)}</p>
      </div>
      <div style="display: flex; gap: 12px; padding: 16px 20px; border-radius: 14px; border: 1px solid #e2e8f0; background: #f8fafc; min-width: 180px;">
        <div>
          <small style="font-weight: 700; color: #64748b; font-size: 12px; text-transform: uppercase; letter-spacing: 0.05em;">Net Salary</small><br />
          <span style="font-size: 24px; font-weight: 900; color: #0f172a;">${esc(currency(s.net_salary))}</span>
        </div>
      </div>
    </div>

    <div style="margin-bottom: 32px;">
      <h3 style="font-size: 16px; font-weight: 700; color: #334155; margin: 0 0 16px; text-transform: uppercase; letter-spacing: 0.05em;">Salary Summary</h3>
      <table style="width: 100%; border-collapse: separate; border-spacing: 0;">
        <thead>
          <tr>
            <th style="padding: 12px 16px; border-bottom: 2px solid #f1f5f9; text-align: left; background: #f8fafc; font-size: 13px; font-weight: 700; color: #475569; border-radius: 8px 0 0 8px;">Description</th>
            <th style="padding: 12px 16px; border-bottom: 2px solid #f1f5f9; text-align: right; background: #f8fafc; font-size: 13px; font-weight: 700; color: #475569; border-radius: 0 8px 8px 0;">Amount</th>
          </tr>
        </thead>
        <tbody>
          <tr><td style="padding: 16px; border-bottom: 1px solid #f1f5f9; color: #334155;">Gross salary</td><td style="padding: 16px; border-bottom: 1px solid #f1f5f9; text-align: right; font-weight: 600; color: #0f172a;">${esc(currency(s.gross_salary))}</td></tr>
          <tr><td style="padding: 16px; border-bottom: 1px solid #f1f5f9; color: #334155;">Total earnings</td><td style="padding: 16px; border-bottom: 1px solid #f1f5f9; text-align: right; font-weight: 600; color: #0f172a;">${esc(currency(s.total_earnings))}</td></tr>
          <tr><td style="padding: 16px; border-bottom: 1px solid #f1f5f9; color: #334155;">Total deductions</td><td style="padding: 16px; border-bottom: 1px solid #f1f5f9; text-align: right; font-weight: 600; color: #dc2626;">${esc(currency(s.total_deductions))}</td></tr>
          <tr style="background: #f8fafc;"><td style="padding: 16px; border-radius: 0 0 0 8px;"><strong>Net salary (Payable)</strong></td><td style="padding: 16px; text-align: right; border-radius: 0 0 8px 0;"><strong>${esc(currency(s.net_salary))}</strong></td></tr>
        </tbody>
      </table>
    </div>

    <div style="display: flex; gap: 24px;">
      <div style="flex: 1; border: 1px solid #e2e8f0; border-radius: 14px; padding: 20px; background: #ffffff;">
        <h3 style="margin: 0 0 16px; font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b;">Attendance</h3>
        <div style="display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px dashed #e2e8f0;"><span style="color: #475569;">Paid days</span><strong style="color: #0f172a;">${esc(fmt2(s.paid_days))}</strong></div>
        <div style="display: flex; justify-content: space-between; padding: 10px 0;"><span style="color: #475569;">LOP days</span><strong style="color: #dc2626;">${esc(fmt2(s.lop_days))}</strong></div>
        <p style="color: #94a3b8; font-size: 11px; margin-top: 12px; font-style: italic;">Calculated based on attendance records and approved leave requests.</p>
      </div>
      <div style="flex: 1; border: 1px solid #e2e8f0; border-radius: 14px; padding: 20px; background: #ffffff;">
        <h3 style="margin: 0 0 16px; font-size: 14px; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b;">Components</h3>
        ${breakdown
        ? `
        <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px dashed #e2e8f0;"><span style="color: #475569;">Basic</span><strong style="color: #0f172a;">${esc(currency(breakdown.basic ?? 0))}</strong></div>
        <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px dashed #e2e8f0;"><span style="color: #475569;">HRA</span><strong style="color: #0f172a;">${esc(currency(breakdown.hra ?? 0))}</strong></div>
        <div style="display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px dashed #e2e8f0;"><span style="color: #475569;">Allowances</span><strong style="color: #0f172a;">${esc(currency(breakdown.allowances ?? 0))}</strong></div>
        <div style="display: flex; justify-content: space-between; padding: 8px 0;"><span style="color: #475569;">Deductions</span><strong style="color: #dc2626;">${esc(currency(breakdown.deductions ?? 0))}</strong></div>
        `
        : `<p style="color: #94a3b8; font-size: 12px;">Component breakdown details are not available for this period.</p>`
      }
      </div>
    </div>

    <div style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #f1f5f9; text-align: center;">
      <p style="color: #94a3b8; font-size: 12px; margin: 0;">This is a computer generated document and does not require a signature.</p>
      <p style="color: #64748b; font-size: 13px; font-weight: 600; margin: 4px 0 0;">${esc(companyName || "Softwiz Infotech")}</p>
    </div>
  </div>
</div>`;

    const container = document.createElement("div");
    container.style.position = "fixed";
    container.style.left = "-9999px";
    container.style.top = "0";
    container.innerHTML = html;
    document.body.appendChild(container);

    const codePart = (employeeMeta.code || "EMP").replace(/[^\w-]+/g, "_");
    const fileName = `payslip_${codePart}_${per.replace(/[^\w-]+/g, "_")}.pdf`;

    // Wait a tiny bit for fonts/styles to stabilize
    setTimeout(() => {
      html2canvas(container, {
        scale: 2, // Better quality
        useCORS: true,
        backgroundColor: "#ffffff",
        width: 800,
      }).then((canvas) => {
        const imgData = canvas.toDataURL("image/png");
        const pdf = new jsPDF({
          orientation: "p",
          unit: "pt",
          format: "a4",
        });

        const pdfWidth = pdf.internal.pageSize.getWidth();
        const pdfHeight = (canvas.height * pdfWidth) / canvas.width;

        pdf.addImage(imgData, "PNG", 0, 0, pdfWidth, pdfHeight);
        pdf.save(fileName);

        document.body.removeChild(container);
      }).catch((err) => {
        console.error("PDF generation failed:", err);
        document.body.removeChild(container);
        alert("Failed to generate PDF. Please try again.");
      });
    }, 100);
  };

  useEffect(() => {
    setLoading(true);
    const myEmployeeId = user?.employee_id ?? null;
    Promise.all([
      myEmployeeId ? payrollApi.payslips({ employee_id: myEmployeeId }) : payrollApi.payslips({ employee_id: -1 }),
      payrollApi.periods(),
    ])
      .then(([pRes, perRes]) => {
        const rows = (pRes.data || []) as Payslip[];
        setPayslips(myEmployeeId ? rows.filter((r) => r.employee_id === myEmployeeId) : []);
        setPeriods(perRes.data);
      })
      .catch(() => { })
      .finally(() => setLoading(false));

    companyApi.config().then(r => setCompanyName(r.data?.name || "Softwiz Infotech")).catch(() => setCompanyName("Softwiz Infotech"));
  }, [user?.employee_id]);

  useEffect(() => {
    if (!user?.employee_id) {
      setEmployeeMeta({});
      return;
    }
    employeesApi
      .get(user.employee_id)
      .then((r) => {
        const e: any = r.data;
        const name = e?.full_name || `${e?.first_name || ""} ${e?.last_name || ""}`.trim();
        const code = e?.employee_code ? String(e.employee_code) : "";
        setEmployeeMeta({ code, name });
      })
      .catch(() => setEmployeeMeta({}));
  }, [user?.employee_id]);

  const periodLabel = (periodId: number) => {
    const p = periods.find((x) => x.id === periodId);
    if (!p) return `#${periodId}`;
    return `${p.year}-${String(p.month).padStart(2, "0")}`;
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


  return (
    <>
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="page-title">My Payslips</h1>
          <div className="page-subtitle">
            Download and view your payslips
            {employeeMeta.code || employeeMeta.name ? ` · ${[employeeMeta.code, employeeMeta.name].filter(Boolean).join(" - ")}` : ""}
          </div>
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
        ) : filteredPayslips.length === 0 ? (
          <p className="text-muted" style={{ padding: "1.5rem", textAlign: "center", background: "rgba(255,255,255,0.02)", borderRadius: "12px", border: "1px dashed rgba(255,255,255,0.1)" }}>
            No payslips found for the selected period.
          </p>
        ) : (
          <div className="table-wrap table-wrap--dark">
            <table className="table-modern table-modern--dark">
              <thead>
                <tr>
                  <th style={{ textAlign: 'left', paddingLeft: '1.5rem', width: '120px' }}>Period</th>
                  <th style={{ width: '150px', textAlign: 'left' }}>Net Salary</th>
                  <th className="hide-md" style={{ width: '120px', textAlign: 'left' }}>Paid Days</th>
                  <th className="hide-md" style={{ width: '120px', textAlign: 'left' }}>LOP Days</th>
                  <th className="hide-sm" style={{ width: '220px', textAlign: 'center' }}>
                    <div style={{ display: 'flex', justifyContent: 'center' }}>Generated</div>
                  </th>
                  <th style={{ width: '320px', textAlign: 'center' }}>
                    <div style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>Actions</div>
                  </th>
                </tr>
              </thead>
              <tbody>
                {filteredPayslips.map((s) => (
                  <tr key={s.id}>
                    <td style={{ textAlign: 'left', paddingLeft: '1.5rem' }}>{periodLabel(s.payroll_period_id)}</td>
                    <td style={{ fontWeight: 600, textAlign: 'center' }}>
                      <div style={{ display: 'flex', justifyContent: 'center' }}>₹ {Number(s.net_salary || 0).toLocaleString('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</div>
                    </td>
                    <td className="hide-md" style={{ textAlign: 'center' }}>
                      <div style={{ display: 'flex', justifyContent: 'center' }}>{Number(s.paid_days || 0).toFixed(2)}</div>
                    </td>
                    <td className="hide-md" style={{ textAlign: 'center' }}>
                      <div style={{ display: 'flex', justifyContent: 'center' }}>{Number(s.lop_days || 0).toFixed(2)}</div>
                    </td>
                    <td className="hide-sm" style={{ textAlign: 'center' }}>
                      <div style={{ display: 'flex', justifyContent: 'center' }}>{formatDate(s.generated_at)} {new Date(s.generated_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</div>
                    </td>
                    <td style={{ textAlign: 'center' }}>
                      <div style={{ display: 'flex', gap: '0.75rem', justifyContent: 'center' }}>
                        <button
                          type="button"
                          className="btn btn-secondary btn-sm"
                          style={{ minWidth: '130px', justifyContent: 'center', backgroundColor: "var(--brand-500)" }}
                          onClick={() => setFormulaPayslip(s)}
                          title="View Detailed Salary Calculation Breakdown"
                        >
                          View calculation
                        </button>
                        <button
                          type="button"
                          className="btn btn-primary btn-sm"
                          style={{ minWidth: '130px', justifyContent: 'center' }}
                          onClick={() => downloadPayslip(s)}
                          title="Download Payslip as PDF"
                        >
                          Download
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
            <div style={{ marginBottom: "1rem" }}>
              <div style={{ fontSize: "0.75rem", fontWeight: 800, color: "var(--brand-400)", textTransform: "uppercase", letterSpacing: "0.1em" }}>{companyName || "Softwiz Infotech"}</div>
              <h3 style={{ marginTop: "0.25rem", marginBottom: 0 }}>
                Salary calculation – {periodLabel(formulaPayslip.payroll_period_id)}
              </h3>
            </div>
            <SalaryFormulaView payslip={formulaPayslip} />
            <div style={{ marginTop: "1rem", display: "flex", justifyContent: "flex-end", gap: "0.5rem" }}>
              <button type="button" className="btn btn-secondary" onClick={() => setFormulaPayslip(null)} title="Close Calculation View">
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
