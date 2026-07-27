/**
 * Displays salary calculation with formula and actual values.
 * Uses 30-day month: Per Day = Gross ÷ 30, Salary Payable = Per Day × Paid Days.
 */

export interface ComponentBreakdown {
  basic?: number;
  hra?: number;
  medical?: number;
  travelling?: number;
  miscellaneous?: number;
  allowances?: number;
  deductions?: number;
  paid_days?: number;
  lop_days?: number;
  lop_dates?: string[];
  short_leaves_used?: number;
  per_hour_salary?: number;
  expected_hours?: number;
}

export interface PayslipForFormula {
  gross_salary: number;
  total_earnings: number;
  total_deductions: number;
  net_salary: number;
  paid_days: number;
  lop_days: number;
  component_breakdown?: string | null;
}

const DAYS_IN_MONTH = 30;
const fmt = (n: number) => Number(n).toFixed(2);
const currency = (n: number) => `₹ ${fmt(n)}`;

export default function SalaryFormulaView({
  payslip,
  title,
}: {
  payslip: PayslipForFormula;
  title?: string;
}) {
  const gross = Number(payslip.gross_salary);
  const paidDays = Number(payslip.paid_days);
  const lopDays = Number(payslip.lop_days);
  const totalEarnings = Number(payslip.total_earnings);
  const totalDeductions = Number(payslip.total_deductions);
  const net = Number(payslip.net_salary);
  const perDay = gross / DAYS_IN_MONTH;
  let breakdown: ComponentBreakdown | null = null;
  if (payslip.component_breakdown) {
    try {
      breakdown = JSON.parse(payslip.component_breakdown) as ComponentBreakdown;
    } catch {
      breakdown = null;
    }
  }

  const med = breakdown?.medical ?? 0;
  const tr = breakdown?.travelling ?? 0;
  const misc = breakdown?.miscellaneous ?? 0;

  return (
    <div className="salary-formula-view" style={{ fontSize: "0.9rem" }}>
      {title && (
        <h4 style={{ marginTop: 0, marginBottom: "0.75rem", borderBottom: "1px solid rgba(255, 255, 255, 0.10)", paddingBottom: "0.5rem" }}>
          {title}
        </h4>
      )}
      <section style={{ marginBottom: "1rem" }}>
        <div style={{ fontWeight: 600, marginBottom: "0.5rem" }}>Salary components (monthly)</div>
        {breakdown ? (
          <table style={{ width: "100%", borderCollapse: "collapse", marginBottom: "0.5rem" }}>
            <tbody>
              <tr><td style={{ padding: "2px 8px 2px 0" }}>Basic</td><td style={{ textAlign: "right" }}>{currency(breakdown.basic ?? 0)}</td></tr>
              <tr><td style={{ padding: "2px 8px 2px 0" }}>HRA</td><td style={{ textAlign: "right" }}>{currency(breakdown.hra ?? 0)}</td></tr>
              <tr><td style={{ padding: "2px 8px 2px 0" }}>Medical</td><td style={{ textAlign: "right" }}>{currency(med)}</td></tr>
              <tr><td style={{ padding: "2px 8px 2px 0" }}>Travelling</td><td style={{ textAlign: "right" }}>{currency(tr)}</td></tr>
              <tr><td style={{ padding: "2px 8px 2px 0" }}>Miscellaneous</td><td style={{ textAlign: "right" }}>{currency(misc)}</td></tr>
              <tr><td style={{ padding: "2px 8px 2px 0" }}>Allowances</td><td style={{ textAlign: "right" }}>{currency(breakdown.allowances ?? 0)}</td></tr>
            </tbody>
          </table>
        ) : null}
        <div style={{ padding: "6px 0", borderTop: "1px solid rgba(255, 255, 255, 0.10)" }}>
          <strong>Gross Salary</strong>
          {breakdown ? (
            <span style={{ color: "rgba(255, 255, 255, 0.72)", fontWeight: "normal" }}>
              {" "}
              = Basic + HRA + Medical + Travelling + Miscellaneous + Allowances ={" "}
              {currency(breakdown.basic ?? 0)} + {currency(breakdown.hra ?? 0)} + {currency(med)} + {currency(tr)} + {currency(misc)} + {currency(breakdown.allowances ?? 0)}
            </span>
          ) : null}
          <span style={{ float: "right", fontWeight: 600 }}>{currency(gross)}</span>
        </div>
      </section>

      <section style={{ marginBottom: "1rem" }}>
        <div style={{ fontWeight: 600, marginBottom: "0.5rem" }}>Calculation (30-day month)</div>
        <table style={{ width: "100%", borderCollapse: "collapse" }}>
          <tbody>
            <tr>
              <td style={{ padding: "4px 8px 4px 0", verticalAlign: "top" }}>Days in month</td>
              <td style={{ textAlign: "right", padding: "4px 0" }}>30</td>
              <td style={{ padding: "4px 0 4px 8px", color: "rgba(255, 255, 255, 0.62)", fontSize: "0.85rem" }}>(fixed)</td>
            </tr>
            <tr>
              <td style={{ padding: "4px 8px 4px 0" }}>Paid days</td>
              <td style={{ textAlign: "right", padding: "4px 0" }}>{fmt(paidDays)}</td>
              <td style={{ padding: "4px 0 4px 8px", color: "rgba(255, 255, 255, 0.62)", fontSize: "0.85rem" }}>(from attendance)</td>
            </tr>
            <tr>
              <td style={{ padding: "4px 8px 4px 0" }}>LOP days</td>
              <td style={{ textAlign: "right", padding: "4px 0" }}>{fmt(lopDays)}</td>
              <td style={{ padding: "4px 0 4px 8px", color: "rgba(255, 255, 255, 0.62)", fontSize: "0.85rem" }}>(loss of pay)</td>
            </tr>
            {breakdown?.short_leaves_used != null && breakdown.short_leaves_used > 0 && (
            <tr>
              <td style={{ padding: "4px 8px 4px 0" }}>Short leaves used</td>
              <td style={{ textAlign: "right", padding: "4px 0" }}>{breakdown.short_leaves_used}</td>
              <td style={{ padding: "4px 0 4px 8px", color: "rgba(255, 255, 255, 0.62)", fontSize: "0.85rem" }}>(First 2 are free)</td>
            </tr>
            )}
            {breakdown?.lop_dates && breakdown.lop_dates.length > 0 && (
              <tr>
                <td colSpan={3} style={{ padding: "0 0 8px 0" }}>
                  <div style={{ fontSize: "0.75rem", color: "rgba(239, 68, 68, 0.8)", background: "rgba(239, 68, 68, 0.08)", padding: "6px 10px", borderRadius: "6px", border: "1px solid rgba(239, 68, 68, 0.15)" }}>
                    <strong>LOP Dates:</strong> {breakdown.lop_dates.map(d => d.split('-').reverse().join('/')).join(', ')}
                  </div>
                </td>
              </tr>
            )}
            <tr style={{ borderTop: "1px solid rgba(255, 255, 255, 0.10)" }}>
              <td style={{ padding: "6px 8px 6px 0" }}><strong>Per day salary</strong></td>
              <td style={{ textAlign: "right", padding: "6px 0" }}><strong>{currency(perDay)}</strong></td>
              <td style={{ padding: "6px 0 6px 8px", color: "rgba(255, 255, 255, 0.62)", fontSize: "0.85rem" }}>Gross ÷ 30 = {currency(gross)} ÷ 30</td>
            </tr>
            {breakdown?.per_hour_salary != null && (
              <tr>
                <td style={{ padding: "4px 8px 4px 0" }}><strong>Per hour salary</strong></td>
                <td style={{ textAlign: "right", padding: "4px 0" }}><strong>{currency(breakdown.per_hour_salary)}</strong></td>
                <td style={{ padding: "4px 0 4px 8px", color: "rgba(255, 255, 255, 0.62)", fontSize: "0.85rem" }}>Per day ÷ {breakdown.expected_hours || 9} hrs = {currency(perDay)} ÷ {breakdown.expected_hours || 9}</td>
              </tr>
            )}
            <tr>
              <td style={{ padding: "4px 8px 4px 0" }}><strong>Total earnings</strong></td>
              <td style={{ textAlign: "right", padding: "4px 0" }}><strong>{currency(totalEarnings)}</strong></td>
              <td style={{ padding: "4px 0 4px 8px", color: "rgba(255, 255, 255, 0.62)", fontSize: "0.85rem" }}>Per day × Paid days = {currency(perDay)} × {fmt(paidDays)}</td>
            </tr>
            <tr>
              <td style={{ padding: "4px 8px 4px 0" }}>Deductions (proportional)</td>
              <td style={{ textAlign: "right", padding: "4px 0" }}>{currency(totalDeductions)}</td>
              <td style={{ padding: "4px 0 4px 8px", color: "rgba(255, 255, 255, 0.62)", fontSize: "0.85rem" }}>(Deductions ÷ 30) × Paid days</td>
            </tr>
            <tr style={{ borderTop: "2px solid rgba(255, 255, 255, 0.18)" }}>
              <td style={{ padding: "8px 8px 8px 0" }}><strong>Net salary</strong></td>
              <td style={{ textAlign: "right", padding: "8px 0", fontWeight: 700 }}>{currency(net)}</td>
              <td style={{ padding: "8px 0 8px 8px", color: "rgba(255, 255, 255, 0.62)", fontSize: "0.85rem" }}>Total earnings − Total deductions</td>
            </tr>
          </tbody>
        </table>
      </section>

      <div
        style={{
          marginTop: "0.75rem",
          padding: "8px 10px",
          background: "rgba(255, 255, 255, 0.06)",
          borderRadius: 10,
          fontSize: "0.8rem",
          color: "rgba(255, 255, 255, 0.78)",
          border: "1px solid rgba(255, 255, 255, 0.10)",
        }}
      >
        <strong>Formula:</strong> Salary Payable = (Monthly Gross ÷ 30) × Paid Days  &nbsp;|&nbsp;  Net = Salary Payable − Deductions (proportional)
      </div>
    </div>
  );
}
