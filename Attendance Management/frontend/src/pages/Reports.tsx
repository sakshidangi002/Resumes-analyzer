import { useState, useEffect } from "react";
import { reports as api } from "../api/client";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import CustomSelect from "../components/CustomSelect";

const Icons = {
  Bars: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="20" x2="18" y2="10" /><line x1="12" y1="20" x2="12" y2="4" /><line x1="6" y1="20" x2="6" y2="14" />
    </svg>
  ),
  Building: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 21V8l9-5 9 5v13" /><line x1="3" y1="21" x2="21" y2="21" /><rect x="9" y="13" width="6" height="8" />
    </svg>
  ),
  Refresh: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20.5 12a8.5 8.5 0 1 1-2.5-6" /><polyline points="20.5 4 20.5 9.5 15 9.5" />
    </svg>
  ),
  Download: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" /><polyline points="7 10 12 15 17 10" /><line x1="12" y1="15" x2="12" y2="3" />
    </svg>
  ),
};

export default function Reports() {
  const [month, setMonth] = useState(new Date().getMonth() + 1);
  const [year, setYear] = useState(new Date().getFullYear());
  const [attSummary, setAttSummary] = useState<{ summary: Array<{ employee_id: number; employee_code: string; first_name: string; last_name: string; present: number; absent: number; half_day: number; on_leave: number; week_off: number; total_attendance: number; total_leaves: number; working_days: number }> } | null>(null);
  const [headcount, setHeadcount] = useState<Array<{ department_id: number; department_name: string; count: number }>>([]);
  const [loading, setLoading] = useState(false);

  const loadAttendance = () => {
    setLoading(true);
    api.monthlyAttendance(month, year).then((r) => setAttSummary(r.data)).catch(() => setAttSummary(null)).finally(() => setLoading(false));
  };

  useEffect(() => {
    api.headcount().then((r) => setHeadcount(r.data.headcount || [])).catch(() => setHeadcount([]));
  }, []);

  const exportExcel = () => {
    api.exportAttendanceExcel(month, year).then((r) => {
      const url = URL.createObjectURL(r.data as Blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "attendance_" + year + "_" + month + ".xlsx";
      a.click();
      URL.revokeObjectURL(url);
    });
  };

  return (
    <div className="eds">
      <header className="eds-topbar">
        <div>
          <h1 className="eds-title">Reports</h1>
          <p className="eds-subtitle">Attendance, leave, headcount, and payroll reports</p>
        </div>
        <GlobalHeaderControls />
      </header>

      <div className="eds-page">
        <section className="eds-card">
          <div className="eds-card-head">
            <span className="eds-chip eds-chip--sky"><Icons.Bars /></span>
            <div className="eds-card-titles">
              <h2 className="eds-card-title">Monthly attendance</h2>
            </div>
          </div>
          <div className="eds-card-body">
            <div style={{ display: "flex", alignItems: "flex-end", gap: 14, flexWrap: "wrap" }}>
              <div className="eds-fieldset">
                <span className="eds-fieldset-label">Month</span>
                <CustomSelect
                  className="eds-cselect eds-cselect--month"
                  value={String(month)}
                  onChange={(val) => setMonth(Number(val))}
                  options={[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map((m) => ({
                    value: String(m),
                    label: new Date(2000, m - 1).toLocaleString("default", { month: "long" })
                  }))}
                />
              </div>
              <div className="eds-fieldset">
                <span className="eds-fieldset-label">Year</span>
                <CustomSelect
                  className="eds-cselect eds-cselect--year"
                  value={String(year)}
                  onChange={(val) => setYear(Number(val))}
                  options={[2026].map(y => ({ value: String(y), label: String(y) }))}
                />
              </div>
              <div className="eds-controls-end">
                <button type="button" className="eds-action" onClick={loadAttendance} disabled={loading}>
                  <Icons.Refresh />
                  Load
                </button>
                <button type="button" className="eds-action eds-action--go" onClick={exportExcel}>
                  <Icons.Download />
                  Export Excel
                </button>
              </div>
            </div>
          </div>

          {attSummary && (
            <div className="eds-table-wrap">
              <table className="eds-table eds-table--auto">
                <thead>
                  <tr>
                    <th>Code</th>
                    <th>Name</th>
                    <th className="is-actions">Present</th>
                    <th className="is-actions">Absent</th>
                    <th className="is-actions">Half day</th>
                    <th className="is-actions">Week Off</th>
                    <th className="is-actions">Working Days</th>
                    <th className="is-actions">Total Leaves</th>
                    <th className="is-actions">Total Attendance</th>
                  </tr>
                </thead>
                <tbody>
                  {attSummary.summary.map((s) => (
                    <tr key={s.employee_id}>
                      <td><span className="eds-code eds-code--sky">{s.employee_code || "-"}</span></td>
                      <td className="eds-cell-strong">{s.first_name} {s.last_name}</td>
                      <td className="eds-money">{s.present}</td>
                      <td className={`eds-money${s.absent > 0 ? " eds-money--minus" : " eds-money--zero"}`}>{s.absent}</td>
                      <td className={`eds-money${s.half_day > 0 ? " eds-money--lop" : " eds-money--zero"}`}>{s.half_day}</td>
                      <td className="eds-money eds-money--zero">{s.week_off}</td>
                      <td className="eds-money">{s.working_days}</td>
                      <td className="eds-money">{s.total_leaves}</td>
                      <td className="eds-money eds-money--net">{s.total_attendance}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>

        <section className="eds-card">
          <div className="eds-card-head">
            <span className="eds-chip eds-chip--violet"><Icons.Building /></span>
            <div className="eds-card-titles">
              <h2 className="eds-card-title">Department headcount</h2>
            </div>
          </div>
          {headcount.length === 0 ? (
            <div className="eds-empty--card">
              <span className="eds-empty-tile"><Icons.Building /></span>
              <span>No headcount data available.</span>
            </div>
          ) : (
            <div className="eds-table-wrap">
              <table className="eds-table eds-table--auto">
                <thead>
                  <tr>
                    <th>Department</th>
                    <th className="is-actions">Count</th>
                  </tr>
                </thead>
                <tbody>
                  {headcount.map((h) => (
                    <tr key={h.department_id}>
                      <td className="eds-cell-strong">{h.department_name || "N/A"}</td>
                      <td className="eds-money eds-money--net">{h.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
