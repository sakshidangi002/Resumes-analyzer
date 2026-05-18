import { useState, useEffect } from "react";
import { reports as api } from "../api/client";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import CustomSelect from "../components/CustomSelect";

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
    <>
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="page-title">Reports</h1>
          <div className="page-subtitle">Attendance, leave, headcount, and payroll reports</div>
        </div>
        <GlobalHeaderControls />
      </div>
      <div className="card">
        <h3 style={{ marginTop: 0 }}>Monthly attendance</h3>
        <div style={{ display: 'flex', flexWrap: 'wrap', alignItems: 'flex-end', marginBottom: '1rem', justifyContent: 'space-between' }}>
          <div style={{ display: 'flex', gap: '1rem' }}>
            <div className="form-group" style={{ marginBottom: 0, width: "130px" }}>
              <label>Month</label>
              <CustomSelect
                value={String(month)}
                onChange={(val) => setMonth(Number(val))}
                options={[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map((m) => ({
                  value: String(m),
                  label: new Date(2000, m - 1).toLocaleString("default", { month: "long" })
                }))}
              />
            </div>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <label>Year</label>
              <CustomSelect
                value={String(year)}
                onChange={(val) => setYear(Number(val))}
                style={{ width: "120px" }}
                options={[2026].map(y => ({ value: String(y), label: String(y) }))}
              />
            </div>
          </div>
          <div style={{ alignSelf: "flex-end" }}>
            <button type="button" className="btn btn-primary btn-uniform" onClick={loadAttendance} disabled={loading}>Load</button>
            <button type="button" className="btn btn-secondary btn-uniform" onClick={exportExcel} style={{ marginLeft: "0.5rem", backgroundColor: "var(--brand-500)" }}>Export Excel</button>
          </div>
        </div>
        {attSummary && (
          <div className="table-wrap table-wrap--dark">
            <table className="table-modern table-modern--dark">
              <thead>
                <tr>
                  <th style={{ textAlign: 'left', paddingLeft: '1.5rem' }}>Code</th>
                  <th style={{ textAlign: 'left' }}>Name</th>
                  <th style={{ textAlign: 'center' }}>Present</th>
                  <th style={{ textAlign: 'center' }}>Absent</th>
                  <th style={{ textAlign: 'center' }}>Half day</th>
                  <th style={{ textAlign: 'center' }}>Week Off</th>
                  <th style={{ textAlign: 'center' }}>Working Days</th>
                  <th style={{ textAlign: 'center' }}>Total Leaves</th>
                  <th style={{ textAlign: 'center' }}>Total Attendance</th>
                </tr>
              </thead>
              <tbody>
                {attSummary.summary.map((s) => (
                  <tr key={s.employee_id}>
                    <td style={{ fontWeight: 600, color: "var(--brand-400)", textAlign: 'left', paddingLeft: '1.5rem' }}>{s.employee_code || "-"}</td>
                    <td style={{ textAlign: 'left' }}>{s.first_name} {s.last_name}</td>
                    <td style={{ textAlign: 'center' }}>
                      <div style={{ display: 'flex', justifyContent: 'center' }}>{s.present}</div>
                    </td>
                    <td style={{ textAlign: 'center' }}>
                      <div style={{ display: 'flex', justifyContent: 'center' }}>{s.absent}</div>
                    </td>
                    <td style={{ textAlign: 'center' }}>
                      <div style={{ display: 'flex', justifyContent: 'center' }}>{s.half_day}</div>
                    </td>
                    <td style={{ textAlign: 'center' }}>
                      <div style={{ display: 'flex', justifyContent: 'center' }}>{s.week_off}</div>
                    </td>
                    <td style={{ textAlign: 'center' }}>
                      <div style={{ display: 'flex', justifyContent: 'center' }}>{s.working_days}</div>
                    </td>
                    <td style={{ textAlign: 'center' }}>
                      <div style={{ display: 'flex', justifyContent: 'center' }}>{s.total_leaves}</div>
                    </td>
                    <td style={{ fontWeight: 800, color: "#fff", textAlign: 'center' }}>
                      <div style={{ display: 'flex', justifyContent: 'center' }}>{s.total_attendance}</div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
      <div className="card">
        <h3 style={{ marginTop: 0 }}>Department headcount</h3>
        <div className="table-wrap table-wrap--dark">
          <table className="table-modern table-modern--dark" style={{ width: '100%' }}>
            <thead>
              <tr>
                <th style={{ width: '50%', textAlign: 'left', paddingLeft: '1.5rem' }}>Department</th>
                <th style={{ width: '50%', textAlign: 'right', paddingRight: '1.5rem' }}>Count</th>
              </tr>
            </thead>
            <tbody>
              {headcount.map((h) => (
                <tr key={h.department_id}>
                  <td style={{ textAlign: 'left', paddingLeft: '1.5rem' }}>{h.department_name || "N/A"}</td>
                  <td style={{ textAlign: 'right', paddingRight: '1.5rem' }}>{h.count}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}
