/** Shared monthly calendar for attendance (employee self-serve or HR profile view). */
import { useMemo } from "react";
import { SectionLoader } from "./LoadingState";
import CustomSelect from "./CustomSelect";

export interface AttendanceRecordLite {
  date: string;
  status: string;
  sign_in_time: string | null;
  sign_out_time: string | null;
  total_work_hours: number | null;
}

function formatLocalDate(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function formatHours(decimalHours: number): string {
  const totalMins = Math.round(decimalHours * 60);
  const h = Math.floor(totalMins / 60);
  const m = totalMins % 60;
  return `${h}:${m.toString().padStart(2, "0")} Hrs`;
}


/* The frame shows check-in and check-out as directional glyphs rather than the
   source's ⏱ emoji: arrival points down into the day, departure points out. */
const ArrowIn = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
    <line x1="12" y1="4" x2="12" y2="19" />
    <polyline points="6 13 12 19 18 13" />
  </svg>
);

const ArrowOut = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round">
    <line x1="12" y1="20" x2="12" y2="5" />
    <polyline points="6 11 12 5 18 11" />
  </svg>
);

export default function MonthlyAttendanceGrid({
  month,
  year,
  setMonth,
  setYear,
  records,
  loading,
  onCellClick,
}: {
  month: number;
  year: number;
  setMonth: (n: number) => void;
  setYear: (n: number) => void;
  records: AttendanceRecordLite[];
  loading: boolean;
  onCellClick?: (date: string) => void;
}) {
  const recordByDate = new Map<string, AttendanceRecordLite>();
  records.forEach((r) => recordByDate.set(r.date, r));

  const firstOfMonth = new Date(year, month - 1, 1);
  const mondayIndex = (firstOfMonth.getDay() + 6) % 7;
  const start = new Date(firstOfMonth);
  start.setDate(firstOfMonth.getDate() - mondayIndex);

  const weeks: Date[][] = [];
  for (let w = 0; w < 6; w++) {
    const row: Date[] = [];
    for (let d = 0; d < 7; d++) {
      const dt = new Date(start);
      dt.setDate(start.getDate() + w * 7 + d);
      row.push(dt);
    }
    weeks.push(row);
  }

  const weekdayLabels = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];
  const monthName = new Date(year, month - 1, 1).toLocaleString("default", { month: "long" });

  const shiftEmpMonth = (delta: number) => {
    const d = new Date(year, month - 1 + delta, 1);
    setMonth(d.getMonth() + 1);
    setYear(d.getFullYear());
  };

  const todayRef = new Date();
  const currentYear = todayRef.getFullYear();
  const minYear = 2026; // Company started in 2026, no data before this
  
  const canGoNextMonth =
    year < currentYear ||
    (year === currentYear && month < todayRef.getMonth() + 1);

  const canGoPrevMonth = 
    year > minYear || 
    (year === minYear && month > 1);

  const barClassForStatus = (status: string) => {
    if (status === "ABSENT") return "emp-cal-bar--absent";
    if (status === "ON_LEAVE" || status === "PAID_LEAVE") return "emp-cal-bar--leave";
    if (status === "HALF_DAY") return "emp-cal-bar--half";
    if (status === "SHORT") return "emp-cal-bar--half"; // Same color as half day
    if (status === "PRESENT") return "emp-cal-bar--present";
    if (status === "WEEKLY_OFF") return "emp-cal-bar--wo";
    if (status === "HOLIDAY") return "emp-cal-bar--holiday";
    return "emp-cal-bar--empty";
  };

  const badgeClassForStatus = (status: string) => {
    if (status === "ON_LEAVE" || status === "PAID_LEAVE") return "emp-cal-badge--leave";
    if (status === "HALF_DAY" || status === "SHORT") return "emp-cal-badge--half";
    if (status === "ABSENT") return "emp-cal-badge--absent";
    if (status === "WEEKLY_OFF") return "emp-cal-badge--wo";
    if (status === "HOLIDAY") return "emp-cal-badge--holiday";
    return "emp-cal-badge--wo";
  };

  // Month summary, from the records this component already holds. Graded against
  // weekdays in the month so the rate cannot exceed 100%.
  const summary = useMemo(() => {
    const inMonth = records.filter((r: AttendanceRecordLite) => {
      const d = new Date(r.date + "T00:00:00");
      return d.getMonth() === month - 1 && d.getFullYear() === year;
    });
    const count = (...st: string[]) => inMonth.filter((r: AttendanceRecordLite) => st.includes(r.status)).length;
    const present = count("PRESENT");
    const partial = count("HALF_DAY", "SHORT");
    const leave = count("ON_LEAVE", "PAID_LEAVE");
    const absent = count("ABSENT");
    const holiday = count("HOLIDAY");

    const daysInMonth = new Date(year, month, 0).getDate();
    let workingDays = 0;
    for (let d = 1; d <= daysInMonth; d++) {
      const dow = new Date(year, month - 1, d).getDay();
      if (dow !== 0 && dow !== 6) workingDays++;
    }
    const credited = present + partial * 0.5;
    return {
      workingDays,
      rate: workingDays > 0 ? Math.min(100, Math.round((credited / workingDays) * 100)) : 0,
      rows: [
        { label: "Present", count: present, swatch: "emp-cal-bar--present" },
        { label: "Half / short", count: partial, swatch: "emp-cal-bar--half" },
        { label: "On leave", count: leave, swatch: "emp-cal-bar--leave" },
        { label: "Absent", count: absent, swatch: "emp-cal-bar--absent" },
        { label: "Holiday", count: holiday, swatch: "emp-cal-bar--holiday" },
      ],
    };
  }, [records, month, year]);


  return (
    <>
      <div className="emp-cal-toolbar">
        <div className="emp-cal-title-block">
          <h2 className="emp-cal-month-title">
            {monthName} {year}
          </h2>
          <div className="emp-cal-nav" aria-label="Change month">
            <button 
              type="button" 
              className="emp-cal-nav-btn" 
              onClick={() => shiftEmpMonth(-1)} 
              disabled={!canGoPrevMonth}
              title={canGoPrevMonth ? "Previous month" : "No attendance data before this"}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="15 18 9 12 15 6"></polyline></svg>
            </button>
            <button
              type="button"
              className="emp-cal-nav-btn"
              onClick={() => shiftEmpMonth(1)}
              disabled={!canGoNextMonth}
              title={canGoNextMonth ? "Next month" : "Cannot go beyond current month"}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="9 18 15 12 9 6"></polyline></svg>
            </button>
          </div>
        </div>
        <div className="emp-cal-controls">
          <div className="form-group">
            <label>Month</label>
            <CustomSelect
              value={String(month)}
              onChange={(val) => setMonth(Number(val))}
              options={[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map((m) => ({
                value: String(m),
                label: new Date(2000, m - 1, 1).toLocaleString("default", { month: "long" })
              }))}
              style={{ width: "130px" }}
            />
          </div>
          <div className="form-group">
            <label>Year</label>
            <CustomSelect
              value={String(year)}
              onChange={(val) => setYear(Number(val))}
              options={Array.from({ length: currentYear - minYear + 1 }, (_, i) => minYear + i).map(y => ({
                value: String(y),
                label: String(y)
              }))}
            />
          </div>
        </div>
      </div>
      {!loading && (
        <>
          <div className="emp-cal-summary">
            {summary.rows.map((r) => (
              <div className="emp-cal-sum-row" key={r.label}>
                <span className={`emp-cal-legend-swatch ${r.swatch}`} />
                <span className="emp-cal-sum-label">{r.label}</span>
                <span className="emp-cal-sum-count">{r.count}</span>
              </div>
            ))}
          </div>
          <div className="emp-cal-rate">
            <span className="emp-cal-rate-label">Attendance rate</span>
            <span className="emp-cal-rate-value">
              {summary.workingDays > 0 ? `${summary.rate}%` : "—"}
            </span>
          </div>
        </>
      )}
      {loading ? (
        <SectionLoader size="md" />
      ) : (
        <div className="emp-cal-wrap">
          <table className="emp-cal-table">
            <thead>
              <tr>
                {weekdayLabels.map((label) => (
                  <th key={label} style={{ textAlign: "center" }}>{label}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {weeks.map((week, wIdx) => (
                <tr key={wIdx}>
                  {week.map((dt, dIdx) => {
                    const yyyyMmDd = formatLocalDate(dt);
                    const inMonth = dt.getMonth() === month - 1;
                    const isWeekend = dt.getDay() === 0 || dt.getDay() === 6;
                    const rec = recordByDate.get(yyyyMmDd);
                    const status = rec?.status || (isWeekend ? "WEEKLY_OFF" : "");
                    const hoursText =
                      rec && rec.total_work_hours != null ? formatHours(rec.total_work_hours) : "";
                    const timeIn = rec?.sign_in_time || "";
                    const timeOut = rec?.sign_out_time || "";

                    const labelText =
                      status === "ON_LEAVE"
                        ? "On Leave"
                        : status === "PAID_LEAVE"
                        ? "Paid Leave"
                        : status === "HALF_DAY"
                        ? "Half Day"
                        : status === "SHORT"
                        ? "Short Leave"
                        : status === "ABSENT"
                        ? "Absent"
                        : status === "WEEKLY_OFF"
                        ? "WO"
                        : status === "HOLIDAY"
                        ? "Holiday"
                        : "";

                    return (
                      <td 
                        key={dIdx} 
                        className={`emp-cal-cell ${inMonth ? "" : "emp-cal-cell--dim"} ${onCellClick && inMonth ? "emp-cal-cell--clickable" : ""}`}
                        onClick={() => onCellClick && inMonth && onCellClick(yyyyMmDd)}
                      >
                        <div className="emp-cal-day-num">{dt.getDate()}</div>
                        <div className={`emp-cal-bar ${barClassForStatus(status)}`} />
                        {isWeekend && !rec && <span className="emp-cal-badge emp-cal-badge--wo">WO</span>}
                        {hoursText && <div className="emp-cal-hours">{hoursText}</div>}
                        {(timeIn || timeOut) && (
                          <div className="emp-cal-times">
                            {timeIn && (
                              <div className="emp-cal-time-row emp-cal-time-row--in">
                                <span className="emp-cal-time-icon" aria-hidden>
                                  <ArrowIn />
                                </span>
                                <span>{timeIn}</span>
                              </div>
                            )}
                            {timeOut && (
                              <div className="emp-cal-time-row emp-cal-time-row--out">
                                <span className="emp-cal-time-icon" aria-hidden>
                                  <ArrowOut />
                                </span>
                                <span>{timeOut}</span>
                              </div>
                            )}
                          </div>
                        )}
                        {labelText && !(isWeekend && !rec) && (
                          <span className={`emp-cal-badge ${badgeClassForStatus(status)}`}>{labelText}</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {!loading && records.length === 0 && (
        <p className="text-muted" style={{ marginTop: "0.75rem" }}>
          No attendance for this month.
        </p>
      )}
      {!loading && (
        <div className="emp-cal-legend" aria-hidden>
          <span className="emp-cal-legend-item">
            <span className="emp-cal-legend-swatch emp-cal-bar--present" /> Present
          </span>
          <span className="emp-cal-legend-item">
            <span className="emp-cal-legend-swatch emp-cal-bar--leave" /> Leave
          </span>
          <span className="emp-cal-legend-item">
            <span className="emp-cal-legend-swatch emp-cal-bar--wo" /> Weekend
          </span>
          <span className="emp-cal-legend-item">
            <span className="emp-cal-legend-swatch emp-cal-bar--half" /> Half day
          </span>
          <span className="emp-cal-legend-item">
            <span className="emp-cal-legend-swatch emp-cal-bar--absent" /> Absent
          </span>
        </div>
      )}
    </>
  );
}
