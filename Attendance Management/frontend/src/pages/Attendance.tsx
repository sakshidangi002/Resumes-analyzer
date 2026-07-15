import { useState, useEffect } from "react";
import { useAuth } from "../auth/AuthContext";
import { attendance as api, employees as employeesApi } from "../api/client";
import CustomSelect from "../components/CustomSelect";
import MonthlyAttendanceGrid from "../components/MonthlyAttendanceGrid";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import { SectionLoader } from "../components/LoadingState";
import { useTableControls, SortableHeader, TableToolbar } from "../components/dataTable";

interface AttendanceRow {
  id: number;
  employee_id: number;
  date: string;
  status: string;
  sign_in_time: string | null;
  sign_out_time: string | null;
  total_work_hours: number | null;
  total_break_hours?: number | null;
}

interface AttendanceEventRow {
  id: number;
  event_time: string;
  event_type: string;
  source: string;
}

interface AttendanceDetails {
  employee_id: number;
  employee_name: string;
  date: string;
  events: AttendanceEventRow[];
  sign_in_time: string | null;
  sign_out_time: string | null;
  total_work_hours: number | null;
  total_break_hours: number | null;
  status: string;
  is_late: boolean;
  is_early_exit: boolean;
  check_in_count?: number;
  check_out_count?: number;
  break_in_count?: number;
  break_out_count?: number;
}

// Event types that mean "started working" (arrival / back from break) vs
// "stopped working" (leaving / going on break). The DB stores either the plain
// IN/OUT (manual) or the CHECK_IN/BREAK_IN/BREAK_OUT variants (camera flow).
const IN_LIKE = ["IN", "CHECK_IN", "BREAK_IN"];
const OUT_LIKE = ["OUT", "CHECK_OUT", "BREAK_OUT"];

// Label each event by its POSITION in the day, matching the summary counts:
// the earliest work-start is the "Check-In", the final work-end (only when the
// day is closed) is the "Check-Out"; everything between is a Break In/Out.
// `events` arrive sorted ascending from the backend.
function makeEventLabeler(events: AttendanceEventRow[], hasFinalCheckout: boolean) {
  const ins = events.filter((e) => IN_LIKE.includes(e.event_type));
  const outs = events.filter((e) => OUT_LIKE.includes(e.event_type));
  const firstInId = ins.length ? ins[0].id : null;
  const finalOutId = hasFinalCheckout && outs.length ? outs[outs.length - 1].id : null;
  return (e: AttendanceEventRow): { label: string; kind: "in" | "out" | "other" } => {
    if (IN_LIKE.includes(e.event_type)) {
      return { label: e.id === firstInId ? "Check-In" : "Break In", kind: "in" };
    }
    if (OUT_LIKE.includes(e.event_type)) {
      return { label: e.id === finalOutId ? "Check-Out" : "Break Out", kind: "out" };
    }
    return { label: e.event_type, kind: "other" };
  };
}

interface EmployeeInfo {
  id: number;
  employee_code: string;
  first_name: string;
  last_name: string;
  expected_working_hours: number;
  staff_type?: string | null;
}

function formatLocalDate(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

const Icons = {
  Edit: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
    </svg>
  ),
  Eye: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
      <circle cx="12" cy="12" r="3"></circle>
    </svg>
  ),
  Calendar: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
      <line x1="16" y1="2" x2="16" y2="6"></line>
      <line x1="8" y1="2" x2="8" y2="6"></line>
      <line x1="3" y1="10" x2="21" y2="10"></line>
    </svg>
  )
};


// Monthly Attendance Summary (Feature 3) — reads the existing attendance
// monthly-summary endpoint (no new calculation), shown under the daily view.
type MonthSummary = {
  total_calendar_days: number;
  working_days: number;
  present: number;
  half_day: number;
  leave: number;
  absent: number;
  holiday: number;
  weekly_off: number;
  attendance_percentage: number;
};

function MonthlySummaryCard({ employeeId, month, year }: { employeeId: number; month: number; year: number }) {
  const [summary, setSummary] = useState<MonthSummary | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (!employeeId) return;
    setLoading(true);
    api
      .monthlySummary(employeeId, year, month)
      .then((r) => setSummary(r.data as MonthSummary))
      .catch(() => setSummary(null))
      .finally(() => setLoading(false));
  }, [employeeId, month, year]);

  const monthName = new Date(year, month - 1, 1).toLocaleString("en-US", { month: "long" });
  const rows: { label: string; value: React.ReactNode }[] = summary
    ? [
        { label: "Total Calendar Days", value: summary.total_calendar_days },
        { label: "Working Days", value: summary.working_days },
        { label: "Present", value: summary.present },
        { label: "Half Day", value: summary.half_day },
        { label: "Leave", value: summary.leave },
        { label: "Absent", value: summary.absent },
        { label: "Holiday", value: summary.holiday },
        { label: "Weekly Off", value: summary.weekly_off },
        { label: "Attendance %", value: `${summary.attendance_percentage}%` },
      ]
    : [];

  return (
    <div className="card" style={{ marginTop: "1rem" }}>
      <h3 style={{ margin: "0 0 0.75rem", fontSize: "1.1rem", fontWeight: 700 }}>
        {monthName} {year} — Month Summary
      </h3>
      {loading ? (
        <div className="text-muted">Loading summary…</div>
      ) : !summary ? (
        <div className="text-muted">Summary unavailable.</div>
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
            gap: "0.75rem",
          }}
        >
          {rows.map((r) => (
            <div
              key={r.label}
              style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: 10,
                padding: "0.7rem 0.8rem",
              }}
            >
              <div style={{ fontSize: "0.68rem", textTransform: "uppercase", letterSpacing: "0.04em", opacity: 0.6 }}>
                {r.label}
              </div>
              <div style={{ fontSize: "1.15rem", fontWeight: 800, marginTop: 2 }}>{r.value}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

export default function Attendance() {
  const { hasRole, user } = useAuth();
  const isAdmin = hasRole("Admin");
  const isHR = hasRole("HR");
  const isHrOrAdmin = isAdmin || isHR;

  const now = new Date();
  const todayIso = formatLocalDate(now);

  const [month, setMonth] = useState(now.getMonth() + 1);
  const [year, setYear] = useState(now.getFullYear());
  const [selectedDate, setSelectedDate] = useState(todayIso);
  const [records, setRecords] = useState<AttendanceRow[]>([]);
  const [employees, setEmployees] = useState<EmployeeInfo[]>([]);
  // Non-Employee staff (Housekeeping, Security, …). They are recognised on
  // camera and their attendance is recorded, but it is reported in its own tab
  // and never mixed into employee counts or reports.
  const [staffMembers, setStaffMembers] = useState<EmployeeInfo[]>([]);
  const [rosterTab, setRosterTab] = useState<"employees" | "staff">("employees");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const [editCell, setEditCell] = useState<{
    employee_id: number;
    date: string;
    sign_in_time: string;
    sign_out_time: string;
    break_minutes: string;   // blank = leave the break to the camera
    status: string;
  } | null>(null);

  const [attendanceDialogEmployee, setAttendanceDialogEmployee] = useState<EmployeeInfo | null>(null);
  const [detailsEmployee, setDetailsEmployee] = useState<EmployeeInfo | null>(null);
  const [detailsData, setDetailsData] = useState<AttendanceDetails | null>(null);
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [dialogMonth, setDialogMonth] = useState(now.getMonth() + 1);
  const [dialogYear, setDialogYear] = useState(now.getFullYear());
  const [dialogRecords, setDialogRecords] = useState<AttendanceRow[]>([]);
  const [dialogLoading, setDialogLoading] = useState(false);
  // Employee self-service: the signed-in user's own day summary + timeline.
  const [myDetails, setMyDetails] = useState<AttendanceDetails | null>(null);
  const [myDetailsLoading, setMyDetailsLoading] = useState(false);

  const formatCompactDuration = (hours: number | null | undefined) => {
    if (hours == null) return "-";
    const totalMinutes = Math.round(Number(hours) * 60);
    const h = Math.floor(totalMinutes / 60);
    const m = totalMinutes % 60;
    if (h <= 0) return `${m}m`;
    if (m <= 0) return `${h}h`;
    return `${h}h ${m}m`;
  };

  const formatTime12h = (timeStr: string | null | undefined) => {
    if (!timeStr) return "-";
    const [hh, mm] = timeStr.split(":").map(Number);
    const period = hh >= 12 ? "PM" : "AM";
    const hour12 = hh % 12 || 12;
    return `${hour12.toString().padStart(2, "0")}:${String(mm).padStart(2, "0")} ${period}`;
  };

  const formatEventTime12h = (iso: string) => {
    const d = new Date(iso);
    return d.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", hour12: true });
  };

  const formatStatusLabel = (status: string, totalWorkHours?: number | null) => {
    if (!status) return "-";
    switch (status) {
      case "PRESENT": return totalWorkHours != null ? "Full Day" : "Present";
      case "ABSENT": return "Absent";
      case "HALF_DAY": return "Half Day";
      case "SHORT": return "Short Leave";
      case "WEEKLY_OFF": return "Week Off";
      case "HOLIDAY": return "Holiday";
      case "PAID_LEAVE": return "Paid Leave";
      case "ON_LEAVE": return "On Leave";
      default: return status;
    }
  };

  useEffect(() => {
    setLoading(true);
    setError("");

    let from, to, eid;
    if (isHrOrAdmin) {
      from = selectedDate;
      to = selectedDate;
      eid = undefined;
    } else {
      const daysInMonth = new Date(year, month, 0).getDate();
      from = `${year}-${String(month).padStart(2, "0")}-01`;
      to = `${year}-${String(month).padStart(2, "0")}-${String(daysInMonth).padStart(2, "0")}`;
      eid = user?.employee_id || undefined;
    }

    const listPromise = api.list(from, to, eid);
    const empsPromise = isHrOrAdmin
      ? employeesApi.list({ status: "Active" })
      : Promise.resolve({ data: [] });

    Promise.allSettled([listPromise, empsPromise])
      .then(([attResult, empsResult]) => {
        if (attResult.status === "fulfilled") {
          setRecords(attResult.value.data || []);
        } else {
          console.error("Failed to load attendance records", attResult.reason);
          setError("Failed to load attendance. Please refresh.");
        }
        if (isHrOrAdmin) {
          if (empsResult.status === "fulfilled") {
            const all: EmployeeInfo[] = empsResult.value.data || [];
            const isEmployee = (e: EmployeeInfo) =>
              (e.staff_type ?? "Employee").toLowerCase() === "employee";
            setEmployees(all.filter(isEmployee));
            setStaffMembers(all.filter((e) => !isEmployee(e)));
          }
        }
      })
      .finally(() => setLoading(false));
  }, [selectedDate, month, year, isHrOrAdmin, user?.employee_id]);

  // Per-second update for live tracking
  const [nowTick, setNowTick] = useState(new Date());
  useEffect(() => {
    const timer = setInterval(() => setNowTick(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (!attendanceDialogEmployee) return;
    const daysInDialogMonth = new Date(dialogYear, dialogMonth, 0).getDate();
    const from = `${dialogYear}-${String(dialogMonth).padStart(2, "0")}-01`;
    const to = `${dialogYear}-${String(dialogMonth).padStart(2, "0")}-${String(daysInDialogMonth).padStart(2, "0")}`;
    setDialogLoading(true);
    api.list(from, to, attendanceDialogEmployee.id)
      .then((res) => setDialogRecords(res.data || []))
      .finally(() => setDialogLoading(false));
  }, [attendanceDialogEmployee, dialogMonth, dialogYear]);

  useEffect(() => {
    if (!detailsEmployee) {
      setDetailsData(null);
      return;
    }
    setDetailsLoading(true);
    api.details(detailsEmployee.id, selectedDate)
      .then((res) => setDetailsData(res.data))
      .catch(() => setDetailsData(null))
      .finally(() => setDetailsLoading(false));
  }, [detailsEmployee, selectedDate]);

  // Employee self-service view: load the signed-in user's OWN day summary +
  // timeline for the selected date (own record only; backend enforces this).
  useEffect(() => {
    if (isHrOrAdmin || !user?.employee_id) {
      setMyDetails(null);
      return;
    }
    setMyDetailsLoading(true);
    api.details(user.employee_id, selectedDate)
      .then((res) => setMyDetails(res.data))
      .catch(() => setMyDetails(null))
      .finally(() => setMyDetailsLoading(false));
  }, [isHrOrAdmin, user?.employee_id, selectedDate]);

  const openEdit = (employee_id: number, date: string, dialogOverride?: boolean) => {
    let rec = records.find((r) => r.employee_id === employee_id && r.date === date);
    if (dialogOverride) {
      rec = dialogRecords.find((r) => r.employee_id === employee_id && r.date === date);
    }
    const breakHours = Number(rec?.total_break_hours ?? 0);
    setEditCell({
      employee_id,
      date,
      sign_in_time: rec?.sign_in_time || "",
      sign_out_time: rec?.sign_out_time || "",
      break_minutes: breakHours > 0 ? String(Math.round(breakHours * 60)) : "",
      status: rec?.status || "PRESENT",
    });
  };


  // Auto-calculate status from time-in/time-out vs expected working hours.
  // Break time is time away from work, so it never counts toward hours worked.
  const calcStatusFromTimes = (
    signIn: string,
    signOut: string,
    employeeId: number,
    breakMinutes: string | number = 0
  ): { status: string; hoursWorked: number } => {
    const emp = employees.find(e => e.id === employeeId);
    const expected = emp?.expected_working_hours || 9;
    if (!signIn || !signOut) return { status: "ABSENT", hoursWorked: 0 };
    const [ih, im] = signIn.split(":").map(Number);
    const [oh, om] = signOut.split(":").map(Number);
    const inMins = ih * 60 + im;
    const outMins = oh * 60 + om;
    const breakMins = Math.max(0, Number(breakMinutes) || 0);
    const workedMins = outMins - inMins - breakMins;
    if (workedMins <= 0) return { status: "ABSENT", hoursWorked: 0 };
    const hoursWorked = workedMins / 60;
    const expectedMins = expected * 60;
    if (workedMins >= expectedMins * 0.9) return { status: "PRESENT", hoursWorked };
    if (workedMins >= expectedMins * 0.5) return { status: "HALF_DAY", hoursWorked };
    return { status: "SHORT", hoursWorked };
  };

  const handleSaveCell = (e: React.FormEvent) => {
    e.preventDefault();
    if (!editCell) return;

    // If status is "PRESENT" and we just added times, let backend decide if it's Full/Short/Half
    // unless the user specifically changed the status.
    const normalizedStatus = editCell.status;
    // Blank break = leave it to the camera; a typed value pins it and is
    // deducted from working hours.
    const breakEntered = editCell.break_minutes.trim() !== "";
    api.adminSet({
      employee_id: editCell.employee_id,
      date: editCell.date,
      sign_in_time: editCell.sign_in_time || null,
      sign_out_time: editCell.sign_out_time || null,
      break_hours: breakEntered ? Number(editCell.break_minutes) / 60 : null,
      status: normalizedStatus,
    })
      .then(() => {
        setSuccess("Attendance updated.");
        setEditCell(null);

        const daysInMonth = new Date(year, month, 0).getDate();
        const from = `${year}-${String(month).padStart(2, "0")}-01`;
        const to = `${year}-${String(month).padStart(2, "0")}-${String(daysInMonth).padStart(2, "0")}`;
        api.list(from, to).then(r => setRecords(r.data));

        if (attendanceDialogEmployee) {
          const dialogDays = new Date(dialogYear, dialogMonth, 0).getDate();
          const dFrom = `${dialogYear}-${String(dialogMonth).padStart(2, "0")}-01`;
          const dTo = `${dialogYear}-${String(dialogMonth).padStart(2, "0")}-${String(dialogDays).padStart(2, "0")}`;
          api.list(dFrom, dTo, attendanceDialogEmployee.id).then(res => setDialogRecords(res.data || []));
        }

        setTimeout(() => setSuccess(""), 3000);
      })
      .catch((err) => setError(err.response?.data?.detail || "Failed"));
  };

  const changeDay = (offset: number) => {
    const d = new Date(selectedDate);
    d.setDate(d.getDate() + offset);

    if (offset > 0 && d > now) return;

    const iso = formatLocalDate(d);
    setSelectedDate(iso);
    if (d.getMonth() + 1 !== month) setMonth(d.getMonth() + 1);
    if (d.getFullYear() !== year) setYear(d.getFullYear());
  };

  // Shared renderer: day summary card + labeled event timeline. Used by BOTH the
  // HR "Attendance Details" modal and the employee "My Attendance" view, so the
  // two stay identical. Events are labeled by position (Check-In / Break Out /
  // Break In / Check-Out), consistent with the backend break counts.
  const renderDaySummary = (d: AttendanceDetails | null, loading: boolean) => {
    if (loading) return <SectionLoader size="sm" />;
    const labelFor = makeEventLabeler(d?.events || [], !!d?.sign_out_time);
    const metric = (label: string, value: React.ReactNode, color?: string) => (
      <div style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.07)", borderRadius: "10px", padding: "0.55rem 0.7rem" }}>
        <div style={{ fontSize: "0.68rem", textTransform: "uppercase", letterSpacing: "0.04em", opacity: 0.6 }}>{label}</div>
        <div style={{ fontSize: "1.05rem", fontWeight: 800, marginTop: 2, color: color || "#fff" }}>{value}</div>
      </div>
    );
    return (
      <>
        {/* Summary card */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: "0.6rem", marginBottom: "1.1rem" }}>
          {metric("First Check-In", formatTime12h(d?.sign_in_time), "#22c55e")}
          {metric("Last Check-Out", formatTime12h(d?.sign_out_time), "#f59e0b")}
          {metric("Working Hours", formatCompactDuration(d?.total_work_hours))}
          {metric("Break Time", formatCompactDuration(d?.total_break_hours))}
          {metric("Break Outs", d?.break_out_count ?? 0)}
          {metric("Break Ins", d?.break_in_count ?? 0)}
        </div>

        {/* Detailed timeline */}
        <div style={{ fontSize: "0.7rem", textTransform: "uppercase", letterSpacing: "0.04em", opacity: 0.6, margin: "0 0 0.4rem 2px" }}>Timeline</div>
        <div style={{
          background: "rgba(255,255,255,0.04)",
          border: "1px solid rgba(255,255,255,0.08)",
          borderRadius: "10px",
          padding: "0.4rem 1rem",
          maxHeight: "240px",
          overflowY: "auto",
        }}>
          {(d?.events?.length || 0) === 0 ? (
            <div style={{ opacity: 0.65, textAlign: "center", padding: "0.6rem 0" }}>No attendance events recorded for this date.</div>
          ) : (
            d?.events.map((evt) => {
              const { label, kind } = labelFor(evt);
              const color = kind === "in" ? "#22c55e" : kind === "out" ? "#f59e0b" : "#94a3b8";
              return (
                <div key={evt.id} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0.4rem 0", borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                  <span style={{ fontVariantNumeric: "tabular-nums" }}>{formatEventTime12h(evt.event_time)}</span>
                  <span style={{ fontWeight: 700, color }}>{label}</span>
                </div>
              );
            })
          )}
        </div>

        <div style={{ marginTop: "0.9rem", fontSize: "0.85rem" }}>
          Status: <strong>{formatStatusLabel(d?.status || "ABSENT", d?.total_work_hours)}</strong>
        </div>
      </>
    );
  };

  if (!isHrOrAdmin) {
    const empDateObj = new Date(selectedDate);
    const empDateLabel = empDateObj.toLocaleString("en-IN", { weekday: "long", day: "numeric", month: "short", year: "numeric" });
    const navBtn: React.CSSProperties = { background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "8px", fontSize: "0.85rem", height: "42px", padding: "0 1rem", color: "#fff", cursor: "pointer" };
    return (
      <>
        <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h1 className="page-title">My Attendance</h1>
          <GlobalHeaderControls />
        </div>

        {/* Daily summary + timeline for the selected date */}
        <div className="card" style={{ padding: "1.5rem", marginBottom: "1rem" }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1.25rem", flexWrap: "wrap", gap: "1rem" }}>
            <div>
              <h3 style={{ margin: 0, fontSize: "1.15rem", fontWeight: 700 }}>Daily Summary</h3>
              <div className="text-muted" style={{ fontSize: "0.85rem", marginTop: "2px" }}>{empDateLabel}</div>
            </div>
            <div style={{ display: "flex", alignItems: "flex-end", gap: "0.6rem", flexWrap: "wrap" }}>
              <input
                type="date"
                value={selectedDate}
                min="2026-01-01"
                max={todayIso}
                onChange={(e) => setSelectedDate(e.target.value)}
                className="date-input-white"
                style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.15)", borderRadius: "8px", padding: "0 12px", color: "#fff", fontSize: "0.85rem", width: "150px", height: "42px" }}
              />
              <button type="button" onClick={() => changeDay(-1)} style={navBtn} title="Previous Day">Prev</button>
              <button type="button" onClick={() => setSelectedDate(todayIso)} style={navBtn} title="Today">Today</button>
              <button
                type="button"
                onClick={() => changeDay(1)}
                disabled={selectedDate >= todayIso}
                style={{ ...navBtn, opacity: selectedDate >= todayIso ? 0.3 : 1, cursor: selectedDate >= todayIso ? "not-allowed" : "pointer" }}
                title="Next Day"
              >
                Next
              </button>
            </div>
          </div>
          {renderDaySummary(myDetails, myDetailsLoading)}
        </div>

        <div className="card">
          <MonthlyAttendanceGrid month={month} year={year} setMonth={setMonth} setYear={setYear} records={records} loading={loading} />
        </div>

        {user?.employee_id && (
          <MonthlySummaryCard employeeId={user.employee_id} month={month} year={year} />
        )}
      </>
    );
  }

  const selectedDateObj = new Date(selectedDate);
  const dayName = selectedDateObj.toLocaleString("en-IN", { weekday: "long" });
  const dayNum = selectedDateObj.getDate();
  const monthName = selectedDateObj.toLocaleString("en-IN", { month: "short" });
  const dayLabelFull = `${dayName}, ${dayNum} ${monthName} ${selectedDateObj.getFullYear()}`;

  const selectedIsWeekend = [0, 6].includes(selectedDateObj.getDay());

  const activeRoster = rosterTab === "staff" ? staffMembers : employees;

  const employeeRows = [...activeRoster]
    .sort((a, b) => (Number(a.employee_code) || 0) - (Number(b.employee_code) || 0))
    .map(e => ({ info: e, rec: records.find(r => r.employee_id === e.id && r.date === selectedDate) }));

  const dayCounts = employeeRows.reduce((acc, { rec }) => {
    const s = rec?.status || (selectedIsWeekend ? "WEEKLY_OFF" : "ABSENT");
    if (s === "ABSENT") acc.absent++;
    else if (["PRESENT", "HALF_DAY", "SHORT"].includes(s)) acc.present++;
    return acc;
  }, { present: 0, absent: 0 });

  type AttendanceCombined = (typeof employeeRows)[number];
  const effectiveStatus = (row: AttendanceCombined) => row.rec?.status || (selectedIsWeekend ? "WEEKLY_OFF" : "ABSENT");

  const {
    displayed: displayedEmployeeRows,
    search: attendanceSearch,
    setSearch: setAttendanceSearch,
    sort: attendanceSort,
    toggleSort: toggleAttendanceSort,
    clearAll: clearAttendanceControls,
    hasActiveControls: attendanceHasActive,
  } = useTableControls<AttendanceCombined>({
    rows: employeeRows,
    columns: {
      member: (r) => `${r.info.first_name} ${r.info.last_name}`,
      sign_in_time: (r) => r.rec?.sign_in_time || "",
      sign_out_time: (r) => r.rec?.sign_out_time || "",
      required: (r) => r.info.expected_working_hours || 9,
      working_hours: (r) => Number(r.rec?.total_work_hours ?? 0),
      break_time: (r) => Number(r.rec?.total_break_hours ?? 0),
      status: (r) => effectiveStatus(r),
    },
    searchableText: (r) =>
      `${r.info.employee_code} ${r.info.first_name} ${r.info.last_name} ${effectiveStatus(r)}`,
  });

  return (
    <>
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="page-title">Attendance</h1>
          <div className="page-subtitle">{isAdmin ? "Admin view" : "HR view"} · Daily attendance</div>
        </div>
        <GlobalHeaderControls />
      </div>

      {success && <div className="alert alert-success">{success}</div>}
      {error && !editCell && <div className="alert alert-error">{error}</div>}

      <div className="card" style={{ padding: "1.5rem" }}>
        <div style={{ display: "flex", gap: "0.5rem", marginBottom: "1.25rem" }}>
          {([
            { key: "employees", label: "Employees", count: employees.length },
            { key: "staff", label: "Non-Employee Staff", count: staffMembers.length },
          ] as const).map((tab) => (
            <button
              key={tab.key}
              type="button"
              onClick={() => { setRosterTab(tab.key); setAttendanceSearch(""); }}
              style={{
                background: rosterTab === tab.key ? "rgba(59,130,246,0.18)" : "rgba(255,255,255,0.04)",
                border: `1px solid ${rosterTab === tab.key ? "rgba(59,130,246,0.55)" : "rgba(255,255,255,0.1)"}`,
                color: rosterTab === tab.key ? "#60a5fa" : "rgba(255,255,255,0.65)",
                borderRadius: "8px",
                padding: "0.5rem 1rem",
                fontSize: "0.85rem",
                fontWeight: 600,
                cursor: "pointer",
              }}
            >
              {tab.label} <span style={{ opacity: 0.7 }}>({tab.count})</span>
            </button>
          ))}
        </div>

        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1.5rem", flexWrap: "wrap", gap: "1rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "1.5rem" }}>
            <div>
              <h3 style={{ margin: 0, fontSize: "1.25rem", fontWeight: 700 }}>
                {rosterTab === "staff" ? "Non-Employee Staff Attendance" : "Daily Attendance"}
              </h3>
              <div className="text-muted" style={{ fontSize: "0.85rem", marginTop: "2px" }}>
                {rosterTab === "staff"
                  ? `${dayLabelFull} · not counted in employee reports`
                  : dayLabelFull}
              </div>
            </div>

            <div style={{ display: "flex", gap: "0.75rem" }}>
              <div style={{ background: "rgba(34, 197, 94, 0.15)", color: "#22c55e", padding: "0.4rem 1rem", borderRadius: "99px", fontSize: "0.85rem", fontWeight: 700, display: "flex", alignItems: "center" }}>
                Present <span style={{ marginLeft: "0.5rem" }}>{dayCounts.present}</span>
              </div>
              <div style={{ background: "rgba(239, 68, 68, 0.15)", color: "#ef4444", padding: "0.4rem 1rem", borderRadius: "99px", fontSize: "0.85rem", fontWeight: 700, display: "flex", alignItems: "center" }}>
                Absent <span style={{ marginLeft: "0.5rem" }}>{dayCounts.absent}</span>
              </div>
            </div>
          </div>

          <div style={{ display: "flex", alignItems: "flex-end", gap: "0.75rem" }}>
            <div className="form-group" style={{ marginBottom: 0 }}>
              <input
                type="date"
                value={selectedDate}
                min="2026-01-01"
                max={todayIso}
                onChange={(e) => setSelectedDate(e.target.value)}
                className="date-input-white"
                style={{
                  background: "rgba(255,255,255,0.04)",
                  border: "1px solid rgba(255,255,255,0.15)",
                  borderRadius: "8px",
                  padding: "0 12px",
                  color: "#fff",
                  fontSize: "0.85rem",
                  width: "150px",
                  height: "42px",
                  textAlign: "left",
                }}
              />
            </div>
            <button type="button" className="btn btn-secondary" onClick={() => changeDay(-1)} style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", padding: "0", borderRadius: "8px", fontSize: "0.85rem", width: "130px", height: "42px", textAlign: "center" }} title="Go to Previous Day">Previous Day</button>
            <button type="button" className="btn btn-secondary" onClick={() => setSelectedDate(todayIso)} style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.1)", padding: "0", borderRadius: "8px", fontSize: "0.85rem", width: "100px", height: "42px", textAlign: "center" }} title="Go to Today">Today</button>
            <button
              type="button"
              className="btn btn-secondary"
              onClick={() => changeDay(1)}
              disabled={selectedDate >= todayIso}
              style={{
                background: "rgba(255,255,255,0.06)",
                border: "1px solid rgba(255,255,255,0.1)",
                padding: "0",
                borderRadius: "8px",
                fontSize: "0.85rem",
                width: "130px",
                height: "42px",
                textAlign: "center",
                opacity: selectedDate >= todayIso ? 0.3 : 1,
                cursor: selectedDate >= todayIso ? "not-allowed" : "pointer"
              }}
              title="Go to Next Day"
            >
              Next Day
            </button>
          </div>
        </div>

        <TableToolbar
          search={attendanceSearch}
          onSearchChange={setAttendanceSearch}
          placeholder="Search by name, code, status..."
          showClear={attendanceHasActive}
          onClear={clearAttendanceControls}
          count={{ shown: displayedEmployeeRows.length, total: employeeRows.length }}
        />
        <div className="table-wrap table-wrap--dark">
          <table className="table-modern table-modern--dark">
            <thead>
              <tr>
                <SortableHeader label="Member" columnKey="member" sort={attendanceSort} onToggle={toggleAttendanceSort} style={{ paddingLeft: '1.5rem' }} />
                <SortableHeader label="First In" columnKey="sign_in_time" sort={attendanceSort} onToggle={toggleAttendanceSort} align="center" />
                <SortableHeader label="Last Out" columnKey="sign_out_time" sort={attendanceSort} onToggle={toggleAttendanceSort} align="center" />
                <SortableHeader label="Required Time" columnKey="required" sort={attendanceSort} onToggle={toggleAttendanceSort} align="center" />
                <SortableHeader label="Working Hours" columnKey="working_hours" sort={attendanceSort} onToggle={toggleAttendanceSort} align="center" />
                <SortableHeader label="Break Time" columnKey="break_time" sort={attendanceSort} onToggle={toggleAttendanceSort} align="center" />
                <SortableHeader label="Status" columnKey="status" sort={attendanceSort} onToggle={toggleAttendanceSort} align="center" />
                <SortableHeader label="Actions" columnKey="__actions" sort={attendanceSort} onToggle={toggleAttendanceSort} notSortable align="center" />
              </tr>
            </thead>
            <tbody>
              {loading ? (
                <tr>
                  <td colSpan={8}>
                    <SectionLoader size="md" />
                  </td>
                </tr>
              ) : displayedEmployeeRows.length === 0 ? (
                <tr>
                  <td colSpan={8} style={{ textAlign: 'center', padding: '1.25rem', opacity: 0.65 }}>
                    No attendance rows match your search.
                  </td>
                </tr>
              ) : (
                displayedEmployeeRows.map(({ info, rec }) => {
                  const rawStatus = rec?.status || (selectedIsWeekend ? "WEEKLY_OFF" : "ABSENT");

                  // If DB says ABSENT but employee has actual working hours recorded,
                  // derive the real status from total_work_hours vs expected
                  let s = rawStatus;
                  if (rawStatus === "ABSENT" && rec?.total_work_hours && rec.total_work_hours > 0) {
                    const expected = info.expected_working_hours || 9;
                    if (rec.total_work_hours >= expected * 0.9) s = "PRESENT";
                    else if (rec.total_work_hours >= expected * 0.5) s = "HALF_DAY";
                    else s = "SHORT";
                  }

                  return (
                    <tr key={info.id} style={{ background: "transparent" }}>
                      <td style={{ textAlign: 'left', paddingLeft: '1.5rem' }}>
                        <div style={{ fontWeight: 600, fontSize: "0.95rem" }}>{info.first_name} {info.last_name}</div>
                      </td>
                      <td style={{ opacity: 0.9, textAlign: 'center' }}>
                        {formatTime12h(rec?.sign_in_time)}
                      </td>
                      <td style={{ opacity: 0.9, textAlign: 'center' }}>
                        {(rec?.sign_out_time && rec.sign_out_time !== "00:00:00") ? formatTime12h(rec.sign_out_time) : "-"}
                      </td>
                      <td style={{ opacity: 0.9, textAlign: 'center' }}>
                        {s === "WEEKLY_OFF" || s === "HOLIDAY" ? "-" : `${info.expected_working_hours || 9} Hours`}
                      </td>
                      <td style={{ opacity: 0.9, textAlign: 'center' }}>
                        {(() => {
                          const signedOut = rec?.sign_out_time && rec.sign_out_time !== "00:00:00";
                          // LIVE: an employee who is checked in today and NOT
                          // checked out / on a break (sign_out_time is null only
                          // while actively working) → tick the working hours up
                          // each second: time since first check-in minus recorded
                          // break time. nowTick updates every second.
                          if (rec?.sign_in_time && selectedDate === todayIso && !signedOut) {
                            const [h, m, sec] = rec.sign_in_time.split(':').map(Number);
                            const start = new Date();
                            start.setHours(h, m, sec || 0, 0);
                            if (nowTick > start) {
                              const elapsed = (nowTick.getTime() - start.getTime()) / (1000 * 60 * 60);
                              const live = Math.max(0, elapsed - Number(rec?.total_break_hours || 0));
                              return (
                                <span style={{ color: "rgb(34, 192, 93)", fontWeight: 600, fontSize: "0.95rem" }}>
                                  {formatCompactDuration(live)}
                                </span>
                              );
                            }
                          }
                          // Static snapshot: checked out, on a break, or past days.
                          if (rec?.total_work_hours != null && rec.total_work_hours > 0) {
                            return formatCompactDuration(rec.total_work_hours);
                          }
                          return "-";
                        })()}
                      </td>
                      <td style={{ opacity: 0.9, textAlign: 'center' }}>
                        {formatCompactDuration(rec?.total_break_hours)}
                      </td>
                      <td style={{ borderBottom: "1px solid rgba(255,255,255,0.04)", textAlign: 'center' }}>
                        <span style={{
                          color: s === "PRESENT" ? "rgb(34 192 93)" :
                            s === "ABSENT" ? "#ef4444" :
                              (s === "ON_LEAVE" || s === "PAID_LEAVE" || s === "HALF_DAY" || s === "SHORT") ? "#3b82f6" :
                                "inherit",
                          fontWeight: 500
                        }}>
                          {formatStatusLabel(s, (rec?.sign_out_time && rec.sign_out_time !== "00:00:00") ? rec?.total_work_hours : null)}
                        </span>
                      </td>
                      <td style={{ textAlign: 'center' }}>
                        <div className="actions-stack" style={{ justifyContent: 'center', display: 'flex', gap: '0.35rem' }}>
                          <button className="btn-icon-circle" onClick={() => setDetailsEmployee(info)} title="View Attendance Details">
                            <Icons.Eye />
                          </button>
                          <button className="btn-icon-circle" onClick={() => { setAttendanceDialogEmployee(info); setDialogMonth(month); setDialogYear(year); }} title="View Monthly Attendance History">
                            <Icons.Calendar />
                          </button>
                          {isHR && (
                            <button className="btn-icon-circle" onClick={() => openEdit(info.id, selectedDate)} title="Edit Attendance for this Day">
                              <Icons.Edit />
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      <style>{`
        .btn-icon-circle {
          background: rgba(255,255,255,0.06);
          border: 1px solid rgba(255,255,255,0.1);
          border-radius: 8px;
          width: 32px;
          height: 32px;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          color: #fff;
          transition: all 0.2s;
        }
        .btn-icon-circle:hover {
          background: rgba(255,255,255,0.12);
          border-color: rgba(255,255,255,0.2);
        }
      `}</style>

      {/* Edit Modal */}
      {
        editCell && (
          <div className="modal-backdrop" style={{ zIndex: 1100 }} onClick={() => setEditCell(null)}>
            <div className="modal" onClick={e => e.stopPropagation()} style={{ overflow: "visible" }}>
              <h3>Update Attendance - {editCell.date}</h3>
              {error && <div className="alert alert-error" style={{ marginBottom: "1rem", padding: "0.75rem" }}>{error}</div>}
              <form onSubmit={handleSaveCell}>
                {/* Four fields on one row (the shared .modal-form-grid is 3-up,
                    which pushed Status onto its own line). */}
                <div
                  className="modal-form-grid"
                  style={{ gridTemplateColumns: "repeat(4, minmax(0, 1fr))", alignItems: "start", gap: "0.85rem" }}
                >
                  <div className="form-group"><label>Time In</label><input type="time" value={editCell.sign_in_time} onChange={e => {
                    const newIn = e.target.value;
                    if (editCell.sign_out_time) {
                      const { status } = calcStatusFromTimes(newIn, editCell.sign_out_time, editCell.employee_id, editCell.break_minutes);
                      setEditCell({ ...editCell, sign_in_time: newIn, status });
                    } else {
                      setEditCell({ ...editCell, sign_in_time: newIn });
                    }
                  }} /></div>
                  <div className="form-group">
                    <label>Time Out</label>
                    <input type="time" value={editCell.sign_out_time} onChange={e => {
                      const newOut = e.target.value;
                      if (editCell.sign_in_time && newOut) {
                        const { status } = calcStatusFromTimes(editCell.sign_in_time, newOut, editCell.employee_id, editCell.break_minutes);
                        setEditCell({ ...editCell, sign_out_time: newOut, status });
                      } else {
                        setEditCell({ ...editCell, sign_out_time: newOut });
                      }
                    }} />
                  </div>
                  <div className="form-group">
                    <label>Break Time (minutes)</label>
                    <input
                      type="number"
                      min={0}
                      step={5}
                      placeholder="e.g. 60"
                      value={editCell.break_minutes}
                      onChange={e => {
                        const newBreak = e.target.value;
                        if (editCell.sign_in_time && editCell.sign_out_time) {
                          const { status } = calcStatusFromTimes(editCell.sign_in_time, editCell.sign_out_time, editCell.employee_id, newBreak);
                          setEditCell({ ...editCell, break_minutes: newBreak, status });
                        } else {
                          setEditCell({ ...editCell, break_minutes: newBreak });
                        }
                      }}
                    />
                  </div>
                  <div className="form-group">
                    <label>Status</label>
                    <CustomSelect
                      value={editCell.status}
                      onChange={(val) => setEditCell({ ...editCell, status: val })}
                      options={[
                        { value: "PRESENT", label: "Present" },
                        { value: "HALF_DAY", label: "Half Day" },
                        { value: "SHORT", label: "Short Leave" },
                        { value: "ABSENT", label: "Absent" },
                        { value: "PAID_LEAVE", label: "Paid Leave" },
                        { value: "WEEKLY_OFF", label: "Week Off" },
                        { value: "HOLIDAY", label: "Holiday" },
                      ]}
                    />
                  </div>
                  <div style={{ gridColumn: "1 / -1", fontSize: '0.72rem', color: 'rgba(255,255,255,0.45)', marginTop: '-0.15rem' }}>
                    Break is deducted from working hours — leave it blank to use the break the camera recorded.
                  </div>
                  {editCell.sign_in_time && editCell.sign_out_time && (() => {
                    const { hoursWorked } = calcStatusFromTimes(editCell.sign_in_time, editCell.sign_out_time, editCell.employee_id, editCell.break_minutes);
                    const h = Math.floor(hoursWorked);
                    const m = Math.round((hoursWorked - h) * 60);
                    const brk = Math.max(0, Number(editCell.break_minutes) || 0);
                    return hoursWorked > 0 ? (
                      <div style={{ gridColumn: "1 / -1", fontSize: '0.78rem', color: '#60a5fa', fontWeight: 600 }}>
                        ⏱ {h}h {m}m worked{brk > 0 ? ` (${brk}m break deducted)` : ""} · status auto-set
                      </div>
                    ) : null;
                  })()}
                </div>
                <div className="modal-actions" style={{ display: "flex", justifyContent: "space-between" }}>
                  <button
                    type="button"
                    className="btn"
                    style={{ background: "rgba(239, 68, 68, 0.15)", color: "#ef4444" }}
                    onClick={() => {
                      setEditCell({ ...editCell, status: "", sign_in_time: "", sign_out_time: "", break_minutes: "" });
                      // Provide a slight delay so state updates before submit is simulated, 
                      // or just call adminSet directly. It's safer to just set state, and let user click save, 
                      // or we can invoke handleSaveCell programmatically by simulating the form submit.
                    }}
                    title="Clear Attendance Data"
                  >
                    Clear Data
                  </button>
                  <div style={{ display: "flex", gap: "0.5rem" }}>
                    <button type="submit" className="btn btn-primary" style={{ padding: '0.65rem 1.5rem' }} title="Save Attendance Changes">Save</button>
                    <button type="button" className="btn btn-cancel-alt" style={{ padding: '0.65rem 1.5rem' }} onClick={() => setEditCell(null)} title="Cancel Changes">Cancel</button>

                  </div>
                </div>
              </form>
            </div>
          </div>
        )
      }

      {
        detailsEmployee && (
          <div className="modal-backdrop" onClick={() => setDetailsEmployee(null)} style={{ zIndex: 1100 }}>
            <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 560, width: "95%" }}>
              <h3>Attendance Details</h3>
              <div style={{ marginBottom: "1rem", opacity: 0.85 }}>
                Employee: <strong>{detailsEmployee.first_name} {detailsEmployee.last_name}</strong>
                <div className="text-muted" style={{ fontSize: "0.85rem", marginTop: "4px" }}>{dayLabelFull}</div>
              </div>
              {renderDaySummary(detailsData, detailsLoading)}
              <div className="modal-actions" style={{ marginTop: "1.25rem", justifyContent: "flex-end" }}>
                <button type="button" className="btn btn-secondary" onClick={() => setDetailsEmployee(null)}>Close</button>
              </div>
            </div>
          </div>
        )
      }

      {
        attendanceDialogEmployee && (
          <div className="modal-backdrop" onClick={() => setAttendanceDialogEmployee(null)}>
            <div className="modal" onClick={e => e.stopPropagation()} style={{ maxWidth: 1200, width: "95%", maxHeight: "90vh", overflowY: "auto", position: "relative" }}>
              <button
                type="button"
                className="btn-icon-circle"
                onClick={() => setAttendanceDialogEmployee(null)}
                style={{ position: "absolute", top: "1rem", right: "1rem", zIndex: 10 }}
                title="Close Monthly View"
              >
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
              </button>
              <h3>Monthly View - {attendanceDialogEmployee.first_name} {attendanceDialogEmployee.last_name}</h3>
              <MonthlyAttendanceGrid
                month={dialogMonth}
                year={dialogYear}
                setMonth={setDialogMonth}
                setYear={setDialogYear}
                records={dialogRecords}
                loading={dialogLoading}
                onCellClick={isHR ? (date) => openEdit(attendanceDialogEmployee.id, date, true) : undefined}
              />

            </div>
          </div>
        )}
    </>
  );
}
