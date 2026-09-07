import { useState, useEffect, useMemo, useRef } from "react";
import { useAuth } from "../auth/AuthContext";
import { attendance as api, employees as employeesApi } from "../api/client";
import CustomSelect from "../components/CustomSelect";
import MonthlyAttendanceGrid from "../components/MonthlyAttendanceGrid";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import { dailyTarget } from "../utils/workingHours";
import { SectionLoader } from "../components/LoadingState";
import { useTableControls } from "../components/dataTable";
import type { SortState, SortDirection } from "../components/dataTable";

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

/* 24-box strokes, round caps, currentColor. Rendered size comes from the
   control that holds them (.eds-iconbtn 15px, .eds-sort 9px, and so on). */
const Icons = {
  Edit: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 4H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-5"></path>
      <path d="M18.5 2.5a2.1 2.1 0 0 1 3 3L12 15l-4 1 1-4z"></path>
    </svg>
  ),
  Eye: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <path d="M1.5 12S5 5.5 12 5.5 22.5 12 22.5 12 19 18.5 12 18.5 1.5 12 1.5 12z"></path>
      <circle cx="12" cy="12" r="3"></circle>
    </svg>
  ),
  Calendar: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4.5" width="18" height="16.5" rx="2.5"></rect>
      <line x1="3" y1="10" x2="21" y2="10"></line>
      <line x1="8" y1="2.5" x2="8" y2="6"></line>
      <line x1="16" y1="2.5" x2="16" y2="6"></line>
    </svg>
  ),
  CalendarPlain: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4.5" width="18" height="16.5" rx="2.5"></rect>
      <line x1="3" y1="10" x2="21" y2="10"></line>
    </svg>
  ),
  Search: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="7.5"></circle>
      <line x1="21" y1="21" x2="16.7" y2="16.7"></line>
    </svg>
  ),
  Users: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path>
      <circle cx="9" cy="7" r="4"></circle>
      <path d="M23 21v-2a4 4 0 0 0-3-3.87"></path>
    </svg>
  ),
  UserPlus: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"></path>
      <circle cx="9" cy="7" r="4"></circle>
      <line x1="19" y1="8" x2="19" y2="14"></line>
      <line x1="22" y1="11" x2="16" y2="11"></line>
    </svg>
  ),
  ChevronLeft: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="15 18 9 12 15 6"></polyline>
    </svg>
  ),
  ChevronRight: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="9 18 15 12 9 6"></polyline>
    </svg>
  ),
  Close: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18"></line>
      <line x1="6" y1="6" x2="18" y2="18"></line>
    </svg>
  ),
  /** Sort affordance: both chevrons when idle, one when the column is active. */
  Sort: ({ direction }: { direction: SortDirection | null }) => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round">
      {direction !== "asc" && <polyline points="7 15 12 20 17 15"></polyline>}
      {direction !== "desc" && <polyline points="7 9 12 4 17 9"></polyline>}
    </svg>
  )
};

/** Column header for the attendance table. Sorting itself stays in
 *  useTableControls; this only renders the design's affordance. */
function SortTh({
  label,
  columnKey,
  sort,
  onToggle,
  notSortable,
}: {
  label: string;
  columnKey: string;
  sort: SortState;
  onToggle: (key: string) => void;
  notSortable?: boolean;
}) {
  if (notSortable) return <th className="is-actions">{label}</th>;
  const active = sort.key === columnKey;
  return (
    <th>
      <button
        type="button"
        className={`eds-sort${active ? " is-active" : ""}`}
        onClick={() => onToggle(columnKey)}
        title={`Sort by ${label}`}
      >
        {label}
        <Icons.Sort direction={active ? sort.direction : null} />
      </button>
    </th>
  );
}

/** Identity tint for a member avatar, stable across sorts and searches. */
const AVATAR_TINTS = ["eds-avatar--blue", "eds-avatar--green", "eds-avatar--purple", "eds-avatar--rose", ""];

function initialsOf(first: string, last: string): string {
  const a = (first || "").trim()[0] || "";
  const b = (last || "").trim()[0] || "";
  return (a + b).toUpperCase() || "?";
}

/** An em dash reads better than a hyphen for an empty cell. */
function orDash(value: string) {
  return value === "-" ? <span className="eds-dash">—</span> : value;
}


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
    <section className="eds-card">
      <div className="eds-card-head">
        <span className="eds-chip"><Icons.Calendar /></span>
        <div className="eds-card-titles">
          <h2 className="eds-card-title">Month Summary</h2>
          <p className="eds-card-sub">{monthName} {year}</p>
        </div>
      </div>
      <div className="eds-card-body">
        {loading ? (
          <div className="eds-note">Loading summary…</div>
        ) : !summary ? (
          <div className="eds-note">Summary unavailable.</div>
        ) : (
          <div className="eds-metrics">
            {rows.map((r) => (
              <div key={r.label} className="eds-metric">
                <div className="eds-metric-lab">{r.label}</div>
                <div className="eds-metric-val">{r.value}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

const parseBreakInputToMinutes = (val: string): number => {
  const clean = val.trim().toLowerCase();
  if (!clean) return 0;

  const hourMatch = clean.match(/(\d+(?:\.\d+)?)\s*h/);
  const minMatch = clean.match(/(\d+(?:\.\d+)?)\s*m/);

  let totalMins = 0;
  if (hourMatch) {
    totalMins += parseFloat(hourMatch[1]) * 60;
  }
  if (minMatch) {
    totalMins += parseFloat(minMatch[1]);
  }

  if (hourMatch || minMatch) {
    return Math.round(totalMins);
  }

  if (clean.includes(":")) {
    const [hStr, mStr] = clean.split(":");
    const h = parseInt(hStr, 10) || 0;
    const m = parseInt(mStr, 10) || 0;
    return h * 60 + m;
  }

  const num = parseFloat(clean) || 0;
  if (clean.includes(".") || num <= 5) {
    return Math.round(num * 60);
  }
  return num;
};

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
  // Stamped when the roster/records load settles, so the card footer can say
  // how fresh the table is instead of claiming a refresh cadence it lacks.
  const [lastSynced, setLastSynced] = useState<Date | null>(null);

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
      .finally(() => {
        setLoading(false);
        setLastSynced(new Date());
      });
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
    let breakStr = "";
    if (breakHours > 0) {
      const totalMins = Math.round(breakHours * 60);
      const h = Math.floor(totalMins / 60);
      const m = totalMins % 60;
      if (h > 0) {
        breakStr = `${h}h${m > 0 ? ` ${m}m` : ""}`;
      } else {
        breakStr = `${m}m`;
      }
    }
    setEditCell({
      employee_id,
      date,
      sign_in_time: rec?.sign_in_time || "",
      sign_out_time: rec?.sign_out_time || "",
      break_minutes: breakStr,
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
    const target = dailyTarget(emp?.expected_working_hours);
    if (!signIn || !signOut) return { status: "ABSENT", hoursWorked: 0 };
    const [ih, im] = signIn.split(":").map(Number);
    const [oh, om] = signOut.split(":").map(Number);
    const inMins = ih * 60 + im;
    const outMins = oh * 60 + om;
    const breakMins = Math.max(0, typeof breakMinutes === "string" ? parseBreakInputToMinutes(breakMinutes) : Number(breakMinutes) || 0);
    const workedMins = outMins - inMins - breakMins;
    if (workedMins <= 0) return { status: "ABSENT", hoursWorked: 0 };
    const hoursWorked = workedMins / 60;
    // No daily target: they worked, so they were present. There is no shorter
    // -than-required to be short of.
    if (target === null) return { status: "PRESENT", hoursWorked };
    const expectedMins = target * 60;
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
    const breakMins = breakEntered ? parseBreakInputToMinutes(editCell.break_minutes) : 0;
    api.adminSet({
      employee_id: editCell.employee_id,
      date: editCell.date,
      sign_in_time: editCell.sign_in_time || null,
      sign_out_time: editCell.sign_out_time || null,
      break_hours: breakEntered ? breakMins / 60 : null,
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
    const metric = (label: string, value: React.ReactNode, tone?: "emerald" | "amber") => (
      <div className="eds-metric">
        <div className="eds-metric-lab">{label}</div>
        <div className={`eds-metric-val${tone ? ` is-${tone}` : ""}`}>{value}</div>
      </div>
    );
    return (
      <div className="eds-card-body">
        <div className="eds-metrics">
          {metric("First Check-In", formatTime12h(d?.sign_in_time), "emerald")}
          {metric("Last Check-Out", formatTime12h(d?.sign_out_time), "amber")}
          {metric("Working Hours", formatCompactDuration(d?.total_work_hours))}
          {metric("Break Time", formatCompactDuration(d?.total_break_hours))}
          {metric("Break Outs", d?.break_out_count ?? 0)}
          {metric("Break Ins", d?.break_in_count ?? 0)}
        </div>

        <div className="eds-eyebrow">Timeline</div>
        <div className="eds-timeline">
          {(d?.events?.length || 0) === 0 ? (
            <div className="eds-timeline-empty">No attendance events recorded for this date.</div>
          ) : (
            d?.events.map((evt) => {
              const { label, kind } = labelFor(evt);
              return (
                <div key={evt.id} className="eds-timeline-row">
                  <span className="eds-timeline-time">{formatEventTime12h(evt.event_time)}</span>
                  <span className={`eds-timeline-label is-${kind}`}>{label}</span>
                </div>
              );
            })
          )}
        </div>

        <div className="eds-note">
          Status: <b>{formatStatusLabel(d?.status || "ABSENT", d?.total_work_hours)}</b>
        </div>
      </div>
    );
  };

  const dateInputRef = useRef<HTMLInputElement | null>(null);

  /**
   * Open the native date picker.
   *
   * `showPicker()` is Chromium 99+ / Safari 16+ / Firefox 101+. It throws when
   * the browser does not have it, or when the call is not treated as
   * user-activated — neither of which should leave the field unusable, so the
   * fallback focuses it and the user can still type or use the arrow keys.
   */
  const openDatePicker = () => {
    const el = dateInputRef.current;
    if (!el) return;
    try {
      el.showPicker();
    } catch {
      el.focus();
    }
  };

  const dayNavigator = (
    <div className="eds-att-nav">
      {/* Clicking ANYWHERE on this field opens the calendar.
          By default a date input only opens its picker when you hit the
          browser's own tiny indicator glyph — clicking the date text just
          focuses a segment and looks broken. `showPicker()` is the only way to
          open it from elsewhere, so the whole field becomes the target and the
          native indicator is hidden in CSS (see .eds-datefield input) to leave
          one affordance instead of two overlapping ones. */}
      <label
        className="eds-datefield"
        title="Pick a date"
        onClick={openDatePicker}
      >
        <Icons.CalendarPlain />
        <input
          ref={dateInputRef}
          type="date"
          value={selectedDate}
          min="2026-01-01"
          max={todayIso}
          onChange={(e) => setSelectedDate(e.target.value)}
          // The label's onClick already covers pointer users. This keeps the
          // keyboard path working: tab to the field, press Enter or Space.
          onKeyDown={(e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              openDatePicker();
            }
          }}
        />
      </label>
      <div className="eds-seg">
        <button type="button" className="eds-seg-btn" onClick={() => changeDay(-1)} title="Go to Previous Day">
          <Icons.ChevronLeft />
          Previous Day
        </button>
        <button
          type="button"
          className="eds-seg-btn eds-seg-btn--today"
          onClick={() => setSelectedDate(todayIso)}
          title="Go to Today"
        >
          Today
        </button>
        <button
          type="button"
          className="eds-seg-btn"
          onClick={() => changeDay(1)}
          disabled={selectedDate >= todayIso}
          title={selectedDate >= todayIso ? "Cannot navigate past today" : "Go to Next Day"}
        >
          Next Day
          <Icons.ChevronRight />
        </button>
      </div>
    </div>
  );

  if (!isHrOrAdmin) {
    const empDateObj = new Date(selectedDate);
    const empDateLabel = empDateObj.toLocaleString("en-IN", { weekday: "long", day: "numeric", month: "short", year: "numeric" });
    return (
      <div className="eds">
        <header className="eds-topbar">
          <div>
            <h1 className="eds-title">My Attendance</h1>
            <p className="eds-subtitle">Employee view · Daily attendance</p>
          </div>
          <GlobalHeaderControls />
        </header>

        <div className="eds-page">
          {/* Daily summary + timeline for the selected date */}
          <section className="eds-card">
            <div className="eds-card-head eds-att-head">
              <div className="eds-att-title">
                <span className="eds-chip"><Icons.Calendar /></span>
                <div className="eds-card-titles">
                  <h2 className="eds-card-title">Daily Summary</h2>
                  <p className="eds-card-sub">{empDateLabel}</p>
                </div>
              </div>
              {dayNavigator}
            </div>
            {renderDaySummary(myDetails, myDetailsLoading)}
          </section>

          <section className="eds-card">
            <div className="eds-card-body">
              <MonthlyAttendanceGrid month={month} year={year} setMonth={setMonth} setYear={setYear} records={records} loading={loading} />
            </div>
          </section>

          {user?.employee_id && (
            <MonthlySummaryCard employeeId={user.employee_id} month={month} year={year} />
          )}
        </div>
      </div>
    );
  }

  const selectedDateObj = new Date(selectedDate);
  const dayName = selectedDateObj.toLocaleString("en-IN", { weekday: "long" });
  const dayNum = selectedDateObj.getDate();
  const monthName = selectedDateObj.toLocaleString("en-IN", { month: "short" });
  const dayLabelFull = `${dayName}, ${dayNum} ${monthName} ${selectedDateObj.getFullYear()}`;

  const selectedIsWeekend = [0, 6].includes(selectedDateObj.getDay());

  const activeRoster = rosterTab === "staff" ? staffMembers : employees;

  // Index the day's records by employee id, once.
  //
  // This used to be `records.find(...)` called from inside the .map() below —
  // a linear scan of every record for every employee, so the cost was
  // employees x records. A 200-person roster with a month of history is well
  // over a million comparisons, and because none of this was memoised it ran
  // again on EVERY render: each keystroke in the search box, every sort click.
  // One pass to build the index, then O(1) lookups.
  const recordsForSelectedDate = useMemo(() => {
    const byEmployee = new Map<number, (typeof records)[number]>();
    for (const r of records) {
      if (r.date === selectedDate) byEmployee.set(r.employee_id, r);
    }
    return byEmployee;
  }, [records, selectedDate]);

  const employeeRows = useMemo(
    () =>
      [...activeRoster]
        .sort((a, b) => (Number(a.employee_code) || 0) - (Number(b.employee_code) || 0))
        .map(e => ({ info: e, rec: recordsForSelectedDate.get(e.id) })),
    [activeRoster, recordsForSelectedDate],
  );

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
      required: (r) => dailyTarget(r.info.expected_working_hours) ?? 0,
      working_hours: (r) => Number(r.rec?.total_work_hours ?? 0),
      break_time: (r) => Number(r.rec?.total_break_hours ?? 0),
      status: (r) => effectiveStatus(r),
    },
    searchableText: (r) =>
      `${r.info.employee_code} ${r.info.first_name} ${r.info.last_name} ${effectiveStatus(r)}`,
  });

  // Worked hours for a row: ticks live for anyone checked in today who has not
  // checked out yet, otherwise the stored snapshot. Same rule the cell applied
  // before, lifted out so the progress bar and the footer total read the same
  // number the cell prints.
  const workedHoursFor = (rec: AttendanceRow | undefined): number | null => {
    const signedOut = !!(rec?.sign_out_time && rec.sign_out_time !== "00:00:00");
    if (rec?.sign_in_time && selectedDate === todayIso && !signedOut) {
      const [h, m, sec] = rec.sign_in_time.split(":").map(Number);
      const start = new Date();
      start.setHours(h, m, sec || 0, 0);
      if (nowTick > start) {
        const elapsed = (nowTick.getTime() - start.getTime()) / (1000 * 60 * 60);
        return Math.max(0, elapsed - Number(rec?.total_break_hours || 0));
      }
    }
    if (rec?.total_work_hours != null && rec.total_work_hours > 0) return rec.total_work_hours;
    return null;
  };

  // Card footer aggregate — a straight sum of the Working hours column above it.
  let loggedHours = 0;
  let loggedMembers = 0;
  displayedEmployeeRows.forEach(({ rec }) => {
    const worked = workedHoursFor(rec);
    if (worked != null && worked > 0) {
      loggedHours += worked;
      loggedMembers++;
    }
  });

  // Same tones the status text used before: present emerald, absent rose,
  // leave/part-day sky, week off and holiday neutral.
  const statusTone = (s: string) =>
    s === "PRESENT"
      ? " eds-status--present"
      : s === "ABSENT"
      ? " eds-status--absent"
      : ["ON_LEAVE", "PAID_LEAVE", "HALF_DAY", "SHORT"].includes(s)
      ? " eds-status--info"
      : "";

  return (
    <div className="eds">
      <header className="eds-topbar">
        <div>
          <h1 className="eds-title">Attendance</h1>
          <p className="eds-subtitle">{isAdmin ? "Admin view" : "HR view"} · Daily attendance</p>
        </div>
        <GlobalHeaderControls />
      </header>

      <div className="eds-page">
        {success && <div className="alert alert-success">{success}</div>}
        {error && !editCell && <div className="alert alert-error">{error}</div>}

        <div className="eds-tabs">
          {([
            { key: "employees", label: "Employees", count: employees.length },
            { key: "staff", label: "Non-Employee Staff", count: staffMembers.length },
          ] as const).map((tab) => (
            <button
              key={tab.key}
              type="button"
              className={`eds-tab${rosterTab === tab.key ? " is-active" : ""}`}
              onClick={() => { setRosterTab(tab.key); setAttendanceSearch(""); }}
            >
              {tab.key === "staff" ? <Icons.UserPlus /> : <Icons.Users />}
              <span>{tab.label}</span>
              <span className="eds-tab-count">{tab.count}</span>
            </button>
          ))}
        </div>

        <section className="eds-card">
          <div className="eds-card-head eds-att-head">
            <div className="eds-att-title">
              <span className="eds-chip"><Icons.Calendar /></span>
              <div className="eds-card-titles">
                <h2 className="eds-card-title">
                  {rosterTab === "staff" ? "Non-Employee Staff Attendance" : "Daily Attendance"}
                </h2>
                <p className="eds-card-sub">
                  {rosterTab === "staff"
                    ? `${dayLabelFull} · not counted in employee reports`
                    : dayLabelFull}
                </p>
              </div>
            </div>

            <div className="eds-att-counts">
              <span className="eds-count-pill eds-count-pill--present">
                <i className="eds-live-dot"></i>
                Present
                <b>{dayCounts.present}</b>
              </span>
              <span className="eds-count-pill">
                Absent
                <b>{dayCounts.absent}</b>
              </span>
            </div>

            {dayNavigator}
          </div>

          <div className="eds-tablebar">
            <label className="eds-search">
              <Icons.Search />
              <input
                type="search"
                value={attendanceSearch}
                onChange={(e) => setAttendanceSearch(e.target.value)}
                placeholder="Search by name, code, status..."
              />
            </label>
            {attendanceHasActive && (
              <button
                type="button"
                className="eds-action"
                onClick={clearAttendanceControls}
                title="Clear search, sort and column filters"
              >
                Clear filters
              </button>
            )}
            <span className="eds-showing">
              Showing <b>{displayedEmployeeRows.length}</b> of <b>{employeeRows.length}</b>
            </span>
          </div>

          <div className="eds-table-wrap">
            <table className="eds-table">
              <colgroup>
                <col style={{ width: "21.5%" }} />
                <col style={{ width: "10.3%" }} />
                <col style={{ width: "10.3%" }} />
                <col style={{ width: "11.3%" }} />
                <col style={{ width: "14.4%" }} />
                <col style={{ width: "10.3%" }} />
                <col style={{ width: "10.3%" }} />
                <col style={{ width: "11.8%" }} />
              </colgroup>
              <thead>
                <tr>
                  <SortTh label="Member" columnKey="member" sort={attendanceSort} onToggle={toggleAttendanceSort} />
                  <SortTh label="First in" columnKey="sign_in_time" sort={attendanceSort} onToggle={toggleAttendanceSort} />
                  <SortTh label="Last out" columnKey="sign_out_time" sort={attendanceSort} onToggle={toggleAttendanceSort} />
                  <SortTh label="Required time" columnKey="required" sort={attendanceSort} onToggle={toggleAttendanceSort} />
                  <SortTh label="Working hours" columnKey="working_hours" sort={attendanceSort} onToggle={toggleAttendanceSort} />
                  <SortTh label="Break time" columnKey="break_time" sort={attendanceSort} onToggle={toggleAttendanceSort} />
                  <SortTh label="Status" columnKey="status" sort={attendanceSort} onToggle={toggleAttendanceSort} />
                  <SortTh label="Actions" columnKey="__actions" sort={attendanceSort} onToggle={toggleAttendanceSort} notSortable />
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
                    <td colSpan={8} className="eds-table-empty">
                      No attendance rows match your search.
                    </td>
                  </tr>
                ) : (
                  displayedEmployeeRows.map(({ info, rec }) => {
                    const rawStatus = rec?.status || (selectedIsWeekend ? "WEEKLY_OFF" : "ABSENT");

                    // If DB says ABSENT but employee has actual working hours recorded,
                    // derive the real status from total_work_hours vs expected
                    let s = rawStatus;
                    const target = dailyTarget(info.expected_working_hours);
                    if (rawStatus === "ABSENT" && rec?.total_work_hours && rec.total_work_hours > 0) {
                      if (target === null) s = "PRESENT";
                      else if (rec.total_work_hours >= target * 0.9) s = "PRESENT";
                      else if (rec.total_work_hours >= target * 0.5) s = "HALF_DAY";
                      else s = "SHORT";
                    }

                    const signedOut = !!(rec?.sign_out_time && rec.sign_out_time !== "00:00:00");
                    const worked = workedHoursFor(rec);

                    return (
                      <tr key={info.id}>
                        <td>
                          <div className="eds-member">
                            <span className={`eds-avatar eds-avatar--lg ${AVATAR_TINTS[info.id % AVATAR_TINTS.length]}`}>
                              {initialsOf(info.first_name, info.last_name)}
                            </span>
                            <span className="eds-member-name">{info.first_name} {info.last_name}</span>
                          </div>
                        </td>
                        <td className="eds-cell-time">{orDash(formatTime12h(rec?.sign_in_time))}</td>
                        <td className="eds-cell-time">
                          {orDash(signedOut ? formatTime12h(rec?.sign_out_time) : "-")}
                        </td>
                        <td className="eds-cell-dim">
                          {s === "WEEKLY_OFF" || s === "HOLIDAY"
                            ? orDash("-")
                            : target === null
                              ? "No fixed hours"
                              : `${target} Hours`}
                        </td>
                        <td>
                          {worked == null ? (
                            orDash("-")
                          ) : (
                            <div className="eds-hours">
                              <span className="eds-hours-val">{formatCompactDuration(worked)}</span>
                              {target === null ? null : (
                                <div
                                  className="eds-hours-track"
                                  role="img"
                                  aria-label={`${formatCompactDuration(worked)} worked of ${target} hours required`}
                                >
                                  <div
                                    className="eds-hours-fill"
                                    style={{ width: `${Math.min(100, Math.round((worked / target) * 100))}%` }}
                                  ></div>
                                </div>
                              )}
                            </div>
                          )}
                        </td>
                        <td className="eds-cell-dim">
                          {orDash(formatCompactDuration(rec?.total_break_hours))}
                        </td>
                        <td>
                          <span className={`eds-status${statusTone(s)}`}>
                            <i></i>
                            {formatStatusLabel(s, signedOut ? rec?.total_work_hours : null)}
                          </span>
                        </td>
                        <td>
                          <div className="eds-rowactions">
                            <button
                              type="button"
                              className="eds-iconbtn eds-iconbtn--view"
                              onClick={() => setDetailsEmployee(info)}
                              title="View Attendance Details"
                            >
                              <Icons.Eye />
                            </button>
                            <button
                              type="button"
                              className="eds-iconbtn eds-iconbtn--month"
                              onClick={() => { setAttendanceDialogEmployee(info); setDialogMonth(month); setDialogYear(year); }}
                              title="View Monthly Attendance History"
                            >
                              <Icons.CalendarPlain />
                            </button>
                            {isHR && (
                              <button
                                type="button"
                                className="eds-iconbtn eds-iconbtn--edit"
                                onClick={() => openEdit(info.id, selectedDate)}
                                title="Edit Attendance for this Day"
                              >
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

          <div className="eds-card-foot eds-att-foot">
            <span>
              {displayedEmployeeRows.length} members ·{" "}
              <b className="is-emerald">{loggedHours > 0 ? formatCompactDuration(loggedHours) : "0m"}</b> logged today · avg{" "}
              <b className="is-text">{loggedMembers ? formatCompactDuration(loggedHours / loggedMembers) : "—"}</b>
            </span>
            <span>
              Working hours tick live · last synced{" "}
              {lastSynced
                ? lastSynced.toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", hour12: true })
                : "—"}
            </span>
          </div>
        </section>
      </div>

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
                    <label>Break Time</label>
                    <input
                      type="text"
                      placeholder="e.g. 90m, 1.5h, or 1:30"
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
                  <div style={{ gridColumn: "1 / -1" }} className="eds-note">
                    Break is deducted from working hours — enter in minutes (e.g. 90m or 90), hours (e.g. 1.5h or 1.5), or HH:MM (e.g. 1:30). Leave blank to use the camera recorded break.
                  </div>
                  {editCell.sign_in_time && editCell.sign_out_time && (() => {
                    const { hoursWorked } = calcStatusFromTimes(editCell.sign_in_time, editCell.sign_out_time, editCell.employee_id, editCell.break_minutes);
                    const h = Math.floor(hoursWorked);
                    const m = Math.round((hoursWorked - h) * 60);
                    const brk = Math.max(0, parseBreakInputToMinutes(editCell.break_minutes));
                    const brkH = Math.floor(brk / 60);
                    const brkM = Math.round(brk % 60);
                    const brkStr = brkH > 0 ? `${brkH}h ${brkM}m` : `${brkM}m`;
                    return hoursWorked > 0 ? (
                      <div style={{ gridColumn: "1 / -1", fontSize: "12px", color: "#60a5fa", fontWeight: 600 }}>
                        ⏱ {h}h {m}m worked{brk > 0 ? ` (${brkStr} break deducted)` : ""} · status auto-set
                      </div>
                    ) : null;
                  })()}
                </div>
                <div className="modal-actions" style={{ display: "flex", justifyContent: "space-between" }}>
                  <button
                    type="button"
                    className="btn"
                    style={{ background: "rgba(251, 113, 133, 0.12)", color: "#fb7185" }}
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
              <div className="eds-note" style={{ marginBottom: "0.75rem" }}>
                Employee: <b>{detailsEmployee.first_name} {detailsEmployee.last_name}</b> · {dayLabelFull}
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
                className="eds-iconbtn"
                onClick={() => setAttendanceDialogEmployee(null)}
                style={{ position: "absolute", top: "1rem", right: "1rem", zIndex: 10 }}
                title="Close Monthly View"
              >
                <Icons.Close />
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
    </div>
  );
}
