import { useEffect, useMemo, useState } from "react";
import { NavLink } from "react-router-dom";
import {
  calendar as calendarApi,
  leave as leaveApi,
  company as companyApi,
  attendance as attendanceApi,
  employees as employeeApi
} from "../api/client";
import { useAuth } from "../auth/AuthContext";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import { dailyTarget } from "../utils/workingHours";
interface ReminderBirthday {
  employee_id: number;
  employee_code: string;
  name: string;
  date: string;
}

interface ReminderWorkAnniversary {
  employee_id: number;
  employee_code: string;
  name: string;
  date_of_joining: string;
  date: string;
  years: number;
}

interface ReminderMarriageAnniversary {
  employee_id: number;
  employee_code: string;
  name: string;
  date_of_marriage: string;
  date: string;
  years: number;
}

interface ReminderEvent {
  id: number;
  title: string;
  date: string;
  event_type: string;
  description?: string | null;
  employee_id?: number | null;
  employee_name?: string | null;
}

interface ReminderDay {
  for_date: string;
  birthdays: ReminderBirthday[];
  work_anniversaries: ReminderWorkAnniversary[];
  marriage_anniversaries?: ReminderMarriageAnniversary[];
  events: ReminderEvent[];
}

type CelebrationKind = "birthday" | "work" | "marriage";

interface CelebrationItem {
  key: string;
  kind: CelebrationKind;
  name: string;
  date: string;
  years?: number;
}

function formatLocalDate(d: Date): string {
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

interface HolidayRow {
  id: number;
  date: string;
  name: string;
  is_optional: boolean;
}

/* 24-box strokes at 1.8, round caps, currentColor. Rendered size comes from
   the chip that holds them (.eds-chip 16px, .eds-tile-chip 17px). */
const Icons = {
  Clock: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="9"></circle><polyline points="12 7 12 12 15.5 14"></polyline></svg>
  ),
  Calendar: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="4.5" width="18" height="16.5" rx="2.5"></rect><line x1="3" y1="10" x2="21" y2="10"></line><line x1="8" y1="2.5" x2="8" y2="6"></line><line x1="16" y1="2.5" x2="16" y2="6"></line></svg>
  ),
  Users: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M17 20v-1.5A3.5 3.5 0 0 0 13.5 15h-6A3.5 3.5 0 0 0 4 18.5V20"></path><circle cx="10.5" cy="8" r="3.5"></circle><path d="M20 20v-1.5a3.5 3.5 0 0 0-2.6-3.4"></path></svg>
  ),
  Person: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M19 20v-1.5A4.5 4.5 0 0 0 14.5 14h-5A4.5 4.5 0 0 0 5 18.5V20"></path><circle cx="12" cy="7.5" r="4"></circle></svg>
  ),
  Operations: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
  ),
  Birthday: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M4 20.5v-4a2.5 2.5 0 0 1 2.5-2.5h11a2.5 2.5 0 0 1 2.5 2.5v4z"></path><line x1="3" y1="20.5" x2="21" y2="20.5"></line><line x1="12" y1="10.5" x2="12" y2="14"></line><path d="M12 7.5c1 -1.4 0 -2.5 0 -2.5s-1 1.1 0 2.5z"></path></svg>
  ),
  Team: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M15 20v-1.5A3.5 3.5 0 0 0 11.5 15h-5A3.5 3.5 0 0 0 3 18.5V20"></path><circle cx="9" cy="8" r="3.5"></circle><polyline points="16 11.5 18 13.5 22 9.5"></polyline></svg>
  ),
  Target: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="9"></circle><circle cx="12" cy="12" r="4.5"></circle><circle cx="12" cy="12" r="0.6"></circle></svg>
  ),
  Payroll: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round"><path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"></path><polyline points="14 3 14 8 19 8"></polyline><line x1="15" y1="13" x2="9" y2="13"></line><line x1="15" y1="17" x2="9" y2="17"></line></svg>
  ),
  Star: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
  )
};

/** Both ring paths trace the same 36-box circle. */
const RING_PATH = "M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831";

export default function Dashboard() {
  const { user, hasRole } = useAuth();
  const isAdmin = hasRole("Admin");
  const isHR = hasRole("HR");
  const isAdminOrHr = isAdmin || isHR;

  const [reminderDays, setReminderDays] = useState<ReminderDay[]>([]);
  const [holidays, setHolidays] = useState<HolidayRow[]>([]);
  const [currentTime, setCurrentTime] = useState(new Date());
  const [empDetails, setEmpDetails] = useState<any>(null);

  const [attendanceStats, setAttendanceStats] = useState({
    present: 0,
    leave: 0,
    percentage: 0,
    avgHours: "08:15",
    onTime: 92,
    totalHours: "0h 0m",
    requiredHours: "0h"
  });

  const [dailyHoursMap, setDailyHoursMap] = useState<Record<string, string>>({});
  const [pendingLeaveList, setPendingLeaveList] = useState<any[]>([]);
  const [teamStats, setTeamStats] = useState({
    available: 0,
    onLeave: 0,
    weeklyOff: 0,
    totalEmps: 0,
    totalDepts: 0,
    lateToday: 0
  });
  const todayISO = useMemo(() => formatLocalDate(new Date()), []);
  // null when this person has no fixed daily target -- see utils/workingHours.
  // Kept as null rather than 0 so the weekly meter below can be hidden instead
  // of dividing by it and rendering a full bar against a "0 Hours" target.
  const dailyHoursTarget = dailyTarget(empDetails?.expected_working_hours);
  const expectedHoursPerDay = dailyHoursTarget ?? 0;

  const tomorrowISO = useMemo(() => {
    const d = new Date();
    d.setDate(d.getDate() + 1);
    return formatLocalDate(d);
  }, []);

  const relativeDayLabel = (iso: string): string => {
    if (iso === todayISO) return "Today";
    if (iso === tomorrowISO) return "Tomorrow";
    const target = new Date(iso + "T00:00:00");
    const base = new Date(todayISO + "T00:00:00");
    const diff = Math.round((target.getTime() - base.getTime()) / (24 * 60 * 60 * 1000));
    if (diff > 1) return `In ${diff} days`;
    const fmt = target.toLocaleDateString(undefined, { day: "2-digit", month: "short" });
    return fmt;
  };

  const upcomingCelebrations = useMemo<CelebrationItem[]>(() => {
    const items: CelebrationItem[] = [];
    reminderDays.forEach((day) => {
      (day.birthdays || []).forEach((b) =>
        items.push({
          key: `b-${b.employee_id}-${day.for_date}`,
          kind: "birthday",
          name: b.name,
          date: day.for_date,
        })
      );
      (day.work_anniversaries || []).forEach((w) =>
        items.push({
          key: `w-${w.employee_id}-${day.for_date}`,
          kind: "work",
          name: w.name,
          date: day.for_date,
          years: w.years,
        })
      );
      (day.marriage_anniversaries || []).forEach((m) =>
        items.push({
          key: `m-${m.employee_id}-${day.for_date}`,
          kind: "marriage",
          name: m.name,
          date: day.for_date,
          years: m.years,
        })
      );
    });
    return items.sort((a, b) => a.date.localeCompare(b.date));
  }, [reminderDays]);

  const upcomingEvents = useMemo<ReminderEvent[]>(() => {
    const all: ReminderEvent[] = [];
    reminderDays.forEach((day) => {
      (day.events || []).forEach((e) => all.push(e));
    });
    return all.sort((a, b) => a.date.localeCompare(b.date) || a.title.localeCompare(b.title));
  }, [reminderDays]);

  const weekDates = useMemo(() => {
    const out: string[] = [];
    const now = new Date();
    const day = now.getDay();
    // Start from the current week's Monday
    const diff = now.getDate() - day + (day === 0 ? -6 : 1);
    const monday = new Date(now.setDate(diff));

    for (let i = 0; i < 7; i++) {
      const d = new Date(monday);
      d.setDate(monday.getDate() + i);
      out.push(formatLocalDate(d));
    }
    return out;
  }, []);

  const weeklyHours = useMemo(() => {
    let total = 0;
    weekDates.forEach(d => {
      const val = dailyHoursMap[d];
      if (val && typeof val === 'string') {
        // Format is "Hh Mm|In|Out"
        const hoursPart = val.split('|')[0];
        const hMatch = hoursPart.match(/(\d+)h/);
        const mMatch = hoursPart.match(/(\d+)m/);
        const h = hMatch ? parseInt(hMatch[1], 10) : 0;
        const m = mMatch ? parseInt(mMatch[1], 10) : 0;
        total += h + (m / 60);
      }
    });
    return Math.round(total);
  }, [weekDates, dailyHoursMap]);

  const todayFormatted = useMemo(() => {
    const d = currentTime;
    const dayName = d.toLocaleDateString(undefined, { weekday: "long" });
    const fullDate = d.toLocaleDateString(undefined, { day: '2-digit', month: 'long', year: 'numeric' });
    const timeStr = d.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: true });
    return { dayName, fullDate, timeStr };
  }, [currentTime]);

  useEffect(() => {
    const timer = setInterval(() => setCurrentTime(new Date()), 1000);
    return () => clearInterval(timer);
  }, []);

  useEffect(() => {
    if (user?.employee_id) {
      employeeApi.get(user.employee_id).then(res => setEmpDetails(res.data));
    }
  }, [user?.employee_id]);

  useEffect(() => {
    const now = new Date();
    const firstDay = new Date(now.getFullYear(), now.getMonth(), 1);
    const lastDay = new Date(now.getFullYear(), now.getMonth() + 1, 0);
    const from = formatLocalDate(firstDay);
    const to = formatLocalDate(lastDay);

    // Working days elapsed so far this month (weekday, not a holiday). The
    // attendance percentage is graded against these days only, so a record
    // logged on a weekend or holiday can never push the ring past 100%.
    const currentMonthHolidays = holidays.filter(h => {
      const hd = new Date(h.date);
      return hd.getMonth() === now.getMonth() && hd.getFullYear() === now.getFullYear() && hd.getDate() <= now.getDate();
    });
    const workingDays = new Set<string>();
    for (let d = 1; d <= now.getDate(); d++) {
      const dateObj = new Date(now.getFullYear(), now.getMonth(), d);
      const dayOfWeek = dateObj.getDay();
      const isHoliday = currentMonthHolidays.some(h => new Date(h.date).getDate() === d);
      if (dayOfWeek !== 0 && dayOfWeek !== 6 && !isHoliday) workingDays.add(formatLocalDate(dateObj));
    }
    const workingDaysCount = workingDays.size;

    const loadAttendance = () => {
      attendanceApi.list(from, to, user?.employee_id || undefined).then(res => {
        const records = res.data || [];

        let present = 0;
        let leaveCount = 0;
        let totalMinutes = 0;
        let onTimeCount = 0;
        let presentRecordsCount = 0;
        const dailyMins: Record<string, { totalMins: number, count: number, inTime?: string, outTime?: string }> = {};

        let weekendsPassed = 0;
        for (let d = 1; d <= now.getDate(); d++) {
          const dateObj = new Date(now.getFullYear(), now.getMonth(), d);
          const dayOfWeek = dateObj.getDay();
          if (dayOfWeek === 0 || dayOfWeek === 6) weekendsPassed++;
        }

        let personalMinutes = 0;
        let personalPresent = 0;
        let personalLeave = 0;
        let personalPresentOnWorkingDays = 0;

        records.forEach((r: any) => {
          const isPresent = ['PRESENT', 'SHORT', 'HALF_DAY'].includes(r.status);
          const isMe = r.employee_id === user?.employee_id;

          if (isPresent) {
            presentRecordsCount++;
            if (r.status === 'HALF_DAY') present += 0.5;
            else present++;
            if (!r.is_late) onTimeCount++;

            let mins = 0;
            if (r.total_work_hours != null && Number(r.total_work_hours) > 0) {
              mins = Math.round(Number(r.total_work_hours) * 60);
            } else if (r.date === todayISO && r.sign_in_time && !r.sign_out_time) {
              const [h, m] = r.sign_in_time.split(':').map(Number);
              const punchIn = new Date(now.getFullYear(), now.getMonth(), now.getDate(), h, m);
              const diffMs = now.getTime() - punchIn.getTime();
              if (diffMs > 0) mins = Math.floor(diffMs / (1000 * 60));
            }

            if (mins > 0) {
              totalMinutes += mins;
              if (isMe) personalMinutes += mins;
              if (!dailyMins[r.date]) dailyMins[r.date] = { totalMins: 0, count: 0 };
              dailyMins[r.date].totalMins += mins;
              dailyMins[r.date].count++;
              if (r.sign_in_time) dailyMins[r.date].inTime = r.sign_in_time;
              if (r.sign_out_time) dailyMins[r.date].outTime = r.sign_out_time;
            }
            if (isMe) {
              const credit = r.status === 'HALF_DAY' ? 0.5 : 1;
              personalPresent += credit;
              if (workingDays.has(r.date)) personalPresentOnWorkingDays += credit;
            }
          } else if (r.status === 'ON_LEAVE' || r.status === 'PAID_LEAVE') {
            leaveCount++;
            if (isMe) personalLeave++;
          }
        });

        const weekendMinutes = weekendsPassed * expectedHoursPerDay * 60;
        const personalTotalMinutes = personalMinutes + weekendMinutes;
        const totalRequiredMinutes = 30 * expectedHoursPerDay * 60;

        const hoursWorked = Math.floor(personalTotalMinutes / 60);
        const minsWorked = Math.round(personalTotalMinutes % 60);
        const reqHours = Math.floor(totalRequiredMinutes / 60);

        setAttendanceStats({
          present: personalPresent,
          leave: personalLeave,
          percentage: workingDaysCount > 0
            ? Math.min(100, Math.round((personalPresentOnWorkingDays / workingDaysCount) * 100))
            : 0,
          avgHours: personalPresent > 0 ? `${Math.floor((personalMinutes / personalPresent) / 60).toString().padStart(2, '0')}:${Math.round((personalMinutes / personalPresent) % 60).toString().padStart(2, '0')}` : "00:00",
          onTime: presentRecordsCount > 0 ? Math.round((onTimeCount / presentRecordsCount) * 100) : 100,
          totalHours: `${hoursWorked}h ${minsWorked}m`,
          requiredHours: `${reqHours}h`
        });

        // If Admin/HR, we can still use company averages for other logic if needed,
        // but the main stats card now shows PERSONAL data.
        const dailyAvgStr: Record<string, string> = {};
        Object.keys(dailyMins).forEach(date => {
          const { totalMins, count, inTime, outTime } = dailyMins[date];
          const avg = totalMins / count;
          const h = Math.floor(avg / 60);
          const m = Math.round(avg % 60);
          dailyAvgStr[date] = `${h}h ${m}m|${inTime?.slice(0, 5) || '--'}|${outTime?.slice(0, 5) || '--'}`;
        });
        setDailyHoursMap(dailyAvgStr);
      }).catch(() => { });
    };

    loadAttendance();
    const interval = setInterval(loadAttendance, 30 * 60 * 1000);
    return () => clearInterval(interval);
  }, [user?.employee_id, empDetails, holidays.length]);

  useEffect(() => {
    calendarApi
      .reminders(todayISO, 7)
      .then((r) => {
        const raw = r.data;
        const list: ReminderDay[] = Array.isArray(raw) ? raw : raw ? [raw] : [];
        setReminderDays(list);
      })
      .catch(() => setReminderDays([]));
    companyApi.holidays().then(r => setHolidays(r.data)).catch(() => setHolidays([]));

    leaveApi.approvals({ status: "PENDING" }).then(res => setPendingLeaveList(Array.isArray(res.data) ? res.data.slice(0, 4) : []))
      .catch(() => setPendingLeaveList([]));
  }, [todayISO]);

  useEffect(() => {
    companyApi.stats().then(res => {
      const s = res.data;
      setTeamStats({
        available: s.present_today,
        onLeave: s.on_leave_today,
        weeklyOff: s.total_employees - s.present_today - s.on_leave_today,
        totalEmps: s.total_employees,
        totalDepts: s.total_departments,
        lateToday: s.late_today
      });
    }).catch(() => { });
  }, []);


  return (
    <div className="eds">
      <header className="eds-topbar">
        <div>
          <h1 className="eds-title">Welcome, {user?.username?.split(' ')[0] || "User"} 👋</h1>
          <p className="eds-subtitle">{user?.designation || "Admin"}</p>
        </div>
        <GlobalHeaderControls />
      </header>

      {/* <section className="card" style={{ padding: "1rem 1.1rem", marginBottom: "1rem", background: "linear-gradient(135deg, rgba(15,23,42,0.96), rgba(51,65,85,0.94))", color: "#fff", border: "none" }}>
        <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap", alignItems: "center" }}>
          <div>
            <div className="eyebrow" style={{ background: "rgba(255,255,255,0.12)", borderColor: "rgba(255,255,255,0.16)", color: "#fff", marginBottom: "0.6rem" }}>Face Detection</div>
            <div style={{ fontSize: "1.15rem", fontWeight: 800 }}>Open the dedicated face detection section</div>
            <div style={{ color: "rgba(255,255,255,0.78)", marginTop: "0.25rem" }}>A separate workflow for employee enrollment, recognition, and attendance review.</div>
          </div>
          <NavLink to="/face-detection" className="btn btn-primary" style={{ textDecoration: "none" }}>Open Face Detection Section</NavLink>
        </div>
      </section> */}

      <div className="eds-grid">
        {/* Clock — compact card, one live figure */}
        <section className="eds-card eds-c1-3">
          <div className="eds-card-head">
            <span className="eds-chip"><Icons.Clock /></span>
            <div className="eds-card-titles">
              <h2 className="eds-card-title">Current Time</h2>
              <p className="eds-card-sub">{todayFormatted.dayName}</p>
            </div>
            <span className="eds-live"><i className="eds-live-dot"></i>Live</span>
          </div>
          <div className="eds-card-body">
            <p className="eds-eyebrow">{todayFormatted.fullDate}</p>
            <div className="eds-clock">
              <span className="eds-clock-time">{todayFormatted.timeStr.split(' ')[0]}</span>
              <span className="eds-clock-ampm">{todayFormatted.timeStr.split(' ')[1]}</span>
            </div>
          </div>
          <div className="eds-card-foot">
            <span className="eds-pill"><Icons.Star />Have a productive day!</span>
          </div>
        </section>

        {hasRole("Admin") ? (
          <>
            <section className="eds-card eds-c4-5">
              <div className="eds-card-head">
                <span className="eds-chip"><Icons.Operations /></span>
                <div className="eds-card-titles">
                  <h2 className="eds-card-title">Workforce Attendance</h2>
                </div>
              </div>
              <div className="eds-card-body">
                <div className="eds-tiles">
                  <div className="eds-tile eds-tile--emerald">
                    <span className="eds-tile-chip"><Icons.Operations /></span>
                    <div className="eds-tile-figures">
                      <span className="eds-figure">{teamStats.totalEmps > 0 ? Math.round((teamStats.available / teamStats.totalEmps) * 100) : 0}%</span>
                      <span className="eds-label">Attendance Rate</span>
                      <span className="eds-sub">{teamStats.available} Employees Present</span>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            <section className="eds-card eds-c9-4">
              <div className="eds-card-head">
                <span className="eds-chip"><Icons.Clock /></span>
                <div className="eds-card-titles">
                  <h2 className="eds-card-title">Absence &amp; Punctuality</h2>
                </div>
              </div>
              <div className="eds-card-body">
                <div className="eds-tiles">
                  <div className="eds-tile eds-tile--amber">
                    <span className="eds-tile-chip"><Icons.Clock /></span>
                    <div className="eds-tile-figures">
                      <span className="eds-figure">{String(teamStats.lateToday).padStart(2, '0')}</span>
                      <span className="eds-label">Late Arrivals</span>
                    </div>
                  </div>
                  <div className="eds-tile eds-tile--sky">
                    <span className="eds-tile-chip"><Icons.Person /></span>
                    <div className="eds-tile-figures">
                      <span className="eds-figure">{String(teamStats.onLeave).padStart(2, '0')}</span>
                      <span className="eds-label">Employees On Leave</span>
                    </div>
                  </div>
                </div>
              </div>
            </section>
          </>
        ) : (
          <>
            <section className="eds-card eds-c4-9">
              <div className="eds-card-head">
                <span className="eds-chip"><Icons.Calendar /></span>
                <div className="eds-card-titles">
                  <h2 className="eds-card-title">Monthly Overview</h2>
                </div>
              </div>
              <div className="eds-card-body">
                <div className="eds-monthly">
                  <div className="eds-ring">
                    <div className="eds-ring-plot">
                      <svg viewBox="0 0 36 36">
                        <path className="eds-ring-track" d={RING_PATH} />
                        <path className="eds-ring-fill" strokeDasharray={`${attendanceStats.percentage}, 100`} d={RING_PATH} />
                      </svg>
                      <div className="eds-ring-value">{attendanceStats.percentage}%</div>
                    </div>
                    <span className="eds-key">
                      <span><i className="eds-key-dot eds-key-dot--emerald"></i>Present</span>
                    </span>
                  </div>
                  <div className="eds-tiles">
                    <div className="eds-tile eds-tile--emerald">
                      <span className="eds-tile-chip"><Icons.Team /></span>
                      <div className="eds-tile-figures">
                        <span className="eds-figure">{String(Math.round(attendanceStats.present)).padStart(2, '0')} Days</span>
                        <span className="eds-label">Present</span>
                      </div>
                    </div>
                    <div className="eds-tile eds-tile--sky">
                      <span className="eds-tile-chip"><Icons.Calendar /></span>
                      <div className="eds-tile-figures">
                        <span className="eds-figure">{String(Math.round(attendanceStats.leave)).padStart(2, '0')} Days</span>
                        <span className="eds-label">Leave</span>
                      </div>
                    </div>
                    <div className="eds-tile">
                      <span className="eds-tile-chip"><Icons.Clock /></span>
                      <div className="eds-tile-figures">
                        <span className="eds-figure">{attendanceStats.totalHours}</span>
                        <span className="eds-label">Worked Hours</span>
                      </div>
                    </div>
                    <div className="eds-tile">
                      <span className="eds-tile-chip"><Icons.Target /></span>
                      <div className="eds-tile-figures">
                        <span className="eds-figure">{attendanceStats.requiredHours}</span>
                        <span className="eds-label">Required Hours</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            {/* Weekly — wide strip; the footer meter is bounded by the two
                figures printed on either side of it. */}
            <section className="eds-card eds-c1-12">
              <div className="eds-card-head">
                <span className="eds-chip"><Icons.Calendar /></span>
                <div className="eds-card-titles">
                  <h2 className="eds-card-title">Weekly Overview</h2>
                </div>
                <span className="eds-key">
                  <span><i className="eds-key-dot eds-key-dot--emerald"></i>Today</span>
                </span>
              </div>
              <div className="eds-card-body">
                <div className="eds-week-grid">
                  {weekDates.map((d) => {
                    const dt = new Date(d);
                    const isToday = d === todayISO;
                    return (
                      <div key={d} className={`eds-week-day ${isToday ? 'eds-week-day--today' : ''}`}>
                        <span className="eds-week-dow">{dt.toLocaleDateString(undefined, { weekday: 'short' })}</span>
                        <span className="eds-week-date">{dt.getDate()}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
              <div className="eds-card-foot eds-week-foot">
                <div className="eds-week-block eds-week-block--logged">
                  <span className="eds-eyebrow">Weekly Performance</span>
                  <span className="eds-figure">{Math.round(weeklyHours)} Hours Logged</span>
                </div>
                {dailyHoursTarget === null ? null : (
                  <div
                    className="eds-meter"
                    role="img"
                    aria-label={`${Math.round(weeklyHours)} of ${(dailyHoursTarget * 5).toFixed(0)} target hours logged`}
                  >
                    <div
                      className="eds-meter-fill"
                      style={{ width: `${Math.min(100, Math.round((weeklyHours / (dailyHoursTarget * 5)) * 100))}%` }}
                    ></div>
                  </div>
                )}
                <div className="eds-week-block eds-week-block--end">
                  <span className="eds-eyebrow">Target</span>
                  <span className="eds-figure">
                    {dailyHoursTarget === null
                      ? "No fixed hours"
                      : `${(dailyHoursTarget * 5).toFixed(0)} Hours`}
                  </span>
                </div>
              </div>
            </section>
          </>
        )}

        {/* Admin-only: System Overview (This Month) */}
        {isAdmin && (
          <section className="eds-card eds-c1-12">
            <div className="eds-card-head">
              <span className="eds-chip"><Icons.Calendar /></span>
              <div className="eds-card-titles">
                <h2 className="eds-card-title">System Overview</h2>
                <p className="eds-card-sub">This Month</p>
              </div>
            </div>
            <div className="eds-card-body">
              <div className="eds-tiles">
                <div className="eds-tile eds-tile--emerald">
                  <span className="eds-tile-chip"><Icons.Operations /></span>
                  <div className="eds-tile-figures">
                    <span className="eds-figure">
                      {teamStats.totalEmps > 0 ? Math.round((teamStats.available / teamStats.totalEmps) * 100) : 0}%
                    </span>
                    <span className="eds-label">Average Attendance</span>
                  </div>
                </div>
                <div className="eds-tile">
                  <span className="eds-tile-chip"><Icons.Clock /></span>
                  <div className="eds-tile-figures">
                    <span className="eds-figure">
                      {teamStats.totalEmps > 0 ? `${(teamStats.available * 9).toLocaleString()}h` : '0h'}
                    </span>
                    <span className="eds-label">Total Working Hours</span>
                  </div>
                </div>
                <div className="eds-tile eds-tile--sky">
                  <span className="eds-tile-chip"><Icons.Calendar /></span>
                  <div className="eds-tile-figures">
                    <span className="eds-figure">{pendingLeaveList.length}</span>
                    <span className="eds-label">Leave Requests</span>
                  </div>
                </div>
                <div className="eds-tile">
                  <span className="eds-tile-chip"><Icons.Payroll /></span>
                  <div className="eds-tile-figures">
                    <span className="eds-figure eds-figure--text">In Progress</span>
                    <span className="eds-label">Payroll Status</span>
                    <span className="eds-sub">
                      {new Date().toLocaleString('en-IN', { month: 'long' })} {new Date().getFullYear()}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        <section className={`eds-card ${hasRole("Admin") ? "eds-c1-8" : "eds-c1-4"}`}>
          <div className="eds-card-head">
            <span className="eds-chip"><Icons.Users /></span>
            <div className="eds-card-titles">
              <h2 className="eds-card-title">Workforce Overview</h2>
            </div>
          </div>
          <div className="eds-card-body">
            <div className="eds-tiles">
              <div className="eds-tile">
                <span className="eds-tile-chip"><Icons.Users /></span>
                <div className="eds-tile-figures">
                  <span className="eds-figure">{String(teamStats.totalEmps).padStart(2, '0')}</span>
                  <span className="eds-label">Total Staff</span>
                  <span className="eds-sub">Active workforce</span>
                </div>
              </div>
              <div className="eds-tile eds-tile--violet">
                <span className="eds-tile-chip"><Icons.Team /></span>
                <div className="eds-tile-figures">
                  <span className="eds-figure">{String(teamStats.totalDepts).padStart(2, '0')}</span>
                  <span className="eds-label">Departments</span>
                  <span className="eds-sub">Operational units</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {!hasRole("Admin") && (
          <section className="eds-card eds-c5-4">
            <div className="eds-card-head">
              <span className="eds-chip"><Icons.Operations /></span>
              <div className="eds-card-titles">
                <h2 className="eds-card-title">Today's Operations</h2>
              </div>
            </div>
            <div className="eds-card-body">
              <div className="eds-tiles">
                <div className="eds-tile eds-tile--emerald">
                  <span className="eds-tile-chip"><Icons.Operations /></span>
                  <div className="eds-tile-figures">
                    <span className="eds-figure">{teamStats.totalEmps > 0 ? Math.round((teamStats.available / teamStats.totalEmps) * 100) : 0}%</span>
                    <span className="eds-label">Attendance Rate</span>
                    <span className="eds-sub">Present today</span>
                  </div>
                </div>
                <div className="eds-tile eds-tile--amber">
                  <span className="eds-tile-chip"><Icons.Clock /></span>
                  <div className="eds-tile-figures">
                    <span className="eds-figure">{String(teamStats.lateToday).padStart(2, '0')}</span>
                    <span className="eds-label">Late Arrivals</span>
                    <span className="eds-sub">Past scheduled time</span>
                  </div>
                </div>
              </div>
            </div>
          </section>
        )}

        <section className="eds-card eds-c9-4">
          <div className="eds-card-head">
            <span className="eds-chip"><Icons.Calendar /></span>
            <div className="eds-card-titles">
              <h2 className="eds-card-title">Upcoming Holidays</h2>
            </div>
            <NavLink to="/calendar" className="eds-action">View all holidays →</NavLink>
          </div>
          {holidays.length > 0 && (
            <div className="eds-card-body">
              <div className="eds-list">
                {holidays.slice(0, 3).map(h => (
                  <div key={h.id} className="eds-row">
                    <div className="eds-row-main">
                      <span className="eds-row-name">{h.name}</span>
                    </div>
                    <span className="eds-date-pill">
                      {new Date(h.date).toLocaleDateString('en-GB', { day: '2-digit', month: '2-digit' }).replace(/\//g, '-')}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </section>

        {/* Fills its grid row: the Celebrations panel beside it is a tall list,
            and a three-tile card left a block of dead space under itself. */}
        <section className="eds-card eds-c1-8 eds-card--fill">
          <div className="eds-card-head">
            <span className="eds-chip"><Icons.Team /></span>
            <div className="eds-card-titles">
              <h2 className="eds-card-title">My Team Status</h2>
              <p className="eds-card-sub">Today</p>
            </div>
            {!isAdmin && <NavLink to="/leave" className="eds-action eds-action--primary">Request Leave</NavLink>}
          </div>
          <div className="eds-card-body">
            <div className="eds-tiles eds-tiles--fill">
              <div className="eds-tile eds-tile--emerald">
                <span className="eds-tile-chip"><Icons.Users /></span>
                <div className="eds-tile-figures">
                  <span className="eds-figure">{String(teamStats.available).padStart(2, '0')}</span>
                  <span className="eds-label">Available</span>
                </div>
              </div>
              <div className="eds-tile eds-tile--sky">
                <span className="eds-tile-chip"><Icons.Calendar /></span>
                <div className="eds-tile-figures">
                  <span className="eds-figure">{String(teamStats.onLeave).padStart(2, '0')}</span>
                  <span className="eds-label">On Leave</span>
                </div>
              </div>
              <div className="eds-tile eds-tile--rose">
                <span className="eds-tile-chip"><Icons.Users /></span>
                <div className="eds-tile-figures">
                  <span className="eds-figure">{String(teamStats.weeklyOff).padStart(2, '0')}</span>
                  <span className="eds-label">Absent</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* Celebrations — tall list panel, aggregate in the footer strip */}
        <section className="eds-card eds-c9-4">
          <div className="eds-card-head">
            <span className="eds-chip"><Icons.Birthday /></span>
            <div className="eds-card-titles">
              <h2 className="eds-card-title">Upcoming Celebrations</h2>
              <p className="eds-card-sub">Next 7 days</p>
            </div>
            <NavLink to="/calendar" className="eds-action">View all →</NavLink>
          </div>
          <div className="eds-card-body">
            {upcomingCelebrations.length === 0 ? (
              <div className="eds-empty">
                <span className="eds-chip"><Icons.Birthday /></span>
                <span className="eds-empty-title">No Celebrations This Week</span>
                <span className="eds-empty-note">Nothing on the calendar for the next 7 days.</span>
              </div>
            ) : (
              <div className="eds-list">
                {upcomingCelebrations.slice(0, 8).map((c) => {
                  const tagText =
                    c.kind === "birthday"
                      ? "🎂 Birthday"
                      : c.kind === "work"
                      ? `💼 ${c.years} yr work`
                      : `💍 ${c.years} yr marriage`;
                  return (
                    <div key={c.key} className="eds-row">
                      <span className="eds-avatar">{c.name[0]}</span>
                      <div className="eds-row-main">
                        <span className="eds-row-name">{c.name}</span>
                        <span className="eds-tag">{tagText}</span>
                      </div>
                      <span className="eds-row-when">{relativeDayLabel(c.date)}</span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
          {upcomingCelebrations.length > 0 && (
            <div className="eds-card-foot">
              <div className="eds-agg">
                <span>🎂 Birthdays <b>{upcomingCelebrations.filter((c) => c.kind === "birthday").length}</b></span>
                <span>💼 Work <b>{upcomingCelebrations.filter((c) => c.kind === "work").length}</b></span>
                <span>💍 Marriage <b>{upcomingCelebrations.filter((c) => c.kind === "marriage").length}</b></span>
              </div>
            </div>
          )}
        </section>

        {isAdminOrHr && (
          <section className="eds-card eds-c1-8">
            <div className="eds-card-head">
              <span className="eds-chip"><Icons.Calendar /></span>
              <div className="eds-card-titles">
                <h2 className="eds-card-title">Pending Leave Requests</h2>
                <p className="eds-card-sub">Recent requests waiting for approval</p>
              </div>
              <NavLink to="/leave-approvals" className="eds-action">View all requests →</NavLink>
            </div>
            <div className="eds-card-body">
              {pendingLeaveList.length === 0 ? (
                <div className="eds-empty">
                  <span className="eds-chip"><Icons.Clock /></span>
                  <span className="eds-empty-title">No pending leave requests</span>
                </div>
              ) : (
                <div className="eds-list">
                  {pendingLeaveList.map(req => (
                    <div key={req.id} className="eds-row">
                      <div className="eds-row-main">
                        <span className="eds-row-name">{req.employee_name || `Emp #${req.employee_id}`}</span>
                      </div>
                      <span className="eds-row-meta">{req.leave_type_name} • {req.total_days} Days</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </section>
        )}

        <section className={`eds-card ${isAdminOrHr ? "eds-c9-4" : "eds-c1-12"}`}>
          <div className="eds-card-head">
            <span className="eds-chip"><Icons.Calendar /></span>
            <div className="eds-card-titles">
              <h2 className="eds-card-title">Upcoming Events</h2>
              <p className="eds-card-sub">Next 7 days</p>
            </div>
            <NavLink to="/calendar" className="eds-action">View all →</NavLink>
          </div>
          <div className="eds-card-body">
            {upcomingEvents.length === 0 ? (
              <div className="eds-empty">
                <span className="eds-chip"><Icons.Calendar /></span>
                <span className="eds-empty-title">No Events This Week</span>
                <span className="eds-empty-note">The calendar is clear for the next 7 days.</span>
              </div>
            ) : (
              <div className="eds-list">
                {upcomingEvents.slice(0, 6).map((e) => (
                  <div key={e.id} className="eds-row" title={e.description || undefined}>
                    <span className="eds-chip"><Icons.Calendar /></span>
                    <div className="eds-row-main">
                      <span className="eds-row-name">{e.title}</span>
                      <span className="eds-tag">
                        {e.event_type}
                        {e.employee_name ? ` • ${e.employee_name}` : ""}
                      </span>
                    </div>
                    <span className="eds-row-when">{relativeDayLabel(e.date)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </section>
      </div>

      <footer className="eds-foot">
        <span>© {new Date().getFullYear()} Softwiz HRMS. All rights reserved.</span>
        <span>Version 1.0.0</span>
      </footer>
    </div>
  );
}
