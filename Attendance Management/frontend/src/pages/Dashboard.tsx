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

const Icons = {
  Clock: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
  ),
  Calendar: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>
  ),
  Users: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><path d="M23 21v-2a4 4 0 0 0-3-3.87"></path><path d="M16 3.13a4 4 0 0 1 0 7.75"></path></svg>
  ),
  Operations: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
  ),
  Birthday: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>
  ),
  Team: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"></path><circle cx="8.5" cy="7" r="4"></circle><polyline points="17 11 19 13 23 9"></polyline></svg>
  ),
  ThreeDots: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="1"></circle><circle cx="12" cy="5" r="1"></circle><circle cx="12" cy="19" r="1"></circle></svg>
  ),
  Star: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="currentColor" stroke="none"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
  )
};

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
  const expectedHoursPerDay = Number(empDetails?.expected_working_hours || 9);

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
              if (r.status === 'HALF_DAY') personalPresent += 0.5;
              else personalPresent++;
            }
          } else if (r.status === 'ON_LEAVE' || r.status === 'PAID_LEAVE') {
            leaveCount++;
            if (isMe) personalLeave++;
          }
        });

        // Calculate Working Days
        let workingDaysCount = 0;
        const currentMonthHolidays = holidays.filter(h => {
          const hd = new Date(h.date);
          return hd.getMonth() === now.getMonth() && hd.getFullYear() === now.getFullYear() && hd.getDate() <= now.getDate();
        });
        for (let d = 1; d <= now.getDate(); d++) {
          const dateObj = new Date(now.getFullYear(), now.getMonth(), d);
          const dayOfWeek = dateObj.getDay();
          const isHoliday = currentMonthHolidays.some(h => new Date(h.date).getDate() === d);
          if (dayOfWeek !== 0 && dayOfWeek !== 6 && !isHoliday) workingDaysCount++;
        }

        const weekendMinutes = weekendsPassed * expectedHoursPerDay * 60;
        const personalTotalMinutes = personalMinutes + weekendMinutes;
        const totalRequiredMinutes = 30 * expectedHoursPerDay * 60;

        const hoursWorked = Math.floor(personalTotalMinutes / 60);
        const minsWorked = Math.round(personalTotalMinutes % 60);
        const reqHours = Math.floor(totalRequiredMinutes / 60);

        setAttendanceStats({
          present: personalPresent,
          leave: personalLeave,
          percentage: workingDaysCount > 0 ? Math.round((personalPresent / workingDaysCount) * 100) : 0,
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
    <div className="dash-container">
      {/* Header Section */}
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="page-title" style={{ textTransform: 'capitalize' }}>Welcome, {user?.username?.split(' ')[0] || "User"} 👋</h1>
          <div className="page-subtitle">{user?.designation || "Admin"}</div>
        </div>
        <GlobalHeaderControls />
      </div>

      <div className="dash-grid" style={{ marginTop: '-1rem' }}>
        {/* Row 1 for Admin: 3 Separate Cards | Row 1 for Others: Clock + Overviews */}
        <div className="dash-card dash-card--clock">
          <div className="card-top">
            <div className="card-title-group">
              <Icons.Clock />
              <span>Current Time</span>
            </div>
          </div>
          <div className="clock-body">
            <p className="clock-date">{todayFormatted.fullDate}</p>
            <div className="clock-time">
              {todayFormatted.timeStr.split(' ')[0]}
              <span className="clock-ampm">{todayFormatted.timeStr.split(' ')[1]}</span>
            </div>
          </div>
          <div className="clock-footer">
            <div className="quote-pill">
              <Icons.Star />
              <span>Have a productive day!</span>
            </div>
          </div>
        </div>

        {hasRole("Admin") ? (
          <>
            <div className="dash-card">
              <div className="card-top">
                <div className="card-title-group">
                  <Icons.Operations />
                  <span>Workforce Attendance</span>
                </div>
              </div>
              <div className="overview-stack">
                <div className="info-box info-box--green">
                  <div className="box-icon"><Icons.Operations /></div>
                  <div className="box-content">
                    <span className="box-val">{teamStats.totalEmps > 0 ? Math.round((teamStats.available / teamStats.totalEmps) * 100) : 0}%</span>
                    <span className="box-lab">Attendance Rate</span>
                    <span className="box-sub">{teamStats.available} Employees Present</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="dash-card">
              <div className="card-top">
                <div className="card-title-group">
                  <Icons.Clock />
                  <span>Absence & Punctuality</span>
                </div>
              </div>
              <div className="overview-stack">
                <div className="info-box info-box--red" style={{ marginBottom: '0.75rem' }}>
                  <div className="box-icon" style={{ color: "#ffa500" }}><Icons.Clock /></div>
                  <div className="box-content">
                    <span className="box-val" style={{ color: "rgb(251, 146, 60)" }}>{String(teamStats.lateToday).padStart(2, '0')}</span>
                    <span className="box-lab">Late Arrivals</span>
                  </div>
                </div>
                <div className="info-box info-box--blue">
                  <div className="box-icon" style={{ background: 'rgba(59, 130, 246, 0.15)', color: '#60a5fa' }}>
                    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#60a5fa" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <path d="M19 21v-2a4 4 0 0 0-4-4H9a4 4 0 0 0-4 4v2"></path>
                      <circle cx="12" cy="7" r="4"></circle>
                    </svg>
                  </div>
                  <div className="box-content">
                    <span className="box-val" style={{ color: '#60a5fa' }}>{String(teamStats.onLeave).padStart(2, '0')}</span>
                    <span className="box-lab">Employees On Leave</span>
                  </div>
                </div>
              </div>
            </div>
          </>
        ) : (
          <>
            <div className="dash-card dash-card--monthly">
              <div className="card-top">
                <div className="card-title-group">
                  <Icons.Calendar />
                  <span>Monthly Overview</span>
                </div>
              </div>
              <div className="monthly-body">
                <div className="progress-circle-wrap">
                  <svg viewBox="0 0 36 36" className="progress-circle">
                    <path className="circle-bg" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                    <path className="circle-fill" strokeDasharray={`${attendanceStats.percentage}, 100`} d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" />
                  </svg>
                  <div className="progress-text">{attendanceStats.percentage}%</div>
                </div>
                <div className="stats-group">
                  <div className="stat-item">
                    <span className="stat-val">{String(Math.round(attendanceStats.present)).padStart(2, '0')} Days</span>
                    <span className="stat-lab">Present</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-val">{String(Math.round(attendanceStats.leave)).padStart(2, '0')} Days</span>
                    <span className="stat-lab">Leave</span>
                  </div>
                </div>
                <div className="divider-v"></div>
                <div className="stats-group">
                  <div className="stat-item">
                    <span className="stat-val stat-val--blue">{attendanceStats.totalHours}</span>
                    <span className="stat-lab">Worked Hours</span>
                  </div>
                  <div className="stat-item">
                    <span className="stat-val stat-val--blue">{attendanceStats.requiredHours}</span>
                    <span className="stat-lab">Required Hours</span>
                  </div>
                </div>
              </div>
            </div>
            <div className="dash-card dash-card--weekly">
              <div className="card-top">
                <div className="card-title-group">
                  <Icons.Calendar />
                  <span>Weekly Overview</span>
                </div>
              </div>
              <div className="weekly-grid">
                {weekDates.map((d) => {
                  const dt = new Date(d);
                  const isToday = d === todayISO;
                  return (
                    <div key={d} className={`week-day ${isToday ? 'active' : ''}`}>
                      <span className="day-name">{dt.toLocaleDateString(undefined, { weekday: 'short' })}</span>
                      <span className="day-num">{dt.getDate()}</span>
                      {isToday && null}
                    </div>
                  );
                })}
              </div>
              <div className="weekly-summary" style={{ marginTop: '2.5rem', paddingTop: '1.5rem', borderTop: '1px solid rgba(255,255,255,0.06)', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <div>
                  <div style={{ fontSize: '0.75rem', opacity: 0.5, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 700 }}>Weekly Performance</div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 800, color: 'rgb(34 192 93)', marginTop: '4px' }}>{Math.round(weeklyHours)} Hours Logged</div>
                </div>
                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '0.75rem', opacity: 0.5, textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: 700 }}>Target</div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 800, color: 'rgba(255,255,255,0.9)', marginTop: '4px' }}>{(expectedHoursPerDay * 5).toFixed(0)} Hours</div>
                </div>
              </div>
            </div>
          </>
        )}

        {/* Admin-only: System Overview (This Month) */}
        {isAdmin && (
          <div className="dash-card dash-card--full">
            <div className="card-top" style={{ marginBottom: '1.25rem' }}>
              <div className="card-title-group">
                <Icons.Calendar />
                <span>System Overview <span style={{ fontWeight: 500, opacity: 0.65, fontSize: '0.9em' }}>(This Month)</span></span>
              </div>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: '1rem' }}>
              {/* Average Attendance */}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.6rem', padding: '1.1rem 0.75rem', borderRadius: '14px', background: 'rgba(34,197,94,0.07)', border: '1px solid rgba(34,197,94,0.15)' }}>
                <div style={{ width: 44, height: 44, borderRadius: '50%', background: 'rgba(34,197,94,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#4ade80' }}>
                  <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>
                </div>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', opacity: 0.65 }}>Average Attendance</span>
                <span style={{ fontSize: '2rem', fontWeight: 900, lineHeight: 1, color: '#4ade80' }}>
                  {teamStats.totalEmps > 0 ? Math.round((teamStats.available / teamStats.totalEmps) * 100) : 0}%
                </span>
              </div>
              {/* Total Working Hours */}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.6rem', padding: '1.1rem 0.75rem', borderRadius: '14px', background: 'rgba(59,130,246,0.07)', border: '1px solid rgba(59,130,246,0.15)' }}>
                <div style={{ width: 44, height: 44, borderRadius: '50%', background: 'rgba(59,130,246,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#60a5fa' }}>
                  <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"></circle><polyline points="12 6 12 12 16 14"></polyline></svg>
                </div>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', opacity: 0.65 }}>Total Working Hours</span>
                <span style={{ fontSize: '2rem', fontWeight: 900, lineHeight: 1, color: '#60a5fa' }}>
                  {teamStats.totalEmps > 0 ? `${(teamStats.available * 9).toLocaleString()}h` : '0h'}
                </span>
              </div>
              {/* Leave Requests */}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.6rem', padding: '1.1rem 0.75rem', borderRadius: '14px', background: 'rgba(251,146,60,0.07)', border: '1px solid rgba(251,146,60,0.15)' }}>
                <div style={{ width: 44, height: 44, borderRadius: '50%', background: 'rgba(223, 51, 8, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#fb923c' }}>
                  <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="rgb(251, 146, 60)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>
                </div>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', opacity: 0.65 }}>Leave Requests</span>
                <span style={{ fontSize: '2rem', fontWeight: 900, lineHeight: 1, color: '#FB923C' }}>
                  {pendingLeaveList.length}
                </span>
              </div>
              {/* Payroll Status */}
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.6rem', padding: '1.1rem 0.75rem', borderRadius: '14px', background: 'rgba(168,85,247,0.07)', border: '1px solid rgba(168,85,247,0.15)' }}>
                <div style={{ width: 44, height: 44, borderRadius: '50%', background: 'rgba(168,85,247,0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#c084fc' }}>
                  <svg xmlns="http://www.w3.org/2000/svg" width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
                </div>
                <span style={{ fontSize: '0.75rem', fontWeight: 700, textTransform: 'uppercase', letterSpacing: '0.06em', opacity: 0.65 }}>Payroll Status</span>
                <span style={{ fontSize: '1.15rem', fontWeight: 900, lineHeight: 1, color: '#c084fc' }}>In Progress</span>
                <span style={{ fontSize: '0.8rem', color: 'rgba(255,255,255,0.5)', fontWeight: 600 }}>
                  {new Date().toLocaleString('en-IN', { month: 'long' })} {new Date().getFullYear()}
                </span>
              </div>
            </div>
          </div>
        )}

        {/* Row 2: Workforce Overview & Holidays */}
        <div className={`dash-card ${hasRole("Admin") ? "dash-card--wide" : ""}`}>
          <div className="card-top">
            <div className="card-title-group">
              <Icons.Users />
              <span>Workforce Overview</span>
            </div>
          </div>
          <div className="overview-stack" style={hasRole("Admin") ? { flexDirection: 'row', gap: '1.5rem' } : {}}>
            <div className="info-box info-box--blue" style={hasRole("Admin") ? { flex: 1 } : {}}>
              <div className="box-icon"><Icons.Users /></div>
              <div className="box-content">
                <span className="box-val">{String(teamStats.totalEmps).padStart(2, '0')}</span>
                <span className="box-lab">Total Staff</span>
                <span className="box-sub">Active workforce</span>
              </div>
            </div>
            <div className="info-box info-box--purple" style={hasRole("Admin") ? { flex: 1 } : {}}>
              <div className="box-icon"><Icons.Team /></div>
              <div className="box-content">
                <span className="box-val">{String(teamStats.totalDepts).padStart(2, '0')}</span>
                <span className="box-lab">Departments</span>
                <span className="box-sub">Operational units</span>
              </div>
            </div>
          </div>
        </div>

        {!hasRole("Admin") && (
          <div className="dash-card">
            <div className="card-top">
              <div className="card-title-group">
                <Icons.Operations />
                <span>Today's Operations</span>
              </div>
            </div>
            <div className="overview-stack">
              <div className="info-box info-box--green">
                <div className="box-icon"><Icons.Operations /></div>
                <div className="box-content">
                  <span className="box-val">{teamStats.totalEmps > 0 ? Math.round((teamStats.available / teamStats.totalEmps) * 100) : 0}%</span>
                  <span className="box-lab">Attendance Rate</span>
                  <span className="box-sub">Present today</span>
                </div>
              </div>
              <div className="info-box info-box--red">
                <div className="box-icon" style={{ color: "rgb(251, 146, 60)" }}><Icons.Clock /></div>
                <div className="box-content">
                  <span className="box-val">{String(teamStats.lateToday).padStart(2, '0')}</span>
                  <span className="box-lab">Late Arrivals</span>
                  <span className="box-sub">Past scheduled time</span>
                </div>
              </div>
            </div>
          </div>
        )}

        <div className="dash-card">
          <div className="card-top">
            <div className="card-title-group">
              <Icons.Calendar />
              <span>Upcoming Holidays</span>
            </div>
          </div>
          <div className="holiday-list">
            {holidays.slice(0, 3).map(h => (
              <div key={h.id} className="holiday-item">
                <span className="holiday-name">{h.name}</span>
                <span className="holiday-date">{new Date(h.date).toLocaleDateString('en-GB', { day: '2-digit', month: '2-digit' }).replace(/\//g, '-')}</span>
              </div>
            ))}
            <NavLink to="/calendar" className="view-link">View all holidays →</NavLink>
          </div>
        </div>

        {/* Row 3: Team Status, Birthdays */}
        <div className="dash-card dash-card--wide">
          <div className="card-top">
            <div className="card-title-group">
              <Icons.Team />
              <span>My Team Status</span>
            </div>
          </div>
          <div className="team-status-row">
            <div className="status-item">
              <div className="status-icon"><Icons.Users /></div>
              <div className="status-data">
                <span className="status-val" style={{ color: "rgb(34,192,93)" }}>{String(teamStats.available).padStart(2, '0')}</span>
                <span className="status-lab">Available</span>
              </div>
            </div>
            <div className="status-item">
              <div className="status-icon"><Icons.Calendar /></div>
              <div className="status-data">
                <span className="status-val" style={{ color: "rgb(31 74 118)" }}>{String(teamStats.onLeave).padStart(2, '0')}</span>
                <span className="status-lab">On Leave</span>
              </div>
            </div>
            <div className="status-item">
              <div className="status-icon"><Icons.Users /></div>
              <div className="status-data">
                <span className="status-val" style={{ color: "#da1f1f" }}>{String(teamStats.weeklyOff).padStart(2, '0')}</span>
                <span className="status-lab">Absent</span>
              </div>
            </div>
            {!isAdmin && <NavLink to="/leave" className="btn btn-primary btn-lg">Request Leave</NavLink>}
          </div>
        </div>

        <div className="dash-card">
          <div className="card-top">
            <div className="card-title-group">
              <Icons.Birthday />
              <div style={{ display: "flex", flexDirection: "column" }}>
                <span style={{ fontSize: "1.05rem", fontWeight: 700 }}>Upcoming Celebrations</span>
                <span style={{ fontSize: "0.75rem", opacity: 0.5, fontWeight: 500 }}>Next 7 days</span>
              </div>
            </div>
            <NavLink to="/calendar" className="view-link">View all →</NavLink>
          </div>
          {upcomingCelebrations.length > 0 && (
            <div
              style={{
                display: "flex",
                gap: "0.5rem",
                marginBottom: "0.75rem",
                flexWrap: "wrap",
              }}
            >
              {[
                { label: "Birthdays", count: upcomingCelebrations.filter((c) => c.kind === "birthday").length, color: "rgba(236, 72, 153, 0.18)", border: "rgba(236, 72, 153, 0.45)", emoji: "🎂" },
                { label: "Work", count: upcomingCelebrations.filter((c) => c.kind === "work").length, color: "rgba(34, 197, 94, 0.18)", border: "rgba(34, 197, 94, 0.45)", emoji: "💼" },
                { label: "Marriage", count: upcomingCelebrations.filter((c) => c.kind === "marriage").length, color: "rgba(168, 85, 247, 0.18)", border: "rgba(168, 85, 247, 0.45)", emoji: "💍" },
              ].map((s) => (
                <div
                  key={s.label}
                  style={{
                    flex: 1,
                    minWidth: 80,
                    padding: "0.5rem 0.6rem",
                    borderRadius: "10px",
                    background: s.color,
                    border: `1px solid ${s.border}`,
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "flex-start",
                    gap: "2px",
                  }}
                >
                  <span style={{ fontSize: "0.7rem", fontWeight: 600, opacity: 0.85, textTransform: "uppercase", letterSpacing: "0.04em" }}>
                    {s.emoji} {s.label}
                  </span>
                  <span style={{ fontSize: "1.2rem", fontWeight: 800, color: "rgba(255,255,255,0.92)", lineHeight: 1 }}>{s.count}</span>
                </div>
              ))}
            </div>
          )}
          <div className="birthday-body" style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            {upcomingCelebrations.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon"><Icons.Birthday /></div>
                <p>No Celebrations This Week</p>
                <span>Nothing on the calendar for the next 7 days.</span>
              </div>
            ) : (
              upcomingCelebrations.slice(0, 8).map((c) => {
                const tagColor =
                  c.kind === "birthday"
                    ? "rgba(236, 72, 153, 0.18)"
                    : c.kind === "work"
                    ? "rgba(34, 197, 94, 0.18)"
                    : "rgba(168, 85, 247, 0.18)";
                const tagBorder =
                  c.kind === "birthday"
                    ? "rgba(236, 72, 153, 0.45)"
                    : c.kind === "work"
                    ? "rgba(34, 197, 94, 0.45)"
                    : "rgba(168, 85, 247, 0.45)";
                const tagText =
                  c.kind === "birthday"
                    ? "🎂 Birthday"
                    : c.kind === "work"
                    ? `💼 ${c.years} yr work`
                    : `💍 ${c.years} yr marriage`;
                return (
                  <div
                    key={c.key}
                    className="birthday-item"
                    style={{ display: "flex", alignItems: "center", gap: "0.75rem", padding: "0.55rem 0.5rem", borderRadius: "10px", background: "rgba(255,255,255,0.03)" }}
                  >
                    <div className="avatar" style={{ flexShrink: 0 }}>{c.name[0]}</div>
                    <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
                      <span style={{ fontWeight: 600, fontSize: "0.92rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{c.name}</span>
                      <span
                        style={{
                          fontSize: "0.7rem",
                          fontWeight: 600,
                          padding: "1px 8px",
                          borderRadius: "8px",
                          background: tagColor,
                          border: `1px solid ${tagBorder}`,
                          color: "rgba(255,255,255,0.88)",
                          alignSelf: "flex-start",
                          marginTop: "2px",
                        }}
                      >
                        {tagText}
                      </span>
                    </div>
                    <span style={{ fontSize: "0.78rem", fontWeight: 600, opacity: 0.7, whiteSpace: "nowrap" }}>{relativeDayLabel(c.date)}</span>
                  </div>
                );
              })
            )}
          </div>
        </div>

        {/* Row 4: Pending Leaves (admin/HR, 2 cols) + Upcoming Events (1 col) */}
        {isAdminOrHr && (
          <div className="dash-card dash-card--wide">
            <div className="card-top">
              <div className="card-title-group">
                <Icons.Calendar />
                <div style={{ display: 'flex', flexDirection: 'column' }}>
                  <span style={{ fontSize: '1.1rem', fontWeight: 700 }}>Pending Leave Requests</span>
                  <span style={{ fontSize: '0.85rem', opacity: 0.5, fontWeight: 500 }}>Recent requests waiting for approval</span>
                </div>
              </div>
              <NavLink to="/leave-approvals" className="view-link">View all requests →</NavLink>
            </div>
            <div className="requests-container" style={{ marginTop: '0.5rem' }}>
              {pendingLeaveList.length === 0 ? (
                <div className="no-data-bar" style={{ background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', padding: '1.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.75rem', color: 'rgba(255,255,255,0.4)' }}>
                  <Icons.Clock />
                  <span>No pending leave requests</span>
                </div>
              ) : (
                <div style={{ display: 'grid', gap: '1rem' }}>
                  {pendingLeaveList.map(req => (
                    <div key={req.id} className="request-card" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)', borderRadius: '12px', padding: '1rem' }}>
                      {/* Request card content */}
                      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <div style={{ fontWeight: 600 }}>{req.employee_name || `Emp #${req.employee_id}`}</div>
                        <div style={{ fontSize: '0.85rem', opacity: 0.7 }}>{req.leave_type_name} • {req.total_days} Days</div>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        <div className={`dash-card ${isAdminOrHr ? "" : "dash-card--full"}`}>
          <div className="card-top">
            <div className="card-title-group">
              <Icons.Calendar />
              <div style={{ display: "flex", flexDirection: "column" }}>
                <span style={{ fontSize: "1.05rem", fontWeight: 700 }}>Upcoming Events</span>
                <span style={{ fontSize: "0.75rem", opacity: 0.5, fontWeight: 500 }}>Next 7 days</span>
              </div>
            </div>
            <NavLink to="/calendar" className="view-link">View all →</NavLink>
          </div>
          <div className="birthday-body" style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            {upcomingEvents.length === 0 ? (
              <div className="empty-state">
                <div className="empty-icon"><Icons.Calendar /></div>
                <p>No Events This Week</p>
                <span>The calendar is clear for the next 7 days.</span>
              </div>
            ) : (
              upcomingEvents.slice(0, 6).map((e) => (
                <div
                  key={e.id}
                  className="birthday-item"
                  style={{ display: "flex", alignItems: "center", gap: "0.75rem", padding: "0.55rem 0.5rem", borderRadius: "10px", background: "rgba(255,255,255,0.03)" }}
                  title={e.description || undefined}
                >
                  <div
                    className="avatar"
                    style={{ flexShrink: 0, background: "rgba(59, 130, 246, 0.2)" }}
                  >
                    <Icons.Calendar />
                  </div>
                  <div style={{ flex: 1, display: "flex", flexDirection: "column", minWidth: 0 }}>
                    <span style={{ fontWeight: 600, fontSize: "0.92rem", overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                      {e.title}
                    </span>
                    <span
                      style={{
                        fontSize: "0.7rem",
                        fontWeight: 600,
                        padding: "1px 8px",
                        borderRadius: "8px",
                        background: "rgba(59,130,246,0.18)",
                        border: "1px solid rgba(59,130,246,0.45)",
                        color: "rgba(255,255,255,0.88)",
                        alignSelf: "flex-start",
                        marginTop: "2px",
                      }}
                    >
                      {e.event_type}
                      {e.employee_name ? ` • ${e.employee_name}` : ""}
                    </span>
                  </div>
                  <span style={{ fontSize: "0.78rem", fontWeight: 600, opacity: 0.7, whiteSpace: "nowrap" }}>
                    {relativeDayLabel(e.date)}
                  </span>
                </div>
              ))
            )}
          </div>
        </div>
      </div>

      <footer className="dash-footer">
        <span>© {new Date().getFullYear()} Softwiz HRMS. All rights reserved.</span>
        <span>Version 1.0.0</span>
      </footer>
    </div>
  );
}
