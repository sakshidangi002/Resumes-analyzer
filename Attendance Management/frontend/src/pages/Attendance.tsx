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
}

interface EmployeeInfo {
  id: number;
  employee_code: string;
  first_name: string;
  last_name: string;
  expected_working_hours: number;
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
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const [editCell, setEditCell] = useState<{
    employee_id: number;
    date: string;
    sign_in_time: string;
    sign_out_time: string;
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
          if (empsResult.status === "fulfilled") setEmployees(empsResult.value.data || []);
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

  const openEdit = (employee_id: number, date: string, dialogOverride?: boolean) => {
    let rec = records.find((r) => r.employee_id === employee_id && r.date === date);
    if (dialogOverride) {
      rec = dialogRecords.find((r) => r.employee_id === employee_id && r.date === date);
    }
    setEditCell({
      employee_id,
      date,
      sign_in_time: rec?.sign_in_time || "",
      sign_out_time: rec?.sign_out_time || "",
      status: rec?.status || "PRESENT",
    });
  };


  // Auto-calculate status from time-in/time-out vs expected working hours
  const calcStatusFromTimes = (
    signIn: string,
    signOut: string,
    employeeId: number
  ): { status: string; hoursWorked: number } => {
    const emp = employees.find(e => e.id === employeeId);
    const expected = emp?.expected_working_hours || 9;
    if (!signIn || !signOut) return { status: "ABSENT", hoursWorked: 0 };
    const [ih, im] = signIn.split(":").map(Number);
    const [oh, om] = signOut.split(":").map(Number);
    const inMins = ih * 60 + im;
    const outMins = oh * 60 + om;
    const workedMins = outMins - inMins;
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
    api.adminSet({
      employee_id: editCell.employee_id,
      date: editCell.date,
      sign_in_time: editCell.sign_in_time || null,
      sign_out_time: editCell.sign_out_time || null,
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

  if (!isHrOrAdmin) {
    return (
      <>
        <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h1 className="page-title">My Attendance</h1>
          <GlobalHeaderControls />
        </div>
        <div className="card">
          <MonthlyAttendanceGrid month={month} year={year} setMonth={setMonth} setYear={setYear} records={records} loading={loading} />
        </div>
      </>
    );
  }

  const selectedDateObj = new Date(selectedDate);
  const dayName = selectedDateObj.toLocaleString("en-IN", { weekday: "long" });
  const dayNum = selectedDateObj.getDate();
  const monthName = selectedDateObj.toLocaleString("en-IN", { month: "short" });
  const dayLabelFull = `${dayName}, ${dayNum} ${monthName} ${selectedDateObj.getFullYear()}`;

  const selectedIsWeekend = [0, 6].includes(selectedDateObj.getDay());

  const employeeRows = [...employees]
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
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: "1.5rem", flexWrap: "wrap", gap: "1rem" }}>
          <div style={{ display: "flex", alignItems: "center", gap: "1.5rem" }}>
            <div>
              <h3 style={{ margin: 0, fontSize: "1.25rem", fontWeight: 700 }}>Daily Attendance</h3>
              <div className="text-muted" style={{ fontSize: "0.85rem", marginTop: "2px" }}>{dayLabelFull}</div>
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
                          if (rec?.total_work_hours != null && rec.total_work_hours > 0) {
                            return formatCompactDuration(rec.total_work_hours);
                          }
                          if (rec?.sign_in_time && selectedDate === todayIso && !rec?.sign_out_time) {
                            const [h, m, sec] = rec.sign_in_time.split(':').map(Number);
                            const start = new Date();
                            start.setHours(h, m, sec, 0);
                            if (nowTick > start) {
                              const diffHours = (nowTick.getTime() - start.getTime()) / (1000 * 60 * 60);
                              return (
                                <span style={{ color: "rgb(34, 192, 93)", fontWeight: 600, fontSize: "0.95rem" }}>
                                  {formatCompactDuration(diffHours)}
                                </span>
                              );
                            }
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
                <div className="modal-form-grid">
                  <div className="form-group"><label>Time In</label><input type="time" value={editCell.sign_in_time} onChange={e => {
                    const newIn = e.target.value;
                    if (editCell.sign_out_time) {
                      const { status } = calcStatusFromTimes(newIn, editCell.sign_out_time, editCell.employee_id);
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
                        const { status } = calcStatusFromTimes(editCell.sign_in_time, newOut, editCell.employee_id);
                        setEditCell({ ...editCell, sign_out_time: newOut, status });
                      } else {
                        setEditCell({ ...editCell, sign_out_time: newOut });
                      }
                    }} />
                    {editCell.sign_in_time && editCell.sign_out_time && (() => {
                      const { hoursWorked } = calcStatusFromTimes(editCell.sign_in_time, editCell.sign_out_time, editCell.employee_id);
                      const h = Math.floor(hoursWorked);
                      const m = Math.round((hoursWorked - h) * 60);
                      return hoursWorked > 0 ? (
                        <div style={{ fontSize: '0.75rem', marginTop: '4px', color: '#60a5fa', fontWeight: 600 }}>
                          ⏱ {h}h {m}m worked · status auto-set
                        </div>
                      ) : null;
                    })()}
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
                </div>
                <div className="modal-actions" style={{ display: "flex", justifyContent: "space-between" }}>
                  <button
                    type="button"
                    className="btn"
                    style={{ background: "rgba(239, 68, 68, 0.15)", color: "#ef4444" }}
                    onClick={() => {
                      setEditCell({ ...editCell, status: "", sign_in_time: "", sign_out_time: "" });
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
              {detailsLoading ? (
                <SectionLoader size="sm" />
              ) : (
                <>
                  <div style={{
                    background: "rgba(255,255,255,0.04)",
                    border: "1px solid rgba(255,255,255,0.08)",
                    borderRadius: "10px",
                    padding: "1rem",
                    marginBottom: "1rem",
                    maxHeight: "240px",
                    overflowY: "auto",
                  }}>
                    {(detailsData?.events?.length || 0) === 0 ? (
                      <div style={{ opacity: 0.65, textAlign: "center" }}>No attendance events recorded for this date.</div>
                    ) : (
                      detailsData?.events.map((evt) => (
                        <div key={evt.id} style={{ display: "flex", justifyContent: "space-between", padding: "0.35rem 0", borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                          <span>{formatEventTime12h(evt.event_time)}</span>
                          <span style={{
                            fontWeight: 700,
                            color: evt.event_type === "IN" ? "#22c55e" : "#f59e0b",
                          }}>
                            {evt.event_type}
                          </span>
                        </div>
                      ))
                    )}
                  </div>
                  <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.75rem", fontSize: "0.9rem" }}>
                    <div>First Check-In: <strong>{formatTime12h(detailsData?.sign_in_time)}</strong></div>
                    <div>Last Check-Out: <strong>{formatTime12h(detailsData?.sign_out_time)}</strong></div>
                    <div>Total Working Hours: <strong>{formatCompactDuration(detailsData?.total_work_hours)}</strong></div>
                    <div>Total Break Time: <strong>{formatCompactDuration(detailsData?.total_break_hours)}</strong></div>
                    <div style={{ gridColumn: "1 / -1" }}>
                      Status: <strong>{formatStatusLabel(detailsData?.status || "ABSENT", detailsData?.total_work_hours)}</strong>
                    </div>
                  </div>
                </>
              )}
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
