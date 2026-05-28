import { useState, useEffect } from "react";
import { calendar as api, employees as employeesApi } from "../api/client";
import CustomSelect from "../components/CustomSelect";
import { useAuth } from "../auth/AuthContext";
import ConfirmModal from "../components/ConfirmModal";
import { SectionLoader } from "../components/LoadingState";
import GlobalHeaderControls from "../components/GlobalHeaderControls";

const Icons = {
  Edit: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
    </svg>
  ),
  Trash: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <line x1="10" y1="11" x2="10" y2="17" />
      <line x1="14" y1="11" x2="14" y2="17" />
    </svg>
  ),
};

interface Holiday {
  id: number;
  date: string;
  name: string;
}

interface Birthday {
  employee_id: number;
  name: string;
  date: string;
}

interface MarriageAnniversary {
  employee_id: number;
  name: string;
  date_of_marriage: string;
}
interface Anniversary {
  employee_id: number;
  name: string;
  date_of_joining: string;
}

interface CalendarEvent {
  id: number;
  title: string;
  date: string;
  event_type: string;
  description?: string | null;
  employee_id?: number | null;
  employee_name?: string | null;
}

/** Axios may expose `data` as a parsed array, a JSON string, or bad proxy shape — normalize to MarriageAnniversary[]. */
function parseMarriageAnniversaryList(raw: unknown): MarriageAnniversary[] {
  let v = raw;
  if (typeof v === "string") {
    try {
      v = JSON.parse(v);
    } catch {
      return [];
    }
  }
  if (!Array.isArray(v)) return [];
  return v
    .filter(
      (row): row is MarriageAnniversary =>
        row != null &&
        typeof row === "object" &&
        typeof (row as MarriageAnniversary).employee_id === "number" &&
        typeof (row as MarriageAnniversary).name === "string" &&
        typeof (row as MarriageAnniversary).date_of_marriage === "string" &&
        /^\d{4}-\d{2}-\d{2}/.test(String((row as MarriageAnniversary).date_of_marriage).trim()),
    )
    .map((row) => ({
      employee_id: row.employee_id,
      name: row.name.trim() || `Employee #${row.employee_id}`,
      date_of_marriage: String(row.date_of_marriage).trim().slice(0, 10),
    }));
}

function parseBirthdayLikeList<
  T extends { employee_id?: unknown; name?: unknown } & Record<string, unknown>,
>(raw: unknown, dateKey: keyof T): T[] {
  let v = raw;
  if (typeof v === "string") {
    try {
      v = JSON.parse(v);
    } catch {
      return [];
    }
  }
  if (!Array.isArray(v)) return [];
  return v.filter((row) => {
    if (row == null || typeof row !== "object") return false;
    const id = (row as T).employee_id;
    const name = (row as T).name;
    const d = (row as T)[dateKey];
    return typeof id === "number" && typeof name === "string" && typeof d === "string" && /^\d{4}-\d{2}-\d{2}/.test(d.trim());
  }) as T[];
}

/** Parse YYYY-MM-DD without UTC shift (fixes invalid / NaN from bad strings). */
function localDayMonthFromIso(iso: string): { day: number; monthDaySort: number } | null {
  const m = /^(\d{4})-(\d{2})-(\d{2})/.exec(String(iso).trim());
  if (!m) return null;
  const day = Number(m[3]);
  const mo = Number(m[2]);
  if (!Number.isFinite(day) || !Number.isFinite(mo)) return null;
  return { day, monthDaySort: mo * 100 + day };
}

function formatBirthdayDateSafe(iso: string): string {
  const p = localDayMonthFromIso(iso);
  if (!p) return "—";
  const d = new Date(2000, Math.floor(p.monthDaySort / 100) - 1, p.day, 12, 0, 0);
  return d.toLocaleDateString("en-IN", { day: "2-digit", month: "short" });
}

function compareMonthDayIso(aIso: string, bIso: string): number {
  const a = localDayMonthFromIso(aIso);
  const b = localDayMonthFromIso(bIso);
  if (!a || !b) return 0;
  return a.monthDaySort - b.monthDaySort;
}

export default function Calendar() {
  const { hasRole } = useAuth();
  const [holidays, setHolidays] = useState<Holiday[]>([]);
  const [birthdays, setBirthdays] = useState<Birthday[]>([]);
  const [anniversaries, setAnniversaries] = useState<Anniversary[]>([]);
  const [marriageAnniversaries, setMarriageAnniversaries] = useState<MarriageAnniversary[]>([]);
  const [events, setEvents] = useState<CalendarEvent[]>([]);
  const [month, setMonth] = useState(new Date().getMonth() + 1);
  const [loading, setLoading] = useState(true);
  const [showEventModal, setShowEventModal] = useState(false);
  const [editingEvent, setEditingEvent] = useState<CalendarEvent | null>(null);
  const [eventForm, setEventForm] = useState({
    title: "",
    date: "",
    event_type: "EVENT",
    description: "",
    employee_id: null as number | null,
  });
  const [employeeList, setEmployeeList] = useState<Array<{ id: number; full_name: string; employee_code: string }>>([]);
  const [confirmDeleteEvent, setConfirmDeleteEvent] = useState<CalendarEvent | null>(null);
  const [confirmDeleteHoliday, setConfirmDeleteHoliday] = useState<Holiday | null>(null);

  const canEditEvents = hasRole("HR");
  const canManageHolidays = hasRole("Admin") || hasRole("HR");
  const [showHolidayModal, setShowHolidayModal] = useState(false);
  const [holidayForm, setHolidayForm] = useState({ date: "", name: "", is_optional: false });

  const formatNiceDate = (iso: string) => {
    const d = new Date(iso + "T12:00:00");
    return d.toLocaleDateString("en-IN", { day: "2-digit", month: "short", year: "numeric" });
  };

  const formatBirthdayDate = (iso: string) => {
    const d = new Date(iso + "T12:00:00");
    return d.toLocaleDateString("en-IN", { day: "2-digit", month: "short" });
  };

  const sortByMonthDay = (isoA: string, isoB: string) => compareMonthDayIso(isoA, isoB);

  const loadData = () => {
    const y = new Date().getFullYear();
    const from = `${y}-${String(month).padStart(2, "0")}-01`;
    const to = month === 12 ? `${y}-12-31` : `${y}-${String(month + 1).padStart(2, "0")}-01`;
    setLoading(true);
    Promise.allSettled([
      api.holidays({ from_date: from, to_date: to }),
      api.birthdays(month),
      api.anniversaries(month),
      api.events({ from_date: from, to_date: to }),
      api.marriageAnniversaries(month),
    ])
      .then(([h, b, a, ev, m]) => {
        setHolidays(
          h.status === "fulfilled" && Array.isArray(h.value.data) ? (h.value.data as Holiday[]) : [],
        );
        setBirthdays(
          b.status === "fulfilled"
            ? parseBirthdayLikeList<Birthday & Record<string, unknown>>(b.value.data, "date")
            : [],
        );
        setAnniversaries(
          a.status === "fulfilled"
            ? parseBirthdayLikeList<Anniversary & Record<string, unknown>>(a.value.data, "date_of_joining")
            : [],
        );
        setEvents(ev.status === "fulfilled" && Array.isArray(ev.value.data) ? ev.value.data : []);
        setMarriageAnniversaries(
          m.status === "fulfilled" ? parseMarriageAnniversaryList(m.value.data) : [],
        );
      })
      .finally(() => setLoading(false));
  };

  useEffect(() => {
    loadData();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [month]);

  useEffect(() => {
    if (canEditEvents) {
      employeesApi.list({ status: "Active" }).then((r) => {
        const list = (r.data || []).map((e: { id: number; first_name: string; last_name: string; employee_code?: string }) => ({
          id: e.id,
          full_name: `${e.first_name || ""} ${e.last_name || ""}`.trim() || "—",
          employee_code: e.employee_code || "",
        }));
        setEmployeeList(list);
      }).catch(() => { });
    }
  }, [canEditEvents]);

  const openAddEvent = () => {
    setEditingEvent(null);
    const today = new Date();
    const defaultDate = `${today.getFullYear()}-${String(month).padStart(2, "0")}-${String(
      today.getDate()
    ).padStart(2, "0")}`;
    setEventForm({
      title: "",
      date: defaultDate,
      event_type: "EVENT",
      description: "",
      employee_id: null,
    });
    setShowEventModal(true);
  };

  const openEditEvent = (ev: CalendarEvent) => {
    setEditingEvent(ev);
    setEventForm({
      title: ev.title,
      date: ev.date,
      event_type: ev.event_type,
      description: ev.description || "",
      employee_id: ev.employee_id ?? null,
    });
    setShowEventModal(true);
  };

  const saveEvent = (e: React.FormEvent) => {
    e.preventDefault();
    const payload = {
      title: eventForm.title,
      date: eventForm.date,
      event_type: eventForm.event_type,
      description: eventForm.description || undefined,
      employee_id: eventForm.employee_id || undefined,
    };
    const req = editingEvent
      ? api.updateEvent(editingEvent.id, payload)
      : api.createEvent(payload);
    req
      .then(() => {
        setShowEventModal(false);
        loadData();
      })
      .catch(() => { });
  };

  const deleteEvent = (ev: CalendarEvent) => {
    setConfirmDeleteEvent(ev);
  };

  const confirmActualDeleteEvent = () => {
    if (!confirmDeleteEvent) return;
    api
      .deleteEvent(confirmDeleteEvent.id)
      .then(() => {
        setConfirmDeleteEvent(null);
        loadData();
      })
      .catch(() => { });
  };

  const openAddHoliday = () => {
    const y = new Date().getFullYear();
    const defaultDate = `${y}-${String(month).padStart(2, "0")}-01`;
    setHolidayForm({ date: defaultDate, name: "", is_optional: false });
    setShowHolidayModal(true);
  };

  const saveHoliday = (e: React.FormEvent) => {
    e.preventDefault();
    api
      .createHoliday({ date: holidayForm.date, name: holidayForm.name, is_optional: holidayForm.is_optional })
      .then(() => {
        setShowHolidayModal(false);
        loadData();
      })
      .catch(() => { });
  };

  const deleteHoliday = (h: Holiday) => {
    setConfirmDeleteHoliday(h);
  };

  const confirmActualDeleteHoliday = () => {
    if (!confirmDeleteHoliday) return;
    api
      .deleteHoliday(confirmDeleteHoliday.id)
      .then(() => {
        setConfirmDeleteHoliday(null);
        loadData();
      })
      .catch(() => { });
  };

  const monthName = new Date(2000, month - 1).toLocaleString("default", { month: "long" });
  const prevMonth = () => setMonth(m => m === 1 ? 12 : m - 1);
  const nextMonth = () => setMonth(m => m === 12 ? 1 : m + 1);

  return (
    <>
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="page-title">Calendar</h1>
          <div className="page-subtitle">Holidays, events, birthdays, and anniversaries</div>
        </div>
        <GlobalHeaderControls />
      </div>
      {loading ? (
        <div style={{ padding: "4rem 0" }}><SectionLoader size="md" /></div>
      ) : (
        <>

          {/* <div className="card">
            <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
              <label style={{ fontWeight: 700, margin: 0, color: "#fff" }}>Month:</label>
              <div style={{ width: "200px" }}>
                <CustomSelect
                  value={month}
                  onChange={(val) => setMonth(Number(val))}
                  options={[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map(m => ({
                    value: m,
                    label: new Date(2000, m - 1).toLocaleString("default", { month: "long" })
                  }))}
                />
              </div>
            </div>
          </div> */}
          <div style={{ display: "flex", justifyContent: "flex-end", alignItems: "center", marginBottom: "1rem", gap: "0.5rem" }}>
            <button
              type="button"
              className="btn-icon-action"
              onClick={prevMonth}
              title="Previous month"
              style={{ borderRadius: "8px" }}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="15 18 9 12 15 6"></polyline>
              </svg>
            </button>
            <span style={{ fontWeight: 700, fontSize: "1rem", minWidth: "110px", textAlign: "center", color: "var(--text-primary, #fff)" }}>
              {monthName}
            </span>
            <button
              type="button"
              className="btn-icon-action"
              onClick={nextMonth}
              title="Next month"
              style={{ borderRadius: "8px" }}
            >
              <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <polyline points="9 18 15 12 9 6"></polyline>
              </svg>
            </button>
          </div>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
              gap: "1rem",
              marginBottom: "1.5rem",
            }}
          >


            <div className="card" style={{ borderTop: "4px solid #22c55e", margin: 0 }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "baseline" }}>
                <h3 style={{ marginTop: 0, marginBottom: 6 }}>Joining dates</h3>
                <span className="text-muted" style={{ fontSize: "0.9rem" }}>
                  {anniversaries.length} this month
                </span>
              </div>
              <p className="text-muted" style={{ marginTop: 0, fontSize: "0.8rem" }}>
                Work anniversaries occurring in {new Date(2000, month - 1).toLocaleString('default', { month: 'long' })}.
              </p>
              {anniversaries.length === 0 ? (
                <p className="text-muted">No joining dates in this month.</p>
              ) : (
                <div className="table-wrap table-wrap--dark">
                  <table className="table-modern table-modern--dark" style={{ tableLayout: 'fixed', width: '100%' }}>
                    <thead>
                      <tr>
                        <th style={{ width: '20%', textAlign: 'center' }}>Day</th>
                        <th style={{ width: '45%', textAlign: 'center' }}>Employee</th>
                        <th style={{ width: '35%', textAlign: 'center' }}>Date of joining</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...anniversaries]
                        .sort((a, b) => sortByMonthDay(a.date_of_joining, b.date_of_joining))
                        .map((a) => (
                          <tr key={a.employee_id}>
                            <td style={{ fontWeight: 700, textAlign: 'center' }}>{new Date(a.date_of_joining + "T12:00:00").getDate()}</td>
                            <td style={{ fontWeight: 500, textAlign: 'center' }}>{a.name}</td>
                            <td style={{ textAlign: 'center' }}>{formatNiceDate(a.date_of_joining)}</td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
            <div className="card" style={{ borderTop: "4px solid #22c55e", margin: 0 }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "baseline" }}>
                <h3 style={{ marginTop: 0, marginBottom: 6 }}>Birthdays</h3>
                <span className="text-muted" style={{ fontSize: "0.9rem" }}>
                  {birthdays.length} this month
                </span>
              </div>
              <p className="text-muted" style={{ marginTop: 0, fontSize: "0.8rem" }}>
                Automatically synced from employee profiles for {new Date(2000, month - 1).toLocaleString('default', { month: 'long' })}.
              </p>
              {birthdays.length === 0 ? (
                <p className="text-muted">No birthdays in this month.</p>
              ) : (
                <div className="table-wrap table-wrap--dark">
                  <table className="table-modern table-modern--dark" style={{ tableLayout: 'fixed', width: '100%' }}>
                    <thead>
                      <tr>
                        <th style={{ width: '20%', textAlign: 'center' }}>Day</th>
                        <th style={{ width: '45%', textAlign: 'center' }}>Employee</th>
                        <th style={{ width: '35%', textAlign: 'center' }}>Date</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...birthdays]
                        .sort((a, b) => sortByMonthDay(a.date, b.date))
                        .map((b) => (
                          <tr key={b.employee_id}>
                            <td style={{ fontWeight: 700, textAlign: 'center' }}>{new Date(b.date + "T12:00:00").getDate()}</td>
                            <td style={{ fontWeight: 500, textAlign: 'center' }}>{b.name}</td>
                            <td style={{ textAlign: 'center' }}>{formatBirthdayDate(b.date)}</td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
            <div className="card" style={{ borderTop: "4px solid #22c55e", margin: 0 }}>
              <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "baseline" }}>
                <h3 style={{ marginTop: 0, marginBottom: 6 }}>Marriage Anniversary</h3>
                <span className="text-muted" style={{ fontSize: "0.9rem" }}>
                  {marriageAnniversaries.length} this month
                </span>
              </div>
              <p className="text-muted" style={{ marginTop: 0, fontSize: "0.8rem" }}>
                Automatically synced from employee profiles for {new Date(2000, month - 1).toLocaleString('default', { month: 'long' })}.
              </p>
              {marriageAnniversaries.length === 0 ? (
                <p className="text-muted">No Marriage Anniversary in this month.</p>
              ) : (
                <div className="table-wrap table-wrap--dark">
                  <table className="table-modern table-modern--dark" style={{ tableLayout: 'fixed', width: '100%' }}>
                    <thead>
                      <tr>
                        <th style={{ width: '20%', textAlign: 'center' }}>Day</th>
                        <th style={{ width: '45%', textAlign: 'center' }}>Employee</th>
                        <th style={{ width: '35%', textAlign: 'center' }}>Date of marriage</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...marriageAnniversaries]
                        .sort((a, b) => sortByMonthDay(a.date_of_marriage, b.date_of_marriage))
                        .map((mRow) => {
                          const dm = localDayMonthFromIso(mRow.date_of_marriage);
                          return (
                            <tr key={`${mRow.employee_id}-${mRow.date_of_marriage}`}>
                              <td style={{ textAlign: "center" }}>{dm?.day ?? "—"}</td>
                              <td style={{ fontWeight: 500, textAlign: "center" }}>{mRow.name}</td>
                              <td style={{ fontWeight: "500", textAlign: "center" }}>
                                {formatBirthdayDateSafe(mRow.date_of_marriage)}
                              </td>
                            </tr>
                          );
                        })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>



            {/* <div className="card" style={{ borderTop: "4px solid var(--brand-500)", margin: 0 }}>
          <div style={{ display: "flex", justifyContent: "space-between", gap: "0.75rem", alignItems: "baseline" }}>
            <h3 style={{ marginTop: 0, marginBottom: 6 }}>Joining dates</h3>
            <span className="text-muted" style={{ fontSize: "0.9rem" }}>
              {anniversaries.length} this month
            </span>
          </div>
          <p className="text-muted" style={{ marginTop: 0, fontSize: "0.8rem" }}>
            Work anniversaries occurring in {new Date(2000, month - 1).toLocaleString('default', { month: 'long' })}.
          </p>
          {anniversaries.length === 0 ? (
            <p className="text-muted">No joining dates in this month.</p>
          ) : (
            <div className="table-wrap table-wrap--dark">
              <table className="table-modern table-modern--dark">
                <thead>
                  <tr>
                    <th style={{ width: 70 }}>Day</th>
                    <th>Employee</th>
                    <th style={{ textAlign: "right" }}>Date of joining</th>
                  </tr>
                </thead>
                <tbody>
                  {[...anniversaries]
                    .sort((a, b) => sortByMonthDay(a.date_of_joining, b.date_of_joining))
                    .map((a) => (
                      <tr key={a.employee_id}>
                        <td style={{ fontWeight: 700 }}>{new Date(a.date_of_joining + "T12:00:00").getDate()}</td>
                        <td style={{ fontWeight: 500 }}>{a.name}</td>
                        <td style={{ textAlign: "right" }}>{formatNiceDate(a.date_of_joining)}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
            </div>
          )}
        </div> */}
          </div>

          <div className="card">
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "0.75rem", flexWrap: "wrap", marginBottom: "1.5rem" }}>
              <h3 style={{ marginTop: 0, marginBottom: 0 }}>Holidays</h3>
              {canManageHolidays && (
                <button type="button" className="btn btn-primary btn-uniform" onClick={openAddHoliday}>
                  Add holiday
                </button>
              )}
            </div>
            {holidays.length > 0 ? (
              <div className="table-wrap table-wrap--dark">
                <table className="table-modern table-modern--dark" style={{ tableLayout: 'fixed', width: '100%' }}>
                  <thead>
                    <tr>
                      <th style={{ width: canManageHolidays ? '35%' : '40%', textAlign: 'center' }}>Date</th>
                      <th style={{ width: canManageHolidays ? '45%' : '60%', textAlign: 'center' }}>Name</th>
                      {canManageHolidays && <th style={{ width: '20%', textAlign: 'center' }}>Actions</th>}
                    </tr>
                  </thead>
                  <tbody>
                    {[...holidays].sort((a, b) => new Date(a.date + "T12:00:00").getTime() - new Date(b.date + "T12:00:00").getTime()).map((h) => (
                      <tr key={h.id}>
                        <td style={{ textAlign: 'center' }}>{formatNiceDate(h.date)}</td>
                        <td style={{ textAlign: 'center' }}>{h.name}</td>
                        {canManageHolidays && (
                          <td style={{ textAlign: 'center' }}>
                            <div style={{ display: 'flex', justifyContent: 'center' }}>
                              <button
                                type="button"
                                className="btn btn-icon-action btn-icon-action--danger"
                                onClick={() => deleteHoliday(h)}
                                title="Delete Holiday"
                                aria-label="Delete"
                              >
                                <Icons.Trash />
                              </button>
                            </div>
                          </td>
                        )}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            ) : (
              <p className="text-muted">No holidays in this month.</p>
            )}
          </div>
          <div className="card" style={{ marginTop: "1.5rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", gap: "0.75rem", flexWrap: "wrap", marginBottom: "1.5rem" }}>
              <h3 style={{ marginTop: 0, marginBottom: 0 }}>Events & Special Days</h3>
              {canEditEvents && (
                <button type="button" className="btn btn-primary btn-uniform" onClick={openAddEvent}>
                  Add Event
                </button>
              )}
            </div>
            {events.length === 0 ? (
              <p className="text-muted">No events in this month.</p>
            ) : (
              <div className="table-wrap table-wrap--dark">
                <table className="table-modern table-modern--dark" style={{ tableLayout: 'fixed', width: '100%' }}>
                  <thead>
                    <tr>
                      <th style={{ width: canEditEvents ? '15%' : '18%', textAlign: 'center' }}>Date</th>
                      <th style={{ width: canEditEvents ? '20%' : '22%', textAlign: 'center' }}>Title</th>
                      <th style={{ width: canEditEvents ? '12%' : '15%', textAlign: 'center' }}>Type</th>
                      <th style={{ width: canEditEvents ? '18%' : '20%', textAlign: 'center' }}>Employee</th>
                      <th style={{ width: canEditEvents ? '20%' : '25%', textAlign: 'center' }}>Description</th>
                      {canEditEvents && <th style={{ width: '15%', textAlign: 'center' }}>Actions</th>}
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      ...events,
                      ...birthdays.map((b) => ({
                        id: -b.employee_id,
                        title: `Birthday: ${b.name}`,
                        date: b.date,
                        event_type: "BIRTHDAY",
                        employee_id: b.employee_id,
                        employee_name: b.name,
                        description: "Auto-generated from employee profile",
                        is_auto: true,
                      })),
                      ...anniversaries.map((a) => ({
                        id: -(a.employee_id + 1000000),
                        title: `Work Anniversary: ${a.name}`,
                        date: a.date_of_joining,
                        event_type: "ANNIVERSARY",
                        employee_id: a.employee_id,
                        employee_name: a.name,
                        description: "Auto-generated from employee profile",
                        is_auto: true,
                      })),
                    ]
                      .sort((a, b) => new Date(b.date + "T12:00:00").getTime() - new Date(a.date + "T12:00:00").getTime())
                      .map((ev) => (
                        <tr key={ev.id} style={{ opacity: (ev as any).is_auto ? 0.85 : 1 }}>
                          <td style={{ color: (ev as any).is_auto ? "var(--brand-300)" : "inherit", textAlign: 'center' }}>
                            {formatNiceDate(ev.date)}
                          </td>
                          <td style={{ fontWeight: 600, textAlign: 'center' }}>
                            <div style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", gap: "0.5rem" }}>
                              {ev.title}
                              {(ev as any).is_auto && (
                                <span style={{ fontSize: "0.6rem", padding: "2px 6px", background: "rgba(255,255,255,0.06)", borderRadius: "4px", color: "var(--brand-300)", fontWeight: 700 }}>AUTO</span>
                              )}
                            </div>
                          </td>
                          <td style={{ textAlign: 'center' }}>
                            <span style={{
                              fontSize: "0.7rem",
                              padding: "3px 10px",
                              borderRadius: "999px",
                              background: ev.event_type === "BIRTHDAY" ? "rgba(236, 72, 153, 0.1)" : ev.event_type === "ANNIVERSARY" ? "rgba(var(--brand-rgb) / 0.1)" : "rgba(255,255,255,0.05)",
                              color: ev.event_type === "BIRTHDAY" ? "#f472b6" : ev.event_type === "ANNIVERSARY" ? "var(--brand-300)" : "rgba(255,255,255,0.7)",
                              border: "1px solid currentColor",
                              fontWeight: 700,
                              display: "inline-block"
                            }}>
                              {ev.event_type}
                            </span>
                          </td>
                          <td style={{ textAlign: 'center' }}>{ev.employee_name ?? "—"}</td>
                          <td style={{ fontSize: "0.85rem", opacity: 0.7, textAlign: 'center' }}>{ev.description}</td>
                          {canEditEvents && (
                            <td style={{ textAlign: 'center' }}>
                              {!(ev as any).is_auto && (
                                <div style={{ display: 'inline-flex', gap: '0.5rem', justifyContent: 'center' }}>
                                  <button
                                    type="button"
                                    className="btn btn-icon-action btn-icon-action--neutral"
                                    onClick={() => openEditEvent(ev)}
                                    title="Edit Event"
                                    aria-label="Edit"
                                  >
                                    <Icons.Edit />
                                  </button>
                                  <button
                                    type="button"
                                    className="btn btn-icon-action btn-icon-action--danger"
                                    onClick={() => deleteEvent(ev)}
                                    title="Delete Event"
                                    aria-label="Delete"
                                  >
                                    <Icons.Trash />
                                  </button>
                                </div>
                              )}
                            </td>
                          )}
                        </tr>
                      ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}


      {showHolidayModal && (
        <div className="modal-backdrop" onClick={() => setShowHolidayModal(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 420 }}>
            <h3 style={{ marginTop: 0 }}>Add holiday</h3>
            <form onSubmit={saveHoliday} className="modal-stack">
              <div className="form-group">
                <label>Date</label>
                <input
                  type="date"
                  required
                  value={holidayForm.date}
                  onChange={(e) => setHolidayForm((f) => ({ ...f, date: e.target.value }))}
                  style={{ maxWidth: "100%" }}
                />
              </div>
              <div className="form-group">
                <label>Holiday name</label>
                <input
                  required
                  value={holidayForm.name}
                  onChange={(e) => setHolidayForm((f) => ({ ...f, name: e.target.value }))}
                  placeholder="e.g. Holi, Diwali, Company Holiday"
                  style={{ maxWidth: "100%" }}
                />
              </div>
              <label className="modal-checkbox-group">
                <input
                  type="checkbox"
                  checked={holidayForm.is_optional}
                  onChange={(e) => setHolidayForm((f) => ({ ...f, is_optional: e.target.checked }))}
                />
                Optional holiday
              </label>
              <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.5rem", marginTop: "0.5rem" }}>

                <button type="submit" className="btn btn-primary btn-uniform">
                  Save
                </button>
                <button type="button" className="btn btn-secondary btn-uniform" onClick={() => setShowHolidayModal(false)} style={{ color: "#ef4444", background: "rgba(239, 68, 68, 0.15)", }}>
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {showEventModal && (
        <div className="modal-backdrop">
          <div className="modal" style={{ maxWidth: 480 }}>
            <h3 style={{ marginTop: 0 }}>{editingEvent ? "Edit Event" : "Add Event"}</h3>
            <form onSubmit={saveEvent} className="modal-stack">
              <div className="form-group">
                <label>Title</label>
                <input
                  required
                  value={eventForm.title}
                  onChange={(e) => setEventForm((f) => ({ ...f, title: e.target.value }))}
                  style={{ maxWidth: "100%" }}
                />
              </div>
              <div className="form-group">
                <label>Date</label>
                <input
                  type="date"
                  required
                  value={eventForm.date}
                  onChange={(e) => setEventForm((f) => ({ ...f, date: e.target.value }))}
                  style={{ maxWidth: "100%" }}
                />
              </div>
              <div className="form-group">
                <label>Type</label>
                <CustomSelect
                  value={eventForm.event_type}
                  onChange={(val) => setEventForm((f) => ({ ...f, event_type: val }))}
                  options={[
                    { value: "EVENT", label: "Event" },
                    { value: "SPECIAL_DAY", label: "Special Day" },
                    { value: "ANNOUNCEMENT", label: "Announcement" },
                    { value: "INTERNSHIP_END", label: "6 month / Internship completed" },
                    { value: "TRAINING_COMPLETED", label: "Training completed" }
                  ]}
                />
              </div>
              <div className="form-group">
                <label>Employee (optional)</label>
                <CustomSelect
                  value={String(eventForm.employee_id ?? "")}
                  onChange={(val) =>
                    setEventForm((f) => ({
                      ...f,
                      employee_id: val ? Number(val) : null,
                    }))
                  }
                  placeholder="— None —"
                  options={[
                    { value: "", label: "— None —" },
                    ...employeeList.map((emp) => ({
                      value: String(emp.id),
                      label: `${emp.full_name} ${emp.employee_code ? `(${emp.employee_code})` : ""}`
                    }))
                  ]}
                />
              </div>
              <div className="form-group">
                <label>Description</label>
                <textarea
                  rows={3}
                  value={eventForm.description}
                  onChange={(e) => setEventForm((f) => ({ ...f, description: e.target.value }))}
                  style={{ maxWidth: "100%" }}
                />
              </div>
              <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.5rem", marginTop: "0.5rem" }}>
                <button type="button" className="btn btn-secondary btn-uniform" onClick={() => setShowEventModal(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary btn-uniform">
                  Save
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      <ConfirmModal
        isOpen={!!confirmDeleteEvent}
        onClose={() => setConfirmDeleteEvent(null)}
        onConfirm={confirmActualDeleteEvent}
        title="Are you absolutely sure?"
        message={
          confirmDeleteEvent ? (
            <>
              You are about to delete event <strong>{confirmDeleteEvent.title}</strong> on <strong>{confirmDeleteEvent.date}</strong>.
            </>
          ) : (
            ""
          )
        }
        confirmText="Yes, Delete Event"
      />

      <ConfirmModal
        isOpen={!!confirmDeleteHoliday}
        onClose={() => setConfirmDeleteHoliday(null)}
        onConfirm={confirmActualDeleteHoliday}
        title="Are you absolutely sure?"
        message={
          confirmDeleteHoliday ? (
            <>
              You are about to delete holiday <strong>{confirmDeleteHoliday.name}</strong> on <strong>{confirmDeleteHoliday.date}</strong>.
            </>
          ) : (
            ""
          )
        }
        confirmText="Yes, Delete Holiday"
      />
    </>
  );
}
