import { useState, useEffect } from "react";
import { calendar as api, employees as employeesApi } from "../api/client";
import CustomSelect from "../components/CustomSelect";
import { useAuth } from "../auth/AuthContext";
import ConfirmModal from "../components/ConfirmModal";
import { SectionLoader } from "../components/LoadingState";
import GlobalHeaderControls from "../components/GlobalHeaderControls";

/* 24-box strokes, round caps, currentColor. Size comes from the chip or button
   that holds them (.eds-chip 16px, .eds-iconbtn 15px, .eds-monthnav 13px). */
const Icons = {
  Edit: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 4H6a2 2 0 0 0-2 2v12a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-5"></path>
      <path d="M18.5 2.5a2.1 2.1 0 0 1 3 3L12 15l-4 1 1-4z"></path>
    </svg>
  ),
  Trash: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6M14 11v6" />
    </svg>
  ),
  Building: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <path d="M3 21V8l9-5 9 5v13" />
      <line x1="3" y1="21" x2="21" y2="21" />
      <rect x="9" y="13" width="6" height="8" />
    </svg>
  ),
  Person: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <path d="M19 21v-1.5A4.5 4.5 0 0 0 14.5 15h-5A4.5 4.5 0 0 0 5 19.5V21" />
      <circle cx="12" cy="8" r="4" />
    </svg>
  ),
  UserCheck: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <path d="M17 20v-1.5A3.5 3.5 0 0 0 13.5 15h-6A3.5 3.5 0 0 0 4 18.5V20" />
      <circle cx="10.5" cy="8" r="3.5" />
      <polyline points="17 11 19 13 22 9" />
    </svg>
  ),
  Calendar: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4.5" width="18" height="16.5" rx="2.5" />
      <line x1="3" y1="10" x2="21" y2="10" />
      <line x1="8" y1="2.5" x2="8" y2="6" />
      <line x1="16" y1="2.5" x2="16" y2="6" />
    </svg>
  ),
  Tasks: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="9 11 12 14 20 6" />
      <path d="M20 12v7a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h9" />
    </svg>
  ),
  ChevronLeft: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="15 18 9 12 15 6" />
    </svg>
  ),
  ChevronRight: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="9 18 15 12 9 6" />
    </svg>
  ),
};

/** Identity tint for a person avatar, stable per employee. */
const AVATAR_TINTS = ["eds-avatar--blue", "eds-avatar--green", "eds-avatar--purple", "eds-avatar--rose", ""];

function initialsOf(name: string): string {
  const parts = (name || "").trim().split(/\s+/).filter(Boolean);
  if (!parts.length) return "?";
  return parts.slice(0, 2).map((p) => p[0]).join("").toUpperCase();
}

/** Accent for an event type pill: birthdays rose, work anniversaries sky,
 *  everything else (SPECIAL_DAY, custom types) violet. */
function eventTone(type: string): string {
  if (type === "BIRTHDAY") return " eds-type--rose";
  if (type === "ANNIVERSARY") return " eds-type--sky";
  return " eds-type--violet";
}

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

/**
 * Weekday that a recurring annual date falls on THIS year ("Mon", "Tue", …).
 *
 * The "Day" column used to print the day-of-month number, which just repeated
 * the date already shown in the next column. What people actually plan around
 * is the weekday — and it has to be computed for the CURRENT year, not the
 * stored one: the stored year is the person's birth year or their joining
 * year, whose weekday is irrelevant. The lists themselves are loaded for the
 * current year (see loadData).
 */
function weekdayForThisYear(iso: string): string {
  const parsed = localDayMonthFromIso(iso);
  if (!parsed) return "—";
  const monthIndex = Math.floor(parsed.monthDaySort / 100) - 1;
  const year = new Date().getFullYear();
  // Clamp to the month's real length so 29 Feb in a non-leap year shows the
  // 28 Feb weekday instead of silently rolling into March and showing that one.
  const lastDayOfMonth = new Date(year, monthIndex + 1, 0).getDate();
  const day = Math.min(parsed.day, lastDayOfMonth);
  // Midday avoids any DST/timezone edge shifting the date across midnight.
  return new Date(year, monthIndex, day, 12, 0, 0)
    .toLocaleDateString("en-IN", { weekday: "short" });
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
    <div className="eds">
      <header className="eds-topbar">
        <div>
          <h1 className="eds-title">Calendar</h1>
          <p className="eds-subtitle">Holidays, events, birthdays, and anniversaries</p>
        </div>
        <GlobalHeaderControls />
      </header>
      {loading ? (
        <div style={{ padding: "4rem 0" }}><SectionLoader size="md" /></div>
      ) : (
        <div className="eds-page">
          <div className="eds-actionbar">
            <div className="eds-monthnav">
              <button type="button" onClick={prevMonth} title="Previous month" aria-label="Previous month">
                <Icons.ChevronLeft />
              </button>
              <span className="eds-monthnav-label">{monthName}</span>
              <button type="button" onClick={nextMonth} title="Next month" aria-label="Next month">
                <Icons.ChevronRight />
              </button>
            </div>
          </div>

          <div className="eds-cards-3">
            <section className="eds-card">
              <div className="eds-card-head">
                <span className="eds-chip eds-chip--violet"><Icons.Building /></span>
                <div className="eds-card-titles">
                  <h2 className="eds-card-title">Joining dates</h2>
                  <p className="eds-card-sub">
                    Work anniversaries occurring in {new Date(2000, month - 1).toLocaleString('default', { month: 'long' })}.
                  </p>
                </div>
                <span className="eds-count-chip"><b>{anniversaries.length}</b>this month</span>
              </div>
              {anniversaries.length === 0 ? (
                <div className="eds-well">
                  <div className="eds-empty--dashed">
                    <span className="eds-chip"><Icons.Building /></span>
                    <span className="eds-empty-title">No joining dates in this month.</span>
                  </div>
                </div>
              ) : (
                <div className="eds-minitable">
                  <div className="eds-minitable-head">
                    <span>Day</span><span>Employee</span><span style={{ textAlign: "right" }}>Date of joining</span>
                  </div>
                  {[...anniversaries]
                    .sort((a, b) => sortByMonthDay(a.date_of_joining, b.date_of_joining))
                    .map((a) => (
                      <div className="eds-minitable-row" key={a.employee_id}>
                        <span className="eds-minitable-day">{weekdayForThisYear(a.date_of_joining)}</span>
                        <div className="eds-minitable-who">
                          <span className={`eds-avatar eds-avatar--sm ${AVATAR_TINTS[a.employee_id % AVATAR_TINTS.length]}`}>
                            {initialsOf(a.name)}
                          </span>
                          <span>{a.name}</span>
                        </div>
                        <span className="eds-minitable-date">{formatNiceDate(a.date_of_joining)}</span>
                      </div>
                    ))}
                </div>
              )}
            </section>

            <section className="eds-card">
              <div className="eds-card-head">
                <span className="eds-chip eds-chip--rose"><Icons.Person /></span>
                <div className="eds-card-titles">
                  <h2 className="eds-card-title">Birthdays</h2>
                  <p className="eds-card-sub">
                    Automatically synced from employee profiles for {new Date(2000, month - 1).toLocaleString('default', { month: 'long' })}.
                  </p>
                </div>
                <span className={`eds-count-chip${birthdays.length > 0 ? " eds-count-chip--rose" : ""}`}>
                  <b>{birthdays.length}</b>this month
                </span>
              </div>
              {birthdays.length === 0 ? (
                <div className="eds-well">
                  <div className="eds-empty--dashed">
                    <span className="eds-chip"><Icons.Person /></span>
                    <span className="eds-empty-title">No birthdays in this month.</span>
                  </div>
                </div>
              ) : (
                <div className="eds-minitable">
                  <div className="eds-minitable-head">
                    <span>Day</span><span>Employee</span><span style={{ textAlign: "right" }}>Date</span>
                  </div>
                  {[...birthdays]
                    .sort((a, b) => sortByMonthDay(a.date, b.date))
                    .map((b) => (
                      <div className="eds-minitable-row" key={b.employee_id}>
                        <span className="eds-minitable-day">{weekdayForThisYear(b.date)}</span>
                        <div className="eds-minitable-who">
                          <span className={`eds-avatar eds-avatar--sm ${AVATAR_TINTS[b.employee_id % AVATAR_TINTS.length]}`}>
                            {initialsOf(b.name)}
                          </span>
                          <span>{b.name}</span>
                        </div>
                        <span className="eds-minitable-date">{formatBirthdayDate(b.date)}</span>
                      </div>
                    ))}
                </div>
              )}
            </section>

            <section className="eds-card">
              <div className="eds-card-head">
                <span className="eds-chip eds-chip--sky"><Icons.UserCheck /></span>
                <div className="eds-card-titles">
                  <h2 className="eds-card-title">Marriage Anniversary</h2>
                  <p className="eds-card-sub">
                    Automatically synced from employee profiles for {new Date(2000, month - 1).toLocaleString('default', { month: 'long' })}.
                  </p>
                </div>
                <span className="eds-count-chip"><b>{marriageAnniversaries.length}</b>this month</span>
              </div>
              {marriageAnniversaries.length === 0 ? (
                <div className="eds-well">
                  <div className="eds-empty--dashed">
                    <span className="eds-chip"><Icons.UserCheck /></span>
                    <span className="eds-empty-title">No Marriage Anniversary in this month.</span>
                  </div>
                </div>
              ) : (
                <div className="eds-minitable">
                  <div className="eds-minitable-head">
                    <span>Day</span><span>Employee</span><span style={{ textAlign: "right" }}>Date of marriage</span>
                  </div>
                  {[...marriageAnniversaries]
                    .sort((a, b) => sortByMonthDay(a.date_of_marriage, b.date_of_marriage))
                    .map((mRow) => (
                      <div className="eds-minitable-row" key={`${mRow.employee_id}-${mRow.date_of_marriage}`}>
                        <span className="eds-minitable-day">{weekdayForThisYear(mRow.date_of_marriage)}</span>
                        <div className="eds-minitable-who">
                          <span className={`eds-avatar eds-avatar--sm ${AVATAR_TINTS[mRow.employee_id % AVATAR_TINTS.length]}`}>
                            {initialsOf(mRow.name)}
                          </span>
                          <span>{mRow.name}</span>
                        </div>
                        <span className="eds-minitable-date">{formatBirthdayDateSafe(mRow.date_of_marriage)}</span>
                      </div>
                    ))}
                </div>
              )}
            </section>
          </div>

          <section className="eds-card">
            <div className="eds-card-head">
              <span className="eds-chip eds-chip--emerald"><Icons.Calendar /></span>
              <div className="eds-card-titles">
                <h2 className="eds-card-title">Holidays</h2>
              </div>
              {canManageHolidays && (
                <button type="button" className="eds-action eds-action--info" onClick={openAddHoliday}>
                  Add holiday
                </button>
              )}
            </div>
            {holidays.length > 0 ? (
              <div className="eds-table-wrap">
                <table className="eds-table eds-table--auto">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Name</th>
                      {canManageHolidays && <th className="is-actions">Actions</th>}
                    </tr>
                  </thead>
                  <tbody>
                    {[...holidays].sort((a, b) => new Date(a.date + "T12:00:00").getTime() - new Date(b.date + "T12:00:00").getTime()).map((h) => (
                      <tr key={h.id}>
                        <td className="eds-cell-dim">{formatNiceDate(h.date)}</td>
                        <td className="eds-cell-strong">{h.name}</td>
                        {canManageHolidays && (
                          <td>
                            <div className="eds-rowactions">
                              <button
                                type="button"
                                className="eds-iconbtn eds-iconbtn--del"
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
              <div className="eds-well">
                <div className="eds-empty--dashed">
                  <span className="eds-chip"><Icons.Calendar /></span>
                  <span className="eds-empty-title">No holidays in this month.</span>
                </div>
              </div>
            )}
          </section>

          <section className="eds-card">
            <div className="eds-card-head">
              <span className="eds-chip eds-chip--amber"><Icons.Tasks /></span>
              <div className="eds-card-titles">
                <h2 className="eds-card-title">Events &amp; Special Days</h2>
              </div>
              {canEditEvents && (
                <button type="button" className="eds-action eds-action--info" onClick={openAddEvent}>
                  Add Event
                </button>
              )}
            </div>
            {events.length === 0 ? (
              <div className="eds-well">
                <div className="eds-empty--dashed">
                  <span className="eds-chip"><Icons.Tasks /></span>
                  <span className="eds-empty-title">No events in this month.</span>
                </div>
              </div>
            ) : (
              <div className="eds-table-wrap">
                <table className="eds-table eds-table--auto">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Title</th>
                      <th>Type</th>
                      <th>Employee</th>
                      <th>Description</th>
                      {canEditEvents && <th className="is-actions">Actions</th>}
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
                        <tr key={ev.id}>
                          <td className="eds-cell-dim">{formatNiceDate(ev.date)}</td>
                          <td>
                            <span className="eds-title-inline">
                              <span>{ev.title}</span>
                              {(ev as any).is_auto && <span className="eds-auto">AUTO</span>}
                            </span>
                          </td>
                          <td>
                            <span className={`eds-type${eventTone(ev.event_type)}`}>{ev.event_type}</span>
                          </td>
                          <td className="eds-cell-strong">{ev.employee_name ?? "—"}</td>
                          <td className="eds-cell-dim eds-cell-clip">{ev.description || "—"}</td>
                          {canEditEvents && (
                            <td>
                              {(ev as any).is_auto ? (
                                <div className="eds-rowactions"><span className="eds-dash">—</span></div>
                              ) : (
                                <div className="eds-rowactions">
                                  <button
                                    type="button"
                                    className="eds-iconbtn eds-iconbtn--edit"
                                    onClick={() => openEditEvent(ev)}
                                    title="Edit Event"
                                    aria-label="Edit"
                                  >
                                    <Icons.Edit />
                                  </button>
                                  <button
                                    type="button"
                                    className="eds-iconbtn eds-iconbtn--del"
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
          </section>
        </div>
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
    </div>
  );
}
