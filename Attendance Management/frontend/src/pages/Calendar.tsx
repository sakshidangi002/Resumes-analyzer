import { useState, useEffect } from "react";
import { calendar as api, employees as employeesApi } from "../api/client";
import CustomSelect from "../components/CustomSelect";
import { useAuth } from "../auth/AuthContext";
import ConfirmModal from "../components/ConfirmModal";
import { SectionLoader } from "../components/LoadingState";
import GlobalHeaderControls from "../components/GlobalHeaderControls";

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

export default function Calendar() {
  const { hasRole } = useAuth();
  const [holidays, setHolidays] = useState<Holiday[]>([]);
  const [birthdays, setBirthdays] = useState<Birthday[]>([]);
  const [anniversaries, setAnniversaries] = useState<Anniversary[]>([]);
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

  const sortByMonthDay = (isoA: string, isoB: string) => {
    const a = new Date(isoA + "T12:00:00");
    const b = new Date(isoB + "T12:00:00");
    return a.getDate() - b.getDate();
  };

  const loadData = () => {
    const y = new Date().getFullYear();
    const from = `${y}-${String(month).padStart(2, "0")}-01`;
    const to = month === 12 ? `${y}-12-31` : `${y}-${String(month + 1).padStart(2, "0")}-01`;
    setLoading(true);
    Promise.all([
      api.holidays({ from_date: from, to_date: to }),
      api.birthdays(month),
      api.anniversaries(month),
      api.events({ from_date: from, to_date: to }),
    ])
      .then(([h, b, a, ev]) => {
        setHolidays(h.data);
        setBirthdays(b.data);
        setAnniversaries(a.data);
        setEvents(ev.data);
      })
      .catch(() => { })
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
          <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "1rem" }}>
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
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
              gap: "1rem",
              marginBottom: "1.5rem",
            }}
          >


            <div className="card" style={{ borderTop: "4px solid var(--brand-500)", margin: 0 }}>
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
                  <table className="table-modern table-modern--dark">
                    <thead>
                      <tr>
                        <th style={{ width: 70 }}>Day</th>
                        <th>Employee</th>
                        <th style={{ textAlign: "right" }}>Date</th>
                      </tr>
                    </thead>
                    <tbody>
                      {[...birthdays]
                        .sort((a, b) => sortByMonthDay(a.date, b.date))
                        .map((b) => (
                          <tr key={b.employee_id}>
                            <td style={{ fontWeight: 700 }}>{new Date(b.date + "T12:00:00").getDate()}</td>
                            <td style={{ fontWeight: 500 }}>{b.name}</td>
                            <td style={{ textAlign: "right" }}>{formatBirthdayDate(b.date)}</td>
                          </tr>
                        ))}
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
                <table className="table-modern table-modern--dark">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Name</th>
                      {canManageHolidays && <th className="actions-center" style={{ width: 120, paddingRight: "3rem" }}>Actions</th>}
                    </tr>
                  </thead>
                  <tbody>
                    {[...holidays].sort((a, b) => new Date(a.date + "T12:00:00").getTime() - new Date(b.date + "T12:00:00").getTime()).map((h) => (
                      <tr key={h.id}>
                        <td>{formatNiceDate(h.date)}</td>
                        <td>{h.name}</td>
                        {canManageHolidays && (
                          <td className="actions-center">
                            <div className="actions-stack">
                              <button type="button" className="btn btn-danger btn-sm" onClick={() => deleteHoliday(h)}>
                                Delete
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
                <table className="table-modern table-modern--dark">
                  <thead>
                    <tr>
                      <th>Date</th>
                      <th>Title</th>
                      <th>Type</th>
                      <th>Employee</th>
                      <th>Description</th>
                      {canEditEvents && <th className="actions-center" style={{ paddingRight: "4rem" }}>Actions</th>}
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
                          <td style={{ color: (ev as any).is_auto ? "var(--brand-300)" : "inherit" }}>
                            {formatNiceDate(ev.date)}
                          </td>
                          <td style={{ fontWeight: 600 }}>
                            <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                              {ev.title}
                              {(ev as any).is_auto && (
                                <span style={{ fontSize: "0.6rem", padding: "2px 6px", background: "rgba(255,255,255,0.06)", borderRadius: "4px", color: "var(--brand-300)", fontWeight: 700 }}>AUTO</span>
                              )}
                            </div>
                          </td>
                          <td>
                            <span style={{
                              fontSize: "0.7rem",
                              padding: "3px 10px",
                              borderRadius: "999px",
                              background: ev.event_type === "BIRTHDAY" ? "rgba(236, 72, 153, 0.1)" : ev.event_type === "ANNIVERSARY" ? "rgba(var(--brand-rgb) / 0.1)" : "rgba(255,255,255,0.05)",
                              color: ev.event_type === "BIRTHDAY" ? "#f472b6" : ev.event_type === "ANNIVERSARY" ? "var(--brand-300)" : "rgba(255,255,255,0.7)",
                              border: "1px solid currentColor",
                              fontWeight: 700
                            }}>
                              {ev.event_type}
                            </span>
                          </td>
                          <td>{ev.employee_name ?? "—"}</td>
                          <td style={{ fontSize: "0.85rem", opacity: 0.7 }}>{ev.description}</td>
                          {canEditEvents && (
                            <td className="actions-center">
                              {!(ev as any).is_auto && (
                                <div className="actions-stack">
                                  <button type="button" className="btn btn-secondary btn-sm" onClick={() => openEditEvent(ev)}>
                                    Edit
                                  </button>
                                  <button type="button" className="btn btn-danger btn-sm" onClick={() => deleteEvent(ev)}>
                                    Delete
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
