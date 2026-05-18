import { useState, useEffect } from "react";
import { NavLink } from "react-router-dom";
import { calendar as api } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { SectionLoader } from "../components/LoadingState";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import { formatDate } from "../utils/dateFormatter";

function formatOrdinal(n: number): string {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return n + (s[(v - 20) % 10] || s[v] || s[0]);
}

function formatDisplayDate(iso: string): string {
  return formatDate(iso);
}

interface ReminderBirthday {
  employee_id: number;
  employee_code: string;
  name: string;
  date: string;
}

interface ReminderAnniversary {
  employee_id: number;
  employee_code: string;
  name: string;
  date_of_joining: string;
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

interface RemindersData {
  for_date: string;
  birthdays: ReminderBirthday[];
  work_anniversaries: ReminderAnniversary[];
  events: ReminderEvent[];
}

export default function Notifications() {
  const { hasRole } = useAuth();
  const [forDate, setForDate] = useState("");
  const [data, setData] = useState<RemindersData | null>(null);
  const [loading, setLoading] = useState(true);

  const isAdminOrHr = hasRole("Admin") || hasRole("HR");

  useEffect(() => {
    const tomorrow = new Date();
    tomorrow.setDate(tomorrow.getDate() + 1);
    const defaultDate = tomorrow.toISOString().slice(0, 10);
    if (!forDate) setForDate(defaultDate);
  }, [forDate]);

  useEffect(() => {
    if (!forDate) return;
    setLoading(true);
    api
      .reminders(forDate)
      .then((r) => setData(r.data))
      .catch(() => setData(null))
      .finally(() => setLoading(false));
  }, [forDate]);

  if (!isAdminOrHr) {
    return (
      <div className="card">
        <p className="text-muted">Reminders are available only to Admin and HR.</p>
      </div>
    );
  }

  const total =
    (data?.birthdays?.length ?? 0) + (data?.work_anniversaries?.length ?? 0) + (data?.events?.length ?? 0);

  return (
    <>
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="page-title">Reminders & Notifications</h1>
          <div className="page-subtitle">Upcoming events and reminders</div>
        </div>
        <GlobalHeaderControls />
      </div>
      <div className="card">
        <p className="text-muted" style={{ marginTop: 0 }}>
          View reminders <strong>a day before</strong> the event date so you can plan (e.g. birthdays, work
          anniversaries, internship/training completion). Default date is tomorrow.
        </p>
        <div className="form-group" style={{ maxWidth: "220px" }}>
          <label>Reminders for date</label>
          <input
            type="date"
            value={forDate}
            onChange={(e) => setForDate(e.target.value)}
          />
        </div>
      </div>

      {loading ? (
        <div style={{ padding: "3rem 0" }}><SectionLoader size="md" /></div>
      ) : data ? (
        <>
          <div className="card">
            <h3 style={{ marginTop: 0 }}>
              {formatDisplayDate(data.for_date)}
              {total > 0 && (
                <span className="text-muted" style={{ fontWeight: "normal", fontSize: "0.95rem" }}>
                  {" "}
                  — {total} reminder{total !== 1 ? "s" : ""}
                </span>
              )}
            </h3>

            {data.birthdays && data.birthdays.length > 0 && (
              <section style={{ marginBottom: "1.5rem" }}>
                <h4 style={{ marginBottom: "0.5rem", color: "var(--primary)" }}>Birthdays</h4>
                <ul style={{ margin: 0, paddingLeft: "1.25rem" }}>
                  {data.birthdays.map((b) => (
                    <li key={b.employee_id}>
                      <strong>{b.name}</strong> ({b.employee_code})
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {data.work_anniversaries && data.work_anniversaries.length > 0 && (
              <section style={{ marginBottom: "1.5rem" }}>
                <h4 style={{ marginBottom: "0.5rem", color: "var(--primary)" }}>Work anniversaries</h4>
                <ul style={{ margin: 0, paddingLeft: "1.25rem" }}>
                  {data.work_anniversaries.map((a) => (
                    <li key={a.employee_id}>
                      <strong>{a.name}</strong> ({a.employee_code}) — {formatOrdinal(a.years)} work anniversary
                      (joined {formatDate(a.date_of_joining)})
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {data.events && data.events.length > 0 && (
              <section style={{ marginBottom: "1.5rem" }}>
                <h4 style={{ marginBottom: "0.5rem", color: "var(--primary)" }}>Other events</h4>
                <ul style={{ margin: 0, paddingLeft: "1.25rem" }}>
                  {data.events.map((e) => (
                    <li key={e.id}>
                      <strong>{e.title}</strong>
                      {e.employee_name ? ` — ${e.employee_name}` : ""}
                      {e.description ? ` — ${e.description}` : ""}
                    </li>
                  ))}
                </ul>
              </section>
            )}

            {total === 0 && (
              <p className="text-muted">No birthdays, work anniversaries, or events on this date.</p>
            )}
          </div>
          <div className="card">
            <p className="text-muted" style={{ margin: 0 }}>
              To add events (e.g. 6‑month internship completed, training completed), go to{" "}
              <NavLink to="/calendar">Calendar</NavLink> and create an event with the exact date; you can link it to an
              employee. Reminders will appear here the day before.
            </p>
          </div>
        </>
      ) : (
        <div className="card">
          <p className="text-muted">Could not load reminders.</p>
        </div>
      )}
    </>
  );
}
