import { useEffect, useState } from "react";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import { audit, type AuditLogRow } from "../api/client";

const Icons = {
  Search: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="7.5" />
      <line x1="21" y1="21" x2="16.7" y2="16.7" />
    </svg>
  ),
  Funnel: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polygon points="22 3 2 3 10 12.5 10 19 14 21 14 12.5 22 3" />
    </svg>
  ),
  Calendar: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="5" width="18" height="16" rx="2" />
      <line x1="3" y1="10" x2="21" y2="10" />
      <line x1="8" y1="3" x2="8" y2="7" />
      <line x1="16" y1="3" x2="16" y2="7" />
    </svg>
  ),
  Ledger: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <rect x="4" y="3" width="16" height="18" rx="2.5" />
      <line x1="8" y1="8" x2="16" y2="8" />
      <line x1="8" y1="12" x2="16" y2="12" />
      <line x1="8" y1="16" x2="13" y2="16" />
    </svg>
  ),
};

/** Audit actions tint by outcome: creates/logins emerald, updates sky,
 *  deletes and failures rose, exports/reveals amber, everything else neutral. */
function actionTone(action: string): string {
  const a = (action || "").toUpperCase();
  if (a.includes("FAIL") || a.includes("DELETE") || a.includes("LOCKOUT")) return " eds-code--rose";
  if (a.includes("CREATE") || a.includes("SUCCESS") || a.includes("APPROVE")) return " eds-code--emerald";
  if (a.includes("UPDATE") || a.includes("PATCH") || a.includes("LOGIN")) return " eds-code--sky";
  if (a.includes("REVEAL") || a.includes("EXPORT") || a.includes("VIEW")) return " eds-code--amber";
  return "";
}

const MONTHS = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

const CURRENT_YEAR = new Date().getFullYear();

/** Years offered in the picker, newest first.
 *
 *  Five is enough to cover any audit history this system can have while keeping
 *  the list short enough to scan. A year with no activity is not an error -- the
 *  empty state names the period, so an empty result reads as "nothing happened
 *  then" rather than as a broken filter. */
const YEARS = Array.from({ length: 5 }, (_, i) => CURRENT_YEAR - i);

/** The naive ISO bounds selecting a whole year, or one month within it.
 *
 *  `audit_logs.created_at` is a NAIVE UTC column, and the rows are rendered with
 *  `new Date(...).toLocaleString()`, which parses a suffix-less string as local
 *  time -- so the month a row DISPLAYS under is the month of its stored digits.
 *  Building the bounds from those same digits is therefore what makes the filter
 *  agree with the table the operator is looking at. Sending a UTC-converted or
 *  offset-bearing string would quietly disagree with it near month ends.
 *
 *  `month` is 1-12, or 0 for the whole year. */
function periodBounds(year: number, month: number): { from_date: string; to_date: string } {
  const firstMonth = month || 1;
  const lastMonth = month || 12;
  // Day 0 of the NEXT month is the last day of this one -- the only form that
  // needs no leap-year or 30/31 special-casing.
  const lastDay = new Date(Date.UTC(year, lastMonth, 0)).getUTCDate();
  const pad = (n: number) => String(n).padStart(2, "0");
  return {
    from_date: `${year}-${pad(firstMonth)}-01T00:00:00`,
    to_date: `${year}-${pad(lastMonth)}-${pad(lastDay)}T23:59:59.999`,
  };
}

export default function AuditLogs() {
  const [logs, setLogs] = useState<AuditLogRow[]>([]);
  const [action, setAction] = useState("");
  // Both empty means "all time". Held as strings because that is what a <select>
  // gives back; 0 / "" is the "all" sentinel for each.
  const [month, setMonth] = useState("");
  const [year, setYear] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await audit.list({
        limit: 500,
        action: action.trim() || undefined,
        // No year chosen means "all time", so send no bounds at all rather than
        // a range that would have to guess how far back to reach.
        ...(year ? periodBounds(Number(year), Number(month || 0)) : {}),
      });
      setLogs(response.data || []);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Could not load audit logs.");
    } finally {
      setLoading(false);
    }
  };

  // Also covers the initial load -- a separate mount-only effect would fire a
  // second, identical request alongside this one on first render.
  useEffect(() => { void load(); }, [month, year]);

  return (
    <div className="eds">
      <header className="eds-topbar">
        <div>
          <h1 className="eds-title">Audit Logs</h1>
          <p className="eds-subtitle">Track important HRMS and account activity</p>
        </div>
        <GlobalHeaderControls />
      </header>

      <div className="eds-page">
        <div className="eds-controls">
          <label className="eds-search eds-search--grow">
            <Icons.Search />
            <input
              value={action}
              onChange={(e) => setAction(e.target.value)}
              placeholder="Filter by action"
              onKeyDown={(e) => { if (e.key === "Enter") void load(); }}
            />
          </label>
          <span className="eds-selectfield">
            <Icons.Calendar />
            <select
              className="eds-select-inline"
              value={month}
              onChange={(e) => {
                setMonth(e.target.value);
                // Picking a month with no year would be ambiguous, so the
                // current year is assumed -- which is what somebody choosing
                // "August" almost always means.
                if (e.target.value && !year) setYear(String(CURRENT_YEAR));
              }}
              aria-label="Filter by month"
            >
              <option value="">All months</option>
              {MONTHS.map((name, i) => (
                <option key={name} value={i + 1}>{name}</option>
              ))}
            </select>
          </span>
          <span className="eds-selectfield">
            <select
              className="eds-select-inline"
              value={year}
              onChange={(e) => {
                setYear(e.target.value);
                // "All years" cannot carry a month: a month with no year has no
                // bounds to build. Clear it rather than leaving a selection
                // showing that is not being applied.
                if (!e.target.value) setMonth("");
              }}
              aria-label="Filter by year"
            >
              <option value="">All years</option>
              {YEARS.map((y) => (
                <option key={y} value={y}>{y}</option>
              ))}
            </select>
          </span>
          {year ? (
            <button
              type="button"
              className="eds-action"
              onClick={() => { setMonth(""); setYear(""); }}
            >
              All time
            </button>
          ) : null}
          <button type="button" className="eds-action eds-action--go" onClick={() => void load()}>
            <Icons.Funnel />
            Filter
          </button>
        </div>

        {error && <div className="alert alert-error">{error}</div>}

        <section className="eds-card">
          {loading ? (
            <div className="eds-empty--card">
              <span className="eds-empty-tile"><Icons.Ledger /></span>
              <span>Loading audit logs…</span>
            </div>
          ) : logs.length === 0 ? (
            <div className="eds-empty--card">
              <span className="eds-empty-tile"><Icons.Ledger /></span>
              <span>
                {year
                  ? `No audit activity in ${month ? `${MONTHS[Number(month) - 1]} ` : ""}${year}.`
                  : "No audit activity found."}
              </span>
            </div>
          ) : (
            <div className="eds-table-wrap">
              <table className="eds-table eds-table--auto">
                <thead>
                  <tr>
                    <th>Date</th>
                    <th>User</th>
                    <th>Action</th>
                    <th>Entity</th>
                    <th>Details</th>
                    <th className="is-actions">IP</th>
                  </tr>
                </thead>
                <tbody>{logs.map((log) => (
                  <tr key={log.id}>
                    <td className="eds-cell-mid">{new Date(log.created_at).toLocaleString()}</td>
                    <td className="eds-cell-strong">{log.username || `User #${log.user_id ?? "-"}`}</td>
                    <td><span className={`eds-code${actionTone(log.action)}`}>{log.action}</span></td>
                    <td className="eds-cell-mid">{log.entity_type || "-"}{log.entity_id ? ` #${log.entity_id}` : ""}</td>
                    <td className="eds-cell-dim eds-cell-clip">{log.details || "-"}</td>
                    <td className="eds-money">{log.ip_address || "-"}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          )}
        </section>
      </div>
    </div>
  );
}
