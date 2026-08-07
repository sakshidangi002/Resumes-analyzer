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

export default function AuditLogs() {
  const [logs, setLogs] = useState<AuditLogRow[]>([]);
  const [action, setAction] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const load = async () => {
    setLoading(true);
    setError("");
    try {
      const response = await audit.list({ limit: 500, action: action.trim() || undefined });
      setLogs(response.data || []);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Could not load audit logs.");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { void load(); }, []);

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
              <span>No audit activity found.</span>
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
