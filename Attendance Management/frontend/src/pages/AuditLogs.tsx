import { useEffect, useState } from "react";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import { audit, type AuditLogRow } from "../api/client";

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
    <>
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="page-title">Audit Logs</h1>
          <div className="page-subtitle">Track important HRMS and account activity</div>
        </div>
        <GlobalHeaderControls />
      </div>
      <div className="card">
        <div style={{ display: "flex", gap: "0.75rem", marginBottom: "1rem" }}>
          <input value={action} onChange={(e) => setAction(e.target.value)} placeholder="Filter by action" onKeyDown={(e) => { if (e.key === "Enter") void load(); }} />
          <button className="btn btn-primary" onClick={() => void load()}>Filter</button>
        </div>
        {error && <div className="alert alert-error">{error}</div>}
        {loading ? <p className="text-muted">Loading audit logs…</p> : logs.length === 0 ? <p className="text-muted">No audit activity found.</p> : (
          <div className="table-wrap table-wrap--dark">
            <table className="table-modern table-modern--dark">
              <thead><tr><th>Date</th><th>User</th><th>Action</th><th>Entity</th><th>Details</th><th>IP</th></tr></thead>
              <tbody>{logs.map((log) => (
                <tr key={log.id}>
                  <td>{new Date(log.created_at).toLocaleString()}</td>
                  <td>{log.username || `User #${log.user_id ?? "-"}`}</td>
                  <td><strong>{log.action}</strong></td>
                  <td>{log.entity_type || "-"}{log.entity_id ? ` #${log.entity_id}` : ""}</td>
                  <td>{log.details || "-"}</td>
                  <td>{log.ip_address || "-"}</td>
                </tr>
              ))}</tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}
