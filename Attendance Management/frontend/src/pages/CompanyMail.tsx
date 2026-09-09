import { useCallback, useEffect, useState } from "react";
import { companyMail, type MailRow, type MailMessage } from "../api/client";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import { SectionLoader } from "../components/LoadingState";

// The mailbox is read through the email MCP server, which spawns a subprocess
// and talks IMAP. That is slow and rate-limited, so this page fetches on
// explicit action (mount, Refresh, filter change) and never polls.

const FILTERS: { label: string; value: string }[] = [
  { label: "All", value: "ALL" },
  { label: "Unread", value: "UNSEEN" },
];

function shortDate(raw: string): string {
  // IMAP dates arrive RFC-2822 ("Sun, 6 Sep 2026 07:39:06 +0000"). Date can
  // parse that, but fall back to the raw string rather than showing "Invalid
  // Date" if a server sends something unusual.
  const d = new Date(raw);
  if (Number.isNaN(d.getTime())) return raw;
  return d.toLocaleString(undefined, {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function CompanyMail() {
  const [rows, setRows] = useState<MailRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [search, setSearch] = useState("ALL");

  const [openUid, setOpenUid] = useState<string | null>(null);
  const [message, setMessage] = useState<MailMessage | null>(null);
  const [messageLoading, setMessageLoading] = useState(false);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await companyMail.list({ limit: 25, search });
      setRows(res.data);
    } catch (err: any) {
      setError(
        err?.response?.status === 502
          ? "Could not reach the mail server. Check that the MCP server and its credentials are configured."
          : "Failed to load the mailbox."
      );
    } finally {
      setLoading(false);
    }
  }, [search]);

  useEffect(() => {
    load();
  }, [load]);

  const openMessage = useCallback(async (uid: string) => {
    setOpenUid(uid);
    setMessage(null);
    setMessageLoading(true);
    try {
      const res = await companyMail.read(uid);
      setMessage(res.data);
    } catch {
      setMessage(null);
    } finally {
      setMessageLoading(false);
    }
  }, []);

  return (
    <div className="eds">
      <header className="eds-topbar">
        <div>
          <h1 className="eds-title">Company Mail</h1>
          <p className="eds-subtitle">The shared HR mailbox, read over MCP.</p>
        </div>
        <GlobalHeaderControls />
      </header>

      <div className="eds-page">
        <div style={{ display: "flex", gap: 8, alignItems: "center", marginBottom: 16 }}>
          {FILTERS.map((f) => (
            <button
              key={f.value}
              type="button"
              className={"btn btn-sm " + (search === f.value ? "btn-primary" : "btn-secondary")}
              onClick={() => setSearch(f.value)}
            >
              {f.label}
            </button>
          ))}
          <button type="button" className="btn btn-secondary btn-sm" onClick={load} disabled={loading}>
            {loading ? "Loading…" : "Refresh"}
          </button>
        </div>

        {error && <div className="alert alert-error" style={{ marginBottom: 16 }}>{error}</div>}

        {loading ? (
          <SectionLoader />
        ) : rows.length === 0 ? (
          <p className="eds-subtitle">No messages.</p>
        ) : (
          <table className="table">
            <thead>
              <tr>
                <th style={{ width: "28%" }}>From</th>
                <th>Subject</th>
                <th style={{ width: "18%" }}>Date</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr
                  key={r.uid}
                  onClick={() => openMessage(r.uid)}
                  style={{ cursor: "pointer" }}
                >
                  <td title={r.from}>{r.from}</td>
                  <td>{r.subject || "(no subject)"}</td>
                  <td>{shortDate(r.date)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {openUid && (
        <div className="modal-backdrop" onClick={() => setOpenUid(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 760 }}>
            {messageLoading ? (
              <SectionLoader />
            ) : message ? (
              <>
                <h2 style={{ marginTop: 0 }}>{message.subject || "(no subject)"}</h2>
                <p className="eds-subtitle" style={{ marginTop: 0 }}>
                  From {message.from} · {shortDate(message.date)}
                </p>
                <pre
                  style={{
                    whiteSpace: "pre-wrap",
                    wordBreak: "break-word",
                    maxHeight: "55vh",
                    overflowY: "auto",
                    fontFamily: "inherit",
                  }}
                >
                  {message.body || "(no plain-text body)"}
                </pre>
              </>
            ) : (
              <p>Could not load this message.</p>
            )}
            <div style={{ textAlign: "right", marginTop: 12 }}>
              <button type="button" className="btn btn-secondary btn-sm" onClick={() => setOpenUid(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
