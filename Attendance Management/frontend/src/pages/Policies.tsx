import { useCallback, useEffect, useState } from "react";
import { useAuth } from "../auth/AuthContext";
import {
  policies as policiesApi,
  type PolicyGroupRow,
  type PolicyVersion,
} from "../api/client";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import { SectionLoader } from "../components/LoadingState";
import ConfirmModal from "../components/ConfirmModal";
import { formatDate } from "../utils/dateFormatter";

const emptyForm = {
  name: "",
  title: "",
  category: "",
  content: "",
  effective_date: new Date().toISOString().slice(0, 10),
};

async function downloadAttachment(p: PolicyVersion) {
  const res = await policiesApi.attachment(p.id);
  const url = URL.createObjectURL(res.data as Blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = p.attachment_name || `policy-${p.id}`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

function VersionCard({
  v,
  isCurrent,
  canEdit,
  onDelete,
}: {
  v: PolicyVersion;
  isCurrent: boolean;
  canEdit: boolean;
  onDelete: (id: number) => void;
}) {
  return (
    <div
      style={{
        border: "1px solid rgba(255,255,255,0.1)",
        borderRadius: 10,
        padding: "0.8rem 1rem",
        marginTop: "0.6rem",
        background: isCurrent ? "rgba(34,197,94,0.08)" : "rgba(255,255,255,0.03)",
      }}
    >
      <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", alignItems: "center", flexWrap: "wrap" }}>
        <div style={{ fontWeight: 700 }}>
          {v.title || v.name}{" "}
          <span style={{ fontSize: "0.72rem", fontWeight: 800, color: isCurrent ? "#22c55e" : "rgba(255,255,255,0.5)" }}>
            v{v.version}{isCurrent ? " · CURRENT" : ""}
          </span>
        </div>
        <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.6)" }}>
          Effective {formatDate(v.effective_date)}
          {v.published_by_name ? ` · by ${v.published_by_name}` : ""}
        </div>
      </div>
      {v.content && <div style={{ whiteSpace: "pre-wrap", marginTop: "0.5rem", color: "rgba(255,255,255,0.85)" }}>{v.content}</div>}
      <div style={{ display: "flex", gap: "0.75rem", marginTop: "0.6rem", alignItems: "center" }}>
        {v.attachment_name && (
          <button type="button" className="btn btn-secondary btn-sm" onClick={() => downloadAttachment(v)}>
            ⬇ {v.attachment_name}
          </button>
        )}
        {canEdit && (
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            style={{ color: "#f87171" }}
            onClick={() => onDelete(v.id)}
          >
            Delete
          </button>
        )}
      </div>
    </div>
  );
}

export default function Policies() {
  const { hasRole } = useAuth();
  const canEdit = hasRole("Admin") || hasRole("HR");

  const [groups, setGroups] = useState<PolicyGroupRow[]>([]);
  const [loading, setLoading] = useState(true);
  const [expanded, setExpanded] = useState<string | null>(null);
  const [historyMap, setHistoryMap] = useState<Record<string, PolicyVersion[]>>({});
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState(emptyForm);
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<number | null>(null);

  const load = useCallback(() => {
    setLoading(true);
    policiesApi
      .list()
      .then((r) => setGroups(r.data))
      .catch(() => setGroups([]))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  const toggleHistory = (name: string) => {
    if (expanded === name) {
      setExpanded(null);
      return;
    }
    setExpanded(name);
    if (!historyMap[name]) {
      policiesApi
        .history(name)
        .then((r) => setHistoryMap((prev) => ({ ...prev, [name]: r.data.versions })))
        .catch(() => setHistoryMap((prev) => ({ ...prev, [name]: [] })));
    }
  };

  const submit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.name.trim()) {
      setError("Policy name is required");
      return;
    }
    if (!form.content.trim() && !file) {
      setError("Provide policy text and/or an attachment");
      return;
    }
    const fd = new FormData();
    fd.append("name", form.name.trim());
    fd.append("effective_date", form.effective_date);
    if (form.title.trim()) fd.append("title", form.title.trim());
    if (form.category.trim()) fd.append("category", form.category.trim());
    if (form.content.trim()) fd.append("content", form.content.trim());
    if (file) fd.append("file", file);

    setBusy(true);
    setError("");
    policiesApi
      .create(fd)
      .then(() => {
        setForm(emptyForm);
        setFile(null);
        setShowForm(false);
        setHistoryMap({});
        setExpanded(null);
        load();
      })
      .catch((e: any) => setError(e?.response?.data?.detail || "Failed to publish policy"))
      .finally(() => setBusy(false));
  };

  const doDelete = () => {
    if (confirmDelete == null) return;
    const id = confirmDelete;
    policiesApi
      .remove(id)
      .then(() => {
        setConfirmDelete(null);
        setHistoryMap({});
        setExpanded(null);
        load();
      })
      .catch(() => setConfirmDelete(null));
  };

  return (
    <div>
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="page-title">Company Policies</h1>
          <div className="page-subtitle">Current policies and previous versions.</div>
        </div>
        <GlobalHeaderControls />
      </div>

      {canEdit && (
        <div style={{ display: "flex", justifyContent: "flex-end", marginBottom: "1rem" }}>
          <button type="button" className="btn btn-primary" onClick={() => { setError(""); setShowForm((s) => !s); }}>
            {showForm ? "Cancel" : "Publish Policy / New Version"}
          </button>
        </div>
      )}

      {error && <div className="alert alert-error">{error}</div>}

      {canEdit && showForm && (
        <form className="card" onSubmit={submit} style={{ marginBottom: "1rem" }}>
          <p className="text-muted" style={{ marginTop: 0 }}>
            Use an existing policy name to publish a new version (the old one is kept in history), or a new name to create a new policy.
          </p>
          <div className="form-group">
            <label>Policy Name</label>
            <input
              list="policy-names"
              value={form.name}
              onChange={(e) => setForm({ ...form, name: e.target.value })}
              placeholder="e.g. Leave Policy"
              required
            />
            <datalist id="policy-names">
              {groups.map((g) => (
                <option key={g.name} value={g.name} />
              ))}
            </datalist>
          </div>
          <div className="form-group">
            <label>Version Title (optional)</label>
            <input value={form.title} onChange={(e) => setForm({ ...form, title: e.target.value })} placeholder="e.g. Leave Policy 2026" />
          </div>
          <div className="form-group">
            <label>Category (optional)</label>
            <input value={form.category} onChange={(e) => setForm({ ...form, category: e.target.value })} placeholder="e.g. HR, Attendance" />
          </div>
          <div className="form-group">
            <label>Effective Date</label>
            <input type="date" value={form.effective_date} onChange={(e) => setForm({ ...form, effective_date: e.target.value })} required />
          </div>
          <div className="form-group">
            <label>Policy Text</label>
            <textarea rows={6} value={form.content} onChange={(e) => setForm({ ...form, content: e.target.value })} placeholder="Type the policy here (optional if attaching a file)" />
          </div>
          <div className="form-group">
            <label>Attachment (optional — PDF/DOC)</label>
            <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
          </div>
          <button type="submit" className="btn btn-primary" disabled={busy}>{busy ? "Publishing…" : "Publish"}</button>
        </form>
      )}

      {loading ? (
        <div style={{ padding: "3rem 0" }}><SectionLoader size="md" /></div>
      ) : groups.length === 0 ? (
        <div className="card" style={{ color: "rgba(255,255,255,0.92)" }}>No company policies published yet.</div>
      ) : (
        groups.map((g) => (
          <div key={g.name} className="card" style={{ marginBottom: "0.75rem" }}>
            <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", alignItems: "center", flexWrap: "wrap" }}>
              <div>
                <div style={{ fontSize: "1.1rem", fontWeight: 800 }}>{g.name}</div>
                {g.category && <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.6)" }}>{g.category}</div>}
              </div>
              {g.versions_count > 1 && (
                <button type="button" className="btn btn-secondary btn-sm" onClick={() => toggleHistory(g.name)}>
                  {expanded === g.name ? "Hide history" : `View history (${g.versions_count} versions)`}
                </button>
              )}
            </div>

            {/* Current version */}
            <VersionCard v={g.current} isCurrent canEdit={canEdit} onDelete={setConfirmDelete} />

            {/* Previous versions */}
            {expanded === g.name &&
              (historyMap[g.name] || [])
                .filter((v) => v.id !== g.current.id)
                .map((v) => (
                  <VersionCard key={v.id} v={v} isCurrent={false} canEdit={canEdit} onDelete={setConfirmDelete} />
                ))}
          </div>
        ))
      )}

      <ConfirmModal
        isOpen={confirmDelete != null}
        onClose={() => setConfirmDelete(null)}
        onConfirm={doDelete}
        title="Delete Policy Version"
        message="Delete this policy version? This cannot be undone."
        confirmText="Yes, Delete"
      />
    </div>
  );
}
