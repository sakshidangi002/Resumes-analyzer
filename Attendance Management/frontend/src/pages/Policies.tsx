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
import RichTextEditor from "../components/RichTextEditor";
import { formatDate } from "../utils/dateFormatter";

// Policy text is authored by Admin/HR as rich HTML (bold/size). Strip anything
// executable before rendering it to viewers.
function sanitizeHtml(html: string): string {
  const tpl = document.createElement("template");
  tpl.innerHTML = html || "";
  tpl.content.querySelectorAll("script,style,iframe,object,embed,link,meta").forEach((n) => n.remove());
  tpl.content.querySelectorAll("*").forEach((el) => {
    Array.from(el.attributes).forEach((attr) => {
      const name = attr.name.toLowerCase();
      if (name.startsWith("on")) el.removeAttribute(attr.name);
      if ((name === "href" || name === "src") && /^\s*javascript:/i.test(attr.value)) el.removeAttribute(attr.name);
    });
  });
  return tpl.innerHTML;
}

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
  onEdit,
}: {
  v: PolicyVersion;
  isCurrent: boolean;
  canEdit: boolean;
  onDelete: (id: number) => void;
  onEdit?: (v: PolicyVersion) => void;
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
      {v.content && (
        <div
          style={{ marginTop: "0.5rem", color: "rgba(255,255,255,0.85)", lineHeight: 1.5 }}
          dangerouslySetInnerHTML={{ __html: sanitizeHtml(v.content) }}
        />
      )}
      <div style={{ display: "flex", gap: "0.75rem", marginTop: "0.6rem", alignItems: "center" }}>
        {v.attachment_name && (
          <button type="button" className="btn btn-secondary btn-sm" onClick={() => downloadAttachment(v)}>
            ⬇ {v.attachment_name}
          </button>
        )}
        {canEdit && onEdit && (
          <button type="button" className="btn btn-secondary btn-sm" onClick={() => onEdit(v)}>
            Edit
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
  const [historyMap, setHistoryMap] = useState<Record<string, PolicyVersion[]>>({});
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState(emptyForm);
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<number | null>(null);
  const [selected, setSelected] = useState<PolicyGroupRow | null>(null);
  const [editing, setEditing] = useState<PolicyVersion | null>(null);
  const [editForm, setEditForm] = useState({ title: "", category: "", content: "", effective_date: "" });
  const [editBusy, setEditBusy] = useState(false);

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

  const openPolicy = (g: PolicyGroupRow) => {
    setSelected(g);
    if (!historyMap[g.name]) {
      policiesApi
        .history(g.name)
        .then((r) => setHistoryMap((prev) => ({ ...prev, [g.name]: r.data.versions })))
        .catch(() => setHistoryMap((prev) => ({ ...prev, [g.name]: [] })));
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
        setSelected(null);
        load();
      })
      .catch(() => setConfirmDelete(null));
  };

  const openEdit = (v: PolicyVersion) => {
    setError("");
    setEditing(v);
    setEditForm({
      title: v.title || "",
      category: v.category || "",
      content: v.content || "",
      effective_date: v.effective_date,
    });
  };

  const saveEdit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!editing) return;
    setEditBusy(true);
    setError("");
    policiesApi
      .update(editing.id, {
        title: editForm.title || undefined,
        category: editForm.category || undefined,
        content: editForm.content || undefined,
        effective_date: editForm.effective_date || undefined,
      })
      .then((r) => {
        const updated = r.data;
        // Update in place so the modal + grid reflect the edit without a full reload.
        setHistoryMap((prev) => ({
          ...prev,
          [updated.name]: (prev[updated.name] || []).map((x) => (x.id === updated.id ? updated : x)),
        }));
        setSelected((prev) =>
          prev && prev.current.id === updated.id
            ? { ...prev, current: updated, category: updated.category }
            : prev
        );
        setGroups((prev) =>
          prev.map((g) => (g.current.id === updated.id ? { ...g, current: updated, category: updated.category } : g))
        );
        setEditing(null);
      })
      .catch((e: any) => setError(e?.response?.data?.detail || "Failed to save changes"))
      .finally(() => setEditBusy(false));
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
            <RichTextEditor
              value={form.content}
              onChange={(html) => setForm({ ...form, content: html })}
              placeholder="Type the policy here (optional if attaching a file)"
            />
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
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(260px, 1fr))", gap: "1rem" }}>
          {groups.map((g) => (
            <div
              key={g.name}
              className="card"
              onClick={() => openPolicy(g)}
              style={{ cursor: "pointer", display: "flex", flexDirection: "column", gap: "0.5rem", transition: "border-color 0.15s" }}
            >
              <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: "0.5rem" }}>
                <div style={{ fontSize: "1.05rem", fontWeight: 800 }}>{g.name}</div>
                <span style={{ fontSize: "0.68rem", fontWeight: 800, color: "#22c55e", whiteSpace: "nowrap" }}>v{g.current.version}</span>
              </div>
              {g.category && (
                <div style={{ fontSize: "0.72rem", color: "var(--brand-400)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.04em" }}>
                  {g.category}
                </div>
              )}
              <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.55)" }}>
                Effective {formatDate(g.current.effective_date)}
              </div>
              {g.current.content && (
                <div
                  style={{
                    fontSize: "0.85rem",
                    color: "rgba(255,255,255,0.75)",
                    display: "-webkit-box",
                    WebkitLineClamp: 3,
                    WebkitBoxOrient: "vertical",
                    overflow: "hidden",
                  }}
                >
                  {g.current.content}
                </div>
              )}
              <div style={{ marginTop: "auto", display: "flex", justifyContent: "space-between", alignItems: "center", paddingTop: "0.4rem" }}>
                <span style={{ fontSize: "0.72rem", color: "rgba(255,255,255,0.5)" }}>
                  {g.versions_count} version{g.versions_count === 1 ? "" : "s"}
                  {g.current.attachment_name ? " · 📎 file" : ""}
                </span>
                <span style={{ fontSize: "0.78rem", color: "var(--brand-400)", fontWeight: 800 }}>View →</span>
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Full-policy detail modal */}
      {selected && (
        <div
          onClick={() => setSelected(null)}
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.6)",
            zIndex: 3000,
            display: "flex",
            alignItems: "flex-start",
            justifyContent: "center",
            padding: "3rem 1rem",
            overflowY: "auto",
          }}
        >
          <div
            className="card"
            onClick={(e) => e.stopPropagation()}
            style={{ maxWidth: 760, width: "100%", maxHeight: "85vh", overflowY: "auto" }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", gap: "1rem" }}>
              <div>
                <div style={{ fontSize: "1.35rem", fontWeight: 800 }}>{selected.name}</div>
                {selected.category && (
                  <div style={{ fontSize: "0.75rem", color: "var(--brand-400)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.04em", marginTop: 2 }}>
                    {selected.category}
                  </div>
                )}
              </div>
              <button type="button" className="btn btn-secondary btn-sm" onClick={() => setSelected(null)}>✕ Close</button>
            </div>

            {editing ? (
              /* Edit an existing version in place */
              <form onSubmit={saveEdit} style={{ marginTop: "0.75rem" }}>
                {error && <div className="alert alert-error">{error}</div>}
                <div className="form-group">
                  <label>Version Title</label>
                  <input value={editForm.title} onChange={(e) => setEditForm({ ...editForm, title: e.target.value })} placeholder="Title" />
                </div>
                <div className="form-group">
                  <label>Category</label>
                  <input value={editForm.category} onChange={(e) => setEditForm({ ...editForm, category: e.target.value })} placeholder="Category" />
                </div>
                <div className="form-group">
                  <label>Effective Date</label>
                  <input type="date" value={editForm.effective_date} onChange={(e) => setEditForm({ ...editForm, effective_date: e.target.value })} />
                </div>
                <div className="form-group">
                  <label>Policy Text</label>
                  <RichTextEditor
                    key={editing.id}
                    value={editForm.content}
                    onChange={(html) => setEditForm({ ...editForm, content: html })}
                    placeholder="Policy text"
                  />
                </div>
                <div style={{ display: "flex", gap: "0.5rem" }}>
                  <button type="submit" className="btn btn-primary" disabled={editBusy}>{editBusy ? "Saving…" : "Save Changes"}</button>
                  <button type="button" className="btn btn-secondary" onClick={() => setEditing(null)}>Cancel</button>
                </div>
                <div className="text-muted" style={{ fontSize: "0.75rem", marginTop: "0.5rem" }}>
                  Editing v{editing.version} in place — this corrects the version without creating a new one.
                </div>
              </form>
            ) : (
              <>
                {/* Current version (full content) */}
                <div style={{ marginTop: "0.5rem" }}>
                  <VersionCard v={selected.current} isCurrent canEdit={canEdit} onDelete={setConfirmDelete} onEdit={openEdit} />
                </div>

                {/* Previous versions */}
                {selected.versions_count > 1 && (
                  <div style={{ marginTop: "1rem" }}>
                    <div style={{ fontSize: "0.8rem", fontWeight: 800, color: "rgba(255,255,255,0.7)", textTransform: "uppercase", letterSpacing: "0.04em" }}>
                      Previous versions
                    </div>
                    {(historyMap[selected.name] || [])
                      .filter((v) => v.id !== selected.current.id)
                      .map((v) => (
                        <VersionCard key={v.id} v={v} isCurrent={false} canEdit={canEdit} onDelete={setConfirmDelete} onEdit={openEdit} />
                      ))}
                    {!historyMap[selected.name] && <p className="text-muted" style={{ marginTop: "0.5rem" }}>Loading previous versions…</p>}
                  </div>
                )}
              </>
            )}
          </div>
        </div>
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
