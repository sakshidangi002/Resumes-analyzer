import { useMemo, useState, useEffect } from "react";
import { useAuth } from "../auth/AuthContext";
import { departments as deptApi } from "../api/client";
import { designations as desigApi } from "../api/client";
import ConfirmModal from "../components/ConfirmModal";
import { SectionLoader } from "../components/LoadingState";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import CustomSelect from "../components/CustomSelect";

interface Department {
  id: number;
  name: string;
  code: string | null;
}

interface Designation {
  id: number;
  title: string;
}

// Premium SVG Icons for Actions
const Icons = {
  Edit: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
    </svg>
  ),
  Delete: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6"></polyline>
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
      <line x1="10" y1="11" x2="10" y2="17"></line>
      <line x1="14" y1="11" x2="14" y2="17"></line>
    </svg>
  ),
  ChevronDown: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
      <path d="m6 9 6 6 6-6" />
    </svg>
  ),
};

const DESIGNATION_GROUP_ORDER = [
  "Backend developers",
  "Frontend developers",
  "HR",
  "SEO",
  "Others",
] as const;

type DesignationGroupId = (typeof DESIGNATION_GROUP_ORDER)[number];

function seniorityRank(title: string): number {
  const t = title.toLowerCase();
  if (/\b(principal|staff|distinguished|director|head\s+of)\b/.test(t)) return 6;
  if (/\b(lead|architect)\b/.test(t)) return 5;
  if (/\bsenior\b/.test(t)) return 4;
  if (/\bintern\b/.test(t)) return 0;
  if (/\btrainee\b/.test(t)) return 1;
  if (/\bjunior\b/.test(t)) return 2;
  return 3;
}

function sortDesignationsInGroup(a: Designation, b: Designation): number {
  const ra = seniorityRank(a.title);
  const rb = seniorityRank(b.title);
  if (ra !== rb) return ra - rb;
  return a.title.localeCompare(b.title);
}

type TrackDef = { id: string; label: string; test: (title: string) => boolean };

const BACKEND_TRACKS: TrackDef[] = [
  { id: "dotnet", label: ".NET / C#", test: (t) => /\.net|dot\s*net|c#/i.test(t) },
  { id: "python", label: "Python", test: (t) => /python|django|flask/i.test(t) },
  { id: "node", label: "Node / API", test: (t) => /node|express|nest\.?js/i.test(t) },
  { id: "java", label: "Java / Spring", test: (t) => /java(?!script)|spring|kotlin/i.test(t) },
  { id: "php", label: "PHP", test: (t) => /php|laravel|symfony/i.test(t) },
  { id: "go", label: "Go", test: (t) => /\bgo\b|golang/i.test(t) },
  { id: "ruby", label: "Ruby", test: (t) => /ruby|rails/i.test(t) },
  { id: "devops", label: "DevOps / Cloud", test: (t) => /devops|sre|kubernetes|docker|aws|azure|gcp|ci\/cd/i.test(t) },
  { id: "data", label: "Data / DBA / SQL", test: (t) => /database|dba|\bsql\b|postgres|mysql|mongo/i.test(t) },
];

const FRONTEND_TRACKS: TrackDef[] = [
  { id: "react", label: "React", test: (t) => /react/i.test(t) },
  { id: "angular", label: "Angular", test: (t) => /angular/i.test(t) },
  { id: "vue", label: "Vue", test: (t) => /vue/i.test(t) },
  { id: "next", label: "Next.js", test: (t) => /next\.?js|nextjs/i.test(t) },
  { id: "js", label: "JavaScript / TypeScript", test: (t) => /javascript|typescript|\bjs\b|\bts\b/i.test(t) },
  { id: "ui", label: "UI / UX", test: (t) => /\bui\b|\bux\b|figma|web\s*designer/i.test(t) },
];

function firstMatchingTrack(title: string, tracks: TrackDef[]): string {
  for (const tr of tracks) {
    if (tr.test(title)) return tr.id;
  }
  return "other";
}

function trackOptionsForGroup(
  groupId: DesignationGroupId,
  rows: Designation[]
): { id: string; label: string }[] {
  if (groupId === "Backend developers") {
    const seen = new Set<string>();
    const opts: { id: string; label: string }[] = [];
    for (const d of rows) {
      const id = firstMatchingTrack(d.title, BACKEND_TRACKS);
      if (id === "other" || seen.has(id)) continue;
      const def = BACKEND_TRACKS.find((t) => t.id === id);
      if (def) {
        seen.add(id);
        opts.push({ id: def.id, label: def.label });
      }
    }
    opts.sort((a, b) => a.label.localeCompare(b.label));
    const base = [{ id: "all", label: "All roles" }, ...opts];
    const hasOther = rows.some((d) => firstMatchingTrack(d.title, BACKEND_TRACKS) === "other");
    return hasOther ? [...base, { id: "other", label: "Other backend" }] : base;
  }
  if (groupId === "Frontend developers") {
    const seen = new Set<string>();
    const opts: { id: string; label: string }[] = [];
    for (const d of rows) {
      const id = firstMatchingTrack(d.title, FRONTEND_TRACKS);
      if (id === "other" || seen.has(id)) continue;
      const def = FRONTEND_TRACKS.find((t) => t.id === id);
      if (def) {
        seen.add(id);
        opts.push({ id: def.id, label: def.label });
      }
    }
    opts.sort((a, b) => a.label.localeCompare(b.label));
    const base = [{ id: "all", label: "All roles" }, ...opts];
    const hasOther = rows.some((d) => firstMatchingTrack(d.title, FRONTEND_TRACKS) === "other");
    return hasOther ? [...base, { id: "other", label: "Other frontend" }] : base;
  }
  if (groupId === "Others" && rows.length > 0) {
    const buckets = new Map<string, string>();
    for (const d of rows) {
      const t = d.title.trim();
      const first = t.split(/\s+/)[0] || t;
      const key = first.toLowerCase();
      if (!buckets.has(key)) buckets.set(key, first);
    }
    if (buckets.size >= 2) {
      const opts = [...buckets.entries()]
        .sort((a, b) => a[1].localeCompare(b[1]))
        .map(([id, label]) => ({ id: `prefix:${id}`, label }));
      return [{ id: "all", label: "All roles" }, ...opts];
    }
  }
  return [{ id: "all", label: "All roles" }];
}

function filterRowsByTrack(
  groupId: DesignationGroupId,
  rows: Designation[],
  trackId: string
): Designation[] {
  if (trackId === "all" || !trackId) return rows;
  if (groupId === "Backend developers") {
    if (trackId === "other") return rows.filter((d) => firstMatchingTrack(d.title, BACKEND_TRACKS) === "other");
    const def = BACKEND_TRACKS.find((t) => t.id === trackId);
    return def ? rows.filter((d) => def.test(d.title)) : rows;
  }
  if (groupId === "Frontend developers") {
    if (trackId === "other") return rows.filter((d) => firstMatchingTrack(d.title, FRONTEND_TRACKS) === "other");
    const def = FRONTEND_TRACKS.find((t) => t.id === trackId);
    return def ? rows.filter((d) => def.test(d.title)) : rows;
  }
  if (groupId === "Others" && trackId.startsWith("prefix:")) {
    const prefix = trackId.slice("prefix:".length);
    return rows.filter((d) => d.title.trim().toLowerCase().startsWith(prefix));
  }
  return rows;
}

export default function DepartmentsDesignations() {
  const { hasRole } = useAuth();
  const canEdit = hasRole("Admin") || hasRole("HR");

  const [deptList, setDeptList] = useState<Department[]>([]);
  const [desigList, setDesigList] = useState<Designation[]>([]);
  const [openSection, setOpenSection] = useState<"departments" | "designations" | null>("departments");
  const [loading, setLoading] = useState(true);
  const [desigSearch, setDesigSearch] = useState("");
  const [desigGroupTrack, setDesigGroupTrack] = useState<Record<string, string>>({});
  const [openGroupName, setOpenGroupName] = useState<string | null>(null);
  const [deptModal, setDeptModal] = useState<"add" | "edit" | null>(null);
  const [desigModal, setDesigModal] = useState<"add" | "edit" | null>(null);
  const [editingDeptId, setEditingDeptId] = useState<number | null>(null);
  const [editingDesigId, setEditingDesigId] = useState<number | null>(null);
  const [deptForm, setDeptForm] = useState({ name: "", code: "" });
  const [desigForm, setDesigForm] = useState({ title: "" });
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [confirmDeleteDept, setConfirmDeleteDept] = useState<Department | null>(null);
  const [confirmDeleteDesig, setConfirmDeleteDesig] = useState<Designation | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const load = async () => {
    setLoading(true);
    const [deptRes, desigRes] = await Promise.allSettled([deptApi.list(), desigApi.list()]);
    if (deptRes.status === "fulfilled") setDeptList(deptRes.value.data);
    else setDeptList([]);
    if (desigRes.status === "fulfilled") setDesigList(desigRes.value.data);
    else setDesigList([]);
    setLoading(false);
  };

  useEffect(() => {
    void load();
  }, []);

  const [desigGroupSelectOpen, setDesigGroupSelectOpen] = useState<string | null>(null);

  const desigGroups = useMemo(() => {
    const q = desigSearch.trim().toLowerCase();
    const rows = [...desigList].filter((d) => (q ? d.title.toLowerCase().includes(q) : true));

    const groups: Record<DesignationGroupId, Designation[]> = {
      "Backend developers": [],
      "Frontend developers": [],
      HR: [],
      SEO: [],
      Others: [],
    };

    const isBackend = (t: string) =>
      /backend|\.net|dot\s*net|c#|java(?!script)|spring|python|django|flask|node|api|devops|database|sql/i.test(t);
    const isFrontend = (t: string) => /frontend|react|angular|vue|javascript|ui|ux|web\s*designer/i.test(t);
    const isHr = (t: string) => /\bhr\b|human\s*resources|recruit/i.test(t);
    const isSeo = (t: string) => /\bseo\b|search\s*engine/i.test(t);

    for (const d of rows) {
      const t = d.title || "";
      if (isHr(t)) groups["HR"].push(d);
      else if (isSeo(t)) groups["SEO"].push(d);
      else if (isBackend(t)) groups["Backend developers"].push(d);
      else if (isFrontend(t)) groups["Frontend developers"].push(d);
      else groups["Others"].push(d);
    }

    (Object.keys(groups) as DesignationGroupId[]).forEach((k) => {
      groups[k].sort(sortDesignationsInGroup);
    });

    return {
      total: rows.length,
      groups,
    };
  }, [desigList, desigSearch]);

  const openDeptAdd = () => {
    setDeptForm({ name: "", code: "" });
    setEditingDeptId(null);
    setDeptModal("add");
    setError("");
    setSuccess("");
  };

  const openDeptEdit = (d: Department) => {
    setDeptForm({ name: d.name, code: d.code || "" });
    setEditingDeptId(d.id);
    setDeptModal("edit");
    setError("");
    setSuccess("");
  };

  const openDesigAdd = () => {
    setDesigForm({ title: "" });
    setEditingDesigId(null);
    setDesigModal("add");
    setError("");
    setSuccess("");
  };

  const openDesigEdit = (d: Designation) => {
    setDesigForm({ title: d.title });
    setEditingDesigId(d.id);
    setDesigModal("edit");
    setError("");
    setSuccess("");
  };

  const handleDeptSubmitAdd = (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    deptApi
      .create({ name: deptForm.name.trim(), code: deptForm.code.trim() || undefined })
      .then((res) => {
        setSuccess("Department created.");
        setDeptList(prev => [...prev, res.data]);
        setDeptModal(null);
      })
      .catch((err) => setError(err.response?.data?.detail || "Failed"))
      .finally(() => setSubmitting(false));
  };

  const handleDeptSubmitEdit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingDeptId) return;
    setError("");
    setSubmitting(true);
    deptApi
      .update(editingDeptId, { name: deptForm.name.trim(), code: deptForm.code.trim() || undefined })
      .then((res) => {
        setSuccess("Department updated.");
        const updated = res.data as Department;
        setDeptList(prev => prev.map(d => d.id === updated.id ? updated : d));
        setDeptModal(null);
      })
      .catch((err) => setError(err.response?.data?.detail || "Failed"))
      .finally(() => setSubmitting(false));
  };

  const handleDeptDelete = (d: Department) => {
    setConfirmDeleteDept(d);
  };

  const confirmActualDeptDelete = () => {
    if (!confirmDeleteDept) return;
    const deletedId = confirmDeleteDept.id;
    setSubmitting(true);
    deptApi
      .delete(deletedId)
      .then(() => {
        setSuccess("Department deleted.");
        setDeptList((prev) => prev.filter((d) => d.id !== deletedId));
        setConfirmDeleteDept(null);
      })
      .catch((err) => setError(err.response?.data?.detail || "Failed"))
      .finally(() => setSubmitting(false));
  };

  const handleDesigSubmitAdd = (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSubmitting(true);
    desigApi
      .create({
        title: desigForm.title.trim(),
      })
      .then((res) => {
        setSuccess("Designation created.");
        setDesigList(prev => [...prev, res.data]);
        setDesigModal(null);
      })
      .catch((err) => setError(err.response?.data?.detail || "Failed"))
      .finally(() => setSubmitting(false));
  };

  const handleDesigSubmitEdit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingDesigId) return;
    setError("");
    setSubmitting(true);
    desigApi
      .update(editingDesigId, {
        title: desigForm.title.trim(),
      })
      .then((res) => {
        setSuccess("Designation updated.");
        const updated = res.data as Designation;
        setDesigList(prev => prev.map(d => d.id === updated.id ? updated : d));
        setDesigModal(null);
      })
      .catch((err) => setError(err.response?.data?.detail || "Failed"))
      .finally(() => setSubmitting(false));
  };

  const handleDesigDelete = (d: Designation) => {
    setConfirmDeleteDesig(d);
  };

  const confirmActualDesigDelete = () => {
    if (!confirmDeleteDesig) return;
    const deletedId = confirmDeleteDesig.id;
    setSubmitting(true);
    desigApi
      .delete(deletedId)
      .then(() => {
        setSuccess("Designation deleted.");
        setDesigList((prev) => prev.filter((d) => d.id !== deletedId));
        setConfirmDeleteDesig(null);
      })
      .catch((err) => setError(err.response?.data?.detail || "Failed"))
      .finally(() => setSubmitting(false));
  };

  if (!canEdit) {
    return (
      <div className="card">
        <p>Access denied. Admin or HR only.</p>
      </div>
    );
  }


  return (
    <>
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="page-title">Departments & Designations</h1>
          <div className="page-subtitle">Maintain your organization structure</div>
        </div>
        <GlobalHeaderControls />
      </div>
      {success && <div className="alert alert-success">{success}</div>}
      {error && <div className="alert alert-error">{error}</div>}

      <div
        className={`card collapsible-section ${openSection === 'departments' ? 'is-expanded' : ''}`}
        style={{ padding: 0, overflow: 'hidden', marginBottom: '1.5rem' }}
      >
        <div
          className="collapsible-header"
          onClick={() => setOpenSection(openSection === 'departments' ? null : 'departments')}
          style={{
            padding: '1.25rem 1.5rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            cursor: 'pointer',
            background: openSection === 'departments' ? 'rgba(255,255,255,0.04)' : 'rgba(255,255,255,0.02)',
            transition: 'all 0.2s'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 600 }}>Departments</h3>
            <span className="badge-count" style={{ background: 'rgba(255,255,255,0.1)', padding: '2px 8px', borderRadius: '10px', fontSize: '0.8rem' }}>
              {deptList.length}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            {canEdit && openSection === 'departments' && (
              <button type="button" className="btn btn-primary btn-sm" onClick={(e) => { e.stopPropagation(); openDeptAdd(); }} style={{ height: '42px', minWidth: '140px' }}>
                Add Department
              </button>
            )}
            <div style={{ transform: openSection === 'departments' ? 'rotate(180deg)' : 'none', transition: 'transform 0.3s' }}>
              <Icons.ChevronDown />
            </div>
          </div>
        </div>

        {openSection === 'departments' && (
          <div style={{ padding: '1.5rem', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
            <div className="table-wrap table-wrap--dark">
              <table className="table-modern table-modern--dark">
                <thead>
                  <tr>
                    <th>Name</th>
                    <th>Code</th>
                    {canEdit && <th className="actions-center" style={{ width: "140px" }}>Actions</th>}
                  </tr>
                </thead>
                <tbody>
                  {loading ? (
                    <tr><td colSpan={canEdit ? 3 : 2}><SectionLoader size="md" /></td></tr>
                  ) : deptList.map((d) => (
                    <tr key={d.id}>
                      <td>{d.name}</td>
                      <td>{d.code || "-"}</td>
                      {canEdit && (
                        <td className="actions-center">
                          <div className="actions-stack">
                            <button type="button" className="btn btn-secondary btn-icon btn-sm" onClick={() => openDeptEdit(d)} title="Edit Department">
                              <Icons.Edit />
                            </button>
                            <button type="button" className="btn btn-danger btn-icon btn-sm" onClick={() => handleDeptDelete(d)} title="Delete Department">
                              <Icons.Delete />
                            </button>
                          </div>
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {deptList.length === 0 && !loading && <p className="text-muted">No departments. Add one to use in Employees.</p>}
          </div>
        )}
      </div>

      <div
        className={`card collapsible-section ${openSection === 'designations' ? 'is-expanded' : ''}`}
        style={{ padding: 0, overflow: 'hidden' }}
      >
        <div
          className="collapsible-header"
          onClick={() => setOpenSection(openSection === 'designations' ? null : 'designations')}
          style={{
            padding: '1.25rem 1.5rem',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            cursor: 'pointer',
            background: openSection === 'designations' ? 'rgba(255,255,255,0.04)' : 'rgba(255,255,255,0.02)',
            transition: 'all 0.2s'
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
            <h3 style={{ margin: 0, fontSize: '1.1rem', fontWeight: 600 }}>Designations</h3>
            <span className="badge-count" style={{ background: 'rgba(255,255,255,0.1)', padding: '2px 8px', borderRadius: '10px', fontSize: '0.8rem' }}>
              {desigGroups.total}
            </span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
            {canEdit && openSection === 'designations' && (
              <button type="button" className="btn btn-primary btn-sm" onClick={(e) => { e.stopPropagation(); openDesigAdd(); }} style={{ height: "42px", minWidth: "140px" }}>
                Add Designation
              </button>
            )}
            <div style={{ transform: openSection === 'designations' ? 'rotate(180deg)' : 'none', transition: 'transform 0.3s' }}>
              <Icons.ChevronDown />
            </div>
          </div>
        </div>

        {openSection === 'designations' && (
          <div style={{ padding: '1.5rem', borderTop: '1px solid rgba(255,255,255,0.05)' }}>
            <div className="desig-card-toolbar" style={{ marginBottom: '1.5rem' }}>
              <div className="form-group desig-card-search" style={{ marginBottom: 0 }}>
                <label>Search designation</label>
                <input
                  value={desigSearch}
                  onChange={(e) => setDesigSearch(e.target.value)}
                  placeholder="e.g. React, Python, HR…"
                />
              </div>
            </div>

            {loading ? (
              <div style={{ padding: "3rem 0" }}><SectionLoader size="md" /></div>
            ) : desigList.length === 0 ? (
              <p className="text-muted desig-card-empty-msg">No designations. Add one to use in Employees.</p>
            ) : (
              <div className="desig-group-list">
                {DESIGNATION_GROUP_ORDER.map((groupName, groupIndex) => {
                  const rowsAll = desigGroups.groups[groupName];
                  const trackOptions = trackOptionsForGroup(groupName, rowsAll);
                  const selectedTrack = desigGroupTrack[groupName] ?? "all";
                  const validTrack =
                    trackOptions.some((o) => o.id === selectedTrack) ? selectedTrack : "all";
                  const rows = filterRowsByTrack(groupName, rowsAll, validTrack);

                  const isOpen = openGroupName === groupName;
                  const panelDomId = `desig-panel-${groupName.replace(/\s+/g, "-").toLowerCase()}`;
                  const headingDomId = `desig-heading-${groupName.replace(/\s+/g, "-").toLowerCase()}`;

                  return (
                    <div
                      key={groupName}
                      className={`desig-group${isOpen ? " desig-group--open" : ""}`}
                      style={{ 
                        animationDelay: `${Math.min(groupIndex, 8) * 0.055}s`,
                        zIndex: desigGroupSelectOpen === groupName ? 100 : (isOpen ? 2 : 1),
                        position: 'relative'
                      }}
                    >
                      <div className="desig-group-header">
                        <button
                          type="button"
                          className="desig-group-heading"
                          aria-expanded={isOpen}
                          aria-controls={panelDomId}
                          id={headingDomId}
                          onClick={() =>
                            setOpenGroupName(isOpen ? null : groupName)
                          }
                        >
                          <span className={`desig-group-caret${isOpen ? " desig-group-caret--open" : ""}`} aria-hidden />
                          <span className="desig-group-title">{groupName}</span>
                          <span className="desig-group-count" aria-label={`${rowsAll.length} in this group`}>
                            {rowsAll.length}
                          </span>
                        </button>
                        {trackOptions.length > 1 && (
                          <label className="desig-group-track-label">
                            <span className="sr-only">Filter {groupName} by role</span>
                            <CustomSelect
                              value={validTrack}
                              onChange={(val) => {
                                setDesigGroupTrack((prev) => ({ ...prev, [groupName]: val }));
                              }}
                              onToggle={(isOp) => setDesigGroupSelectOpen(isOp ? groupName : null)}
                              options={trackOptions.map((o) => ({
                                value: o.id,
                                label: o.label
                              }))}
                              style={{ width: "200px" }}
                            />
                          </label>
                        )}
                      </div>
                      <div
                        className={`desig-group-panel-outer${isOpen ? " desig-group-panel-outer--open" : ""}`}
                        id={panelDomId}
                        role="region"
                        aria-labelledby={headingDomId}
                        aria-hidden={!isOpen}
                      >
                        <div className="desig-group-panel-inner" {...(!isOpen ? { inert: "" as const } : {})}>
                          <div className="desig-group-panel">
                            {rowsAll.length === 0 ? (
                              <p className="text-muted desig-group-empty">No items in this group.</p>
                            ) : rows.length === 0 ? (
                              <p className="text-muted desig-group-empty">No designations for this filter.</p>
                            ) : (
                              <div className="table-wrap table-wrap--dark desig-group-table">
                                <table className="table-modern table-modern--dark">
                                  <thead>
                                    <tr>
                                      <th>Title</th>
                                      {canEdit && <th className="actions-center" style={{ width: "140px" }}>Actions</th>}
                                    </tr>
                                  </thead>
                                  <tbody>
                                    {rows.map((d) => (
                                      <tr key={d.id}>
                                        <td>{d.title}</td>
                                        {canEdit && (
                                          <td className="actions-center">
                                            <div className="actions-stack">
                                              <button type="button" className="btn btn-secondary btn-icon btn-sm" onClick={() => openDesigEdit(d)} title="Edit Designation">
                                                <Icons.Edit />
                                              </button>
                                              <button type="button" className="btn btn-danger btn-icon btn-sm" onClick={() => handleDesigDelete(d)} title="Delete Designation">
                                                <Icons.Delete />
                                              </button>
                                            </div>
                                          </td>
                                        )}
                                      </tr>
                                    ))}
                                  </tbody>
                                </table>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}
      </div>

      {deptModal && (
        <div className="modal-backdrop" onClick={() => setDeptModal(null)}>
          <div className="modal" style={{ maxWidth: 420 }} onClick={(e) => e.stopPropagation()}>
            <h3 style={{ marginTop: 0 }}>
              {deptModal === "add" ? "Add Department" : "Edit Department"}
            </h3>
            <form
              onSubmit={deptModal === "add" ? handleDeptSubmitAdd : handleDeptSubmitEdit}
              className="modal-stack"
            >
              <div className="form-group">
                <label>Name *</label>
                <input
                  value={deptForm.name}
                  onChange={(e) => setDeptForm({ ...deptForm, name: e.target.value })}
                  required
                  disabled={submitting}
                />
              </div>
              <div className="form-group">
                <label>Code</label>
                <input
                  value={deptForm.code}
                  onChange={(e) => setDeptForm({ ...deptForm, code: e.target.value })}
                  placeholder="e.g. ENG"
                  disabled={submitting}
                />
              </div>
              <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.75rem", marginTop: "1rem" }}>
                <button type="submit" className="btn btn-primary" disabled={submitting}>
                  {submitting ? "Saving..." : (deptModal === "add" ? "Create" : "Update")}
                </button>
                <button type="button" className="btn btn-cancel-alt" onClick={() => setDeptModal(null)} disabled={submitting}> Cancel </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {desigModal && (
        <div className="modal-backdrop" onClick={() => setDesigModal(null)}>
          <div className="modal" style={{ maxWidth: 420 }} onClick={(e) => e.stopPropagation()}>
            <h3 style={{ marginTop: 0 }}>
              {desigModal === "add" ? "Add Designation" : "Edit Designation"}
            </h3>
            <form
              onSubmit={desigModal === "add" ? handleDesigSubmitAdd : handleDesigSubmitEdit}
              className="modal-stack"
            >
              <div className="form-group">
                <label>Title *</label>
                <input
                  value={desigForm.title}
                  onChange={(e) => setDesigForm({ ...desigForm, title: e.target.value })}
                  required
                  disabled={submitting}
                />
              </div>
              <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.75rem", marginTop: "1rem" }}>
                <button type="submit" className="btn btn-primary" disabled={submitting}>
                  {submitting ? "Saving..." : (desigModal === "add" ? "Create" : "Update")}
                </button>
                <button type="button" className="btn btn-cancel-alt" onClick={() => setDesigModal(null)} disabled={submitting}> Cancel </button>
              </div>
            </form>
          </div>
        </div>
      )}

      <ConfirmModal
        isOpen={!!confirmDeleteDept}
        onClose={() => setConfirmDeleteDept(null)}
        onConfirm={confirmActualDeptDelete}
        isLoading={submitting}
        title="Are you absolutely sure?"
        message={
          confirmDeleteDept ? (
            <>
              You are about to delete department <strong>{confirmDeleteDept.name}</strong>.
              This may affect employees linked to this department.
            </>
          ) : (
            ""
          )
        }
        confirmText="Yes, Delete Department"
      />

      <ConfirmModal
        isOpen={!!confirmDeleteDesig}
        onClose={() => setConfirmDeleteDesig(null)}
        onConfirm={confirmActualDesigDelete}
        isLoading={submitting}
        title="Are you absolutely sure?"
        message={
          confirmDeleteDesig ? (
            <>
              You are about to delete designation <strong>{confirmDeleteDesig.title}</strong>.
              This may affect employees linked to this designation.
            </>
          ) : (
            ""
          )
        }
        confirmText="Yes, Delete Designation"
      />
    </>
  );
}
