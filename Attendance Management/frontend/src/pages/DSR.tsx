import { useCallback, useEffect, useMemo, useState } from "react";
import { createPortal } from "react-dom";
import {
  dsr as dsrApi,
  employees as employeesApi,
  type DSRRow,
  type DSRSummaryRow,
  type DSRReminderSettings,
  type DSRPendingEmployee,
} from "../api/client";
import { useAuth } from "../auth/AuthContext";
import CustomSelect from "../components/CustomSelect";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import ConfirmModal from "../components/ConfirmModal";
import { SectionLoader } from "../components/LoadingState";
import { formatDate } from "../utils/dateFormatter";
import { formatApiError } from "../utils/apiError";

type EmployeeLite = {
  id: number;
  employee_code?: string | null;
  first_name?: string | null;
  last_name?: string | null;
  full_name?: string | null;
  designation_id?: number | null;
  department_id?: number | null;
};

// ---------------------------------------------------------------------------
// Icons - inline SVGs matching the rest of HRMS
// ---------------------------------------------------------------------------
const Icons = {
  Document: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <polyline points="14 2 14 8 20 8" />
      <line x1="16" y1="13" x2="8" y2="13" />
      <line x1="16" y1="17" x2="8" y2="17" />
    </svg>
  ),
  Plus: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  ),
  Send: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  ),
  Save: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z" />
      <polyline points="17 21 17 13 7 13 7 21" />
      <polyline points="7 3 7 8 15 8" />
    </svg>
  ),
  Eye: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  ),
  Clock: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
    </svg>
  ),
  Trash: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <line x1="10" y1="11" x2="10" y2="17" />
      <line x1="14" y1="11" x2="14" y2="17" />
    </svg>
  ),
  Edit: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
    </svg>
  ),
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function todayIso(): string {
  const d = new Date();
  const y = d.getFullYear();
  const m = String(d.getMonth() + 1).padStart(2, "0");
  const day = String(d.getDate()).padStart(2, "0");
  return `${y}-${m}-${day}`;
}

function prettyDate(iso: string): string {
  try {
    return formatDate(iso);
  } catch {
    return iso;
  }
}


interface FormState {
  report_date: string;
  project_work: string;
  work_location: string;
  work_done: string;
  plan_for_tomorrow: string;
}

const EMPTY_FORM: FormState = {
  report_date: todayIso(),
  project_work: "",
  work_location: "",
  work_done: "",
  plan_for_tomorrow: "",
};

const MONTHS = [
  "January", "February", "March", "April", "May", "June",
  "July", "August", "September", "October", "November", "December",
];

function prettyTime(iso: string | null): string {
  if (!iso) return "â€”";
  try {
    return new Date(iso).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  } catch {
    return iso;
  }
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function DSR() {
  const { user, hasRole } = useAuth();
  const isAdminOrHR = hasRole("Admin", "HR");
  const isAdmin = hasRole("Admin");
  // Admin role is view-only on DSRs. HR (with or without Admin) keeps full management
  // power. Use this for buttons that create / edit / delete DSRs.
  const canMutateDsr = hasRole("HR") || !isAdmin;
  const canSeeAll = isAdminOrHR;
  const now = new Date();
  const [year, setYear] = useState(now.getFullYear());
  const [month, setMonth] = useState(now.getMonth() + 1);

  const [view, setView] = useState<"mine" | "all" | "pending">("mine");
  const [rows, setRows] = useState<DSRRow[]>([]);
  const [allRows, setAllRows] = useState<DSRRow[]>([]);
  const [summary, setSummary] = useState<DSRSummaryRow | null>(null);
  const [loading, setLoading] = useState(true);
  const [allLoading, setAllLoading] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [showForm, setShowForm] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<DSRRow | null>(null);
  const [viewingDsr, setViewingDsr] = useState<DSRRow | null>(null);
  const [showFormError, setShowFormError] = useState(false);
  const [banner, setBanner] = useState<{ kind: "ok" | "err"; text: string } | null>(null);
  const [showReminderSettings, setShowReminderSettings] = useState(false);
  const [showDraftsNotepad, setShowDraftsNotepad] = useState(false);

  // Filters for All-DSRs view (Admin/HR/Manager only)
  const [employeesList, setEmployeesList] = useState<EmployeeLite[]>([]);
  const [filterEmployeeId, setFilterEmployeeId] = useState<string>("");
  const [filterStatus, setFilterStatus] = useState<string>("");
  const [designations, setDesignations] = useState<Array<{ id: number; title: string }>>([]);
  const [searchName, setSearchName] = useState<string>("");
  const [filterDate, setFilterDate] = useState<string>("");

  const [form, setForm] = useState<FormState>(EMPTY_FORM);

  // Recent DSRs table excludes DRAFT rows — drafts are surfaced in the
  // notepad opened from the "Draft" summary tile.
  const submittedRows = useMemo(
    () => rows.filter((r) => r.status === "SUBMITTED"),
    [rows]
  );
  const recent = useMemo(() => submittedRows.slice(0, 6), [submittedRows]);
  const draftRows = useMemo(
    () => rows.filter((r) => r.status === "DRAFT"),
    [rows]
  );
  const isOwnDsr = useCallback(
    (row: DSRRow) => user?.employee_id != null && row.employee_id === user.employee_id,
    [user?.employee_id]
  );

  const filteredAllRows = useMemo(() => {
    // Drafts are private to each employee — they should never appear in the
    // "All DSRs" view used by HR/Admin/Manager.
    let res = allRows.filter((r) => r.status !== "DRAFT");
    if (searchName.trim()) {
      const term = searchName.toLowerCase();
      res = res.filter((r) =>
        (r.employee_name || "").toLowerCase().includes(term)
      );
    }
    if (filterDate) {
      res = res.filter((r) => r.report_date === filterDate);
    }
    return res;
  }, [allRows, searchName, filterDate]);

  // -------------------------------------------------------------------------
  // Loaders
  // -------------------------------------------------------------------------
  const loadList = useCallback(
    (silent = false) => {
      if (!silent) setLoading(true);
      Promise.all([
        dsrApi.mine({ year, month, limit: 50 }),
        dsrApi.summary({ year, month }),
      ])
        .then(([listRes, sumRes]) => {
          setRows(listRes.data || []);
          setSummary(sumRes.data || null);
        })
        .catch(() => {
          setRows([]);
          setSummary(null);
        })
        .finally(() => setLoading(false));
    },
    [year, month]
  );

  useEffect(() => {
    loadList();
  }, [loadList]);

  // Load all DSRs for Admin/HR/Manager when "All" tab is active or filters change.
  const loadAll = useCallback(() => {
    if (!canSeeAll) return;
    setAllLoading(true);
    dsrApi
      .listAll({
        year,
        month,
        employee_id: filterEmployeeId ? Number(filterEmployeeId) : undefined,
        status: filterStatus || undefined,
        limit: 500,
      })
      .then((res) => setAllRows(res.data || []))
      .catch(() => setAllRows([]))
      .finally(() => setAllLoading(false));
  }, [canSeeAll, year, month, filterEmployeeId, filterStatus]);

  useEffect(() => {
    if (view === "all") loadAll();
  }, [view, loadAll]);

  // Load employees list once, only for Admin/HR/Manager.
  useEffect(() => {
    if (!canSeeAll) return;
    employeesApi
      .list({ status: "Active" })
      .then((res) => {
        const list: EmployeeLite[] = (res.data || []).map((e: any) => ({
          id: e.id,
          employee_code: e.employee_code,
          first_name: e.first_name,
          last_name: e.last_name,
          designation_id: e.designation_id,
          department_id: e.department_id,
          full_name:
            e.full_name ||
            [e.first_name, e.last_name].filter(Boolean).join(" ").trim() ||
            null,
        }));
        setEmployeesList(list);
      })
      .catch(() => setEmployeesList([]));
  }, [canSeeAll]);

  // Load designations list once, only for Admin/HR/Manager.
  useEffect(() => {
    if (!canSeeAll) return;
    employeesApi
      .designations()
      .then((res) => setDesignations(res.data || []))
      .catch(() => setDesignations([]));
  }, [canSeeAll]);

  // Banner auto-clear
  useEffect(() => {
    if (!banner) return;
    const t = setTimeout(() => setBanner(null), 4000);
    return () => clearTimeout(t);
  }, [banner]);

  // -------------------------------------------------------------------------
  // Form handlers
  // -------------------------------------------------------------------------
  const setField = <K extends keyof FormState>(key: K, value: FormState[K]) => {
    setForm((f) => ({ ...f, [key]: value }));
  };

  const resetForm = () => {
    setForm({ ...EMPTY_FORM, report_date: todayIso() });
    setEditingId(null);
    setShowFormError(false);
  };

  const beginEdit = (row: DSRRow) => {
    setEditingId(row.id);
    setShowForm(true);
    setForm({
      report_date: row.report_date,
      project_work: row.project_work || "",
      work_location: row.work_location || "",
      work_done: row.work_done || "",
      plan_for_tomorrow: row.plan_for_tomorrow || "",
    });
    setShowFormError(false);
    requestAnimationFrame(() => window.scrollTo({ top: 0, behavior: "smooth" }));
  };

  const buildPayload = (status: "DRAFT" | "SUBMITTED") => ({
    report_date: form.report_date,
    project_work: form.project_work.trim() || null,
    work_location: form.work_location.trim() || null,
    work_done: form.work_done.trim(),
    plan_for_tomorrow: form.plan_for_tomorrow.trim() || null,
    status,
  });

  const validate = () => form.work_done.trim().length > 0;

  const saveDraft = async () => {
    if (!validate()) {
      setShowFormError(true);
      return;
    }
    setShowFormError(false);
    setSubmitting(true);
    try {
      const payload = buildPayload("DRAFT");
      if (editingId) {
        await dsrApi.update(editingId, payload);
        setBanner({ kind: "ok", text: "Draft saved." });
      } else {
        await dsrApi.create(payload);
        setBanner({ kind: "ok", text: "Draft saved." });
      }
      resetForm();
      setShowForm(false);
      loadList(true);
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        "Could not save draft.";
      setBanner({ kind: "err", text: msg });
    } finally {
      setSubmitting(false);
    }
  };

  // Save a draft straight from the Notepad modal (no Add DSR form involved).
  // If `id` is provided it updates that draft; otherwise it creates a new one.
  const saveDraftFromNotepad = async (params: {
    id?: number;
    report_date: string;
    work_done: string;
  }): Promise<boolean> => {
    const text = (params.work_done || "").trim();
    if (!text) {
      setBanner({ kind: "err", text: "Please write something before saving." });
      return false;
    }
    try {
      const payload = {
        report_date: params.report_date,
        project_work: null,
        work_location: null,
        work_done: text,
        plan_for_tomorrow: null,
        status: "DRAFT" as const,
      };
      if (params.id) {
        await dsrApi.update(params.id, payload);
      } else {
        await dsrApi.create(payload);
      }
      setBanner({ kind: "ok", text: "Draft saved." });
      loadList(true);
      return true;
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        "Could not save draft.";
      setBanner({ kind: "err", text: msg });
      return false;
    }
  };

  const submitDsr = async () => {
    if (!validate()) {
      setShowFormError(true);
      return;
    }
    setShowFormError(false);
    setSubmitting(true);
    try {
      const payload = buildPayload("SUBMITTED");
      if (editingId) {
        await dsrApi.update(editingId, payload);
      } else {
        await dsrApi.create(payload);
      }
      setBanner({ kind: "ok", text: "DSR submitted successfully." });
      resetForm();
      setShowForm(false);
      loadList(true);
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        "Could not submit DSR.";
      setBanner({ kind: "err", text: msg });
    } finally {
      setSubmitting(false);
    }
  };

  const confirmActualDelete = async () => {
    if (!confirmDelete) return;
    const id = confirmDelete.id;
    try {
      await dsrApi.remove(id);
      setConfirmDelete(null);
      setBanner({ kind: "ok", text: "DSR deleted." });
      loadList(true);
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail ||
        "Could not delete DSR.";
      setConfirmDelete(null);
      setBanner({ kind: "err", text: msg });
    }
  };

  // -------------------------------------------------------------------------
  // Year options (last 3 years + next 1)
  // -------------------------------------------------------------------------
  const yearOptions = useMemo(() => {
    const y = now.getFullYear();
    return [y - 2, y - 1, y, y + 1].map((v) => ({ value: String(v), label: String(v) }));
  }, [now]);

  const monthOptions = MONTHS.map((m, i) => ({ value: String(i + 1), label: m }));

  // -------------------------------------------------------------------------
  // Render
  // -------------------------------------------------------------------------
  return (
    <div>
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="page-title">Daily Status Report</h1>
          <div className="page-subtitle">Track your daily work, progress, and plan for tomorrow.</div>
        </div>
        <GlobalHeaderControls />
      </div>

      {banner && (
        <div
          className="card"
          style={{
            marginBottom: "1rem",
            padding: "0.8rem 1rem",
            borderColor: banner.kind === "ok" ? "rgb(16 185 129 / 0.4)" : "rgb(239 68 68 / 0.4)",
            background: banner.kind === "ok" ? "rgba(16, 185, 129, 0.08)" : "rgba(239, 68, 68, 0.08)",
            color: banner.kind === "ok" ? "#a7f3d0" : "#fecaca",
            fontSize: "0.875rem",
            fontWeight: 600,
          }}
        >
          {banner.text}
        </div>
      )}

      {/* Tabs (only visible to Admin/HR/Manager) */}
      {canSeeAll && (
        <div
          style={{
            display: "flex",
            gap: "0.4rem",
            marginBottom: "1rem",
            borderBottom: "1px solid rgba(255,255,255,0.08)",
          }}
        >
          <TabButton
            active={view === "mine"}
            onClick={() => {
              setView("mine");
              resetForm();
              setShowForm(false);
            }}
            label="My DSRs"
          />
          <TabButton
            active={view === "all"}
            onClick={() => {
              setView("all");
              setShowForm(false);
            }}
            label="All DSRs"
          />
          <TabButton
            active={view === "pending"}
            onClick={() => {
              setView("pending");
              setShowForm(false);
            }}
            label="Pending Today"
          />
        </div>
      )}

      {/* Filter & Add toolbar */}
      <div
        style={{
          display: "flex",
          gap: "0.75rem",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: "1rem",
          flexWrap: "wrap",
        }}
      >
        {/* Left side: month / year / extra filters */}
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", flexWrap: "wrap" }}>
          {/* <span style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.55)", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.05em" }}>
            View
          </span> */}
          <CustomSelect
            value={String(month)}
            onChange={(v) => setMonth(Number(v))}
            options={monthOptions}
            style={{ width: 150 }}
          />
          <CustomSelect
            value={String(year)}
            onChange={(v) => setYear(Number(v))}
            options={yearOptions}
            style={{ width: 110 }}
          />

          {view === "all" && (
            <>
              <input
                type="text"
                className="form-control"
                value={searchName}
                onChange={(e) => setSearchName(e.target.value)}
                placeholder="Search by name..."
                style={{
                  width: 170,
                  background: "rgba(255, 255, 255, 0.05)",
                  border: "1px solid rgba(255, 255, 255, 0.1)",
                  borderRadius: 8,
                  padding: "0.45rem 0.75rem",
                  color: "#fff",
                  fontSize: "0.82rem",
                  height: "38px",
                  outline: "none",
                }}
              />
              <input
                type="date"
                className="form-control"
                value={filterDate}
                onChange={(e) => setFilterDate(e.target.value)}
                style={{
                  width: 140,
                  background: "rgba(255, 255, 255, 0.05)",
                  border: "1px solid rgba(255, 255, 255, 0.1)",
                  borderRadius: 8,
                  padding: "0.45rem 0.75rem",
                  color: "#fff",
                  fontSize: "0.82rem",
                  height: "38px",
                  outline: "none",
                }}
              />
              <CustomSelect
                value={filterEmployeeId}
                onChange={(v) => setFilterEmployeeId(v)}
                options={[
                  { value: "", label: "All employees" },
                  ...employeesList.map((e) => ({
                    value: String(e.id),
                    label:
                      (e.full_name ||
                        [e.first_name, e.last_name].filter(Boolean).join(" ").trim() ||
                        `Employee #${e.id}`) +
                      (e.employee_code ? ` (${e.employee_code})` : ""),
                  })),
                ]}
                style={{ width: 200 }}
              />
              <CustomSelect
                value={filterStatus}
                onChange={(v) => setFilterStatus(v)}
                options={[
                  { value: "", label: "All statuses" },
                  { value: "SUBMITTED", label: "Submitted" },
                ]}
                style={{ width: 140 }}
              />
              {(searchName || filterDate || filterEmployeeId || filterStatus) && (
                <button
                  type="button"
                  className="btn btn-secondary btn-sm"
                  style={{
                    height: "38px",
                    padding: "0 0.8rem",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    fontWeight: 600,
                    borderRadius: 8,
                  }}
                  onClick={() => {
                    setSearchName("");
                    setFilterDate("");
                    setFilterEmployeeId("");
                    setFilterStatus("");
                  }}
                >
                  Clear
                </button>
              )}
            </>
          )}
        </div>

        {/* Right side: action buttons */}
        {view === "mine" && (
          <div style={{ display: "flex", gap: "0.5rem", alignItems: "center" }}>
            {/* <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => {
                resetForm();
                setShowForm(false);
                requestAnimationFrame(() => window.scrollTo({ top: 0, behavior: "smooth" }));
              }}
              title="Hide the form and show your recent DSRs"
            >
              <Icons.Eye /> &nbsp; View My DSRs
            </button> */}
            {isAdminOrHR && (
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                style={{ padding: "0.55rem 1.0rem" }}
                onClick={() => setShowReminderSettings(true)}
                title="Configure when the daily DSR reminder is sent"
              >
                <Icons.Clock /> &nbsp; Reminder
              </button>
            )}
            {canMutateDsr && (
              <button
                type="button"
                className="btn btn-primary btn-sm"
                style={{ padding: "0.55rem 1.2rem" }}
                onClick={() => {
                  if (showForm) {
                    if (!editingId) {
                      setShowForm(false);
                      return;
                    }
                  }
                  resetForm();
                  setShowForm(true);
                  requestAnimationFrame(() => window.scrollTo({ top: 0, behavior: "smooth" }));
                }}
              >
                <Icons.Plus /> &nbsp; {showForm && !editingId ? "Close" : "Add DSR"}
              </button>
            )}
          </div>
        )}
      </div>

      {view === "pending" ? (
        <PendingTodayView onBanner={(b) => setBanner(b)} />
      ) : view === "all" ? (
        <AllDsrsView
          rows={filteredAllRows}
          loading={allLoading}
          year={year}
          month={month}
          canEdit={canMutateDsr && isAdminOrHR}
          employeesList={employeesList}
          designations={designations}
          onView={(row) => setViewingDsr(row)}
          onDelete={(row) => setConfirmDelete(row)}
        />
      ) : (
        <div>
          {/* Compact horizontal summary strip */}
          <DsrSummaryStrip
            summary={summary}
            year={year}
            month={month}
            onDraftClick={() => setShowDraftsNotepad(true)}
          />

          {/* Recent DSRs table */}
          <div className="card" style={{ padding: 0, overflow: "hidden" }}>
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "1rem 1.25rem",
                borderBottom: "1px solid rgba(255,255,255,0.06)",
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <Icons.Document />
                <h3 style={{ margin: 0, fontSize: "0.95rem", fontWeight: 800, color: "#fff" }}>
                  Recent DSRs
                </h3>
              </div>
              <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.55)", fontWeight: 600 }}>
                {MONTHS[month - 1]} {year}
              </div>
            </div>

            {loading ? (
              <div style={{ padding: "2.5rem 0" }}>
                <SectionLoader size="md" />
              </div>
            ) : recent.length === 0 ? (
              <div style={{ padding: "2.5rem 1.25rem", textAlign: "center", color: "rgba(255,255,255,0.6)" }}>
                No DSRs filed yet for {MONTHS[month - 1]} {year}. Use the form above to create one.
              </div>
            ) : (
              <div className="table-wrap table-wrap--dark dsr-recent-table">
                <table className="table-modern table-modern--dark" style={{ tableLayout: "fixed", width: "100%" }}>
                  <colgroup>
                    <col style={{ width: "22%" }} />
                    <col style={{ width: "22%" }} />
                    <col style={{ width: "18%" }} />
                    <col style={{ width: "18%" }} />
                    <col style={{ width: "20%" }} />
                  </colgroup>
                  <thead>
                    <tr>
                      <th style={{ textAlign: "center" }}>Date</th>
                      <th style={{ textAlign: "center" }}>Designation</th>
                      <th style={{ textAlign: "center" }}>Status</th>
                      <th style={{ textAlign: "center" }}>Submitted On</th>
                      <th style={{ textAlign: "center" }}>Action</th>
                    </tr>
                  </thead>
                  <tbody>
                    {recent.map((r) => (
                      <tr key={r.id}>
                        <td style={{ textAlign: "center" }}>{prettyDate(r.report_date)}</td>
                        <td style={{ textAlign: "center" }}>{r.designation || user?.designation || "-"}</td>
                        <td style={{ textAlign: "center" }}>
                          <div style={{ display: "inline-flex", justifyContent: "center" }}>
                            <StatusPill status={r.status} />
                          </div>
                        </td>
                        <td style={{ textAlign: "center" }}>{prettyTime(r.submitted_at)}</td>
                        <td style={{ textAlign: "center" }}>
                          <div style={{ display: "inline-flex", gap: "0.45rem", justifyContent: "center", alignItems: "center" }}>
                            <button
                              type="button"
                              className="btn btn-icon-action btn-icon-action--neutral"
                              onClick={() => setViewingDsr(r)}
                              title="View details"
                              aria-label="View details"
                            >
                              <Icons.Eye />
                            </button>
                            {isOwnDsr(r) && (
                              <button
                                type="button"
                                className="btn btn-icon-action btn-icon-action--neutral"
                                onClick={() => beginEdit(r)}
                                title="Edit your DSR"
                                aria-label="Edit"
                              >
                                <Icons.Edit />
                              </button>
                            )}
                            {isOwnDsr(r) && (
                              <button
                                type="button"
                                className="btn btn-icon-action btn-icon-action--danger"
                                onClick={() => setConfirmDelete(r)}
                                title="Delete your DSR"
                                aria-label="Delete"
                              >
                                <Icons.Trash />
                              </button>
                            )}
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {user && (
            <div
              style={{
                marginTop: "0.75rem",
                fontSize: "0.72rem",
                color: "rgba(255,255,255,0.5)",
                textAlign: "right",
              }}
            >
              Filing as{" "}
              {/* <span style={{ color: "rgba(255,255,255,0.85)", fontWeight: 700 }}>
                {user.username}
              </span> */}
              {user.employee_code ? ` (${user.employee_code})` : ""}
            </div>
          )}
        </div>
      )}

      {/* Add / Edit DSR popup */}
      {showForm && (
        <DsrFormModal
          editing={editingId !== null}
          form={form}
          submitting={submitting}
          showFormError={showFormError}
          onClose={() => {
            resetForm();
            setShowForm(false);
          }}
          onField={setField}
          onSaveDraft={saveDraft}
          onSubmit={submitDsr}
        />
      )}

      {viewingDsr && (
        <DsrViewModal row={viewingDsr} onClose={() => setViewingDsr(null)} />
      )}

      {showReminderSettings && (
        <ReminderSettingsModal
          onClose={() => setShowReminderSettings(false)}
          onSaved={(s) =>
            setBanner({
              kind: "ok",
              text: s.enabled
                ? `DSR reminder set to ${s.time} IST on ${s.weekdays.join(", ")}.`
                : "DSR reminder disabled.",
            })
          }
        />
      )}

      {showDraftsNotepad && (
        <DraftsNotepadModal
          drafts={draftRows}
          onClose={() => setShowDraftsNotepad(false)}
          onSave={saveDraftFromNotepad}
          onOpenInForm={(row) => {
            setShowDraftsNotepad(false);
            beginEdit(row);
          }}
          onDelete={(row) => {
            setShowDraftsNotepad(false);
            setConfirmDelete(row);
          }}
        />
      )}

      <ConfirmModal
        isOpen={!!confirmDelete}
        onClose={() => setConfirmDelete(null)}
        onConfirm={confirmActualDelete}
        title="Delete DSR"
        message={
          confirmDelete
            ? `Delete the DSR for ${prettyDate(confirmDelete.report_date)}? This cannot be undone.`
            : ""
        }
        confirmText="Yes, Delete"
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function StatusPill({ status }: { status: "DRAFT" | "SUBMITTED" }) {
  const submitted = status === "SUBMITTED";
  return (
    <span
      style={{
        display: "inline-block",
        padding: "2px 10px",
        borderRadius: 999,
        fontSize: "0.7rem",
        fontWeight: 700,
        background: submitted ? "rgba(16, 185, 129, 0.18)" : "rgba(234, 179, 8, 0.18)",
        color: submitted ? "#a7f3d0" : "#fde68a",
        border: `1px solid ${submitted ? "rgba(16, 185, 129, 0.4)" : "rgba(234, 179, 8, 0.4)"}`,
        textTransform: "capitalize",
      }}
    >
      {submitted ? "Submitted" : "Draft"}
    </span>
  );
}

// Writable notepad modal for DSR drafts.
// - Top section: composer for a brand-new draft (date + note + Save).
// - Below: existing drafts shown as inline-editable cards.
function DraftsNotepadModal({
  drafts,
  onClose,
  onSave,
  onOpenInForm,
  onDelete,
}: {
  drafts: DSRRow[];
  onClose: () => void;
  onSave: (params: {
    id?: number;
    report_date: string;
    work_done: string;
  }) => Promise<boolean>;
  onOpenInForm: (row: DSRRow) => void;
  onDelete: (row: DSRRow) => void;
}) {
  const sortedDrafts = useMemo(
    () =>
      [...drafts].sort(
        (a, b) =>
          new Date(b.report_date + "T12:00:00").getTime() -
          new Date(a.report_date + "T12:00:00").getTime()
      ),
    [drafts]
  );

  // Composer (new draft) state
  const [newDate, setNewDate] = useState<string>(todayIso());
  const [newText, setNewText] = useState<string>("");
  const [savingNew, setSavingNew] = useState(false);

  // Inline edit state per draft id
  const [editText, setEditText] = useState<Record<number, string>>({});
  const [savingId, setSavingId] = useState<number | null>(null);

  const fullText = (d: DSRRow) => {
    const parts: string[] = [];
    if (d.project_work) parts.push(`Project / work: ${d.project_work}`);
    if (d.work_location) parts.push(`Work location: ${d.work_location}`);
    if (d.work_done) parts.push(d.work_done);
    if (d.plan_for_tomorrow)
      parts.push(`Plan for tomorrow:\n${d.plan_for_tomorrow}`);
    return parts.join("\n\n");
  };

  const getValue = (d: DSRRow): string =>
    editText[d.id] !== undefined ? editText[d.id] : fullText(d);

  const isDirty = (d: DSRRow): boolean =>
    editText[d.id] !== undefined && editText[d.id].trim() !== fullText(d).trim();

  const handleSaveNew = async () => {
    if (!newText.trim()) return;
    setSavingNew(true);
    const ok = await onSave({ report_date: newDate, work_done: newText.trim() });
    setSavingNew(false);
    if (ok) {
      setNewText("");
      setNewDate(todayIso());
    }
  };

  const handleSaveExisting = async (d: DSRRow) => {
    const text = (editText[d.id] ?? "").trim();
    if (!text) return;
    setSavingId(d.id);
    const ok = await onSave({
      id: d.id,
      report_date: d.report_date,
      work_done: text,
    });
    setSavingId(null);
    if (ok) {
      setEditText((prev) => {
        const next = { ...prev };
        delete next[d.id];
        return next;
      });
    }
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal"
        onClick={(e) => e.stopPropagation()}
        style={{
          maxWidth: 720,
          width: "100%",
          padding: 0,
          overflow: "hidden",
        }}
      >
        {/* Header */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            padding: "1rem 1.25rem",
            background: "rgba(255,255,255,0.04)",
            borderBottom: "1px solid rgba(255,255,255,0.08)",
            gap: "0.75rem",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: "0.65rem",
              minWidth: 0,
            }}
          >
            <span
              style={{
                display: "inline-grid",
                placeItems: "center",
                width: 36,
                height: 36,
                borderRadius: 10,
                background: "rgba(var(--brand-rgb) / 0.14)",
                color: "var(--brand-300)",
                border: "1px solid rgba(var(--brand-rgb) / 0.3)",
              }}
            >
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="18"
                height="18"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
              >
                <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
                <polyline points="14 2 14 8 20 8" />
                <line x1="16" y1="13" x2="8" y2="13" />
                <line x1="16" y1="17" x2="8" y2="17" />
              </svg>
            </span>
            <div style={{ minWidth: 0 }}>
              <h3
                style={{
                  margin: 0,
                  fontSize: "1.05rem",
                  fontWeight: 800,
                  color: "#fff",
                }}
              >
                My DSR Drafts
              </h3>
              <div
                style={{
                  fontSize: "0.75rem",
                  color: "rgba(255,255,255,0.6)",
                  marginTop: 2,
                }}
              >
                Write here and save — your drafts won't be visible to anyone else.
              </div>
            </div>
          </div>
          <button
            type="button"
            className="btn btn-icon-action btn-icon-action--neutral"
            onClick={onClose}
            title="Close"
            aria-label="Close"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="16"
              height="16"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div
          style={{
            maxHeight: "75vh",
            overflowY: "auto",
            padding: "1rem 1.25rem 1.25rem",
          }}
        >
          {/* New draft composer */}
          <div
            style={{
              padding: "0.9rem 1rem 1rem",
              borderRadius: 12,
              background: "rgba(255,255,255,0.03)",
              border: "1px solid rgba(255,255,255,0.08)",
              display: "flex",
              flexDirection: "column",
              gap: "0.65rem",
              marginBottom: "1rem",
            }}
          >
            <div
              style={{
                display: "flex",
                gap: "0.75rem",
                alignItems: "center",
                flexWrap: "wrap",
                justifyContent: "space-between",
              }}
            >
              <div
                style={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: "0.5rem",
                  color: "#fff",
                  fontWeight: 700,
                  fontSize: "0.9rem",
                }}
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  style={{ color: "var(--brand-300)" }}
                >
                  <line x1="12" y1="5" x2="12" y2="19" />
                  <line x1="5" y1="12" x2="19" y2="12" />
                </svg>
                Write a new draft
              </div>
              <input
                type="date"
                value={newDate}
                onChange={(e) => setNewDate(e.target.value)}
                style={{
                  padding: "0.4rem 0.55rem",
                  borderRadius: 8,
                  background: "rgba(0,0,0,0.35)",
                  border: "1px solid rgba(255,255,255,0.12)",
                  color: "#fff",
                  fontSize: "0.8rem",
                  outline: "none",
                }}
              />
            </div>
            <textarea
              value={newText}
              onChange={(e) => setNewText(e.target.value)}
              placeholder="Type your note here... (e.g. tasks worked on, blockers, plan for tomorrow)"
              rows={5}
              style={{
                width: "100%",
                resize: "vertical",
                minHeight: 120,
                padding: "0.75rem 0.85rem",
                borderRadius: 10,
                background: "rgba(0,0,0,0.35)",
                border: "1px solid rgba(255,255,255,0.1)",
                color: "rgba(255,255,255,0.95)",
                fontSize: "0.88rem",
                lineHeight: 1.55,
                fontFamily: "inherit",
                outline: "none",
                boxSizing: "border-box",
              }}
            />
            <div
              style={{
                display: "flex",
                justifyContent: "flex-end",
                gap: "0.5rem",
              }}
            >
              <button
                type="button"
                className="btn btn-primary btn-sm"
                onClick={handleSaveNew}
                disabled={!newText.trim() || savingNew}
                title="Save this as a draft"
              >
                {savingNew ? "Saving..." : "Save Draft"}
              </button>
            </div>
          </div>

          {/* Existing drafts */}
          {sortedDrafts.length > 0 && (
            <div
              style={{
                fontSize: "0.75rem",
                fontWeight: 700,
                color: "rgba(255,255,255,0.55)",
                textTransform: "uppercase",
                letterSpacing: "0.06em",
                margin: "0.5rem 0 0.75rem",
              }}
            >
              {sortedDrafts.length} saved draft
              {sortedDrafts.length === 1 ? "" : "s"}
            </div>
          )}

          <div
            style={{
              display: "flex",
              flexDirection: "column",
              gap: "0.85rem",
            }}
          >
            {sortedDrafts.map((d) => {
              const dirty = isDirty(d);
              return (
                <div
                  key={d.id}
                  style={{
                    padding: "0.85rem 1rem",
                    borderRadius: 12,
                    background: "rgba(255,255,255,0.025)",
                    border: dirty
                      ? "1px solid rgba(var(--brand-rgb) / 0.4)"
                      : "1px solid rgba(255,255,255,0.08)",
                    display: "flex",
                    flexDirection: "column",
                    gap: "0.6rem",
                  }}
                >
                  <div
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      gap: "0.75rem",
                      flexWrap: "wrap",
                    }}
                  >
                    <div
                      style={{
                        display: "inline-flex",
                        alignItems: "center",
                        gap: "0.5rem",
                        color: "#fff",
                        fontWeight: 700,
                        fontSize: "0.9rem",
                      }}
                    >
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="14"
                        height="14"
                        viewBox="0 0 24 24"
                        fill="none"
                        stroke="currentColor"
                        strokeWidth="2"
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        style={{ color: "var(--brand-300)" }}
                      >
                        <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
                        <line x1="16" y1="2" x2="16" y2="6" />
                        <line x1="8" y1="2" x2="8" y2="6" />
                        <line x1="3" y1="10" x2="21" y2="10" />
                      </svg>
                      {prettyDate(d.report_date)}
                      {dirty && (
                        <span
                          style={{
                            fontSize: "0.65rem",
                            padding: "1px 7px",
                            borderRadius: 999,
                            background: "rgba(var(--brand-rgb) / 0.18)",
                            color: "var(--brand-300)",
                            border: "1px solid rgba(var(--brand-rgb) / 0.4)",
                            fontWeight: 700,
                            textTransform: "uppercase",
                            letterSpacing: "0.04em",
                          }}
                        >
                          Unsaved
                        </span>
                      )}
                    </div>
                    <div style={{ display: "inline-flex", gap: "0.4rem" }}>
                      <button
                        type="button"
                        className="btn btn-primary btn-sm"
                        onClick={() => handleSaveExisting(d)}
                        disabled={!dirty || savingId === d.id}
                        title="Save changes"
                      >
                        {savingId === d.id ? "Saving..." : "Save"}
                      </button>
                      <button
                        type="button"
                        className="btn btn-icon-action btn-icon-action--neutral"
                        onClick={() => onOpenInForm(d)}
                        title="Open in full DSR form"
                        aria-label="Open in full form"
                      >
                        <Icons.Edit />
                      </button>
                      <button
                        type="button"
                        className="btn btn-icon-action btn-icon-action--danger"
                        onClick={() => onDelete(d)}
                        title="Delete this draft"
                        aria-label="Delete draft"
                      >
                        <Icons.Trash />
                      </button>
                    </div>
                  </div>
                  <textarea
                    value={getValue(d)}
                    onChange={(e) =>
                      setEditText((prev) => ({ ...prev, [d.id]: e.target.value }))
                    }
                    rows={5}
                    style={{
                      width: "100%",
                      resize: "vertical",
                      minHeight: 110,
                      padding: "0.75rem 0.85rem",
                      borderRadius: 10,
                      background: "rgba(0,0,0,0.35)",
                      border: "1px solid rgba(255,255,255,0.1)",
                      color: "rgba(255,255,255,0.92)",
                      fontSize: "0.86rem",
                      lineHeight: 1.55,
                      fontFamily: "inherit",
                      outline: "none",
                      boxSizing: "border-box",
                    }}
                  />
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

function DsrSummaryStrip({
  summary,
  year,
  month,
  showDraft = true,
  onDraftClick,
}: {
  summary: DSRSummaryRow | null;
  year: number;
  month: number;
  showDraft?: boolean;
  onDraftClick?: () => void;
}) {
  const monthLabel = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
  ][month - 1];

  const cards: Array<{
    label: string;
    value: number;
    iconTint: string;
    iconBg: string;
    iconBorder: string;
    icon: React.ReactNode;
    onClick?: () => void;
    hint?: string;
  }> = [
    {
      label: "Total DSRs",
      value: summary?.total ?? 0,
      iconTint: "#60a5fa",
      iconBg: "rgba(59, 130, 246, 0.14)",
      iconBorder: "rgba(96, 165, 250, 0.28)",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
          <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
        </svg>
      ),
    },
    {
      label: "Submitted",
      value: summary?.submitted ?? 0,
      iconTint: "#34d399",
      iconBg: "rgba(16, 185, 129, 0.14)",
      iconBorder: "rgba(52, 211, 153, 0.28)",
      icon: (
        <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
          <polyline points="14 2 14 8 20 8" />
          <polyline points="9 14 11 16 15 12" />
        </svg>
      ),
    },
    ...(showDraft
      ? [{
          label: "Draft",
          value: summary?.draft ?? 0,
          iconTint: "#c084fc",
          iconBg: "rgba(168, 85, 247, 0.14)",
          iconBorder: "rgba(192, 132, 252, 0.28)",
          onClick: onDraftClick,
          hint: onDraftClick ? "Click to open" : undefined,
          icon: (
            <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
              <line x1="16" y1="13" x2="8" y2="13" />
              <line x1="16" y1="17" x2="8" y2="17" />
            </svg>
          ),
        }]
      : []),
  ];

  return (
    <div
      className="card dsr-summary-card"
      style={{
        marginBottom: "1rem",
        padding: "1rem 1.1rem 1.05rem",
        borderRadius: 14,
      }}
    >
      {/* Title row */}
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          marginBottom: "0.85rem",
          gap: "0.75rem",
          flexWrap: "wrap",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.55rem" }}>
          <span style={{ color: "rgba(255,255,255,0.85)", display: "inline-flex" }}>
            <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
              <polyline points="14 2 14 8 20 8" />
            </svg>
          </span>
          <span
            style={{
              fontSize: "0.98rem",
              fontWeight: 800,
              color: "#fff",
              letterSpacing: "0.01em",
            }}
          >
            DSR Summary
          </span>
        </div>
        <div
          style={{
            display: "inline-flex",
            alignItems: "center",
            gap: "0.45rem",
            padding: "0.38rem 0.75rem",
            borderRadius: 10,
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.08)",
            fontSize: "0.78rem",
            fontWeight: 600,
            color: "rgba(255,255,255,0.85)",
          }}
        >
          <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.85 }}>
            <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
            <line x1="16" y1="2" x2="16" y2="6" />
            <line x1="8" y1="2" x2="8" y2="6" />
            <line x1="3" y1="10" x2="21" y2="10" />
          </svg>
          {monthLabel} {year}
          <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" style={{ opacity: 0.55 }}>
            <polyline points="6 9 12 15 18 9" />
          </svg>
        </div>
      </div>

      {/* Stat tiles */}
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
          gap: "0.85rem",
        }}
        className="dsr-summary-strip"
      >
        {cards.map((c) => {
          const isClickable = typeof c.onClick === "function";
          return (
            <div
              key={c.label}
              role={isClickable ? "button" : undefined}
              tabIndex={isClickable ? 0 : undefined}
              onClick={isClickable ? c.onClick : undefined}
              onKeyDown={
                isClickable
                  ? (e) => {
                      if (e.key === "Enter" || e.key === " ") {
                        e.preventDefault();
                        c.onClick?.();
                      }
                    }
                  : undefined
              }
              style={{
                padding: "0.95rem 1.1rem",
                borderRadius: 12,
                background: "rgba(255,255,255,0.025)",
                border: "1px solid rgba(255,255,255,0.07)",
                display: "flex",
                flexDirection: "column",
                gap: "0.7rem",
                minWidth: 0,
                cursor: isClickable ? "pointer" : "default",
                transition: "background 0.15s, border-color 0.15s, transform 0.1s",
              }}
              onMouseEnter={
                isClickable
                  ? (e) => {
                      (e.currentTarget as HTMLDivElement).style.background =
                        "rgba(255,255,255,0.05)";
                      (e.currentTarget as HTMLDivElement).style.borderColor =
                        c.iconBorder;
                    }
                  : undefined
              }
              onMouseLeave={
                isClickable
                  ? (e) => {
                      (e.currentTarget as HTMLDivElement).style.background =
                        "rgba(255,255,255,0.025)";
                      (e.currentTarget as HTMLDivElement).style.borderColor =
                        "rgba(255,255,255,0.07)";
                    }
                  : undefined
              }
            >
              <span
                style={{
                  width: 38,
                  height: 38,
                  display: "inline-grid",
                  placeItems: "center",
                  borderRadius: 10,
                  background: c.iconBg,
                  color: c.iconTint,
                  border: `1px solid ${c.iconBorder}`,
                  flexShrink: 0,
                }}
              >
                {c.icon}
              </span>
              <div style={{ display: "flex", flexDirection: "column", minWidth: 0, gap: 4 }}>
                <span
                  style={{
                    fontSize: "0.78rem",
                    color: "rgba(255,255,255,0.6)",
                    fontWeight: 500,
                    letterSpacing: "0.01em",
                    whiteSpace: "nowrap",
                    overflow: "hidden",
                    textOverflow: "ellipsis",
                  }}
                >
                  {c.label}
                </span>
                <span
                  style={{
                    fontSize: "1.6rem",
                    fontWeight: 800,
                    color: "#fff",
                    fontVariantNumeric: "tabular-nums",
                    lineHeight: 1,
                  }}
                >
                  {c.value}
                </span>
                {c.hint && (
                  <span
                    style={{
                      fontSize: "0.7rem",
                      color: c.iconTint,
                      fontWeight: 600,
                      marginTop: 2,
                    }}
                  >
                    {c.hint} →
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function DsrFormModal({
  editing,
  form,
  submitting,
  showFormError,
  onClose,
  onField,
  onSaveDraft,
  onSubmit,
}: {
  editing: boolean;
  form: FormState;
  submitting: boolean;
  showFormError: boolean;
  onClose: () => void;
  onField: <K extends keyof FormState>(key: K, value: FormState[K]) => void;
  onSaveDraft: () => void;
  onSubmit: () => void;
}) {
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: 760, width: "100%" }}
      >
        <div className="modal-titlebar">
          <div>
            <h3 className="modal-title" style={{ fontSize: "1.1rem" }}>
              {editing ? "Edit DSR" : "Add Daily Status Report"}
            </h3>
            <div className="modal-subtitle">
              {form.report_date ? prettyDate(form.report_date) : "Today"}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="btn btn-secondary btn-icon btn-sm"
            title="Close"
            disabled={submitting}
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))",
            gap: "1rem",
            marginBottom: "1rem",
          }}
        >
          {/* <div className="form-group" style={{ margin: 0 }}>
            <label>Project / Work</label>
            <input
              type="text"
              value={form.project_work}
              onChange={(e) => onField("project_work", e.target.value)}
              placeholder="What project / module did you work on?"
              maxLength={255}
            />
          </div> */}
          <div className="form-group" style={{ margin: 0 }}>
            <label>Work Location</label>
            <input
              type="text"
              value={form.work_location}
              onChange={(e) => onField("work_location", e.target.value)}
              placeholder="e.g. Office / Remote / Client Site"
              maxLength={50}
            />
          </div>
          <div className="form-group" style={{ margin: 0 }}>
            <label>Report Date</label>
            <input
              type="date"
              value={form.report_date}
              onChange={(e) => onField("report_date", e.target.value)}
              max={todayIso()}
            />
          </div>
        </div>

        <div className="form-group">
          <label>
            1. What did you do today? <span style={{ color: "#ef4444" }}>*</span>
          </label>
          <textarea
            rows={4}
            value={form.work_done}
            onChange={(e) => onField("work_done", e.target.value)}
            placeholder="Describe your work, tasks completed, and key achievements..."
            maxLength={1000}
            style={{
              border: showFormError
                ? "1px solid #ef4444"
                : "1px solid rgba(255,255,255,0.1)",
              boxShadow: showFormError ? "0 0 0 2px rgba(239, 68, 68, 0.2)" : undefined,
            }}
          />
          <div style={{ textAlign: "right", fontSize: "0.7rem", color: "rgba(255,255,255,0.45)", marginTop: 4 }}>
            {form.work_done.length}/1000
          </div>
        </div>

        <div className="form-group" style={{ marginBottom: "1.25rem" }}>
          <label>2. Plan for Tomorrow</label>
          <textarea
            rows={3}
            value={form.plan_for_tomorrow}
            onChange={(e) => onField("plan_for_tomorrow", e.target.value)}
            placeholder="What are your top priorities for tomorrow?"
            maxLength={500}
          />
          <div style={{ textAlign: "right", fontSize: "0.7rem", color: "rgba(255,255,255,0.45)", marginTop: 4 }}>
            {form.plan_for_tomorrow.length}/500
          </div>
        </div>

        <div className="modal-actions" style={{ display: "flex", justifyContent: "flex-end", gap: "0.6rem", borderTop: "1px solid rgba(255,255,255,0.06)", paddingTop: "1rem" }}>

          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={onSaveDraft}
            disabled={submitting}
            title="Save without submitting; you can finish it later"
          >
            {submitting ? "Saving..." : "Save Draft"}
          </button>
          <button
            type="button"
            className="btn btn-primary btn-sm"
            onClick={onSubmit}
            disabled={submitting}
          >
            {submitting ? "Submitting..." : "Submit DSR"}
          </button>
          <button
            type="button"
            className="btn btn-cancel-alt btn-sm"
            onClick={onClose}
            disabled={submitting}
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}

function TabButton({
  active,
  onClick,
  label,
}: {
  active: boolean;
  onClick: () => void;
  label: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      style={{
        background: "transparent",
        border: "none",
        cursor: "pointer",
        padding: "0.7rem 1.1rem",
        fontSize: "0.85rem",
        fontWeight: 700,
        color: active ? "#fff" : "rgba(255,255,255,0.6)",
        borderBottom: active
          ? "2px solid var(--brand-400, #60a5fa)"
          : "2px solid transparent",
        marginBottom: -1,
      }}
    >
      {label}
    </button>
  );
}

function AllDsrsView({
  rows,
  loading,
  year,
  month,
  canEdit,
  employeesList,
  designations,
  onView,
  onDelete,
}: {
  rows: DSRRow[];
  loading: boolean;
  year: number;
  month: number;
  canEdit: boolean;
  employeesList: EmployeeLite[];
  designations: Array<{ id: number; title: string }>;
  onView: (row: DSRRow) => void;
  onDelete: (row: DSRRow) => void;
}) {
  const monthName = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
  ][month - 1];

  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(20);

  // Reset page to 1 when the rows length or content changes
  useEffect(() => {
    setPage(1);
  }, [rows.length]);

  const totalPages = Math.max(1, Math.ceil(rows.length / pageSize));
  const pageStart = (page - 1) * pageSize;
  const pageEnd = pageStart + pageSize;
  const pagedRows = rows.slice(pageStart, pageEnd);

  const pageNumbers = (() => {
    const result: (number | "...")[] = [];
    if (totalPages <= 7) {
      for (let i = 1; i <= totalPages; i++) result.push(i);
      return result;
    }
    const window = new Set<number>([1, totalPages, page, page - 1, page + 1]);
    const sorted = [...window].filter((n) => n >= 1 && n <= totalPages).sort((a, b) => a - b);
    let prev = 0;
    for (const n of sorted) {
      if (prev && n - prev > 1) result.push("...");
      result.push(n);
      prev = n;
    }
    return result;
  })();

  const getEmployeeDesignation = (employeeId: number) => {
    const emp = employeesList.find((e) => e.id === employeeId);
    if (!emp || !emp.designation_id) return "-";
    const des = designations.find((d) => d.id === emp.designation_id);
    return des ? des.title : "-";
  };

  const canDeleteRow = () => canEdit;

  // Build a summary for the four metric tiles. "Pending" here means the
  // number of active employees who have NOT filed a single DSR in the
  // selected month (no submitted, no draft).
  const allSummary: DSRSummaryRow = useMemo(() => {
    const total = rows.length;
    const submitted = rows.filter((r) => r.status === "SUBMITTED").length;
    const draft = rows.filter((r) => r.status === "DRAFT").length;
    const submittedEmployeeIds = new Set<number>(
      rows.filter((r) => r.status === "SUBMITTED").map((r) => r.employee_id)
    );
    const totalActive = employeesList.length;
    const pending = Math.max(0, totalActive - submittedEmployeeIds.size);
    return { year, month, total, submitted, draft, pending };
  }, [rows, employeesList, year, month]);

  return (
    <div>
      <DsrSummaryStrip summary={allSummary} year={year} month={month} showDraft={false} />

      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          padding: "1rem 1.25rem",
          borderBottom: "1px solid rgba(255,255,255,0.06)",
        }}
      >
        <h3 style={{ margin: 0, fontSize: "0.95rem", fontWeight: 800, color: "#fff" }}>
          All Employee DSRs
        </h3>
        <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.55)", fontWeight: 600 }}>
          {monthName} {year} &middot; {rows.length} record{rows.length === 1 ? "" : "s"}
        </div>
      </div>

      {loading ? (
        <div style={{ padding: "2.5rem 0" }}>
          <SectionLoader size="md" />
        </div>
      ) : rows.length === 0 ? (
        <div style={{ padding: "2.5rem 1.25rem", textAlign: "center", color: "rgba(255,255,255,0.6)" }}>
          No DSRs found for the current filters.
        </div>
      ) : (
        <div style={{ padding: "0 0 1rem 0" }}>
          <div className="table-wrap table-wrap--dark">
            <table className="table-modern table-modern--dark" style={{ width: "100%" }}>
              <thead>
                <tr>
                  <th>Employee</th>
                  <th>Designation</th>
                  <th>Date</th>
                  <th>Status</th>
                  <th>Submitted On</th>
                  <th style={{ textAlign: "center" }}>Action</th>
                </tr>
              </thead>
              <tbody>
                {pagedRows.map((r) => (
                  <tr key={r.id}>
                    <td>
                      <div style={{ fontWeight: 700, color: "#fff" }}>
                        {r.employee_name || "—"}
                      </div>
                    </td>
                    <td>{r.designation || getEmployeeDesignation(r.employee_id)}</td>
                    <td>{formatDate(r.report_date)}</td>
                    <td>
                      <StatusPill status={r.status} />
                    </td>
                    <td>{prettyTime(r.submitted_at)}</td>
                    <td style={{ textAlign: "center" }}>
                      <div style={{ display: "inline-flex", gap: "0.45rem", justifyContent: "center", alignItems: "center" }}>
                        <button
                          type="button"
                          className="btn btn-icon-action btn-icon-action--neutral"
                          onClick={() => onView(r)}
                          title="View details"
                          aria-label="View details"
                        >
                          <Icons.Eye />
                        </button>
                        {canDeleteRow() && (
                          <button
                            type="button"
                            className="btn btn-icon-action btn-icon-action--danger"
                            onClick={() => onDelete(r)}
                            title="Delete your DSR"
                            aria-label="Delete"
                          >
                            <Icons.Trash />
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <div
            style={{
              display: 'flex',
              flexWrap: 'wrap',
              justifyContent: 'space-between',
              alignItems: 'center',
              gap: '0.75rem',
              marginTop: '1rem',
              padding: '0.5rem 1.25rem 0 1.25rem',
            }}
          >
            <div style={{ fontSize: '0.85rem', opacity: 0.7, color: 'rgba(255,255,255,0.7)' }}>
              Showing <strong>{pageStart + 1}</strong>–<strong>{Math.min(pageEnd, rows.length)}</strong> of <strong>{rows.length}</strong>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <label style={{ fontSize: '0.85rem', opacity: 0.7, color: 'rgba(255,255,255,0.7)' }}>Rows:</label>
              <select
                value={pageSize}
                onChange={(e) => setPageSize(Number(e.target.value))}
                style={{
                  background: 'rgba(255,255,255,0.05)',
                  color: '#fff',
                  border: '1px solid rgba(255,255,255,0.15)',
                  borderRadius: '6px',
                  padding: '4px 8px',
                  fontSize: '0.85rem',
                  outline: 'none',
                }}
              >
                {[10, 20, 50, 100].map((n) => (
                  <option key={n} value={n} style={{ background: '#153273' }}>{n}</option>
                ))}
              </select>
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setPage((p) => Math.max(1, p - 1))}
                disabled={page === 1}
                style={{ padding: '0.3rem 0.75rem' }}
              >
                Prev
              </button>
              {pageNumbers.map((n, idx) =>
                n === '...' ? (
                  <span key={`e-${idx}`} style={{ padding: '0 4px', opacity: 0.5, color: '#fff' }}>…</span>
                ) : (
                  <button
                    key={n}
                    type="button"
                    className={`btn btn-sm ${n === page ? 'btn-primary' : 'btn-secondary'}`}
                    onClick={() => setPage(n)}
                    style={{ padding: '0.3rem 0.7rem', minWidth: '34px' }}
                  >
                    {n}
                  </button>
                )
              )}
              <button
                type="button"
                className="btn btn-secondary btn-sm"
                onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                disabled={page === totalPages}
                style={{ padding: '0.3rem 0.75rem' }}
              >
                Next
              </button>
            </div>
          </div>
        </div>
      )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Pending Today view (Admin/HR only)
// ---------------------------------------------------------------------------
function PendingTodayView({
  onBanner,
}: {
  onBanner: (b: { kind: "ok" | "err"; text: string }) => void;
}) {
  const [loading, setLoading] = useState(true);
  const [todayIst, setTodayIst] = useState<string>("");
  const [totalActive, setTotalActive] = useState(0);
  const [submittedCount, setSubmittedCount] = useState(0);
  const [pending, setPending] = useState<DSRPendingEmployee[]>([]);
  const [selected, setSelected] = useState<Set<number>>(new Set());
  const [reminding, setReminding] = useState(false);
  const [designationFilter, setDesignationFilter] = useState<string>("");
  const [searchName, setSearchName] = useState<string>("");

  const load = useCallback(() => {
    setLoading(true);
    dsrApi
      .pendingToday()
      .then((r) => {
        setTodayIst(r.data.today_ist);
        setTotalActive(r.data.total_active_employees);
        setSubmittedCount(r.data.submitted);
        setPending(r.data.pending || []);
        setSelected(new Set());
      })
      .catch((err) => {
        onBanner({ kind: "err", text: formatApiError(err, "Could not load pending DSR list.") });
        setPending([]);
      })
      .finally(() => setLoading(false));
  }, [onBanner]);

  useEffect(() => {
    load();
  }, [load]);

  // Unique designations (using the effective value the backend returns, which
  // falls back to department when designation isn't set on the employee).
  const designationOptions = useMemo(() => {
    const set = new Set<string>();
    pending.forEach((p) => {
      const v = (p.designation || "").trim();
      if (v) set.add(v);
    });
    return Array.from(set).sort();
  }, [pending]);

  const filtered = useMemo(() => {
    let rows = pending;
    if (designationFilter) {
      rows = rows.filter((p) => (p.designation || "") === designationFilter);
    }
    const term = searchName.trim().toLowerCase();
    if (term) {
      rows = rows.filter((p) =>
        (p.employee_name || "").toLowerCase().includes(term) ||
        (p.employee_code || "").toLowerCase().includes(term)
      );
    }
    // Sort ascending by employee_code (numeric when possible, then alpha),
    // falling back to employee_name for rows without a code.
    const codeKey = (p: DSRPendingEmployee): number | string => {
      const raw = (p.employee_code || "").trim();
      if (!raw) return Number.POSITIVE_INFINITY;
      const n = Number(raw);
      return Number.isFinite(n) && /^\d+$/.test(raw) ? n : raw.toLowerCase();
    };
    return [...rows].sort((a, b) => {
      const ka = codeKey(a);
      const kb = codeKey(b);
      if (typeof ka === "number" && typeof kb === "number") return ka - kb;
      if (typeof ka === "number") return -1;
      if (typeof kb === "number") return 1;
      const cmp = String(ka).localeCompare(String(kb));
      if (cmp !== 0) return cmp;
      return (a.employee_name || "").localeCompare(b.employee_name || "");
    });
  }, [pending, designationFilter, searchName]);

  // When filters change, drop selections that fall out of view.
  useEffect(() => {
    setSelected((prev) => {
      const visible = new Set(filtered.map((p) => p.user_id));
      const next = new Set<number>();
      prev.forEach((id) => {
        if (visible.has(id)) next.add(id);
      });
      return next;
    });
  }, [filtered]);

  const toggleAll = () => {
    if (selected.size === filtered.length) {
      setSelected(new Set());
    } else {
      setSelected(new Set(filtered.map((p) => p.user_id)));
    }
  };

  const toggleOne = (user_id: number) => {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(user_id)) next.delete(user_id);
      else next.add(user_id);
      return next;
    });
  };

  const sendReminders = async (mode: "selected" | "all") => {
    const ids =
      mode === "selected"
        ? Array.from(selected)
        : filtered.map((p) => p.user_id);
    if (ids.length === 0) {
      onBanner({ kind: "err", text: "Select at least one employee to remind." });
      return;
    }
    setReminding(true);
    try {
      const res = await dsrApi.remindPendingToday(ids);
      onBanner({
        kind: "ok",
        text: `Reminder sent to ${res.data.notified} employee${res.data.notified === 1 ? "" : "s"}.`,
      });
      setSelected(new Set());
    } catch (err: unknown) {
      onBanner({ kind: "err", text: formatApiError(err, "Could not send reminders.") });
    } finally {
      setReminding(false);
    }
  };

  return (
    <div className="card" style={{ padding: 0, overflow: "hidden" }}>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: "0.75rem",
          padding: "1rem 1.25rem",
          borderBottom: "1px solid rgba(255,255,255,0.06)",
          flexWrap: "wrap",
        }}
      >
        <div>
          <h3 style={{ margin: 0, fontSize: "0.95rem", fontWeight: 800, color: "#fff" }}>
            Pending DSR — {todayIst || "today"}
          </h3>
          <div style={{ marginTop: 4, fontSize: "0.75rem", color: "rgba(255,255,255,0.55)" }}>
            {submittedCount} of {totalActive} active employees submitted ·{" "}
            <span style={{ color: "#fcd34d", fontWeight: 700 }}>
              {pending.length} pending
            </span>
          </div>
        </div>
        <div style={{ display: "flex", gap: "0.5rem", alignItems: "center", flexWrap: "wrap" }}>
          <input
            type="text"
            value={searchName}
            onChange={(e) => setSearchName(e.target.value)}
            placeholder="Search name or code..."
            style={{
              width: 200,
              background: "rgba(255, 255, 255, 0.05)",
              border: "1px solid rgba(255, 255, 255, 0.1)",
              borderRadius: 8,
              padding: "0.45rem 0.75rem",
              color: "#fff",
              fontSize: "0.82rem",
              height: "38px",
              outline: "none",
            }}
          />
          <CustomSelect
            value={designationFilter}
            onChange={(v) => setDesignationFilter(v)}
            options={[
              { value: "", label: "All designations" },
              ...designationOptions.map((d) => ({ value: d, label: d })),
            ]}
            style={{ width: 200 }}
          />
          {(designationFilter || searchName) && (
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              style={{ height: 38, padding: "0 0.8rem" }}
              onClick={() => {
                setDesignationFilter("");
                setSearchName("");
              }}
            >
              Clear
            </button>
          )}
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={load}
            disabled={loading}
            title="Refresh"
          >
            {loading ? "Refreshing…" : "Refresh"}
          </button>
          <button
            type="button"
            className="btn btn-primary btn-sm"
            onClick={() => sendReminders("selected")}
            disabled={reminding || selected.size === 0}
            style={{ padding: "0.5rem 1rem" }}
          >
            {reminding ? "Sending…" : `Remind Selected (${selected.size})`}
          </button>
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={() => sendReminders("all")}
            disabled={reminding || filtered.length === 0}
          >
            Remind All
          </button>
        </div>
      </div>

      {loading ? (
        <div style={{ padding: "2.5rem 0" }}>
          <SectionLoader size="md" />
        </div>
      ) : filtered.length === 0 ? (
        <div
          style={{
            padding: "2.5rem 1.25rem",
            textAlign: "center",
            color: "rgba(255,255,255,0.7)",
          }}
        >
          Everyone has submitted today's DSR. Nothing pending.
        </div>
      ) : (
        <div className="table-wrap table-wrap--dark">
          <table className="table-modern table-modern--dark" style={{ width: "100%" }}>
            <thead>
              <tr>
                <th style={{ width: 36, textAlign: "center" }}>
                  <input
                    type="checkbox"
                    checked={selected.size > 0 && selected.size === filtered.length}
                    ref={(el) => {
                      if (el) el.indeterminate = selected.size > 0 && selected.size < filtered.length;
                    }}
                    onChange={toggleAll}
                    aria-label="Select all"
                  />
                </th>
                <th>Employee</th>
                <th>Code</th>
                <th>Designation</th>
                <th>Email</th>
                <th style={{ textAlign: "center" }}>Status</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((p) => (
                <tr key={p.user_id}>
                  <td style={{ textAlign: "center" }}>
                    <input
                      type="checkbox"
                      checked={selected.has(p.user_id)}
                      onChange={() => toggleOne(p.user_id)}
                      aria-label={`Select ${p.employee_name}`}
                    />
                  </td>
                  <td style={{ fontWeight: 700, color: "#fff" }}>{p.employee_name}</td>
                  <td>{p.employee_code || "—"}</td>
                  <td>{p.designation || "—"}</td>
                  <td
                    style={{
                      maxWidth: 220,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                    title={p.official_email || ""}
                  >
                    {p.official_email || "—"}
                  </td>
                  <td style={{ textAlign: "center" }}>
                    {p.has_draft ? (
                      <span
                        style={{
                          display: "inline-block",
                          padding: "2px 10px",
                          borderRadius: 999,
                          fontSize: "0.7rem",
                          fontWeight: 700,
                          background: "rgba(234, 179, 8, 0.18)",
                          color: "#fde68a",
                          border: "1px solid rgba(234, 179, 8, 0.4)",
                        }}
                      >
                        Draft only
                      </span>
                    ) : (
                      <span
                        style={{
                          display: "inline-block",
                          padding: "2px 10px",
                          borderRadius: 999,
                          fontSize: "0.7rem",
                          fontWeight: 700,
                          background: "rgba(239, 68, 68, 0.18)",
                          color: "#fca5a5",
                          border: "1px solid rgba(239, 68, 68, 0.4)",
                        }}
                      >
                        Not started
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

function DsrViewModal({ row, onClose }: { row: DSRRow; onClose: () => void }) {
  const reportDateObj = new Date(`${row.report_date}T00:00:00`);
  const weekday = isNaN(reportDateObj.getTime())
    ? ""
    : reportDateObj.toLocaleDateString(undefined, { weekday: "long" });
  const reportDateStr = `${formatDate(row.report_date)}${weekday ? `, ${weekday}` : ""}`;

  const fmtDateTime = (iso: string | null | undefined): string => {
    if (!iso) return "—";
    try {
      const d = new Date(iso);
      if (isNaN(d.getTime())) return "—";
      const t = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      return `${formatDate(iso)}, ${t}`;
    } catch {
      return "—";
    }
  };
  const submittedAt = fmtDateTime(row.submitted_at);
  const updatedAt = fmtDateTime(row.updated_at);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div
        className="modal dsr-view-modal"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: 920, width: "100%", padding: 0, maxHeight: "90vh", overflowY: "auto", overflowX: "hidden" }}
        role="dialog"
        aria-modal="true"
        aria-labelledby="dsr-view-title"
      >
        {/* ---- Header ---- */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.95rem",
            padding: "1.2rem 1.4rem",
            borderBottom: "1px solid rgba(255,255,255,0.06)",
          }}
        >
          <div
            style={{
              width: 48,
              height: 48,
              borderRadius: 12,
              background: "rgba(59,130,246,0.14)",
              color: "#60a5fa",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              border: "1px solid rgba(96,165,250,0.25)",
              flexShrink: 0,
            }}
          >
            <Icons.Document />
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <h3
              id="dsr-view-title"
              className="modal-title"
              style={{ fontSize: "1.18rem", margin: 0, lineHeight: 1.2 }}
            >
              DSR Details
            </h3>
            <div
              className="modal-subtitle"
              style={{ marginTop: 4, color: "rgba(255,255,255,0.55)", fontSize: "0.82rem" }}
            >
              Here's the detailed report for {row.employee_name ? row.employee_name + "'s" : "this"} daily work.
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="btn btn-secondary btn-icon btn-sm"
            title="Close"
            aria-label="Close"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {/* ---- Stat-pill row ---- */}
        <div style={{ padding: "1rem 1.4rem 0.4rem" }}>
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
              gap: "0.65rem",
              background: "rgba(255,255,255,0.025)",
              border: "1px solid rgba(255,255,255,0.06)",
              borderRadius: 14,
              padding: "0.85rem",
            }}
          >
            <StatChip
              icon={<EmployeeIcon />}
              tint="#a78bfa"
              tintBg="rgba(167,139,250,0.14)"
              label="Employee"
              value={row.employee_name || "—"}
            />
            <StatChip
              icon={<CalendarIcon />}
              tint="#f472b6"
              tintBg="rgba(244,114,182,0.14)"
              label="Report Date"
              value={reportDateStr}
            />
            <StatChip
              icon={<BriefcaseIcon />}
              tint="#fbbf24"
              tintBg="rgba(251,191,36,0.14)"
              label="Project / Work"
              value={row.project_work || "—"}
            />
            <StatChip
              icon={<PinIcon />}
              tint="#fb923c"
              tintBg="rgba(251,146,60,0.14)"
              label="Work Location"
              value={row.work_location || "—"}
            />
            <StatChip
              icon={<ClockIcon />}
              tint="#38bdf8"
              tintBg="rgba(56,189,248,0.14)"
              label="Submitted On"
              value={submittedAt}
            />
            <StatChip
              icon={null}
              tint="#34d399"
              tintBg="transparent"
              label="Status"
              value={<StatusPill status={row.status} />}
              valueIsNode
            />
          </div>
        </div>

        {/* ---- Body sections ---- */}
        <div style={{ padding: "1.1rem 1.4rem 0.4rem" }}>
          <SectionBlock
            number={1}
            tint="#60a5fa"
            tintBg="rgba(96,165,250,0.14)"
            icon={<ChecklistIcon />}
            title="What did you work on today?"
            body={row.work_done}
          />
          <SectionBlock
            number={2}
            tint="#34d399"
            tintBg="rgba(52,211,153,0.14)"
            icon={<CalendarIcon />}
            title="Plan for tomorrow"
            body={row.plan_for_tomorrow}
            isLast
          />
        </div>

        {/* ---- Footer ---- */}
        <div
          style={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: "1rem",
            padding: "0.95rem 1.4rem",
            borderTop: "1px solid rgba(255,255,255,0.06)",
            background: "rgba(255,255,255,0.015)",
          }}
        >
          <div
            style={{
              display: "inline-flex",
              alignItems: "center",
              gap: 6,
              color: "rgba(255,255,255,0.5)",
              fontSize: "0.78rem",
            }}
          >
            <Icons.Clock /> Last updated on {updatedAt}
          </div>
          <button type="button" className="btn btn-primary btn-sm" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

function StatChip({
  icon,
  tint,
  tintBg,
  label,
  value,
  valueIsNode,
}: {
  icon: React.ReactNode;
  tint: string;
  tintBg: string;
  label: string;
  value: React.ReactNode;
  valueIsNode?: boolean;
}) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: "0.6rem",
        padding: "0.45rem 0.55rem",
        minWidth: 0,
      }}
    >
      {icon ? (
        <div
          style={{
            width: 34,
            height: 34,
            borderRadius: 9,
            background: tintBg,
            color: tint,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            border: `1px solid ${tint}33`,
            flexShrink: 0,
          }}
        >
          {icon}
        </div>
      ) : null}
      <div style={{ minWidth: 0, flex: 1 }}>
        <div
          style={{
            fontSize: "0.66rem",
            color: "rgba(255,255,255,0.5)",
            fontWeight: 600,
            letterSpacing: "0.02em",
            marginBottom: 2,
            textTransform: "none",
          }}
        >
          {label}
        </div>
        {valueIsNode ? (
          <div>{value}</div>
        ) : (
          <div
            style={{
              color: "rgba(255,255,255,0.95)",
              fontWeight: 600,
              fontSize: "0.84rem",
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
            title={typeof value === "string" ? value : undefined}
          >
            {value}
          </div>
        )}
      </div>
    </div>
  );
}

function SectionBlock({
  number,
  tint,
  tintBg,
  icon,
  title,
  body,
  isLast,
}: {
  number: number;
  tint: string;
  tintBg: string;
  icon: React.ReactNode;
  title: string;
  body: string | null | undefined;
  isLast?: boolean;
}) {
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "44px 1fr",
        gap: "0.95rem",
        position: "relative",
        paddingBottom: isLast ? "0.6rem" : "1.1rem",
      }}
    >
      {/* Left rail: icon + vertical line */}
      <div style={{ position: "relative", display: "flex", justifyContent: "center" }}>
        <div
          style={{
            width: 40,
            height: 40,
            borderRadius: 11,
            background: tintBg,
            color: tint,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            border: `1px solid ${tint}33`,
            zIndex: 1,
          }}
        >
          {icon}
        </div>
        {!isLast && (
          <div
            aria-hidden
            style={{
              position: "absolute",
              top: 44,
              bottom: -6,
              width: 2,
              background: "rgba(255,255,255,0.07)",
              borderRadius: 2,
            }}
          />
        )}
      </div>

      {/* Right column: title + body card */}
      <div
        style={{
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.07)",
          borderRadius: 12,
          padding: "0.85rem 1rem 0.95rem",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "0.5rem",
            marginBottom: "0.55rem",
          }}
        >
          <div style={{ fontWeight: 700, color: "rgba(255,255,255,0.95)", fontSize: "0.95rem" }}>
            {number}. {title}
          </div>
        </div>
        <div
          style={{
            height: 1,
            background: "rgba(255,255,255,0.06)",
            marginBottom: "0.65rem",
          }}
        />
        <div
          style={{
            color: "rgba(255,255,255,0.82)",
            fontSize: "0.86rem",
            lineHeight: 1.55,
            whiteSpace: "pre-wrap",
            minHeight: 36,
          }}
        >
          {body && body.trim() ? body : "—"}
        </div>
      </div>
    </div>
  );
}

// --- Tiny icons used only by the DSR detail modal ---
function EmployeeIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
      <circle cx="12" cy="7" r="4" />
    </svg>
  );
}
function CalendarIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
      <line x1="16" y1="2" x2="16" y2="6" />
      <line x1="8" y1="2" x2="8" y2="6" />
      <line x1="3" y1="10" x2="21" y2="10" />
    </svg>
  );
}
function BriefcaseIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <rect x="2" y="7" width="20" height="14" rx="2" ry="2" />
      <path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16" />
    </svg>
  );
}
function PinIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
      <circle cx="12" cy="10" r="3" />
    </svg>
  );
}
function ClockIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
    </svg>
  );
}
function ChecklistIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 11l3 3L22 4" />
      <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Reminder settings modal — Admin/HR only
// ---------------------------------------------------------------------------

const WEEKDAY_OPTIONS: { value: string; label: string }[] = [
  { value: "mon", label: "Mon" },
  { value: "tue", label: "Tue" },
  { value: "wed", label: "Wed" },
  { value: "thu", label: "Thu" },
  { value: "fri", label: "Fri" },
  { value: "sat", label: "Sat" },
  { value: "sun", label: "Sun" },
];

function ReminderSettingsModal({
  onClose,
  onSaved,
}: {
  onClose: () => void;
  onSaved: (s: DSRReminderSettings) => void;
}) {
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [enabled, setEnabled] = useState(true);
  const [time, setTime] = useState("17:00");
  const [weekdays, setWeekdays] = useState<string[]>(["mon", "tue", "wed", "thu", "fri"]);
  const [currentIst, setCurrentIst] = useState("");

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    dsrApi
      .reminderSettings()
      .then((r) => {
        if (cancelled) return;
        const data = r.data;
        setEnabled(Boolean(data.enabled));
        setTime(typeof data.time === "string" && data.time ? data.time : "17:00");
        setWeekdays(
          Array.isArray(data.weekdays) && data.weekdays.length
            ? data.weekdays
            : ["mon", "tue", "wed", "thu", "fri"]
        );
        setCurrentIst(data.current_ist || "");
      })
      .catch((e) => {
        if (!cancelled) {
          setError(formatApiError(e, "Could not load reminder settings."));
        }
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const toggleDay = (d: string) => {
    setWeekdays((prev) => {
      if (prev.includes(d)) return prev.filter((x) => x !== d);
      // Re-sort by calendar order so display is stable.
      const next = [...prev, d];
      return WEEKDAY_OPTIONS.map((o) => o.value).filter((v) => next.includes(v));
    });
  };

  const onSave = async () => {
    if (!safeWeekdays.length) {
      setError("Pick at least one weekday.");
      return;
    }
    if (!/^\d{1,2}:\d{2}$/.test(time)) {
      setError("Time must be HH:MM, e.g. 17:00");
      return;
    }
    setSaving(true);
    setError(null);
    try {
      const r = await dsrApi.updateReminderSettings({
        enabled,
        time,
        weekdays: safeWeekdays,
      });
      onSaved(r.data);
      onClose();
    } catch (e: unknown) {
      setError(formatApiError(e, "Could not save settings."));
    } finally {
      setSaving(false);
    }
  };

  const safeWeekdays = Array.isArray(weekdays) ? weekdays : [];

  const modal = (
    <div className="modal-backdrop" onClick={onClose} role="presentation">
      <div
        className="modal"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: 560, width: "100%" }}
        role="dialog"
        aria-labelledby="dsr-reminder-title"
        aria-modal="true"
      >
        <div className="modal-titlebar">
          <div>
            <h3 id="dsr-reminder-title" className="modal-title" style={{ fontSize: "1.1rem" }}>
              DSR reminder schedule
            </h3>
            <div className="modal-subtitle">
              All times are in <strong>IST</strong>
              {currentIst ? ` · server ${currentIst}` : ""}
            </div>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="btn btn-secondary btn-icon btn-sm"
            title="Close"
            aria-label="Close"
          >
            <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>

        {loading ? (
          <div style={{ padding: "2rem 0", textAlign: "center", color: "rgba(255,255,255,0.7)" }}>
            Loading settings…
          </div>
        ) : (
          <div className="modal-stack">
            <label className="modal-checkbox-group" style={{ alignItems: "flex-start" }}>
              <input
                type="checkbox"
                checked={enabled}
                onChange={(e) => setEnabled(e.target.checked)}
              />
              <div>
                <div style={{ fontWeight: 600, color: "#fff" }}>Send daily DSR reminder</div>
                <div style={{ fontSize: "0.78rem", color: "rgba(255,255,255,0.55)", marginTop: 2 }}>
                  When off, no in-app / email / web push reminder is sent.
                </div>
              </div>
            </label>

            <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
              <label style={{ fontWeight: 600, minWidth: 120, color: "#fff" }}>Reminder time</label>
              <input
                type="time"
                className="form-control"
                value={time}
                disabled={!enabled}
                onChange={(e) => setTime(e.target.value)}
                style={{ width: 140, maxWidth: "100%" }}
              />

            </div>

            <div>
              <div style={{ fontWeight: 600, marginBottom: 8, color: "#fff" }}>Active weekdays</div>
              <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
                {WEEKDAY_OPTIONS.map((opt) => {
                  const active = safeWeekdays.includes(opt.value);
                  return (
                    <button
                      key={opt.value}
                      type="button"
                      disabled={!enabled}
                      onClick={() => toggleDay(opt.value)}
                      style={{
                        padding: "0.45rem 0.85rem",
                        borderRadius: 999,
                        border: active
                          ? "1px solid rgba(99,102,241,0.7)"
                          : "1px solid rgba(255,255,255,0.14)",
                        background: active
                          ? "rgba(99,102,241,0.25)"
                          : "rgba(255,255,255,0.04)",
                        color: active ? "#fff" : "rgba(255,255,255,0.78)",
                        fontWeight: 600,
                        fontSize: "0.82rem",
                        cursor: enabled ? "pointer" : "not-allowed",
                        opacity: enabled ? 1 : 0.5,
                      }}
                    >
                      {opt.label}
                    </button>
                  );
                })}
              </div>
              <div
                style={{
                  fontSize: "0.78rem",
                  color: "rgba(255,255,255,0.55)",
                  marginTop: 6,
                }}
              >
                The reminder only fires on the selected days.
              </div>
            </div>

            {error && (
              <div
                style={{
                  background: "rgba(239,68,68,0.12)",
                  color: "#fecaca",
                  border: "1px solid rgba(239,68,68,0.35)",
                  borderRadius: 8,
                  padding: "0.55rem 0.8rem",
                  fontSize: "0.85rem",
                }}
              >
                {error}
              </div>
            )}
          </div>
        )}

        <div className="modal-actions">

          <button
            type="button"
            className="btn btn-primary btn-sm"
            onClick={onSave}
            disabled={saving || loading}
            style={{ padding: "0.5rem 1.1rem" }}
          >
            {saving ? "Saving…" : "Save"}
          </button>
          <button
            type="button"
            className="btn btn-cancel-alt btn-sm"
            onClick={onClose}
            disabled={saving}
          >
            Cancel
          </button>
        </div>
      </div>
    </div>
  );

  return createPortal(modal, document.body);
}



