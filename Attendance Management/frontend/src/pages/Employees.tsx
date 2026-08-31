import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { employees as api, users as usersApi } from "../api/client";
import ConfirmModal from "../components/ConfirmModal";
import { SectionLoader } from "../components/LoadingState";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import CustomSelect from "../components/CustomSelect";
import { useTableControls } from "../components/dataTable";
import type { SortState, SortDirection } from "../components/dataTable";

interface Emp {
  id: number;
  employee_code: string;
  staff_type?: string;
  first_name: string;
  last_name: string;
  official_email: string;
  personal_email?: string;
  phone?: string;
  date_of_joining: string;
  designation_id?: number;
  department_id?: number;
  employment_type: string;
  reporting_manager_id?: number;
  employment_status: string;
  date_of_birth?: string;
  date_of_marriage?: string | null;
  marital_status?: string | null;
  date_of_leaving?: string | null;
  expected_working_hours: number;
}

const EMPLOYMENT_TYPES = ["Full-time", "Intern", "Contract"];
const EMPLOYMENT_STATUSES = ["Active", "Resigned", "Terminated"];
const MARITAL_STATUSES = ["Single", "Married", "Divorced", "Widowed"];

const STAFF_TYPE_SUGGESTIONS = ["Employee", "Housekeeping", "Security", "Driver", "Contractor", "Other"];

const emptyForm = (): Record<string, string | number | undefined> => ({
  employee_code: "",
  staff_type: "Employee",
  first_name: "",
  last_name: "",
  official_email: "",
  personal_email: "",
  phone: "",
  date_of_joining: "",
  designation_id: "",
  department_id: "",
  employment_type: "Full-time",
  employment_status: "Active",
  date_of_birth: "",
  date_of_marriage: "",
  marital_status: "",
  date_of_leaving: "",
  expected_working_hours: 9.0,
  login_username: "",
  login_password: "",
});

const maxDobDate = (() => {
  const d = new Date();
  d.setFullYear(d.getFullYear() - 18);
  return d.toISOString().split("T")[0];
})();

// Premium SVG Icons for Actions
/* 24-box strokes, round caps, currentColor. Size comes from the control that
   holds them (.eds-iconbtn 15px, .eds-search 15px, .eds-sort 9px). */
const Icons = {
  View: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <path d="M1.5 12S5 5.5 12 5.5 22.5 12 22.5 12 19 18.5 12 18.5 1.5 12 1.5 12z"></path>
      <circle cx="12" cy="12" r="3"></circle>
    </svg>
  ),
  Delete: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6"></polyline>
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"></path>
      <path d="M10 11v6M14 11v6"></path>
    </svg>
  ),
  Search: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="11" cy="11" r="7.5"></circle>
      <line x1="21" y1="21" x2="16.7" y2="16.7"></line>
    </svg>
  ),
  Person: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <path d="M19 21v-1.5A4.5 4.5 0 0 0 14.5 15h-5A4.5 4.5 0 0 0 5 19.5V21"></path>
      <circle cx="12" cy="8" r="4"></circle>
    </svg>
  ),
  Sort: ({ direction }: { direction: SortDirection | null }) => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round">
      {direction !== "asc" && <polyline points="7 15 12 20 17 15"></polyline>}
      {direction !== "desc" && <polyline points="7 9 12 4 17 9"></polyline>}
    </svg>
  ),
};

/** Column header. Sorting itself stays in useTableControls; this only renders
 *  the design's affordance. */
function SortTh({
  label,
  columnKey,
  sort,
  onToggle,
  notSortable,
  className,
}: {
  label: string;
  columnKey: string;
  sort: SortState;
  onToggle: (key: string) => void;
  notSortable?: boolean;
  className?: string;
}) {
  if (notSortable) return <th className={`is-actions ${className || ""}`}>{label}</th>;
  const active = sort.key === columnKey;
  return (
    <th className={className}>
      <button
        type="button"
        className={`eds-sort${active ? " is-active" : ""}`}
        onClick={() => onToggle(columnKey)}
        title={`Sort by ${label}`}
      >
        {label}
        <Icons.Sort direction={active ? sort.direction : null} />
      </button>
    </th>
  );
}

/** Identity tint for a member avatar, stable across sorts and searches. */
const AVATAR_TINTS = ["eds-avatar--blue", "eds-avatar--green", "eds-avatar--purple", "eds-avatar--rose", ""];

function initialsOf(first: string, last: string): string {
  const a = (first || "").trim()[0] || "";
  const b = (last || "").trim()[0] || "";
  return (a + b).toUpperCase() || "?";
}

export default function Employees() {
  const { hasRole } = useAuth();
  const navigate = useNavigate();
  const canEdit = hasRole("Admin") || hasRole("HR");
  const [list, setList] = useState<Emp[]>([]);
  const [departments, setDepartments] = useState<Array<{ id: number; name: string }>>([]);
  const [designations, setDesignations] = useState<Array<{ id: number; title: string }>>([]);
  const [loading, setLoading] = useState(true);
  const [modal, setModal] = useState<"add" | "staff" | null>(null);
  const [form, setForm] = useState(emptyForm());
  const [filterDept, setFilterDept] = useState<string>("");
  const [filterStatus, setFilterStatus] = useState<string>("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [createdLogin, setCreatedLogin] = useState<{ username: string; password: string } | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<Emp | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const sortEmployees = (rows: Emp[]) => {
    const toKey = (code: string) => {
      const n = Number(code);
      // numeric codes first (1,2,3...), otherwise fallback to string
      if (Number.isFinite(n) && String(code).trim() !== "") return { kind: 0 as const, n, s: "" };
      return { kind: 1 as const, n: 0, s: String(code || "").toLowerCase() };
    };
    return [...rows].sort((a, b) => {
      const ka = toKey(a.employee_code);
      const kb = toKey(b.employee_code);
      if (ka.kind !== kb.kind) return ka.kind - kb.kind;
      if (ka.kind === 0) return ka.n - kb.n;
      if (ka.s < kb.s) return -1;
      if (ka.s > kb.s) return 1;
      return a.id - b.id;
    });
  };

  const load = async () => {
    const params: { department_id?: number; status?: string } = {};
    if (filterDept) params.department_id = Number(filterDept);
    if (filterStatus) params.status = filterStatus;
    setLoading(true);
    const [employeesRes, departmentsRes, designationsRes] = await Promise.allSettled([
      api.list(params),
      api.departments(),
      api.designations(),
    ]);
    if (employeesRes.status === "fulfilled") setList(sortEmployees(employeesRes.value.data));
    else setList([]);
    if (departmentsRes.status === "fulfilled") setDepartments(departmentsRes.value.data);
    else setDepartments([]);
    if (designationsRes.status === "fulfilled") setDesignations(designationsRes.value.data);
    else setDesignations([]);
    setLoading(false);
  };

  useEffect(() => {
    void load();
  }, [filterDept, filterStatus]);

  const deptName = (id?: number) => departments.find((d) => d.id === id)?.name || "";

  const {
    displayed: displayedList,
    search,
    setSearch,
    sort,
    toggleSort,
    clearAll,
    hasActiveControls,
  } = useTableControls<Emp>({
    rows: list,
    columns: {
      employee_code: (e) => e.employee_code,
      name: (e) => `${e.first_name} ${e.last_name}`,
      staff_type: (e) => e.staff_type || "",
      official_email: (e) => e.official_email,
      department: (e) => deptName(e.department_id),
      date_of_joining: (e) => e.date_of_joining,
      date_of_leaving: (e) => e.date_of_leaving || "",
      employment_status: (e) => e.employment_status,
    },
    searchableText: (e) =>
      `${e.employee_code} ${e.first_name} ${e.last_name} ${e.staff_type || ""} ${e.official_email} ${deptName(e.department_id)} ${e.employment_status}`,
  });

  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(10);
  const totalPages = Math.max(1, Math.ceil(displayedList.length / pageSize));
  // Reset to page 1 when filters / search / sort / page size change
  useEffect(() => {
    setPage(1);
  }, [search, filterDept, filterStatus, sort.key, sort.direction, pageSize]);
  // Clamp page if the data shrinks (e.g. after delete)
  useEffect(() => {
    if (page > totalPages) setPage(totalPages);
  }, [page, totalPages]);
  const pageStart = (page - 1) * pageSize;
  const pageEnd = pageStart + pageSize;
  const pagedList = displayedList.slice(pageStart, pageEnd);

  const pageNumbers = (() => {
    // Render up to 7 buttons with smart ellipses around current page.
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

  const openAdd = async () => {
    setError("");
    setSuccess("");
    setModal("add");
    let nextCode = "";
    try {
      const r = await api.list();
      const allEmps = r.data as Emp[];
      let maxNum = 0;
      allEmps.forEach((e) => {
        const str = String(e.employee_code).replace(/\D/g, "");
        if (str) {
          const num = parseInt(str, 10);
          if (!isNaN(num) && num > maxNum) maxNum = num;
        }
      });
      nextCode = String(maxNum + 1);
    } catch {
      // ignore, leave blank
    }
    setForm({ ...emptyForm(), employee_code: nextCode });
  };


  const setField = (key: string, value: string | number) => {
    setForm((f) => ({ ...f, [key]: value }));
  };

  // ── Non-employee staff (housekeeping/security/etc.) — minimal form ──────
  const [staffForm, setStaffForm] = useState({ first_name: "", last_name: "", staff_type: "Housekeeping", phone: "" });
  const setStaffField = (key: string, value: string) => setStaffForm((f) => ({ ...f, [key]: value }));

  const openStaff = () => {
    setError("");
    setSuccess("");
    setStaffForm({ first_name: "", last_name: "", staff_type: "Housekeeping", phone: "" });
    setModal("staff");
  };

  const handleSubmitStaff = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!staffForm.first_name.trim() || !staffForm.staff_type.trim()) {
      setError("Name and staff type are required.");
      return;
    }
    setSubmitting(true);
    setError("");
    try {
      await api.createStaff({
        first_name: staffForm.first_name.trim(),
        last_name: staffForm.last_name.trim(),
        staff_type: staffForm.staff_type.trim(),
        phone: staffForm.phone.trim() || undefined,
      });
      setModal(null);
      setSuccess("Staff member added. Open their profile to register a face.");
      await load();
    } catch (err) {
      const e = err as { response?: { data?: { detail?: string } } };
      setError(e.response?.data?.detail || "Failed to add staff member.");
    } finally {
      setSubmitting(false);
    }
  };

  const handleSubmitAdd = (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    const payload: Record<string, unknown> = {
      employee_code: form.employee_code,
      staff_type: form.staff_type || "Employee",
      first_name: form.first_name,
      last_name: form.last_name,
      official_email: form.official_email,
      personal_email: form.personal_email || null,
      phone: form.phone || null,
      date_of_joining: form.date_of_joining,
      designation_id: form.designation_id ? Number(form.designation_id) : null,
      department_id: form.department_id ? Number(form.department_id) : null,
      employment_type: form.employment_type,
      employment_status: form.employment_status,
      date_of_birth: form.date_of_birth || null,
      date_of_marriage: form.date_of_marriage || null,
      marital_status: form.marital_status || null,
      date_of_leaving: form.date_of_leaving || null,
      expected_working_hours: Number(form.expected_working_hours) || 9.0,
    };

    // Validation: If date_of_leaving is set, status must be Resigned or Terminated
    if (payload.date_of_leaving) {
      if (payload.employment_status === "Active") {
        setError("Please select either Resigned or Terminated when adding a date of leaving.");
        return;
      }
    }
    setSubmitting(true);
    api
      .create(payload)
      .then(async (res) => {
        const emp = res.data as Emp;
        if (form.login_username && form.login_password) {
          try {
            await usersApi.create({
              username: String(form.login_username),
              password: String(form.login_password),
              official_email: String(form.official_email),
              employee_id: emp.id,
              role_names: ["Employee"],
            });
            setCreatedLogin({
              username: String(form.login_username),
              password: String(form.login_password),
            });
          } catch (e) {
            console.error("Failed to create login", e);
          }
        } else {
          setCreatedLogin(null);
        }
        setSuccess("Employee created.");
        setList((prev) => sortEmployees([...prev, emp]));
        setModal(null);
      })
      .catch((err) => setError(err.response?.data?.detail || "Failed"))
      .finally(() => setSubmitting(false));
  };

  const confirmActualDelete = () => {
    if (!confirmDelete) return;
    setSubmitting(true);
    api
      .delete(confirmDelete.id)
      .then(() => {
        setSuccess("Employee deleted.");
        setList((prev) => prev.filter((e) => e.id !== confirmDelete.id));
        setConfirmDelete(null);
      })
      .catch((err) => setError(err.response?.data?.detail || "Failed"))
      .finally(() => setSubmitting(false));
  };


  // Footer aggregate over the rows actually on this page.
  const pageActive = pagedList.filter((e) => e.employment_status === "Active").length;
  const pageLeft = pagedList.length - pageActive;

  return (
    <div className="eds">
      <header className="eds-topbar">
        <div>
          <h1 className="eds-title">Employees</h1>
          <p className="eds-subtitle">View and manage employee records</p>
        </div>
        <GlobalHeaderControls />
      </header>
      <div className="eds-page">
      {createdLogin && (
        <div className="eds-card">
          <div className="eds-card-body">
            <p className="eds-note"><b>Login created for employee.</b> Share these credentials securely with the employee:</p>
            <p className="eds-note">
              Username: <b>{createdLogin.username}</b> &nbsp;|&nbsp;
              Password: <b>{createdLogin.password}</b>
            </p>
          </div>
        </div>
      )}
      {success && <div className="alert alert-success">{success}</div>}
      {error && <div className="alert alert-error">{error}</div>}
      <section className="eds-card">
        <div className="eds-toolbar">
          <CustomSelect
            className="eds-cselect"
            value={filterDept}
            onChange={setFilterDept}
            placeholder="All Departments"
            options={[
              { value: "", label: "All Departments" },
              ...departments.map((d) => ({ value: String(d.id), label: d.name }))
            ]}
          />
          <CustomSelect
            className="eds-cselect eds-cselect--status"
            value={filterStatus}
            onChange={setFilterStatus}
            placeholder="All Status"
            options={[
              { value: "", label: "All Status" },
              ...EMPLOYMENT_STATUSES.map((s) => ({ value: s, label: s }))
            ]}
          />
          <label className="eds-search eds-search--wide">
            <Icons.Search />
            <input
              type="search"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search employees (name, code, email, department)"
            />
          </label>
          <span className="eds-showing">
            <b>{displayedList.length}</b> of <b>{list.length}</b>
          </span>
          {(hasActiveControls || !!filterDept || !!filterStatus) && (
            <button
              type="button"
              className="eds-action"
              onClick={() => { clearAll(); setFilterDept(""); setFilterStatus(""); }}
              title="Clear search, sort and column filters"
            >
              Clear filters
            </button>
          )}
          {canEdit && (
            <div style={{ marginLeft: "auto", display: "flex", alignItems: "center", gap: 9, flexShrink: 0 }}>
              <button type="button" className="eds-action" onClick={openStaff} title="Add non-employee staff (housekeeping, security, etc.)">
                + Add Staff
              </button>
              <button type="button" className="eds-action eds-action--info" onClick={openAdd} title="Add New Employee">
                Add Employee
              </button>
            </div>
          )}
        </div>
        {loading ? (
          <div style={{ padding: "3rem 0" }}><SectionLoader size="md" /></div>
        ) : list.length === 0 ? (
          <div className="eds-well">
            <div className="eds-empty--dashed">
              <span className="eds-chip"><Icons.Person /></span>
              <span className="eds-empty-title">No employee records found.</span>
            </div>
          </div>
        ) : (
          <>
            <div className="eds-table-wrap">
              <table className="eds-table eds-table--employees">
                <colgroup>
                  <col style={{ width: '4.7%' }} />
                  <col style={{ width: '14%' }} />
                  <col style={{ width: '9.3%' }} />
                  <col style={{ width: '19.6%' }} />
                  <col style={{ width: '14%' }} />
                  <col style={{ width: '10.3%' }} />
                  <col style={{ width: '10.3%' }} />
                  <col style={{ width: '9.3%' }} />
                  {canEdit && <col style={{ width: '8.4%' }} />}
                </colgroup>
                <thead>
                  <tr>
                    <SortTh className="hide-md" label="ID" columnKey="employee_code" sort={sort} onToggle={toggleSort} />
                    <SortTh label="Name" columnKey="name" sort={sort} onToggle={toggleSort} />
                    <SortTh label="Staff type" columnKey="staff_type" sort={sort} onToggle={toggleSort} />
                    <SortTh label="Email" columnKey="official_email" sort={sort} onToggle={toggleSort} />
                    <SortTh className="hide-sm" label="Department" columnKey="department" sort={sort} onToggle={toggleSort} />
                    <SortTh className="hide-md" label="DOJ" columnKey="date_of_joining" sort={sort} onToggle={toggleSort} />
                    <SortTh className="hide-md" label="DOL" columnKey="date_of_leaving" sort={sort} onToggle={toggleSort} />
                    <SortTh className="hide-sm" label="Status" columnKey="employment_status" sort={sort} onToggle={toggleSort} />
                    {canEdit && (
                      <SortTh label="Actions" columnKey="__actions" sort={sort} onToggle={toggleSort} notSortable />
                    )}
                  </tr>
                </thead>
                <tbody>
                  {displayedList.length === 0 ? (
                    <tr>
                      <td colSpan={canEdit ? 9 : 8} className="eds-table-empty">
                        No employees match your search / filters.
                      </td>
                    </tr>
                  ) : null}
                  {pagedList.map((e) => {
                    const hasLeft = !!e.date_of_leaving;
                    const dolText = e.date_of_leaving
                      ? new Date(e.date_of_leaving).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })
                      : '-';
                    return (
                    <tr key={e.id} title={hasLeft ? `Employee left on ${dolText}` : undefined}>
                      <td className="hide-md eds-cell-dim">{e.employee_code}</td>
                      <td>
                        <div className="eds-member">
                          <span className={`eds-avatar eds-avatar--md ${AVATAR_TINTS[e.id % AVATAR_TINTS.length]}`}>
                            {initialsOf(e.first_name, e.last_name)}
                          </span>
                          <span className="eds-member-name">{e.first_name} {e.last_name}</span>
                        </div>
                      </td>
                      <td className="eds-cell-dim">{e.staff_type || "Employee"}</td>
                      <td className="eds-cell-dim eds-cell-clip">{e.official_email}</td>
                      <td className="hide-sm eds-cell-strong eds-cell-clip">{departments.find((d) => d.id === e.department_id)?.name || "-"}</td>
                      <td className="hide-md eds-cell-dim">{e.date_of_joining ? new Date(e.date_of_joining).toLocaleDateString("en-GB", { day: 'numeric', month: 'short', year: 'numeric' }) : "-"}</td>
                      <td className={`hide-md ${hasLeft ? "eds-cell-strong" : "eds-cell-dim"}`} style={hasLeft ? { fontWeight: 600 } : undefined}>
                        {hasLeft ? dolText : <span className="eds-dash">–</span>}
                      </td>
                      <td className="hide-sm">
                        <span className={`eds-status${e.employment_status === "Active" ? " eds-status--present" : " eds-status--absent"}`}>
                          <i></i>
                          {e.employment_status}
                        </span>
                      </td>
                      {canEdit && (
                        <td className="is-actions">
                          <div className="eds-rowactions">
                            <button
                              type="button"
                              className="eds-iconbtn eds-iconbtn--view"
                              onClick={() => navigate(`/employees/${e.id}`)}
                              title="View Employee Profile"
                            >
                              <Icons.View />
                            </button>
                            <button
                              type="button"
                              className="eds-iconbtn eds-iconbtn--del"
                              onClick={() => setConfirmDelete(e)}
                              title="Delete Employee Permanently"
                            >
                              <Icons.Delete />
                            </button>
                          </div>
                        </td>
                      )}
                    </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
            {displayedList.length > 0 && (
              <div className="eds-card-foot eds-att-foot">
                <span>
                  <b className="is-emerald">{pageActive}</b> active · <b className="is-text">{pageLeft}</b> not active on this page
                </span>
                <span className="eds-pager">
                  <span className="eds-showing" style={{ margin: 0 }}>
                    Showing <b>{pageStart + 1}</b>–<b>{Math.min(pageEnd, displayedList.length)}</b> of <b>{displayedList.length}</b>
                  </span>
                  <label className="eds-pager-rows">
                    Rows
                    <select value={pageSize} onChange={(e) => setPageSize(Number(e.target.value))}>
                      {[10, 25, 50, 100].map((n) => (
                        <option key={n} value={n}>{n}</option>
                      ))}
                    </select>
                  </label>
                  <button
                    type="button"
                    className="eds-action"
                    onClick={() => setPage((p) => Math.max(1, p - 1))}
                    disabled={page === 1}
                  >
                    Prev
                  </button>
                  {pageNumbers.map((n, idx) =>
                    n === '...' ? (
                      <span key={`e-${idx}`} className="eds-dash">…</span>
                    ) : (
                      <button
                        key={n}
                        type="button"
                        className={`eds-action${n === page ? " eds-action--info" : ""}`}
                        onClick={() => setPage(n)}
                      >
                        {n}
                      </button>
                    )
                  )}
                  <button
                    type="button"
                    className="eds-action"
                    onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
                    disabled={page === totalPages}
                  >
                    Next
                  </button>
                </span>
              </div>
            )}
          </>
        )}
      </section>
      </div>

      {modal === "staff" && (
        <div className="modal-backdrop" onClick={() => setModal(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 480 }}>
            <h3 style={{ marginTop: 0 }}>Add Staff</h3>
            <p style={{ marginTop: 0, fontSize: "0.85rem", opacity: 0.7 }}>
              For non-employee staff (housekeeping, security, driver…). They are recognised on
              camera but not marked for attendance. Register their face from the profile after saving.
            </p>
            <form onSubmit={handleSubmitStaff}>
              <div style={{ display: "grid", gap: "0.9rem" }}>
                <div className="form-group">
                  <label>Staff Type *</label>
                  <input
                    list="staff-type-options"
                    value={staffForm.staff_type}
                    onChange={(e) => setStaffField("staff_type", e.target.value)}
                    required
                    placeholder="e.g. Housekeeping, Security, Driver"
                  />
                  <datalist id="staff-type-options">
                    {STAFF_TYPE_SUGGESTIONS.filter((s) => s !== "Employee").map((s) => (
                      <option key={s} value={s} />
                    ))}
                  </datalist>
                </div>
                <div className="form-group">
                  <label>First Name *</label>
                  <input value={staffForm.first_name} onChange={(e) => setStaffField("first_name", e.target.value)} required placeholder="First Name" />
                </div>
                <div className="form-group">
                  <label>Last Name</label>
                  <input value={staffForm.last_name} onChange={(e) => setStaffField("last_name", e.target.value)} placeholder="Last Name (optional)" />
                </div>
                <div className="form-group">
                  <label>Phone</label>
                  <input value={staffForm.phone} onChange={(e) => setStaffField("phone", e.target.value)} placeholder="Phone (optional)" />
                </div>
              </div>
              {error ? <div style={{ color: "#f87171", marginTop: "0.75rem", fontSize: "0.85rem" }}>{error}</div> : null}
              <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.6rem", marginTop: "1.25rem" }}>
                <button type="button" className="btn btn-cancel-alt" onClick={() => setModal(null)} disabled={submitting}>Cancel</button>
                <button type="submit" className="btn btn-primary" disabled={submitting}>{submitting ? "Saving…" : "Save Staff"}</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {modal === "add" && (
        <div className="modal-backdrop" onClick={() => setModal(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 900 }}>
            <h3 style={{ marginTop: 0 }}>Add Employee</h3>
            <form onSubmit={handleSubmitAdd}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "1rem" }}>

                <div className="form-group">
                  <label>Staff Type *</label>
                  <input
                    list="staff-type-options"
                    value={form.staff_type ?? "Employee"}
                    onChange={(e) => setField("staff_type", e.target.value)}
                    required
                    placeholder="e.g. Employee, Housekeeping, Security"
                  />
                  <datalist id="staff-type-options">
                    {STAFF_TYPE_SUGGESTIONS.map((s) => (
                      <option key={s} value={s} />
                    ))}
                  </datalist>
                </div>
                <div className="form-group">
                  <label>First Name *</label>
                  <input value={form.first_name} onChange={(e) => setField("first_name", e.target.value)} required placeholder="First Name" />
                </div>
                <div className="form-group">
                  <label>Last Name *</label>
                  <input value={form.last_name} onChange={(e) => setField("last_name", e.target.value)} required placeholder="Last Name" />
                </div>
                <div className="form-group">
                  <label>Official Email *</label>
                  <input type="email" value={form.official_email} onChange={(e) => setField("official_email", e.target.value)} required placeholder="Official Email" />
                </div>
                <div className="form-group">
                  <label>Personal Email *</label>
                  <input type="email" value={form.personal_email} onChange={(e) => setField("personal_email", e.target.value)} required placeholder="Personal Email" />
                </div>
                <div className="form-group">
                  <label>Phone *</label>
                  <input value={form.phone} onChange={(e) => setField("phone", e.target.value)} required placeholder="Phone Number" />
                </div>
                <div className="form-group">
                  <label>Date of Joining *</label>
                  <input type="date" value={form.date_of_joining} onChange={(e) => setField("date_of_joining", e.target.value)} required />
                </div>
                <div className="form-group">
                  <label>Department *</label>
                  <CustomSelect
                    value={String(form.department_id || "")}
                    onChange={(val) => setField("department_id", val)}
                    options={[
                      { value: "", label: "Select Department" },
                      ...departments.map((d) => ({ value: String(d.id), label: d.name }))
                    ]}
                  />
                </div>
                <div className="form-group">
                  <label>Designation *</label>
                  <CustomSelect
                    value={String(form.designation_id || "")}
                    onChange={(val) => setField("designation_id", val)}
                    options={[
                      { value: "", label: "Select Designation" },
                      ...designations.map((d) => ({ value: String(d.id), label: d.title }))
                    ]}
                  />
                </div>
                <div className="form-group">
                  <label>Employment Type *</label>
                  <CustomSelect
                    value={String(form.employment_type || "")}
                    onChange={(val) => setField("employment_type", val)}
                    options={EMPLOYMENT_TYPES.map((t) => ({ value: t, label: t }))}
                  />
                </div>
                <div className="form-group">
                  <label>Date of Birth *</label>
                  <input type="date" value={form.date_of_birth} max={maxDobDate} onChange={(e) => setField("date_of_birth", e.target.value)} required />
                </div>
                <div className="form-group">
                  <label>Marital Status</label>
                  <CustomSelect
                    value={String(form.marital_status || "")}
                    onChange={(val) => {
                      setField("marital_status", val);
                      if (val !== "Married") {
                        setField("date_of_marriage", "");
                      }
                    }}
                    placeholder="Select Marital Status"
                    options={[
                      { value: "", label: "Select Marital Status" },
                      ...MARITAL_STATUSES.map((s) => ({ value: s, label: s })),
                    ]}
                  />
                </div>
                <div className="form-group">
                  <label>Date of Marriage</label>
                  <input
                    type="date"
                    value={form.date_of_marriage}
                    onChange={(e) => setField("date_of_marriage", e.target.value)}
                    disabled={form.marital_status !== "Married"}
                    title={form.marital_status !== "Married" ? "Available only when Marital Status is Married" : undefined}
                  />
                </div>
                <div className="form-group">
                  <label>Date of leaving</label>
                  <input
                    type="date"
                    value={form.date_of_leaving}
                    onChange={(e) => {
                      const val = e.target.value;
                      setField("date_of_leaving", val);
                      if (val) {
                        if (String(form.employment_status || "") === "Active") {
                          setField("employment_status", "Resigned");
                        }
                      } else {
                        setField("employment_status", "Active");
                      }
                    }}
                  />
                </div>
                <div className="form-group">
                  <label>{form.date_of_leaving ? "Reason for leaving *" : "Employment Status *"}</label>
                  <CustomSelect
                    value={String(form.employment_status || "")}
                    onChange={(val) => setField("employment_status", val)}
                    options={(form.date_of_leaving
                      ? ["Resigned", "Terminated"]
                      : EMPLOYMENT_STATUSES
                    ).map((s) => ({ value: s, label: s }))}
                  />
                </div>
                <div className="form-group">
                  <label>Expected Working Hours *</label>
                  <input type="number" step="0.5" value={form.expected_working_hours} onChange={(e) => setField("expected_working_hours", e.target.value)} required placeholder="e.g. 9.0" />
                </div>
              </div>
              <div className="card" style={{ marginTop: "1rem", background: "rgba(255, 255, 255, 0.06)", border: "1px solid rgba(255, 255, 255, 0.12)" }}>
                <h4 style={{ marginTop: 0 }}>Login credentials (optional)</h4>
                <p className="text-muted">If you fill these, a login account will be created for the employee.</p>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "1rem" }}> <div className="form-group"> <label>Username</label>
                  <input
                    value={String(form.login_username || "")}
                    onChange={(e) => setField("login_username", e.target.value)}
                    placeholder="Username"
                  />
                </div>
                  <div className="form-group">
                    <label>Password</label>
                    <input
                      type="password"
                      value={String(form.login_password || "")}
                      onChange={(e) => setField("login_password", e.target.value)}
                      placeholder="Password"
                    />
                  </div>
                </div>
              </div>
              <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.75rem", marginTop: "2rem" }}>
                <button type="submit" className="btn btn-primary" title="Create and Save New Employee Record" disabled={submitting}>
                  {submitting ? "Creating..." : "Create"}
                </button>
                <button type="button" className="btn btn-cancel-alt" onClick={() => setModal(null)} title="Cancel and Discard Changes" disabled={submitting}>Cancel</button>
              </div>
            </form>
          </div>
        </div>
      )}

      <ConfirmModal
        isOpen={!!confirmDelete}
        onClose={() => setConfirmDelete(null)}
        onConfirm={confirmActualDelete}
        isLoading={submitting}
        title="Are you absolutely sure?"
        message={
          confirmDelete ? (
            <>
              You are about to delete employee <strong>{confirmDelete.first_name} {confirmDelete.last_name}</strong>.
              This action will permanently remove their records from the system.
            </>
          ) : (
            ""
          )
        }
        confirmText="Yes, Delete Employee"
      />

    </div>
  );
}
