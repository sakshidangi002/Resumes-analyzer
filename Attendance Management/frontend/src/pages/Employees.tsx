import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { employees as api, users as usersApi } from "../api/client";
import ConfirmModal from "../components/ConfirmModal";
import { SectionLoader } from "../components/LoadingState";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import CustomSelect from "../components/CustomSelect";
import { useTableControls, SortableHeader, TableToolbar } from "../components/dataTable";

interface Emp {
  id: number;
  employee_code: string;
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

const emptyForm = (): Record<string, string | number | undefined> => ({
  employee_code: "",
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
const Icons = {
  View: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z"></path>
      <circle cx="12" cy="12" r="3"></circle>
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
};

export default function Employees() {
  const { hasRole } = useAuth();
  const navigate = useNavigate();
  const canEdit = hasRole("Admin") || hasRole("HR");
  const [list, setList] = useState<Emp[]>([]);
  const [departments, setDepartments] = useState<Array<{ id: number; name: string }>>([]);
  const [designations, setDesignations] = useState<Array<{ id: number; title: string }>>([]);
  const [loading, setLoading] = useState(true);
  const [modal, setModal] = useState<"add" | null>(null);
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
      official_email: (e) => e.official_email,
      department: (e) => deptName(e.department_id),
      date_of_joining: (e) => e.date_of_joining,
      date_of_leaving: (e) => e.date_of_leaving || "",
      employment_status: (e) => e.employment_status,
    },
    searchableText: (e) =>
      `${e.employee_code} ${e.first_name} ${e.last_name} ${e.official_email} ${deptName(e.department_id)} ${e.employment_status}`,
  });

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

  const handleSubmitAdd = (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    const payload: Record<string, unknown> = {
      employee_code: form.employee_code,
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


  return (
    <>
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="page-title">Employees</h1>
          <div className="page-subtitle">View and manage employee records</div>
        </div>
        <GlobalHeaderControls />
      </div>
      {createdLogin && (
        <div className="card">
          <p><strong>Login created for employee.</strong> Share these credentials securely with the employee:</p>
          <p className="text-muted">
            Username: <strong>{createdLogin.username}</strong> &nbsp;|&nbsp;
            Password: <strong>{createdLogin.password}</strong>
          </p>
        </div>
      )}
      {success && <div className="alert alert-success">{success}</div>}
      {error && <div className="alert alert-error">{error}</div>}
      <div className="card">
        <TableToolbar
          search={search}
          onSearchChange={setSearch}
          placeholder="Search employees (name, code, email, department)..."
          showClear={hasActiveControls || !!filterDept || !!filterStatus}
          onClear={() => {
            clearAll();
            setFilterDept("");
            setFilterStatus("");
          }}
          count={{ shown: displayedList.length, total: list.length }}
          leftControls={
            <>
              <div className="form-group" style={{ marginBottom: 0, minWidth: "180px" }}>
                <CustomSelect
                  value={filterDept}
                  onChange={setFilterDept}
                  placeholder="All Departments"
                  options={[
                    { value: "", label: "All Departments" },
                    ...departments.map((d) => ({ value: String(d.id), label: d.name }))
                  ]}
                />
              </div>
              <div className="form-group" style={{ marginBottom: 0, minWidth: "160px" }}>
                <CustomSelect
                  value={filterStatus}
                  onChange={setFilterStatus}
                  placeholder="All Status"
                  options={[
                    { value: "", label: "All Status" },
                    ...EMPLOYMENT_STATUSES.map((s) => ({ value: s, label: s }))
                  ]}
                />
              </div>
            </>
          }
          rightControls={
            canEdit ? (
              <button type="button" className="btn btn-primary" onClick={openAdd} title="Add New Employee" style={{ height: "42px", minWidth: "140px" }}>
                Add Employee
              </button>
            ) : null
          }
        />
        {loading ? (
          <div style={{ padding: "3rem 0" }}><SectionLoader size="md" /></div>
        ) : list.length === 0 ? (
          <p className="text-muted">No employee records found.</p>
        ) : (
          <div className="table-responsive">
            <div className="table-wrap table-wrap--dark">
              <table className="table-modern table-modern--dark" style={{ tableLayout: 'fixed', width: '100%' }}>
                <colgroup>
                  <col style={{ width: '80px' }} />
                  <col style={{ width: '16%' }} />
                  <col style={{ width: '20%' }} />
                  <col style={{ width: '16%' }} />
                  <col style={{ width: '130px' }} />
                  <col style={{ width: '130px' }} />
                  <col style={{ width: '110px' }} />
                  {canEdit && <col style={{ width: '120px' }} />}
                </colgroup>
                <thead>
                  <tr>
                    <SortableHeader className="hide-md" label="ID" columnKey="employee_code" sort={sort} onToggle={toggleSort} style={{ paddingLeft: '1.5rem' }} />
                    <SortableHeader label="Name" columnKey="name" sort={sort} onToggle={toggleSort} />
                    <SortableHeader label="Email" columnKey="official_email" sort={sort} onToggle={toggleSort} />
                    <SortableHeader className="hide-sm" label="Department" columnKey="department" sort={sort} onToggle={toggleSort} />
                    <SortableHeader className="hide-md" label="DOJ" columnKey="date_of_joining" sort={sort} onToggle={toggleSort} />
                    <SortableHeader className="hide-md" label="DOL" columnKey="date_of_leaving" sort={sort} onToggle={toggleSort} />
                    <SortableHeader className="hide-sm" label="Status" columnKey="employment_status" sort={sort} onToggle={toggleSort} />
                    {canEdit && (
                      <SortableHeader label="Actions" columnKey="__actions" sort={sort} onToggle={toggleSort} align="center" notSortable />
                    )}
                  </tr>
                </thead>
                <tbody>
                  {displayedList.length === 0 ? (
                    <tr>
                      <td colSpan={canEdit ? 8 : 7} style={{ textAlign: 'center', padding: '1.5rem', opacity: 0.65 }}>
                        No employees match your search / filters.
                      </td>
                    </tr>
                  ) : null}
                  {displayedList.map((e) => {
                    const hasLeft = !!e.date_of_leaving;
                    const rowStyle: React.CSSProperties = hasLeft
                      ? {
                          background: 'rgba(239, 68, 68, 0.08)',
                          color: 'rgba(248, 113, 113, 0.95)',
                        }
                      : {};
                    const dolText = e.date_of_leaving
                      ? new Date(e.date_of_leaving).toLocaleDateString('en-GB', { day: 'numeric', month: 'short', year: 'numeric' })
                      : '-';
                    return (
                    <tr key={e.id} style={rowStyle} title={hasLeft ? `Employee left on ${dolText}` : undefined}>
                      <td className="hide-md" style={{ textAlign: 'left', paddingLeft: '1.5rem', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.employee_code}</td>
                      <td style={{ textAlign: 'left', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.first_name} {e.last_name}</td>
                      <td style={{ textAlign: 'left', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{e.official_email}</td>
                      <td className="hide-sm" style={{ textAlign: 'left', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{departments.find((d) => d.id === e.department_id)?.name || "-"}</td>
                      <td className="hide-md" style={{ textAlign: 'left', whiteSpace: 'nowrap' }}>{e.date_of_joining ? new Date(e.date_of_joining).toLocaleDateString("en-GB", { day: 'numeric', month: 'short', year: 'numeric' }) : "-"}</td>
                      <td className="hide-md" style={{ textAlign: 'left', whiteSpace: 'nowrap', fontWeight: hasLeft ? 600 : 400 }}>{dolText}</td>
                      <td className="hide-sm" style={{ textAlign: 'left', whiteSpace: 'nowrap' }}>{e.employment_status}</td>
                      {canEdit && (
                        <td style={{ textAlign: 'center' }}>
                          <div className="actions-stack" style={{ display: 'flex', gap: '0.75rem', justifyContent: 'center' }}>
                            <button
                              type="button"
                              className="btn btn-secondary btn-icon btn-sm"
                              onClick={() => navigate(`/employees/${e.id}`)}
                              title="View Employee Profile"
                              style={{ padding: '0.4rem 0.6rem' }}
                            >
                              <Icons.View />
                            </button>
                            <button
                              type="button"
                              className="btn btn-danger btn-icon btn-sm"
                              onClick={() => setConfirmDelete(e)}
                              title="Delete Employee Permanently"
                              style={{ padding: '0.4rem 0.6rem' }}
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
          </div>
        )}
      </div>

      {modal === "add" && (
        <div className="modal-backdrop" onClick={() => setModal(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 900 }}>
            <h3 style={{ marginTop: 0 }}>Add Employee</h3>
            <form onSubmit={handleSubmitAdd}>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "1rem" }}>

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

    </>
  );
}
