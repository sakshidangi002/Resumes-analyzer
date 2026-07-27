import { useState, useEffect } from "react";
import { useAuth } from "../auth/AuthContext";
import { users as usersApi } from "../api/client";
import { employees as employeesApi } from "../api/client";
import ConfirmModal from "../components/ConfirmModal";
import { SectionLoader } from "../components/LoadingState";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import CustomSelect from "../components/CustomSelect";

const ROLES = ["HR", "Manager", "Employee"];

interface UserRow {
  id: number;
  username: string;
  official_email: string | null;
  is_active: boolean;
  employee_id: number | null;
  created_at: string;
  roles: string[];
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
};

export default function ManageUsers() {
  const { hasRole } = useAuth();
  const [list, setList] = useState<UserRow[]>([]);
  const [employees, setEmployees] = useState<Array<{ id: number; employee_code: string; first_name: string; last_name: string }>>([]);
  const [loading, setLoading] = useState(true);
  const [modal, setModal] = useState<"add" | "edit" | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<{ id: number; username: string } | null>(null);
  const [editingId, setEditingId] = useState<number | null>(null);
  const [form, setForm] = useState({
    username: "",
    password: "",
    official_email: "",
    employee_id: "" as string | number,
    role_names: [] as string[],
    is_active: true,
  });
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");

  const load = async () => {
    setLoading(true);
    const [usersRes, employeesRes] = await Promise.allSettled([usersApi.list(), employeesApi.list()]);
    if (usersRes.status === "fulfilled") setList(usersRes.value.data);
    else setList([]);
    if (employeesRes.status === "fulfilled") setEmployees(employeesRes.value.data);
    else setEmployees([]);
    setLoading(false);
  };

  useEffect(() => {
    if (hasRole("Admin")) {
      void load();
    } else {
      setLoading(false);
    }
  }, []);

  const openAdd = () => {
    setForm({
      username: "",
      password: "",
      official_email: "",
      employee_id: "",
      role_names: [],
      is_active: true,
    });
    setEditingId(null);
    setModal("add");
    setError("");
    setSuccess("");
  };

  const openEdit = (u: UserRow) => {
    setForm({
      username: u.username,
      password: "",
      official_email: u.official_email || "",
      employee_id: u.employee_id ?? "",
      role_names: u.roles || [],
      is_active: u.is_active,
    });
    setEditingId(u.id);
    setModal("edit");
    setError("");
    setSuccess("");
  };

  const handleRoleToggle = (role: string) => {
    setForm((f) => ({
      ...f,
      role_names: f.role_names.includes(role)
        ? f.role_names.filter((r) => r !== role)
        : [...f.role_names, role],
    }));
  };

  const handleSubmitAdd = (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (!form.username.trim()) {
      setError("Username required");
      return;
    }
    if (!form.password && modal === "add") {
      setError("Password required");
      return;
    }
    usersApi
      .create({
        username: form.username.trim(),
        password: form.password,
        official_email: form.official_email || undefined,
        employee_id: form.employee_id ? Number(form.employee_id) : undefined,
        role_names: form.role_names,
      })
      .then(() => {
        setSuccess("User created.");
        load();
        setModal(null);
      })
      .catch((err) => setError(err.response?.data?.detail || "Failed"));
  };

  const handleSubmitEdit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!editingId) return;
    setError("");
    usersApi
      .update(editingId, {
        password: form.password || undefined,
        official_email: form.official_email || undefined,
        employee_id: form.employee_id ? Number(form.employee_id) : undefined,
        is_active: form.is_active,
        role_names: form.role_names,
      })
      .then(() => {
        setSuccess("User updated.");
        load();
        setModal(null);
      })
      .catch((err) => setError(err.response?.data?.detail || "Failed"));
  };

  const handleDelete = (id: number, username: string) => {
    setConfirmDelete({ id, username });
  };

  const confirmActualDelete = () => {
    if (!confirmDelete) return;
    const { id } = confirmDelete;
    usersApi
      .delete(id)
      .then(() => {
        setSuccess("User deleted.");
        setList((prev) => prev.filter((u) => u.id !== id));
        setConfirmDelete(null);
      })
      .catch((err) => {
        setError(err.response?.data?.detail || "Failed");
        setConfirmDelete(null);
      });
  };

  if (!hasRole("Admin")) {
    return (
      <div className="card">
        <p>Access denied. Admin only.</p>
      </div>
    );
  }


  return (
    <>
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="page-title">Manage Users / Add HR</h1>
          <div className="page-subtitle">Create users and assign roles</div>
        </div>
        <GlobalHeaderControls />
      </div>
      {success && <div className="alert alert-success">{success}</div>}
      {error && <div className="alert alert-error">{error}</div>}
      <div className="card">
        <div className="flex gap-2 mb-2" style={{ justifyContent: "flex-end" }}>
          <button
            type="button"
            className="btn btn-primary"
            onClick={openAdd}
            style={{ padding: "0.9rem 1.25rem", lineHeight: 1.1 }}
          >
            Add User (HR / Manager / Employee)
          </button>
        </div>
        {loading ? (
          <div style={{ padding: "3rem 0" }}><SectionLoader size="md" /></div>
        ) : (
          <div className="table-wrap table-wrap--dark">
            <table className="table-modern table-modern--dark">
              <thead>
                <tr>
                  <th>Username</th>
                  <th>Email</th>
                  <th>Roles</th>
                  <th style={{ textAlign: "center" }}>Employee</th>
                  <th style={{ textAlign: "center" }}>Active</th>
                  <th className="actions-center">Actions</th>
                </tr>
              </thead>
              <tbody>
                {list.map((u) => (
                  <tr key={u.id}>
                    <td style={{ color: "#fff", fontWeight: 700 }}>{u.username}</td>
                    <td style={{ color: "#fff" }}>{u.official_email || "-"}</td>
                    <td style={{ color: "#fff" }}>{u.roles.join(", ")}</td>
                    <td style={{ color: "#fff", textAlign: "center" }}>{u.employee_id ? employees.find((e) => e.id === u.employee_id)?.employee_code || u.employee_id : "-"}</td>
                    <td style={{ color: "#fff", textAlign: "center" }}>{u.is_active ? "Yes" : "No"}</td>
                    <td className="actions-center">
                      <div className="actions-stack">
                        <button type="button" className="btn btn-secondary btn-icon btn-sm" onClick={() => openEdit(u)} title="Edit User">
                          <Icons.Edit />
                        </button>
                        <button type="button" className="btn btn-danger btn-icon btn-sm" onClick={() => handleDelete(u.id, u.username)} title="Delete User">
                          <Icons.Delete />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        {!loading && list.length === 0 && <p className="text-muted">No users.</p>}
      </div>

      {modal === "add" && (
        <div className="modal-backdrop" onClick={() => setModal(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 600 }}>
            <h3 style={{ marginTop: 0 }}>Add User</h3>
            <form onSubmit={handleSubmitAdd}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
                <div className="form-group">
                  <label>Username *</label>
                  <input value={form.username} onChange={(e) => setForm({ ...form, username: e.target.value })} required />
                </div>
                <div className="form-group">
                  <label>Password *</label>
                  <input type="password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} required />
                </div>
                <div className="form-group">
                  <label>Email</label>
                  <input type="email" value={form.official_email} onChange={(e) => setForm({ ...form, official_email: e.target.value })} />
                </div>
                <div className="form-group">
                  <label>Link to Employee</label>
                  <CustomSelect
                    value={String(form.employee_id || "")}
                    onChange={(val) => setForm({ ...form, employee_id: val })}
                    options={[
                      { value: "", label: "— None —" },
                      ...employees.map((e) => ({ value: String(e.id), label: `${e.employee_code} – ${e.first_name} ${e.last_name}` }))
                    ]}
                  />
                </div>
              </div>
              <div className="form-group">
                <label style={{ marginBottom: "8px", display: "block" }}>Roles</label>
                <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
                  {ROLES.map((r) => (
                    <label key={r} style={{ display: "flex", alignItems: "center", gap: "6px", cursor: "pointer" }}>
                      <input type="checkbox" checked={form.role_names.includes(r)} onChange={() => handleRoleToggle(r)} style={{ width: "auto" }} /> {r}
                    </label>
                  ))}
                </div>
              </div>
              <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.5rem", marginTop: "1.5rem" }}>
                <button type="submit" className="btn btn-primary">Create</button>
                <button type="button" className="btn btn-cancel-alt" onClick={() => setModal(null)}>Cancel</button>
              </div>
            </form>
          </div>
        </div>
      )}

      {modal === "edit" && editingId && (
        <div className="modal-backdrop" onClick={() => setModal(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 600 }}>
            <h3 style={{ marginTop: 0 }}>Edit User</h3>
            <form onSubmit={handleSubmitEdit}>
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem" }}>
                <div className="form-group">
                  <label>Username</label>
                  <input value={form.username} disabled />
                </div>
                <div className="form-group">
                  <label>New password (leave blank to keep)</label>
                  <input type="password" value={form.password} onChange={(e) => setForm({ ...form, password: e.target.value })} />
                </div>
                <div className="form-group">
                  <label>Email</label>
                  <input type="email" value={form.official_email} onChange={(e) => setForm({ ...form, official_email: e.target.value })} />
                </div>
                <div className="form-group">
                  <label>Link to Employee</label>
                  <CustomSelect
                    value={String(form.employee_id || "")}
                    onChange={(val) => setForm({ ...form, employee_id: val })}
                    options={[
                      { value: "", label: "— None —" },
                      ...employees.map((e) => ({ value: String(e.id), label: `${e.employee_code} – ${e.first_name} ${e.last_name}` }))
                    ]}
                  />
                </div>
              </div>
              <div className="form-group">
                <label style={{ marginBottom: "8px", display: "block" }}>Roles</label>
                <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
                  {ROLES.map((r) => (
                    <label key={r} style={{ display: "flex", alignItems: "center", gap: "6px", cursor: "pointer" }}>
                      <input type="checkbox" checked={form.role_names.includes(r)} onChange={() => handleRoleToggle(r)} style={{ width: "auto" }} /> {r}
                    </label>
                  ))}
                </div>
              </div>
              <div className="form-group">
                <label style={{ display: "flex", alignItems: "center", gap: "6px", cursor: "pointer" }}>
                  <input type="checkbox" checked={form.is_active} onChange={(e) => setForm({ ...form, is_active: e.target.checked })} style={{ width: "auto" }} /> Active
                </label>
              </div>
              <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.5rem", marginTop: "1.5rem" }}>
                <button type="submit" className="btn btn-primary">Update</button>
                <button type="button" className="btn btn-cancel-alt" onClick={() => setModal(null)}>Cancel</button>
              </div>
            </form>
          </div>
        </div>
      )}
      <ConfirmModal
        isOpen={!!confirmDelete}
        onClose={() => setConfirmDelete(null)}
        onConfirm={confirmActualDelete}
        title="Are you absolutely sure?"
        message={
          confirmDelete ? (
            <>
              You are about to delete user <strong>{confirmDelete.username}</strong>. This action will permanently remove
              their login access.
            </>
          ) : (
            ""
          )
        }
        confirmText="Yes, Delete User"
      />
    </>
  );
}
