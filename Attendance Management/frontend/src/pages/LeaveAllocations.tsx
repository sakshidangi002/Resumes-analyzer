import { useEffect, useState } from "react";
import { useAuth } from "../auth/AuthContext";
import { leave as leaveApi, company as companyApi, employees as employeesApi } from "../api/client";
import { SectionLoader } from "../components/LoadingState";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import { formatDate } from "../utils/dateFormatter";
import CustomSelect from "../components/CustomSelect";

interface LeaveType {
  id: number;
  code: string;
  name: string;
}

interface FinancialYear {
  id: number;
  name: string;
  start_date: string;
  end_date: string;
}

interface Allocation {
  id: number;
  employee_id: number;
  financial_year_id: number;
  leave_type_id: number;
  allocated_days: number;
  used_days: number;
  balance_days: number;
}

interface EmployeeOption {
  id: number;
  employee_code: string;
  full_name: string;
}

// Premium SVG Icons for Actions
const Icons = {
  Edit: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7"></path>
      <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z"></path>
    </svg>
  ),
};

export default function LeaveAllocations() {
  const { hasRole } = useAuth();
  const [financialYears, setFinancialYears] = useState<FinancialYear[]>([]);
  const [selectedFyId, setSelectedFyId] = useState<number | null>(null);
  const [types, setTypes] = useState<LeaveType[]>([]);
  const [allocations, setAllocations] = useState<Allocation[]>([]);
  const [employees, setEmployees] = useState<EmployeeOption[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [modal, setModal] = useState<"add" | "edit" | null>(null);
  const [editRow, setEditRow] = useState<Allocation | null>(null);
  const [form, setForm] = useState({
    employee_id: "",
    leave_type_id: "",
    allocated_days: "",
  });

  const canManage = hasRole("Admin") || hasRole("HR");

  useEffect(() => {
    Promise.all([companyApi.financialYears(), leaveApi.types(), employeesApi.list()])
      .then(([fyRes, tRes, eRes]) => {
        setFinancialYears(fyRes.data || []);
        setTypes(tRes.data || []);
        setEmployees(
          (eRes.data as any[]).map((e: any) => ({
            id: e.id,
            employee_code: e.employee_code,
            full_name: e.full_name,
          }))
        );
        if ((fyRes.data || []).length > 0 && !selectedFyId)
          setSelectedFyId((fyRes.data as FinancialYear[])[0].id);
      })
      .catch(() => { })
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!selectedFyId || !canManage) {
      setAllocations([]);
      return;
    }
    setLoading(true);
    leaveApi
      .allocations({ financial_year_id: selectedFyId })
      .then((res) => setAllocations(res.data || []))
      .catch(() => setAllocations([]))
      .finally(() => setLoading(false));
  }, [selectedFyId, canManage]);

  const employeeLabel = (id: number) => {
    const e = employees.find((x) => x.id === id);
    return e ? `${e.employee_code} - ${e.full_name}` : `#${id}`;
  };

  const typeName = (id: number) => types.find((t) => t.id === id)?.name ?? `#${id}`;

  const openAdd = () => {
    setEditRow(null);
    setForm({ employee_id: "", leave_type_id: "", allocated_days: "" });
    setModal("add");
    setError("");
    setSuccess("");
  };

  const openEdit = (row: Allocation) => {
    setEditRow(row);
    setForm({
      employee_id: String(row.employee_id),
      leave_type_id: String(row.leave_type_id),
      allocated_days: String(row.allocated_days),
    });
    setModal("edit");
    setError("");
    setSuccess("");
  };

  const handleSave = (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedFyId) return;
    const allocated = Number(form.allocated_days);
    if (isNaN(allocated) || allocated < 0) {
      setError("Allocated days must be 0 or more.");
      return;
    }
    if (editRow && allocated < editRow.used_days) {
      setError(`Allocated days cannot be less than already used (${editRow.used_days}).`);
      return;
    }
    setError("");
    leaveApi
      .setAllocation({
        employee_id: Number(form.employee_id),
        leave_type_id: Number(form.leave_type_id),
        allocated_days: allocated,
        financial_year_id: selectedFyId,
      })
      .then(() => {
        setSuccess(editRow ? "Allocation updated." : "Allocation added.");
        setModal(null);
        leaveApi
          .allocations({ financial_year_id: selectedFyId })
          .then((res) => setAllocations(res.data || []));
      })
      .catch((err) => setError(err.response?.data?.detail || "Failed to save."));
  };

  if (!canManage) {
    return (
      <div className="card">
        <p>Access denied. HR/Admin only.</p>
      </div>
    );
  }

  return (
    <>
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="page-title">Leave Allocations</h1>
          <div className="page-subtitle">
            Set or edit paid/unpaid leave days per employee per financial year.
          </div>
        </div>
        <GlobalHeaderControls />
      </div>

      {success && <div className="alert alert-success">{success}</div>}
      {error && <div className="alert alert-error">{error}</div>}

      <div className="card">
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "flex-end", justifyContent: "space-between", gap: "1rem", marginBottom: "2rem" }}>
          <div className="form-group" style={{ marginBottom: 0 }}>
            <label>Financial year</label>
            <CustomSelect
              value={String(selectedFyId ?? "")}
              onChange={(val) => setSelectedFyId(Number(val) || null)}
              style={{ width: "320px", maxWidth: "100%" }}
              options={financialYears.map((fy) => ({
                value: String(fy.id),
                label: `${fy.name} (${formatDate(fy.start_date)} to ${formatDate(fy.end_date)})`
              }))}
            />
          </div>
          <div style={{ alignSelf: "flex-end" }}>
            <button type="button" className="btn btn-primary btn-uniform" onClick={openAdd} title="Assign New Leave Allocation for an Employee">
              Add allocation
            </button>
          </div>
        </div>

        {loading ? (
          <SectionLoader rows={4} />
        ) : allocations.length === 0 ? (
          <p className="text-muted">
            No leave allocations for this financial year. Click &quot;Add allocation&quot; to set days for an employee (e.g. 1 paid leave, 0, or more as per policy).
          </p>
        ) : (() => {
          const sortedAllocations = [...allocations].sort((a, b) => {
            const empA = employees.find(e => e.id === a.employee_id);
            const empB = employees.find(e => e.id === b.employee_id);
            const codeA = parseInt(String(empA?.employee_code || "0").replace(/\D/g, ""), 10) || 0;
            const codeB = parseInt(String(empB?.employee_code || "0").replace(/\D/g, ""), 10) || 0;
            if (codeA !== codeB) return codeA - codeB;
            return a.leave_type_id - b.leave_type_id;
          });

          const groupedMap = new Map<number, Allocation[]>();
          sortedAllocations.forEach(a => {
            if (!groupedMap.has(a.employee_id)) groupedMap.set(a.employee_id, []);
            groupedMap.get(a.employee_id)!.push(a);
          });

          return (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fill, minmax(320px, 1fr))", gap: "1.5rem" }}>
              {Array.from(groupedMap.entries()).map(([empId, allocs]) => {
                const label = employeeLabel(empId);
                const parts = label.split("-");
                const code = parts[0]?.trim() || "";
                const name = parts[1]?.trim() || label;
                const initial = name.charAt(0).toUpperCase();
                return (
                  <div key={empId} className="card" style={{ display: "flex", flexDirection: "column", padding: "1.5rem", background: "rgba(255,255,255,0.02)", margin: 0, color: "#fff" }}>
                    <div style={{ display: "flex", alignItems: "center", gap: "1rem", marginBottom: "1.5rem", paddingBottom: "1rem", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
                      <div style={{
                        width: 44, height: 44, borderRadius: "50%", background: "var(--brand-500)",
                        display: "flex", alignItems: "center", justifyContent: "center", fontWeight: "bold", fontSize: "1.2rem", color: "#fff", flexShrink: 0
                      }}>
                        {initial}
                      </div>
                      <div style={{ minWidth: 0, flex: 1 }}>
                        <h3 style={{ margin: 0, fontSize: "1.1rem", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }}>{name}</h3>
                        <div style={{ fontSize: "0.85rem", opacity: 0.7 }}>{code}</div>
                      </div>
                    </div>
                    <div style={{ display: "flex", flexDirection: "column", gap: "1rem", flex: 1 }}>
                      {allocs.map((a) => (
                        <div key={a.id} style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "0.75rem", background: "rgba(0,0,0,0.2)", borderRadius: "8px" }}>
                          <div>
                            <div style={{ fontWeight: 600 }}>{typeName(a.leave_type_id)}</div>
                            <div style={{ fontSize: "0.8rem", opacity: 0.7, marginTop: "0.2rem" }}>
                              Alloc: {Math.round(a.allocated_days)} &bull; Used: {Math.round(Number(a.used_days))}
                            </div>
                          </div>
                          <div style={{ display: "flex", alignItems: "center", gap: "1rem" }}>
                            <div style={{ textAlign: "right" }}>
                              <div style={{ fontSize: "0.75rem", opacity: 0.7 }}>Balance</div>
                              <div style={{ fontWeight: 700, color: "var(--brand-400)", fontSize: "1.1rem" }}>{Math.round(Number(a.balance_days))}</div>
                            </div>
                            <button
                              type="button"
                              className="btn btn-secondary btn-icon btn-sm"
                              onClick={() => openEdit(a)}
                              title="Edit Allocation"
                            >
                              <Icons.Edit />
                            </button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          );
        })()}
      </div>

      {modal && (
        <div className="modal-backdrop" onClick={() => setModal(null)}>
          <div className="modal" style={{ maxWidth: 420 }} onClick={(e) => e.stopPropagation()}>
            <h3 style={{ marginTop: 0 }}>{editRow ? "Edit allocation" : "Add allocation"}</h3>
            <form onSubmit={handleSave}>
              <div className="form-group">
                <label>Employee</label>
                <CustomSelect
                  value={form.employee_id}
                  onChange={(val) => setForm((f) => ({ ...f, employee_id: val }))}
                  options={[
                    { value: "", label: "Select employee" },
                    ...[...employees].sort((a, b) => {
                      const nA = parseInt(a.employee_code.replace(/\D/g, ""), 10) || 0;
                      const nB = parseInt(b.employee_code.replace(/\D/g, ""), 10) || 0;
                      return nA - nB;
                    }).map((e) => ({
                      value: String(e.id),
                      label: `${e.employee_code} - ${e.full_name}`
                    }))
                  ]}
                />
              </div>
              <div className="form-group">
                <label>Leave type</label>
                <CustomSelect
                  value={form.leave_type_id}
                  onChange={(val) => setForm((f) => ({ ...f, leave_type_id: val }))}
                  options={[
                    { value: "", label: "Select leave type" },
                    ...types.map((t) => ({
                      value: String(t.id),
                      label: `${t.name} (${t.code})`
                    }))
                  ]}
                />
              </div>
              <div className="form-group">
                <label>Allocated days</label>
                <input
                  type="number"
                  min={editRow ? editRow.used_days : 0}
                  step="0.5"
                  required
                  value={form.allocated_days}
                  onChange={(e) => setForm((f) => ({ ...f, allocated_days: e.target.value }))}
                />
                {editRow && (
                  <p className="text-muted" style={{ fontSize: "0.8rem", marginTop: 4 }}>
                    Cannot be less than already used ({editRow.used_days}).
                  </p>
                )}
              </div>
              <div style={{ marginTop: "1rem", display: "flex", justifyContent: "flex-end", gap: "0.5rem" }}>
                <button type="submit" className="btn btn-primary btn-uniform" title="Save Allocation Changes">
                  Save
                </button>
                <button type="button" className="btn btn-cancel-alt btn-uniform" onClick={() => setModal(null)} title="Cancel Changes">
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </>
  );
}
