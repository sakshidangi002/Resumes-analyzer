import { useEffect, useState } from "react";
import { payroll as payrollApi } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import ConfirmModal from "../components/ConfirmModal";
import { SectionLoader } from "../components/LoadingState";
import CustomSelect from "../components/CustomSelect";
import { useTableControls, SortableHeader, TableToolbar } from "../components/dataTable";


interface PayrollPeriod {
  id: number;
  month: number;
  year: number;
  status: string;
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

export default function Payroll() {
  const { hasRole } = useAuth();
  const [periods, setPeriods] = useState<PayrollPeriod[]>([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [runningPeriodId, setRunningPeriodId] = useState<number | null>(null);
  const [editing, setEditing] = useState<PayrollPeriod | null>(null);
  const [editForm, setEditForm] = useState({ month: 1, year: 2020, status: "OPEN" });
  const [showAdd, setShowAdd] = useState(false);
  const now = new Date();
  const [addForm, setAddForm] = useState({ month: now.getMonth() + 1, year: now.getFullYear() });
  const [confirmDelete, setConfirmDelete] = useState<PayrollPeriod | null>(null);

  const canRun = hasRole("Admin") || hasRole("HR");
  const canManage = hasRole("Admin") || hasRole("HR");

  const loadPeriods = async () => {
    setLoading(true);
    try {
      const r = await payrollApi.periods();
      setPeriods(r.data);
    } catch {
      setPeriods([]);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    void loadPeriods();
  }, []);

  const runPayroll = (periodId: number) => {
    if (!canRun) return;
    if (runningPeriodId === periodId) return;
    setRunningPeriodId(periodId);
    payrollApi
      .runPayroll(periodId)
      .then(() => {
        // Run payroll usually updates status or creates slips, 
        // it's safer to reload the whole list here as multiple records change.
        loadPeriods();
      })
      .catch((err) => {
        alert(err.response?.data?.detail || "Failed to run payroll");
      })
      .finally(() => setRunningPeriodId(null));
  };

  const openEdit = (p: PayrollPeriod) => {
    setEditing(p);
    setEditForm({ month: p.month, year: p.year, status: p.status });
  };

  const saveEdit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!editing) return;
    const payload: { status?: string; month?: number; year?: number } = {};
    if (editForm.status !== editing.status) payload.status = editForm.status;
    if (editForm.month !== editing.month) payload.month = editForm.month;
    if (editForm.year !== editing.year) payload.year = editForm.year;

    if (Object.keys(payload).length === 0) {
      setEditing(null);
      return;
    }

    setSubmitting(true);
    payrollApi
      .updatePeriod(editing.id, payload)
      .then((res) => {
        const updated = res.data as PayrollPeriod;
        setPeriods(prev => prev.map(p => p.id === updated.id ? updated : p));
        setEditing(null);
      })
      .catch((err) => alert(err.response?.data?.detail || "Update failed"))
      .finally(() => setSubmitting(false));
  };

  const deletePeriod = (p: PayrollPeriod) => {
    setConfirmDelete(p);
  };

  const confirmActualDelete = () => {
    if (!confirmDelete) return;
    setSubmitting(true);
    payrollApi
      .deletePeriod(confirmDelete.id)
      .then(() => {
        setPeriods(prev => prev.filter(p => p.id !== confirmDelete.id));
        setConfirmDelete(null);
      })
      .catch((err) => alert(err.response?.data?.detail || "Delete failed"))
      .finally(() => setSubmitting(false));
  };

  const monthName = (m: number) => new Date(2000, m - 1).toLocaleString("default", { month: "long" });

  const {
    displayed: displayedPeriods,
    search: periodSearch,
    setSearch: setPeriodSearch,
    sort: periodSort,
    toggleSort: togglePeriodSort,
    clearAll: clearPeriodControls,
    hasActiveControls: periodHasActive,
  } = useTableControls<PayrollPeriod>({
    rows: periods,
    columns: {
      month: (p) => p.month,
      year: (p) => p.year,
      status: (p) => p.status,
    },
    searchableText: (p) => `${monthName(p.month)} ${p.year} ${p.status}`,
  });

  const handleAddSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setSubmitting(true);
    payrollApi
      .createPeriod(addForm.month, addForm.year)
      .then((res) => {
        setPeriods(prev => [...prev, res.data]);
        setShowAdd(false);
      })
      .catch((err) => alert(err.response?.data?.detail || "Create failed"))
      .finally(() => setSubmitting(false));
  };

  if (!canManage) {
    return (
      <div className="card">
        <p className="text-muted">You do not have access to payroll periods.</p>
      </div>
    );
  }


  return (
    <>
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="page-title">Payroll Periods</h1>
          <div className="page-subtitle">Create periods, run payroll, and manage payouts</div>
        </div>
        <GlobalHeaderControls />
      </div>
      <div className="card">
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem" }}>
          <h3 style={{ marginTop: 0, marginBottom: 0 }}>Payroll periods</h3>
        </div>
        <TableToolbar
          search={periodSearch}
          onSearchChange={setPeriodSearch}
          placeholder="Search by month, year, status..."
          showClear={periodHasActive}
          onClear={clearPeriodControls}
          count={{ shown: displayedPeriods.length, total: periods.length }}
          rightControls={
            <button
              type="button"
              className="btn btn-primary btn-uniform"
              onClick={() => setShowAdd(true)}
            >
              Add period
            </button>
          }
        />
        {loading ? (
          <div style={{ padding: "3rem 0" }}><SectionLoader size="md" /></div>
        ) : periods.length === 0 ? (
          <p className="text-muted">No payroll periods.</p>
        ) : (
          <div className="table-wrap table-wrap--dark" style={{ marginTop: "1.25rem" }}>
            <table className="table-modern table-modern--dark" style={{ tableLayout: 'fixed', width: '100%' }}>
              <thead>
                <tr>
                  <SortableHeader label="Month" columnKey="month" sort={periodSort} onToggle={togglePeriodSort} style={{ width: '25%', paddingLeft: '1.5rem' }} />
                  <SortableHeader label="Year" columnKey="year" sort={periodSort} onToggle={togglePeriodSort} style={{ width: '25%' }} />
                  <SortableHeader label="Status" columnKey="status" sort={periodSort} onToggle={togglePeriodSort} style={{ width: '25%' }} />
                  <SortableHeader label="Action" columnKey="__actions" sort={periodSort} onToggle={togglePeriodSort} notSortable align="right" style={{ width: '25%' }} />
                </tr>
              </thead>
              <tbody>
                {displayedPeriods.length === 0 && (
                  <tr>
                    <td colSpan={4} style={{ textAlign: 'center', padding: '1.25rem', opacity: 0.65 }}>
                      No payroll periods match your search.
                    </td>
                  </tr>
                )}
                {displayedPeriods.map((p) => (
                  <tr key={p.id}>
                    <td style={{ textAlign: 'left', paddingLeft: '1.5rem' }}>{monthName(p.month)}</td>
                    <td>{p.year}</td>
                    <td>{p.status}</td>
                    <td className="actions-center">
                      <div className="actions-stack">
                        <button type="button" className="btn btn-secondary btn-icon btn-sm" onClick={() => openEdit(p)} title="Edit Period">
                          <Icons.Edit />
                        </button>
                        <button type="button" className="btn btn-danger btn-icon btn-sm" onClick={() => deletePeriod(p)} title="Delete Period">
                          <Icons.Delete />
                        </button>
                        {canRun && p.status === "OPEN" && (
                          <button
                            type="button"
                            className="btn btn-primary btn-sm btn-uniform"
                            onClick={() => runPayroll(p.id)}
                            disabled={runningPeriodId === p.id}
                          >
                            {runningPeriodId === p.id ? "Running…" : "Run payroll"}
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

      {showAdd && (
        <div className="modal-backdrop">
          <div className="modal" style={{ maxWidth: 360 }}>
            <h3 style={{ marginTop: 0 }}>Add period</h3>
            <form onSubmit={handleAddSubmit}>
              <div className="form-group">
                <label>Month</label>
                <CustomSelect
                  value={String(addForm.month)}
                  onChange={(val) => setAddForm((f) => ({ ...f, month: Number(val) }))}
                  options={[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map((m) => ({
                    value: String(m),
                    label: new Date(2000, m - 1).toLocaleString("default", { month: "long" })
                  }))}
                />
              </div>
              <div className="form-group">
                <label>Year</label>
                <CustomSelect
                  value={String(addForm.year)}
                  onChange={(val) => setAddForm((f) => ({ ...f, year: Number(val) }))}
                  disabled={submitting}
                  options={[2026].map(y => ({ value: String(y), label: String(y) }))}
                />
              </div>
              <div style={{ marginTop: "1rem", display: "flex", gap: "0.5rem", justifyContent: "flex-end" }}>

                <button type="submit" className="btn btn-primary" disabled={submitting}>
                  {submitting ? "Saving..." : "Save"}
                </button>
                <button type="button" className="btn btn-secondary" onClick={() => setShowAdd(false)} disabled={submitting} style={{ color: "#ef4444", background: "rgba(239, 68, 68, 0.15)", }}>
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {editing && (
        <div className="modal-backdrop">
          <div className="modal" style={{ maxWidth: 360 }}>
            <h3 style={{ marginTop: 0 }}>Edit period</h3>
            <form onSubmit={saveEdit}>
              <div className="form-group">
                <label>Month</label>
                <CustomSelect
                  value={String(editForm.month)}
                  onChange={(val) => setEditForm((f) => ({ ...f, month: Number(val) }))}
                  options={[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12].map((m) => ({
                    value: String(m),
                    label: new Date(2000, m - 1).toLocaleString("default", { month: "long" })
                  }))}
                />
              </div>
              <div className="form-group">
                <label>Year</label>
                <CustomSelect
                  value={String(editForm.year)}
                  onChange={(val) => setEditForm((f) => ({ ...f, year: Number(val) }))}
                  disabled={submitting}
                  options={[2026].map(y => ({ value: String(y), label: String(y) }))}
                />
              </div>
              <div className="form-group">
                <label>Status</label>
                <CustomSelect
                  value={editForm.status}
                  onChange={(val) => setEditForm((f) => ({ ...f, status: val }))}
                  options={[
                    { value: "OPEN", label: "OPEN" },
                    { value: "PROCESSED", label: "PROCESSED" },
                    { value: "LOCKED", label: "LOCKED" },
                  ]}
                />
              </div>
              <div style={{ marginTop: "1rem", display: "flex", justifyContent: "flex-end", gap: "0.5rem" }}>

                <button type="submit" className="btn btn-primary" disabled={submitting}>
                  {submitting ? "Saving..." : "Save"}
                </button>
                <button type="button" className="btn btn-secondary" onClick={() => setEditing(null)} disabled={submitting} style={{ color: "#ef4444", background: "rgba(239, 68, 68, 0.15)", }}>
                  Cancel
                </button>
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
              You are about to delete payroll period <strong>{confirmDelete.month}/{confirmDelete.year}</strong>.
              This is only allowed if no payslips exist for this period.
            </>
          ) : (
            ""
          )
        }
        confirmText="Yes, Delete Period"
      />
    </>
  );
}
