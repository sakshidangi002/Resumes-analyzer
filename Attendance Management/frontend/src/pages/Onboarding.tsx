import { useEffect, useState } from "react";
import { employees as employeesApi, onboarding as onboardingApi, type OnboardingTaskRow } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import ConfirmModal from "../components/ConfirmModal";
import { SectionLoader } from "../components/LoadingState";
import CustomSelect from "../components/CustomSelect";
import GlobalHeaderControls from "../components/GlobalHeaderControls";

type EmpOpt = { id: number; full_name: string; employee_code: string };

// Premium SVG Icons for Actions
const Icons = {
  Delete: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6"></polyline>
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path>
      <line x1="10" y1="11" x2="10" y2="17"></line>
      <line x1="14" y1="11" x2="14" y2="17"></line>
    </svg>
  ),
};

export default function Onboarding() {
  const { user, hasRole } = useAuth();
  const isHr = hasRole("Admin") || hasRole("HR");
  const [mine, setMine] = useState<OnboardingTaskRow[]>([]);
  const [empList, setEmpList] = useState<EmpOpt[]>([]);
  const [selectedId, setSelectedId] = useState<number | "" | "all">("all");
  const [hrTasks, setHrTasks] = useState<OnboardingTaskRow[]>([]);
  const [newTitle, setNewTitle] = useState("");
  const [newPriority, setNewPriority] = useState("Medium");
  const [newDueDate, setNewDueDate] = useState("");
  const [loading, setLoading] = useState(true);
  const [hrLoading, setHrLoading] = useState(false);
  const [confirmDelete, setConfirmDelete] = useState<OnboardingTaskRow | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [showError, setShowError] = useState(false);

  useEffect(() => {
    setLoading(true);
    onboardingApi
      .mine()
      .then((r) => setMine(r.data))
      .catch(() => setMine([]))
      .finally(() => setLoading(false));
  }, []);

  useEffect(() => {
    if (!isHr) return;
    employeesApi
      .list({ status: "Active" })
      .then((r) => {
        const rows = (r.data as { id: number; full_name: string; employee_code: string }[]) || [];
        setEmpList(rows.map((e) => ({ id: e.id, full_name: e.full_name, employee_code: e.employee_code })));
      })
      .catch(() => setEmpList([]));
  }, [isHr]);

  useEffect(() => {
    if (!isHr || selectedId === "") {
      setHrTasks([]);
      return;
    }
    setHrLoading(true);
    const fetchTasks = selectedId === "all"
      ? onboardingApi.listAll()
      : onboardingApi.forEmployee(selectedId as number);

    fetchTasks
      .then((r) => setHrTasks(r.data))
      .catch(() => setHrTasks([]))
      .finally(() => setHrLoading(false));
  }, [isHr, selectedId]);

  const toggleMine = async (task: OnboardingTaskRow) => {
    try {
      const res = await onboardingApi.updateTask(task.id, { is_completed: !task.is_completed });
      const updated = res.data as OnboardingTaskRow;
      setMine(prev => prev.map(t => t.id === updated.id ? updated : t));
      // Also update hrTasks if the same task is visible there
      setHrTasks(prev => prev.map(t => t.id === updated.id ? updated : t));
    } catch {
      /* ignore */
    }
  };

  const addHrTask = async () => {
    if (!newTitle.trim()) {
      setShowError(true);
      return;
    }
    if (selectedId === "") return;
    setShowError(false);
    setSubmitting(true);
    try {
      const res = await onboardingApi.createTask({
        employee_id: selectedId as number,
        title: newTitle.trim(),
        priority: newPriority,
        due_date: newDueDate || undefined
      });
      setNewTitle("");
      setNewPriority("Medium");
      setNewDueDate("");

      const newTask = res.data as OnboardingTaskRow;
      setHrTasks(prev => [newTask, ...prev]);

      // If it belongs to me, update mine too
      if (newTask.employee_id === user?.employee_id) {
        setMine(prev => [newTask, ...prev]);
      }
    } catch {
      /* ignore */
    } finally {
      setSubmitting(false);
    }
  };

  const toggleHr = async (task: OnboardingTaskRow) => {
    try {
      const res = await onboardingApi.updateTask(task.id, { is_completed: !task.is_completed });
      const updated = res.data as OnboardingTaskRow;
      setHrTasks(prev => prev.map(t => t.id === updated.id ? updated : t));
      // Also update mine if it's there
      setMine(prev => prev.map(t => t.id === updated.id ? updated : t));
    } catch {
      /* ignore */
    }
  };

  const removeHr = (task: OnboardingTaskRow) => {
    setConfirmDelete(task);
  };

  const confirmActualDelete = async () => {
    if (!confirmDelete) return;
    setSubmitting(true);
    try {
      await onboardingApi.deleteTask(confirmDelete.id);
      setHrTasks(prev => prev.filter(t => t.id !== confirmDelete.id));
      setMine(prev => prev.filter(t => t.id !== confirmDelete.id));
      setConfirmDelete(null);
    } catch (err: any) {
      alert(err?.response?.data?.detail || "Failed to delete task.");
    } finally {
      setSubmitting(false);
    }
  };

  const done = mine.filter((t) => t.is_completed).length;
  const total = mine.length;


  return (
    <>
      <div className="content-shell--fade-in">
        <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <h1 className="page-title">Task Hub</h1>
            <div className="page-subtitle">Manage and track your tasks</div>
          </div>
          <GlobalHeaderControls />
        </div>


        {isHr && (
          <div className="card" style={{ marginBottom: "1.5rem", border: "1px solid rgb(var(--brand-rgb) / 0.25)", position: "relative", padding: "1.5rem 2rem" }}>
            <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: "1.5rem" }} >
              <h5 style={{ margin: "0", color: "#e6e7e8ff", fontSize: "0.9rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em" }}>Assigned Tasks</h5>
              <div style={{ display: "flex", alignItems: "center", gap: "0.75rem", background: "rgba(255, 255, 255, 0.02)" }}>
                <CustomSelect
                  style={{ width: "220px" }}
                  value={String(selectedId)}
                  disabled={submitting}
                  onChange={(val) => {
                    if (val === "all") setSelectedId("all");
                    else if (val === "") setSelectedId("");
                    else setSelectedId(Number(val));
                  }}
                  options={[
                    { value: "", label: "Select..." },
                    { value: "all", label: "All Employees" },
                    ...[...empList].sort((a, b) => (Number(a.employee_code) || 0) - (Number(b.employee_code) || 0)).map((e) => ({
                      value: String(e.id),
                      label: `${e.employee_code} — ${e.full_name}`
                    }))
                  ]}
                />
              </div>
            </div>
            <div >

              {selectedId !== "" && (
                <div>
                  {selectedId !== "all" && (
                    <div style={{ marginBottom: "2.5rem", background: "rgba(255,255,255,0.02)", padding: "1.5rem", borderRadius: "16px", border: "1px solid rgba(255,255,255,0.05)" }}>
                      <h5 style={{ margin: "0 0 1rem 0", color: "white", fontSize: "0.9rem", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em" }}>Create a new task</h5>
                      <div style={{ display: "flex", gap: "0.75rem", flexWrap: "wrap" }}>
                        <input
                          className="input"
                          style={{
                            flex: 3,
                            minWidth: "280px",
                            background: "rgba(255,255,255,0.03)",
                            border: showError ? "1px solid #ef4444" : "1px solid rgba(255,255,255,0.1)",
                            borderRadius: "10px",
                            boxShadow: showError ? "0 0 0 2px rgba(239, 68, 68, 0.2)" : "none"
                          }}
                          placeholder="Describe a new task"
                          value={newTitle}
                          onChange={(e) => {
                            setNewTitle(e.target.value);
                            if (e.target.value.trim()) setShowError(false);
                          }}
                          disabled={submitting}
                        />
                        <CustomSelect
                          style={{ flex: 1, minWidth: "120px" }}
                          value={newPriority}
                          onChange={(val) => setNewPriority(val)}
                          disabled={submitting}
                          options={[
                            { value: "Low", label: "Low" },
                            { value: "Medium", label: "Medium" },
                            { value: "High", label: "High" }
                          ]}
                        />
                        <input
                          type="date"
                          className="input"
                          style={{ flex: 1.5, minWidth: "160px", background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "10px" }}
                          value={newDueDate}
                          onChange={(e) => setNewDueDate(e.target.value)}
                          disabled={submitting}
                        />
                        <button
                          type="button"
                          className="btn btn-primary"
                          onClick={addHrTask}
                          disabled={submitting}
                          style={{ minWidth: "160px", height: "46px", borderRadius: "10px", fontWeight: 700 }}
                          title="Assign this New Task to the Selected Employee"
                        >
                          {submitting ? "Assigning..." : "Assign Task"}
                        </button>
                      </div>
                    </div>
                  )}

                  {hrLoading ? (
                    <SectionLoader rows={4} />
                  ) : hrTasks.length === 0 ? (
                    <div style={{ textAlign: "center", padding: "4rem 2rem", border: "1px solid  rgba(255,255,255,0.08)", borderRadius: "16px" }}>
                      <p className="text-muted" style={{ fontSize: "1rem" }}>No tasks assigned {selectedId === "all" ? "in the organization" : "to this employee"} yet.</p>
                    </div>
                  ) : (
                    <div style={{ display: "grid", gap: "0.75rem" }}>
                      {hrTasks.map((t) => (
                        <div
                          key={t.id}
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: "1.25rem",
                            padding: "1.25rem 1.5rem",
                            background: "rgba(255,255,255,0.03)",
                            borderRadius: "14px",
                            border: "1px solid rgba(255,255,255,0.05)",
                            transition: "all 0.2s ease"
                          }}
                          className="checklist-item-hover"
                        >
                          <input
                            type="checkbox"
                            checked={t.is_completed}
                            onChange={() => toggleHr(t)}
                            disabled={submitting}
                            style={{ cursor: "pointer", accentColor: "#15731e" }}
                            title={t.is_completed ? "Mark as Incomplete" : "Mark as Completed"}
                          />
                          <div style={{ flex: 1, minWidth: 0 }}>
                            <div style={{
                              fontWeight: 600,
                              fontSize: "1.1rem",
                              color: t.is_completed ? "rgba(255,255,255,0.3)" : "rgba(255,255,255,0.95)",
                              textDecoration: t.is_completed ? "line-through" : "none"
                            }}>
                              {t.title}
                            </div>
                            <div style={{ fontSize: "0.85rem", color: "rgba(255,255,255,0.4)", marginTop: "4px" }}>
                              {hasRole("Admin") ? "Assigned by Admin" : "Assigned by HR"} • {selectedId === "all" && (
                                <span style={{ color: "#54A832", fontWeight: 700 }}>
                                  {empList.find(e => e.id === t.employee_id)?.full_name || `Emp #${t.employee_id}`} •{" "}
                                </span>
                              )}
                              {t.due_date ? `Due: ${new Date(t.due_date).toLocaleDateString("en-GB", { day: 'numeric', month: 'short', year: 'numeric' })}` : "No due date"}
                            </div>
                          </div>

                          <div style={{
                            padding: "6px 16px",
                            borderRadius: "999px",
                            fontSize: "0.75rem",
                            fontWeight: 800,
                            background: t.priority === "High" ? "rgba(239, 68, 68, 0.15)" : t.priority === "Medium" ? "rgba(34, 197, 94, 0.15)" : "rgba(21, 50, 115, 0.15)",
                            color: t.priority === "High" ? "#f87171" : t.priority === "Medium" ? "#4ade80" : "#153273",
                            border: `1px solid ${t.priority === "High" ? "rgba(239, 68, 68, 0.2)" : t.priority === "Medium" ? "rgba(34, 197, 94, 0.2)" : "rgba(21, 50, 115, 0.2)"}`,
                            minWidth: "90px",
                            textAlign: "center"
                          }}>
                            {t.priority.toUpperCase()}
                          </div>

                          <button
                            type="button"
                            className="btn btn-danger btn-icon"
                            onClick={() => removeHr(t)}
                            disabled={submitting}
                            title="Delete this Task Permanently"
                            style={{
                              width: "36px",
                              height: "36px",
                              borderRadius: "50%",
                              background: "rgba(239, 68, 68, 0.1)",
                              border: "1px solid rgba(239, 68, 68, 0.2)",
                              color: "#f87171",
                              padding: "8px",
                              display: "flex",
                              alignItems: "center",
                              justifyContent: "center"
                            }}
                          >
                            <Icons.Delete />
                          </button>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        )}

        {user?.employee_id != null && (
          <div style={{ marginTop: "1rem" }}>
            <div className="card" style={{ padding: "1.5rem" }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", marginBottom: "1.5rem" }}>
                <div>
                  <h3 style={{ margin: 0, fontSize: "1.25rem", fontWeight: 800 }}>My Tasks</h3>
                  <div className="text-muted" style={{ fontSize: "0.9rem", marginTop: "4px" }}>
                    Your progress: <strong>{done}</strong> of <strong>{total}</strong> tasks completed
                  </div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div style={{ fontSize: "1.25rem", fontWeight: 900, color: "#54A832" }}>
                    {total > 0 ? Math.round((done / total) * 100) : 0}%
                  </div>
                </div>
              </div>

              <div style={{ height: "8px", background: "rgba(255,255,255,0.06)", borderRadius: "4px", marginBottom: "2rem", overflow: "hidden" }}>
                <div
                  style={{
                    height: "100%",
                    width: `${total > 0 ? (done / total) * 100 : 0}%`,
                    background: "#54A832",
                    boxShadow: "0 0 12px rgba(84, 168, 50, 0.4)",
                    transition: "width 0.4s ease"
                  }}
                />
              </div>

              {loading ? (
                <div style={{ padding: "3rem 0" }}><SectionLoader size="md" /></div>
              ) : mine.length === 0 ? (
                <div style={{ textAlign: "center", padding: "3rem", background: "rgba(255,255,255,0.02)", borderRadius: "12px", border: "1px solid rgba(255,255,255,0.1)" }}>
                  <p className="text-muted">You have no tasks assigned yet.</p>
                </div>
              ) : (
                <div style={{ display: "grid", gap: "0.75rem" }}>
                  {mine.map((t) => (
                    <div
                      key={t.id}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: "1.25rem",
                        padding: "1.15rem 1.25rem",
                        background: "rgba(255,255,255,0.03)",
                        borderRadius: "12px",
                        border: "1px solid rgba(255,255,255,0.05)",
                        transition: "all 0.2s ease",
                        opacity: t.is_completed ? 0.7 : 1
                      }}
                      className="checklist-item-hover"
                    >
                      <input
                        type="checkbox"
                        checked={t.is_completed}
                        onChange={() => toggleMine(t)}
                        disabled={submitting}
                        style={{ cursor: "pointer", accentColor: "#15731e" }}
                        title={t.is_completed ? "Mark as Incomplete" : "Mark as Completed"}
                      />
                      <div style={{ flex: 1, minWidth: 0 }}>
                        <div style={{
                          fontWeight: 700,
                          fontSize: "1.05rem",
                          color: t.is_completed ? "rgba(255,255,255,0.35)" : "rgba(255,255,255,0.95)",
                          textDecoration: t.is_completed ? "line-through" : "none"
                        }}>
                          {t.title}
                        </div>
                        {t.due_date && (
                          <div style={{ fontSize: "0.8rem", color: "rgba(255,255,255,0.45)", marginTop: "2px" }}>
                            Due: {new Date(t.due_date).toLocaleDateString("en-GB", { day: 'numeric', month: 'short', year: 'numeric' })}
                          </div>
                        )}
                      </div>
                      <div style={{
                        padding: "4px 12px",
                        borderRadius: "999px",
                        fontSize: "0.7rem",
                        fontWeight: 800,
                        background: t.priority === "High" ? "rgba(239, 68, 68, 0.1)" : t.priority === "Medium" ? "rgba(34, 197, 94, 0.1)" : "rgba(21, 50, 115, 0.1)",
                        color: t.priority === "High" ? "#f87171" : t.priority === "Medium" ? "#4ade80" : "#153273",
                        border: "1px solid currentColor"
                      }}>
                        {t.priority}
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {user?.employee_id == null && !isHr && (
          <div className="card" style={{ padding: "3rem", textAlign: "center" }}>
            <p className="text-muted" style={{ fontWeight: 700, textTransform: "uppercase", fontSize: "0.85rem", letterSpacing: "0.1em" }}>Account Pending Linkage</p>
            <p style={{ fontSize: "1.1rem", maxWidth: "500px", margin: "1rem auto", lineHeight: 1.6 }}>Please contact HR to link your user account to an employee record so you can begin your onboarding journey.</p>
          </div>
        )}
      </div>

      <style>{`
        .checklist-item-hover:hover {
          background: rgba(255, 255, 255, 0.08) !important;
          border-color: rgba(255, 255, 255, 0.2) !important;
          transform: scale(1.005) translateX(4px);
        }
        .content-shell--fade-in {
          animation: onboardingFadeIn 0.6s cubic-bezier(0.22, 1, 0.36, 1) forwards;
        }
        @keyframes onboardingFadeIn {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
      `}</style>

      <ConfirmModal
        isOpen={!!confirmDelete}
        onClose={() => setConfirmDelete(null)}
        onConfirm={confirmActualDelete}
        isLoading={submitting}
        title="Are you absolutely sure?"
        message={
          confirmDelete ? (
            <>
              You are about to remove the onboarding task: <strong>{confirmDelete.title}</strong>.
            </>
          ) : (
            ""
          )
        }
        confirmText="Yes, Remove Task"
      />
    </>
  );
}
