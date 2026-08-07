import { useEffect, useState } from "react";
import { employees as employeesApi, onboarding as onboardingApi, type OnboardingTaskRow } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import ConfirmModal from "../components/ConfirmModal";
import { SectionLoader } from "../components/LoadingState";
import CustomSelect from "../components/CustomSelect";
import GlobalHeaderControls from "../components/GlobalHeaderControls";

type EmpOpt = { id: number; full_name: string; employee_code: string };

// Premium SVG Icons for Actions
/* 24-box strokes, round caps, currentColor. Size comes from the control that
   holds them (.eds-chip 16px, .eds-iconbtn 15px, .eds-empty-tile 20px). */
const Icons = {
  Delete: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 21 6"></polyline>
      <path d="M8 6V4h8v2"></path>
      <path d="M6 6l1 14h10l1-14"></path>
    </svg>
  ),
  Tasks: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="9 11 12 14 20 6"></polyline>
      <path d="M20 12v7a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h9"></path>
    </svg>
  ),
  TaskDone: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 11l3 3 8-8"></path>
      <path d="M20 12v7a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2h9"></path>
    </svg>
  ),
  List: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <rect x="4" y="4" width="16" height="16" rx="3"></rect>
      <line x1="8" y1="10" x2="16" y2="10"></line>
      <line x1="8" y1="14" x2="13" y2="14"></line>
    </svg>
  ),
  Clock: () => (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="9"></circle>
      <polyline points="12 7 12 12 15 14"></polyline>
    </svg>
  ),
};

/** High is urgent (rose), Medium is in-flight (amber), Low is informational. */
function priorityTone(priority: string): string {
  if (priority === "High") return " eds-type--rose";
  if (priority === "Medium") return " eds-type--amber";
  return " eds-type--sky";
}

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
    <div className="eds">
      <header className="eds-topbar">
        <div>
          <h1 className="eds-title">Task Hub</h1>
          <p className="eds-subtitle">Manage and track your tasks</p>
        </div>
        <GlobalHeaderControls />
      </header>

      <div className="eds-page">
        {isHr && (
          <section className="eds-card">
            <div className="eds-card-head">
              <span className="eds-chip eds-chip--sky"><Icons.Tasks /></span>
              <div className="eds-card-titles">
                <h2 className="eds-card-title">Assigned Tasks</h2>
              </div>
              <div style={{ marginLeft: "auto", flexShrink: 0 }}>
                <CustomSelect
                  className="eds-cselect eds-cselect--wide"
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
            <div className="eds-card-body">

              {selectedId !== "" && (
                <div>
                  {selectedId !== "all" && (
                    <div style={{ marginBottom: "1.25rem", background: "rgba(255,255,255,0.03)", padding: "16px", borderRadius: "12px", border: "1px solid var(--eds-border)" }}>
                      <div className="eds-eyebrow" style={{ marginBottom: "0.75rem" }}>Create a new task</div>
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
                    <div className="eds-empty--card">
                      <span className="eds-empty-tile"><Icons.List /></span>
                      <span>No tasks assigned {selectedId === "all" ? "in the organization" : "to this employee"} yet.</span>
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

                          <span className={`eds-type${priorityTone(t.priority)}`} style={{ justifyContent: "center", minWidth: "82px" }}>
                            {t.priority.toUpperCase()}
                          </span>

                          <button
                            type="button"
                            className="eds-iconbtn eds-iconbtn--del"
                            onClick={() => removeHr(t)}
                            disabled={submitting}
                            title="Delete this Task Permanently"
                            style={{
                              width: "36px",
                              height: "36px",
                              borderRadius: "8px",
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
          </section>
        )}

        {user?.employee_id != null && (
          <section className="eds-card">
            <div className="eds-card-head">
              <span className="eds-chip eds-chip--emerald"><Icons.TaskDone /></span>
              <div className="eds-card-titles">
                <h2 className="eds-card-title">My Tasks</h2>
                <p className="eds-card-sub">
                  Your progress: <b style={{ color: "var(--eds-text)", fontWeight: 600 }}>{done}</b> of <b style={{ color: "var(--eds-text)", fontWeight: 600 }}>{total}</b> tasks completed
                </p>
              </div>
              <span className="eds-bigpct">{total > 0 ? Math.round((done / total) * 100) : 0}%</span>
            </div>

            <div className="eds-progress">
              <div style={{ width: `${total > 0 ? (done / total) * 100 : 0}%` }} />
            </div>

            <div className="eds-card-body">
              {loading ? (
                <div style={{ padding: "3rem 0" }}><SectionLoader size="md" /></div>
              ) : mine.length === 0 ? (
                <div className="eds-empty--card">
                  <span className="eds-empty-tile"><Icons.Clock /></span>
                  <span>You have no tasks assigned yet.</span>
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
                          fontWeight: 600,
                          fontSize: "14px",
                          color: t.is_completed ? "var(--eds-dim)" : "var(--eds-text)",
                          textDecoration: t.is_completed ? "line-through" : "none"
                        }}>
                          {t.title}
                        </div>
                        {t.due_date && (
                          <div style={{ fontSize: "12px", color: "var(--eds-dim)", marginTop: "2px" }}>
                            Due: {new Date(t.due_date).toLocaleDateString("en-GB", { day: 'numeric', month: 'short', year: 'numeric' })}
                          </div>
                        )}
                      </div>
                      <span className={`eds-type${priorityTone(t.priority)}`}>
                        {t.priority}
                      </span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </section>
        )}

        {user?.employee_id == null && !isHr && (
          <section className="eds-card">
            <div className="eds-empty--card">
              <span className="eds-empty-tile"><Icons.Clock /></span>
              <span className="eds-empty-title">Account Pending Linkage</span>
              <span>Please contact HR to link your user account to an employee record so you can begin your onboarding journey.</span>
            </div>
          </section>
        )}
      </div>

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
    </div>
  );
}
