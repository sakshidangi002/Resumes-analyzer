import { useState, useEffect, useMemo } from "react";
import { letters as api, employees as employeesApi } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import ConfirmModal from "../components/ConfirmModal";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import { SectionLoader } from "../components/LoadingState";
import { formatDate } from "../utils/dateFormatter";
import CustomSelect from "../components/CustomSelect";
import { useTableControls, SortableHeader, TableToolbar } from "../components/dataTable";

interface LetterTemplate {
  id: number;
  code: string;
  name: string;
}

interface LetterInstance {
  id: number;
  employee_id: number;
  generated_at: string;
  subject: string | null;
  sent_via_email: boolean;
  employee_code?: string | null;
  employee_name?: string | null;
  employee_official_email?: string | null;
  employee_personal_email?: string | null;
}

interface LetterReply {
  id: number;
  letter_instance_id: number;
  author_employee_id: number | null;
  author_user_id: number | null;
  message: string;
  created_at: string;
}

// Premium SVG Icons for Actions
const Icons = {
  View: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7z"></path>
      <circle cx="12" cy="12" r="3"></circle>
    </svg>
  ),
  PDF: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
      <polyline points="7 10 12 15 17 10"></polyline>
      <line x1="12" y1="15" x2="12" y2="3"></line>
    </svg>
  ),
  Reply: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <polyline points="9 17 4 12 9 7"></polyline>
      <path d="M20 18v-2a4 4 0 0 0-4-4H4"></path>
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
  Email: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path>
      <polyline points="22,6 12,13 2,6"></polyline>
    </svg>
  ),
};

export default function Letters({ forceEmployeeView = false }: { forceEmployeeView?: boolean }) {
  const { user, hasRole } = useAuth();
  const canManageLetters = (hasRole("Admin") || hasRole("HR")) && !forceEmployeeView;
  const [templates, setTemplates] = useState<LetterTemplate[]>([]);
  const [instances, setInstances] = useState<LetterInstance[]>([]);
  const [employees, setEmployees] = useState<
    { id: number; employee_code: string; first_name: string; last_name: string }[]
  >([]);
  const [selectedTemplate, setSelectedTemplate] = useState<string>("");
  const [selectedEmployeeId, setSelectedEmployeeId] = useState<number | "">("");
  const [subject, setSubject] = useState("");
  const [simpleBody, setSimpleBody] = useState("");
  const [sendEmail, setSendEmail] = useState(false);
  const [emailTarget, setEmailTarget] = useState<"official" | "personal" | "both">("official");
  const [fromEmail, setFromEmail] = useState("");
  const [loading, setLoading] = useState(true);
  const [generating, setGenerating] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [confirmDeleteId, setConfirmDeleteId] = useState<number | null>(null);
  const [sortOrder, setSortOrder] = useState<"newest" | "earliest">("newest");
  const [emailModalFor, setEmailModalFor] = useState<LetterInstance | null>(null);
  const [emailModalTarget, setEmailModalTarget] = useState<"official" | "personal" | "both">("official");
  const [emailModalFromEmail, setEmailModalFromEmail] = useState("");
  const [emailModalSending, setEmailModalSending] = useState(false);

  useEffect(() => {
    setLoading(true);
    if (canManageLetters) {
      api
        .templates()
        .then((r: { data: LetterTemplate[] }) => {
          setTemplates(r.data);
          if (!selectedTemplate && r.data.length > 0) {
            setSelectedTemplate(r.data[0].code);
          }
        })
        .catch(() => setTemplates([]));
    } else {
      setTemplates([]);
    }
    api
      .instances()
      .then((r: { data: LetterInstance[] }) => {
        setInstances(r.data);
      })
      .catch(() => setInstances([]));
    if (canManageLetters) {
      employeesApi
        .list()
        .then((r) => {
          const rows = r.data as {
            id: number;
            employee_code: string;
            first_name: string;
            last_name: string;
          }[];
          setEmployees(rows);
          if (!selectedEmployeeId && user?.employee_id) {
            setSelectedEmployeeId(user.employee_id);
          }
        })
        .catch(() => setEmployees([]))
        .finally(() => setLoading(false));
    } else {
      // Employee view: no employee picker; show only own letters
      setEmployees([]);
      setSelectedEmployeeId(user?.employee_id ?? "");
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [user?.employee_id, canManageLetters]);

  const sortedInstances = useMemo(() => {
    return [...instances].sort((a, b) => {
      const timeA = new Date(a.generated_at).getTime();
      const timeB = new Date(b.generated_at).getTime();
      return sortOrder === "newest" ? timeB - timeA : timeA - timeB;
    });
  }, [instances, sortOrder]);

  const {
    displayed: displayedInstances,
    search: letterSearch,
    setSearch: setLetterSearch,
    sort: letterSort,
    toggleSort: toggleLetterSort,
    clearAll: clearLetterControls,
    hasActiveControls: letterHasActive,
  } = useTableControls<LetterInstance>({
    rows: sortedInstances,
    columns: {
      generated_at: (i) => i.generated_at,
      employee: (i) => `${i.employee_code ?? ""} ${i.employee_name ?? ""}`.trim(),
      email: (i) => i.employee_official_email || i.employee_personal_email || "",
      subject: (i) => i.subject || "",
      sent_via_email: (i) => (i.sent_via_email ? "Yes" : "No"),
    },
    searchableText: (i) =>
      [
        i.employee_code,
        i.employee_name,
        i.employee_official_email,
        i.employee_personal_email,
        i.subject,
        i.sent_via_email ? "sent" : "not sent",
      ]
        .filter(Boolean)
        .join(" "),
  });

  const ownInstances = useMemo(
    () => displayedInstances.filter((i) => user?.employee_id && i.employee_id === user.employee_id),
    [displayedInstances, user?.employee_id]
  );


  const htmlToSimpleText = (html: string) => {
    if (!html) return "";
    let s = html;
    s = s.replace(/\r\n/g, "\n");
    s = s.replace(/<\s*br\s*\/?\s*>/gi, "\n");
    s = s.replace(/<\s*\/p\s*>/gi, "\n");
    s = s.replace(/<\s*p[^>]*>/gi, "");
    s = s.replace(/<\s*\/?strong\s*>/gi, "");
    s = s.replace(/<\s*\/?em\s*>/gi, "");
    // Strip any remaining tags
    s = s.replace(/<[^>]+>/g, "");
    // Decode a few common entities
    s = s.replace(/&nbsp;/g, " ");
    s = s.replace(/&amp;/g, "&");
    s = s.replace(/&lt;/g, "<");
    s = s.replace(/&gt;/g, ">");
    // Clean up extra blank lines
    s = s.replace(/\n{3,}/g, "\n\n").trim();
    return s;
  };

  const escapeHtml = (text: string) =>
    text
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");

  const simpleTextToHtml = (text: string) => {
    const t = (text || "").replace(/\r\n/g, "\n").trim();
    if (!t) return "";
    const paras = t.split(/\n\s*\n/g);
    return paras
      .map((p) => `<p>${escapeHtml(p).replace(/\n/g, "<br/>")}</p>`)
      .join("");
  };

  useEffect(() => {
    // Auto-generate content when both template and employee are selected
    if (!canManageLetters) return;
    if (!selectedTemplate || !selectedEmployeeId) {
      setSubject("");
      setSimpleBody("");
      return;
    }
    api
      .preview(selectedTemplate, Number(selectedEmployeeId))
      .then((r: { data: { subject: string; body: string } }) => {
        setSubject(r.data.subject || "");
        setSimpleBody(htmlToSimpleText(r.data.body || ""));
      })
      .catch(() => {
        setSubject("");
        setSimpleBody("");
      });
  }, [selectedTemplate, selectedEmployeeId]);

  const handleGenerate = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setSuccess("");
    if (!selectedTemplate || !selectedEmployeeId) {
      setError("Please choose both template and employee.");
      return;
    }
    const finalBody = simpleTextToHtml(simpleBody);
    const finalSubject = subject;
    if (!finalBody.trim()) {
      setError("Letter content cannot be empty.");
      return;
    }
    try {
      setGenerating(true);
      const { data } = await api.generate(
        selectedTemplate,
        Number(selectedEmployeeId),
        sendEmail,
        { subject: finalSubject, body: finalBody, from_email: fromEmail || undefined },
        emailTarget
      );
      const newInstance: LetterInstance = {
        id: data.letter_instance_id,
        employee_id: Number(selectedEmployeeId),
        generated_at: new Date().toISOString(),
        subject: data.subject ?? finalSubject,
        sent_via_email: !!data.sent_email,
      };
      setInstances((prev) => [...prev, newInstance]);
      setSuccess("Letter generated successfully.");
    } catch (err: unknown) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data
          ?.detail || "Failed to generate letter.";
      setError(msg);
    } finally {
      setGenerating(false);
    }
  };

  const handleView = async (id: number) => {
    try {
      const { data } = await api.letterBody(id);
      const win = window.open("", "_blank");
      if (win) {
        win.document.open();
        win.document.write(data as string);
        win.document.close();
      }
    } catch (err) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail || "Failed to load letter.";
      setError(msg);
    }
  };

  const handleDownloadPDF = async (id: number) => {
    try {
      const { data } = await api.letterBody(id);
      const win = window.open("", "_blank");
      if (win) {
        win.document.open();
        win.document.write(`
          <!DOCTYPE html>
          <html>
            <head>
              <title>Letter Download</title>
              <style>
                body { 
                  font-family: 'Inter', system-ui, sans-serif; 
                  padding: 40px; 
                  line-height: 1.6; 
                  color: #000; 
                  background: #fff;
                }
                @media print {
                  @page {
                    margin: 0; /* Hides browser headers/footers */
                  }
                  body { 
                    padding: 0;
                    margin: 1.5cm 2cm; /* Real content margins */
                    padding-top: 5cm;   /* Space for letterhead */
                  }
                }
              </style>
            </head>
            <body>
              ${data}
              <script>
                window.onload = function() {
                  window.print();
                  setTimeout(() => window.close(), 100);
                }
              </script>
            </body>
          </html>
        `);
        win.document.close();
      }
    } catch (err) {
      const msg = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail || "Failed to download PDF.";
      setError(msg);
    }
  };

  const [replyForId, setReplyForId] = useState<number | null>(null);
  const [replies, setReplies] = useState<LetterReply[]>([]);
  const [newReply, setNewReply] = useState("");
  const [replyLoading, setReplyLoading] = useState(false);

  const openReplies = async (instanceId: number) => {
    setReplyForId(instanceId);
    setNewReply("");
    setReplyLoading(true);
    setError("");
    try {
      const { data } = await api.replies(instanceId);
      setReplies(data as LetterReply[]);
    } catch (err) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data
          ?.detail || "Failed to load replies.";
      setError(msg);
      setReplies([]);
    } finally {
      setReplyLoading(false);
    }
  };

  const submitReply = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!replyForId) return;
    if (!newReply.trim()) return;
    setReplyLoading(true);
    setError("");
    try {
      const { data } = await api.addReply(replyForId, newReply.trim());
      setReplies((prev) => [...prev, data as LetterReply]);
      setNewReply("");
    } catch (err) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data
          ?.detail || "Failed to send reply.";
      setError(msg);
    } finally {
      setReplyLoading(false);
    }
  };

  const handleDelete = (id: number) => {
    setConfirmDeleteId(id);
  };

  const openEmailModal = (instance: LetterInstance) => {
    setEmailModalFor(instance);
    if (instance.employee_official_email) {
      setEmailModalTarget("official");
    } else if (instance.employee_personal_email) {
      setEmailModalTarget("personal");
    } else {
      setEmailModalTarget("official");
    }
    setEmailModalFromEmail("");
    setError("");
  };

  const submitEmailInstance = async () => {
    if (!emailModalFor) return;
    setEmailModalSending(true);
    setError("");
    setSuccess("");
    try {
      const { data } = await api.emailInstance(
        emailModalFor.id,
        emailModalTarget,
        emailModalFromEmail.trim() || undefined
      );
      const sentTo: string[] = (data as { sent_to?: string[] })?.sent_to || [];
      setInstances((prev) =>
        prev.map((i) =>
          i.id === emailModalFor.id ? { ...i, sent_via_email: true } : i
        )
      );
      setSuccess(
        sentTo.length
          ? `Email sent with PDF attachment to ${sentTo.join(", ")}.`
          : "Email sent."
      );
      setEmailModalFor(null);
    } catch (err) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data
          ?.detail || "Failed to send email.";
      setError(msg);
    } finally {
      setEmailModalSending(false);
    }
  };

  const confirmActualDelete = async () => {
    if (!confirmDeleteId) return;
    setError("");
    try {
      await api.deleteInstance(confirmDeleteId);
      setInstances((prev) => prev.filter((i) => i.id !== confirmDeleteId));
      if (replyForId === confirmDeleteId) {
        setReplyForId(null);
        setReplies([]);
      }
      setSuccess("Letter deleted.");
    } catch (err) {
      const msg =
        (err as { response?: { data?: { detail?: string } } })?.response?.data
          ?.detail || "Failed to delete letter.";
      setError(msg);
    } finally {
      setConfirmDeleteId(null);
    }
  };


  return (
    <>
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div style={{ flex: 1 }}>
          <h1 className="page-title">{canManageLetters ? "Employee Letters" : "My Documents"}</h1>
          <div className="page-subtitle">
            {canManageLetters ? "View and manage all letters generated for employees." : "View your generated letters"}
          </div>
        </div>
        <GlobalHeaderControls />
      </div>
      {error && <div className="alert alert-error">{error}</div>}
      {success && <div className="alert alert-success">{success}</div>}
      {canManageLetters && (
        <>
          <div className="card" style={{ marginBottom: '1.5rem' }}>
            <div className="flex items-center justify-between mb-6" style={{ gap: "1rem" }}>
              <h3 style={{ margin: 0, fontSize: "1.25rem", color: "#fff", whiteSpace: "nowrap" }}>Generate Document</h3>
            </div>

            <form onSubmit={handleGenerate}>
              <div
                style={{
                  display: "grid",
                  gridTemplateColumns: "1fr 1fr 1fr auto", gap: "2rem", alignItems: "end", marginBottom: "1.5rem",
                }}
              >
                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label style={{ color: "rgba(255, 255, 255, 0.7)", fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>Select Template</label>
                  <CustomSelect
                    value={selectedTemplate}
                    onChange={(val) => setSelectedTemplate(val)}
                    options={templates.map((t: LetterTemplate) => ({
                      value: t.code,
                      label: `${t.code} – ${t.name}`
                    }))}
                  />
                </div>

                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label style={{ color: "rgba(255, 255, 255, 0.7)", fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>Target Employee</label>
                  <CustomSelect
                    value={String(selectedEmployeeId)}
                    onChange={(val) => setSelectedEmployeeId(val ? Number(val) : "")}
                    placeholder="Select employee"
                    options={[
                      { value: "", label: "Select employee" },
                      ...employees.map((emp) => ({
                        value: String(emp.id),
                        label: `${emp.employee_code} – ${emp.first_name} ${emp.last_name}`
                      }))
                    ]}
                  />
                </div>

                <div className="form-group" style={{ marginBottom: 0 }}>
                  <label style={{ color: "rgba(255, 255, 255, 0.7)", fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>Delivery Options</label>
                  <div style={{ display: "flex", alignItems: "center", gap: "1rem", height: "48px" }}>
                    <label style={{ display: "flex", alignItems: "center", gap: "0.5rem", cursor: "pointer", marginBottom: 0, whiteSpace: "nowrap" }}>
                      <input
                        type="checkbox"
                        checked={sendEmail}
                        onChange={(e) => setSendEmail(e.target.checked)}
                        style={{ width: "18px", height: "18px" }}
                      />
                      <span style={{ fontSize: "0.9rem" }}>Send Email</span>
                    </label>
                    {sendEmail && (
                      <CustomSelect
                        value={emailTarget}
                        style={{ height: "32px", fontSize: "0.85rem" }}
                        onChange={(val) => setEmailTarget(val as "official" | "personal" | "both")}
                        options={[
                          { value: "official", label: "Official Only" },
                          { value: "personal", label: "Personal Only" },
                          { value: "both", label: "Both Emails" }
                        ]}
                      />
                    )}
                  </div>
                </div>

                {sendEmail && (
                  <div className="form-group" style={{ gridColumn: "1 / -1", marginBottom: 0 }}>
                    <label style={{ color: "rgba(255, 255, 255, 0.7)", fontSize: "0.75rem", textTransform: "uppercase", letterSpacing: "0.05em" }}>Custom Sender (Optional)</label>
                    <input
                      type="email"
                      value={fromEmail}
                      onChange={(e) => setFromEmail(e.target.value)}
                      placeholder="e.g. hr@company.com (leave blank for system default)"
                    />
                  </div>
                )}
              </div>

              {selectedEmployeeId && (
                <>
                  <div className="form-group" style={{ marginBottom: "1.5rem" }}>
                    <label style={{ fontSize: "0.9rem", fontWeight: 600, color: "#fff" }}>Document Subject</label>
                    <input
                      type="text"
                      value={subject}
                      onChange={(e) => setSubject(e.target.value)}
                      placeholder="The subject will be populated based on the selected template"
                      style={{ fontSize: "1rem", fontWeight: 500 }}
                    />
                  </div>

                  <div style={{ marginBottom: "1.5rem" }}>
                    <div className="flex items-center justify-between mb-4">
                      <label style={{ margin: "3px", fontSize: "0.9rem", fontWeight: 600, color: "#fff" }}>Body Content</label>
                    </div>

                    <div
                      style={{
                        position: "relative",
                        border: "1px solid rgba(255, 255, 255, 0.12)",
                        borderRadius: "12px",
                        overflow: "hidden",
                        background: "rgba(0, 0, 0, 0.2)"
                      }}
                    >
                      <textarea
                        value={simpleBody}
                        onChange={(e) => setSimpleBody(e.target.value)}
                        rows={10}
                        style={{
                          width: "100%",
                          maxWidth: "100%",
                          border: "none",
                          background: "transparent",
                          padding: "1.5rem",
                          fontSize: "0.95rem",
                          resize: "vertical",
                          color: "#eee"
                        }}
                      />
                      <div
                        style={{
                          padding: "0.5rem 1rem",
                          background: "rgba(255, 255, 255, 0.04)",
                          borderTop: "1px solid rgba(255, 255, 255, 0.08)",
                          fontSize: "0.75rem",
                          color: "rgba(255, 255, 255, 0.4)"
                        }}
                      >
                        Smart formatting is enabled. Line breaks will be preserved.
                      </div>
                    </div>
                  </div>

                  <div style={{ marginBottom: "2rem" }}>
                    <div className="flex items-center gap-4 mb-6" style={{ color: "#fff" }}>
                      <Icons.View />
                      <span style={{ fontSize: "1.1rem", fontWeight: 700, letterSpacing: "0.02em" }}>Live Document Preview</span>
                    </div>
                    <div
                      className="preview-container"
                      style={{
                        marginTop: "1.5rem",
                        border: "1px solid rgba(255, 255, 255, 0.2)",
                        borderRadius: 16,
                        padding: "3rem",
                        background: "rgba(255, 255, 255, 0.04)",
                        color: "#fff",
                        minHeight: "250px",
                        boxShadow: "inset 0 4px 20px rgba(0,0,0,0.3)"
                      }}
                      dangerouslySetInnerHTML={{
                        __html: simpleTextToHtml(simpleBody),
                      }}
                    />
                  </div>

                  <div style={{ display: "flex", alignItems: "center", justifyContent: "end" }}>
                    <button type="submit" className="btn btn-primary"
                      style={{ padding: "0.75rem 2.5rem", fontSize: "1rem", fontWeight: 600, borderRadius: "10px" }}
                      disabled={generating || loading || templates.length === 0}
                      title="Generate and Finalize Letter"
                    >
                      {generating ? (
                        <span className="flex items-center gap-2">
                          <span className="spinner-small" /> Generating...
                        </span>
                      ) : "Generate & Finalize"}
                    </button>
                  </div>
                </>
              )}
            </form>
          </div>
        </>
      )}

      {canManageLetters ? (
        <>
          <div className="card" style={{ marginBottom: '1.5rem' }}>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "0.75rem", gap: "1rem", flexWrap: "wrap" }}>
              <h3 style={{ margin: 0 }}>Generated Documents History</h3>
            </div>
            <TableToolbar
              search={letterSearch}
              onSearchChange={setLetterSearch}
              placeholder="Search by employee, subject, email..."
              showClear={letterHasActive}
              onClear={clearLetterControls}
              count={{ shown: displayedInstances.length, total: sortedInstances.length }}
              leftControls={
                <div style={{ minWidth: "160px" }}>
                  <CustomSelect
                    value={sortOrder}
                    onChange={(val) => setSortOrder(val as any)}
                    options={[
                      { value: "newest", label: "Newest first" },
                      { value: "earliest", label: "Earliest first" }
                    ]}
                  />
                </div>
              }
            />
            {loading ? (
              <div style={{ padding: "3rem 0" }}><SectionLoader size="md" /></div>
            ) : sortedInstances.length === 0 ? (
              <p className="text-muted">No employee letters generated yet.</p>
            ) : (
              <div className="table-wrap table-wrap--dark">
                <table className="table-modern table-modern--dark" style={{ tableLayout: 'fixed', width: '100%' }}>
                  <thead>
                    <tr>
                      <SortableHeader className="hide-xl" label="Generated" columnKey="generated_at" sort={letterSort} onToggle={toggleLetterSort} align="center" style={{ width: '12%', textAlign: 'center' }} />
                      <SortableHeader label="Employee" columnKey="employee" sort={letterSort} onToggle={toggleLetterSort} align="center" style={{ width: '17%', textAlign: 'center' }} />
                      <SortableHeader className="hide-md" label="Email" columnKey="email" sort={letterSort} onToggle={toggleLetterSort} align="center" style={{ width: '20%', textAlign: 'center' }} />
                      <SortableHeader label="Subject" columnKey="subject" sort={letterSort} onToggle={toggleLetterSort} align="center" style={{ width: '18%', textAlign: 'center' }} />
                      <SortableHeader className="hide-lg" label="Email sent" columnKey="sent_via_email" sort={letterSort} onToggle={toggleLetterSort} align="center" style={{ width: '8%', textAlign: 'center' }} />
                      <SortableHeader label="Actions" columnKey="__actions" sort={letterSort} onToggle={toggleLetterSort} align="center" notSortable style={{ width: '25%', minWidth: 220, textAlign: 'center' }} />
                    </tr>
                  </thead>
                  <tbody>
                    {displayedInstances.length === 0 && (
                      <tr>
                        <td colSpan={6} style={{ textAlign: 'center', padding: '1.25rem', opacity: 0.65, color: '#fff' }}>
                          No letters match your search.
                        </td>
                      </tr>
                    )}
                    {displayedInstances.map((i: LetterInstance) => (
                      <tr key={i.id}>
                        <td className="hide-xl" style={{ textAlign: 'center' }}>
                          {formatDate(i.generated_at)} {new Date(i.generated_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </td>
                        <td style={{ textAlign: 'center' }}>
                          <div
                            className="text-truncate"
                            style={{ width: '100%' }}
                            title={
                              i.employee_code || i.employee_name
                                ? `${i.employee_code ? i.employee_code + " – " : ""}${i.employee_name ?? ""}`
                                : "-"
                            }
                          >
                            {i.employee_code || i.employee_name
                              ? `${i.employee_code ? i.employee_code + " – " : ""}${i.employee_name ?? ""}`
                              : "-"}
                          </div>
                        </td>
                        <td className="hide-md" style={{ textAlign: 'center' }}>
                          <div
                            className="text-truncate"
                            style={{ width: '100%' }}
                            title={i.employee_official_email || i.employee_personal_email || "-"}
                          >
                            {i.employee_official_email || i.employee_personal_email || "-"}
                          </div>
                        </td>
                        <td style={{ textAlign: 'center', fontWeight: 500 }}>
                          <div
                            className="text-truncate"
                            style={{ width: '100%' }}
                            title={i.subject || "-"}
                          >
                            {i.subject || "-"}
                          </div>
                        </td>
                        <td className="hide-lg" style={{ textAlign: 'center' }}>
                          {i.sent_via_email ? "✅" : "No"}
                        </td>
                        <td style={{ textAlign: 'center', whiteSpace: 'nowrap', minWidth: 220 }}>
                          <div style={{ display: 'inline-flex', gap: '6px', justifyContent: 'center', flexWrap: 'nowrap' }}>
                            <button type="button" className="btn btn-secondary btn-icon btn-sm" onClick={() => handleView(i.id)} title="View Letter Content">
                              <Icons.View />
                            </button>
                            <button type="button" className="btn btn-secondary btn-icon btn-sm" onClick={() => handleDownloadPDF(i.id)} title="Download Letter as PDF">
                              <Icons.PDF />
                            </button>
                            <button type="button" className="btn btn-secondary btn-icon btn-sm" onClick={() => openReplies(i.id)} title="Open Conversation/Replies">
                              <Icons.Reply />
                            </button>
                            <button
                              type="button"
                              className="btn btn-secondary btn-icon btn-sm"
                              onClick={() => openEmailModal(i)}
                              title={i.sent_via_email ? "Resend Letter by Email (PDF attached)" : "Send Letter by Email (PDF attached)"}
                              style={i.sent_via_email ? { borderColor: "#22c55e", color: "#22c55e" } : undefined}
                            >
                              <Icons.Email />
                            </button>
                            <button type="button" className="btn btn-danger btn-icon btn-sm" onClick={() => handleDelete(i.id)} title="Delete Letter Permanently">
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
          </div>
        </>
      ) : (
        <div className="card">
          <h3 style={{ marginTop: 0 }}>My Documents</h3>
          {instances.length === 0 ? (
            <p className="text-muted">No documents issued yet.</p>
          ) : (
            <div className="table-wrap table-wrap--dark">
              <table className="table-modern table-modern--dark" style={{ tableLayout: 'fixed', width: '100%' }}>
                <thead>
                  <tr>
                    <th className="hide-xl" style={{ width: '25%', textAlign: 'center' }}>Generated</th>
                    <th style={{ width: '35%', textAlign: 'center' }}>Subject</th>
                    <th className="hide-lg" style={{ width: '15%', textAlign: 'center' }}>Email received</th>
                    <th style={{ width: '25%', textAlign: 'center' }}>Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {ownInstances.map((i: LetterInstance) => (
                    <tr key={i.id}>
                      <td className="hide-xl" style={{ textAlign: 'center' }}>
                        {formatDate(i.generated_at)} {new Date(i.generated_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </td>
                      <td style={{ textAlign: 'center', fontWeight: 500 }}>{i.subject || "-"}</td>
                      <td className="hide-lg" style={{ textAlign: 'center' }}>{i.sent_via_email ? "✅" : "Not yet"}</td>
                      <td style={{ textAlign: 'center' }}>
                        <div style={{ display: 'inline-flex', gap: '8px', justifyContent: 'center' }}>
                          <button type="button" className="btn btn-secondary btn-icon btn-sm" onClick={() => handleView(i.id)} title="View Document">
                            <Icons.View />
                          </button>
                          <button type="button" className="btn btn-secondary btn-icon btn-sm" onClick={() => handleDownloadPDF(i.id)} title="Download as PDF">
                            <Icons.PDF />
                          </button>
                          <button type="button" className="btn btn-secondary btn-icon btn-sm" onClick={() => openReplies(i.id)} title="View Conversation">
                            <Icons.Reply />
                          </button>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )}

      {replyForId && (
        <div className="modal-backdrop" onClick={() => setReplyForId(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 720 }}>
            <h3 style={{ marginTop: 0 }}>Conversation for Letter #{replyForId}</h3>
            {replyLoading ? (
              <SectionLoader rows={3} compact />
            ) : (
              <>
                <div
                  style={{
                    maxHeight: "300px",
                    overflowY: "auto",
                    border: "1px solid rgba(var(--brand-rgb) / 0.25)",
                    borderRadius: 8,
                    padding: "0.75rem",
                    marginBottom: "1rem",
                    background: "rgba(255, 255, 255, 0.06)",
                  }}
                >
                  {replies.length === 0 && (
                    <p className="text-muted">No replies yet. You can send the first reply.</p>
                  )}
                  {replies.map((r) => (
                    <div key={r.id} style={{ marginBottom: "0.75rem" }}>
                      <div style={{ fontSize: "0.75rem", color: "rgba(255, 255, 255, 0.65)" }}>
                        {formatDate(r.created_at)} {new Date(r.created_at + (r.created_at.endsWith("Z") ? "" : "Z")).toLocaleTimeString("en-IN", { timeZone: "Asia/Kolkata", hour: "2-digit", minute: "2-digit" })}
                      </div>
                      <div>{r.message}</div>
                    </div>
                  ))}
                </div>
                <form onSubmit={submitReply}>
                  <div className="form-group">
                    <label>Your reply</label>
                    <textarea
                      rows={3}
                      value={newReply}
                      onChange={(e) => setNewReply(e.target.value)}
                      style={{ maxWidth: "100%" }}
                    />
                  </div>
                  <button
                    type="submit"
                    className="btn btn-primary"
                    disabled={replyLoading || !newReply.trim()}
                    title="Submit Reply"
                  >
                    Send reply
                  </button>
                  <button
                    type="button"
                    className="btn btn-cancel-alt"
                    style={{ marginLeft: "0.5rem" }}
                    onClick={() => setReplyForId(null)}
                    title="Close Dialog"
                  >
                    Close
                  </button>
                </form>
              </>
            )}
          </div>
        </div>
      )}

      <ConfirmModal
        isOpen={!!confirmDeleteId}
        onClose={() => setConfirmDeleteId(null)}
        onConfirm={confirmActualDelete}
        title="Are you absolutely sure?"
        message={
          <>
            You are about to delete letter <strong>#{confirmDeleteId}</strong>. This action cannot be undone.
          </>
        }
        confirmText="Yes, Delete Letter"
      />

      {emailModalFor && (
        <div
          style={{
            position: "fixed", inset: 0, background: "rgba(0,0,0,0.6)",
            display: "flex", alignItems: "center", justifyContent: "center", zIndex: 1000,
          }}
          onClick={() => !emailModalSending && setEmailModalFor(null)}
        >
          <div
            className="card"
            style={{ width: "min(520px, 92vw)", padding: "1.5rem" }}
            onClick={(e) => e.stopPropagation()}
          >
            <h3 style={{ marginTop: 0 }}>
              {emailModalFor.sent_via_email ? "Resend letter by email" : "Send letter by email"}
            </h3>
            <p className="text-muted" style={{ marginTop: 0 }}>
              {emailModalFor.employee_code ? `${emailModalFor.employee_code} – ` : ""}
              {emailModalFor.employee_name || "Employee"}
              <br />
              The letter will be sent with a PDF attachment.
            </p>
            <div className="form-group">
              <label>Send to</label>
              <CustomSelect
                value={emailModalTarget}
                onChange={(val) => setEmailModalTarget(val as "official" | "personal" | "both")}
                options={[
                  {
                    value: "official",
                    label: emailModalFor.employee_official_email
                      ? `Official (${emailModalFor.employee_official_email})`
                      : "Official (none on record)",
                  },
                  {
                    value: "personal",
                    label: emailModalFor.employee_personal_email
                      ? `Personal (${emailModalFor.employee_personal_email})`
                      : "Personal (none on record)",
                  },
                  { value: "both", label: "Both" },
                ]}
              />
            </div>
            <div className="form-group">
              <label>Reply-to (optional)</label>
              <input
                type="email"
                placeholder="hr@company.com"
                value={emailModalFromEmail}
                onChange={(e) => setEmailModalFromEmail(e.target.value)}
              />
            </div>
            <div style={{ display: "flex", gap: "0.75rem", justifyContent: "flex-end", marginTop: "1rem" }}>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={() => setEmailModalFor(null)}
                disabled={emailModalSending}
              >
                Cancel
              </button>
              <button
                type="button"
                className="btn btn-primary"
                onClick={submitEmailInstance}
                disabled={emailModalSending}
              >
                {emailModalSending ? "Sending..." : emailModalFor.sent_via_email ? "Resend" : "Send"}
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
