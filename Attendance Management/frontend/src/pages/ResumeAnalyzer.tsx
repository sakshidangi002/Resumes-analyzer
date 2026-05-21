import { useCallback, useEffect, useState } from "react";
import axios from "axios";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import ConfirmModal from "../components/ConfirmModal";
import { SectionLoader } from "../components/LoadingState";
import { useAuth } from "../auth/AuthContext";

// Premium inline SVG Icons
const Icons = {
  Search: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>
  ),
  Upload: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="17 8 12 3 7 8"></polyline><line x1="12" y1="3" x2="12" y2="15"></line></svg>
  ),
  Sparkles: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
  ),
  Chat: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path></svg>
  ),
  Delete: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>
  ),
  Star: ({ filled }: { filled?: boolean }) => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill={filled ? "var(--brand-400)" : "none"} stroke={filled ? "var(--brand-400)" : "currentColor"} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polygon points="12 2 15.09 8.26 22 9.27 17 14.14 18.18 21.02 12 17.77 5.82 21.02 7 14.14 2 9.27 8.91 8.26 12 2"></polygon></svg>
  ),
  FileText: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
  ),
  Close: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
  ),
  Send: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>
  ),
  Briefcase: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path></svg>
  ),
  Phone: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72 12.84 12.84 0 0 0 .7 2.81 2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45 12.84 12.84 0 0 0 2.81.7A2 2 0 0 1 22 16.92z"></path></svg>
  ),
  Mail: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M4 4h16c1.1 0 2 .9 2 2v12c0 1.1-.9 2-2 2H4c-1.1 0-2-.9-2-2V6c0-1.1.9-2 2-2z"></path><polyline points="22,6 12,13 2,6"></polyline></svg>
  )
};

// Create axios client for Resume Analyzer API.
// Unified server mounts the Resume API at /resume-api on the same origin.
const resumeApiUrl = `${window.location.origin}/resume-api`;
const resClient = axios.create({
  baseURL: resumeApiUrl,
  headers: { "Content-Type": "application/json" }
});

resClient.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

interface Resume {
  id: string;
  name: string;
  email: string;
  phone: string;
  location: string;
  skills: string[];
  primary_skills: string[];
  other_skills: string[];
  experience_years: number;
  total_experience_years: number;
  experience_level: string;
  experience_summary: string;
  summary: string;
  education: string[];
  projects: string[];
  resume_link: string;
  is_shortlisted: boolean;
  companies_worked_at: string[];
  role: string;
  created_at: string;
}

interface Note {
  id: string;
  note: string;
  status: string | null;
  created_at: string;
}

export default function ResumeAnalyzer() {
  const { hasRole, loading: authLoading } = useAuth();

  const [resumes, setResumes] = useState<Resume[]>([]);
  const [filteredResumes, setFilteredResumes] = useState<Resume[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchType, setSearchType] = useState<"text" | "semantic">("text");

  // Filtering States
  const [experienceFilter, setExperienceFilter] = useState<"all" | "junior" | "mid" | "senior">("all");
  const [shortlistFilter, setShortlistFilter] = useState<"all" | "shortlisted">("all");

  // Selection & Details Sidebar
  const [selectedResume, setSelectedResume] = useState<Resume | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [notes, setNotes] = useState<Note[]>([]);
  const [newNote, setNewNote] = useState("");
  const [notesLoading, setNotesLoading] = useState(false);

  // Chatbot Modal
  const [chatbotOpen, setChatbotOpen] = useState(false);
  const [chatResumeObj, setChatResumeObj] = useState<Resume | null>(null);
  const [chatMessages, setChatMessages] = useState<{ sender: "user" | "bot"; text: string }[]>([]);
  const [chatInput, setChatInput] = useState("");
  const [chatLoading, setChatLoading] = useState(false);

  // Compare Tab / Panel
  const [compareOpen, setCompareOpen] = useState(false);
  const [jobDescription, setJobDescription] = useState("");
  const [selectedForCompare, setSelectedForCompare] = useState<string[]>([]);
  const [compareResults, setCompareResults] = useState<any[]>([]);
  const [compareLoading, setCompareLoading] = useState(false);

  // File Upload
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState("");
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  // Load resumes
  const loadResumes = useCallback(async () => {
    if (!hasRole("Admin", "HR")) return; // Don't fetch if no access
    setLoading(true);
    try {
      const response = await resClient.get<Resume[]>("/resumes");
      setResumes(response.data);
      setFilteredResumes(response.data);
    } catch (err) {
      console.error("Failed to load resumes", err);
    } finally {
      setLoading(false);
    }
  }, [hasRole]);

  useEffect(() => {
    if (!authLoading) {
      loadResumes();
    }
  }, [loadResumes, authLoading]);

  // Access Control UI
  if (authLoading) return <SectionLoader text="Verifying Access..." />;
  if (!hasRole("Admin", "HR")) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[70vh] text-center px-4 space-y-4">
        <div className="p-4 bg-red-500/10 rounded-full border border-red-500/20 mb-2 shadow-lg shadow-red-500/10">
          <svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--red-400)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
            <path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10"/>
            <path d="m9 12 2 2 4-4"/>
          </svg>
        </div>
        <h2 className="text-2xl font-outfit font-bold text-white tracking-tight">Access Restricted</h2>
        <p className="text-slate-400 max-w-md text-sm leading-relaxed">
          The Resume Analyzer workspace contains sensitive candidate data and is restricted to Administrators and HR personnel. You do not have the required permissions to view this module.
        </p>
      </div>
    );
  }

  // Execute Local / Semantic Search
  const handleSearch = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!searchQuery.trim()) {
      setFilteredResumes(resumes);
      return;
    }

    setLoading(true);
    try {
      if (searchType === "semantic") {
        // Chat endpoint does semantic query
        const response = await resClient.post("/chat", { question: searchQuery });
        if (response.data.best_matches) {
          const matchedIds = new Set(response.data.best_matches.map((m: any) => m.id));
          const matchedList = resumes.filter((r) => matchedIds.has(r.id));
          setFilteredResumes(matchedList);
        } else {
          setFilteredResumes([]);
        }
      } else {
        // Standard textual matching
        const response = await resClient.get<Resume[]>("/resumes", {
          params: { skills: searchQuery }
        });
        setFilteredResumes(response.data);
      }
    } catch (err) {
      console.error("Search failed", err);
    } finally {
      setLoading(false);
    }
  };

  // Run filters whenever dependencies change
  useEffect(() => {
    let result = resumes;

    // Filter by Experience Level
    if (experienceFilter === "junior") {
      result = result.filter((r) => r.experience_years < 2);
    } else if (experienceFilter === "mid") {
      result = result.filter((r) => r.experience_years >= 2 && r.experience_years <= 5);
    } else if (experienceFilter === "senior") {
      result = result.filter((r) => r.experience_years > 5);
    }

    // Filter by Shortlisted
    if (shortlistFilter === "shortlisted") {
      result = result.filter((r) => r.is_shortlisted);
    }

    setFilteredResumes(result);
  }, [experienceFilter, shortlistFilter, resumes]);

  // Toggle Shortlist status
  const toggleShortlist = async (resume: Resume, e: React.MouseEvent) => {
    e.stopPropagation();
    const updatedStatus = !resume.is_shortlisted;
    try {
      if (updatedStatus) {
        await resClient.post(`/resumes/${resume.id}/shortlist`);
      } else {
        await resClient.delete(`/resumes/${resume.id}/shortlist`);
      }
      setResumes((prev) =>
        prev.map((r) => (r.id === resume.id ? { ...r, is_shortlisted: updatedStatus } : r))
      );
      if (selectedResume && selectedResume.id === resume.id) {
        setSelectedResume({ ...selectedResume, is_shortlisted: updatedStatus });
      }
    } catch (err) {
      console.error("Failed to toggle shortlist", err);
    }
  };

  // Delete Resume
  const handleDeleteResume = async () => {
    if (!confirmDelete) return;
    try {
      await resClient.delete("/resumes/bulk", {
        data: { resume_ids: [confirmDelete] }
      });
      setResumes((prev) => prev.filter((r) => r.id !== confirmDelete));
      if (selectedResume?.id === confirmDelete) {
        setSelectedResume(null);
        setSidebarOpen(false);
      }
      setConfirmDelete(null);
    } catch (err) {
      console.error("Failed to delete candidate", err);
    }
  };

  // Open Candidate Details
  const handleOpenDetails = async (resume: Resume) => {
    setSelectedResume(resume);
    setSidebarOpen(true);
    setNotesLoading(true);
    try {
      const response = await resClient.get<Note[]>(`/resumes/${resume.id}/notes`);
      setNotes(response.data);
    } catch (err) {
      console.error("Failed to load notes", err);
    } finally {
      setNotesLoading(false);
    }
  };

  // Add a Candidate Note
  const handleAddNote = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!selectedResume || !newNote.trim()) return;
    try {
      const response = await resClient.post(`/resumes/${selectedResume.id}/notes`, {
        note: newNote,
        status: "Note"
      });
      setNotes((prev) => [
        {
          id: response.data.id,
          note: newNote,
          status: "Note",
          created_at: new Date().toISOString()
        },
        ...prev
      ]);
      setNewNote("");
    } catch (err) {
      console.error("Failed to add note", err);
    }
  };

  // Delete a Candidate Note
  const handleDeleteNote = async (noteId: string) => {
    if (!selectedResume) return;
    try {
      await resClient.delete(`/resumes/${selectedResume.id}/notes/${noteId}`);
      setNotes((prev) => prev.filter((n) => n.id !== noteId));
    } catch (err) {
      console.error("Failed to delete note", err);
    }
  };

  // Open Chatbot modal
  const handleOpenChat = (resume: Resume, e: React.MouseEvent) => {
    e.stopPropagation();
    setChatResumeObj(resume);
    setChatMessages([
      {
        sender: "bot",
        text: `Hello! I've fully parsed ${resume.name}'s resume. You can ask me anything about their background, skills, projects, or work history!`
      }
    ]);
    setChatbotOpen(true);
  };

  // Send message to chatbot
  const handleSendChatMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!chatResumeObj || !chatInput.trim()) return;

    const userText = chatInput.trim();
    setChatMessages((prev) => [...prev, { sender: "user", text: userText }]);
    setChatInput("");
    setChatLoading(true);

    try {
      const response = await resClient.post(`/resume/${chatResumeObj.id}/chat`, {
        question: userText
      });
      setChatMessages((prev) => [...prev, { sender: "bot", text: response.data.answer }]);
    } catch (err) {
      setChatMessages((prev) => [
        ...prev,
        { sender: "bot", text: "Sorry, I had trouble processing that request. Please try again." }
      ]);
    } finally {
      setChatLoading(false);
    }
  };

  // Trigger candidate comparison
  const handleCompare = async () => {
    if (selectedForCompare.length < 2) return;
    if (!jobDescription.trim()) return;

    setCompareLoading(true);
    try {
      const response = await resClient.get("/resumes/compare", {
        params: {
          ids: selectedForCompare.join(","),
          job_description: jobDescription
        }
      });
      setCompareResults(response.data);
    } catch (err) {
      console.error("Compare failed", err);
    } finally {
      setCompareLoading(false);
    }
  };

  // Toggle selection for comparison
  const toggleSelectForCompare = (id: string, e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedForCompare((prev) => {
      if (prev.includes(id)) {
        return prev.filter((x) => x !== id);
      }
      if (prev.length >= 5) return prev; // Limit comparison to 5
      return [...prev, id];
    });
  };

  // Drag and Drop File Upload
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = e.target.files;
    if (!selectedFiles || selectedFiles.length === 0) return;

    setUploading(true);
    setUploadProgress("Reading files…");
    const formData = new FormData();

    // Check if it's a single ZIP file or multiple PDF/DOCX
    const firstFile = selectedFiles[0];
    if (selectedFiles.length === 1 && firstFile.name.endsWith(".zip")) {
      formData.append("file", firstFile);
      setUploadProgress("Uploading bulk ZIP package…");
      try {
        await resClient.post("/upload/bulk", formData, {
          headers: { "Content-Type": "multipart/form-data" }
        });
        setUploadProgress("Parsing and building embeddings…");
        await loadResumes();
      } catch (err) {
        console.error("ZIP upload failed", err);
      }
    } else {
      for (let i = 0; i < selectedFiles.length; i++) {
        formData.append("files", selectedFiles[i]);
      }
      setUploadProgress(`Uploading ${selectedFiles.length} resumes…`);
      try {
        await resClient.post("/upload", formData, {
          headers: { "Content-Type": "multipart/form-data" }
        });
        setUploadProgress("Parsing and indexing text…");
        await loadResumes();
      } catch (err) {
        console.error("Resume upload failed", err);
      }
    }
    setUploading(false);
    setUploadProgress("");
  };

  return (
    <div style={{ position: "relative", minHeight: "100%" }}>
      {/* Page Header */}
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="page-title">Resume Analyzer & AI Matcher</h1>
          <div className="page-subtitle">Perform semantic query searches, tag profiles, upload bulk ZIPs, and chat with candidate profiles.</div>
        </div>
        <GlobalHeaderControls />
      </div>

      {/* Metrics Row */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: "1rem", marginBottom: "1.5rem" }}>
        <div className="card" style={{ background: "linear-gradient(135deg, rgba(21, 50, 115, 0.4), rgba(255, 255, 255, 0.03))", border: "1px solid rgba(21, 50, 115, 0.3)" }}>
          <div style={{ fontSize: "0.8rem", textTransform: "uppercase", letterSpacing: "1px", color: "rgba(255,255,255,0.6)" }}>Total Parsed Resumes</div>
          <div style={{ fontSize: "2rem", fontWeight: 800, marginTop: "0.5rem", color: "#fff" }}>{resumes.length}</div>
        </div>
        <div className="card" style={{ background: "linear-gradient(135deg, rgba(21, 50, 115, 0.4), rgba(255, 255, 255, 0.03))", border: "1px solid rgba(21, 50, 115, 0.3)" }}>
          <div style={{ fontSize: "0.8rem", textTransform: "uppercase", letterSpacing: "1px", color: "rgba(255,255,255,0.6)" }}>Shortlisted Candidates</div>
          <div style={{ fontSize: "2rem", fontWeight: 800, marginTop: "0.5rem", color: "var(--brand-400)" }}>
            {resumes.filter((r) => r.is_shortlisted).length}
          </div>
        </div>
        <div className="card" style={{ background: "linear-gradient(135deg, rgba(21, 50, 115, 0.4), rgba(255, 255, 255, 0.03))", border: "1px solid rgba(21, 50, 115, 0.3)" }}>
          <div style={{ fontSize: "0.8rem", textTransform: "uppercase", letterSpacing: "1px", color: "rgba(255,255,255,0.6)" }}>Average Experience</div>
          <div style={{ fontSize: "2rem", fontWeight: 800, marginTop: "0.5rem", color: "#fff" }}>
            {resumes.length ? (resumes.reduce((acc, r) => acc + r.experience_years, 0) / resumes.length).toFixed(1) : 0} Years
          </div>
        </div>
      </div>

      {/* Main Grid Layout */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 340px", gap: "1.5rem", alignItems: "flex-start" }}>

        {/* Left Side: Table list, filters & search */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>

          {/* Search bar & Query Toggle */}
          <div className="card" style={{ padding: "1rem" }}>
            <form onSubmit={handleSearch} style={{ display: "flex", gap: "0.5rem", width: "100%" }}>
              <div style={{ display: "flex", borderRadius: "var(--border-radius)", border: "1px solid rgba(255,255,255,0.15)", background: "rgba(0,0,0,0.2)", flexGrow: 1, overflow: "hidden" }}>
                <select
                  value={searchType}
                  onChange={(e) => setSearchType(e.target.value as any)}
                  style={{ background: "rgba(255,255,255,0.05)", color: "#fff", border: "none", padding: "0 0.75rem", fontSize: "0.85rem", borderRight: "1px solid rgba(255,255,255,0.1)" }}
                >
                  <option value="text" style={{ background: "#153273" }}>Key Skills</option>
                  <option value="semantic" style={{ background: "#153273" }}>AI Query</option>
                </select>
                <input
                  type="text"
                  placeholder={searchType === "semantic" ? "Search: 'Who has 3+ years experience with React?'" : "Search by skills: 'React, Node, Typescript'"}
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  style={{ background: "transparent", border: "none", color: "#fff", padding: "0.5rem 1rem", flexGrow: 1, fontSize: "0.9rem" }}
                />
              </div>
              <button type="submit" className="btn btn-secondary" style={{ backgroundColor: "var(--brand-500)", display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <Icons.Search /> {searchType === "semantic" ? "AI Search" : "Search"}
              </button>
            </form>

            {/* Quick Filter Pill Buttons */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.5rem", marginTop: "1rem", borderTop: "1px solid rgba(255,255,255,0.05)", paddingTop: "0.75rem" }}>
              <div style={{ fontSize: "0.8rem", color: "rgba(255,255,255,0.5)", alignSelf: "center", marginRight: "0.5rem" }}>Experience:</div>
              {(["all", "junior", "mid", "senior"] as const).map((lvl) => (
                <button
                  key={lvl}
                  className={`btn btn-sm ${experienceFilter === lvl ? "btn-secondary" : "btn-secondary"}`}
                  onClick={() => setExperienceFilter(lvl)}
                  style={{
                    backgroundColor: experienceFilter === lvl ? "var(--brand-500)" : "rgba(255,255,255,0.05)",
                    border: "none",
                    textTransform: "capitalize",
                    fontSize: "0.75rem",
                    padding: "4px 10px"
                  }}
                >
                  {lvl === "all" ? "All Levels" : lvl === "junior" ? "Junior (< 2y)" : lvl === "mid" ? "Mid-Level (2-5y)" : "Senior (> 5y)"}
                </button>
              ))}

              <div style={{ fontSize: "0.8rem", color: "rgba(255,255,255,0.5)", alignSelf: "center", margin: "0 0.5rem 0 1rem" }}>Shortlisted:</div>
              <button
                className="btn btn-sm"
                onClick={() => setShortlistFilter((prev) => (prev === "all" ? "shortlisted" : "all"))}
                style={{
                  backgroundColor: shortlistFilter === "shortlisted" ? "var(--brand-500)" : "rgba(255,255,255,0.05)",
                  border: "none",
                  fontSize: "0.75rem",
                  padding: "4px 10px",
                  display: "flex",
                  alignItems: "center",
                  gap: "0.3rem"
                }}
              >
                <Icons.Star filled={shortlistFilter === "shortlisted"} /> Shortlisted Only
              </button>

              {selectedForCompare.length >= 2 && (
                <button
                  className="btn btn-sm"
                  onClick={() => setCompareOpen(true)}
                  style={{ marginLeft: "auto", background: "var(--brand-400)", color: "#153273", fontWeight: 800, border: "none", fontSize: "0.75rem", padding: "4px 10px" }}
                >
                  Compare ({selectedForCompare.length}) Candidates
                </button>
              )}
            </div>
          </div>

          {/* Resumes Grid/Table */}
          <div className="card" style={{ padding: "0" }}>
            {loading ? (
              <div style={{ padding: "5rem 0" }}><SectionLoader size="lg" /></div>
            ) : filteredResumes.length === 0 ? (
              <div style={{ padding: "3rem", textAlign: "center", color: "rgba(255,255,255,0.6)" }}>No parsed candidates found matching filters.</div>
            ) : (
              <div className="table-responsive" style={{ overflowX: "auto" }}>
                <table className="table" style={{ width: "100%", borderCollapse: "collapse", color: "#fff" }}>
                  <thead>
                    <tr style={{ background: "rgba(255,255,255,0.02)", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
                      <th style={{ padding: "0.75rem 1rem", textAlign: "left", fontSize: "0.8rem", color: "rgba(255,255,255,0.6)" }}>Name & Role</th>
                      <th style={{ padding: "0.75rem 1rem", textAlign: "left", fontSize: "0.8rem", color: "rgba(255,255,255,0.6)" }}>Contact Info</th>
                      <th style={{ padding: "0.75rem 1rem", textAlign: "center", fontSize: "0.8rem", color: "rgba(255,255,255,0.6)" }}>Exp</th>
                      <th style={{ padding: "0.75rem 1rem", textAlign: "left", fontSize: "0.8rem", color: "rgba(255,255,255,0.6)" }}>Primary Core Skills</th>
                      <th style={{ padding: "0.75rem 1rem", textAlign: "center", fontSize: "0.8rem", color: "rgba(255,255,255,0.6)" }}>Date Added</th>
                      <th style={{ padding: "0.75rem 1rem", textAlign: "right", fontSize: "0.8rem", color: "rgba(255,255,255,0.6)" }}>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {filteredResumes.map((r) => (
                      <tr
                        key={r.id}
                        onClick={() => handleOpenDetails(r)}
                        style={{
                          borderBottom: "1px solid rgba(255,255,255,0.05)",
                          cursor: "pointer",
                          transition: "background 0.2s"
                        }}
                        onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.02)")}
                        onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
                      >
                        <td style={{ padding: "0.75rem 1rem", verticalAlign: "middle" }}>
                          <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                            <input
                              type="checkbox"
                              checked={selectedForCompare.includes(r.id)}
                              onChange={(e) => toggleSelectForCompare(r.id, e as any)}
                              onClick={(e) => e.stopPropagation()}
                              style={{ marginRight: "0.3rem", cursor: "pointer" }}
                              title="Select to compare"
                            />
                            <div>
                              <div style={{ fontWeight: 800 }}>{r.name}</div>
                              <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.5)", marginTop: "2px" }}>
                                {r.role || "Candidate Designation"}
                              </div>
                            </div>
                          </div>
                        </td>
                        <td style={{ padding: "0.75rem 1rem", verticalAlign: "middle" }}>
                          <div style={{ fontSize: "0.8rem", display: "flex", flexDirection: "column", gap: "2px" }}>
                            {r.email && <span style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}><Icons.Mail /> {r.email}</span>}
                            {r.phone && <span style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}><Icons.Phone /> {r.phone}</span>}
                          </div>
                        </td>
                        <td style={{ padding: "0.75rem 1rem", textAlign: "center", verticalAlign: "middle", fontWeight: 800 }}>
                          {r.experience_years.toFixed(1)} yr
                        </td>
                        <td style={{ padding: "0.75rem 1rem", verticalAlign: "middle" }}>
                          <div style={{ display: "flex", flexWrap: "wrap", gap: "0.25rem", maxWidth: "300px" }}>
                            {r.primary_skills.slice(0, 4).map((sk, idx) => (
                              <span
                                key={idx}
                                style={{
                                  fontSize: "0.7rem",
                                  padding: "2px 6px",
                                  borderRadius: "4px",
                                  background: "rgba(21, 50, 115, 0.4)",
                                  color: "var(--brand-300)"
                                }}
                              >
                                {sk}
                              </span>
                            ))}
                            {r.primary_skills.length > 4 && (
                              <span style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.4)", alignSelf: "center" }}>
                                +{r.primary_skills.length - 4}
                              </span>
                            )}
                          </div>
                        </td>
                        <td style={{ padding: "0.75rem 1rem", textAlign: "center", verticalAlign: "middle", fontSize: "0.8rem", color: "rgba(255,255,255,0.7)" }}>
                          {r.created_at ? new Date(r.created_at).toLocaleDateString() : 'N/A'}
                        </td>
                        <td style={{ padding: "0.75rem 1rem", textAlign: "right", verticalAlign: "middle" }}>
                          <div style={{ display: "inline-flex", gap: "0.4rem" }}>
                            {/* <button
                              type="button"
                              className="btn btn-secondary btn-icon btn-sm"
                              onClick={(e) => handleOpenChat(r, e)}
                              title="Open AI Resume Chatbot"
                              style={{ background: "rgba(255,255,255,0.05)", border: "none", padding: "4px" }}
                            >
                              <Icons.Chat />
                            </button> */}
                            <button
                              type="button"
                              className="btn btn-secondary btn-icon btn-sm"
                              onClick={(e) => toggleShortlist(r, e)}
                              title={r.is_shortlisted ? "Remove from Shortlist" : "Add to Shortlist"}
                              style={{ background: "rgba(255,255,255,0.05)", border: "none", padding: "4px" }}
                            >
                              <Icons.Star filled={r.is_shortlisted} />
                            </button>
                            <button
                              type="button"
                              className="btn btn-secondary btn-icon btn-sm"
                              onClick={(e) => {
                                e.stopPropagation();
                                setConfirmDelete(r.id);
                              }}
                              title="Delete Candidate profile"
                              style={{ background: "rgba(255,255,255,0.05)", border: "none", color: "rgba(255,255,255,0.4)", padding: "4px" }}
                            >
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
        </div>

        {/* Right Side: Upload Zone & Actions */}
        <div style={{ display: "flex", flexDirection: "column", gap: "1rem" }}>

          {/* File Upload Zone */}
          <div className="card" style={{ padding: "1.25rem", textAlign: "center" }}>
            <div style={{ fontSize: "1rem", fontWeight: 800, marginBottom: "0.5rem" }}>Upload Resumes</div>
            <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.6)", marginBottom: "1rem" }}>Supports single PDF/DOCX or multiple files at once. Upload a ZIP file for bulk extraction.</div>

            <label
              style={{
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                border: "2px dashed rgba(255,255,255,0.15)",
                borderRadius: "var(--border-radius)",
                padding: "2rem 1rem",
                cursor: "pointer",
                background: "rgba(0,0,0,0.1)",
                transition: "border 0.2s"
              }}
              onMouseEnter={(e) => (e.currentTarget.style.borderColor = "var(--brand-400)")}
              onMouseLeave={(e) => (e.currentTarget.style.borderColor = "rgba(255,255,255,0.15)")}
            >
              <Icons.Upload />
              <span style={{ fontSize: "0.85rem", marginTop: "0.75rem", fontWeight: 700 }}>Choose Files</span>
              <span style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.5)", marginTop: "2px" }}>PDF, DOCX, ZIP</span>
              <input
                type="file"
                multiple
                accept=".pdf,.docx,.zip"
                onChange={handleFileUpload}
                style={{ display: "none" }}
                disabled={uploading}
              />
            </label>

            {uploading && (
              <div style={{ marginTop: "1rem", background: "rgba(255,255,255,0.05)", borderRadius: "4px", padding: "0.5rem" }}>
                <SectionLoader size="sm" />
                <div style={{ fontSize: "0.75rem", color: "var(--brand-400)", fontWeight: 800, marginTop: "5px" }}>{uploadProgress}</div>
              </div>
            )}
          </div>

          {/* Quick AI Search Helper Card */}
          <div className="card" style={{ padding: "1.25rem", background: "linear-gradient(135deg, rgba(21, 50, 115, 0.3), rgba(0,0,0,0.2))", border: "1px solid rgba(21, 50, 115, 0.4)" }}>
            <div style={{ display: "flex", alignItems: "center", gap: "0.4rem", color: "var(--brand-400)", fontWeight: 800, fontSize: "0.9rem", marginBottom: "0.5rem" }}>
              <Icons.Sparkles /> <span>AI Search Prompt Tips</span>
            </div>
            <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.7)", display: "flex", flexDirection: "column", gap: "0.5rem" }}>
              <div>You can write complete semantic requests:</div>
              <div style={{ fontFamily: "monospace", background: "rgba(0,0,0,0.3)", padding: "4px 8px", borderRadius: "4px", color: "var(--brand-300)" }}>
                "Show candidates with 3+ years experience with React"
              </div>
              <div style={{ fontFamily: "monospace", background: "rgba(0,0,0,0.3)", padding: "4px 8px", borderRadius: "4px", color: "var(--brand-300)" }}>
                "Find .NET backend developers in Chandigarh"
              </div>
            </div>
          </div>
        </div>

      </div>

      {/* Candidate Details Drawer Sidebar */}
      {sidebarOpen && selectedResume && (
        <div
          style={{
            position: "fixed",
            top: 0,
            right: 0,
            width: "500px",
            height: "100vh",
            background: "#153273", // Deep navy solid sidebar
            boxShadow: "-4px 0 20px rgba(0,0,0,0.5)",
            zIndex: 100,
            display: "flex",
            flexDirection: "column",
            animation: "slideIn 0.3s ease"
          }}
        >
          {/* Header */}
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "1.25rem", borderBottom: "1px solid rgba(255,255,255,0.1)" }}>
            <div>
              <div style={{ fontSize: "1.1rem", fontWeight: 900, color: "#fff" }}>{selectedResume.name}</div>
              <div style={{ fontSize: "0.8rem", color: "rgba(255,255,255,0.6)" }}>{selectedResume.role || "Designation"}</div>
            </div>
            <button
              onClick={() => setSidebarOpen(false)}
              style={{ background: "transparent", border: "none", color: "rgba(255,255,255,0.6)", cursor: "pointer" }}
            >
              <Icons.Close />
            </button>
          </div>

          {/* Body Content */}
          <div style={{ flexGrow: 1, overflowY: "auto", padding: "1.25rem", display: "flex", flexDirection: "column", gap: "1rem" }}>

            {/* Contact details */}
            <div style={{ display: "flex", flexWrap: "wrap", gap: "0.75rem", fontSize: "0.8rem", background: "rgba(0,0,0,0.15)", padding: "0.75rem", borderRadius: "var(--border-radius)" }}>
              {selectedResume.email && <span style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}><Icons.Mail /> {selectedResume.email}</span>}
              {selectedResume.phone && <span style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}><Icons.Phone /> {selectedResume.phone}</span>}
              {selectedResume.location && <span style={{ display: "inline-flex", alignItems: "center", gap: "4px" }}><Icons.Briefcase /> {selectedResume.location}</span>}
            </div>

            {/* Experience Stats */}
            <div style={{ display: "flex", gap: "0.5rem" }}>
              <div style={{ flex: 1, background: "rgba(255,255,255,0.03)", padding: "0.5rem", borderRadius: "4px", textAlign: "center" }}>
                <div style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.5)" }}>CORE EXP</div>
                <div style={{ fontSize: "1rem", fontWeight: 800 }}>{selectedResume.experience_years.toFixed(1)} Yrs</div>
              </div>
              <div style={{ flex: 1, background: "rgba(255,255,255,0.03)", padding: "0.5rem", borderRadius: "4px", textAlign: "center" }}>
                <div style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.5)" }}>TOTAL EXP</div>
                <div style={{ fontSize: "1rem", fontWeight: 800 }}>{selectedResume.total_experience_years.toFixed(1)} Yrs</div>
              </div>
              <div style={{ flex: 1, background: "rgba(255,255,255,0.03)", padding: "0.5rem", borderRadius: "4px", textAlign: "center" }}>
                <div style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.5)" }}>LEVEL</div>
                <div style={{ fontSize: "1rem", fontWeight: 800, textTransform: "capitalize" }}>{selectedResume.experience_level || "Mid"}</div>
              </div>
            </div>

            {/* Primary Skills */}
            <div>
              <div style={{ fontSize: "0.8rem", fontWeight: 800, color: "rgba(255,255,255,0.6)", marginBottom: "0.3rem" }}>Primary Core Skills</div>
              <div style={{ display: "flex", flexWrap: "wrap", gap: "0.25rem" }}>
                {selectedResume.primary_skills.map((s, idx) => (
                  <span key={idx} style={{ fontSize: "0.75rem", padding: "2px 6px", borderRadius: "4px", background: "rgba(21, 50, 115, 0.6)", color: "var(--brand-300)" }}>{s}</span>
                ))}
              </div>
            </div>

            {/* Summary */}
            {selectedResume.summary && (
              <div>
                <div style={{ fontSize: "0.8rem", fontWeight: 800, color: "rgba(255,255,255,0.6)", marginBottom: "0.3rem" }}>Profile Summary</div>
                <p style={{ fontSize: "0.8rem", color: "rgba(255,255,255,0.8)", margin: 0, lineHeight: 1.4, background: "rgba(0,0,0,0.1)", padding: "0.5rem", borderRadius: "4px" }}>{selectedResume.summary}</p>
              </div>
            )}

            {/* Education */}
            {selectedResume.education.length > 0 && (
              <div>
                <div style={{ fontSize: "0.8rem", fontWeight: 800, color: "rgba(255,255,255,0.6)", marginBottom: "0.3rem" }}>Education</div>
                <div style={{ display: "flex", flexDirection: "column", gap: "4px", fontSize: "0.8rem", color: "rgba(255,255,255,0.8)" }}>
                  {selectedResume.education.map((edu, idx) => (
                    <div key={idx} style={{ display: "flex", gap: "0.5rem" }}>
                      <span style={{ color: "var(--brand-400)" }}>•</span>
                      <span>{edu}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Actions Panel */}
            <div style={{ borderTop: "1px solid rgba(255,255,255,0.1)", paddingTop: "1rem", display: "flex", gap: "0.5rem" }}>
              <a
                href={`${resumeApiUrl}/files/${selectedResume.id}`} // Serve file directly
                target="_blank"
                rel="noopener noreferrer"
                className="btn btn-secondary"
                style={{ flex: 1, textAlign: "center", fontSize: "0.8rem", background: "rgba(255,255,255,0.1)", border: "none" }}
              >
                <Icons.FileText /> View PDF Resume
              </a>
              <button
                onClick={(e) => handleOpenChat(selectedResume, e)}
                className="btn btn-secondary"
                style={{ flex: 1, fontSize: "0.8rem", backgroundColor: "var(--brand-500)", border: "none" }}
              >
                <Icons.Chat /> Start AI Chat
              </button>
            </div>

            {/* Notes History */}
            <div style={{ borderTop: "1px solid rgba(255,255,255,0.1)", paddingTop: "1rem" }}>
              <div style={{ fontSize: "0.85rem", fontWeight: 800, color: "rgba(255,255,255,0.6)", marginBottom: "0.5rem" }}>HR Interview / Candidate Notes</div>

              {/* Add Note Form */}
              <form onSubmit={handleAddNote} style={{ display: "flex", gap: "0.3rem", marginBottom: "0.75rem" }}>
                <input
                  type="text"
                  placeholder="Type note and hit Enter…"
                  value={newNote}
                  onChange={(e) => setNewNote(e.target.value)}
                  style={{ background: "rgba(0,0,0,0.2)", border: "1px solid rgba(255,255,255,0.15)", color: "#fff", padding: "4px 8px", flexGrow: 1, borderRadius: "4px", fontSize: "0.8rem" }}
                />
                <button type="submit" className="btn btn-secondary btn-sm" style={{ background: "var(--brand-400)", color: "#153273", border: "none" }}>Add</button>
              </form>

              {/* Note List */}
              {notesLoading ? (
                <SectionLoader size="sm" />
              ) : notes.length === 0 ? (
                <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.4)" }}>No interview history found for this profile.</div>
              ) : (
                <div style={{ display: "flex", flexDirection: "column", gap: "0.4rem" }}>
                  {notes.map((n) => (
                    <div key={n.id} style={{ background: "rgba(0,0,0,0.15)", padding: "0.5rem", borderRadius: "4px", display: "flex", justifyContent: "space-between", gap: "0.5rem" }}>
                      <div style={{ minWidth: 0 }}>
                        <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.8)" }}>{n.note}</div>
                        <div style={{ fontSize: "0.65rem", color: "rgba(255,255,255,0.4)", marginTop: "2px" }}>
                          {new Date(n.created_at).toLocaleDateString()} {new Date(n.created_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                        </div>
                      </div>
                      <button
                        onClick={() => handleDeleteNote(n.id)}
                        style={{ border: "none", background: "transparent", color: "rgba(255,255,255,0.35)", cursor: "pointer", alignSelf: "center", padding: "4px" }}
                        title="Delete note"
                      >
                        <Icons.Delete />
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

          </div>
        </div>
      )}

      {/* AI Resume Chatbot Modal */}
      {chatbotOpen && chatResumeObj && (
        <div style={{ position: "fixed", top: 0, left: 0, width: "100%", height: "100%", background: "rgba(0,0,0,0.6)", zIndex: 120, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <div className="card" style={{ width: "500px", height: "600px", padding: 0, display: "flex", flexDirection: "column", overflow: "hidden", border: "1px solid rgba(255,255,255,0.15)" }}>

            {/* Modal Header */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "1rem", borderBottom: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.02)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: "0.5rem" }}>
                <Icons.Sparkles />
                <div>
                  <div style={{ fontWeight: 800 }}>Chat with {chatResumeObj.name}</div>
                  <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.5)" }}>AI Interview Helper</div>
                </div>
              </div>
              <button
                onClick={() => setChatbotOpen(false)}
                style={{ background: "transparent", border: "none", color: "rgba(255,255,255,0.6)", cursor: "pointer" }}
              >
                <Icons.Close />
              </button>
            </div>

            {/* Messages Screen */}
            <div style={{ flexGrow: 1, overflowY: "auto", padding: "1rem", display: "flex", flexDirection: "column", gap: "0.75rem" }}>
              {chatMessages.map((m, idx) => (
                <div
                  key={idx}
                  style={{
                    alignSelf: m.sender === "user" ? "flex-end" : "flex-start",
                    background: m.sender === "user" ? "var(--brand-500)" : "rgba(255,255,255,0.06)",
                    color: "#fff",
                    padding: "0.5rem 0.8rem",
                    borderRadius: m.sender === "user" ? "12px 12px 2px 12px" : "12px 12px 12px 2px",
                    maxWidth: "80%",
                    fontSize: "0.8rem",
                    lineHeight: 1.4
                  }}
                >
                  {m.text}
                </div>
              ))}
              {chatLoading && (
                <div style={{ alignSelf: "flex-start", background: "rgba(255,255,255,0.06)", padding: "0.5rem 0.8rem", borderRadius: "12px 12px 12px 2px" }}>
                  <SectionLoader size="sm" />
                </div>
              )}
            </div>

            {/* Input Form */}
            <form onSubmit={handleSendChatMessage} style={{ display: "flex", gap: "0.4rem", padding: "1rem", borderTop: "1px solid rgba(255,255,255,0.1)", background: "rgba(0,0,0,0.2)" }}>
              <input
                type="text"
                placeholder={`Ask me: "What projects has ${chatResumeObj.name.split(" ")[0]} worked on?"`}
                value={chatInput}
                onChange={(e) => setChatInput(e.target.value)}
                style={{ background: "rgba(0,0,0,0.3)", border: "1px solid rgba(255,255,255,0.15)", color: "#fff", padding: "8px 12px", flexGrow: 1, borderRadius: "4px", fontSize: "0.85rem" }}
              />
              <button type="submit" className="btn btn-secondary btn-icon" style={{ background: "var(--brand-400)", color: "#153273", border: "none", padding: "8px" }} disabled={chatLoading}>
                <Icons.Send />
              </button>
            </form>

          </div>
        </div>
      )}

      {/* Compare Modal Panel */}
      {compareOpen && (
        <div style={{ position: "fixed", top: 0, left: 0, width: "100%", height: "100%", background: "rgba(0,0,0,0.6)", zIndex: 120, display: "flex", alignItems: "center", justifyContent: "center" }}>
          <div className="card" style={{ width: "700px", height: "550px", display: "flex", flexDirection: "column", overflow: "hidden", border: "1px solid rgba(255,255,255,0.15)" }}>

            {/* Header */}
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "1rem", borderBottom: "1px solid rgba(255,255,255,0.1)", background: "rgba(255,255,255,0.02)" }}>
              <div>
                <div style={{ fontWeight: 800, fontSize: "1.1rem" }}>Candidate Fit Comparison</div>
                <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.5)" }}>Compare up to 5 candidates against a Job Description</div>
              </div>
              <button
                onClick={() => setCompareOpen(false)}
                style={{ background: "transparent", border: "none", color: "rgba(255,255,255,0.6)", cursor: "pointer" }}
              >
                <Icons.Close />
              </button>
            </div>

            {/* Input Form */}
            <div style={{ padding: "1rem", borderBottom: "1px solid rgba(255,255,255,0.05)", background: "rgba(0,0,0,0.15)" }}>
              <div style={{ fontSize: "0.8rem", fontWeight: 700, color: "rgba(255,255,255,0.6)", marginBottom: "4px" }}>Enter Job Description</div>
              <div style={{ display: "flex", gap: "0.5rem" }}>
                <textarea
                  placeholder="Paste your Job Description here to analyze fit scores..."
                  rows={2}
                  value={jobDescription}
                  onChange={(e) => setJobDescription(e.target.value)}
                  style={{ background: "rgba(0,0,0,0.2)", border: "1px solid rgba(255,255,255,0.15)", color: "#fff", padding: "8px", flexGrow: 1, borderRadius: "4px", fontSize: "0.8rem" }}
                />
                <button
                  onClick={handleCompare}
                  className="btn btn-secondary"
                  style={{ backgroundColor: "var(--brand-500)", border: "none", alignSelf: "center", height: "fit-content" }}
                  disabled={compareLoading}
                >
                  {compareLoading ? "Analyzing..." : "Compare"}
                </button>
              </div>
            </div>

            {/* Compare Results Screen */}
            <div style={{ flexGrow: 1, overflowY: "auto", padding: "1rem" }}>
              {compareLoading ? (
                <div style={{ padding: "3rem 0" }}><SectionLoader size="md" /></div>
              ) : compareResults.length === 0 ? (
                <div style={{ color: "rgba(255,255,255,0.5)", textAlign: "center", padding: "3rem" }}>Write a Job Description and click Compare to compute AI match statistics.</div>
              ) : (
                <div style={{ display: "grid", gridTemplateColumns: `repeat(${compareResults.length}, 1fr)`, gap: "0.75rem", height: "100%" }}>
                  {compareResults.map((r, idx) => (
                    <div key={idx} style={{ background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "var(--border-radius)", padding: "0.75rem", display: "flex", flexDirection: "column", gap: "0.5rem", minWidth: 0 }}>
                      <div style={{ fontWeight: 800, fontSize: "0.9rem", borderBottom: "1px solid rgba(255,255,255,0.08)", paddingBottom: "4px" }}>{r.name}</div>

                      {/* Fit Score */}
                      <div style={{ textAlign: "center", padding: "0.5rem", background: "rgba(21,50,115,0.3)", borderRadius: "4px" }}>
                        <div style={{ fontSize: "0.65rem", color: "rgba(255,255,255,0.5)" }}>FIT SCORE</div>
                        <div style={{ fontSize: "1.5rem", fontWeight: 900, color: "var(--brand-400)" }}>{r.fit_score ?? "N/A"}/10</div>
                      </div>

                      <div style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.5)" }}>CORE EXPERIENCE</div>
                      <div style={{ fontSize: "0.8rem", fontWeight: 700 }}>{r.experience_years.toFixed(1)} yr</div>

                      <div style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.5)" }}>PRIMARY TECH</div>
                      <div style={{ display: "flex", flexWrap: "wrap", gap: "3px" }}>
                        {r.primary_skills.slice(0, 3).map((sk: string, i: number) => (
                          <span key={i} style={{ fontSize: "0.65rem", background: "rgba(255,255,255,0.06)", padding: "2px 4px", borderRadius: "3px" }}>{sk}</span>
                        ))}
                      </div>

                      {r.fit_summary && (
                        <div style={{ marginTop: "auto", flexGrow: 1, display: "flex", flexDirection: "column" }}>
                          <div style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.5)", marginBottom: "2px" }}>AI FEEDBACK</div>
                          <p style={{ fontSize: "0.7rem", color: "rgba(255,255,255,0.85)", margin: 0, overflowY: "auto", maxHeight: "120px", background: "rgba(0,0,0,0.15)", padding: "6px", borderRadius: "3px", lineHeight: 1.3 }}>{r.fit_summary}</p>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>

          </div>
        </div>
      )}

      {/* Confirm Delete Candidate modal */}
      <ConfirmModal
        isOpen={!!confirmDelete}
        onClose={() => setConfirmDelete(null)}
        onConfirm={handleDeleteResume}
        title="Delete Candidate profile"
        message="Are you sure you want to delete this candidate? This will wipe out their profile data, resume notes history, and Chroma DB vector indexes. This cannot be undone."
        confirmText="Delete Profile"
      />
    </div>
  );
}
