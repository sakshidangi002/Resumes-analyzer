import { useEffect, useRef, useState } from "react";
import { useAuth } from "../auth/AuthContext";
import { chatbot } from "../api/client";
import type { ChatbotContext, ChatbotKind, ChatbotReply } from "../api/client";

/**
 * The HRMS chatbot, as a floating panel available on every dashboard page.
 *
 * Deliberately thin: every decision about what the user is told — refusals,
 * clarifying questions, "payroll hasn't run yet" — is made by the backend and
 * arrives as `answer`. This component renders that string and never composes a
 * message of its own from `data`, so there is exactly one place in the product
 * that decides what the chatbot says.
 *
 * History is per-session and lives only in this component's state. The endpoint
 * answers each question on its own, so persisting the transcript would imply a
 * continuity the backend does not have.
 */

type Message = {
  id: number;
  role: "user" | "bot";
  text: string;
  kind?: ChatbotKind;
  /** Aggregates the answer already stated, rendered as chips underneath it. */
  facts?: { label: string; value: string }[];
  /** Who the answer was about, when it was about a person. */
  about?: string;
};

const GREETING =
  "Hi! Ask me about your attendance, leave balance or payslips — for example " +
  '"how many days was I absent last month?"';

// Admin and HR can ask about any employee and about the company as a whole, and
// the backend gates that by role. Showing them only the personal prompts would
// hide most of what the chatbot does from the people who need it most.
const GREETING_HR =
  "Hi! Ask me about any employee or about the company — for example " +
  '"how many people are absent today?" or "what was Priya\'s net pay last month?"';

const SUGGESTIONS = [
  "How many days was I absent last month?",
  "What is my leave balance?",
  "What was my net pay last month?",
  "When is the next holiday?",
];

const SUGGESTIONS_HR = [
  "How many people are absent today?",
  "Who is on leave today?",
  "What is the headcount by department?",
  "How many leave requests are pending?",
];

const inr = (v: unknown) => `₹${Math.round(Number(v) || 0).toLocaleString("en-IN")}`;
const num = (v: unknown) => (v === null || v === undefined ? "—" : String(v));

/**
 * Which figures to repeat as chips, per skill.
 *
 * Keyed by skill name so a new skill simply has no chips until someone adds an
 * entry — the sentence above them is always the real answer, and a missing
 * mapping degrades to text rather than to a broken row.
 */
const CHIPS: Record<string, (d: Record<string, any>) => { label: string; value: string }[]> = {
  "attendance.employee_month": (d) => [
    { label: "Present", value: num(d.summary?.present) },
    { label: "Absent", value: num(d.summary?.absent) },
    { label: "Leave", value: num(d.summary?.leave) },
    { label: "Attendance", value: `${num(d.summary?.attendance_percentage)}%` },
  ],
  "attendance.company_day": (d) => [
    { label: "Present", value: num(d.present) },
    { label: "Absent", value: num(d.absent) },
    { label: "On leave", value: num(d.on_leave) },
    { label: "Not marked", value: num(d.not_marked) },
  ],
  "leave.balance": (d) => [
    { label: "Available", value: num(d.remaining) },
    { label: "Earned", value: num(d.earned) },
    { label: "Used", value: num(d.used_paid) },
  ],
  "leave.who_is_off": (d) => [{ label: "On leave", value: num(d.count) }],
  "leave.pending_approvals": (d) => [
    { label: "Pending", value: num(d.count) },
    { label: "Oldest", value: `${num(d.oldest_days)}d` },
  ],
  "payroll.payslip": (d) =>
    d.found
      ? [
          { label: "Net", value: inr(d.net_salary) },
          { label: "Earnings", value: inr(d.total_earnings) },
          { label: "Deductions", value: inr(d.total_deductions) },
        ]
      : [],
  "payroll.company_run": (d) =>
    d.found
      ? [
          { label: "Net", value: inr(d.total_net) },
          { label: "Deductions", value: inr(d.total_deductions) },
          { label: "Employees", value: num(d.payslip_count) },
        ]
      : [],
  "breakdown.leave": (d) => [
    { label: "Employees", value: num((d.employees as unknown[])?.length) },
    { label: "Total days", value: num(d.total_days) },
  ],
  "breakdown.payroll": (d) => [
    { label: "Total", value: inr(d.total_net) },
    { label: "Payslips", value: num((d.employees as unknown[])?.length) },
  ],
  "breakdown.attendance": (d) => [
    { label: "Employees", value: num((d.employees as unknown[])?.length) },
  ],
  "workforce.headcount": (d) => [{ label: "Active", value: num(d.total_active) }],
  "workforce.movement": (d) => [
    { label: "Joined", value: num(d.joiners) },
    { label: "Left", value: num(d.exits) },
  ],
};

function factsFrom(reply: ChatbotReply): { label: string; value: string }[] {
  const data = reply.data as Record<string, any> | null | undefined;
  if (!data || !reply.skill) return [];
  const build = CHIPS[reply.skill];
  if (!build) return [];
  try {
    return build(data).filter((f) => f.value !== "—");
  } catch {
    // A chip row is a nicety; a malformed payload must not blank the answer.
    return [];
  }
}

export default function ChatWidget() {
  const { hasRole } = useAuth();
  // Mirrors the backend's own rule. This only chooses which examples to show;
  // the gate that matters is `Actor` on the server, so a wrong guess here shows
  // an unhelpful prompt, never data the user may not see.
  const isHr = hasRole("Admin") || hasRole("HR");

  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    { id: 0, role: "bot", text: isHr ? GREETING_HR : GREETING },
  ]);

  // What the last answer settled — which lookup, which person, which period —
  // echoed back so "and Neha?" or "what about her leave?" mean something. A ref,
  // not state: it must be current when `send` runs and never needs a re-render.
  const context = useRef<ChatbotContext>({});

  const scrollRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLInputElement | null>(null);
  const nextId = useRef(1);

  // Keep the newest message in view as the conversation and the typing
  // indicator change height.
  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, busy]);

  useEffect(() => {
    if (open) inputRef.current?.focus();
  }, [open]);

  // Escape closes the panel, matching the app's other overlays.
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  const send = async (text: string) => {
    const question = text.trim();
    if (!question || busy) return;

    setMessages((prev) => [
      ...prev,
      { id: nextId.current++, role: "user", text: question },
    ]);
    setInput("");
    setBusy(true);

    try {
      const { data } = await chatbot.ask(question, context.current);
      // A clarification carries no skill of its own; keep the previous one so
      // the answer to "which did you mean?" still has context behind it.
      if (data.skill) context.current.skill = data.skill;
      // The person and period are overwritten every turn, including with null:
      // after a company-wide answer there is no "her" to carry forward, and
      // leaving a stale employee behind would answer the next follow-up about
      // somebody the user has stopped talking about.
      context.current.employeeId = data.employee?.id ?? null;
      context.current.period = data.period?.label ?? null;
      setMessages((prev) => [
        ...prev,
        {
          id: nextId.current++,
          role: "bot",
          text: data.answer || "I couldn't produce an answer for that.",
          kind: data.kind,
          facts: factsFrom(data),
          about: data.employee?.name,
        },
      ]);
    } catch (err: any) {
      // The backend answers refusals and failures with a 200 and an `answer`,
      // so reaching here means the request itself failed — offline, timed out,
      // or the session expired. Say which rather than inventing an HR answer.
      const timedOut = err?.code === "ECONNABORTED";
      setMessages((prev) => [
        ...prev,
        {
          id: nextId.current++,
          role: "bot",
          kind: "error",
          text: timedOut
            ? "That took too long to answer. The chatbot may still be starting up — please try again in a moment."
            : "I couldn't reach the chatbot just now. Please check your connection and try again.",
        },
      ]);
    } finally {
      setBusy(false);
    }
  };

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    void send(input);
  };

  // Only the opening greeting so far — offer starting points rather than an
  // empty box the user has to guess at.
  const showSuggestions = messages.length === 1 && !busy;

  return (
    <>
      <button
        type="button"
        className={`chatbot-fab${open ? " is-open" : ""}`}
        onClick={() => setOpen((v) => !v)}
        aria-label={open ? "Close HR chatbot" : "Open HR chatbot"}
        aria-expanded={open}
      >
        {open ? (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round">
            <path d="M18 6L6 18M6 6l12 12" />
          </svg>
        ) : (
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z" />
          </svg>
        )}
      </button>

      {open && (
        <section className="chatbot-panel" role="dialog" aria-label="HR chatbot">
          <header className="chatbot-head">
            <div>
              <strong>HR Assistant</strong>
              <span className="chatbot-sub">
                {isHr ? "Employees · Attendance · Leave · Payroll" : "Attendance · Leave · Payslips"}
              </span>
            </div>
            <button
              type="button"
              className="chatbot-close"
              onClick={() => setOpen(false)}
              aria-label="Close"
            >
              ×
            </button>
          </header>

          <div className="chatbot-body" ref={scrollRef}>
            {messages.map((m) => (
              <div
                key={m.id}
                className={`chatbot-msg chatbot-msg--${m.role}${
                  m.kind && m.kind !== "answer" && m.kind !== "capabilities"
                    ? " chatbot-msg--muted"
                    : ""
                }`}
              >
                {m.about && <span className="chatbot-about">{m.about}</span>}
                <p>{m.text}</p>
                {m.facts && m.facts.length > 0 && (
                  <div className="chatbot-facts">
                    {m.facts.map((f) => (
                      <span key={f.label} className="chatbot-fact">
                        <em>{f.label}</em>
                        {f.value}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}

            {busy && (
              <div className="chatbot-msg chatbot-msg--bot">
                <span className="chatbot-typing" aria-label="Thinking">
                  <i />
                  <i />
                  <i />
                </span>
              </div>
            )}

            {showSuggestions && (
              <div className="chatbot-suggestions">
                {(isHr ? SUGGESTIONS_HR : SUGGESTIONS).map((s) => (
                  <button key={s} type="button" onClick={() => void send(s)}>
                    {s}
                  </button>
                ))}
              </div>
            )}
          </div>

          <form className="chatbot-input" onSubmit={onSubmit}>
            <input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder={
                isHr ? "Ask about an employee or the company…" : "Ask about attendance, leave or payslips…"
              }
              maxLength={1000}
              disabled={busy}
            />
            <button type="submit" disabled={busy || !input.trim()} aria-label="Send">
              <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M22 2L11 13" />
                <path d="M22 2l-7 20-4-9-9-4 20-7z" />
              </svg>
            </button>
          </form>
        </section>
      )}
    </>
  );
}
