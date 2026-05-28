/** Turn an Axios / FastAPI error into a user-visible string. */
export function formatApiError(err: unknown, fallback: string): string {
  const detail = (err as { response?: { data?: { detail?: unknown } } })?.response
    ?.data?.detail;

  if (typeof detail === "string" && detail.trim()) {
    if (detail === "Method Not Allowed") {
      return (
        "Save failed (server does not allow this request yet). Restart the HRMS " +
        "service on the server, hard-refresh the page (Ctrl+F5), and try again."
      );
    }
    return detail;
  }

  if (Array.isArray(detail)) {
    const msgs = detail
      .map((item) => {
        if (item && typeof item === "object" && "msg" in item) {
          return String((item as { msg: string }).msg);
        }
        return null;
      })
      .filter((m): m is string => Boolean(m));

    const joined = msgs.join(" ");
    if (joined.includes("dsr_id") && joined.includes("integer")) {
      return (
        "The server is running old code. Restart it (Ctrl+C, then " +
        "python run_app.py), hard-refresh this page (Ctrl+F5), and try again."
      );
    }
    if (joined) return joined;
  }

  return fallback;
}
