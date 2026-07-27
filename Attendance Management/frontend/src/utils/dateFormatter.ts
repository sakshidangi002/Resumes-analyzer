/**
 * Formats an ISO date string or Date object to DD-MM-YYYY format.
 * If the input is invalid, returns the original input or an empty string.
 */
export function formatDate(date: string | Date | null | undefined): string {
  if (!date) return "";

  const d = new Date(date);
  if (isNaN(d.getTime())) {
    return typeof date === "string" ? date : "";
  }

  const day = String(d.getDate()).padStart(2, "0");
  const month = String(d.getMonth() + 1).padStart(2, "0");
  const year = d.getFullYear();

  return `${day}-${month}-${year}`;
}

/** Format a UTC ISO timestamp as IST (independent of browser timezone). */
export function formatTimeIST(date: string | Date | null | undefined): string {
  if (!date) return "";
  let d: Date;
  if (typeof date === "string") {
    const raw = date.trim();
    const iso =
      raw.endsWith("Z") || /[+-]\d{2}:\d{2}$/.test(raw) ? raw : `${raw}Z`;
    d = new Date(iso);
  } else {
    d = new Date(date);
  }
  if (isNaN(d.getTime())) return "";
  return d.toLocaleTimeString("en-IN", {
    timeZone: "Asia/Kolkata",
    hour: "2-digit",
    minute: "2-digit",
  });
}
