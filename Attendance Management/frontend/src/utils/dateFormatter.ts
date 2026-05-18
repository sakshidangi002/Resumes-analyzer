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
