/** How to read `expected_working_hours`.
 *
 *  0 is a REAL value meaning "this person has no fixed daily target" -- the
 *  backend writes exactly that for support staff, whose hours vary and whose
 *  pay is a flat monthly figure. It is not the same as the field being unset.
 *
 *  Every reader used to write `expected_working_hours || 9`, and `0 || 9` is 9,
 *  so somebody on no fixed hours was shown a nine-hour target, measured against
 *  it, and marked SHORT for not meeting it.
 *
 *  `null` is returned rather than 0 so callers cannot accidentally divide by it
 *  -- a progress bar computing `worked / 0` renders a NaN width.
 */
export const DEFAULT_DAILY_HOURS = 9;

export function dailyTarget(hours: number | null | undefined): number | null {
  if (hours === null || hours === undefined) return DEFAULT_DAILY_HOURS;
  const value = Number(hours);
  if (!Number.isFinite(value) || value <= 0) return null;
  return value;
}
