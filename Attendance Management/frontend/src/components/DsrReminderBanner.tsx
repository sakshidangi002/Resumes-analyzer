/**
 * 5 PM IST DSR reminder banner.
 *
 * Behavior:
 *  - Computes "now in IST" without depending on the browser's timezone DB
 *    (IST = UTC + 5:30, fixed offset).
 *  - Every 60 s, if IST hour >= 17 AND the server says today's DSR isn't
 *    submitted yet, the banner is shown.
 *  - One browser desktop notification is fired per IST calendar day
 *    (subject to the user granting permission).
 *  - "Dismiss" hides the banner for the rest of the IST day; it returns
 *    only on the next IST day (or after the user submits today's DSR).
 *
 * This is the client-side belt to the backend braces: APScheduler also
 * creates an in-app `AppNotification` at 17:00 IST, so the bell badge in
 * the header lights up even if the user wasn't on this page at 5 PM.
 */
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import {
  dsr,
  type DSRReminderSettings,
  type DSRTodayStatus,
} from "../api/client";
import { useAuth } from "../auth/AuthContext";
import { ensurePushSubscription } from "../lib/push";

const DEFAULT_REMINDER_HOUR_IST = 17;
const DEFAULT_REMINDER_MIN_IST = 0;
const DEFAULT_WEEKDAYS = ["mon", "tue", "wed", "thu", "fri"];
const POLL_MS = 60_000;
const SETTINGS_REFRESH_MS = 10 * 60_000; // refresh schedule every 10 min
const STORAGE_KEY_BANNER = "softwiz.dsrReminder.bannerDismissed";
const STORAGE_KEY_BROWSER = "softwiz.dsrReminder.browserNotified";
const STORAGE_KEY_INBOX = "softwiz.dsrReminder.inboxNotified";
const NOTIF_PERMISSION_DELAY_MS = 4_000;

const WEEKDAY_TOKENS = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"] as const;

type IstNow = { hour: number; minute: number; ymd: string; weekday: string };

function istNow(): IstNow {
  const now = new Date();
  const utcMs = now.getTime() + now.getTimezoneOffset() * 60_000;
  const ist = new Date(utcMs + (5 * 60 + 30) * 60_000);
  const y = ist.getFullYear();
  const m = String(ist.getMonth() + 1).padStart(2, "0");
  const d = String(ist.getDate()).padStart(2, "0");
  // JS getDay(): 0 = Sun ... 6 = Sat. Re-index to match Mon-first tokens.
  const jsDay = ist.getDay();
  const mondayIndex = (jsDay + 6) % 7;
  return {
    hour: ist.getHours(),
    minute: ist.getMinutes(),
    ymd: `${y}-${m}-${d}`,
    weekday: WEEKDAY_TOKENS[mondayIndex],
  };
}

function parseHhmm(value: string): { hour: number; minute: number } {
  const m = /^(\d{1,2}):(\d{2})$/.exec(value || "");
  if (!m) {
    return { hour: DEFAULT_REMINDER_HOUR_IST, minute: DEFAULT_REMINDER_MIN_IST };
  }
  return { hour: parseInt(m[1], 10), minute: parseInt(m[2], 10) };
}

export default function DsrReminderBanner() {
  const { token } = useAuth();
  const [show, setShow] = useState(false);
  const [dateLabel, setDateLabel] = useState<string>("");
  const lastCheckedYmd = useRef<string>("");
  const settingsRef = useRef<DSRReminderSettings | null>(null);

  useEffect(() => {
    if (!token) return;
    // Defer the permission prompt + Web Push subscribe slightly. Pestering at
    // login time is rude; once the dashboard has settled, ensurePushSubscription
    // handles permission + service worker + server-side subscription. After
    // that, the backend can push a desktop notification at 5 PM IST even if the
    // browser tab is closed (provided Chrome/Edge keeps running in the
    // background, which they do by default on Windows).
    const t = window.setTimeout(() => {
      ensurePushSubscription().catch(() => {});
    }, NOTIF_PERMISSION_DELAY_MS);
    return () => window.clearTimeout(t);
  }, [token]);

  useEffect(() => {
    if (!token) {
      setShow(false);
      settingsRef.current = null;
      return;
    }
    let cancelled = false;
    let lastSettingsAt = 0;

    const refreshSettings = async () => {
      try {
        const res = await dsr.reminderSettings();
        if (cancelled) return;
        settingsRef.current = res.data;
      } catch {
        // Endpoint may be missing on older backends — fall back to defaults
        // baked in at the top of this file.
      }
      lastSettingsAt = Date.now();
    };

    const evaluate = async () => {
      const now = istNow();
      lastCheckedYmd.current = now.ymd;

      if (Date.now() - lastSettingsAt > SETTINGS_REFRESH_MS) {
        await refreshSettings();
        if (cancelled) return;
      }

      const settings = settingsRef.current;
      const enabled = settings ? settings.enabled : true;
      if (!enabled) {
        setShow(false);
        return;
      }

      const { hour: targetHour, minute: targetMinute } = settings
        ? parseHhmm(settings.time)
        : { hour: DEFAULT_REMINDER_HOUR_IST, minute: DEFAULT_REMINDER_MIN_IST };
      const allowedDays = settings && settings.weekdays.length
        ? settings.weekdays
        : DEFAULT_WEEKDAYS;
      if (!allowedDays.includes(now.weekday)) {
        setShow(false);
        return;
      }

      const targetTotal = targetHour * 60 + targetMinute;
      const nowTotal = now.hour * 60 + now.minute;
      if (nowTotal < targetTotal) {
        setShow(false);
        return;
      }

      let status: DSRTodayStatus | null = null;
      try {
        const res = await dsr.todayStatus();
        status = res.data;
      } catch {
        return;
      }
      if (cancelled || !status) return;
      const ymd = now.ymd;

      if (!status.needs_dsr) {
        setShow(false);
        return;
      }

      const dismissedFor = localStorage.getItem(STORAGE_KEY_BANNER);
      if (dismissedFor !== ymd) {
        setDateLabel(formatIstDateLabel(status.today_ist || ymd));
        setShow(true);
      }

      // Belt-and-braces: drop today's DSR reminder into the in-app inbox so
      // the bell badge lights up even if the server-side 5 PM scheduler
      // missed this user (offline server, late login, etc.). Backend is
      // idempotent per IST day; we still gate by localStorage to avoid an
      // extra request every 60 s once we've done it.
      const inboxFor = localStorage.getItem(STORAGE_KEY_INBOX);
      if (inboxFor !== ymd) {
        try {
          const res = await dsr.notifyMe();
          localStorage.setItem(STORAGE_KEY_INBOX, ymd);
          if (res.data?.created) {
            // Nudge NotificationBell to re-fetch immediately rather than
            // wait up to 60 s for its next poll.
            try {
              window.dispatchEvent(new CustomEvent("softwiz:notif-refresh"));
            } catch {
              // older browsers: bell will catch up on its 60 s timer
            }
          }
        } catch {
          // older backend without /notify-me — fail silently, the 5 PM
          // server scheduler still covers the inbox row.
        }
      }

      const browserFor = localStorage.getItem(STORAGE_KEY_BROWSER);
      if (
        browserFor !== ymd &&
        "Notification" in window &&
        Notification.permission === "granted"
      ) {
        try {
          const n = new Notification("Submit your DSR before leaving", {
            body: "Reminder: please submit today's Daily Status Report.",
            icon: "/favicon.ico",
            tag: `dsr-reminder-${ymd}`,
          });
          n.onclick = () => {
            try {
              window.focus();
            } catch {
              // ignore
            }
            window.location.assign("/dsr");
          };
          localStorage.setItem(STORAGE_KEY_BROWSER, ymd);
        } catch {
          // browser may block in background tabs — that's fine.
        }
      }
    };

    evaluate();
    const t = window.setInterval(evaluate, POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(t);
    };
  }, [token]);

  if (!show) return null;

  const dismiss = () => {
    localStorage.setItem(STORAGE_KEY_BANNER, lastCheckedYmd.current || istNow().ymd);
    setShow(false);
  };

  return (
    <div
      role="alert"
      aria-live="polite"
      style={{
        position: "fixed",
        bottom: 24,
        right: 24,
        zIndex: 1100,
        maxWidth: 360,
        padding: "1rem 1.1rem",
        borderRadius: 14,
        background: "linear-gradient(135deg, rgba(99,102,241,0.96), rgba(168,85,247,0.96))",
        color: "#fff",
        boxShadow: "0 16px 40px rgba(0,0,0,0.35)",
        border: "1px solid rgba(255,255,255,0.16)",
        fontSize: "0.92rem",
        backdropFilter: "blur(6px)",
      }}
    >
      <div style={{ display: "flex", alignItems: "flex-start", gap: 12 }}>
        <div style={{ fontSize: "1.5rem", lineHeight: 1 }} aria-hidden>
          📝
        </div>
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontWeight: 700, marginBottom: 4 }}>
            Submit your DSR before leaving
          </div>
          <div style={{ opacity: 0.92, lineHeight: 1.45 }}>
            It's after 5 PM IST and today's Daily Status Report
            {dateLabel ? ` (${dateLabel})` : ""} isn't submitted yet.
          </div>
          <div style={{ marginTop: 12, display: "flex", gap: 8, flexWrap: "wrap" }}>
            <Link
              to="/dsr"
              onClick={dismiss}
              style={{
                background: "#fff",
                color: "#4338ca",
                fontWeight: 700,
                padding: "0.45rem 0.9rem",
                borderRadius: 8,
                textDecoration: "none",
                fontSize: "0.85rem",
              }}
            >
              Go to DSR
            </Link>
            <button
              type="button"
              onClick={dismiss}
              style={{
                background: "rgba(255,255,255,0.16)",
                color: "#fff",
                fontWeight: 600,
                padding: "0.45rem 0.9rem",
                borderRadius: 8,
                border: "1px solid rgba(255,255,255,0.25)",
                cursor: "pointer",
                fontSize: "0.85rem",
              }}
            >
              Dismiss for today
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function formatIstDateLabel(ymd: string): string {
  // Render "26 May 2026" without depending on the user's locale.
  const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(ymd);
  if (!m) return ymd;
  const [, y, mo, d] = m;
  const months = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
  ];
  const idx = Math.max(0, Math.min(11, parseInt(mo, 10) - 1));
  return `${parseInt(d, 10)} ${months[idx]} ${y}`;
}
