import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { activity } from "../api/client";

export default function NotificationBell() {
  const [count, setCount] = useState(0);

  const load = () => {
    activity
      .unreadCount()
      .then((r) => setCount(r.data.count))
      .catch(() => setCount(0));
  };

  useEffect(() => {
    load();
    // 60 s safety-net poll. With the WebSocket connected this is almost
    // never the trigger, but we keep it so the badge still updates if the
    // socket is closed by an aggressive proxy.
    const t = window.setInterval(load, 60_000);
    const onRefresh = () => load();
    // Realtime push from /ws/notifications — the freshest possible signal.
    const onRealtime = () => load();
    // Components that just created a new notification (e.g. the DSR reminder
    // banner calling /notify-me) dispatch this event so the badge updates
    // immediately rather than after the next 60 s poll.
    window.addEventListener("softwiz:notif-refresh", onRefresh);
    window.addEventListener("softwiz:notif-realtime", onRealtime);
    // Inbox tab in another window/tab may mark items read; resync on focus.
    window.addEventListener("focus", onRefresh);
    // The service worker forwards every Web Push to open tabs so we can
    // refresh the bell badge the moment a leave / letter / task / event /
    // DSR push lands — without waiting for the 60 s timer.
    const onSwMessage = (e: MessageEvent) => {
      if (e && e.data && e.data.type === "softwiz:notif-refresh") load();
    };
    navigator.serviceWorker?.addEventListener("message", onSwMessage);
    return () => {
      window.clearInterval(t);
      window.removeEventListener("softwiz:notif-refresh", onRefresh);
      window.removeEventListener("softwiz:notif-realtime", onRealtime);
      window.removeEventListener("focus", onRefresh);
      navigator.serviceWorker?.removeEventListener("message", onSwMessage);
    };
  }, []);

  return (
    <Link
      to="/inbox"
      title="In-app notifications"
      style={{
        position: "relative",
        display: "inline-flex",
        alignItems: "center",
        justifyContent: "center",
        width: 44,
        height: 44,
        borderRadius: 10,
        background: "rgba(255,255,255,0.08)",
        border: "1px solid rgba(255,255,255,0.12)",
        color: "#e2e8f0",
        textDecoration: "none",
        fontSize: "1.15rem",
      }}
    >
      <span aria-hidden>🔔</span>
      {count > 0 && (
        <span
          style={{
            position: "absolute",
            top: -4,
            right: -4,
            minWidth: 18,
            height: 18,
            padding: "0 5px",
            borderRadius: 999,
            background: "var(--brand-500)",
            color: "#fff",
            fontSize: "0.65rem",
            fontWeight: 800,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            lineHeight: 1,
          }}
        >
          {count > 99 ? "99+" : count}
        </span>
      )}
    </Link>
  );
}
