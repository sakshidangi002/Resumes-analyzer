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
    const t = window.setInterval(load, 60_000);
    return () => window.clearInterval(t);
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
