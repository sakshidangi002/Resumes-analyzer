import { useEffect, useState } from "react";
import { attendance } from "../api/client";
import { SectionLoader } from "../components/LoadingState";

type EmployeeStatus = {
  employee_id: number;
  employee_code: string;
  employee_name: string;
  department: string | null;
  status: string;
  sign_in_time: string | null;
  sign_out_time: string | null;
  total_work_hours: number;
  total_break_hours: number;
  last_event_type: string | null;
  last_event_time: string | null;
  current_state: string;
};

type LiveStatusResponse = {
  currently_working: EmployeeStatus[];
  currently_outside: EmployeeStatus[];
  checked_out: EmployeeStatus[];
  absent: EmployeeStatus[];
  last_recognition_events: Array<{
    employee_id: number;
    employee_name: string;
    event_type: string;
    event_time: string;
    camera_id: string | null;
  }>;
  summary: {
    total_employees: number;
    currently_working_count: number;
    currently_outside_count: number;
    checked_out_count: number;
    absent_count: number;
  };
};

function formatTime12h(timeStr: string | null) {
  if (!timeStr) return "—";
  const parts = timeStr.split(":");
  if (parts.length < 2) return timeStr;
  const hh = Number(parts[0]);
  const mm = Number(parts[1]);
  const period = hh >= 12 ? "PM" : "AM";
  const hour12 = hh % 12 || 12;
  return `${hour12.toString().padStart(2, "0")}:${String(mm).padStart(2, "0")} ${period}`;
}

function formatDuration(hours: number) {
  if (hours === 0) return "0h";
  const totalMinutes = Math.round(hours * 60);
  const h = Math.floor(totalMinutes / 60);
  const m = totalMinutes % 60;
  if (h === 0) return `${m}m`;
  if (m === 0) return `${h}h`;
  return `${h}h ${m}m`;
}

export default function LiveAttendanceDashboard() {
  const [data, setData] = useState<LiveStatusResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = () => {
    setLoading(true);
    setError(null);
    attendance
      .liveStatus()
      .then((res) => {
        setData(res.data);
        setLoading(false);
      })
      .catch(() => {
        setError("Failed to load attendance data");
        setLoading(false);
      });
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  if (loading && !data) {
    return <SectionLoader />;
  }

  if (error) {
    return (
      <div style={{ textAlign: "center", padding: "3rem" }}>
        <div style={{ color: "#ef4444", marginBottom: "1rem" }}>{error}</div>
        <button
          onClick={fetchData}
          style={{
            padding: "0.5rem 1rem",
            background: "rgba(59,130,246,0.15)",
            color: "#60a5fa",
            border: "1px solid rgba(59,130,246,0.3)",
            borderRadius: "8px",
            cursor: "pointer",
          }}
        >
          Retry
        </button>
      </div>
    );
  }

  if (!data) return null;

  const StatCard = ({ title, count, color }: { title: string; count: number; color: string }) => (
    <div
      style={{
        padding: "1.25rem",
        borderRadius: "12px",
        background: `${color}10`,
        border: `1px solid ${color}30`,
        textAlign: "center",
      }}
    >
      <div style={{ fontSize: "0.8rem", color: "rgba(255,255,255,0.7)", marginBottom: "0.5rem", textTransform: "uppercase" }}>
        {title}
      </div>
      <div style={{ fontSize: "2rem", fontWeight: 700, color }}>{count}</div>
    </div>
  );

  const EmployeeRow = ({ emp }: { emp: EmployeeStatus }) => {
    const statusColors = {
      WORKING: "#22c55e",
      OUTSIDE: "#f59e0b",
      CHECKED_OUT: "#64748b",
      ABSENT: "#ef4444",
    };
    const color = statusColors[emp.current_state as keyof typeof statusColors] || "#64748b";

    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: "1rem",
          padding: "0.75rem",
          borderRadius: "8px",
          background: "rgba(255,255,255,0.03)",
          border: "1px solid rgba(255,255,255,0.08)",
        }}
      >
        <div
          style={{
            width: "8px",
            height: "8px",
            borderRadius: "50%",
            background: color,
            flexShrink: 0,
          }}
        />
        <div style={{ flex: 1, minWidth: 0 }}>
          <div style={{ fontWeight: 600, color: "rgba(255,255,255,0.9)", fontSize: "0.9rem" }}>
            {emp.employee_name}
          </div>
          <div style={{ fontSize: "0.8rem", color: "rgba(255,255,255,0.6)" }}>
            {emp.employee_code} • {emp.department || "No Department"}
          </div>
        </div>
        <div style={{ textAlign: "right", whiteSpace: "nowrap" }}>
          <div style={{ fontSize: "0.85rem", color: "rgba(255,255,255,0.8)" }}>
            {formatTime12h(emp.sign_in_time)}
          </div>
          <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.5)" }}>
            {formatDuration(emp.total_work_hours)}
          </div>
        </div>
      </div>
    );
  };

  return (
    <div style={{ maxWidth: "1400px", margin: "0 auto", padding: "0 1.5rem" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.5rem" }}>
        <div>
          <h1 style={{ margin: 0, fontSize: "1.75rem" }}>Live Attendance Dashboard</h1>
          <p style={{ margin: "0.25rem 0 0 0", color: "rgba(255,255,255,0.6)", fontSize: "0.9rem" }}>
            Real-time employee attendance status
          </p>
        </div>
        <button
          onClick={fetchData}
          style={{
            padding: "0.5rem 1rem",
            background: "rgba(59,130,246,0.15)",
            color: "#60a5fa",
            border: "1px solid rgba(59,130,246,0.3)",
            borderRadius: "8px",
            cursor: "pointer",
          }}
        >
          Refresh
        </button>
      </div>

      {/* Summary Stats */}
      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: "1rem", marginBottom: "2rem" }}>
        <StatCard title="Currently Working" count={data.summary.currently_working_count} color="#22c55e" />
        <StatCard title="Currently Outside" count={data.summary.currently_outside_count} color="#f59e0b" />
        <StatCard title="Checked Out" count={data.summary.checked_out_count} color="#64748b" />
        <StatCard title="Absent" count={data.summary.absent_count} color="#ef4444" />
      </div>

      <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(400px, 1fr))", gap: "1.5rem" }}>
        {/* Currently Working */}
        <div
          style={{
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: "12px",
            padding: "1.25rem",
          }}
        >
          <h3 style={{ margin: "0 0 1rem 0", fontSize: "1.1rem", color: "#22c55e" }}>
            Currently Working ({data.currently_working.length})
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", maxHeight: "400px", overflowY: "auto" }}>
            {data.currently_working.length === 0 ? (
              <div style={{ opacity: 0.6, textAlign: "center", padding: "1rem" }}>No employees currently working</div>
            ) : (
              data.currently_working.map((emp) => <EmployeeRow key={emp.employee_id} emp={emp} />)
            )}
          </div>
        </div>

        {/* Currently Outside */}
        <div
          style={{
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: "12px",
            padding: "1.25rem",
          }}
        >
          <h3 style={{ margin: "0 0 1rem 0", fontSize: "1.1rem", color: "#f59e0b" }}>
            Currently Outside ({data.currently_outside.length})
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", maxHeight: "400px", overflowY: "auto" }}>
            {data.currently_outside.length === 0 ? (
              <div style={{ opacity: 0.6, textAlign: "center", padding: "1rem" }}>No employees currently outside</div>
            ) : (
              data.currently_outside.map((emp) => <EmployeeRow key={emp.employee_id} emp={emp} />)
            )}
          </div>
        </div>

        {/* Checked Out */}
        <div
          style={{
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: "12px",
            padding: "1.25rem",
          }}
        >
          <h3 style={{ margin: "0 0 1rem 0", fontSize: "1.1rem", color: "#64748b" }}>
            Checked Out ({data.checked_out.length})
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", maxHeight: "400px", overflowY: "auto" }}>
            {data.checked_out.length === 0 ? (
              <div style={{ opacity: 0.6, textAlign: "center", padding: "1rem" }}>No employees checked out</div>
            ) : (
              data.checked_out.map((emp) => <EmployeeRow key={emp.employee_id} emp={emp} />)
            )}
          </div>
        </div>

        {/* Absent */}
        <div
          style={{
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: "12px",
            padding: "1.25rem",
          }}
        >
          <h3 style={{ margin: "0 0 1rem 0", fontSize: "1.1rem", color: "#ef4444" }}>
            Absent ({data.absent.length})
          </h3>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem", maxHeight: "400px", overflowY: "auto" }}>
            {data.absent.length === 0 ? (
              <div style={{ opacity: 0.6, textAlign: "center", padding: "1rem" }}>No absent employees</div>
            ) : (
              data.absent.map((emp) => <EmployeeRow key={emp.employee_id} emp={emp} />)
            )}
          </div>
        </div>
      </div>

      {/* Last Recognition Events */}
      {data.last_recognition_events.length > 0 && (
        <div
          style={{
            marginTop: "1.5rem",
            background: "rgba(255,255,255,0.04)",
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: "12px",
            padding: "1.25rem",
          }}
        >
          <h3 style={{ margin: "0 0 1rem 0", fontSize: "1.1rem" }}>Last Recognition Events</h3>
          <div style={{ display: "flex", flexDirection: "column", gap: "0.5rem" }}>
            {data.last_recognition_events.map((event, idx) => (
              <div
                key={idx}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: "1rem",
                  padding: "0.75rem",
                  borderRadius: "8px",
                  background: "rgba(255,255,255,0.03)",
                }}
              >
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 600, color: "rgba(255,255,255,0.9)" }}>{event.employee_name}</div>
                  <div style={{ fontSize: "0.8rem", color: "rgba(255,255,255,0.6)" }}>
                    {event.camera_id || "Unknown Camera"}
                  </div>
                </div>
                <div style={{ textAlign: "right" }}>
                  <div
                    style={{
                      fontSize: "0.85rem",
                      fontWeight: 600,
                      color: event.event_type === "IN" ? "#22c55e" : "#ef4444",
                    }}
                  >
                    {event.event_type}
                  </div>
                  <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.6)" }}>
                    {new Date(event.event_time).toLocaleTimeString("en-IN", {
                      hour: "2-digit",
                      minute: "2-digit",
                      hour12: true,
                    })}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
