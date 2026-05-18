import softwizLogo from '../assets/softwiz New Logo1 (1).png';
import { useState } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { auth } from "../api/client";


export default function Login() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const [showForgot, setShowForgot] = useState(false);
  const [forgotUser, setForgotUser] = useState("");
  const [forgotMsg, setForgotMsg] = useState("");
  const [forgotErr, setForgotErr] = useState("");
  const [forgotLoading, setForgotLoading] = useState(false);

  const handleForgotSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setForgotErr("");
    setForgotMsg("");
    setForgotLoading(true);
    try {
      const res = await auth.forgotPassword(forgotUser);
      setForgotMsg(res.data.detail);
      setForgotUser("");
    } catch (err: any) {
      setForgotErr(err.response?.data?.detail || "Error resetting password");
    } finally {
      setForgotLoading(false);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await login(username, password);
      navigate("/", { replace: true });
    } catch (err: any) {
      if (err.code === "ECONNABORTED" || err.message?.toLowerCase().includes("timeout")) {
        setError("Server is taking too long to respond. Please check your connection and try again.");
      } else if (!err.response) {
        setError("Cannot connect to server. Please ensure the backend is running.");
      } else if (err.response?.status === 401) {
        setError("Invalid username or password.");
      } else {
        setError(err.response?.data?.detail || "Login failed. Please try again.");
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <div className="login-box">
        <div className="login-brand">
          <div className="login-brand-logo" aria-hidden>
            <img src={softwizLogo} alt="" />
          </div>
          <div style={{ minWidth: 0 }}>
            <div className="login-brand-title">Attendance & HRMS</div>
            <div className="login-brand-sub text-muted">Sign in to your account</div>
          </div>
        </div>

        {error && <div className="alert alert-error">{error}</div>}

        <form onSubmit={handleSubmit} className="login-form">
          <div className="form-group">
            <label>Username</label>
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              required
              autoComplete="username"
              placeholder="Enter your username"
            />
          </div>
          <div className="form-group">
            <label>Password</label>
            <div className="password-field">
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                autoComplete="current-password"
                placeholder="Enter your password"
              />
              <button
                type="button"
                className="password-toggle"
                onClick={() => setShowPassword((v) => !v)}
                aria-label={showPassword ? "Hide password" : "Show password"}
                aria-pressed={showPassword}
              >
                {showPassword ? (
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M9.88 9.88a3 3 0 1 0 4.24 4.24"></path><path d="M10.73 5.08A10.43 10.43 0 0 1 12 5c7 0 10 7 10 7a13.16 13.16 0 0 1-1.67 2.68"></path><path d="M6.61 6.61A13.52 13.52 0 0 0 2 12s3 7 10 7a9.74 9.74 0 0 0 5.39-1.61"></path><line x1="2" y1="2" x2="22" y2="22"></line></svg>
                ) : (
                  <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"></path><circle cx="12" cy="12" r="3"></circle></svg>
                )}
              </button>
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '0.4rem' }}>
              <button 
                type="button" 
                onClick={() => setShowForgot(true)}
                style={{ background: 'none', border: 'none', color: 'var(--brand-400)', fontSize: '0.85rem', cursor: 'pointer', padding: 0 }}
              >
                Forgot password?
              </button>
            </div>
          </div>
          <button type="submit" className="btn btn-primary login-submit" disabled={loading}>
            {loading ? "Signing in..." : "Sign in"}
          </button>
        </form>

        <div className="login-footer text-muted">
          No account yet? <Link to="/signup">Create first account</Link>
        </div>
      </div>

      {showForgot && (
        <div className="modal-backdrop" onClick={() => setShowForgot(false)}>
          <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 400 }}>
            <h3 style={{ marginTop: 0 }}>Reset Password</h3>
            <p className="text-muted" style={{ fontSize: "0.9rem", marginBottom: "1rem" }}>
              Enter your username. Your password will be securely reset to a default value.
            </p>
            {forgotMsg && <div className="alert alert-success" style={{ padding: "0.5rem", borderRadius: "4px", backgroundColor: "rgba(34, 197, 94, 0.1)", color: "#4ade80", border: "1px solid rgba(34, 197, 94, 0.2)", marginBottom: "1rem" }}>{forgotMsg}</div>}
            {forgotErr && <div className="alert alert-error" style={{ marginBottom: "1rem" }}>{forgotErr}</div>}
            <form onSubmit={handleForgotSubmit}>
              <div className="form-group">
                <label>Username</label>
                <input
                  type="text"
                  value={forgotUser}
                  onChange={(e) => setForgotUser(e.target.value)}
                  required
                  placeholder="Enter your username"
                />
              </div>
              <div style={{ display: "flex", justifyContent: "flex-end", gap: "0.5rem", marginTop: "1.5rem" }}>
                <button type="button" className="btn btn-secondary" onClick={() => setShowForgot(false)}>
                  Close
                </button>
                <button type="submit" className="btn btn-primary" disabled={forgotLoading}>
                  {forgotLoading ? "Resetting..." : "Reset Password"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
