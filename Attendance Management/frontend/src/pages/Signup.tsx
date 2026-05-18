import { useState, useEffect } from "react";
import { useNavigate, Link } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import { auth as authApi } from "../api/client";

export default function Signup() {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [email, setEmail] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [signupAllowed, setSignupAllowed] = useState<boolean | null>(null);
  const { signup } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    authApi.canSignup().then((r) => setSignupAllowed(r.data.allowed)).catch(() => setSignupAllowed(false));
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    if (password !== confirmPassword) {
      setError("Passwords do not match");
      return;
    }
    if (password.length < 6) {
      setError("Password must be at least 6 characters");
      return;
    }
    setLoading(true);
    try {
      await signup(username, password, email || undefined);
      navigate("/", { replace: true });
    } catch (err: unknown) {
      const detail = (err as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
      setError(detail || "Signup failed");
    } finally {
      setLoading(false);
    }
  };

  if (signupAllowed === null) {
    return (
      <div className="login-page">
        <div className="login-box">Checking...</div>
      </div>
    );
  }

  if (signupAllowed === false) {
    return (
      <div className="login-page">
        <div className="login-box">
          <h1>Attendance & HRMS</h1>
          <p className="text-muted">Signup is only available when no users exist (first-time setup).</p>
          <Link to="/login" className="btn btn-primary">Go to Login</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="login-page">
      <div className="login-box">
        <h1>Attendance & HRMS</h1>
      <p className="text-muted mb-2">Create the first account (Admin)</p>
      {error && <div className="alert alert-error">{error}</div>}
      <form onSubmit={handleSubmit}>
        <div className="form-group">
          <label>Username</label>
          <input
            type="text"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
            autoComplete="username"
          />
        </div>
        <div className="form-group">
          <label>Email (optional)</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="email"
          />
        </div>
        <div className="form-group">
          <label>Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            minLength={6}
            autoComplete="new-password"
          />
        </div>
        <div className="form-group">
          <label>Confirm password</label>
          <input
            type="password"
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            required
            minLength={6}
            autoComplete="new-password"
          />
        </div>
        <button type="submit" className="btn btn-primary" disabled={loading} style={{ width: "100%" }}>
          {loading ? "Creating account..." : "Create account"}
        </button>
      </form>
      <p className="text-muted mt-2" style={{ textAlign: "center" }}>
        Already have an account? <Link to="/login">Sign in</Link>
      </p>
    </div>
    </div>
  );
}
