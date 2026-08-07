import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { auth as authApi } from "../api/client";
import { useAuth } from "../auth/AuthContext";
import logo from "../assets/New softwiz Logo.png";

/* The design replaces the source's bare fields with a leading lock glyph. */
const LockIcon = () => (
  <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" strokeLinecap="round" strokeLinejoin="round">
    <rect x="4" y="10.5" width="16" height="10.5" rx="2.5" />
    <path d="M8 10.5V7a4 4 0 0 1 8 0v3.5" />
  </svg>
);

export default function ChangePassword() {
  const navigate = useNavigate();
  const { markPasswordChanged } = useAuth();
  const [currentPassword, setCurrentPassword] = useState("");
  const [newPassword, setNewPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [error, setError] = useState("");
  const [success, setSuccess] = useState("");
  const [saving, setSaving] = useState(false);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault();
    setError("");
    setSuccess("");
    if (newPassword.length < 8) {
      setError("New password must be at least 8 characters.");
      return;
    }
    if (newPassword !== confirmPassword) {
      setError("New passwords do not match.");
      return;
    }
    setSaving(true);
    try {
      await authApi.changePassword(currentPassword, newPassword);
      // Clear the flag BEFORE navigating, otherwise PrivateRoute still sees
      // must_change_password === true and redirects right back here. The
      // endpoint re-sets the session cookie with pwd_change cleared, so the
      // Resume Analyzer gate stops blocking without the client touching a token.
      markPasswordChanged();
      setSuccess("Password changed successfully. Redirecting…");
      window.setTimeout(() => navigate("/", { replace: true }), 700);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Could not change password.");
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="eds eds-auth">
      <div className="eds-auth-card">
        <div className="eds-auth-head">
          <img src={logo} alt="Softwiz" />
          <div>
            <div className="eds-auth-title">Change your password</div>
            <p className="eds-auth-sub">Your temporary password must be replaced before continuing.</p>
          </div>
        </div>

        {error && <div className="alert alert-error">{error}</div>}
        {success && <div className="alert alert-success">{success}</div>}

        <form onSubmit={submit} className="eds-auth-form">
          <div className="eds-fieldset">
            <span className="eds-fieldset-label">Temporary password</span>
            <label className="eds-lockfield">
              <LockIcon />
              <input type="password" value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} required autoComplete="current-password" />
            </label>
          </div>
          <div className="eds-fieldset">
            <span className="eds-fieldset-label">New password</span>
            <label className="eds-lockfield">
              <LockIcon />
              <input type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} required minLength={8} autoComplete="new-password" />
            </label>
          </div>
          <div className="eds-fieldset">
            <span className="eds-fieldset-label">Confirm new password</span>
            <label className="eds-lockfield">
              <LockIcon />
              <input type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} required minLength={8} autoComplete="new-password" />
            </label>
          </div>
          <span className="eds-auth-note">Minimum 8 characters.</span>
          <button type="submit" className="eds-auth-submit" disabled={saving}>{saving ? "Saving…" : "Change password"}</button>
        </form>
      </div>
    </div>
  );
}
