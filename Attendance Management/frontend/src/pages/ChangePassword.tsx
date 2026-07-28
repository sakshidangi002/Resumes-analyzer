import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { auth as authApi } from "../api/client";
import { useAuth } from "../auth/AuthContext";

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
    <div className="login-page">
      <div className="login-box">
        <h1 className="page-title">Change your password</h1>
        <p className="text-muted">Your temporary password must be replaced before continuing.</p>
        {error && <div className="alert alert-error">{error}</div>}
        {success && <div className="alert alert-success">{success}</div>}
        <form onSubmit={submit} className="login-form">
          <div className="form-group"><label>Temporary password</label><input type="password" value={currentPassword} onChange={(e) => setCurrentPassword(e.target.value)} required autoComplete="current-password" /></div>
          <div className="form-group"><label>New password</label><input type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} required minLength={8} autoComplete="new-password" /></div>
          <div className="form-group"><label>Confirm new password</label><input type="password" value={confirmPassword} onChange={(e) => setConfirmPassword(e.target.value)} required minLength={8} autoComplete="new-password" /></div>
          <button type="submit" className="btn btn-primary login-submit" disabled={saving}>{saving ? "Saving…" : "Change password"}</button>
        </form>
      </div>
    </div>
  );
}
