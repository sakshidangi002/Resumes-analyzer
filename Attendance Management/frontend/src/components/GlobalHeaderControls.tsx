import NotificationBell from './NotificationBell';
import { useAuth } from "../auth/AuthContext";
import { useState } from "react";
import ConfirmModal from "./ConfirmModal";

const Icons = {
  Logout: () => (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4"></path>
      <polyline points="16 17 21 12 16 7"></polyline>
      <line x1="21" y1="12" x2="9" y2="12"></line>
    </svg>
  ),
};

export default function GlobalHeaderControls() {
  const { logout } = useAuth();
  const [showLogoutConfirm, setShowLogoutConfirm] = useState(false);

  const handleLogout = () => {
    logout();
  };

  return (
    <div className="global-header-controls-inner" style={{ display: "flex", alignItems: "center", gap: "1.25rem" }}>
      <NotificationBell />
      <button
        type="button"
        onClick={() => setShowLogoutConfirm(true)}
        title="Logout"
        aria-label="Logout"
        className="btn btn-secondary btn-icon btn-sm"
        style={{ width: "44px", height: "44px", borderRadius: "10px", background: "rgba(255,255,255,0.05)", border: "1px solid rgba(255,255,255,0.1)" }}
      >
        <Icons.Logout />
      </button>

      <ConfirmModal
        isOpen={showLogoutConfirm}
        onClose={() => setShowLogoutConfirm(false)}
        onConfirm={handleLogout}
        title="Sign out?"
        message="You will be logged out of the dashboard."
        confirmText="Yes, Logout"
        cancelText="Cancel"
        variant="warning"
      />
    </div>
  );
}
