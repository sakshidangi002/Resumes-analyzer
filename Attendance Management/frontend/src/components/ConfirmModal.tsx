import React from "react";

interface ConfirmModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title?: string;
  message: React.ReactNode;
  confirmText?: string;
  cancelText?: string;
  variant?: "danger" | "warning";
  isLoading?: boolean;
}

export default function ConfirmModal({
  isOpen,
  onClose,
  onConfirm,
  title = "Are you absolutely sure?",
  message,
  confirmText = "Yes, Delete",
  cancelText = "Cancel",
  variant = "danger",
  isLoading = false,
}: ConfirmModalProps) {
  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" style={{ zIndex: 3000 }}>
      <div className="modal-confirm">
        <div className={`modal-confirm-icon ${variant === "danger" ? "text-danger" : "text-warning"}`}>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            width="32"
            height="32"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          >
            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
            <line x1="12" y1="9" x2="12" y2="13"></line>
            <line x1="12" y1="17" x2="12.01" y2="17"></line>
          </svg>
        </div>
        <div className="modal-confirm-title">{title}</div>
        <div className="modal-confirm-text">{message}</div>
        <div className="modal-confirm-actions">
          <button
            type="button"
            className={`btn ${variant === "danger" ? "btn-danger" : "btn-primary"}`}
            disabled={isLoading}
            onClick={() => {
              onConfirm();
            }}
          >
            {isLoading ? "Processing..." : confirmText}
          </button>
          <button type="button" className="btn btn-cancel-alt" onClick={onClose} disabled={isLoading}>
            {cancelText}
          </button>
        </div>
      </div>
    </div>
  );
}
