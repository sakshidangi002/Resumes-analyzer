import logo from '../assets/New softwiz Logo.png';
import { useState } from "react";
import { Outlet, NavLink } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import DsrReminderBanner from "../components/DsrReminderBanner";
import ToastHost from "../components/ToastHost";

// Premium SVG Icons
const Icons = {
  Dashboard: () => <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="3" width="7" height="7"></rect><rect x="14" y="3" width="7" height="7"></rect><rect x="14" y="14" width="7" height="7"></rect><rect x="3" y="14" width="7" height="7"></rect></svg>,
  Attendance: () => <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M12 20v-6M6 20V10M18 20V4"></path></svg>,
  FaceDetection: () => <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M4 8c0-2.2 1.8-4 4-4h8c2.2 0 4 1.8 4 4v8c0 2.2-1.8 4-4 4H8c-2.2 0-4-1.8-4-4V8z"></path><path d="M9 10h0"></path><path d="M15 10h0"></path><path d="M9 15c1 .8 2 .8 3 .8s2-.3 3-.8"></path></svg>,
  Profile: () => <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>,
  Employees: () => (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2" />
      <circle cx="12" cy="7" r="4" />
    </svg>
  ),
  Payroll: () => <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="2" y="4" width="20" height="16" rx="2"></rect><line x1="2" y1="10" x2="22" y2="10"></line></svg>,
  Leave: () => <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"></path><circle cx="9" cy="7" r="4"></circle><polyline points="16 11 18 13 22 9"></polyline></svg>,
  Organization: () => <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M22 10v6M2 10l10-5 10 5-10 5z"></path><path d="M6 12v5c3 3 9 3 12 0v-5"></path></svg>,
  Calendar: () => <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect><line x1="16" y1="2" x2="16" y2="6"></line><line x1="8" y1="2" x2="8" y2="6"></line><line x1="3" y1="10" x2="21" y2="10"></line></svg>,
  Inbox: () => <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><polyline points="22 12 16 12 14 15 10 15 8 12 2 12"></polyline><path d="M5.45 5.11L2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z"></path></svg>,
  Documents: () => <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line></svg>,
  PayrollMgmt: () => <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>,
  Reports: () => <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>,
  Tasks: () => <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 11l3 3L22 4"></path><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"></path></svg>,
  DSR: () => <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><rect x="4" y="3" width="16" height="18" rx="2"></rect><line x1="8" y1="8" x2="16" y2="8"></line><line x1="8" y1="12" x2="16" y2="12"></line><line x1="8" y1="16" x2="13" y2="16"></line></svg>,
  Logout: () => <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M10 17l5-5-5-5"></path><path d="M15 12H3"></path><path d="M21 3v18"></path></svg>,
};

export default function DashboardLayout() {
  const { hasRole, token } = useAuth();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const isEmployeeOnly = hasRole("Employee") && !hasRole("Admin") && !hasRole("HR") && !hasRole("Manager");

  const toggleSidebar = () => setIsSidebarOpen(!isSidebarOpen);
  const closeSidebar = () => setIsSidebarOpen(false);

  return (
    <div className="app-layout">



      {/* Mobile Top Header */}
      <header className="mobile-header">
        <button className="hamburger" onClick={toggleSidebar} aria-label="Toggle menu">
          <span></span>
          <span></span>
          <span></span>
        </button>
        <div className="mobile-brand-name"><img src={logo} alt="Logo" style={{ height: "24px", objectFit: "contain" }} /></div>
      </header>

      {/* Sidebar Backdrop */}
      {isSidebarOpen && <div className="sidebar-backdrop" onClick={closeSidebar}></div>}

      <aside className={`sidebar ${isSidebarOpen ? "is-open" : ""}`}>
        <div className="sidebar-top">
          <div className="sidebar-brand">
            <img src={logo} alt="Softwiz Infotech" style={{ height: "32px", width: "auto", objectFit: "contain", filter: "drop-shadow(0 0 8px rgba(var(--brand-rgb) / 0.25))" }} />
          </div>
        </div>

        <nav className="sidebar-nav no-scrollbar">
          {/* Section 1: Main */}
          <div className="sidebar-section">
            <div className="sidebar-section-label">General</div>
            <NavLink to="/" end onClick={closeSidebar}>
              <Icons.Dashboard /> Dashboard
            </NavLink>
            <NavLink to="/attendance" onClick={closeSidebar}>
              <Icons.Attendance /> Attendance
            </NavLink>
            <NavLink to="/face-detection" onClick={closeSidebar}>
              <Icons.FaceDetection /> Face Detection
            </NavLink>
            <NavLink to="/calendar" onClick={closeSidebar}>
              <Icons.Calendar /> Calendar
            </NavLink>
            <NavLink to="/inbox" onClick={closeSidebar}>
              <Icons.Inbox /> Inbox
            </NavLink>
            <NavLink to="/onboarding" onClick={closeSidebar}>
              <Icons.Tasks /> Task Hub
            </NavLink>
            <NavLink to="/dsr" onClick={closeSidebar}>
              <Icons.DSR /> DSR
            </NavLink>
          </div>

          {/* Section 2: Administration */}
          {(hasRole("Admin") || hasRole("HR") || hasRole("Manager")) && (
            <div className="sidebar-section">
              <div className="sidebar-section-label">Administration</div>
              <NavLink to="/employees" onClick={closeSidebar}>
                <Icons.Employees /> Employees
              </NavLink>
              {hasRole("Admin") && (
                <NavLink to="/users" onClick={closeSidebar}>
                  <Icons.Profile /> Manage Users
                </NavLink>
              )}
              {(hasRole("Admin") || hasRole("HR")) && (
                <NavLink to="/departments-designations" onClick={closeSidebar}>
                  <Icons.Organization /> Departments
                </NavLink>
              )}
            </div>
          )}

          {/* Section 3: HR & Payroll Operations */}
          {(hasRole("Admin") || hasRole("HR")) && (
            <div className="sidebar-section">
              <div className="sidebar-section-label">Operations</div>
              <NavLink to="/leave-approvals" onClick={closeSidebar}>
                <Icons.Leave /> Leave Approvals
              </NavLink>
              <NavLink to="/leave-allocations" onClick={closeSidebar}>
                <Icons.Organization /> Leave Allocations
              </NavLink>
              <NavLink to="/payroll" onClick={closeSidebar}>
                <Icons.Payroll /> Payroll Periods
              </NavLink>
              <NavLink to="/payroll-management" onClick={closeSidebar}>
                <Icons.PayrollMgmt /> Payroll Setup
              </NavLink>
              <NavLink to="/payslip-management" onClick={closeSidebar}>
                <Icons.Documents /> Payslip Registry
              </NavLink>
              <NavLink to="/letters" onClick={closeSidebar}>
                <Icons.Documents /> Employee Letters
              </NavLink>
              <NavLink to="/reports" onClick={closeSidebar}>
                <Icons.Reports /> Full Reports
              </NavLink>
              <a
                href={`/resume/?token=${token}`}
                target="_blank"
                rel="noopener noreferrer"
                onClick={closeSidebar}
              >
                <Icons.Documents /> Resume Analyzer
              </a>
            </div>
          )}

          <div className="sidebar-divider"></div>

          {/* Section 4: Personal */}
          {!hasRole("Admin") && (
            <div className={`sidebar-section ${isEmployeeOnly ? "sidebar-section--compact" : ""}`}>
              <div className="sidebar-section-label">My Space</div>
              <NavLink to="/my-profile" onClick={closeSidebar}>
                <Icons.Profile /> My Profile
              </NavLink>
              <NavLink to="/leave" onClick={closeSidebar}>
                <Icons.Leave /> My Leave
              </NavLink>
              <NavLink to="/my-payslips" onClick={closeSidebar}>
                <Icons.Documents /> My Payslips
              </NavLink>
              <NavLink to="/my-letters" onClick={closeSidebar}>
                <Icons.Documents /> My Documents
              </NavLink>
            </div>
          )}
        </nav>
      </aside>

      <main className="main-content">
        <div className="content-shell">
          <Outlet />
        </div>
      </main>

      <DsrReminderBanner />
      <ToastHost />
    </div>
  );
}
