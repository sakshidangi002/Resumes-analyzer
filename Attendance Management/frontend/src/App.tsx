import { Routes, Route, Navigate } from "react-router-dom";
import { useAuth } from "./auth/AuthContext";
import { AppLoadingScreen } from "./components/LoadingState";
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import DashboardLayout from "./layouts/DashboardLayout";
import Dashboard from "./pages/Dashboard";
import Employees from "./pages/Employees";
import EmployeeProfile from "./pages/EmployeeProfile";
import ManageUsers from "./pages/ManageUsers";
import DepartmentsDesignations from "./pages/DepartmentsDesignations";
import Attendance from "./pages/Attendance";
import Leave from "./pages/Leave";
import LeaveApprovals from "./pages/LeaveApprovals";
import LeaveAllocations from "./pages/LeaveAllocations";
import Payroll from "./pages/Payroll";
import PayrollManagement from "./pages/PayrollManagement";
import PayslipManagement from "./pages/PayslipManagement";
import MyPayslips from "./pages/MyPayslips";
import Letters from "./pages/Letters";
import Reports from "./pages/Reports";
import Calendar from "./pages/Calendar";
import Notifications from "./pages/Notifications";
import Inbox from "./pages/Inbox";
import Onboarding from "./pages/Onboarding";
import DSR from "./pages/DSR";

function PrivateRoute({ children }: { children: React.ReactNode }) {
  const { token, loading } = useAuth();
  if (loading) return <AppLoadingScreen />;
  if (!token) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/signup" element={<Signup />} />
      <Route
        path="/"
        element={
          <PrivateRoute>
            <DashboardLayout />
          </PrivateRoute>
        }
      >
        <Route index element={<Dashboard />} />
        <Route path="employees" element={<Employees />} />
        <Route path="employees/:id" element={<EmployeeProfile />} />
        <Route path="my-profile" element={<EmployeeProfile />} />
        <Route path="departments-designations" element={<DepartmentsDesignations />} />
        <Route path="users" element={<ManageUsers />} />
        <Route path="attendance" element={<Attendance />} />
        <Route path="leave" element={<Leave />} />
        <Route path="leave-approvals" element={<LeaveApprovals />} />
        <Route path="leave-allocations" element={<LeaveAllocations />} />
        <Route path="payroll" element={<Payroll />} />
        <Route path="payroll-management" element={<PayrollManagement />} />
        <Route path="payslip-management" element={<PayslipManagement />} />
        <Route path="my-payslips" element={<MyPayslips />} />
        <Route path="letters" element={<Letters />} />
        <Route path="my-letters" element={<Letters forceEmployeeView />} />
        <Route path="reports" element={<Reports />} />
        <Route path="calendar" element={<Calendar />} />
        <Route path="notifications" element={<Notifications />} />
        <Route path="inbox" element={<Inbox />} />
        <Route path="onboarding" element={<Onboarding />} />
        <Route path="dsr" element={<DSR />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
