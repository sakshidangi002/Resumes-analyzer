import { lazy } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { useAuth } from "./auth/AuthContext";
import { AppLoadingScreen } from "./components/LoadingState";
// Eager: auth entry pages + the app shell (needed immediately).
import Login from "./pages/Login";
import Signup from "./pages/Signup";
import DashboardLayout from "./layouts/DashboardLayout";

// Lazy: every authenticated page is code-split so its JS is fetched only when
// the route is visited, keeping the initial bundle small. The <Suspense>
// boundary lives in DashboardLayout (around <Outlet/>), so the sidebar stays
// visible with a loader in the content area while a page chunk downloads.
const Dashboard = lazy(() => import("./pages/Dashboard"));
const Employees = lazy(() => import("./pages/Employees"));
const EmployeeProfile = lazy(() => import("./pages/EmployeeProfile"));
const ManageUsers = lazy(() => import("./pages/ManageUsers"));
const DepartmentsDesignations = lazy(() => import("./pages/DepartmentsDesignations"));
const Attendance = lazy(() => import("./pages/Attendance"));
const FaceDetection = lazy(() => import("./pages/FaceDetection"));
const CctvAttendance = lazy(() => import("./pages/CctvAttendance"));
const CctvCameraManager = lazy(() => import("./pages/CctvCameraManager"));
const DvrCameraDashboard = lazy(() => import("./pages/DvrCameraDashboard"));
const Leave = lazy(() => import("./pages/Leave"));
const LeaveApprovals = lazy(() => import("./pages/LeaveApprovals"));
const LeaveAllocations = lazy(() => import("./pages/LeaveAllocations"));
const Payroll = lazy(() => import("./pages/Payroll"));
const PayrollManagement = lazy(() => import("./pages/PayrollManagement"));
const PayslipManagement = lazy(() => import("./pages/PayslipManagement"));
const MyPayslips = lazy(() => import("./pages/MyPayslips"));
const Letters = lazy(() => import("./pages/Letters"));
const Reports = lazy(() => import("./pages/Reports"));
const Calendar = lazy(() => import("./pages/Calendar"));
const Notifications = lazy(() => import("./pages/Notifications"));
const Inbox = lazy(() => import("./pages/Inbox"));
const Onboarding = lazy(() => import("./pages/Onboarding"));
const DSR = lazy(() => import("./pages/DSR"));
const Policies = lazy(() => import("./pages/Policies"));

function PrivateRoute({ children }: { children: React.ReactNode }) {
  const { token, loading } = useAuth();
  if (loading) return <AppLoadingScreen />;
  if (!token) return <Navigate to="/login" replace />;
  return <>{children}</>;
}

/** Restrict a route to specific roles; others are bounced to the dashboard. */
function RoleRoute({ roles, children }: { roles: string[]; children: React.ReactNode }) {
  const { hasRole, loading } = useAuth();
  if (loading) return <AppLoadingScreen />;
  if (!hasRole(...roles)) return <Navigate to="/" replace />;
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
        <Route path="face-detection" element={<FaceDetection />} />
        <Route path="cctv-attendance" element={<RoleRoute roles={["Admin", "HR"]}><CctvAttendance /></RoleRoute>} />
        <Route path="cctv-cameras" element={<RoleRoute roles={["Admin", "HR"]}><CctvCameraManager /></RoleRoute>} />
        <Route path="dvr-cameras" element={<RoleRoute roles={["Admin", "HR"]}><DvrCameraDashboard /></RoleRoute>} />
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
        <Route path="policies" element={<Policies />} />
        <Route path="onboarding" element={<Onboarding />} />
        <Route path="dsr" element={<DSR />} />
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
