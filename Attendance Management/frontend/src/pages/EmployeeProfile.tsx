import { useEffect, useMemo, useState } from "react";
import { useParams, NavLink, useMatch } from "react-router-dom";
import { useAuth } from "../auth/AuthContext";
import CustomSelect from "../components/CustomSelect";
import {
  employees as employeesApi,
  departments as departmentsApi,
  designations as designationsApi,
  leave as leaveApi,
  payroll as payrollApi,
  company as companyApi,
  attendance as attendanceApi,
  recognition as recognitionApi,
} from "../api/client";
import { SectionLoader } from "../components/LoadingState";
import GlobalHeaderControls from "../components/GlobalHeaderControls";
import SalaryFormulaView from "../components/SalaryFormulaView";
import MonthlyAttendanceGrid from "../components/MonthlyAttendanceGrid";

type TabKey =
  | "about"
  | "attendance"
  | "attendance_details"
  | "leaves"
  | "salary"
  | "payslips";

const EMPLOYMENT_TYPES = ["Full-time", "Intern", "Contract"];
const EMPLOYMENT_STATUSES = ["Active", "Resigned", "Terminated"];
const MARITAL_STATUSES = ["Single", "Married", "Divorced", "Widowed"];

const maxDobDate = (() => {
  const d = new Date();
  d.setFullYear(d.getFullYear() - 18);
  return d.toISOString().split("T")[0];
})();

type ProfileForm = {
  employee_code: string;
  first_name: string;
  last_name: string;
  official_email: string;
  personal_email: string;
  phone: string;
  date_of_joining: string;
  date_of_birth: string;
  date_of_marriage: string;
  marital_status: string;
  date_of_leaving: string;
  designation_id: number | "";
  department_id: number | "";
  employment_type: string;
  employment_status: string;
  expected_working_hours: number;
  reporting_manager_id: number | "";
  pan_number: string;
  aadhar_number: string;
  passport_number: string;
  passport_expiry_date: string;
  driving_license_number: string;
  driving_license_expiry_date: string;
};

function empToForm(e: Emp): ProfileForm {
  return {
    employee_code: e.employee_code,
    first_name: e.first_name,
    last_name: e.last_name,
    official_email: e.official_email,
    personal_email: e.personal_email || "",
    phone: e.phone || "",
    date_of_joining: (e.date_of_joining || "").slice(0, 10),
    date_of_birth: (e.date_of_birth || "").slice(0, 10),
    date_of_marriage: (e.date_of_marriage || "").slice(0, 10),
    marital_status: e.marital_status || "",
    date_of_leaving: (e.date_of_leaving || "").slice(0, 10),
    designation_id: e.designation_id ?? "",
    department_id: e.department_id ?? "",
    employment_type: e.employment_type,
    employment_status: e.employment_status,
    expected_working_hours: e.expected_working_hours || 9.0,
    reporting_manager_id: e.reporting_manager_id ?? "",
    pan_number: e.pan_number || "",
    aadhar_number: e.aadhar_number || "",
    passport_number: e.passport_number || "",
    passport_expiry_date: (e.passport_expiry_date || "").slice(0, 10),
    driving_license_number: e.driving_license_number || "",
    driving_license_expiry_date: (e.driving_license_expiry_date || "").slice(0, 10),
  };
}

interface Emp {
  id: number;
  employee_code: string;
  first_name: string;
  last_name: string;
  official_email: string;
  personal_email?: string | null;
  phone?: string | null;
  date_of_joining: string;
  designation_id?: number | null;
  department_id?: number | null;
  reporting_manager_id?: number | null;
  reporting_manager?: {
    id: number;
    employee_code: string;
    first_name: string;
    last_name: string;
  } | null;
  employment_type: string;
  employment_status: string;
  expected_working_hours: number;
  date_of_birth?: string | null;
  date_of_marriage?: string | null;
  marital_status?: string | null;
  date_of_leaving?: string | null;
  pan_number?: string | null;
  aadhar_number?: string | null;
  passport_number?: string | null;
  passport_expiry_date?: string | null;
  driving_license_number?: string | null;
  driving_license_expiry_date?: string | null;
}

interface Dept {
  id: number;
  name: string;
}

interface Desig {
  id: number;
  title: string;
}

interface CompanyCfg {
  id: number;
  name: string;
}

interface FinancialYear {
  id: number;
  name?: string | null;
  start_date: string;
  end_date: string;
  is_current: boolean;
}

interface LeaveType {
  id: number;
  code: string;
  name: string;
}

interface Allocation {
  id: number;
  employee_id: number;
  financial_year_id: number;
  leave_type_id: number;
  allocated_days: number;
  used_days: number;
  balance_days: number;
}

interface LeaveRequestRow {
  id: number;
  employee_id: number;
  leave_type_id: number;
  start_date: string;
  end_date: string;
  is_half_day: boolean;
  reason?: string | null;
  status: string;
  rejection_reason?: string | null;
  applied_at: string;
}

interface SalaryStructure {
  id: number;
  employee_id: number;
  basic: number;
  hra: number;
  medical: number;
  travelling: number;
  miscellaneous: number;
  allowances: number;
  deductions: number;
  effective_from: string;
  effective_to?: string | null;
}

interface Payslip {
  id: number;
  employee_id: number;
  payroll_period_id: number;
  gross_salary: number;
  total_earnings: number;
  total_deductions: number;
  net_salary: number;
  paid_days: number;
  lop_days: number;
  component_breakdown?: string | null;
  generated_at: string;
}

const tabLabel: Record<TabKey, string> = {
  about: "About",
  attendance: "Attendance Grid",
  attendance_details: "Attendance Details",
  leaves: "Leaves",
  salary: "Salary",
  payslips: "Payslips",
};

function initials(first?: string, last?: string) {
  const a = (first || "").trim().slice(0, 1).toUpperCase();
  const b = (last || "").trim().slice(0, 1).toUpperCase();
  return (a + b) || "?";
}

function formatNiceDate(dateStr?: string | null) {
  if (!dateStr) return "-";
  const d = new Date(dateStr.length === 10 ? dateStr + "T12:00:00" : dateStr);
  if (isNaN(d.getTime())) return dateStr;
  const day = String(d.getDate()).padStart(2, "0");
  const month = String(d.getMonth() + 1).padStart(2, "0");
  const year = d.getFullYear();
  return `${day}/${month}/${year}`;
}

function formatCompactDuration(hours: number | null | undefined) {
  if (hours == null) return "—";
  const totalSeconds = Math.round(Number(hours) * 3600);
  const h = Math.floor(totalSeconds / 3600);
  const m = Math.floor((totalSeconds % 3600) / 60);
  const s = totalSeconds % 60;
  if (h <= 0 && m <= 0) return `${s}s`;
  if (h <= 0) return `${m}m ${s}s`;
  if (m <= 0 && s <= 0) return `${h}h`;
  if (s <= 0) return `${h}h ${m}m`;
  return `${h}h ${m}m ${s}s`;
}

function formatTime12h(timeStr: string | null | undefined) {
  if (!timeStr) return "—";
  const parts = timeStr.split(":");
  if (parts.length < 2) return timeStr;
  const hh = Number(parts[0]);
  const mm = Number(parts[1]);
  const period = hh >= 12 ? "PM" : "AM";
  const hour12 = hh % 12 || 12;
  return `${hour12.toString().padStart(2, "0")}:${String(mm).padStart(2, "0")} ${period}`;
}

/** Compact label/value for the profile hero strip only */
function InfoItem({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div style={{ minWidth: 0 }}>
      <div
        style={{
          fontSize: "0.75rem",
          color: "rgba(255, 255, 255, 0.72)",
          textTransform: "uppercase",
          letterSpacing: "0.04em",
          marginBottom: 4,
          lineHeight: 1.2,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: "0.95rem",
          fontWeight: 600,
          color: "rgba(255, 255, 255, 0.92)",
          lineHeight: 1.3,
          whiteSpace: "nowrap",
          overflow: "hidden",
          textOverflow: "ellipsis",
        }}
      >
        {value}
      </div>
    </div>
  );
}

function DetailField({
  label,
  value,
  variant = "default",
}: {
  label: string;
  value: React.ReactNode;
  variant?: "default" | "job" | "contact";
}) {
  const empty = value === null || value === undefined || value === "" || value === "-";
  const base =
    variant === "job"
      ? "emp-detail-field emp-detail-field--job"
      : variant === "contact"
        ? "emp-detail-field emp-detail-field--contact"
        : "emp-detail-field emp-detail-field--job";
  return (
    <div className={base}>
      <span className="emp-detail-field__label">{label}</span>
      <div className={empty ? "emp-detail-field__value emp-detail-field__value--muted" : "emp-detail-field__value"}>
        {empty ? "—" : value}
      </div>
    </div>
  );
}

export default function EmployeeProfile() {
  const { hasRole, user } = useAuth();
  const myProfileMatch = useMatch({ path: "my-profile", end: true });
  const isMyProfileRoute = Boolean(myProfileMatch);
  const { id } = useParams();
  const employeeId = isMyProfileRoute
    ? user?.employee_id != null
      ? user.employee_id
      : NaN
    : Number(id);

  const canEdit = hasRole("Admin") || hasRole("HR");
  const isSelf = user?.employee_id === employeeId;
  const canViewDocs = canEdit || (hasRole("Employee") && isSelf);
  const canViewProfile = useMemo(() => {
    if (!employeeId || Number.isNaN(employeeId)) return false;
    if (hasRole("Admin") || hasRole("HR") || hasRole("Manager")) return true;
    if (hasRole("Employee") && user?.employee_id === employeeId) return true;
    return false;
  }, [employeeId, user?.employee_id, user?.roles, hasRole]);

  const [tab, setTab] = useState<TabKey>("about");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [company, setCompany] = useState<CompanyCfg | null>(null);
  const [emp, setEmp] = useState<Emp | null>(null);
  const [depts, setDepts] = useState<Dept[]>([]);
  const [desigs, setDesigs] = useState<Desig[]>([]);
  // Leaves tab
  const [financialYears, setFinancialYears] = useState<FinancialYear[]>([]);
  const [selectedFyId, setSelectedFyId] = useState<number | null>(null);
  const [leaveTypes, setLeaveTypes] = useState<LeaveType[]>([]);
  const [allocs, setAllocs] = useState<Allocation[]>([]);
  const [leaveReqs, setLeaveReqs] = useState<LeaveRequestRow[]>([]);

  // Salary tab
  const [structures, setStructures] = useState<SalaryStructure[]>([]);

  // Payslips tab
  const [payslips, setPayslips] = useState<Payslip[]>([]);
  const [formulaPayslip, setFormulaPayslip] = useState<Payslip | null>(null);


  const [editing, setEditing] = useState(false);
  const [editingBank, setEditingBank] = useState(false);
  const [form, setForm] = useState<ProfileForm | null>(null);
  const [saveError, setSaveError] = useState("");
  const [saveSuccess, setSaveSuccess] = useState("");

  const canViewBank = canEdit || (hasRole("Employee") && isSelf);
  const [bankLoading, setBankLoading] = useState(false);
  const [bankError, setBankError] = useState("");
  const [bankSaved, setBankSaved] = useState("");
  const [bankForm, setBankForm] = useState<{
    bank_name: string;
    branch_name: string;
    account_holder_name: string;
    account_number: string;
    ifsc_code: string;
    account_type: string;
  } | null>(null);

  const [faceImages, setFaceImages] = useState<File[]>([]);
  const [faceUploadLoading, setFaceUploadLoading] = useState(false);
  const [faceUploadError, setFaceUploadError] = useState("");
  const [faceUploadSuccess, setFaceUploadSuccess] = useState("");
  
  const handleFaceUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!faceImages.length) return;
    if (!emp) return;
    if (faceImages.length > 5) {
      setFaceUploadError("You can upload a maximum of 5 images.");
      return;
    }
    setFaceUploadLoading(true);
    setFaceUploadError("");
    setFaceUploadSuccess("");
    try {
      await recognitionApi.registerFace(emp.id, `${emp.first_name} ${emp.last_name}`, deptName, faceImages);
      setFaceUploadSuccess("Face samples uploaded and registered successfully.");
      setFaceImages([]);
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      let message = "Face upload failed.";
      if (typeof detail === "string") {
        message = detail;
      } else if (Array.isArray(detail)) {
        message = detail.map((d: { msg?: string }) => d.msg || String(d)).join("; ");
      } else if (!err.response) {
        message = "Could not reach the HRMS server. Make sure the main backend is running and you are logged in as Admin or HR.";
      }
      setFaceUploadError(message);
    } finally {
      setFaceUploadLoading(false);
    }
  };

  const [attMonth, setAttMonth] = useState(() => new Date().getMonth() + 1);
  const [attYear, setAttYear] = useState(() => new Date().getFullYear());
  const [attRecords, setAttRecords] = useState<
    Array<{
      date: string;
      status: string;
      sign_in_time: string | null;
      sign_out_time: string | null;
      total_work_hours: number | null;
    }>
  >([]);
  const [attLoading, setAttLoading] = useState(false);

  const [selectedDetailDate, setSelectedDetailDate] = useState(() => {
    const d = new Date();
    const y = d.getFullYear();
    const m = String(d.getMonth() + 1).padStart(2, "0");
    const day = String(d.getDate()).padStart(2, "0");
    return `${y}-${m}-${day}`;
  });
  const [timelineData, setTimelineData] = useState<{
    first_check_in: string | null;
    last_check_out: string | null;
    total_work_hours: number | null;
    total_break_hours: number | null;
    status: string;
  } | null>(null);
  const [timelineLoading, setTimelineLoading] = useState(false);
  const [eventHistory, setEventHistory] = useState<
    Array<{
      id: number;
      event_time: string;
      event_type: string;
      source: string;
      attendance_date: string;
    }>
  >([]);
  const [eventsLoading, setEventsLoading] = useState(false);

  const handleDeleteEvent = (eventId: number) => {
    if (!confirm("Are you sure you want to delete this attendance event? This will recalculate break time and working hours.")) return;
    attendanceApi.deleteEvent(eventId)
      .then(() => {
        // Refresh timeline and events after deletion
        if (employeeId && selectedDetailDate) {
          attendanceApi.details(employeeId, selectedDetailDate)
            .then((res) => {
              setTimelineData({
                first_check_in: res.data.first_check_in || res.data.sign_in_time || null,
                last_check_out: res.data.last_check_out || res.data.sign_out_time || null,
                total_work_hours: res.data.total_work_hours,
                total_break_hours: res.data.total_break_hours,
                status: res.data.status,
              });
              setEventHistory(res.data.events || []);
            })
            .catch(() => {
              setTimelineData(null);
              setEventHistory([]);
            });
        }
      })
      .catch((err: unknown) => {
        console.error("Failed to delete event:", err);
        alert("Failed to delete event. Please try again.");
      });
  };

  const deptName = useMemo(() => {
    if (!emp?.department_id) return "-";
    return depts.find((d) => d.id === emp.department_id)?.name || "-";
  }, [emp?.department_id, depts]);

  const desigName = useMemo(() => {
    if (!emp?.designation_id) return "-";
    return desigs.find((d) => d.id === emp.designation_id)?.title || "-";
  }, [emp?.designation_id, desigs]);

  useEffect(() => {
    if (!canViewProfile) return;
    if (!employeeId || Number.isNaN(employeeId)) {
      setError("Invalid employee id.");
      setLoading(false);
      return;
    }

    setLoading(true);
    setError("");
    Promise.all([
      companyApi.config(),
      employeesApi.get(employeeId),
      departmentsApi.list(),
      designationsApi.list(),
      companyApi.financialYears(),
      leaveApi.types(),
    ])
      .then(([cRes, eRes, dRes, desRes, fyRes, ltRes]) => {
        setCompany(cRes.data);
        setEmp(eRes.data);
        setDepts(dRes.data || []);
        setDesigs(desRes.data || []);
        setFinancialYears(fyRes.data || []);
        setLeaveTypes(ltRes.data || []);
        const currentFy = (fyRes.data || []).find((fy: FinancialYear) => fy.is_current) || (fyRes.data || [])[0];
        setSelectedFyId(currentFy?.id ?? null);
      })
      .catch((err) => setError(err.response?.data?.detail || "Failed to load employee."))
      .finally(() => setLoading(false));


    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [employeeId, canViewProfile]);

  useEffect(() => {
    if (!canViewProfile || !canViewBank) return;
    if (!employeeId || Number.isNaN(employeeId)) return;
    setBankLoading(true);
    setBankError("");
    setBankSaved("");
    employeesApi
      .bankGet(employeeId)
      .then((res) => {
        const b = res.data;
        setBankForm({
          bank_name: b.bank_name || "",
          branch_name: b.branch_name || "",
          account_holder_name: b.account_holder_name || "",
          account_number: b.account_number || "",
          ifsc_code: b.ifsc_code || "",
          account_type: b.account_type || "Savings",
        });
      })
      .catch(() => {
        setBankForm({
          bank_name: "",
          branch_name: "",
          account_holder_name: "",
          account_number: "",
          ifsc_code: "",
          account_type: "Savings",
        });
      })
      .finally(() => setBankLoading(false));
  }, [employeeId, canViewProfile, canViewBank]);

  const handleSaveBank = (e: React.FormEvent) => {
    e.preventDefault();
    if (!bankForm) return;
    if (!employeeId || Number.isNaN(employeeId)) return;
    setBankLoading(true);
    setBankError("");
    setBankSaved("");
    const payload = {
      bank_name: (bankForm.bank_name || "").trim(),
      branch_name: (bankForm.branch_name || "").trim() || undefined,
      account_holder_name: (bankForm.account_holder_name || "").trim(),
      account_number: (bankForm.account_number || "").trim(),
      ifsc_code: (bankForm.ifsc_code || "").trim(),
      account_type: bankForm.account_type || "Savings",
    };
    employeesApi
      .bankUpdate(employeeId, payload)
      .then(() => {
        setBankSaved("Bank details saved.");
        setEditingBank(false);
      })
      .catch((err) => {
        const status = err?.response?.status;
        const data = err?.response?.data;
        const detail = data?.detail;
        const msg =
          typeof detail === "string"
            ? detail
            : Array.isArray(detail)
              ? detail.map((d: any) => d?.msg || JSON.stringify(d)).join(", ")
              : typeof data === "string"
                ? data
                : data
                  ? JSON.stringify(data)
                  : err?.message || "Failed to save bank details.";
        setBankError(status ? `${status}: ${msg}` : msg);
      })
      .finally(() => setBankLoading(false));
  };

  useEffect(() => {
    if (!canViewProfile || !employeeId || !selectedFyId) return;
    if (tab !== "leaves") return;
    Promise.all([
      leaveApi.allocations({ employee_id: employeeId, financial_year_id: selectedFyId }),
      leaveApi.requests({ employee_id: employeeId }),
    ])
      .then(([aRes, rRes]) => {
        setAllocs(aRes.data || []);
        setLeaveReqs(rRes.data || []);
      })
      .catch(() => {
        setAllocs([]);
        setLeaveReqs([]);
      });
  }, [tab, selectedFyId, canViewProfile, employeeId]);

  useEffect(() => {
    if (!canViewProfile || !employeeId) return;
    if (tab !== "salary") return;
    payrollApi
      .salaryStructures(employeeId)
      .then((res) => setStructures(res.data || []))
      .catch(() => setStructures([]));
  }, [tab, canViewProfile, employeeId]);

  useEffect(() => {
    if (!canViewProfile || !employeeId) return;
    if (tab !== "payslips") return;
    payrollApi
      .payslips({ employee_id: employeeId })
      .then((res) => setPayslips(res.data || []))
      .catch(() => setPayslips([]));
  }, [tab, canViewProfile, employeeId]);

  useEffect(() => {
    if (tab !== "about") {
      setEditing(false);
      setEditingBank(false);
    }
  }, [tab]);

  useEffect(() => {
    if (!canViewProfile || !employeeId || tab !== "attendance") return;
    setAttLoading(true);
    const daysInMonth = new Date(attYear, attMonth, 0).getDate();
    const from = `${attYear}-${String(attMonth).padStart(2, "0")}-01`;
    const to = `${attYear}-${String(attMonth).padStart(2, "0")}-${String(daysInMonth).padStart(2, "0")}`;
    attendanceApi
      .list(from, to, employeeId)
      .then((res) => setAttRecords(res.data || []))
      .catch(() => setAttRecords([]))
      .finally(() => setAttLoading(false));
  }, [tab, attMonth, attYear, employeeId, canViewProfile]);

  useEffect(() => {
    if (!canViewProfile || !employeeId || tab !== "attendance_details" || !selectedDetailDate) return;
    setTimelineLoading(true);
    attendanceApi
      .details(employeeId, selectedDetailDate)
      .then((res) => {
        setTimelineData({
          first_check_in: res.data.first_check_in || res.data.sign_in_time || null,
          last_check_out: res.data.last_check_out || res.data.sign_out_time || null,
          total_work_hours: res.data.total_work_hours,
          total_break_hours: res.data.total_break_hours,
          status: res.data.status || "ABSENT",
        });
      })
      .catch(() => setTimelineData(null))
      .finally(() => setTimelineLoading(false));
  }, [tab, selectedDetailDate, employeeId, canViewProfile]);

  useEffect(() => {
    if (!canViewProfile || !employeeId || tab !== "attendance_details") return;
    setEventsLoading(true);
    attendanceApi
      .listEvents(employeeId, selectedDetailDate || new Date().toISOString().split('T')[0])
      .then((res) => {
        setEventHistory(res.data || []);
      })
      .catch(() => setEventHistory([]))
      .finally(() => setEventsLoading(false));
  }, [tab, employeeId, canViewProfile, selectedDetailDate]);



  const handleSaveProfile = (e: React.FormEvent) => {
    e.preventDefault();
    if (!form || !emp) return;
    setSaveError("");
    setSaveSuccess("");
    const payload = {
      employee_code: form.employee_code,
      first_name: form.first_name,
      last_name: form.last_name,
      official_email: form.official_email,
      personal_email: form.personal_email.trim() || null,
      phone: form.phone.trim() || null,
      date_of_joining: form.date_of_joining,
      designation_id: form.designation_id === "" ? null : Number(form.designation_id),
      department_id: form.department_id === "" ? null : Number(form.department_id),
      employment_type: form.employment_type,
      employment_status: form.employment_status,
      expected_working_hours: Number(form.expected_working_hours) || 9.0,
      date_of_birth: form.date_of_birth.trim() || null,
      date_of_marriage: form.date_of_marriage.trim() || null,
      marital_status: form.marital_status.trim() || null,
      date_of_leaving: form.date_of_leaving.trim() || null,
      reporting_manager_id: form.reporting_manager_id === "" ? null : Number(form.reporting_manager_id),
      pan_number: form.pan_number.trim() || null,
      aadhar_number: form.aadhar_number.trim() || null,
    };

    // Validation: If date_of_leaving is set, status must be Resigned or Terminated
    if (payload.date_of_leaving) {
      if (payload.employment_status === "Active") {
        setSaveError("Please select either Resigned or Terminated when adding a date of leaving.");
        return;
      }
    }

    employeesApi
      .update(employeeId, payload)
      .then((res) => {
        setEmp(res.data as Emp);
        setEditing(false);
        setForm(null);
        setSaveSuccess("Profile updated.");
      })
      .catch((err) => setSaveError(err.response?.data?.detail || "Failed to save."));
  };

  if (isMyProfileRoute && user?.employee_id == null) {
    return (
      <div className="card">
        <p>
          Your account is not linked to an employee record, so your profile cannot be shown. Please ask HR to link your
          user to your employee ID.
        </p>
        <NavLink to="/" className="btn btn-secondary" style={{ marginTop: "0.75rem", display: "inline-block" }}>
          Back to dashboard
        </NavLink>
      </div>
    );
  }

  if (!canViewProfile) {
    return (
      <div className="card">
        <p>You don&apos;t have permission to view this employee profile.</p>
        <NavLink to="/employees" className="btn btn-secondary" style={{ marginTop: "0.75rem", display: "inline-block" }}>
          Back to Employees
        </NavLink>
      </div>
    );
  }

  if (loading) {
    return (
      <>
        <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <div>
            <h1 className="page-title">{isMyProfileRoute ? "My profile" : "Employee profile"}</h1>
            <div className="page-subtitle">
              {isMyProfileRoute
                ? "Your details, attendance, leave, salary, and payslips"
                : "Employee details, attendance, leave, salary, and payslips"}
            </div>
          </div>
          <GlobalHeaderControls />
        </div>
        <div style={{ padding: "4rem 0" }}><SectionLoader size="md" /></div>
      </>
    );
  }

  if (error) {
    return (
      <div className="card">
        <p className="text-muted">{error}</p>
        <NavLink
          to={isMyProfileRoute ? "/" : "/employees"}
          className="btn btn-secondary"
          style={{ marginTop: "0.75rem", display: "inline-block" }}
        >
          {isMyProfileRoute ? "Back to dashboard" : "Back to Employees"}
        </NavLink>
      </div>
    );
  }

  if (!emp) return null;

  const fullName = `${emp.first_name} ${emp.last_name}`;

  return (
    <>
      <div className="page-header" style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
        <div>
          <h1 className="page-title">{isMyProfileRoute ? "My profile" : "Employee profile"}</h1>
          <div className="page-subtitle">
            {isMyProfileRoute
              ? "Your details, attendance, leave, salary, and payslips"
              : "Employee details, attendance, leave, salary, and payslips"}
          </div>
        </div>
        <GlobalHeaderControls />
      </div>

      <div className="emp-profile-hero">
        <div className="emp-profile-hero-inner">
          <div className="emp-profile-avatar" title={fullName}>
            {initials(emp.first_name, emp.last_name)}
          </div>

          <div style={{ minWidth: 0 }}>
            <div style={{ fontSize: "1.3rem", fontWeight: 800, color: "rgba(255, 255, 255, 0.96)", lineHeight: 1.2 }}>
              {fullName}
            </div>
            <div className="text-muted" style={{ marginTop: 4, fontSize: "0.95rem" }}>
              {company?.name || "Company"} · ID {emp.employee_code}
            </div>

            <div
              style={{
                marginTop: "0.9rem",
                display: "flex",
                flexWrap: "wrap",
                gap: "2rem 3rem",
              }}
            >
              <InfoItem label="Official email" value={emp.official_email || "-"} />
              <InfoItem label="Phone" value={emp.phone || "-"} />
              <InfoItem label="Designation" value={desigName} />
              <InfoItem label="Department" value={deptName} />
            </div>
          </div>

          <div style={{ justifySelf: "end" }}>
            <span
              className={`emp-profile-status-pill ${emp.employment_status === "Active" ? "emp-profile-status-pill--active" : "emp-profile-status-pill--inactive"
                }`}
            >
              {emp.employment_status}
            </span>
          </div>
        </div>

        <div className="emp-profile-tabs">
          {(Object.keys(tabLabel) as TabKey[]).map((k) => (
            <button
              key={k}
              type="button"
              className={`emp-profile-tab ${tab === k ? "emp-profile-tab--active" : ""}`}
              onClick={() => setTab(k)}
            >
              {tabLabel[k]}
            </button>
          ))}
        </div>
      </div>

      {/* Tab content */}
      <div className="card emp-profile-card" style={{ marginTop: "1rem" }}>
        {tab === "about" && (
          <>
            <div className="emp-profile-about-head">
              <div className="emp-profile-about-head__title-block">
                <h3>Employee details</h3>
                <p className="text-muted" style={{ margin: 0, maxWidth: "42rem" }}>
                  {canEdit ? (
                    <>
                      Personal, job, and contact information. Use <strong>Edit details</strong> to update records.
                    </>
                  ) : (
                    <>Personal, job, and contact information. This page is view-only; contact HR to request changes.</>
                  )}
                </p>
              </div>
              <div className="emp-profile-about-head__actions">
                {!editing && canEdit && (
                  <button
                    type="button"
                    className="btn btn-primary"
                    style={{ padding: '0.85rem 3rem' }}
                    onClick={() => {
                      setForm(empToForm(emp));
                      setSaveError("");
                      setSaveSuccess("");
                      setEditing(true);
                    }}
                  >
                    Edit details
                  </button>
                )}
                {editing && form && canEdit && (
                  <>

                    <button type="submit" form="emp-profile-edit-form" className="btn btn-primary" style={{ padding: '0.85rem 2.5rem' }}>
                      Save changes
                    </button>
                    <button
                      type="button"
                      className="btn btn-secondary"
                      style={{ padding: '0.85rem 2.5rem', color: "#ef4444", background: "rgba(239, 68, 68, 0.15)", }}
                      onClick={() => {
                        setEditing(false);
                        setForm(null);
                        setSaveError("");
                      }}
                    >
                      Cancel
                    </button>
                  </>
                )}
              </div>
            </div>

            {saveSuccess && (
              <p className="text-muted" style={{ marginTop: 0, marginBottom: "0.75rem", color: "#22c55e" }}>
                {saveSuccess}
              </p>
            )}
            {saveError && (
              <p className="text-muted" style={{ marginTop: 0, marginBottom: "0.75rem", color: "#b91c1c" }}>
                {saveError}
              </p>
            )}

            {!editing && (
              <div className="emp-detail-stack">
                <section className="emp-detail-section" aria-labelledby="emp-section-personal-title">
                  <div className="emp-detail-section__head">
                    <div>
                      <h4 className="emp-detail-section__title" id="emp-section-personal-title">
                        Personal
                      </h4>
                      <p className="emp-detail-section__subtitle">Identity, employee code, joining and leaving dates</p>
                    </div>
                  </div>
                  <div className="emp-detail-section__grid">
                    <DetailField label="Full name" value={fullName} />
                    <DetailField label="Employee ID" value={emp.employee_code} />
                    <DetailField label="Date of birth" value={emp.date_of_birth?.split('-').reverse().join('-') || "-"} />
                    <DetailField label="Marital status" value={emp.marital_status || "-"} />
                    <DetailField label="Date of marriage" value={emp.date_of_marriage?.split('-').reverse().join('-') || "-"} />
                    <DetailField label="Date of joining" value={emp.date_of_joining?.split('-').reverse().join('-')} />
                    <DetailField label="Date of leaving" value={emp.date_of_leaving?.split('-').reverse().join('-') || "-"} />
                  </div>
                </section>

                <section className="emp-detail-section" aria-labelledby="emp-section-job-title">
                  <div className="emp-detail-section__head">
                    <div>
                      <h4 className="emp-detail-section__title" id="emp-section-job-title">
                        Job
                      </h4>
                      <p className="emp-detail-section__subtitle">Department, role, and employment status</p>
                    </div>
                  </div>
                  <div className="emp-detail-section__grid">
                    <DetailField label="Department" value={deptName} variant="job" />
                    <DetailField label="Designation" value={desigName} variant="job" />
                    <DetailField label="Employment type" value={emp.employment_type} variant="job" />
                    <DetailField label="Status" value={emp.employment_status} variant="job" />
                    <DetailField label="Working hours" value={`${emp.expected_working_hours || 9.0} hrs/day`} variant="job" />
                  </div>
                </section>

                <section className="emp-detail-section" aria-labelledby="emp-section-contact-title">
                  <div className="emp-detail-section__head">
                    <div>
                      <h4 className="emp-detail-section__title" id="emp-section-contact-title">
                        Contact
                      </h4>
                      <p className="emp-detail-section__subtitle">Work and personal communication</p>
                    </div>
                  </div>
                  <div className="emp-detail-section__grid">
                    <DetailField label="Official email" value={emp.official_email || "-"} variant="contact" />
                    <DetailField label="Personal email" value={emp.personal_email || "-"} variant="contact" />
                    <DetailField label="Phone" value={emp.phone || "-"} variant="contact" />
                  </div>
                </section>

                {canViewDocs && (
                  <section className="emp-detail-section" aria-labelledby="emp-section-docs-title">
                    <div className="emp-detail-section__head">
                      <div>
                        <h4 className="emp-detail-section__title" id="emp-section-docs-title">
                          Important documents
                        </h4>
                        <p className="emp-detail-section__subtitle">IDs and expiry dates (metadata only)</p>
                      </div>
                    </div>
                    <div className="emp-detail-section__grid">
                      <DetailField label="PAN" value={emp.pan_number || "-"} variant="contact" />
                      <DetailField label="Aadhar" value={emp.aadhar_number || "-"} variant="contact" />
                    </div>
                  </section>
                )}

                {canViewBank && bankForm && !editingBank && (
                  <section className="emp-detail-section" aria-labelledby="emp-section-bank-title">
                    <div className="emp-detail-section__head" style={{ display: 'flex', flexWrap: 'wrap', gap: '1rem', width: '100%' }}>
                      <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'flex-start' }}>
                        <div>
                          <h4 className="emp-detail-section__title" id="emp-section-bank-title">
                            Bank details
                          </h4>
                          <p className="emp-detail-section__subtitle">
                            Salary account information
                          </p>
                        </div>
                      </div>
                      {canEdit && (
                        <div style={{ marginLeft: 'auto' }}>
                          <button
                            type="button"
                            className="btn btn-secondary btn-sm"
                            onClick={() => {
                              setEditingBank(true);
                              setBankSaved("");
                              setBankError("");
                            }}
                          >
                            Edit bank details
                          </button>
                        </div>
                      )}
                    </div>
                    <div className="emp-detail-section__grid">
                      <DetailField label="Bank name" value={bankForm.bank_name || "-"} variant="job" />
                      <DetailField label="Branch name" value={bankForm.branch_name || "-"} variant="job" />
                      <DetailField label="Account holder" value={bankForm.account_holder_name || "-"} variant="job" />
                      <DetailField label="Account number" value={bankForm.account_number || "-"} variant="job" />
                      <DetailField label="IFSC" value={bankForm.ifsc_code || "-"} variant="job" />
                      <DetailField label="Account type" value={bankForm.account_type || "-"} variant="job" />
                    </div>
                  </section>
                )}

                {canEdit && (
                  <section className="emp-detail-section" aria-labelledby="emp-section-face-title">
                    <div className="emp-detail-section__head">
                      <div>
                        <h4 className="emp-detail-section__title" id="emp-section-face-title">
                          Face Recognition Data
                        </h4>
                        <p className="emp-detail-section__subtitle">Upload 1-5 face samples for attendance detection</p>
                      </div>
                    </div>
                    <div style={{ padding: "1.5rem" }}>
                      <form onSubmit={handleFaceUpload}>
                        <div className="form-group">
                          <input 
                            type="file" 
                            multiple 
                            accept="image/*" 
                            onChange={(e) => setFaceImages(Array.from(e.target.files || []))} 
                          />
                        </div>
                        {faceUploadError && <p style={{ color: "#ef4444", marginTop: "0.5rem" }}>{faceUploadError}</p>}
                        {faceUploadSuccess && <p style={{ color: "#22c55e", marginTop: "0.5rem" }}>{faceUploadSuccess}</p>}
                        <button type="submit" className="btn btn-primary" disabled={faceUploadLoading || faceImages.length === 0} style={{ marginTop: "1rem" }}>
                          {faceUploadLoading ? "Uploading..." : "Upload & Register Face"}
                        </button>
                      </form>
                    </div>
                  </section>
                )}
              </div>
            )}

            {editing && form && (
              <form id="emp-profile-edit-form" className="emp-detail-stack" onSubmit={handleSaveProfile}>
                <section className="emp-detail-section" aria-labelledby="emp-edit-personal-title">
                  <div className="emp-detail-section__head">
                    <div>
                      <h4 className="emp-detail-section__title" id="emp-edit-personal-title">
                        Personal
                      </h4>
                      <p className="emp-detail-section__subtitle">Names, code, and dates</p>
                    </div>
                  </div>
                  <div className="emp-detail-section__grid">
                    <div className="form-group">
                      <label>First name</label>
                      <input
                        value={form.first_name}
                        onChange={(e) => setForm({ ...form, first_name: e.target.value })}
                        required
                      />
                    </div>
                    <div className="form-group">
                      <label>Last name</label>
                      <input
                        value={form.last_name}
                        onChange={(e) => setForm({ ...form, last_name: e.target.value })}
                        required
                      />
                    </div>
                    <div className="form-group">
                      <label>Employee code</label>
                      <input
                        value={form.employee_code}
                        onChange={(e) => setForm({ ...form, employee_code: e.target.value })}
                        required
                      />
                    </div>
                    <div className="form-group">
                      <label>Date of birth</label>
                      <input
                        type="date"
                        value={form.date_of_birth}
                        max={maxDobDate}
                        onChange={(e) => setForm({ ...form, date_of_birth: e.target.value })}
                      />
                    </div>
                    <div className="form-group">
                      <label>Marital status</label>
                      <CustomSelect
                        value={form.marital_status}
                        onChange={(val) =>
                          setForm({
                            ...form,
                            marital_status: val,
                            date_of_marriage: val === "Married" ? form.date_of_marriage : "",
                          })
                        }
                        placeholder="Select Marital Status"
                        options={[
                          { value: "", label: "Select Marital Status" },
                          ...MARITAL_STATUSES.map((s) => ({ value: s, label: s })),
                        ]}
                      />
                    </div>
                    <div className="form-group">
                      <label>Date of marriage</label>
                      <input
                        type="date"
                        value={form.date_of_marriage}
                        onChange={(e) => setForm({ ...form, date_of_marriage: e.target.value })}
                        disabled={form.marital_status !== "Married"}
                        title={form.marital_status !== "Married" ? "Available only when Marital Status is Married" : undefined}
                      />
                    </div>
                    <div className="form-group">
                      <label>Date of joining</label>
                      <input
                        type="date"
                        value={form.date_of_joining}
                        onChange={(e) => {
                          const val = e.target.value;
                          setForm({ ...form, date_of_joining: val });
                          if (form.date_of_leaving && val > form.date_of_leaving) {
                            setForm({ ...form, date_of_joining: val, date_of_leaving: val });
                          }
                        }}
                        required
                      />
                    </div>
                    <div className="form-group">
                      <label>Date of leaving</label>
                      <input
                        type="date"
                        value={form.date_of_leaving}
                        min={form.date_of_joining}
                        onChange={(e) => {
                          const val = e.target.value;
                          setForm((f) => {
                            if (!f) return f;
                            const next: ProfileForm = { ...f, date_of_leaving: val };
                            if (val) {
                              if (next.employment_status === "Active") {
                                next.employment_status = "Resigned";
                              }
                            } else {
                              next.employment_status = "Active";
                            }
                            return next;
                          });
                        }}
                      />
                      <span className="text-muted" style={{ fontSize: "0.78rem", display: "block", marginTop: "0.35rem" }}>
                        Last working day (if resigned or terminated). Leave blank if still employed.
                      </span>
                    </div>
                    {form.date_of_leaving && (
                      <div className="form-group">
                        <label>Reason for leaving *</label>
                        <CustomSelect
                          value={form.employment_status === "Active" ? "Resigned" : form.employment_status}
                          onChange={(val) => setForm({ ...form, employment_status: val })}
                          options={[
                            { value: "Resigned", label: "Resigned" },
                            { value: "Terminated", label: "Terminated" },
                          ]}
                        />
                        <span className="text-muted" style={{ fontSize: "0.78rem", display: "block", marginTop: "0.35rem" }}>
                          Required when a date of leaving is set.
                        </span>
                      </div>
                    )}
                  </div>
                </section>

                <section className="emp-detail-section" aria-labelledby="emp-edit-job-title">
                  <div className="emp-detail-section__head">
                    <div>
                      <h4 className="emp-detail-section__title" id="emp-edit-job-title">
                        Job
                      </h4>
                      <p className="emp-detail-section__subtitle">Org structure and employment</p>
                    </div>
                  </div>
                  <div className="emp-detail-section__grid">
                    <div className="form-group">
                      <label>Department</label>
                      <CustomSelect
                        value={form.department_id === "" ? "" : String(form.department_id)}
                        onChange={(val) =>
                          setForm({
                            ...form,
                            department_id: val === "" ? "" : Number(val),
                          })
                        }
                        options={[
                          { value: "", label: "—" },
                          ...depts.map((d) => ({ value: String(d.id), label: d.name }))
                        ]}
                      />
                    </div>
                    <div className="form-group">
                      <label>Designation</label>
                      <CustomSelect
                        value={form.designation_id === "" ? "" : String(form.designation_id)}
                        onChange={(val) =>
                          setForm({
                            ...form,
                            designation_id: val === "" ? "" : Number(val),
                          })
                        }
                        options={[
                          { value: "", label: "—" },
                          ...desigs.map((d) => ({ value: String(d.id), label: d.title }))
                        ]}
                      />
                    </div>
                    <div className="form-group">
                      <label>Employment type</label>
                      <CustomSelect
                        value={form.employment_type}
                        onChange={(val) => setForm({ ...form, employment_type: val })}
                        options={EMPLOYMENT_TYPES.map((t) => ({ value: t, label: t }))}
                      />
                    </div>
                    <div className="form-group">
                      <label>Employment status</label>
                      <CustomSelect
                        value={form.employment_status}
                        onChange={(val) => setForm({ ...form, employment_status: val })}
                        options={EMPLOYMENT_STATUSES.map((s) => ({ value: s, label: s }))}
                      />
                    </div>
                    <div className="form-group">
                      <label>Expected Working Hours</label>
                      <input
                        type="number"
                        step="0.5"
                        value={form.expected_working_hours}
                        onChange={(e) => setForm({ ...form, expected_working_hours: Number(e.target.value) })}
                        required
                        placeholder="e.g. 9.0"
                      />
                    </div>
                  </div>
                </section>

                <section className="emp-detail-section" aria-labelledby="emp-edit-contact-title">
                  <div className="emp-detail-section__head">
                    <div>
                      <h4 className="emp-detail-section__title" id="emp-edit-contact-title">
                        Contact
                      </h4>
                      <p className="emp-detail-section__subtitle">Email and phone</p>
                    </div>
                  </div>
                  <div className="emp-detail-section__grid">
                    <div className="form-group">
                      <label>Official email</label>
                      <input
                        type="email"
                        value={form.official_email}
                        onChange={(e) => setForm({ ...form, official_email: e.target.value })}
                        required
                      />
                    </div>
                    <div className="form-group">
                      <label>Personal email</label>
                      <input
                        type="email"
                        value={form.personal_email}
                        onChange={(e) => setForm({ ...form, personal_email: e.target.value })}
                      />
                    </div>
                    <div className="form-group">
                      <label>Phone</label>
                      <input value={form.phone} onChange={(e) => setForm({ ...form, phone: e.target.value })} />
                    </div>
                  </div>
                </section>

                {canEdit && (
                  <section className="emp-detail-section" aria-labelledby="emp-edit-docs-title">
                    <div className="emp-detail-section__head">
                      <div>
                        <h4 className="emp-detail-section__title" id="emp-edit-docs-title">
                          Important documents
                        </h4>
                        <p className="emp-detail-section__subtitle">Numbers and expiry dates</p>
                      </div>
                    </div>
                    <div className="emp-detail-section__grid">
                      <div className="form-group">
                        <label>PAN</label>
                        <input value={form.pan_number} onChange={(e) => setForm({ ...form, pan_number: e.target.value })} />
                      </div>
                      <div className="form-group">
                        <label>Aadhar</label>
                        <input value={form.aadhar_number} onChange={(e) => setForm({ ...form, aadhar_number: e.target.value })} />
                      </div>
                    </div>
                  </section>
                )}
              </form>
            )}

            {canEdit && editingBank && (
              <div className="card" style={{ marginTop: "1rem" }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <h3 style={{ marginTop: 0 }}>Edit bank details</h3>
                  <button type="button" className="btn btn-secondary btn-sm" onClick={() => setEditingBank(false)}>Cancel</button>
                </div>
                <p className="text-muted" style={{ marginTop: 0 }}>
                  HR/Admin only. These details are stored separately for payroll.
                </p>
                {bankSaved && <div className="alert alert-success">{bankSaved}</div>}
                {bankError && <div className="alert alert-error">{bankError}</div>}
                {bankLoading && !bankForm ? (
                  <SectionLoader size="sm" />
                ) : (
                  bankForm && (
                    <form onSubmit={handleSaveBank}>
                      <div className="modal-form-grid">
                        <div className="form-group">
                          <label>Bank name</label>
                          <input
                            value={bankForm.bank_name}
                            onChange={(e) => setBankForm({ ...bankForm, bank_name: e.target.value })}
                            required
                          />
                        </div>
                        <div className="form-group">
                          <label>Branch name</label>
                          <input
                            value={bankForm.branch_name}
                            onChange={(e) => setBankForm({ ...bankForm, branch_name: e.target.value })}
                            placeholder="e.g. Andheri West"
                          />
                        </div>
                        <div className="form-group">
                          <label>Account holder name</label>
                          <input
                            value={bankForm.account_holder_name}
                            onChange={(e) => setBankForm({ ...bankForm, account_holder_name: e.target.value })}
                            required
                          />
                        </div>
                        <div className="form-group">
                          <label>Account number</label>
                          <input
                            value={bankForm.account_number}
                            onChange={(e) => setBankForm({ ...bankForm, account_number: e.target.value })}
                            required
                          />
                        </div>
                        <div className="form-group">
                          <label>IFSC code</label>
                          <input
                            value={bankForm.ifsc_code}
                            onChange={(e) => setBankForm({ ...bankForm, ifsc_code: e.target.value })}
                            required
                          />
                        </div>
                        <div className="form-group">
                          <label>Account type</label>
                          <CustomSelect
                            value={bankForm.account_type}
                            onChange={(val) => setBankForm({ ...bankForm, account_type: val })}
                            options={[
                              { value: "Savings", label: "Savings" },
                              { value: "Current", label: "Current" }
                            ]}
                          />
                        </div>
                      </div>
                      <div className="modal-actions">
                        <button type="submit" className="btn btn-primary" disabled={bankLoading}>
                          {bankLoading ? "Saving..." : "Save bank details"}
                        </button>
                      </div>
                    </form>
                  )
                )}
              </div>
            )}
          </>
        )}

        {tab === "attendance" && (
          <>
            <h3 style={{ marginTop: 0, marginBottom: "0.75rem" }}>Attendance</h3>
            <p className="text-muted" style={{ marginTop: 0, marginBottom: "1rem" }}>
              Monthly attendance calendar for this employee (read-only).
            </p>
            <MonthlyAttendanceGrid
              month={attMonth}
              year={attYear}
              setMonth={setAttMonth}
              setYear={setAttYear}
              records={attRecords}
              loading={attLoading}
            />
          </>
        )}

        {tab === "attendance_details" && (
          <>
            <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap", alignItems: "flex-end" }}>
              <div>
                <h3 style={{ marginTop: 0, marginBottom: 4 }}>Attendance Details</h3>
                <p className="text-muted" style={{ margin: 0, fontSize: "0.9rem" }}>
                  Daily timeline and complete check-in/check-out event history.
                </p>
              </div>
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label>Select Date</label>
                <input
                  type="date"
                  value={selectedDetailDate}
                  onChange={(e) => setSelectedDetailDate(e.target.value)}
                  style={{
                    background: "rgba(255,255,255,0.04)",
                    border: "1px solid rgba(255,255,255,0.15)",
                    borderRadius: "8px",
                    padding: "0 12px",
                    color: "#fff",
                    fontSize: "0.85rem",
                    width: "180px",
                    height: "42px",
                  }}
                />
              </div>
            </div>

            <div style={{ marginTop: "1.5rem" }}>
              {/* Attendance Timeline Card */}
              <div style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.08)",
                borderRadius: "12px",
                padding: "1.25rem",
                marginBottom: "1.5rem",
              }}>
                <h4 style={{ marginTop: 0, marginBottom: "1rem", fontSize: "1rem" }}>Attendance Timeline</h4>
                {timelineLoading ? (
                  <SectionLoader size="sm" />
                ) : timelineData ? (
                  <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: "1rem" }}>
                    <div style={{ padding: "0.75rem", borderRadius: "8px", background: "rgba(34,197,94,0.08)", border: "1px solid rgba(34,197,94,0.15)" }}>
                      <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.6)", marginBottom: "0.25rem", textTransform: "uppercase", letterSpacing: "0.04em" }}>First Check-In</div>
                      <div style={{ fontSize: "1rem", fontWeight: 700, color: "#22c55e" }}>{formatTime12h(timelineData.first_check_in)}</div>
                    </div>
                    <div style={{ padding: "0.75rem", borderRadius: "8px", background: "rgba(239,68,68,0.08)", border: "1px solid rgba(239,68,68,0.15)" }}>
                      <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.6)", marginBottom: "0.25rem", textTransform: "uppercase", letterSpacing: "0.04em" }}>Last Check-Out</div>
                      <div style={{ fontSize: "1rem", fontWeight: 700, color: "#ef4444" }}>{formatTime12h(timelineData.last_check_out)}</div>
                    </div>
                    <div style={{ padding: "0.75rem", borderRadius: "8px", background: "rgba(59,130,246,0.08)", border: "1px solid rgba(59,130,246,0.15)" }}>
                      <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.6)", marginBottom: "0.25rem", textTransform: "uppercase", letterSpacing: "0.04em" }}>Working Hours</div>
                      <div style={{ fontSize: "1rem", fontWeight: 700, color: "#3b82f6" }}>{formatCompactDuration(timelineData.total_work_hours)}</div>
                    </div>
                    <div style={{ padding: "0.75rem", borderRadius: "8px", background: "rgba(168,85,247,0.08)", border: "1px solid rgba(168,85,247,0.15)" }}>
                      <div style={{ fontSize: "0.75rem", color: "rgba(255,255,255,0.6)", marginBottom: "0.25rem", textTransform: "uppercase", letterSpacing: "0.04em" }}>Break Time</div>
                      <div style={{ fontSize: "1rem", fontWeight: 700, color: "#a855f7" }}>{formatCompactDuration(timelineData.total_break_hours)}</div>
                    </div>
                  </div>
                ) : (
                  <div style={{ opacity: 0.65, textAlign: "center", padding: "1rem" }}>No attendance data for this date.</div>
                )}
              </div>

              {/* Event History Table */}
              <div>
                <h4 style={{ marginTop: 0, marginBottom: "1rem", fontSize: "1rem" }}>Event History</h4>
                {eventsLoading ? (
                  <SectionLoader size="sm" />
                ) : eventHistory.length === 0 ? (
                  <div style={{ opacity: 0.65, textAlign: "center", padding: "1rem" }}>No attendance events recorded.</div>
                ) : (
                  <div className="table-wrap table-wrap--dark" style={{ maxHeight: "400px", overflowY: "auto" }}>
                    <table className="table-modern table-modern--dark">
                      <thead>
                        <tr>
                          <th style={{ textAlign: "left", paddingLeft: "1.5rem", width: "22%" }}>Date</th>
                          <th style={{ width: "22%" }}>Event Time</th>
                          <th style={{ width: "22%" }}>Event Type</th>
                          <th style={{ width: "22%" }}>Source</th>
                          {canEdit && <th style={{ paddingRight: "1.5rem", width: "12%" }}>Actions</th>}
                        </tr>
                      </thead>
                      <tbody>
                        {eventHistory.map((evt) => {
                          const eventDate = new Date(evt.event_time).toLocaleDateString("en-IN", {
                            day: "2-digit",
                            month: "short",
                            year: "numeric"
                          });
                          const eventTime = new Date(evt.event_time).toLocaleTimeString("en-IN", {
                            hour: "2-digit",
                            minute: "2-digit",
                            hour12: true
                          });
                          const isCheckIn = evt.event_type === "CHECK_IN" || evt.event_type === "IN";
                          return (
                            <tr key={evt.id}>
                              <td style={{ textAlign: "left", paddingLeft: "1.5rem" }}>{eventDate}</td>
                              <td>{eventTime}</td>
                              <td>
                                <span style={{
                                  fontWeight: 700,
                                  color: isCheckIn ? "#22c55e" : "#f59e0b",
                                }}>
                                  {evt.event_type}
                                </span>
                              </td>
                              <td>{evt.source}</td>
                              {canEdit && (
                                <td style={{ paddingRight: "1.5rem", textAlign: "center" }}>
                                  <button
                                    type="button"
                                    className="btn btn-secondary btn-sm"
                                    onClick={() => handleDeleteEvent(evt.id)}
                                    style={{
                                      padding: "0.35rem 0.75rem",
                                      fontSize: "0.75rem",
                                      backgroundColor: "rgba(239, 68, 68, 0.15)",
                                      color: "#ef4444",
                                      border: "1px solid rgba(239, 68, 68, 0.3)",
                                    }}
                                    onMouseOver={(e) => {
                                      e.currentTarget.style.backgroundColor = "rgba(239, 68, 68, 0.25)";
                                    }}
                                    onMouseOut={(e) => {
                                      e.currentTarget.style.backgroundColor = "rgba(239, 68, 68, 0.15)";
                                    }}
                                  >
                                    Delete
                                  </button>
                                </td>
                              )}
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {tab === "leaves" && (
          <>
            <div style={{ display: "flex", justifyContent: "space-between", gap: "1rem", flexWrap: "wrap", alignItems: "flex-end" }}>
              <div>
                <h3 style={{ marginTop: 0, marginBottom: 4 }}>Leaves</h3>
                <p className="text-muted" style={{ margin: 0, fontSize: "0.9rem" }}>
                  Allocations and requests for this employee.
                </p>
              </div>
              <div className="form-group" style={{ marginBottom: 0 }}>
                <label>Financial year</label>
                <CustomSelect
                  value={String(selectedFyId ?? "")}
                  onChange={(val) => setSelectedFyId(Number(val) || null)}
                  style={{ width: "320px", maxWidth: "100%" }}
                  options={financialYears.map((fy) => ({
                    value: String(fy.id),
                    label: `${(fy.name || `FY ${formatNiceDate(fy.start_date)} - ${formatNiceDate(fy.end_date)}`)} (${formatNiceDate(fy.start_date)} to ${formatNiceDate(fy.end_date)})`
                  }))}
                />
              </div>
            </div>

            <div style={{ marginTop: "2.5rem" }}>
              <h4 style={{ marginTop: 0, marginBottom: '0.75rem' }}></h4>
              {allocs.length === 0 ? (
                <p className="text-muted">No allocations.</p>
              ) : (
                <div className="table-wrap table-wrap--dark">
                  <table className="table-modern table-modern--dark" style={{ tableLayout: 'fixed', width: '100%' }}>
                    <thead>
                      <tr>
                        <th style={{ width: '25%', textAlign: 'center' }}>Type</th>
                        <th style={{ width: '25%', textAlign: 'center' }}>Allocated</th>
                        <th style={{ width: '25%', textAlign: 'center' }}>Used</th>
                        <th style={{ width: '25%', textAlign: 'center' }}>Balance</th>
                      </tr>
                    </thead>
                    <tbody>
                      {allocs.map((a) => (
                        <tr key={a.id}>
                          <td style={{ textAlign: 'center' }}>{leaveTypes.find((t) => t.id === a.leave_type_id)?.name ?? a.leave_type_id}</td>
                          <td style={{ textAlign: 'center' }}>{Math.round(Number(a.allocated_days))}</td>
                          <td style={{ textAlign: 'center' }}>{Math.round(Number(a.used_days))}</td>
                          <td style={{ textAlign: 'center' }}>{Math.round(Number(a.balance_days))}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
            </div>

            <div style={{ marginTop: "1rem" }}>
              <h4 style={{ marginTop: 0, marginBottom: '0.75rem' }}>Leave requests</h4>
              {leaveReqs.length === 0 ? (
                <p className="text-muted">No leave requests.</p>
              ) : (
                <div className="table-wrap table-wrap--dark">
                  <table className="table-modern table-modern--dark" style={{ tableLayout: 'fixed' }}>
                    <thead>
                      <tr>
                        <th style={{ textAlign: 'left', paddingLeft: '1.5rem', width: '14.28%' }}>Start</th>
                        <th style={{ width: '14.28%' }}>End</th>
                        <th style={{ width: '14.28%' }}>Type</th>
                        <th style={{ width: '14.28%' }}>Kind</th>
                        <th style={{ width: '14.28%' }}>Reason</th>
                        <th style={{ width: '14.28%' }}>Response</th>
                        <th style={{ paddingRight: '1.5rem', width: '14.28%' }}>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {leaveReqs.map((r) => {
                        const typeName = leaveTypes.find((t) => t.id === r.leave_type_id)?.name ?? r.leave_type_id;
                        const lowerName = String(typeName).toLowerCase();
                        const isUnpaid = lowerName.includes("unpaid") || lowerName.includes("lop");
                        const kindLabel = isUnpaid
                          ? r.is_half_day
                            ? "Half day (Unpaid)"
                            : "Full day (Unpaid)"
                          : r.is_half_day
                            ? "Half day (Paid)"
                            : "Full day (Paid)";
                        return (
                          <tr key={r.id}>
                            <td style={{ textAlign: 'left', paddingLeft: '1.5rem' }}>{formatNiceDate(r.start_date)}</td>
                            <td>{formatNiceDate(r.end_date)}</td>
                            <td>{typeName}</td>
                            <td>{kindLabel}</td>
                            <td title={r.reason || ""}>
                              <div style={{ maxWidth: '100%', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {r.reason || "-"}
                              </div>
                            </td>
                            <td title={r.rejection_reason || ""}>
                              <div style={{ maxWidth: '100%', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                                {r.rejection_reason || "-"}
                              </div>
                            </td>
                            <td style={{ paddingRight: '1.5rem', fontWeight: 600 }}>{r.status}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}
            </div>
          </>
        )}

        {tab === "salary" && (
          <>
            <h3 style={{ marginTop: 0, marginBottom: '0.75rem' }}>Salary</h3>
            <p className="text-muted" style={{ marginTop: 0 }}>
              Salary structures are effective by date. The latest applicable structure is used until an increment adds a new one.
            </p>

            {structures.length === 0 ? (
              <p className="text-muted">No salary structure found for this employee.</p>
            ) : (
              <div className="table-wrap table-wrap--dark">
                <table className="table-modern table-modern--dark" style={{ tableLayout: 'fixed' }}>
                  <thead>
                    <tr>
                      <th style={{ textAlign: 'center', width: '12%' }}>Effective from</th>
                      <th style={{ textAlign: 'center', width: '10%' }}>Basic</th>
                      <th style={{ textAlign: 'center', width: '10%' }}>HRA</th>
                      <th style={{ textAlign: 'center', width: '10%' }}>Medical</th>
                      <th style={{ textAlign: 'center', width: '10%' }}>Travelling</th>
                      <th style={{ textAlign: 'center', width: '10%' }}>Misc.</th>
                      <th style={{ textAlign: 'center', width: '10%' }}>Allowances</th>
                      <th style={{ textAlign: 'center', width: '12%' }}>Gross</th>
                      <th style={{ textAlign: 'center', width: '12%' }}>Deductions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {structures.map((s) => {
                      const gross =
                        Number(s.basic) +
                        Number(s.hra) +
                        Number(s.medical ?? 0) +
                        Number(s.travelling ?? 0) +
                        Number(s.miscellaneous ?? 0) +
                        Number(s.allowances);
                      return (
                        <tr key={s.id}>
                          <td style={{ textAlign: 'center' }}>{formatNiceDate(s.effective_from)}</td>
                          <td style={{ textAlign: 'center' }}>₹ {Number(s.basic).toFixed(2)}</td>
                          <td style={{ textAlign: 'center' }}>₹ {Number(s.hra).toFixed(2)}</td>
                          <td style={{ textAlign: 'center' }}>₹ {Number(s.medical ?? 0).toFixed(2)}</td>
                          <td style={{ textAlign: 'center' }}>₹ {Number(s.travelling ?? 0).toFixed(2)}</td>
                          <td style={{ textAlign: 'center' }}>₹ {Number(s.miscellaneous ?? 0).toFixed(2)}</td>
                          <td style={{ textAlign: 'center' }}>₹ {Number(s.allowances).toFixed(2)}</td>
                          <td style={{ textAlign: 'center', fontWeight: 600 }}>₹ {gross.toFixed(2)}</td>
                          <td style={{ textAlign: 'center' }}>₹ {Number(s.deductions).toFixed(2)}</td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}

        {tab === "payslips" && (
          <>
            <h3 style={{ marginTop: 0, marginBottom: '0.75rem' }}>Payslips</h3>
            {payslips.length === 0 ? (
              <p className="text-muted">No payslips.</p>
            ) : (
              <div className="table-wrap table-wrap--dark">
                <table className="table-modern table-modern--dark" style={{ tableLayout: 'fixed' }}>
                  <thead>
                    <tr>
                      <th style={{ textAlign: 'center', width: '20%' }}>Net salary</th>
                      <th style={{ textAlign: 'center', width: '20%' }}>Paid days</th>
                      <th style={{ textAlign: 'center', width: '20%' }}>LOP days</th>
                      <th style={{ textAlign: 'center', width: '20%' }}>Generated</th>
                      <th style={{ textAlign: 'center', width: '20%' }}>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {payslips.map((p) => (
                      <tr key={p.id}>
                        <td style={{ textAlign: 'center', fontWeight: 600 }}>₹ {Number(p.net_salary).toFixed(2)}</td>
                        <td style={{ textAlign: 'center' }}>{p.paid_days}</td>
                        <td style={{ textAlign: 'center' }}>{p.lop_days}</td>
                        <td style={{ textAlign: 'center' }}>{formatNiceDate(p.generated_at)}</td>
                        <td style={{ textAlign: 'center' }}>
                          <div style={{ display: 'flex', justifyContent: 'center', width: '100%' }}>
                            <button type="button" className="btn btn-secondary btn-sm" onClick={() => setFormulaPayslip(p)} style={{ backgroundColor: "var(--brand-500)", minWidth: '140px' }}>
                              View calculation
                            </button>
                          </div>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </>
        )}


      </div>

      {formulaPayslip && (
        <div className="modal-backdrop" onClick={() => setFormulaPayslip(null)}>
          <div className="modal" onClick={(e) => e.stopPropagation()} style={{ maxWidth: 560 }}>
            <h3 style={{ marginTop: 0 }}>Salary calculation (Payslip #{formulaPayslip.id})</h3>
            <SalaryFormulaView payslip={formulaPayslip} />
            <div style={{ marginTop: "1rem", display: "flex", justifyContent: "flex-end" }}>
              <button type="button" className="btn btn-secondary" onClick={() => setFormulaPayslip(null)}>
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </>
  );
}
