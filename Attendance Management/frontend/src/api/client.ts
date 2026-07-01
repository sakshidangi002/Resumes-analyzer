import axios from "axios";

const rawBase = import.meta.env.VITE_API_BASE_URL;
const apiBase = (rawBase == null || rawBase === "" ? "/api" : String(rawBase).replace(/\/$/, ""));

const api = axios.create({
  baseURL: apiBase,
  headers: { "Content-Type": "application/json" },
  timeout: 15000, // 15 second timeout — prevents login from hanging indefinitely
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("token");
  if (token) config.headers.Authorization = "Bearer " + token;
  return config;
});

api.interceptors.response.use(
  (r) => r,
  (err) => {
    if (err.response && err.response.status === 401) {
      // Don't redirect if it's a login attempt, otherwise error message in Login.tsx disappears on refresh
      const isLoginRequest = err.config && err.config.url && err.config.url.includes("/auth/login");
      
      localStorage.removeItem("token");
      localStorage.removeItem("user");
      
      if (!isLoginRequest) {
        window.location.href = "/login";
      }
    }
    return Promise.reject(err);
  }
);

export default api;

export const auth = {
  login: (username: string, password: string) =>
    api.post("/auth/login", { username, password }),
  signup: (username: string, password: string, official_email?: string) =>
    api.post("/auth/signup", { username, password, official_email }),
  canSignup: () => api.get<{ allowed: boolean }>("/auth/can-signup"),
  me: () => api.get("/auth/me"),
  forgotPassword: (username: string) => api.post("/auth/forgot-password", { username }),
};

export const users = {
  list: () => api.get("/users"),
  get: (id: number) => api.get("/users/" + id),
  create: (data: { username: string; password: string; official_email?: string; employee_id?: number; role_names: string[] }) =>
    api.post("/users", data),
  update: (id: number, data: { password?: string; official_email?: string; employee_id?: number; is_active?: boolean; role_names?: string[] }) =>
    api.patch("/users/" + id, data),
  delete: (id: number) => api.delete("/users/" + id),
};

export const departments = {
  list: () => api.get("/employees/departments"),
  get: (id: number) => api.get("/employees/departments/" + id),
  create: (data: { name: string; code?: string }) => api.post("/employees/departments", data),
  update: (id: number, data: { name?: string; code?: string }) => api.patch("/employees/departments/" + id, data),
  delete: (id: number) => api.delete("/employees/departments/" + id),
};

export const designations = {
  list: () => api.get("/employees/designations"),
  get: (id: number) => api.get("/employees/designations/" + id),
  create: (data: { title: string; level?: number }) => api.post("/employees/designations", data),
  update: (id: number, data: { title?: string; level?: number }) => api.patch("/employees/designations/" + id, data),
  delete: (id: number) => api.delete("/employees/designations/" + id),
};

export const employees = {
  list: (params?: { department_id?: number; status?: string }) => api.get("/employees", { params }),
  get: (id: number) => api.get("/employees/" + id),
  create: (data: object) => api.post("/employees", data),
  update: (id: number, data: object) => api.patch("/employees/" + id, data),
  delete: (id: number) => api.delete("/employees/" + id),
  bankGet: (id: number) => api.get(`/employees/${id}/bank`),
  bankUpdate: (
    id: number,
    data: {
      bank_name: string;
      branch_name?: string;
      account_holder_name: string;
      account_number: string;
      ifsc_code: string;
      account_type?: string;
    }
  ) => api.put(`/employees/${id}/bank`, data),
  departments: () => api.get("/employees/departments"),
  designations: () => api.get("/employees/designations"),
};

export const attendance = {
  list: (from_date: string, to_date: string, employee_id?: number) =>
    api.get("/attendance", { params: { from_date, to_date, employee_id } }),
  signIn: (date: string, sign_in_time: string, employee_id: number) =>
    api.post("/attendance/sign-in", null, { params: { d: date, sign_in_time, employee_id } }),
  signOut: (date: string, sign_out_time: string, employee_id: number) =>
    api.post("/attendance/sign-out", null, { params: { d: date, sign_out_time, employee_id } }),
  adminSet: (data: { employee_id: number; date: string; sign_in_time?: string | null; sign_out_time?: string | null; status?: string | null }) =>
    api.put("/attendance/admin-set", data),
  autoMark: (data: { employee_id: number; date: string; sign_in_time?: string | null; sign_out_time?: string | null; status?: string | null }) =>
    api.post("/attendance/auto-mark", data),
  addEvent: (data: { employee_id: number; event_time?: string | null; event_type?: string | null; source?: string; camera_id?: string | null }) =>
    api.post("/attendance/events", data),
  deleteEvent: (event_id: number) =>
    api.delete("/attendance/events/" + event_id),
  listEvents: (employee_id: number, date: string) =>
    api.get("/attendance/events", { params: { employee_id, date } }),
  details: (employee_id: number, date: string) =>
    api.get("/attendance/details", { params: { employee_id, date } }),
  recalculate: (employee_id: number, date: string) =>
    api.post("/attendance/recalculate", null, { params: { employee_id, date } }),
  dailyReport: (date: string, department_id?: number) =>
    api.get("/attendance/daily-report", { params: { date, department_id } }),
  correctionRequests: (params?: { employee_id?: number; status?: string }) =>
    api.get("/attendance/correction-requests", { params }),
  createCorrection: (data: object) => api.post("/attendance/correction-requests", data),
  approveCorrection: (id: number, approved: boolean, rejection_reason?: string) =>
    api.patch("/attendance/correction-requests/" + id, null, { params: { approved, rejection_reason } }),
  today: (department_id?: number) =>
    api.get("/attendance/today", { params: { department_id } }),
  liveStatus: () =>
    api.get("/attendance/live-status"),
  timeline: (employee_id: number, date: string) =>
    api.get(`/attendance/employee/${employee_id}/timeline`, { params: { date } }),
  monthlySummary: (employee_id: number, year: number, month: number) =>
    api.get(`/attendance/employee/${employee_id}/monthly-summary`, { params: { year, month } }),
  employeeHistory: (employee_id: number, from_date: string, to_date: string) =>
    api.get(`/attendance/employee/${employee_id}/history`, { params: { from_date, to_date } }),
};

export const recognition = {
  recognizeFrame: (file: Blob, threshold?: number, cameraId?: string | null, cameraPurpose?: string | null) => {
    const formData = new FormData();
    formData.append("file", file);
    // Build query params
    const params: Record<string, string | number> = {};
    if (threshold != null) params.threshold = threshold;
    if (cameraId != null) params.camera_id = cameraId;
    if (cameraPurpose != null) params.camera_purpose = cameraPurpose;
    // Call the main HRMS backend (port 5001/5002) directly
    return api.post("/recognize-frame", formData, {
      params: Object.keys(params).length > 0 ? params : undefined,
      headers: { "Content-Type": "multipart/form-data" },
    });
  },
  recognizeCctvFrame: (data: { stream_url: string; threshold?: number; camera_id?: string | null; camera_type?: string }) =>
    api.post("/recognize-cctv-frame", data),
  registerFace: (employee_id: number, name: string, department: string, images: FileList | File[]) => {
    const formData = new FormData();
    formData.append("employee_id", String(employee_id));
    formData.append("name", name);
    formData.append("department", department || "");
    for (let i = 0; i < images.length; i++) {
      formData.append("images", images[i]);
    }
    // Use the headless face registration endpoint via /api/employees/register.
    return api.post("/employees/register", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
  },
};

export const leave = {
  types: () => api.get("/leave/types"),
  allocations: (params?: { employee_id?: number; financial_year_id?: number }) =>
    api.get("/leave/allocations", { params }),
  setAllocation: (params: {
    employee_id: number;
    leave_type_id: number;
    allocated_days: number;
    financial_year_id?: number;
  }) => api.post("/leave/allocations", null, { params: { ...params } }),
  requests: (params?: { employee_id?: number; status?: string }) =>
    api.get("/leave/requests", { params }),
  approvals: (params?: { status?: string }) =>
    api.get("/leave/approvals", { params }),
  apply: (data: object) => api.post("/leave/requests", data),
  approve: (id: number, approved: boolean, comment: string) =>
    api.patch("/leave/requests/" + id, null, { params: { approved, comment } }),
  balance: (leave_type_id: number, employee_id?: number) =>
    api.get("/leave/balance", { params: { leave_type_id, employee_id } }),
  deleteRequest: (id: number) => api.delete("/leave/requests/" + id),
};

export const payroll = {
  periods: () => api.get("/payroll/periods"),
  createPeriod: (month: number, year: number) =>
    api.post("/payroll/periods", null, { params: { month, year } }),
  updatePeriod: (period_id: number, data: { status?: string; month?: number; year?: number }) =>
    api.patch("/payroll/periods/" + period_id, data),
  deletePeriod: (period_id: number) => api.delete("/payroll/periods/" + period_id),
  payslips: (params?: { employee_id?: number; period_id?: number; year?: number }) =>
    api.get("/payroll/payslips", { params }),
  getPayslip: (id: number) => api.get("/payroll/payslips/" + id),
  runPayroll: (period_id: number) => api.post("/payroll/periods/" + period_id + "/run"),
  salaryStructures: (employee_id?: number) => api.get("/payroll/salary-structures", { params: { employee_id } }),
  createSalaryStructure: (data: object) => api.post("/payroll/salary-structures", data),
  updateSalaryStructure: (id: number, data: object) => api.patch("/payroll/salary-structures/" + id, data),
  deleteSalaryStructure: (id: number) => api.delete("/payroll/salary-structures/" + id),
  createPayslip: (data: object) => api.post("/payroll/payslips", data),
  updatePayslip: (id: number, data: object) => api.patch("/payroll/payslips/" + id, data),
  deletePayslip: (id: number) => api.delete("/payroll/payslips/" + id),
};

export const letters = {
  templates: () => api.get("/letters/templates"),
  instances: (employee_id?: number) => api.get("/letters/instances", { params: { employee_id } }),
  preview: (template_code: string, employee_id: number) =>
    api.post("/letters/preview", { template_code, employee_id }),
  generate: (
    template_code: string,
    employee_id: number,
    send_email?: boolean,
    overrides?: { subject?: string; body?: string; from_email?: string },
    email_target?: "official" | "personal" | "both"
  ) =>
    api.post("/letters/generate", overrides ?? null, {
      params: { template_code, employee_id, send_email, email_target },
    }),
  letterBody: (instance_id: number) =>
    api.get("/letters/instances/" + instance_id + "/body", { responseType: "text" }),
  replies: (instance_id: number) =>
    api.get("/letters/instances/" + instance_id + "/replies"),
  addReply: (instance_id: number, message: string) =>
    api.post("/letters/instances/" + instance_id + "/replies", { message }),
  deleteInstance: (instance_id: number) =>
    api.delete("/letters/instances/" + instance_id),
  emailInstance: (
    instance_id: number,
    email_target: "official" | "personal" | "both" = "official",
    from_email?: string
  ) =>
    api.post(
      "/letters/instances/" + instance_id + "/email",
      null,
      { params: { email_target, from_email } }
    ),
};

export const reports = {
  monthlyAttendance: (month: number, year: number, department_id?: number) =>
    api.get("/reports/attendance/monthly", { params: { month, year, department_id } }),
  leaveUsage: (params?: { financial_year_id?: number; department_id?: number }) =>
    api.get("/reports/leave/usage", { params }),
  headcount: () => api.get("/reports/headcount/department"),
  salarySummary: (year: number) => api.get("/reports/salary/summary", { params: { year } }),
  exportAttendanceExcel: (month: number, year: number) =>
    api.get("/reports/export/attendance-excel", { params: { month, year }, responseType: "blob" }),
};

export const calendar = {
  holidays: (params?: { from_date?: string; to_date?: string }) => api.get("/calendar/holidays", { params }),
  createHoliday: (data: { date: string; name: string; is_optional?: boolean; financial_year_id?: number | null }) =>
    api.post("/calendar/holidays", data),
  deleteHoliday: (id: number) => api.delete("/calendar/holidays/" + id),
  events: (params?: { from_date?: string; to_date?: string }) => api.get("/calendar/events", { params }),
  createEvent: (data: {
    title: string;
    date: string;
    event_type: string;
    description?: string;
    employee_id?: number | null;
  }) => api.post("/calendar/events", data),
  updateEvent: (
    id: number,
    data: {
      title: string;
      date: string;
      event_type: string;
      description?: string;
      employee_id?: number | null;
    }
  ) => api.patch("/calendar/events/" + id, data),
  deleteEvent: (id: number) => api.delete("/calendar/events/" + id),
  birthdays: (month: number) => api.get("/calendar/birthdays", { params: { month } }),
  anniversaries: (month: number) => api.get("/calendar/anniversaries", { params: { month } }),
  marriageAnniversaries: (month: number) => api.get("/calendar/marriage-anniversaries", { params: { month } }),
  reminders: (for_date?: string, days?: number) =>
    api.get("/calendar/reminders", { params: { for_date, days } }),
};

export const company = {
  config: () => api.get("/company/config"),
  financialYears: () => api.get("/company/financial-years"),
  holidays: () => api.get("/company/holidays"),
  stats: () => api.get("/company/stats"),
};

export type AppNotificationRow = {
  id: number;
  title: string;
  body: string | null;
  kind: string;
  link_path: string | null;
  read_at: string | null;
  created_at: string;
};

export const activity = {
  notifications: (params?: { unread_only?: boolean; limit?: number }) =>
    api.get<AppNotificationRow[]>("/activity/notifications", { params }),
  unreadCount: () => api.get<{ count: number }>("/activity/notifications/unread-count"),
  markRead: (id: number) => api.patch<AppNotificationRow>("/activity/notifications/" + id + "/read"),
  markAllRead: () => api.post("/activity/notifications/read-all"),
  delete: (id: number) => api.delete("/activity/notifications/" + id),
};

export type OnboardingTaskRow = {
  id: number;
  employee_id: number;
  title: string;
  priority: string;
  due_date: string | null;
  is_completed: boolean;
  completed_at: string | null;
  sort_order: number;
  created_at: string;
};

export const onboarding = {
  mine: () => api.get<OnboardingTaskRow[]>("/onboarding/me"),
  forEmployee: (employeeId: number) => api.get<OnboardingTaskRow[]>("/onboarding/employee/" + employeeId),
  listAll: () => api.get<OnboardingTaskRow[]>("/onboarding/all"),
  createTask: (data: { employee_id: number; title: string; priority?: string; due_date?: string; sort_order?: number }) =>
    api.post<OnboardingTaskRow>("/onboarding/tasks", data),
  updateTask: (id: number, data: { is_completed?: boolean; title?: string; priority?: string; due_date?: string; sort_order?: number }) =>
    api.patch<OnboardingTaskRow>("/onboarding/tasks/" + id, data),
  deleteTask: (id: number) => api.delete("/onboarding/tasks/" + id),
};

// ---------------------------------------------------------------------------
// Daily Status Report (DSR)
// ---------------------------------------------------------------------------
export type DSRStatus = "DRAFT" | "SUBMITTED";

export type DSRRow = {
  id: number;
  employee_id: number;
  employee_name: string | null;
  employee_code: string | null;
  designation?: string | null;
  report_date: string; // YYYY-MM-DD
  project_work: string | null;
  work_location: string | null;
  total_hours: string | number | null;
  work_done: string;
  plan_for_tomorrow: string | null;
  status: DSRStatus;
  submitted_at: string | null;
  created_at: string;
  updated_at: string;
};

export type DSRSummaryRow = {
  year: number;
  month: number;
  total: number;
  submitted: number;
  draft: number;
  pending: number;
};

export type DSRCreatePayload = {
  report_date: string; // YYYY-MM-DD
  project_work?: string | null;
  work_location?: string | null;
  total_hours?: number | string | null;
  work_done: string;
  plan_for_tomorrow?: string | null;
  status?: DSRStatus;
};

export type DSRUpdatePayload = Partial<Omit<DSRCreatePayload, "report_date">>;

export type DSRTodayStatus = {
  today_ist: string;            // YYYY-MM-DD in IST
  has_dsr: boolean;
  submitted: boolean;
  dsr_id: number | null;
  needs_dsr: boolean;           // true when today's DSR isn't SUBMITTED yet
};

// ---------------------------------------------------------------------------
// Web Push (5 PM IST DSR reminder etc.)
// ---------------------------------------------------------------------------
export type PushSubscribePayload = {
  endpoint: string;
  keys: { p256dh: string; auth: string };
};

export const push = {
  vapidPublicKey: () => api.get<{ key: string }>("/push/vapid-public-key"),
  subscribe: (data: PushSubscribePayload) =>
    api.post<{ ok: boolean; subscription_id: number; updated: boolean }>(
      "/push/subscribe",
      data
    ),
  unsubscribe: (endpoint: string) =>
    api.post<{ ok: boolean; deleted: number }>("/push/unsubscribe", { endpoint }),
};

export type DSRReminderSettings = {
  enabled: boolean;
  time: string;        // "HH:MM" 24h, IST
  weekdays: string[];  // ["mon","tue",...]
  current_ist: string; // "YYYY-MM-DD HH:MM"
};

export type DSRReminderSettingsUpdate = {
  enabled?: boolean;
  time?: string;
  weekdays?: string[];
};

export type DSRPendingEmployee = {
  user_id: number;
  employee_id: number;
  employee_name: string;
  employee_code: string | null;
  department: string | null;
  designation: string | null;
  official_email: string | null;
  has_draft: boolean;
};

export type DSRPendingTodayResponse = {
  today_ist: string;
  total_active_employees: number;
  submitted: number;
  pending: DSRPendingEmployee[];
};

export type DSRManualRemindResponse = {
  today_ist: string;
  notified: number;
  skipped_submitted: number;
  skipped_no_target: number;
};

export const dsr = {
  mine: (params?: { year?: number; month?: number; limit?: number }) =>
    api.get<DSRRow[]>("/dsr/me", { params }),
  summary: (params?: { year?: number; month?: number }) =>
    api.get<DSRSummaryRow>("/dsr/me/summary", { params }),
  todayStatus: () => api.get<DSRTodayStatus>("/dsr/me/today-status"),
  reminderSettings: () => api.get<DSRReminderSettings>("/dsr-reminder/settings"),
  updateReminderSettings: (data: DSRReminderSettingsUpdate) =>
    api.put<DSRReminderSettings>("/dsr-reminder/settings", data),
  notifyMe: () =>
    api.post<{ created: boolean; reason: string; today_ist: string }>(
      "/dsr-reminder/notify-me",
    ),
  listAll: (params?: { year?: number; month?: number; employee_id?: number; status?: string; limit?: number }) =>
    api.get<DSRRow[]>("/dsr/all", { params }),
  get: (id: number) => api.get<DSRRow>("/dsr/" + id),
  create: (data: DSRCreatePayload) => api.post<DSRRow>("/dsr/me", data),
  update: (id: number, data: DSRUpdatePayload) => api.patch<DSRRow>("/dsr/" + id, data),
  remove: (id: number) => api.delete("/dsr/" + id),
  pendingToday: () =>
    api.get<DSRPendingTodayResponse>("/dsr-reminder/pending-today"),
  remindPendingToday: (user_ids: number[] = []) =>
    api.post<DSRManualRemindResponse>("/dsr-reminder/pending-today/remind", {
      user_ids,
    }),
};

export const cameras = {
  list: () => api.get("/cameras"),
  get: (id: number) => api.get(`/cameras/${id}`),
  create: (data: {
    name: string;
    location?: string;
    source_url: string;
    source_type?: string;
    camera_purpose: string;
    threshold?: number;
    interval_sec?: number;
    enabled?: boolean;
  }) => api.post("/cameras", data),
  update: (id: number, data: {
    name?: string;
    location?: string;
    source_url?: string;
    source_type?: string;
    camera_purpose?: string;
    threshold?: number;
    interval_sec?: number;
    enabled?: boolean;
  }) => api.put(`/cameras/${id}`, data),
  remove: (id: number) => api.delete(`/cameras/${id}`),
  start: (id: number) => api.post(`/cameras/${id}/start`),
  stop: (id: number) => api.post(`/cameras/${id}/stop`),
  restart: (id: number) => api.post(`/cameras/${id}/restart`),
  status: (id: number) => api.get(`/cameras/${id}/status`),
  previewUrl: (id: number) => `/api/cameras/${id}/preview.jpg`,
  testConnection: (data: { source_url: string; source_type?: string }) =>
    api.post("/cameras/test-connection", data),
  stats: () => api.get("/cameras/stats"),
  discoverDVR: (data: { ip: string; port: number; username: string; password: string }) =>
    api.post("/dvr/discover", data),
};

export const dvr = {
  connect: (data: { ip: string; port: number; username: string; password: string }) =>
    api.post("/dvr/connect", data),
  disconnect: () => api.post("/dvr/disconnect"),
  status: () => api.get("/dvr/status"),
  startCamera: (channelId: number) => api.post(`/dvr/cameras/${channelId}/start`),
  stopCamera: (channelId: number) => api.post(`/dvr/cameras/${channelId}/stop`),
  setRecognition: (channelId: number, enabled: boolean) =>
    api.post(`/dvr/cameras/${channelId}/recognition`, null, { params: { enabled } }),
  startAll: () => api.post("/dvr/cameras/start-all"),
  stopAll: () => api.post("/dvr/cameras/stop-all"),
  previewUrl: (channelId: number) => `/api/dvr/cameras/${channelId}/preview`,
  streamUrl: (channelId: number) => `/api/dvr/cameras/${channelId}/stream`,
};



