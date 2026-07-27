# Playwright Testing Guide: Attendance & HRMS Application

This document provides a comprehensive overview of the "Attendance & HRMS" application to help a Playwright agent effectively write and execute end-to-end tests.

## 1. Environment & Architecture

- **Frontend**: React (Vite), listening on `http://localhost:5173`
- **Backend**: FastAPI (Python), listening on `http://localhost:8000` (API server)
- **Database**: PostgreSQL (connected via the backend)

*Note: For Playwright testing, the primary entry point is the frontend URL (`http://localhost:5173`).*

## 2. Authentication & Roles

The application uses role-based access control. The three primary roles are:
1. **Admin**: Has full access to everything, including `/users` to manage application users.
2. **HR**: Has access to manage Employees, Attendance, Leave Allocations, and Payroll.
3. **Employee**: Has restricted access. Can view their own profile, apply for leave, view attendance, and download their payslips.

**Testing Note:** Tests should cover scenarios for each specific role. Be mindful of logging out and logging back in when testing different permission levels.

## 3. Common UI Selectors & Patterns

The application uses standard semantic HTML combined with custom utility classes. Avoid relying on complex XPath locators; use these predictable CSS classes instead:

### Buttons & Actions
- **Primary Actions (Save, Submit, Add)**: `.btn.btn-primary`
- **Secondary Actions (Cancel, Close)**: `.btn.btn-secondary`
- **Danger Actions (Delete)**: `.btn.btn-danger`
- **Icon Buttons (Edit, Delete in tables)**: `.btn.btn-icon`

### Forms & Inputs
- **Form Containers**: Forms are typically standard `<form>` elements.
- **Input Fields**: Standard `<input>` and `<select>` tags inside a `.form-group` wrapper.
- **Password Toggles**: `.password-field` container with a `.password-toggle` button to show/hide the password.

### Layout & Containers
- **Cards (Main content areas)**: `.card`
- **Page Headers**: `.page-header` containing a `.page-title`
- **Tables**: Data is displayed in `.table-modern` or `.table-modern--dark` wrapped inside `.table-wrap`.
- **Modals (Popups)**: Overlay is `.modal-backdrop`, and the dialog box itself is `.modal`. Wait for `.modal` to be visible when testing popups.
- **Loading States**: Look for `.app-loading-screen` or `.section-loader` to wait for content to finish loading.

## 4. Key Routes & Pages

Here are the main URLs and their purpose:

- `/login` - **Login Page**.
  - Username locator: `input[type="text"]` or `input[placeholder="Enter your username"]`
  - Password locator: `input[type="password"]`
  - Submit button: `button[type="submit"]` or `.login-submit`
- `/signup` - **Signup Page** (Used for creating the initial admin account).
- `/` - **Dashboard**. Overview statistics and quick actions.
- `/employees` - **Employee Management**. List, add, edit, and delete employees.
- `/users` - **User Management**. Create users and assign roles (Admin only).
- `/attendance` - **Attendance Tracking**. View and log clock-ins/outs.
- `/leave` - **Leave Application**. Where employees request time off.
- `/leave-approvals` - **Leave Approvals**. Where HR/Admins approve or reject leave requests.
- `/payroll` - **Payroll Periods**. Create periods (e.g., May 2026) and run payroll.
- `/payslip-management` - **Payslips**. Generate, view, and download individual payslips.

## 5. Example Playwright Test Flow (Login)

```typescript
import { test, expect } from '@playwright/test';

test('Should login successfully', async ({ page }) => {
  // 1. Navigate to the frontend
  await page.goto('http://localhost:5173/login');
  
  // 2. Fill in credentials
  await page.fill('input[placeholder="Enter your username"]', 'admin');
  await page.fill('input[placeholder="Enter your password"]', 'password123');
  
  // 3. Submit the form
  await page.click('button[type="submit"]');
  
  // 4. Wait for navigation and verify Dashboard is visible
  await expect(page).toHaveURL('http://localhost:5173/');
  await expect(page.locator('.page-title')).toContainText('Dashboard');
});
```

## 6. Tips for the Agent

1. **State Management**: The app uses React state heavily. After submitting a form (e.g., adding an employee), wait for the table or list to update by waiting for the new text/row to appear.
2. **Debouncing & API Calls**: Many actions (like saving) trigger API calls. Wait for `.alert-success` (if applicable) or for the `.modal` to disappear before proceeding to the next step.
3. **Dropdowns**: The application uses a custom `<CustomSelect>` component in many places. You may need to click the select container and then click the list item instead of using the native Playwright `.selectOption()` if it's not a native `<select>`.
