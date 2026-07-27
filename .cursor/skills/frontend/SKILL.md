---
name: frontend
description: >-
  React 18 + TypeScript + Vite frontend for Softwiz HRMS (custom CSS, dark
  theme) and the standalone Resume Analyzer UI (Tailwind CDN). Activate whenever
  the user edits any file under `Attendance Management/frontend/src/`, edits
  `frontend/index.html` for the Resume Analyzer, adds or changes a React page /
  component / form / table, configures React Router 6, makes Axios API calls
  via `src/api/client`, touches `src/index.css`, uses HRMS CSS classes
  (`.page-header`, `.dash-card`, `.card`, `.table-modern--dark`,
  `.table-wrap--dark`, `.btn`, `.btn-primary`, `.sidebar-brand`, `--brand-*`),
  or mentions layout, alignment, spacing, header, sidebar, table, modal, dark
  theme, responsive, Vite dev server, HMR, npm run dev / build. HRMS uses
  custom CSS, NOT Bootstrap.
---

# Activation signals

Apply this skill when ANY of these are true:

- Editing files under `Attendance Management/frontend/src/`.
- Editing `frontend/index.html` (the standalone Resume Analyzer UI).
- File extension is `.tsx`, `.ts`, `.jsx`, `.js`, `.css`, or `.html` inside a frontend folder.
- User mentions: page, component, form, table, button, layout, alignment, spacing, header, sidebar, card, modal, route, navigation, axios, JWT in browser, dark theme, `--brand`, `.page-header`, `.dash-card`, `.btn-primary`, npm, vite, HMR, dev server.

Skip when the task is purely Python or DB — use `backend` instead.

# Two frontends, two style systems

| Surface | Lives in | Styling | Build |
|---|---|---|---|
| HRMS app | `Attendance Management/frontend/src/` | Custom CSS in `src/index.css` (dark theme, `--brand-*` vars) | Vite → `Attendance Management/backend/frontend_build/` |
| Resume Analyzer (separate tab) | `frontend/index.html` | Tailwind CDN + inline `<style>` block | None — served as static by FastAPI |

HRMS is the canonical look. When restyling the Resume Analyzer to match HRMS, copy the same CSS variables (`--brand-300`, `--brand-rgb`, `--card`, `--border`, `--header-height`) into the inline `<style>` block of `frontend/index.html` as `--hrms-*` mirrors and recreate the matching component classes there (e.g., `.hrms-page-header`, `.hrms-table-modern`).

# Hard rules

- HRMS uses **custom CSS, NOT Bootstrap**. Do NOT import Bootstrap, MUI, Chakra, etc.
- Reuse components in `src/components/`. Search before creating: `TableToolbar`, `SectionLoader`, `CustomSelect`, `ConfirmModal`, `GlobalHeaderControls`, `SortableHeader`, `SalaryFormulaView`.
- All API calls go through `src/api/client.ts` so JWT and base URL are handled centrally. Do NOT call `axios.get(...)` directly in pages.
- Icons are inline SVG with `stroke="currentColor"`, NOT emojis. Match the icon style in `DashboardLayout.tsx` and `Dashboard.tsx`.
- Dark-theme tables MUST use `.table-wrap--dark` + `.table-modern--dark`. Do not roll your own table styling.
- Pages MUST use `<div className="page-header">` at the top so the global gradient, sticky behavior, and spacing apply.

# UI / layout consistency rules

- Numeric / currency columns: `textAlign: 'right'` + `fontVariantNumeric: 'tabular-nums'`.
- Text columns: left aligned.
- Action button columns: centered, with `paddingRight: '1.5rem'` to mirror the first column's `paddingLeft: '1.5rem'`.
- Header `<th>` `textAlign` MUST match the body `<td>` `textAlign` in the same column.
- DO NOT add `<div style={{ justifyContent: ... }}>` wrappers inside `<td>` to override alignment — set `textAlign` on the cell.
- Action button rows (Apply/Clear, Save/Cancel, Submit/Reset) sit at the right of their card via `justify-end` flex; primary button on the far right, secondary to its left.
- Forms use the existing `.form-group`, `.alert`, `.btn`, `.btn-primary`, `.btn-secondary` classes.
- Spacing rule: page header `margin-bottom: 1.25rem` provides the gap to the next section — DO NOT add a negative `marginTop` to the next element.

# Development workflow (read this — most users miss it)

For DAILY development, do NOT use `npm run build`. Use the Vite dev server with hot module replacement (changes appear in ~100 ms, no rebuild).

```powershell
# Terminal 1: backend (provides /api, /resume-api on 5001)
python run_app.py

# Terminal 2: frontend with HMR
cd "Attendance Management/frontend"
npm run dev
```

Open http://localhost:5173/ — the Vite proxy in `vite.config.ts` forwards `/api`, `/resume-api`, `/resume`, `/portal.html` to the backend on 5001.

`npm run build` is ONLY needed to:

1. Verify the production bundle at http://127.0.0.1:5001 still works.
2. Prepare the deploy zip (see `deployment` skill).

# Workflow: adding a new HRMS page

1. Create `src/pages/MyNewPage.tsx`. Model on `Employees.tsx` or `Dashboard.tsx`.
2. Wrap content in `<div className="page-header">...</div>` then the rest.
3. Add the route in the router.
4. Add a sidebar entry in `src/layouts/DashboardLayout.tsx` with a `<NavLink>` + inline SVG icon matching the existing pattern.
5. All data calls go through `src/api/client.ts`.

# Workflow: fixing alignment / restyle

1. Find the offending file under `src/pages/` or `src/components/`.
2. Prefer existing CSS classes from `src/index.css`. Inline `style` only for one-off tweaks.
3. NEVER override `text-align` with a flex wrapper — set it on `<td>` / `<th>` directly.
4. If both header and body need the same alignment: change both in one edit so they stay paired.

# Workflow: editing the standalone Resume Analyzer (`frontend/index.html`)

- Single static file; no build needed. Hard-refresh (Ctrl+Shift+R) at `/resume/`.
- Add new CSS rules inside the existing `<style>` block before `</style>`.
- Preserve ALL `id="..."` hooks — the inline `<script>` references them.
- To match HRMS look: use the `.hrms-*` classes already defined in the file (`.hrms-page-header`, `.hrms-sidebar-brand`, `.hrms-table-wrap`, `.hrms-table-modern`).

# Stop and ask the user when

- A change would break a page's responsive collapse (`hide-md`, `hide-sm`).
- About to install a new UI framework — HRMS sticks to custom CSS.
- A new component would duplicate an existing one.
- You'd need to change the URL surface (route paths) — that may touch backend routing too.

# Verification (run before declaring done)

- During dev: changes auto-reload at `localhost:5173`. Manually click through the affected page.
- Before shipping: `cd "Attendance Management/frontend" ; npm run build`, then load `localhost:5001` and verify.
- Browser console has no red errors on the affected page.
- For Resume Analyzer edits: hard-refresh `localhost:5001/resume/` and click through the affected tab.
- Network tab: API calls return 200; no `CORS` errors; the `Authorization: Bearer ...` header is present.

# Related skills

- Backend changes that surface in the UI → `backend` skill.
- `npm run build`, deploy zip, Windows ports → `deployment` skill.
- Visual / functional QA checklist → `testing` skill.
