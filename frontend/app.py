import os
import re
import json
import time
from datetime import datetime,timedelta
from io import BytesIO
from typing import Optional

import pandas as pd
import requests
import streamlit as st
from sqlalchemy import create_engine, text

# Backend base URL. Override via environment for LAN access (e.g. http://192.168.29.235:8001)
API_URL = os.getenv("API_URL", "http://127.0.0.1:8001").rstrip("/")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_str(v: object, fallback: str = "Not specified") -> str:
    s = "" if v is None else str(v)
    return s.strip() if s.strip() else fallback


def _join_list(v: object, fallback: str = "Not specified") -> str:
    if isinstance(v, list):
        items = [str(x).strip() for x in v if str(x).strip()]
        return ", ".join(items) if items else fallback
    return _safe_str(v, fallback=fallback)


def _name_only(name: object) -> str:
    """Return name with any embedded email removed (name column = name only)."""
    if name is None or name == "":
        return "—"
    s = str(name).strip()
    if not s:
        return "—"
    # Remove email-like substrings (e.g. word@domain.tld)
    s = re.sub(r"\S+@\S+\.\S+", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s if s else "—"


def _format_years_to_duration(exp_years: object) -> str:
    try:
        y = float(exp_years or 0.0)
    except Exception:
        y = 0.0
    if y <= 0:
        return "Not specified"
    total_months = int(round(y * 12))
    yrs, mos = divmod(total_months, 12)
    parts: list[str] = []
    if yrs:
        parts.append(f"{yrs} year{'s' if yrs != 1 else ''}")
    if mos:
        parts.append(f"{mos} month{'s' if mos != 1 else ''}")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Export to Google Sheets helpers
# ---------------------------------------------------------------------------

# Reuse the same DATABASE_URL as backend (falls back to same default)
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:root@localhost:5432/Resume_analyzer",
)


def _get_engine():
    """Create SQLAlchemy engine for PostgreSQL."""
    return create_engine(DATABASE_URL)


def fetch_resume_data_for_export() -> pd.DataFrame:
    """Fetch resume data used for export to Google Sheets."""
    engine = _get_engine()
    query = text(
        """
        SELECT
            id,
            name AS candidate_name,
            email,
            phone,
            primary_skills,
            other_skills,
            experience_years AS experience,
            resume_link,
            created_at
        FROM resumes
        WHERE deleted_at IS NULL
        ORDER BY created_at DESC
        """
    )
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    # Normalise skills columns to strings
    for col in ("primary_skills", "other_skills"):
        if col in df.columns:
            df[col] = df[col].astype(str).fillna("")

    return df


def connect_google_sheet(creds_json_bytes: bytes, sheet_name: str):
    """
    Authenticate using service-account JSON bytes and return (client, worksheet).
    Creates the sheet if it doesn't exist.
    """
    try:
        import gspread
        from oauth2client.service_account import ServiceAccountCredentials
    except Exception as exc:
        raise RuntimeError(
            "Google Sheets export dependencies are missing. Install:\n\n"
            "  pip install gspread oauth2client\n"
        ) from exc

    creds_dict = json.loads(creds_json_bytes.decode("utf-8"))
    scope = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive",
    ]
    credentials = ServiceAccountCredentials.from_json_keyfile_dict(
        creds_dict, scopes=scope
    )
    client = gspread.authorize(credentials)

    try:
        sh = client.open(sheet_name)
    except gspread.SpreadsheetNotFound:
        sh = client.create(sheet_name)

    try:
        ws = sh.sheet1
    except gspread.exceptions.WorksheetNotFound:
        ws = sh.add_worksheet(title="Sheet1", rows="1000", cols="20")

    return client, ws


def _get_existing_ids(ws) -> list[str]:
    """Read existing IDs from first column (excluding header)."""
    vals = ws.col_values(1)
    if not vals:
        return []
    return [str(v).strip() for v in vals[1:] if str(v).strip()]


def export_to_google_sheet(ws, df: pd.DataFrame) -> None:
    """
    Export DataFrame to worksheet:
    - Writes headers if sheet is empty.
    - Skips rows whose id is already present.
    """
    if df.empty:
        return
    if "id" not in df.columns:
        raise ValueError("DataFrame must include 'id' column.")

    df = df.copy()
    df["id"] = df["id"].astype(str)

    existing_ids = set(_get_existing_ids(ws))
    new_rows = df[~df["id"].isin(existing_ids)]
    if new_rows.empty:
        return

    headers = list(new_rows.columns)
    rows = new_rows.values.tolist()

    if not ws.get_all_values():
        ws.append_row(headers)

    ws.append_rows(rows, value_input_option="RAW")


def render_export_to_google_sheets_tab():
    """Streamlit UI for exporting DB resumes to Google Sheets."""
    st.subheader("Export database to Google Sheets")

    sheet_name = st.text_input("Google Sheet name", value="Resume_Export")

    # OLD (file upload) – remove or comment this
    # creds_file = st.file_uploader(
    #     "Upload Google Service Account JSON credentials",
    #     type=["json"],
    #     key="gsheets_creds",
    # )

    # NEW: paste JSON credentials directly
    creds_text = st.text_area(
        "Paste Google Service Account JSON credentials",
        height=200,
        key="gsheets_creds_text",
    )

    if st.button("Export Database to Google Sheets", key="btn_export_to_gsheets"):
        if not sheet_name.strip():
            st.error("Please enter a Google Sheet name.")
            return
        if not creds_text.strip():
            st.error("Please paste the service account JSON.")
            return

        with st.spinner("Exporting resumes to Google Sheets…"):
            try:
                df = fetch_resume_data_for_export()
                if df.empty:
                    st.warning("No resumes found in the database.")
                    return

                # Convert pasted text to bytes for connect_google_sheet()
                creds_bytes = creds_text.encode("utf-8")
                _, ws = connect_google_sheet(creds_bytes, sheet_name.strip())
                export_to_google_sheet(ws, df)

                st.success("Export completed successfully.")
                st.markdown(
                    f"**Sheet URL:** [Open in Google Sheets]({ws.spreadsheet.url})"
                )
            except json.JSONDecodeError:
                st.error("The pasted credentials are not valid JSON.")
            except Exception as e:
                # Handle missing gspread/oauth2client + API errors without breaking app import.
                msg = str(e)
                if "gspread" in msg or "oauth2client" in msg or "dependencies are missing" in msg.lower():
                    st.error(str(e))
                    st.caption("This does not affect resume parsing; only the Export tab needs these packages.")
                else:
                    st.error(f"Unexpected error: {e}")
            except Exception as e:
                st.error(f"Unexpected error: {e}")
# ---------------------------------------------------------------------------
# Skills dialog (modal) — compatible across Streamlit versions
# ---------------------------------------------------------------------------

_dialog_decorator = getattr(st, "dialog", None) or getattr(st, "experimental_dialog", None)

import html

def _render_skill_chips_html(title: str, icon: str, skills: list[str]) -> str:
    """Return HTML for a skill section (badges)."""
    if not skills:
        content = "<div style='color: rgba(255,255,255,0.4); margin-top: 4px;'>—</div>"
    else:
        chips_html = "".join(f'<div class="skill-chip">{html.escape(str(s))}</div>' for s in skills)
        content = f'<div class="chips-container">{chips_html}</div>'

    return f"""
<div class="skills-section">
  <div class="skills-title">{icon} {title}</div>
  {content}
</div>
"""


def _view_skills_dialog_body(resume_id: str) -> None:
    """
    Fetch candidate from backend using ID and show skills as chips:
    - Primary Skills (key_skills)
    - Other Skills (skills)
    """

    with st.spinner("Loading skills..."):
        try:
            resp = requests.get(f"{API_URL}/resumes/{resume_id}/json", timeout=15)
            resp.raise_for_status()
            resume = resp.json()
        except Exception as exc:
            resume = None
            st.error(f"Could not load skills: {exc}")

    if not resume:
        st.caption("Candidate not found.")
        if st.button("Close", type="primary", key=f"skills_notfound_close_{resume_id}"):
            st.rerun()
        return

    # Prefer backend-classified primary/other skills when available
    primary = resume.get("primary_skills") or resume.get("key_skills") or []
    if not isinstance(primary, list):
        primary = [s.strip() for s in str(primary).split(",") if s.strip()]

    # Backend may already send other_skills; otherwise, derive from full skills list
    other = resume.get("other_skills") or resume.get("skills") or []
    if not isinstance(other, list):
        other = [s.strip() for s in str(other).split(",") if s.strip()]

    # Normalise + de-duplicate, enforce exactly "top 5" primary skills from backend ordering
    def _clean_list(items: list[str], candidate_name: str = "") -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        name_parts = {p.lower() for p in (candidate_name or "").split() if len(p) > 2}
        
        for raw in items:
            s = str(raw).strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            # Filter out candidate's own name parts
            if key in name_parts:
                continue
            # Filter out common noise
            if key in {"candidate", "resume", "curriculum", "vitae", "name", "contact"}:
                continue

            seen.add(key)
            out.append(s)
        return out

    primary = _clean_list(primary, resume.get("name") or "")[:5]
    other = _clean_list(other, resume.get("name") or "")

    # Ensure other skills do not repeat primary skills
    primary_norm = {s.lower() for s in primary}
    other = [s for s in other if s.lower() not in primary_norm]

    # When Streamlit supports dialogs, this function is rendered inside the native modal.
    # In that case, do NOT inject a second fixed-position modal overlay (it breaks layout).
    st.markdown(
        """
<style>
.skills-title { font-size: 1.05rem; font-weight: 750; margin: 10px 0 12px 0; color: #ffffff; display: flex; align-items: center; gap: 10px; }
.chips-container { display: flex; flex-wrap: wrap; gap: 10px; }
.skill-chip {
    padding: 8px 14px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 12px;
    font-size: 0.95rem;
    color: rgba(255,255,255,0.9);
}
.skills-divider { height: 1px; background: rgba(255,255,255,0.1); margin: 18px 0; }
</style>
""",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='font-size:1.35rem; font-weight:800; margin-bottom:0.25rem;'>Candidate Skills</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div class='skills-divider'></div>", unsafe_allow_html=True)
    st.markdown(_render_skill_chips_html("Primary Skills", "⭐", primary[:25]), unsafe_allow_html=True)
    st.markdown("<div class='skills-divider'></div>", unsafe_allow_html=True)
    st.markdown(_render_skill_chips_html("Other Skills", "🧰", other[:50]), unsafe_allow_html=True)

    # No extra close button here: the overlay modal provides a single Close action,
    # and Streamlit dialogs already include a close affordance.


if _dialog_decorator is not None:
    _view_skills_dialog = _dialog_decorator("View Skills")(_view_skills_dialog_body)  # type: ignore[misc]
else:
    def _view_skills_dialog(resume_id: str) -> None:  # type: ignore[no-redef]
        # Fallback: render inline without a modal.
        _view_skills_dialog_body(resume_id)


# ---------------------------------------------------------------------------
# Skills modal overlay (always works; independent of st.dialog)
# ---------------------------------------------------------------------------

def _open_skills_modal(resume_id: str) -> None:
    st.session_state["skills_modal_rid"] = resume_id
    st.session_state["skills_modal_open"] = True


def _close_skills_modal() -> None:
    st.session_state["skills_modal_open"] = False
    st.session_state.pop("skills_modal_rid", None)


def _render_skills_modal_if_open() -> None:
    """
    Render a fixed-position modal overlay for skills.
    This avoids Streamlit dialog edge cases and guarantees a popup UX.
    """
    if not st.session_state.get("skills_modal_open"):
        return
    rid = str(st.session_state.get("skills_modal_rid") or "").strip()
    if not rid:
        _close_skills_modal()
        return

    # Fetch resume JSON (includes skills lists).
    with st.spinner("Loading skills..."):
        try:
            resp = requests.get(f"{API_URL}/resumes/{rid}/json", timeout=15)
            resp.raise_for_status()
            resume = resp.json()
        except Exception as exc:
            resume = None
            st.error(f"Could not load skills: {exc}")

    if not resume:
        if st.button("Close", type="primary", key="skills_modal_close_empty"):
            _close_skills_modal()
            st.rerun()
        return

    primary = resume.get("primary_skills") or resume.get("key_skills") or []
    if not isinstance(primary, list):
        primary = [s.strip() for s in str(primary).split(",") if s.strip()]

    other = resume.get("other_skills") or resume.get("skills") or []
    if not isinstance(other, list):
        other = [s.strip() for s in str(other).split(",") if s.strip()]

    # Basic cleanup / de-dupe
    def _clean_list(items: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in items:
            s = str(raw).strip()
            if not s:
                continue
            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out

    primary = _clean_list(primary)[:5]
    other = [s for s in _clean_list(other) if s.lower() not in {x.lower() for x in primary}]

    sidebar_hidden = bool(st.session_state.get("sidebar_hidden"))
    modal_left = "50%" if sidebar_hidden else "56%"
    modal_width = "min(860px, 92vw)" if sidebar_hidden else "min(820px, 86vw)"

    # IMPORTANT: avoid `.format()`/f-strings on raw CSS (it contains `{}`).
    # Use placeholder replacement instead.
    css = """
<style>
/* Backdrop */
.skills-modal-backdrop{
  position: fixed; inset: 0;
  background: rgba(0,0,0,0.70);
  z-index: 100001;
  pointer-events: auto;
}
/* Centered modal */
.skills-modal{
  position: fixed;
  top: 50%; left: __MODAL_LEFT__;
  transform: translate(-50%, -50%);
  width: __MODAL_WIDTH__;
  max-height: 84vh;
  overflow: auto;
  background: #0e1117;
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 18px;
  box-shadow: 0 28px 90px rgba(0,0,0,0.75);
  padding: 22px 22px 14px 22px;
  z-index: 100002;
  pointer-events: auto;
}
.skills-modal-title{
  font-size: 1.35rem;
  font-weight: 850;
  margin: 0 0 8px 0;
}
.skills-divider { height: 1px; background: rgba(255,255,255,0.10); margin: 14px 0; }
.skills-title { font-size: 1.05rem; font-weight: 750; margin: 10px 0 12px 0; display:flex; align-items:center; gap:10px; }
.chips-container { display:flex; flex-wrap:wrap; gap:10px; }
.skill-chip{
  padding: 8px 14px;
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.14);
  border-radius: 12px;
  font-size: 0.95rem;
  color: rgba(255,255,255,0.92);
}

/* Pin the modal Close button deterministically:
   we render <div id="skills-close-anchor"></div> right before the Close button,
   then style the *next* Streamlit button block. */
#skills-close-anchor + div[data-testid="stButton"]{
  position: fixed !important;
  top: 8vh !important;
  right: 5vw !important;
  z-index: 100003 !important;
}
#skills-close-anchor + div[data-testid="stButton"] button{
  border-radius: 10px !important;
  padding: 0.35rem 0.7rem !important;
}
</style>
<div class="skills-modal-backdrop"></div>
        """
    css = css.replace("__MODAL_LEFT__", modal_left).replace("__MODAL_WIDTH__", modal_width)
    st.markdown(css, unsafe_allow_html=True)

    primary_html = _render_skill_chips_html("Primary Skills", "⭐", primary[:25])
    other_html = _render_skill_chips_html("Other Skills", "🧰", other[:60])
    st.markdown(
        f"""
<div class="skills-modal">
  <div class="skills-modal-title">Candidate Skills</div>
  <div class="skills-divider"></div>
  {primary_html}
  <div class="skills-divider"></div>
  {other_html}
</div>
        """,
        unsafe_allow_html=True,
    )

    # Close action
    st.markdown('<div id="skills-close-anchor"></div>', unsafe_allow_html=True)
    if st.button("Close", type="primary", key=f"skills_modal_close_{rid}"):
        _close_skills_modal()
        st.rerun()


# ---------------------------------------------------------------------------
# View candidate details dialog (lightweight profile view)
# ---------------------------------------------------------------------------

def _view_candidate_dialog_body(resume_id: str, row: Optional[dict] = None) -> None:
    """Show candidate details: name, email, phone, experience, primary skills, resume link."""
    if row:
        r = row
    else:
        try:
            resp = requests.get(f"{API_URL}/resumes", timeout=30)
            resp.raise_for_status()
            all_resumes = resp.json() or []
            r = next((x for x in all_resumes if x.get("id") == resume_id), None)
        except Exception as exc:
            r = None
            st.error(f"Could not load candidate: {exc}")
    if not r:
        st.caption("Candidate not found.")
        if st.button("Close", type="primary"):
            st.rerun()
        return
    st.markdown(f"### {_safe_str(r.get('name'), fallback='—')}")
    st.caption(f"ID: {resume_id}")
    st.write("**Email:** ", _safe_str(r.get("email"), fallback="—"))
    st.write("**Phone:** ", _safe_str(r.get("phone"), fallback="—"))
    st.write("**Experience:** ", _format_years_to_duration(r.get("experience_years")))
    primary = r.get("primary_skills") or r.get("key_skills") or []
    if not isinstance(primary, list):
        primary = [s.strip() for s in str(primary).split(",") if s.strip()]
    st.write("**Primary Skills:** ", ", ".join(primary[:10]) if primary else "—")
    link = r.get("resume_link")
    if link:
        st.markdown(f"**Resume:** [Open file]({link})")
    if st.button("Close", type="primary"):
        st.rerun()


if _dialog_decorator is not None:
    _view_candidate_dialog = _dialog_decorator("Candidate Details")(_view_candidate_dialog_body)  # type: ignore[misc]
else:
    def _view_candidate_dialog(resume_id: str, row: Optional[dict] = None) -> None:  # type: ignore[no-redef]
        _view_candidate_dialog_body(resume_id, row)


# ---------------------------------------------------------------------------
# Delete confirmation dialog
# ---------------------------------------------------------------------------

def _delete_confirm_dialog_body(resume_id: str) -> None:
    st.markdown("**Are you sure you want to delete this candidate?**")
    st.caption("This will remove the candidate record and any related notes and data.")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Yes", type="primary", key="delete_confirm_yes"):
            try:
                resp = requests.delete(
                    f"{API_URL}/resumes/bulk",
                    json={"resume_ids": [resume_id]},
                    timeout=10,
                )
                resp.raise_for_status()
                invalidate_resume_cache()
                st.session_state.pop("delete_confirm_rid", None)
                st.session_state.pop("delete_confirm_reopen", None)
                st.success("Candidate deleted.")
                st.rerun()
            except Exception as e:
                st.error(str(e))
    with col2:
        if st.button("Cancel", key="delete_confirm_cancel"):
            st.session_state.pop("delete_confirm_rid", None)
            st.session_state.pop("delete_confirm_reopen", None)
            st.rerun()
    with col3:
        # Explicit close button (in addition to dialog close "X")
        if st.button("Close", key="delete_confirm_close", type="secondary"):
            st.session_state.pop("delete_confirm_rid", None)
            st.session_state.pop("delete_confirm_reopen", None)
            st.rerun()


if _dialog_decorator is not None:
    _delete_confirm_dialog = _dialog_decorator("Delete candidate")(_delete_confirm_dialog_body)  # type: ignore[misc]
else:
    def _delete_confirm_dialog(resume_id: str) -> None:  # type: ignore[no-redef]
        _delete_confirm_dialog_body(resume_id)


# ---------------------------------------------------------------------------
# Notes dialog (modal) — view all notes, add new, save via API
# ---------------------------------------------------------------------------

def _view_notes_dialog_body(resume_id: str, candidate_name: str = "") -> None:
    """
    Primary Notes dialog:
    - Add new note
    - Button to open full "View Notes" dialog
    """
    # Header
    st.markdown(
        """
        <style>
        .notes-wrap { max-width: 760px; margin: 0 auto; }
        .notes-meta { color: rgba(255,255,255,0.72); font-size: 0.92rem; margin-top: -0.2rem; }
        .notes-list {
            max-height: 420px;
            overflow: auto;
            padding-right: 6px;
            margin-top: 0.4rem;
        }
        .notes-list::-webkit-scrollbar { width: 8px; }
        .notes-list::-webkit-scrollbar-thumb { background: rgba(255,255,255,0.10); border-radius: 999px; }
        .note-card {
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 14px;
            padding: 10px 10px 9px 10px;
            background: rgba(255,255,255,0.015);
            box-shadow: 0 10px 22px rgba(0,0,0,0.18);
            margin: 0.30rem 0;
        }
        .note-head { display: flex; align-items: baseline; gap: 0.55rem; }
        .note-date { color: rgba(255,255,255,0.55); font-size: 0.85rem; }
        .note-body { color: rgba(255,255,255,0.92); font-size: 0.95rem; margin-top: 0.3rem; white-space: pre-wrap; }
        .note-pill {
            display: inline-flex; align-items: center;
            padding: 0.12rem 0.5rem;
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.14);
            background: rgba(255,255,255,0.04);
            color: rgba(255,255,255,0.80);
            font-size: 0.80rem;
            margin-top: 0.5rem;
        }
        .note-actions button {
            background: transparent !important;
            border: none !important;
            box-shadow: none !important;
            padding: 0.0rem 0.1rem !important;
            min-height: 1.0rem !important;
            color: rgba(255,255,255,0.72) !important;
            font-weight: 800 !important;
        }
        .note-actions button:hover { color: rgba(255,255,255,0.92) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="notes-wrap">', unsafe_allow_html=True)
    #st.subheader("Notes")
    st.caption((candidate_name or resume_id).strip())

    # One-shot success/info message for add/delete actions
    flash_key = f"notes_flash_{resume_id}"
    flash_msg = st.session_state.pop(flash_key, None)
    if flash_msg:
        st.success(flash_msg)

    # Fetch existing notes from FastAPI
    try:
        resp = requests.get(f"{API_URL}/resumes/{resume_id}/notes", timeout=10)
        resp.raise_for_status()
        notes_list = resp.json() or []
    except Exception as e:
        st.error(f"Could not load notes: {e}")
        notes_list = []

    # Add note: compact form
    new_note = st.text_area("New note", key=f"notes_input_{resume_id}", height=60)
    status_options = ["", "New", "Screening", "Phone Call", "Follow-Up", "Contacted", "Interview", "Technical Interview", "Rejected", "Offer"]
    note_status = st.selectbox("Status", status_options, key=f"notes_status_{resume_id}")

    b1, b2, _ = st.columns([1, 1, 3])
    with b1:
        if st.button("Save", type="primary", key=f"notes_save_{resume_id}"):
            if not new_note.strip():
                st.warning("Enter note text.")
            else:
                try:
                    requests.post(
                        f"{API_URL}/resumes/{resume_id}/notes",
                        json={"note": new_note.strip(), "status": note_status or None},
                        timeout=10,
                    )
                    st.session_state[flash_key] = "Note added successfully."
                    st.session_state["notes_dialog_rid"] = resume_id
                    st.session_state["notes_dialog_name"] = candidate_name
                    st.session_state["notes_dialog_reopen_after_save"] = True
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    with b2:
        if st.button("Close", type="secondary", key=f"notes_close_{resume_id}"):
            st.session_state.pop("notes_dialog_rid", None)
            st.session_state.pop("notes_dialog_reopen_after_save", None)
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)  # notes-wrap


def _view_notes_list_dialog_body(resume_id: str, candidate_name: str = "") -> None:
    """
    Secondary dialog: list all notes with expander + delete.
    """
    st.subheader("Saved notes")
    st.caption((candidate_name or resume_id).strip())

    flash_key = f"notes_flash_{resume_id}"
    flash_msg = st.session_state.pop(flash_key, None)
    if flash_msg:
        st.success(flash_msg)

    try:
        resp = requests.get(f"{API_URL}/resumes/{resume_id}/notes", timeout=10)
        resp.raise_for_status()
        notes_list = resp.json() or []
    except Exception as e:
        st.error(f"Could not load notes: {e}")
        notes_list = []

    total = len(notes_list)
    if not total:
        st.info("No notes yet for this candidate.")
        if st.button("Close", type="secondary", key=f"notes_list_close_empty_{resume_id}"):
            st.session_state.pop("notes_list_dialog_rid", None)
            st.session_state.pop("notes_list_dialog_reopen", None)
            st.rerun()
        return

    # Pagination: 5 notes per page
    page_size = 5
    total_pages = max(1, (total + page_size - 1) // page_size)
    page_key = f"notes_view_page_{resume_id}"
    if page_key not in st.session_state:
        st.session_state[page_key] = 1
    st.session_state[page_key] = max(1, min(int(st.session_state[page_key]), total_pages))
    page = int(st.session_state[page_key])

    start = (page - 1) * page_size
    end = start + page_size
    page_notes = notes_list[start:end]

    st.caption(f"Showing {min(end, total)} of {total} notes (page {page}/{total_pages})")

    # Compact list with expanders; keep popup height under control
    st.markdown(
        """
        <div style="max-height: 420px; overflow:auto; padding-right:4px;">
        """,
        unsafe_allow_html=True,
    )

    for n in page_notes:
        note_id = n.get("id", "")
        status = (n.get("status") or "Note").strip()
        created = str(n.get("created_at") or "")[:10]
        body = (n.get("note") or "").strip()
        preview = (body[:50] + "…") if len(body) > 50 else (body or "(empty note)")

        with st.expander(f"{created} • {status} • {preview}"):
            c1, c2 = st.columns([6, 1])
            with c1:
                st.markdown(
                    f"""
                    <div style="max-height: 160px; overflow:auto; white-space:pre-wrap;">
                        {body or "(no text)"}
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with c2:
                if st.button("🗑", key=f"notes_list_del_{resume_id}_{note_id}"):
                    try:
                        requests.delete(
                            f"{API_URL}/resumes/{resume_id}/notes/{note_id}",
                            timeout=10,
                        )
                        st.session_state[flash_key] = "Note deleted successfully."
                        st.session_state["notes_list_dialog_rid"] = resume_id
                        st.session_state["notes_list_dialog_name"] = candidate_name
                        st.session_state["notes_list_dialog_reopen"] = True
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

    st.markdown("</div>", unsafe_allow_html=True)

    # Pager controls (Prev 1 2 ... Next)
    win = 7
    half = win // 2
    sp = max(1, page - half)
    ep = min(total_pages, sp + win - 1)
    sp = max(1, ep - win + 1)
    nums = list(range(sp, ep + 1))

    pager_cols = st.columns([0.8] + ([0.25] * len(nums)) + [0.8])
    if pager_cols[0].button("Prev", key=f"notes_view_prev_{resume_id}", disabled=(page <= 1), type="secondary"):
        st.session_state[page_key] = page - 1
        st.session_state["notes_list_dialog_rid"] = resume_id
        st.session_state["notes_list_dialog_name"] = candidate_name
        st.session_state["notes_list_dialog_reopen"] = True
        st.rerun()
    for i2, pnum in enumerate(nums, start=1):
        t = "primary" if pnum == page else "secondary"
        if pager_cols[i2].button(str(pnum), key=f"notes_view_p_{resume_id}_{pnum}", type=t):
            st.session_state[page_key] = pnum
            st.session_state["notes_list_dialog_rid"] = resume_id
            st.session_state["notes_list_dialog_name"] = candidate_name
            st.session_state["notes_list_dialog_reopen"] = True
            st.rerun()
    if pager_cols[-1].button("Next", key=f"notes_view_next_{resume_id}", disabled=(page >= total_pages), type="secondary"):
        st.session_state[page_key] = page + 1
        st.session_state["notes_list_dialog_rid"] = resume_id
        st.session_state["notes_list_dialog_name"] = candidate_name
        st.session_state["notes_list_dialog_reopen"] = True
        st.rerun()

    st.markdown("---")
    if st.button("Close", type="secondary", key=f"notes_list_close_{resume_id}"):
        st.session_state.pop("notes_list_dialog_rid", None)
        st.session_state.pop("notes_list_dialog_reopen", None)
        st.rerun()


if _dialog_decorator is not None:
    _view_notes_dialog = _dialog_decorator("Notes")(_view_notes_dialog_body)  # type: ignore[misc]
    _view_notes_list_dialog = _dialog_decorator("View Notes")(_view_notes_list_dialog_body)  # type: ignore[misc]
else:
    def _view_notes_dialog(resume_id: str, candidate_name: str = "") -> None:  # type: ignore[no-redef]
        _view_notes_dialog_body(resume_id, candidate_name)

    def _view_notes_list_dialog(resume_id: str, candidate_name: str = "") -> None:  # type: ignore[no-redef]
        _view_notes_list_dialog_body(resume_id, candidate_name)


# ---------------------------------------------------------------------------
# Resume list (cached – avoids calling the API once per tab on every rerun)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=15, show_spinner=False)
def fetch_resumes(params: str = "") -> tuple:
    """
    Fetch resumes from the backend.
    Returns (list_of_resumes, error_message_or_None).
    `params` is a URL query-string used as a cache key.
    """
    import urllib.parse
    p = dict(urllib.parse.parse_qsl(params)) if params else {}
    try:
        resp = requests.get(f"{API_URL}/resumes", params=p, timeout=30)
        if resp.status_code >= 400:
            return [], f"Backend error {resp.status_code}: {resp.text[:300]}"
        return resp.json(), None
    except requests.exceptions.ConnectionError:
        return [], "OFFLINE"
    except requests.exceptions.Timeout:
        return [], "STARTING"
    except Exception as exc:
        return [], str(exc)


def invalidate_resume_cache():
    fetch_resumes.clear()


@st.cache_data(ttl=15, show_spinner=False)
def fetch_indeed_leads(params: str = "") -> tuple:
    """
    Fetch Indeed leads (link-only applications) from backend.
    Returns (list_of_leads, error_message_or_None).
    """
    import urllib.parse

    p = dict(urllib.parse.parse_qsl(params)) if params else {}
    try:
        resp = requests.get(f"{API_URL}/indeed-leads", params=p, timeout=30)
        if resp.status_code >= 400:
            return [], f"Backend error {resp.status_code}: {resp.text[:300]}"
        return resp.json() or [], None
    except requests.exceptions.ConnectionError:
        return [], "OFFLINE"
    except requests.exceptions.Timeout:
        return [], "STARTING"
    except Exception as exc:
        return [], str(exc)


def invalidate_indeed_leads_cache():
    fetch_indeed_leads.clear()


# ---------------------------------------------------------------------------
# Backend health check
# ---------------------------------------------------------------------------

def _backend_status() -> str:
    """Returns 'ok', 'starting' (slow response), or 'offline' (no connection)."""
    try:
        r = requests.get(f"{API_URL}/health", timeout=8)
        return "ok" if r.status_code == 200 else "offline"
    except requests.exceptions.Timeout:
        return "starting"
    except Exception:
        return "offline"


def _backend_ok() -> bool:
    return _backend_status() == "ok"


# ---------------------------------------------------------------------------
# Candidate detail renderer (shared across tabs)
# ---------------------------------------------------------------------------

def render_candidate_details(
    resume: dict,
    rid: str,
    *,
    key_prefix: str = "",
    show_notes: bool = True,
    show_chat: bool = True,
):
    # Keep display name consistent with the Resumes table (name-only, no embedded email)
    display_name = _name_only(resume.get("name") or resume.get("email") or "Candidate")
    st.markdown(f"### {display_name}")
    st.caption(f"ID: {rid}")

    if resume.get("summary"):
        st.markdown("#### Overview")
        st.write(resume["summary"])

    st.markdown("#### Profile")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Email:** ", _safe_str(resume.get("email", ""), fallback="—"))
        st.write("**Phone:** ", _safe_str(resume.get("phone", ""), fallback="—"))
        st.write("**Experience:** ", _format_years_to_duration(resume.get("experience_years")))
        exp_line = resume.get("experience_line") or resume.get("experience_summary") or ""
        st.write("**Experience summary:** ", _safe_str(exp_line))
        st.write("**Skills:** ", _join_list(resume.get("skills")))
        if st.button("Recalculate experience from resume dates", key=f"{key_prefix}reextract_exp"):
            with st.spinner("Recalculating experience…"):
                try:
                    resp = requests.post(f"{API_URL}/resumes/{rid}/reextract-experience", timeout=60)
                    resp.raise_for_status()
                    out = resp.json()
                    if out.get("updated"):
                        fetch_resumes.clear()
                        st.success("Experience updated. Refreshing…")
                        st.rerun()
                    else:
                        st.warning(out.get("message", "No update performed."))
                except Exception as e:
                    st.error(str(e))
        if st.button("Re-extract name + skills + experience (recommended)", key=f"{key_prefix}reextract_skills"):
            with st.spinner("Re-extracting (full)…"):
                try:
                    resp = requests.post(f"{API_URL}/resumes/{rid}/reextract-key-skills", timeout=90)
                    resp.raise_for_status()
                    out = resp.json()
                    if out.get("updated"):
                        fetch_resumes.clear()
                        st.success("Re-extraction completed. Refreshing…")
                        st.rerun()
                    else:
                        st.warning(out.get("message", "No update performed."))
                except Exception as e:
                    st.error(str(e))
    with col2:
        st.write("**Education:** ", _join_list(resume.get("education")))
        st.write("**Role:** ", _safe_str(resume.get("role", ""), fallback="—"))
        st.write("**Companies:** ", _join_list(resume.get("companies_worked_at")))
        if resume.get("experience_tags"):
            st.write("**Experience tags:** ", _join_list(resume.get("experience_tags")))

    

    if resume.get("resume_link"):
        st.link_button("Open resume (PDF)", resume["resume_link"], type="secondary")

    email = (resume.get("email") or "").strip()
    if email:
        st.link_button(
            "Email candidate",
            f"mailto:{email}?subject=Regarding your application",
            type="primary",
        )

    if show_notes:
        st.subheader("Notes & status")
        try:
            notes_resp = requests.get(f"{API_URL}/resumes/{rid}/notes", timeout=5)
            notes_resp.raise_for_status()
            notes_list = notes_resp.json()
        except Exception:
            notes_list = []
        for n in notes_list:
            with st.expander(f"{n.get('status', 'Note')} – {str(n.get('created_at', ''))[:10]}"):
                st.write(n.get("note", ""))
        with st.form(f"{key_prefix}add_note_form"):
            new_note = st.text_area("New note", key=f"{key_prefix}new_note")
            new_status = st.selectbox(
                "Status",
                ["", "Contacted", "Interview", "Rejected", "Offer"],
                key=f"{key_prefix}note_status",
            )
            if st.form_submit_button("Add note"):
                if new_note.strip():
                    try:
                        requests.post(
                            f"{API_URL}/resumes/{rid}/notes",
                            json={"note": new_note.strip(), "status": new_status or None},
                            timeout=5,
                        )
                        st.success("Note added.")
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))

    if show_chat:
        st.subheader("Ask about this candidate")
        q = st.text_input("Question", key=f"{key_prefix}detail_question")
        if st.button("Ask", key=f"{key_prefix}btn_ask_detail") and q:
            with st.spinner("Generating answer (may take a minute)…"):
                try:
                    resp = requests.post(
                        f"{API_URL}/resume/{rid}/chat",
                        json={"question": q},
                        timeout=300,
                    )
                    data = resp.json()
                    st.write("**Answer:** ", data.get("answer", ""))
                except requests.exceptions.Timeout:
                    st.error("Request timed out. The model is still loading — please try again in a moment.")
                except Exception as e:
                    st.error(str(e))


# ---------------------------------------------------------------------------
# App layout
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Resume Analyzer", layout="wide")

# Sidebar toggle (single hamburger that moves)
if "sidebar_hidden" not in st.session_state:
    st.session_state["sidebar_hidden"] = False

# When sidebar is hidden, show hamburger in main header so user can reopen it.
# When sidebar is visible, hamburger is shown inside the sidebar only.
if st.session_state.get("sidebar_hidden"):
    top_l, top_r = st.columns([0.12, 0.88], vertical_alignment="center")
    with top_l:
        if st.button("☰", key="btn_toggle_sidebar", help="Show sidebar"):
            st.session_state["sidebar_hidden"] = False
            st.rerun()
    with top_r:
        st.title("Resume Analyzer")
else:
    st.title("Resume Analyzer")

if st.session_state.get("sidebar_hidden"):
    st.markdown(
        """
        <style>
        /* Hide sidebar completely */
        [data-testid="stSidebar"] { display: none !important; }
        /* Remove left padding/margin reserved for sidebar */
        section.main { padding-left: 0 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Global UI: sidebar toggle always visible, consistent spacing
st.markdown(
    """
    <style>
    /* Remove Streamlit's default arrow/chevron sidebar toggle */
    [data-testid="collapsedControl"],
    [data-testid="stSidebarCollapsedControl"] {
        display: none !important;
        visibility: hidden !important;
        pointer-events: none !important;
    }
    button[kind="header"] { z-index: 9998 !important; }
    /* Ensure page title never gets clipped under fixed header/toggle */
    header[data-testid="stHeader"] { height: 3.25rem !important; }
    /* Main content padding for consistency */
    .block-container { padding-top: 3.25rem; padding-bottom: 2rem; padding-left: 1.5rem; padding-right: 1.5rem; max-width: 100%; }

    /* Hamburger button styling (subtle, icon-only) */
    button[data-testid="baseButton-secondary"][key="btn_toggle_sidebar"],
    button[data-testid="baseButton-primary"][key="btn_toggle_sidebar"] { background: transparent !important; border: none !important; box-shadow: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    # Sidebar hamburger (in-sidebar toggle)
    # vertical_alignment is not supported in older Streamlit versions
    try:
        sbtn_l, _sbtn_r = st.columns([1.0, 0.85], vertical_alignment="center")
    except TypeError:
        sbtn_l, _sbtn_r = st.columns([1.0, 0.85])
    with sbtn_l:
        if st.button("☰", key="btn_toggle_sidebar_in_sidebar", help="Hide sidebar"):
            st.session_state["sidebar_hidden"] = True
            st.rerun()

    # Backend health indicator
    _status = _backend_status()
    if _status == "ok":
        st.success("Backend: online", icon="✅")
    elif _status == "starting":
        st.warning("Backend is starting up…", icon="⏳")
        st.caption("Wait ~10 seconds then click Refresh.")
    else:
        st.error("Backend offline", icon="🔴")
        st.caption("Run: uvicorn backend.api:app --host 127.0.0.1 --port 8001")

    if st.button(" Refresh", key="sidebar_refresh"):
        fetch_resumes.clear()
        st.rerun()

    st.caption("💡 First chat may take ~20 sec (model loading).")

    st.caption("Quick filters: use the Filters expander in the Resumes tab.")

# ---------------------------------------------------------------------------
# Auto-open dialogs helper
# ---------------------------------------------------------------------------
def _handle_global_dialogs():
    if (
        st.session_state.get("delete_confirm_rid")
        and st.session_state.get("delete_confirm_reopen")
    ):
        _delete_confirm_dialog(st.session_state["delete_confirm_rid"])
        st.session_state["delete_confirm_reopen"] = False
    elif (
        st.session_state.get("notes_list_dialog_rid")
        and st.session_state.get("notes_list_dialog_reopen")
    ):
        _view_notes_list_dialog(
            st.session_state["notes_list_dialog_rid"],
            st.session_state.get("notes_list_dialog_name", ""),
        )
        st.session_state["notes_list_dialog_reopen"] = False
    elif st.session_state.get("skills_dialog_rid"):
        _view_skills_dialog(st.session_state["skills_dialog_rid"])
        st.session_state.pop("skills_dialog_rid", None)
    elif (
        st.session_state.get("notes_dialog_rid")
        and st.session_state.get("notes_dialog_reopen_after_save")
    ):
        _view_notes_dialog(
            st.session_state["notes_dialog_rid"],
            st.session_state.get("notes_dialog_name", ""),
        )
        st.session_state["notes_dialog_reopen_after_save"] = False


# Tabs  (7 tabs — Resume Chat and Ranking are now separate)
tabs = st.tabs([
    "Upload",
    "Resumes (table)",
    "Compare",
    "Chat (global)",
    "Resume Chat",
    "Export",
])


# ---------------------------------------------------------------------------
# Tab 0 — Upload
# ---------------------------------------------------------------------------
with tabs[0]:
    # Upload UX helpers: show "done" message and reset uploaders by changing widget keys
    if "upload_done_msg" not in st.session_state:
        st.session_state["upload_done_msg"] = ""
    if "resume_uploader_nonce" not in st.session_state:
        st.session_state["resume_uploader_nonce"] = 0
    if "zip_uploader_nonce" not in st.session_state:
        st.session_state["zip_uploader_nonce"] = 0

    if st.session_state.get("upload_done_msg"):
        st.success(st.session_state["upload_done_msg"])
        # Show once, then clear
        st.session_state["upload_done_msg"] = ""

    upload_type = st.radio(
        "Upload type", ["Single/Multiple files", "ZIP (bulk)"], horizontal=True
    )

    if upload_type == "Single/Multiple files":
        uploaded = st.file_uploader(
            "Upload PDF or DOCX resumes",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            key=f"resume_uploader_{st.session_state['resume_uploader_nonce']}",
        )
        if st.button("Send files", key="btn_send_files"):
            if not uploaded:
                st.warning("Please select at least one PDF or DOCX file.")
            else:
                files = []
                for u in uploaded:
                    fname = u.name or "resume.pdf"
                    ct = (
                        "application/pdf"
                        if fname.lower().endswith(".pdf")
                        else "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                    )
                    files.append(("files", (fname, u.getvalue(), ct)))
                with st.spinner(
                    "Processing resumes — this can take several minutes while the model runs…"
                ):
                    try:
                        resp = requests.post(
                            f"{API_URL}/upload", files=files
                        )
                        try:
                            data = resp.json()
                        except ValueError:
                            st.error("Upload failed — server returned invalid response.")
                            if resp.text:
                                with st.expander("Technical details"):
                                    st.code(resp.text[:2000], language="text")
                            st.stop()

                        if resp.status_code >= 400:
                            st.error(
                                f"Upload failed (HTTP {resp.status_code}). "
                                f"{data.get('detail', '')}"
                            )
                            st.stop()

                        if isinstance(data, list):
                            success = [d for d in data if d.get("status") == "success"]
                            dupes = [d for d in data if d.get("status") == "duplicate"]
                            errors = [d for d in data if d.get("status") == "error"]
                            if success:
                                st.success(
                                    f" {len(success)} resume(s) uploaded: "
                                    + ", ".join(d.get("candidate_name", "") for d in success)
                                )
                            for d in dupes:
                                st.warning(
                                    f" Duplicate skipped — {d.get('message', '')}"
                                )
                            for d in errors:
                                st.error(
                                    f" {d.get('file', '')}: {d.get('message', '')}"
                                )
                            invalidate_resume_cache()
                            # Notify user and clear the uploader by bumping the widget key
                            if success:
                                names = ", ".join(d.get("candidate_name", "") for d in success if d.get("candidate_name"))
                                st.session_state["upload_done_msg"] = (
                                    f"Upload + extraction completed for {len(success)} resume(s)"
                                    + (f": {names}" if names else ".")
                                )
                                try:
                                    st.toast("Resume upload and extraction completed.", icon="✅")
                                except Exception:
                                    pass
                                st.session_state["resume_uploader_nonce"] += 1
                                try:
                                    st.rerun()
                                except Exception:
                                    st.experimental_rerun()
                        else:
                            st.json(data)
                    except requests.exceptions.Timeout:
                        st.error(
                            "Upload timed out after 10 minutes. "
                            "The model may still be running — check the backend terminal."
                        )
                    except requests.exceptions.ConnectionError:
                        st.error(
                            f"Cannot reach backend at {API_URL}. "
                            "Make sure uvicorn is running on port 8001."
                        )
                    except Exception as e:
                        st.error(f"Upload failed: {e}")
    else:
        zip_file = st.file_uploader(
            "Upload a ZIP of PDF/DOCX",
            type=["zip"],
            key=f"zip_uploader_{st.session_state['zip_uploader_nonce']}",
        )
        if zip_file and st.button("Process ZIP", key="btn_process_zip"):
            with st.spinner("Processing ZIP…"):
                try:
                    resp = requests.post(
                        f"{API_URL}/upload/bulk",
                        files={"file": (zip_file.name, zip_file.getvalue(), "application/zip")},
                    )
                    resp.raise_for_status()
                    st.json(resp.json())
                    invalidate_resume_cache()
                    st.session_state["upload_done_msg"] = "Bulk ZIP upload + extraction completed."
                    try:
                        st.toast("Bulk ZIP upload and extraction completed.", icon="✅")
                    except Exception:
                        pass
                    st.session_state["zip_uploader_nonce"] += 1
                    try:
                        st.rerun()
                    except Exception:
                        st.experimental_rerun()
                except Exception as e:
                    st.error(str(e))

    st.markdown("---")
    #st.subheader("Fetch resumes from email (IMAP)")
    auto_extract = st.checkbox("Auto-download and extract into database", value=True, key="email_auto_extract")
    force_refetch = st.checkbox("Force re-fetch (ignore processed cache)", value=False, key="email_force_refetch")
    if st.button("Fetch new resumes from email", key="btn_fetch_email"):
        with st.spinner("Fetching matching emails and downloading attachments…"):
            try:
                resp = requests.get(
                    f"{API_URL}/fetch-resumes",
                    params={
                        "auto_extract": "1" if auto_extract else "0",
                        "force": "1" if force_refetch else "0",
                        "indeed": "0",
                    },
                    timeout=180,
                )
                data = resp.json() if resp.content else {}
                if resp.status_code >= 400:
                    st.error(f"Fetch failed (HTTP {resp.status_code}). {data.get('detail', '')}")
                else:
                    if data.get("ok") is False:
                        st.error(f"Email fetch error: {data.get('indeed_error') or data}")
                        with st.expander("Details"):
                            st.json(data)
                        st.stop()
                    downloaded = data.get("downloaded") or []
                    total_files = sum(len(x.get("downloaded_files") or []) for x in downloaded)
                    if auto_extract:
                        ae = data.get("auto_extract") or {}
                        res = ae.get("results") or []
                        ok = [r for r in res if isinstance(r, dict) and r.get("status") == "success"]
                        st.success(f"Completed: {total_files} file(s) downloaded and {len(ok)} new resume(s) added to database.")
                        invalidate_resume_cache()
                    else:
                        st.success(f"Email fetch complete. {total_files} file(s) downloaded.")
                    with st.expander("Details"):
                        st.json(data)
            except requests.exceptions.ConnectionError:
                st.error(f"Cannot reach backend at {API_URL}. Make sure it is running.")
            except Exception as exc:
                st.error(f"Fetch failed: {exc}")

# ---------------------------------------------------------------------------
# Tab 1 — Resumes table
# ---------------------------------------------------------------------------
with tabs[1]:
    _handle_global_dialogs()
    qp = st.query_params
    default_skill = qp.get("skill", "")
    qp_shortlisted = (qp.get("shortlisted", "") or "").strip().lower()
    default_shortlisted_option = (
        "Shortlisted" if qp_shortlisted == "true" else ("Not shortlisted" if qp_shortlisted == "false" else "All")
    )

    # Filters: proper spacing, Shortlisted inside toolbar
    st.markdown(
        """
        <style>
        /* Filter expander: readable padding and spacing between inputs */
        [data-testid="stExpander"] details summary { padding: 0.4rem 0; }
        [data-testid="stExpander"] details summary ~ div { padding: 0.5rem 0.75rem 0.6rem 0.75rem; }
        [data-testid="stExpander"] [data-testid="column"] { padding-left: 0.6rem; padding-right: 0.6rem; }
        [data-testid="stExpander"] [data-testid="column"] [data-testid="stHorizontalBlock"] { gap: 0.1rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    with st.expander("Filters", expanded=True):
        c1, c2, c3, c4, c5 = st.columns([1.0, 1.0, 1.5, 0.7, 0.7])
        with c1:
            name_filter = st.text_input("Name (contains)", "",key="name_filter")
        with c2:
            email_filter = st.text_input("Email (contains)", "",key="email_filter")
        with c3:
            skills_filter = st.text_input(
                "Skills (comma, AND)", default_skill or "", key="skills_filter"
            )
        with c4:
            min_exp = st.number_input("Min exp (yr)", min_value=0, value=0, step=1, key="min_exp_filter")
        with c5:
            max_exp = st.number_input(
                "Max exp (yr)", min_value=0, value=0, step=1, key="max_exp"
            )

        d1, d2, d3, d4, d5 = st.columns([1.0, 1.0, 1.2, 0.8, 0.9])
        with d1:
            from_date = st.date_input("From Date", value=datetime.now().date(),max_value=datetime.now().date(), key="from_date")

        with d2:
           to_date = st.date_input("To Date",value=datetime.now().date(),max_value=datetime.now().date(),key="to_date")
        with d3:
            sort_by = st.selectbox(
                "Sort by",
                options=["created_at", "name", "experience"],
                format_func=lambda x: {"created_at": "Created Date", "name": "Name", "experience": "Experience"}[x],
                key="sort_by_filter",
            )
        with d4:
            sort_order = st.selectbox("Order", options=["desc", "asc"], format_func=lambda x: "Descending" if x == "desc" else "Ascending", key="sort_order_filter")
        with d5:
            
            shortlisted_filter = st.selectbox(
                "Shortlisted filter",
                options=["All", "Shortlisted"],
                index=["All", "Shortlisted"].index(default_shortlisted_option),
                #label_visibility="collapsed",
                key="shortlisted_filter",
            )

        # Build query-string key for cache and fetch
        qs_parts = []
        if min_exp > 0:
            qs_parts.append(f"min_experience={min_exp}")
        if max_exp > 0:
            qs_parts.append(f"max_experience={max_exp}")
        if skills_filter.strip():
            qs_parts.append(f"skills={skills_filter.strip()}")
        if from_date:
           qs_parts.append(f"added_after={from_date.isoformat()}")
        if to_date:
            qs_parts.append(f"added_before={to_date.isoformat()}")
        if sort_by:
            qs_parts.append(f"sort_by={sort_by}")
        if sort_order:
            qs_parts.append(f"sort_order={sort_order}")
        if shortlisted_filter == "Shortlisted":
            qs_parts.append("shortlisted=true")
        elif shortlisted_filter == "Not shortlisted":
            qs_parts.append("shortlisted=false")
        pending_key = "&".join(qs_parts)

        # Apply button: only fetch when user clicks Apply (or on first load)
        if "applied_filter_cache_key" not in st.session_state:
            st.session_state["applied_filter_cache_key"] = pending_key

        # NOTE: Streamlit forbids modifying widget keys after instantiation in the same run.
        # Use button callbacks to update session_state safely.
        def _apply_filters() -> None:
            st.session_state["applied_filter_cache_key"] = pending_key
            st.session_state["resumes_page"] = 1
            invalidate_resume_cache()

        def _clear_filters() -> None:
            st.session_state["name_filter"]=""
            st.session_state["email_filter"]=""
            st.session_state["skills_filter"] = ""
            st.session_state["min_exp_filter"] = 0
            st.session_state["max_exp"] = 0
            today = datetime.now().date()
            st.session_state["from_date"] = today
            st.session_state["to_date"] = today
            st.session_state["sort_by_filter"] = "created_at"
            st.session_state["sort_order_filter"] = "desc"
            st.session_state["shortlisted_filter"] = "All"
            # After clear: show only today's candidates (not all)
            st.session_state["applied_filter_cache_key"] = (
                f"added_after={today.isoformat()}&added_before={today.isoformat()}"
                f"&sort_by=created_at&sort_order=desc"
            )
            st.session_state["resumes_page"] = 1
            invalidate_resume_cache()

        a1, a2, _ = st.columns([0.9, 0.9, 9])
        with a1:
            st.button("Apply", key="btn_apply_filters", type="primary", on_click=_apply_filters)
        with a2:
            st.button("Clear", key="btn_clear_filters", type="secondary", on_click=_clear_filters)

        cache_key = st.session_state.get("applied_filter_cache_key", "")

        # Detect when applied filters actually change (for spinner timing)
        last_key = st.session_state.get("last_filter_cache_key")
        is_new_filter = cache_key != last_key
        st.session_state["last_filter_cache_key"] = cache_key

        with st.spinner("Loading resumes…"):
            start_t = time.time()
            data, fetch_err = fetch_resumes(cache_key)
            if is_new_filter:
                min_duration = 0.8
                elapsed = time.time() - start_t
                remaining = min_duration - elapsed
                if remaining > 0:
                    time.sleep(remaining)dfgn

    if fetch_err == "OFFLINE":
        st.error(
            "Backend is not running. Start it with:\n\n"
            "```\nuvicorn backend.api:app --reload --host 127.0.0.1 --port 8001\n```"
        )
        data = []
    elif fetch_err == "STARTING":
        st.warning(
            "⏳ Backend is still starting up (loading models + migrations). "
            "Please wait ~10 seconds and click **Refresh** in the sidebar."
        )
        st.button("🔄 Retry now", on_click=fetch_resumes.clear, key="retry_resumes")
        data = []
    elif fetch_err:
        st.error(f"Could not load resumes: {fetch_err}")
        data = []

    if not data and not fetch_err:
        if cache_key:
            st.info("No resumes match the current filters.")
        else:
            st.info("No resumes yet. Upload some in the Upload tab.")
    else:
        df = pd.DataFrame(data)

        # Client-side name/email filter
        if name_filter and "name" in df.columns:
            df = df[df["name"].astype(str).str.contains(name_filter, case=False, na=False)]
        if email_filter and "email" in df.columns:
            df = df[df["email"].astype(str).str.contains(email_filter, case=False, na=False)]

        # Clean table layout
        rows = df.to_dict("records")

        # Grid/table borders, icon-only actions, skill badges
        st.markdown("""
<style>
.resume-grid-wrap {
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 8px;
    overflow-x: auto;
    overflow-y: hidden;
    background: rgba(255,255,255,0.02);
}

.resume-grid-header {
    border-bottom: 1px solid rgba(255,255,255,0.15);
    background: rgba(255,255,255,0.05);
    min-width: 1000px;
    font-weight: 600;
}

.resume-grid-row {
    border-bottom: 1px solid rgba(255,255,255,0.08);
    min-width: 1000px;
}

.resume-grid-row div[data-testid="stColumn"],
.resume-grid-header div[data-testid="stColumn"] {
    border-right: 1px solid rgba(255,255,255,0.12);
    padding: 10px 12px;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    vertical-align: middle !important;
}

.resume-grid-header div[data-testid="stColumn"] *,
.resume-grid-row div[data-testid="stColumn"] * {
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    margin-bottom: 0.1rem;
}

.resume-grid-row div[data-testid="stColumn"]:last-child,
.resume-grid-header div[data-testid="stColumn"]:last-child {
    border-right: none;
}

.resume-grid-row:hover {
    background: rgba(255,255,255,0.03);
}

/* Icon-only action buttons: no border, icon centered in middle */
.resume-grid-row div[data-testid="stColumn"] button[kind="secondary"],
.resume-grid-row div[data-testid="stColumn"] button[kind="primary"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0 !important;
    margin: 0 auto !important;
    width: 2.25rem !important;
    min-width: 2.25rem !important;
    max-width: 2.25rem !important;
    height: 2.25rem !important;
    min-height: 2.25rem !important;
    font-size: 1.2rem !important;
    line-height: 1 !important;
    color: rgba(255,255,255,0.75) !important;
    cursor: pointer !important;
    border-radius: 0 !important;
}

.resume-grid-row div[data-testid="stColumn"] button[kind="secondary"] > div,
.resume-grid-row div[data-testid="stColumn"] button[kind="primary"] > div {
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 100% !important;
    line-height: 1 !important;
    margin: 0 !important;
    padding: 0 !important;
}

.resume-grid-row div[data-testid="stColumn"] button[kind="secondary"]:hover,
.resume-grid-row div[data-testid="stColumn"] button[kind="primary"]:hover {
    background: transparent !important;
    color: rgba(255,255,255,0.95) !important;
}

/* Extra-hard override: Streamlit wraps buttons in .stButton */
.resume-grid-row div[data-testid="stColumn"] .stButton > button {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}



.resume-grid-wrap a[target="_blank"] {
    color: #6eb5ff !important;
    text-decoration: none;
}
.resume-grid-wrap a[target="_blank"]:hover {
    text-decoration: underline;
}

/* Email column: plain text, white (no link styling) */
.resume-grid-wrap div[data-testid="stColumn"] p,
.resume-grid-wrap div[data-testid="stColumn"] div[data-testid="stText"] {
    color: #ffffff !important;
}
</style>
""", unsafe_allow_html=True)

        # Pagination (10 rows per page) — numeric pager (Prev / 1 2 3 / Next)
        page_size = 10
        total_rows = len(rows)
        total_pages = max(1, (total_rows + page_size - 1) // page_size)

        if "resumes_page" not in st.session_state:
            st.session_state["resumes_page"] = 1
        st.session_state["resumes_page"] = max(1, min(int(st.session_state["resumes_page"]), total_pages))
        page = int(st.session_state["resumes_page"])

        st.markdown(
            """
            <style>
            .pager-row [data-testid="stHorizontalBlock"] { gap: 0.02rem; }
            .pager-row button[kind="secondary"],
            .pager-row button[kind="primary"] {
                background: transparent !important;
                border: 1px solid rgba(255,255,255,0.14) !important;
                box-shadow: none !important;
                padding: 0.05rem 0.2rem !important;
                min-height: 1.55rem !important;
                line-height: 1.1 !important;
                font-size: 0.82rem !important;
                color: #cfe3ff !important;
                font-weight: 600 !important;
                border-radius: 0.55rem !important;
            }
            .pager-row button[kind="secondary"]:hover,
            .pager-row button[kind="primary"]:hover { border-color: rgba(255,255,255,0.22) !important; }

            /* Active page */
            .pager-row button[kind="primary"] {
                background: rgba(255, 82, 82, 0.95) !important;
                border-color: rgba(255, 82, 82, 0.95) !important;
                color: #0b0f14 !important;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        top_left, top_right = st.columns([4.2, 1.8])
        with top_left:
            st.caption(f"Total candidates: {total_rows} • Showing {page_size} per page")

        with top_right:
            st.markdown('<div class="pager-row">', unsafe_allow_html=True)

            # Build a compact window of page numbers (max 10)
            window = 10
            half = window // 2
            start_page = max(1, page - half)
            end_page = min(total_pages, start_page + window - 1)
            start_page = max(1, end_page - window + 1)

            nums_count = end_page - start_page + 1
            # Right-aligned, tighter pager: [Prev][1][2]...[Next]
            pager_cols = st.columns([0.35] + ([0.26] * nums_count) + [0.6])

            if pager_cols[0].button("Prev", key="pager_prev", disabled=(page <= 1), type="secondary"):
                st.session_state["resumes_page"] = page - 1
                st.rerun()

            for idx, pnum in enumerate(range(start_page, end_page + 1), start=1):
                btn_type = "primary" if pnum == page else "secondary"
                if pager_cols[idx].button(str(pnum), key=f"pager_num_{pnum}", type=btn_type):
                    st.session_state["resumes_page"] = pnum
                    st.rerun()

            if pager_cols[-1].button("Next", key="pager_next", disabled=(page >= total_pages), type="secondary"):
                st.session_state["resumes_page"] = page + 1
                st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        start = (page - 1) * page_size
        end = start + page_size
        page_rows = rows[start:end]

        st.markdown('<div class="resume-grid-wrap">', unsafe_allow_html=True)
        st.markdown('<div class="resume-grid-header resume-grid-row">', unsafe_allow_html=True)
        col_weights = [2.0, 2.5, 1.5, 1.5, 1.0, 1.8, 0.9, 1.5, 2.5]
        header = st.columns(col_weights)
        header[0].markdown("**Name**")
        header[1].markdown("**Email**")
        header[2].markdown("**Phone**")
        header[3].markdown("**Experience**")
        header[4].markdown("**Added**")
        header[5].markdown("**Primary Skills**")
        header[6].markdown("**Resume**")
        header[7].markdown("**Shortlisted**")
        header[8].markdown("**Actions**")
        st.markdown("</div>", unsafe_allow_html=True)

        shortlist_changes: list[tuple[str, bool]] = []

        for i, r in enumerate(page_rows, start=start):
            rid = (r.get("id") or "").strip()
            if not rid:
                continue
            st.markdown('<div class="resume-grid-row">', unsafe_allow_html=True)
            cols = st.columns(col_weights)
            cols[0].write(_name_only(r.get("name")))
            with cols[1]:
                st.text(r.get("email") or "—")
            cols[2].write(r.get("phone") or "—")
            cols[3].write(_format_years_to_duration(r.get("experience_years")))
            created_at = r.get("created_at")
            cols[4].write(str(created_at)[:10] if created_at else "—")

            # Primary Skills column: badges (top 3)
            primary = r.get("primary_skills") or r.get("key_skills") or []
            if not isinstance(primary, list):
                primary = [s.strip() for s in str(primary).split(",") if s.strip()]
            if primary:
                badges_html = " ".join(
                    f'<span class="skill-badge">{html.escape(s)}</span>' for s in primary[:3]
                )
                cols[5].markdown(badges_html, unsafe_allow_html=True)
            else:
                cols[5].write("—")

            with cols[6]:
                link = r.get("resume_link") or "#"
                st.markdown(
                    f'<a href="{link}" target="_blank">Open</a>',
                    unsafe_allow_html=True,
                )
            with cols[7]:
                cur_short = bool(r.get("is_shortlisted"))
                new_short = st.checkbox(
                    "Shortlist",
                    value=cur_short,
                    key=f"short_resumes_{rid}",
                    label_visibility="collapsed",
                )
                if new_short != cur_short:
                    shortlist_changes.append((rid, new_short))

            # Actions column: icon-only buttons with tooltips (View Skills, Add Note, View Notes, Delete)
            with cols[8]:
                act1, act2, act3, act4 = st.columns(4)
                with act1:
                    if st.button("🧩", key=f"viewskills_resumes_{rid}_{i}", help="View Skills"):
                        _open_skills_modal(rid)
                        st.rerun()
                with act2:
                    if st.button("📝", key=f"notes_btn_resumes_{rid}", help="Add Note"):
                        st.session_state["notes_dialog_rid"] = rid
                        st.session_state["notes_dialog_name"] = r.get("name") or ""
                        st.session_state["notes_dialog_reopen_after_save"] = True
                        st.rerun()
                with act3:
                    if st.button("📋", key=f"view_resumes_{rid}_{i}", help="View Notes"):
                        st.session_state["notes_list_dialog_rid"] = rid
                        st.session_state["notes_list_dialog_name"] = r.get("name") or ""
                        st.session_state["notes_list_dialog_reopen"] = True
                        st.rerun()
                with act4:
                    if st.button("🗑", key=f"del_btn_resumes_{rid}_{i}", help="Delete Candidate"):
                        st.session_state["delete_confirm_rid"] = rid
                        st.session_state["delete_confirm_reopen"] = True
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)  # resume-grid-wrap

        # Apply shortlist changes
        if shortlist_changes:
            for rid, is_short in shortlist_changes:
                try:
                    if is_short:
                        requests.post(f"{API_URL}/resumes/{rid}/shortlist", timeout=5)
                    else:
                        requests.delete(f"{API_URL}/resumes/{rid}/shortlist", timeout=5)
                except Exception:
                    pass
            invalidate_resume_cache()
            st.rerun()


# ---------------------------------------------------------------------------
# Tab 2 — Compare
# ---------------------------------------------------------------------------
with tabs[2]:
    # Field/domain-based compare (fast filtering via backend primary_skills)
    field_options = [".NET", "Python", "React", "SEO", "DevOps", "Java","BDE"]
    field = st.selectbox("Select technology field/domain", field_options, key="compare_field")

    compare_list = []
    _err4 = None
    try:
        resp = requests.get(
            f"{API_URL}/resumes/by-field",
            params={"field": field.lower(), "limit": 500},
            timeout=20,
        )
        if resp.status_code >= 400:
            _err4 = (resp.json() or {}).get("detail") or f"HTTP {resp.status_code}"
        else:
            compare_list = resp.json() or []
    except requests.exceptions.Timeout:
        _err4 = "Timeout talking to backend"
    except Exception as e:
        _err4 = str(e)

    if _err4 in ("OFFLINE", "STARTING") or "Backend" in str(_err4):
        st.warning("Backend is not ready yet. Start uvicorn and click Refresh in the sidebar.")
        compare_list = []
    elif _err4:
        st.error(f"Could not load candidates: {_err4}")
        compare_list = []

    if len(compare_list) < 2:
        st.info("Need at least 2 candidates in this field to compare.")
    else:
        st.markdown(
            """
            <style>
            .compare-card {
                border: 1px solid rgba(255,255,255,0.10);
                border-radius: 14px;
                padding: 14px 14px 12px 14px;
                background: rgba(255,255,255,0.02);
                box-shadow: 0 14px 34px rgba(0,0,0,0.28);
                min-height: 260px;
            }
            .compare-title {
                font-size: 1.05rem;
                font-weight: 800;
                margin-bottom: 0.25rem;
            }
            .compare-sub {
                color: rgba(255,255,255,0.72);
                font-size: 0.92rem;
                margin-bottom: 0.75rem;
            }
            .compare-kv {
                display: grid;
                grid-template-columns: 110px 1fr;
                gap: 0.25rem 0.55rem;
                font-size: 0.95rem;
                margin-bottom: 0.75rem;
            }
            .compare-k {
                color: rgba(255,255,255,0.70);
            }
            .compare-v {
                color: rgba(255,255,255,0.94);
                word-break: break-word;
            }
            .compare-score {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.25rem 0.55rem;
                border-radius: 999px;
                background: rgba(255, 82, 82, 0.18);
                border: 1px solid rgba(255, 82, 82, 0.35);
                color: rgba(255,255,255,0.92);
                font-weight: 800;
                font-size: 0.92rem;
                margin-bottom: 0.65rem;
            }
            .compare-score b { color: rgba(255,255,255,0.95); }
            .compare-summary {
                color: rgba(255,255,255,0.78);
                font-size: 0.92rem;
                margin: 0.2rem 0 0.85rem 0;
            }
            .compare-resume-btn {
                display: inline-block;
                padding: 0.45rem 0.7rem;
                border-radius: 0.7rem;
                border: 1px solid rgba(255,255,255,0.14);
                background: rgba(255,255,255,0.03);
                color: rgba(255,255,255,0.92);
                text-decoration: none !important;
                font-weight: 650;
                font-size: 0.92rem;
            }
            .compare-resume-btn:hover {
                border-color: rgba(255,255,255,0.22);
                background: rgba(255,255,255,0.05);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        selected = st.multiselect(
            "Select 2–3 candidates to compare",
            options=compare_list,
            format_func=lambda r: (r.get("name") or "").strip() or (r.get("id") or ""),
            max_selections=3,
            key="compare_sel",
        )
        jd_compare = st.text_area(
            "Optional: job description for fit score", height=80, key="jd_compare"
        )
        can_compare = bool(selected and len(selected) >= 2)
        compare_btn = st.button(
            "Compare Candidates",
            key="btn_compare_candidates",
            type="primary",
            disabled=not can_compare,
        )

        if compare_btn:
            ids = [(r.get("id") or "").strip() for r in selected if (r.get("id") or "").strip()]
            params: dict = {"ids": ",".join(ids)}
            if jd_compare.strip():
                params["job_description"] = jd_compare.strip()
            try:
                resp = requests.get(f"{API_URL}/resumes/compare", params=params, timeout=300)
                resp.raise_for_status()
                st.session_state["compare_data"] = resp.json() or []
            except Exception as e:
                st.error(str(e))
                st.session_state["compare_data"] = []

        compare_data = st.session_state.get("compare_data") or []
        if compare_data:
            cols = st.columns(len(compare_data))
            for i, c in enumerate(compare_data):
                with cols[i]:
                    name = c.get("name", "") or "—"
                    role = c.get("role", "") or "—"

                    exp_txt = _format_years_to_duration(c.get("experience_years"))
                    primary = c.get("primary_skills") or c.get("key_skills") or []
                    if not isinstance(primary, list):
                        primary = [s.strip() for s in str(primary).split(",") if s.strip()]
                    primary_txt = ", ".join(primary[:3]) if primary else "—"

                    projects = c.get("projects") or []
                    if not isinstance(projects, list):
                        projects = [s.strip() for s in str(projects).split(",") if s.strip()]
                    proj_txt = ", ".join(projects[:4]) if projects else "—"

                    

                    resume_link = c.get("resume_link") or ""
                    fit_score = c.get("fit_score")
                    fit_summary = c.get("fit_summary") or ""

                    score_html = ""
                    if fit_score is not None:
                        score_html = f'<div class="compare-score">Match score <b>{fit_score}</b> / 10</div>'

                    summary_html = f'<div class="compare-summary">{fit_summary}</div>' if fit_summary else ""

                    resume_btn_html = ""
                    if resume_link:
                        resume_btn_html = f'<a class="compare-resume-btn" href="{resume_link}" target="_blank">View full resume</a>'

                    st.markdown(
                        f"""
                        <div class="compare-card">
                          <div class="compare-title">{name}</div>
                          <div class="compare-sub">{role}</div>
                          {score_html}
                          <div class="compare-kv">
                            <div class="compare-k">Experience</div><div class="compare-v">{exp_txt}</div>
                            <div class="compare-k">Primary</div><div class="compare-v">{primary_txt}</div>
                           
                          </div>
                          {summary_html}
                          {resume_btn_html}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

# ---------------------------------------------------------------------------
# Tab 3 — Chat (global / vector search)
# ---------------------------------------------------------------------------
with tabs[3]:
    _handle_global_dialogs()
    st.markdown(
        "Ask a question about all stored resumes (e.g. *Show .NET developers*, *Find Python developers*, *List candidates with 3+ years experience*). "
        "Search finds matching candidates and the LLM answers; results are shown in a table below."
    )
    question_global = st.text_input("Question", key="global_chat_q")
    if st.button("Submit question", key="btn_submit_question"):
        if not question_global:
            st.warning("Please enter a question.")
        else:
            with st.spinner("Searching and generating answer (may take a minute)…"):
                try:
                    resp = requests.post(
                        f"{API_URL}/chat",
                        json={"question": question_global},
                        timeout=300,
                    )
                    data = resp.json()
                    if resp.status_code >= 400:
                        st.error(f"Chat failed. {data.get('detail', '')}")
                    else:
                        st.session_state["chat_answer"] = data.get("answer", "")
                        st.session_state["chat_matches"] = data.get("best_matches") or []
                        st.rerun()
                except requests.exceptions.Timeout:
                    st.error("Request timed out — model is still loading. Please try again.")
                except Exception as e:
                    st.error(str(e))

    # Show last answer and matching candidates in same grid layout as Resume table
    chat_matches = st.session_state.get("chat_matches") or []
    if st.session_state.get("chat_answer") is not None:
        st.write("**Answer:**")
        st.write(st.session_state.get("chat_answer", ""))
    if chat_matches:
        st.subheader(f"Matching candidates ({len(chat_matches)})")
        # Same grid style as Resumes table
        st.markdown("""
        <style>
        .chat-grid-row { border-bottom: 1px solid rgba(255,255,255,0.10); }
        .chat-grid-row div[data-testid="stColumn"], .chat-grid-header div[data-testid="stColumn"] {
            border-right: 1px solid rgba(255,255,255,0.10); padding: 10px;
        }
        .chat-grid-row div[data-testid="stColumn"]:last-child, .chat-grid-header div[data-testid="stColumn"]:last-child { border-right: none; }
        .chat-grid-row:hover { background: rgba(255,255,255,0.04); }
        .chat-grid-row button { background: none !important; border: none !important; padding: 0 !important; color: #4ba3ff !important; text-decoration: none !important; box-shadow: none !important; cursor: pointer !important; }
        .chat-grid-row button:hover { text-decoration: underline !important; }
        </style>
        """, unsafe_allow_html=True)
        st.markdown('<div class="chat-grid-header chat-grid-row">', unsafe_allow_html=True)
        h1, h2, h3, h4, h5, h6, h7, h8, h9 = st.columns([2.0, 2.2, 1.5, 1.0, 1.0, 1.8, 0.9, 1.0, 2.5])
        for col, label in [
            (h1, "**Name**"),
            (h2, "**Email**"),
            (h3, "**Phone**"),
            (h4, "**Experience**"),
            (h5, "**Added**"),
            (h6, "**Primary Skills**"),
            (h7, "**Resume**"),
            (h8, "**Shortlisted**"),
            (h9, "**Actions**"),
        ]:
            col.markdown(label)
        st.markdown("</div>", unsafe_allow_html=True)

        chat_shortlist_changes: list[tuple[str, bool]] = []
        chat_delete_ids: list[str] = []

        for i, r in enumerate(chat_matches):
            rid = (r.get("id") or "").strip()
            if not rid:
                continue
            
            st.markdown('<div class="chat-grid-row">', unsafe_allow_html=True)
            cols = st.columns([2.0, 2.2, 1.5, 1.0, 1.0, 1.8, 0.9, 1.0, 2.5])
            cols[0].write(_name_only(r.get("name")))
            with cols[1]:
                st.text(r.get("email") or "—")
            cols[2].write(r.get("phone") or "—")
            exp_txt = _format_years_to_duration(r.get("experience_years"))
            lvl = (r.get("experience_level") or "").strip()
            intern = bool(r.get("internship_present") or False)
            if intern and "Fresher" in (lvl or ""):
                cols[3].write(f"{exp_txt} (Internship)")
            else:
                cols[3].write(exp_txt)
            if lvl:
                cols[3].caption(lvl)
            created_at = r.get("created_at")
            cols[4].write(str(created_at)[:10] if created_at else "—")

            # Primary Skills column: badges (top 3)
            primary = r.get("primary_skills") or r.get("key_skills") or []
            if not isinstance(primary, list):
                primary = [s.strip() for s in str(primary).split(",") if s.strip()]
            if primary:
                badges_html = " ".join(
                    f'<span class="skill-badge">{html.escape(s)}</span>' for s in primary[:3]
                )
                cols[5].markdown(badges_html, unsafe_allow_html=True)
            else:
                cols[5].write("—")

            with cols[6]:
                link = r.get("resume_link") or "#"
                st.markdown(
                    f'<a href="{link}" target="_blank">Open</a>',
                    unsafe_allow_html=True,
                )
            with cols[7]:
                cur_short = bool(r.get("is_shortlisted"))
                new_short = st.checkbox(
                    "Shortlist",
                    value=cur_short,
                    key=f"short_chat_{rid}",
                    label_visibility="collapsed",
                )
                if new_short != cur_short:
                    chat_shortlist_changes.append((rid, new_short))

            # Actions column: icon-only buttons with tooltips (View Skills, Add Note, View Notes, Delete)
            with cols[8]:
                act1, act2, act3, act4 = st.columns(4)
                with act1:
                    if st.button("🧩", key=f"viewskills_chat_{rid}_{i}", help="View Skills"):
                        _open_skills_modal(rid)
                        st.rerun()
                with act2:
                    if st.button("📝", key=f"notes_btn_chat_{rid}", help="Add Note"):
                        st.session_state["notes_dialog_rid"] = rid
                        st.session_state["notes_dialog_name"] = r.get("name") or ""
                        st.session_state["notes_dialog_reopen_after_save"] = True
                        st.rerun()
                with act3:
                    if st.button("📋", key=f"view_chat_{rid}_{i}", help="View Notes"):
                        st.session_state["notes_list_dialog_rid"] = rid
                        st.session_state["notes_list_dialog_name"] = r.get("name") or ""
                        st.session_state["notes_list_dialog_reopen"] = True
                        st.rerun()
                with act4:
                    if st.button("🗑", key=f"del_btn_chat_{rid}_{i}", help="Delete Candidate"):
                        st.session_state["delete_confirm_rid"] = rid
                        st.session_state["delete_confirm_reopen"] = True
                        st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # Apply shortlist changes
        if chat_shortlist_changes:
            for rid, is_short in chat_shortlist_changes:
                try:
                    if is_short:
                        requests.post(f"{API_URL}/resumes/{rid}/shortlist", timeout=5)
                    else:
                        requests.delete(f"{API_URL}/resumes/{rid}/shortlist", timeout=5)
                except Exception:
                    pass
                for m in chat_matches:
                    if str(m.get("id")) == rid:
                        m["is_shortlisted"] = is_short
                        break
            invalidate_resume_cache()
            st.rerun()

        # Delete selected
    #    if chat_delete_ids:
    #        st.warning(f"Delete {len(chat_delete_ids)} selected candidate(s)?")
    #        d1, d2 = st.columns(2)
    #        with d1:
    #            if st.button("Confirm delete", key="chat_confirm_delete"):
    #                try:
    #                    requests.delete(f"{API_URL}/resumes/bulk", json={"resume_ids": chat_delete_ids}, timeout=10)
    #                    st.session_state["chat_matches"] = [m for m in chat_matches if str(m.get("id")) not in chat_delete_ids]
    #                    invalidate_resume_cache()
    #                    st.success("Deleted.")
    #                    st.rerun()
    #                except Exception as e:
    #                    st.error(str(e))
    #        with d2:
    #            if st.button("Cancel", key="chat_cancel_delete"):
    #                st.rerun()
    #elif st.session_state.get("chat_answer") is not None:
    #    st.caption("No matching candidates to display.")

# ---------------------------------------------------------------------------
# Tab 4 — Resume Chat (per-resume Q&A)
# ---------------------------------------------------------------------------
with tabs[4]:
    resumes_chat, _err6 = fetch_resumes()
    if _err6 in ("OFFLINE", "STARTING"):
        st.warning("Backend is not ready yet. Start uvicorn and click Refresh in the sidebar.")
        resumes_chat = []
    elif _err6:
        st.error(f"Could not load resumes: {_err6}")
        resumes_chat = []
    elif not resumes_chat:
        st.info("No resumes in DB. Upload some first.")
    else:
        options = [f"{x['name']}  ({x.get('role') or x['email'] or x['id']})" for x in resumes_chat]
        chat_idx = st.selectbox(
            "Select a resume",
            range(len(options)),
            format_func=lambda i: options[i],
            key="resume_chat_sel",
        )
        chat_q = st.text_input(
            "Ask about this candidate (e.g. strengths, fit for Python role?)",
            key="resume_chat_q",
        )
        if st.button("Ask", key="btn_ask_resume_chat") and chat_q:
            with st.spinner("Generating answer (may take a minute)…"):
                try:
                    rid = resumes_chat[chat_idx]["id"]
                    resp = requests.post(
                        f"{API_URL}/resume/{rid}/chat",
                        json={"question": chat_q},
                        timeout=300,
                    )
                    data = resp.json()
                    if resp.status_code >= 400:
                        st.error(f"Request failed. {data.get('detail', '')}")
                    else:
                        st.write("**Answer:**")
                        st.write(data.get("answer", ""))
                        resume_link = resumes_chat[chat_idx].get("resume_link")
                        if resume_link:
                            st.caption("Cross-check with the resume:")
                            st.link_button("Open resume (PDF)", resume_link, type="secondary")
                except requests.exceptions.Timeout:
                    st.error("Request timed out — model is still loading. Please try again.")
                except Exception as e:
                    st.error(str(e))


# ---------------------------------------------------------------------------
with tabs[5]:
    render_export_to_google_sheets_tab()

# Render modal overlays LAST so buttons are visible and unique.
_render_skills_modal_if_open()