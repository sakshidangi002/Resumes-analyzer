"""Initial schema: roles, users, employees, attendance, leave, payroll, letters, etc.

Revision ID: 001
Revises:
Create Date: 2025-03-02

"""
from typing import Sequence, Union
from alembic import op
import sqlalchemy as sa

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "departments",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("code", sa.String(20), nullable=True),
    )
    op.create_table(
        "designations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("title", sa.String(100), nullable=False),
        sa.Column("level", sa.Integer(), nullable=True),
    )
    op.create_table(
        "company_configs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("default_working_days_per_month", sa.Integer(), default=30),
        sa.Column("weekly_off_days", sa.String(100), nullable=True),
        sa.Column("grace_time_minutes", sa.Integer(), default=15),
        sa.Column("half_day_threshold_hours", sa.Integer(), default=4),
    )
    op.create_table(
        "financial_years",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        sa.Column("is_current", sa.Boolean(), default=False),
        sa.Column("name", sa.String(20), nullable=True),
    )
    op.create_table(
        "employees",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("employee_code", sa.String(50), unique=True, nullable=False),
        sa.Column("first_name", sa.String(100), nullable=False),
        sa.Column("last_name", sa.String(100), nullable=False),
        sa.Column("official_email", sa.String(255), nullable=False),
        sa.Column("personal_email", sa.String(255), nullable=True),
        sa.Column("phone", sa.String(20), nullable=True),
        sa.Column("date_of_joining", sa.Date(), nullable=False),
        sa.Column("designation_id", sa.Integer(), sa.ForeignKey("designations.id"), nullable=True),
        sa.Column("department_id", sa.Integer(), sa.ForeignKey("departments.id"), nullable=True),
        sa.Column("employment_type", sa.String(20), default="Full-time"),
        sa.Column("reporting_manager_id", sa.Integer(), sa.ForeignKey("employees.id"), nullable=True),
        sa.Column("employment_status", sa.String(20), default="Active"),
        sa.Column("date_of_birth", sa.Date(), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_table(
        "employee_bank_details",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("employee_id", sa.Integer(), sa.ForeignKey("employees.id"), nullable=False),
        sa.Column("bank_name", sa.String(100), nullable=False),
        sa.Column("account_holder_name", sa.String(100), nullable=False),
        sa.Column("account_number", sa.String(50), nullable=False),
        sa.Column("ifsc_code", sa.String(20), nullable=False),
        sa.Column("account_type", sa.String(20), default="Savings"),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_table(
        "roles",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("name", sa.String(50), unique=True, nullable=False),
        sa.Column("description", sa.String(255), nullable=True),
    )
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("username", sa.String(100), unique=True, nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("official_email", sa.String(255), nullable=True),
        sa.Column("is_active", sa.Boolean(), default=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("employee_id", sa.Integer(), sa.ForeignKey("employees.id"), nullable=True),
    )
    op.create_table(
        "user_roles",
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), primary_key=True),
        sa.Column("role_id", sa.Integer(), sa.ForeignKey("roles.id"), primary_key=True),
    )
    op.create_table(
        "holidays",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("is_optional", sa.Boolean(), default=False),
        sa.Column("financial_year_id", sa.Integer(), sa.ForeignKey("financial_years.id"), nullable=True),
        sa.Column("company_config_id", sa.Integer(), sa.ForeignKey("company_configs.id"), nullable=True),
    )
    op.create_table(
        "attendance_records",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("employee_id", sa.Integer(), sa.ForeignKey("employees.id"), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("sign_in_time", sa.Time(), nullable=True),
        sa.Column("sign_out_time", sa.Time(), nullable=True),
        sa.Column("total_work_hours", sa.Numeric(5, 2), nullable=True),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("is_late", sa.Boolean(), default=False),
        sa.Column("is_early_exit", sa.Boolean(), default=False),
        sa.Column("is_weekly_off", sa.Boolean(), default=False),
        sa.Column("is_holiday", sa.Boolean(), default=False),
        sa.Column("source", sa.String(20), default="SELF"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_table(
        "attendance_correction_requests",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("employee_id", sa.Integer(), sa.ForeignKey("employees.id"), nullable=False),
        sa.Column("attendance_date", sa.Date(), nullable=False),
        sa.Column("requested_sign_in_time", sa.Time(), nullable=True),
        sa.Column("requested_sign_out_time", sa.Time(), nullable=True),
        sa.Column("requested_status", sa.String(20), nullable=True),
        sa.Column("reason", sa.String(500), nullable=False),
        sa.Column("status", sa.String(20), default="PENDING"),
        sa.Column("approver_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("approved_at", sa.DateTime(), nullable=True),
        sa.Column("rejection_reason", sa.String(500), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_table(
        "leave_types",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("code", sa.String(20), unique=True, nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("is_paid", sa.Boolean(), default=True),
        sa.Column("allow_half_day", sa.Boolean(), default=False),
    )
    op.create_table(
        "leave_allocations",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("employee_id", sa.Integer(), sa.ForeignKey("employees.id"), nullable=False),
        sa.Column("financial_year_id", sa.Integer(), sa.ForeignKey("financial_years.id"), nullable=False),
        sa.Column("leave_type_id", sa.Integer(), sa.ForeignKey("leave_types.id"), nullable=False),
        sa.Column("allocated_days", sa.Numeric(5, 2), default=0),
        sa.Column("used_days", sa.Numeric(5, 2), default=0),
    )
    op.create_table(
        "leave_requests",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("employee_id", sa.Integer(), sa.ForeignKey("employees.id"), nullable=False),
        sa.Column("leave_type_id", sa.Integer(), sa.ForeignKey("leave_types.id"), nullable=False),
        sa.Column("start_date", sa.Date(), nullable=False),
        sa.Column("end_date", sa.Date(), nullable=False),
        sa.Column("is_half_day", sa.Boolean(), default=False),
        sa.Column("reason", sa.String(500), nullable=True),
        sa.Column("status", sa.String(20), default="PENDING"),
        sa.Column("applied_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("manager_approver_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("hr_approver_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("approved_at", sa.DateTime(), nullable=True),
        sa.Column("rejected_at", sa.DateTime(), nullable=True),
        sa.Column("rejection_reason", sa.String(500), nullable=True),
    )
    op.create_table(
        "salary_structures",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("employee_id", sa.Integer(), sa.ForeignKey("employees.id"), nullable=False),
        sa.Column("basic", sa.Numeric(12, 2), default=0),
        sa.Column("hra", sa.Numeric(12, 2), default=0),
        sa.Column("allowances", sa.Numeric(12, 2), default=0),
        sa.Column("deductions", sa.Numeric(12, 2), default=0),
        sa.Column("effective_from", sa.Date(), nullable=False),
        sa.Column("effective_to", sa.Date(), nullable=True),
    )
    op.create_table(
        "payroll_periods",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("month", sa.Integer(), nullable=False),
        sa.Column("year", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(20), default="OPEN"),
        sa.Column("processed_at", sa.DateTime(), nullable=True),
    )
    op.create_table(
        "payslips",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("employee_id", sa.Integer(), sa.ForeignKey("employees.id"), nullable=False),
        sa.Column("payroll_period_id", sa.Integer(), sa.ForeignKey("payroll_periods.id"), nullable=False),
        sa.Column("gross_salary", sa.Numeric(12, 2), default=0),
        sa.Column("total_earnings", sa.Numeric(12, 2), default=0),
        sa.Column("total_deductions", sa.Numeric(12, 2), default=0),
        sa.Column("net_salary", sa.Numeric(12, 2), default=0),
        sa.Column("paid_days", sa.Numeric(5, 2), default=0),
        sa.Column("lop_days", sa.Numeric(5, 2), default=0),
        sa.Column("component_breakdown", sa.Text(), nullable=True),
        sa.Column("generated_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_table(
        "email_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("to_email", sa.String(255), nullable=False),
        sa.Column("subject", sa.String(255), nullable=True),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("template_code", sa.String(50), nullable=True),
        sa.Column("status", sa.String(20), default="PENDING"),
        sa.Column("sent_at", sa.DateTime(), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column("related_entity_type", sa.String(50), nullable=True),
        sa.Column("related_entity_id", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_table(
        "letter_templates",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("code", sa.String(50), unique=True, nullable=False),
        sa.Column("name", sa.String(100), nullable=False),
        sa.Column("subject_template", sa.String(255), nullable=True),
        sa.Column("body_template", sa.Text(), nullable=False),
        sa.Column("is_editable", sa.Boolean(), default=True),
    )
    op.create_table(
        "letter_instances",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("employee_id", sa.Integer(), sa.ForeignKey("employees.id"), nullable=False),
        sa.Column("template_id", sa.Integer(), sa.ForeignKey("letter_templates.id"), nullable=False),
        sa.Column("generated_by_user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("generated_at", sa.DateTime(), server_default=sa.func.now()),
        sa.Column("data_snapshot", sa.Text(), nullable=True),
        sa.Column("subject", sa.String(255), nullable=True),
        sa.Column("body", sa.Text(), nullable=True),
        sa.Column("sent_via_email", sa.Boolean(), default=False),
        sa.Column("email_log_id", sa.Integer(), sa.ForeignKey("email_logs.id"), nullable=True),
    )
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=True),
        sa.Column("action", sa.String(100), nullable=False),
        sa.Column("entity_type", sa.String(50), nullable=True),
        sa.Column("entity_id", sa.String(50), nullable=True),
        sa.Column("details", sa.Text(), nullable=True),
        sa.Column("ip_address", sa.String(50), nullable=True),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now()),
    )
    op.create_table(
        "events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("date", sa.Date(), nullable=False),
        sa.Column("event_type", sa.String(30), nullable=False),
        sa.Column("employee_id", sa.Integer(), sa.ForeignKey("employees.id"), nullable=True),
        sa.Column("description", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    for t in ["events", "audit_logs", "letter_instances", "letter_templates", "email_logs",
              "payslips", "payroll_periods", "salary_structures", "leave_requests", "leave_allocations",
              "leave_types", "attendance_correction_requests", "attendance_records", "holidays",
              "user_roles", "users", "roles", "employee_bank_details", "employees",
              "financial_years", "company_configs", "designations", "departments"]:
        op.drop_table(t)
