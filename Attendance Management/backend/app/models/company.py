"""Company config, financial year, holidays."""
from datetime import date
from sqlalchemy import Column, Integer, String, Date, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from app.db.base_class import Base


class CompanyConfig(Base):
    __tablename__ = "company_configs"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(200), nullable=False)
    default_working_days_per_month = Column(Integer, default=30)
    weekly_off_days = Column(String(100), nullable=True)  # e.g. "SAT,SUN" or JSON
    grace_time_minutes = Column(Integer, default=15)
    half_day_threshold_hours = Column(Integer, default=4)

    # ---- DSR reminder schedule (configurable by Admin / HR) ---------------
    # All values are interpreted in IST. Stored on the singleton company
    # config row; the 5 PM IST scheduler reads these every minute.
    dsr_reminder_enabled = Column(Boolean, nullable=False, default=True)
    # "HH:MM" 24h IST, e.g. "17:00".
    dsr_reminder_time = Column(String(5), nullable=False, default="17:00")
    # Comma-separated lowercase weekday tokens — mon, tue, wed, thu, fri, sat, sun.
    dsr_reminder_weekdays = Column(
        String(64), nullable=False, default="mon,tue,wed,thu,fri"
    )


class FinancialYear(Base):
    __tablename__ = "financial_years"

    id = Column(Integer, primary_key=True, index=True)
    start_date = Column(Date, nullable=False)  # e.g. 1 Apr
    end_date = Column(Date, nullable=False)   # e.g. 31 Mar next year
    is_current = Column(Boolean, default=False)
    name = Column(String(20), nullable=True)  # e.g. "2024-25"

    holidays = relationship("Holiday", back_populates="financial_year")


class Holiday(Base):
    __tablename__ = "holidays"

    id = Column(Integer, primary_key=True, index=True)
    date = Column(Date, nullable=False)
    name = Column(String(100), nullable=False)
    is_optional = Column(Boolean, default=False)
    financial_year_id = Column(Integer, ForeignKey("financial_years.id"), nullable=True)
    company_config_id = Column(Integer, ForeignKey("company_configs.id"), nullable=True)

    financial_year = relationship("FinancialYear", back_populates="holidays")
