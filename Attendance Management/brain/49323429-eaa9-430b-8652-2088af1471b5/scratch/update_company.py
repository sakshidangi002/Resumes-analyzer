from app.db.session import SessionLocal
from app.models import CompanyConfig

db = SessionLocal()
cfg = db.query(CompanyConfig).first()
if cfg:
    print(f"Current Company Name: {cfg.name}")
    cfg.name = "Softwiz Infotech"
    db.commit()
    print(f"Updated Company Name to: {cfg.name}")
else:
    print("No CompanyConfig found. Creating one...")
    new_cfg = CompanyConfig(
        name="Softwiz Infotech",
        default_working_days_per_month=30,
        grace_time_minutes=15,
        half_day_threshold_hours=4
    )
    db.add(new_cfg)
    db.commit()
    print("Created CompanyConfig with name 'Softwiz Infotech'")
db.close()
