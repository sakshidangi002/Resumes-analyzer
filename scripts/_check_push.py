"""Quick check: is Web Push fully configured on the server?

Reports whether VAPID keys are set, whether pywebpush is installed in the
running venv, and how many push subscriptions exist in the DB.
"""
import json, paramiko

with open("scripts/sync_to_server.json") as f:
    cfg = json.load(f)
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(
    cfg["host"],
    port=cfg.get("port", 22),
    username=cfg.get("user"),
    key_filename=cfg.get("key_path"),
)


def run(cmd, timeout=30):
    _, so, se = ssh.exec_command(cmd, timeout=timeout)
    rc = so.channel.recv_exit_status()
    out = so.read().decode(errors="replace")
    err = se.read().decode(errors="replace")
    print("---", cmd[:120])
    if out.strip():
        print(out.strip()[:1500])
    if err.strip():
        print("STDERR:", err.strip()[:1500])
    print(f"(exit={rc})")


run(r'powershell -NoProfile -Command "Get-ChildItem C:\SoftwizApp -Recurse -Filter .env -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName"')
run(r'findstr /b "VAPID_ FRONTEND_URL APP_BASE_URL" C:\SoftwizApp\.env')
run(r'findstr /b "VAPID_ FRONTEND_URL APP_BASE_URL" C:\SoftwizApp\hrms\backend\.env')

# Subscription count + ask backend if it's configured
run(
    r'C:\SoftwizApp\.venv\Scripts\python.exe -c "'
    r"import os, sys; "
    r"os.environ.setdefault('PYTHONPATH', r'C:\SoftwizApp\hrms\backend;C:\SoftwizApp'); "
    r"sys.path.insert(0, r'C:\SoftwizApp\hrms\backend'); "
    r"sys.path.insert(0, r'C:\SoftwizApp'); "
    r"from app.services.web_push_service import is_configured; "
    r"print('web_push is_configured:', is_configured()); "
    r"from app.db.session import SessionLocal; "
    r"from app.models import PushSubscription; "
    r"db = SessionLocal(); "
    r"print('push subscriptions in DB:', db.query(PushSubscription).count()); "
    r"db.close()"
    r'"'
)

ssh.close()
