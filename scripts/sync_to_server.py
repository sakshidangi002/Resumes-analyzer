#!/usr/bin/env python3
"""
Push changed files from deploy/SoftwizApp/ to a remote Windows server over SSH/SFTP.

Three modes:
  - one-shot (default): sync the current deploy/SoftwizApp folder and exit.
  - --watch: re-sync any file you touch inside deploy/SoftwizApp.
  - --auto: full pipeline. Watch your SOURCE code (Attendance Management/{frontend,backend},
    backend/, frontend/). On every save: run "npm run build" (only if a React file
    changed), then package-for-server.ps1 -SkipBuild, then push diff to server, then
    restart the NSSM service if any backend Python was uploaded.

Prerequisites (laptop only, one-time):
    pip install -r scripts/requirements-sync.txt    # paramiko + watchdog

Setup:
    1. Copy sync_to_server.example.json -> sync_to_server.json and edit host/user/key.
    2. ssh-keygen -t ed25519  (if you have no key yet)
    3. Copy public key to server authorized_keys
    4. python scripts/sync_to_server.py --dry-run

Usage:
    python scripts/sync_to_server.py              # one-shot sync, smart restart
    python scripts/sync_to_server.py --watch      # watch deploy folder, sync on save
    python scripts/sync_to_server.py --auto       # FULL PIPELINE: watch source ->
                                                  # npm build (if needed) -> repackage -> sync
    python scripts/sync_to_server.py --dry-run    # preview only
    python scripts/sync_to_server.py --no-restart
    python scripts/sync_to_server.py --restart    # always restart after sync
    python scripts/sync_to_server.py --push-env
    python scripts/sync_to_server.py --include backend/main.py
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import logging
import os
import stat
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterable

try:
    import paramiko
except ImportError:
    print("Missing dependency: pip install paramiko", file=sys.stderr)
    sys.exit(1)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
DEFAULT_LOCAL = REPO_ROOT / "deploy" / "SoftwizApp"
CONFIG_PATH = SCRIPT_DIR / "sync_to_server.json"
LOG_PATH = SCRIPT_DIR / "sync.log"

EXCLUDE_GLOBS = [
    ".venv/**",
    "**/__pycache__/**",
    "*.pyc",
    "*.pyo",
    "backend/uploads/**",
    "backend/temp_*",
    # ChromaDB sqlite + index files MUST stay server-side (they're locked by
    # the running backend, so SFTP upload returns SSH_FX_FAILURE).
    "chromadb/**",
    "backend/chromadb/**",
    "hrms/backend/chromadb/**",
    "*.log",
    "sync.log",
    ".env",
    "hrms/backend/.env",
]

BACKEND_RESTART_GLOBS = [
    "hrms/backend/**/*.py",
    "backend/**/*.py",
]

# Paths that look like backend but must not trigger restart
BACKEND_RESTART_SKIP = [
    "backend/uploads/**",
]

# ---------------------------------------------------------------------------
# Auto mode: which source folders to watch and how changes map to actions
# ---------------------------------------------------------------------------
AUTO_SOURCE_DIRS = [
    "Attendance Management/frontend/src",
    "Attendance Management/frontend/public",
    "Attendance Management/backend/app",
    "Attendance Management/backend/alembic",
    "Attendance Management/backend/scripts",
    "backend",
    "frontend",
]

# Files (not whole dirs) that should trigger React rebuild if changed
REACT_TRIGGER_PREFIXES = [
    "Attendance Management/frontend/src/",
    "Attendance Management/frontend/public/",
    "Attendance Management/frontend/index.html",
    "Attendance Management/frontend/package.json",
    "Attendance Management/frontend/vite.config",
    "Attendance Management/frontend/tailwind.config",
    "Attendance Management/frontend/postcss.config",
    "Attendance Management/frontend/tsconfig",
]

# Substrings that mean "not a real source change, skip it"
AUTO_SKIP_SUBSTRINGS = [
    "__pycache__",
    ".venv",
    "node_modules",
    "frontend_build",
    "/uploads/",
    "\\uploads\\",
    "temp_",
    ".tsbuildinfo",
    ".pyc",
    ".pyo",
    "/chromadb/",
    "\\chromadb\\",
    "/data/",
    "\\data\\",
    "sync.log",
    ".swp",
    "~$",
]


def setup_logging() -> logging.Logger:
    logger = logging.getLogger("sync_to_server")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)
    fh = logging.FileHandler(LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def load_config(path: Path | None = None) -> dict:
    cfg_path = path or CONFIG_PATH
    if not cfg_path.is_file():
        example = SCRIPT_DIR / "sync_to_server.example.json"
        raise FileNotFoundError(
            f"Config not found: {cfg_path}\n"
            f"Copy {example.name} to {CONFIG_PATH.name} and edit it."
        )
    with open(cfg_path, encoding="utf-8") as f:
        cfg = json.load(f)
    required = ("host", "user", "remote_path")
    missing = [k for k in required if not cfg.get(k)]
    if missing:
        raise ValueError(f"Config missing required keys: {', '.join(missing)}")
    cfg.setdefault("port", 22)  # SSH port only — NOT the HTTP port uvicorn listens on
    cfg.setdefault("app_port", 5001)  # Softwiz HTTP (run-server.bat default)
    cfg.setdefault("key_path", "")
    cfg.setdefault("service_name", "SoftwizApp")
    cfg.setdefault("restart_on_backend_change", True)
    if "local_path" not in cfg:
        cfg["local_path"] = str(DEFAULT_LOCAL)
    return cfg


def _norm_rel(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.as_posix()


def path_matches_globs(rel_posix: str, globs: list[str]) -> bool:
    for pattern in globs:
        if fnmatch.fnmatch(rel_posix, pattern):
            return True
        if pattern.endswith("/**"):
            prefix = pattern[:-3]
            if rel_posix == prefix or rel_posix.startswith(prefix + "/"):
                return True
        if "**" in pattern:
            base = pattern.split("**")[0].rstrip("/")
            if base and (rel_posix == base or rel_posix.startswith(base + "/")):
                return True
    return False


def is_excluded(rel_posix: str, excludes: list[str]) -> bool:
    return path_matches_globs(rel_posix, excludes)


SKIP_WALK_DIRS = {".venv", "__pycache__", "uploads", "chromadb"}


def iter_local_files(root: Path, excludes: list[str]) -> Iterable[Path]:
    if not root.is_dir():
        raise FileNotFoundError(f"Local deploy folder not found: {root}")
    for dirpath, dirnames, filenames in os.walk(root):
        rel_dir = "" if Path(dirpath) == root else _norm_rel(Path(dirpath), root)
        pruned = []
        for d in dirnames:
            rel = f"{rel_dir}/{d}".lstrip("/") if rel_dir else d
            if d in SKIP_WALK_DIRS:
                continue
            if rel == "backend/uploads":
                continue
            if rel == "chromadb" and path_matches_globs("chromadb", excludes):
                continue
            if is_excluded(rel, excludes):
                continue
            pruned.append(d)
        dirnames[:] = pruned

        for name in filenames:
            local = Path(dirpath) / name
            rel = _norm_rel(local, root)
            if is_excluded(rel, excludes):
                continue
            yield local


def local_to_remote(remote_root: str, rel_posix: str) -> str:
    root = remote_root.replace("\\", "/").rstrip("/")
    return f"{root}/{rel_posix.replace('/', '/')}"


def open_ssh(cfg: dict) -> tuple[paramiko.SSHClient, paramiko.SFTPClient]:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.WarningPolicy())

    key_path = (cfg.get("key_path") or "").strip()
    connect_kw: dict = {
        "hostname": cfg["host"],
        "port": int(cfg.get("port", 22)),
        "username": cfg["user"],
        "timeout": 30,
        "allow_agent": True,
        "look_for_keys": not bool(key_path),
    }
    if key_path:
        expanded = os.path.expanduser(key_path)
        if not os.path.isfile(expanded):
            raise FileNotFoundError(f"SSH key not found: {expanded}")
        connect_kw["key_filename"] = expanded
        connect_kw["look_for_keys"] = False

    client.connect(**connect_kw)
    sftp = client.open_sftp()
    return client, sftp


def remote_stat(sftp: paramiko.SFTPClient, remote_path: str) -> tuple[int, float] | None:
    try:
        st = sftp.stat(remote_path)
        return int(st.st_size), float(st.st_mtime)
    except FileNotFoundError:
        return None
    except OSError:
        return None


def ensure_remote_dir(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    remote_dir = remote_dir.replace("\\", "/")
    parts = remote_dir.split("/")
    # Handle Windows drive letter C:
    if len(parts) > 1 and parts[0].endswith(":"):
        current = parts[0]
        start = 1
    else:
        current = ""
        start = 0
    for part in parts[start:]:
        if not part:
            continue
        current = f"{current}/{part}" if current else part
        try:
            sftp.mkdir(current)
        except OSError:
            pass


def upload_file(
    sftp: paramiko.SFTPClient,
    local: Path,
    remote: str,
    logger: logging.Logger,
) -> None:
    remote = remote.replace("\\", "/")
    parent = "/".join(remote.split("/")[:-1])
    if parent:
        ensure_remote_dir(sftp, parent)
    sftp.put(str(local), remote)
    local_mtime = local.stat().st_mtime
    try:
        sftp.utime(remote, (local_mtime, local_mtime))
    except OSError:
        logger.debug("Could not set utime on %s", remote)


def needs_upload(local: Path, sftp: paramiko.SFTPClient, remote: str) -> bool:
    remote_info = remote_stat(sftp, remote)
    if remote_info is None:
        return True
    remote_size, remote_mtime = remote_info
    st = local.stat()
    local_size = st.st_size
    local_mtime = st.st_mtime
    if local_size != remote_size:
        return True
    # Allow 2 second tolerance for FAT/Windows timestamp rounding
    if abs(local_mtime - remote_mtime) > 2.0:
        return True
    return False


def needs_migration(uploaded_rel: list[str]) -> bool:
    """True if any new alembic version file was uploaded this sync."""
    return any(
        rel.startswith("hrms/backend/alembic/versions/") and rel.endswith(".py")
        for rel in uploaded_rel
    )


def needs_pip_install(uploaded_rel: list[str]) -> bool:
    """True if requirements changed (combined or per-component)."""
    return any(
        rel in ("requirements-all.txt",)
        or rel.endswith("/requirements.txt")
        or rel.endswith("requirements.txt")
        for rel in uploaded_rel
    )


def run_remote_pip_install(
    ssh: paramiko.SSHClient, remote_path: str, logger: logging.Logger
) -> None:
    """Run ``pip install -r requirements-all.txt`` on the server.

    Triggered automatically when requirements files are uploaded, or via
    ``--install-deps``. Surfaces clear errors instead of letting the backend
    silently boot without APScheduler / pywebpush / etc.
    """
    root = remote_path.rstrip("/\\")
    venv_py = root + r"\.venv\Scripts\python.exe"
    req = root + r"\requirements-all.txt"
    cmd = (
        f'cmd /c ""{venv_py}" -m pip install --disable-pip-version-check '
        f'-r "{req}""'
    )
    logger.info("Installing/updating Python deps on server (this may take a minute)...")
    exit_status, out, err = _run_remote(ssh, cmd, timeout=900)
    if out:
        for line in out.splitlines()[-25:]:
            if line.strip():
                logger.info("  [pip] %s", line.rstrip())
    if err:
        for line in err.splitlines()[-25:]:
            if line.strip():
                logger.info("  [pip] %s", line.rstrip())
    if exit_status != 0:
        logger.error(
            "pip install FAILED on server (exit %s). New features that depend "
            "on missing packages (APScheduler scheduler, Web Push, etc.) will "
            "not work until you run it manually on the server:\n"
            '    cd %s\n'
            '    "%s" -m pip install -r requirements-all.txt',
            exit_status, root, venv_py,
        )
    else:
        logger.info("pip install OK")


def run_remote_migrations(ssh: paramiko.SSHClient, remote_path: str, logger: logging.Logger) -> None:
    """Run ``alembic upgrade head`` on the remote server.

    Assumes the standard layout produced by package-for-server.ps1:
        <remote_path>\\.venv\\Scripts\\python.exe
        <remote_path>\\hrms\\backend\\alembic.ini
    Sets PYTHONPATH so app.* resolves like in setup-server.bat.
    """
    venv_py = remote_path.rstrip("/\\") + r"\.venv\Scripts\python.exe"
    backend = remote_path.rstrip("/\\") + r"\hrms\backend"
    cmd = (
        f'cmd /c "set PYTHONPATH={backend};{remote_path}&& '
        f'cd /d {backend}&& "{venv_py}" -m alembic upgrade head"'
    )
    logger.info("Running alembic upgrade head on server...")
    exit_status, out, err = _run_remote(ssh, cmd, timeout=300)
    if out:
        for line in out.splitlines()[-20:]:
            logger.info("  [alembic] %s", line.rstrip())
    if err:
        # Alembic logs to stderr by default; only escalate on non-zero exit
        for line in err.splitlines()[-20:]:
            if line.strip():
                logger.info("  [alembic] %s", line.rstrip())
    if exit_status != 0:
        logger.error(
            "alembic upgrade head FAILED on server (exit %s). New endpoints "
            "that depend on the new tables will return 500 until you run it "
            "manually:\n"
            "    cd %s\n"
            '    set PYTHONPATH=%s;%s\n'
            '    "%s" -m alembic upgrade head',
            exit_status, backend, backend, remote_path, venv_py,
        )
    else:
        logger.info("alembic upgrade head OK")


def should_restart(uploaded_rel: list[str], smart: bool) -> bool:
    if not smart:
        return False
    for rel in uploaded_rel:
        if path_matches_globs(rel, BACKEND_RESTART_SKIP):
            continue
        if path_matches_globs(rel, BACKEND_RESTART_GLOBS):
            return True
        if rel.startswith("hrms/backend/app/") and rel.endswith(".py"):
            return True
        if (
            rel.startswith("backend/")
            and rel.endswith(".py")
            and not rel.startswith("backend/uploads/")
        ):
            return True
    return False


def uvicorn_port(cfg: dict) -> int:
    """HTTP port for the unified Softwiz app (``cfg['port']`` is SSH, usually 22)."""
    return int(cfg.get("app_port", cfg.get("http_port", 5001)))


def _run_remote(ssh: paramiko.SSHClient, cmd: str, timeout: int = 60) -> tuple[int, str, str]:
    """Run a remote shell command. Returns (exit_status, stdout, stderr)."""
    _, stdout, stderr = ssh.exec_command(cmd, timeout=timeout)
    exit_status = stdout.channel.recv_exit_status()
    out = stdout.read().decode("utf-8", errors="replace").strip()
    err = stderr.read().decode("utf-8", errors="replace").strip()
    return exit_status, out, err


def restart_uvicorn_process(
    ssh: paramiko.SSHClient,
    remote_path: str,
    port: int,
    logger: logging.Logger,
) -> bool:
    """Kill whatever is listening on ``port`` and start uvicorn again (no Windows service).

    Many dev servers run ``run-server.bat`` in a console instead of NSSM. This
    is the fallback when ``SoftwizApp`` is not registered as a service.

    Implementation notes (hard-won):
    - ``$pid`` is a PowerShell *automatic* variable (current process id). Any
      ``$pid = ...`` assignment silently fails and ``Stop-Process -Id $pid``
      targets nothing, so the original uvicorn keeps owning the port while
      we cheerfully log a fake "OK:uvicorn:pid=...". Use ``$listenerPid``.
    - ``Start-Process`` over SSH does NOT properly detach: the child cmd dies
      with the SSH session, so uvicorn never gets past startup. Spawn the
      launcher .bat via ``wmic process call create`` instead — that creates
      the process under WMI, completely independent of the SSH channel.
    - Don't trust spawn-time success. The bat may exit instantly (port busy,
      bad PYTHONPATH, broken import). Poll the port AFTER launching and only
      claim success once something is actually LISTENING.
    """
    root = remote_path.rstrip("/\\")
    backend = root + r"\hrms\backend"
    venv_py = root + r"\.venv\Scripts\python.exe"
    boot_bat = root + r"\_softwiz_boot.bat"
    log_path = root + r"\_softwiz_boot.log"

    logger.info(
        "No Windows service found — restarting uvicorn on port %s under %s",
        port,
        root,
    )

    # Stop anything currently bound to the app port, plus any orphaned
    # `python ... uvicorn app.main` processes from previous half-dead boots.
    # IMPORTANT: cmd.exe (between paramiko and powershell) strips inner
    # double quotes, so the PowerShell here uses single quotes only.
    kill_ps = (
        f"$port={port}; "
        f"$pat = ':' + $port + '\\s+.*LISTENING'; "
        f"netstat -ano | Select-String $pat | ForEach-Object {{ "
        f"$listenerPid = ($_.Line -split '\\s+')[-1]; "
        f"if ($listenerPid -match '^\\d+$') {{ "
        f"try {{ Stop-Process -Id ([int]$listenerPid) -Force -ErrorAction Stop }} "
        f"catch {{ Write-Output ('KILL-FAIL pid=' + $listenerPid + ' : ' + $_.Exception.Message) }} "
        f"}} "
        f"}}; "
        f"Get-CimInstance Win32_Process -Filter ('name=' + [char]39 + 'python.exe' + [char]39) -ErrorAction SilentlyContinue | "
        f"Where-Object {{ $_.CommandLine -like '*uvicorn*app.main*' }} | "
        f"ForEach-Object {{ "
        f"try {{ Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop }} catch {{}} "
        f"}}; "
        f"Start-Sleep -Seconds 2"
    )
    kill_cmd = f'powershell -NoProfile -ExecutionPolicy Bypass -Command "{kill_ps}"'
    _, kill_out, _ = _run_remote(ssh, kill_cmd, timeout=45)
    if kill_out and "KILL-FAIL" in kill_out:
        logger.warning("  [uvicorn-kill] %s", kill_out.strip()[:400])

    # Write a self-contained launcher .bat that sets PYTHONPATH, cd's to the
    # backend folder, redirects all output to a log, and execs uvicorn. We
    # write it via SFTP because cmd /c heredocs over SSH are a quoting maze.
    bat = (
        "@echo off\r\n"
        f"cd /d {backend}\r\n"
        f"set PYTHONPATH={backend};{root}\r\n"
        f"\"{venv_py}\" -m uvicorn app.main:app "
        f"--host 0.0.0.0 --port {port} > \"{log_path}\" 2>&1\r\n"
    )
    try:
        sftp = ssh.open_sftp()
        try:
            with sftp.open(boot_bat.replace("\\", "/"), "w") as f:
                f.write(bat)
        finally:
            sftp.close()
    except Exception as e:
        logger.error("Could not upload boot launcher .bat: %s", e)
        return False

    # Detach the launcher via wmic so it survives this SSH session closing.
    spawn_cmd = f'wmic process call create "cmd /c {boot_bat}", "{root}"'
    exit_status, out, err = _run_remote(ssh, spawn_cmd, timeout=30)
    if exit_status != 0 or "ReturnValue = 0" not in out:
        logger.error(
            "wmic could not spawn the launcher (exit=%s)\nstdout: %s\nstderr: %s",
            exit_status, out.strip()[:600], err.strip()[:400],
        )
        return False

    # Poll the port for up to ~40 s — embedding model + ChromaDB take time on
    # cold start. PowerShell uses single quotes here because cmd.exe strips
    # the inner double quotes of `powershell -Command "..."`.
    poll_ps = (
        f"for ($i=0; $i -lt 40; $i++) {{ "
        f"$c = Get-NetTCPConnection -LocalPort {port} -State Listen -ErrorAction SilentlyContinue; "
        f"if ($c) {{ Write-Output ('LISTENING pid=' + $c[0].OwningProcess + ' after=' + $i + 's'); exit 0 }} "
        f"else {{ Start-Sleep 1 }} "
        f"}}; Write-Output 'NOT-LISTENING'; exit 2"
    )
    poll_cmd = f'powershell -NoProfile -ExecutionPolicy Bypass -Command "{poll_ps}"'
    exit_status, out, _ = _run_remote(ssh, poll_cmd, timeout=60)
    if "LISTENING " in out:
        logger.info("  [uvicorn] %s", out.strip())
        logger.info("Backend restarted via uvicorn process (port %s)", port)
        return True

    # Boot failed. Surface the log so the user knows why.
    # Single-quoted path -> no inner double quotes for cmd to mangle.
    tail_ps = (
        f"$p = '{log_path}'; "
        f"if (Test-Path $p) {{ Get-Content -Tail 30 $p }} else {{ Write-Output 'NO-LOG' }}"
    )
    tail_cmd = f'powershell -NoProfile -Command "{tail_ps}"'
    _, tail_out, _ = _run_remote(ssh, tail_cmd, timeout=15)
    logger.error(
        "uvicorn did not start listening on port %s. Tail of %s:\n%s",
        port, log_path, (tail_out.strip() or "(empty)")[:2000],
    )
    return False


def restart_service(
    ssh: paramiko.SSHClient,
    service_name: str,
    logger: logging.Logger,
    *,
    remote_path: str = "C:/SoftwizApp",
    port: int = 5001,
) -> None:
    """Restart the unified Softwiz backend on the remote Windows host.

    Tries, in order:
      1. Windows service via nssm / sc / Restart-Service
      2. Kill port listener + ``Start-Process`` uvicorn (console / manual deploy)
    """
    logger.info("Restarting service: %s", service_name)

    attempts: list[tuple[str, str]] = [
        ("nssm (PATH)", f'nssm restart "{service_name}"'),
        ("nssm (C:\\nssm)", f'"C:\\nssm\\nssm.exe" restart "{service_name}"'),
        ("nssm (C:\\tools\\nssm)", f'"C:\\tools\\nssm\\nssm.exe" restart "{service_name}"'),
        ("sc stop+start", f'sc stop "{service_name}" & ping 127.0.0.1 -n 3 > nul & sc start "{service_name}"'),
        ("PowerShell Restart-Service", f'powershell -NoProfile -Command "Restart-Service -Name \\"{service_name}\\" -Force"'),
    ]

    last_err = ""
    for label, cmd in attempts:
        exit_status, out, err = _run_remote(ssh, cmd, timeout=60)
        if out:
            logger.info("  [%s] stdout: %s", label, out[:400])
        if err:
            last_err = err
            logger.debug("  [%s] stderr: %s", label, err[:400])
        if exit_status == 0:
            logger.info("Restart succeeded via: %s", label)
            return
        logger.warning("  [%s] failed (exit %s)", label, exit_status)

    # Service not installed — typical when the server runs run-server.bat manually.
    if restart_uvicorn_process(ssh, remote_path, port, logger):
        return

    logger.error(
        "Could not restart '%s' (no Windows service) and uvicorn fallback failed. "
        "On the server (%s), open a console and run:\n"
        "    cd %s\n"
        "    run-server.bat\n"
        "Or install NSSM and register the service (see SERVER-SETUP.txt).\n"
        "Last service error: %s",
        service_name,
        remote_path,
        remote_path,
        last_err or "(none)",
    )


def build_excludes(args: argparse.Namespace) -> list[str]:
    excludes = list(EXCLUDE_GLOBS)
    if args.push_env:
        excludes = [e for e in excludes if e not in (".env", "hrms/backend/.env")]
    if args.include_chroma:
        excludes = [e for e in excludes if e != "chromadb/**"]
    return excludes


def sync_file(
    local: Path,
    root: Path,
    cfg: dict,
    sftp: paramiko.SFTPClient,
    excludes: list[str],
    dry_run: bool,
    logger: logging.Logger,
) -> str | None:
    """Sync one file; returns relative posix path if uploaded, else None."""
    try:
        rel = _norm_rel(local.resolve(), root.resolve())
    except ValueError:
        return None
    if is_excluded(rel, excludes):
        return None
    remote = local_to_remote(cfg["remote_path"], rel)
    if not needs_upload(local, sftp, remote):
        return None
    if dry_run:
        logger.info("[dry-run] would upload: %s", rel)
        return rel
    logger.info("upload: %s", rel)
    upload_file(sftp, local, remote, logger)
    return rel


def sync_once(
    cfg: dict,
    excludes: list[str],
    dry_run: bool,
    include_only: str | None,
    logger: logging.Logger,
) -> list[str]:
    root = Path(cfg["local_path"]).resolve()
    uploaded: list[str] = []

    if include_only:
        local = root / include_only.replace("\\", "/")
        if not local.is_file():
            raise FileNotFoundError(f"File not found under deploy folder: {include_only}")
        client, sftp = open_ssh(cfg)
        try:
            rel = sync_file(local, root, cfg, sftp, excludes, dry_run, logger)
            if rel:
                uploaded.append(rel)
        finally:
            sftp.close()
            client.close()
        return uploaded

    client, sftp = open_ssh(cfg)
    try:
        skipped = 0
        failed = 0
        for local in iter_local_files(root, excludes):
            rel = _norm_rel(local, root)
            remote = local_to_remote(cfg["remote_path"], rel)
            try:
                if not needs_upload(local, sftp, remote):
                    skipped += 1
                    continue
                if dry_run:
                    logger.info("[dry-run] would upload: %s", rel)
                    uploaded.append(rel)
                    continue
                logger.info("upload: %s", rel)
                upload_file(sftp, local, remote, logger)
                uploaded.append(rel)
            except Exception as e:
                # Don't let a single locked/permission-denied file abort the
                # whole sync. Log and keep going so critical files (frontend
                # bundles, .py routes, etc.) still get pushed.
                failed += 1
                logger.error("upload FAILED: %s  -> %s", rel, e)
                continue
        logger.info(
            "Sync complete: %d uploaded, %d unchanged, %d failed%s",
            len(uploaded),
            skipped,
            failed,
            " (dry-run)" if dry_run else "",
        )
    finally:
        sftp.close()
        client.close()
    return uploaded


class DebouncedSyncHandler:
    def __init__(
        self,
        cfg: dict,
        excludes: list[str],
        dry_run: bool,
        restart_force: bool,
        restart_never: bool,
        logger: logging.Logger,
    ):
        self.cfg = cfg
        self.excludes = excludes
        self.dry_run = dry_run
        self.restart_force = restart_force
        self.restart_never = restart_never
        self.logger = logger
        self.root = Path(cfg["local_path"]).resolve()
        self._lock = threading.Lock()
        self._pending: dict[str, float] = {}
        self._timer: threading.Timer | None = None
        self._ssh: paramiko.SSHClient | None = None
        self._sftp: paramiko.SFTPClient | None = None
        self.debounce_sec = 0.5

    def _ensure_connection(self) -> tuple[paramiko.SSHClient, paramiko.SFTPClient]:
        if self._ssh is None or self._sftp is None:
            self._ssh, self._sftp = open_ssh(self.cfg)
        return self._ssh, self._sftp

    def _close_connection(self) -> None:
        if self._sftp:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None
        if self._ssh:
            try:
                self._ssh.close()
            except Exception:
                pass
            self._ssh = None

    def schedule(self, path: str) -> None:
        with self._lock:
            self._pending[path] = time.time()
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_sec, self._flush)
            self._timer.daemon = True
            self._timer.start()

    def _flush(self) -> None:
        with self._lock:
            paths = list(self._pending.keys())
            self._pending.clear()
        if not paths:
            return
        uploaded: list[str] = []
        try:
            ssh, sftp = self._ensure_connection()
            for path in paths:
                local = Path(path)
                if not local.is_file():
                    continue
                try:
                    local.resolve().relative_to(self.root)
                except ValueError:
                    continue
                try:
                    rel = sync_file(local, self.root, self.cfg, sftp, self.excludes, self.dry_run, self.logger)
                    if rel:
                        uploaded.append(rel)
                except Exception as e:
                    # Single-file failures (locked DB on server, perms, etc.)
                    # should not break the watch loop or kill the SFTP session
                    # for the other queued files.
                    self.logger.error("upload FAILED: %s  -> %s", path, e)
            if uploaded and not self.restart_never and not self.dry_run:
                smart = self.cfg.get("restart_on_backend_change", True)
                must_migrate = needs_migration(uploaded)
                must_install = needs_pip_install(uploaded)
                if (
                    must_migrate
                    or must_install
                    or self.restart_force
                    or should_restart(uploaded, smart)
                ):
                    remote_path = self.cfg.get("remote_path", "C:/SoftwizApp")
                    if must_install:
                        run_remote_pip_install(ssh, remote_path, self.logger)
                    if must_migrate:
                        run_remote_migrations(ssh, remote_path, self.logger)
                    restart_service(
                        ssh,
                        self.cfg.get("service_name", "SoftwizApp"),
                        self.logger,
                        remote_path=remote_path,
                        port=uvicorn_port(self.cfg),
                    )
        except Exception as e:
            self.logger.error("Watch sync failed: %s", e)
            self._close_connection()

    def stop(self) -> None:
        if self._timer:
            self._timer.cancel()
        self._flush()
        self._close_connection()


def run_watch(cfg: dict, args: argparse.Namespace, logger: logging.Logger) -> None:
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        print("Missing dependency: pip install watchdog", file=sys.stderr)
        sys.exit(1)

    excludes = build_excludes(args)
    root = Path(cfg["local_path"]).resolve()
    handler_obj = DebouncedSyncHandler(
        cfg,
        excludes,
        args.dry_run,
        args.restart,
        args.no_restart,
        logger,
    )

    class Handler(FileSystemEventHandler):
        def on_modified(self, event):
            if not event.is_directory:
                handler_obj.schedule(event.src_path)

        def on_created(self, event):
            if not event.is_directory:
                handler_obj.schedule(event.src_path)

    observer = Observer()
    observer.schedule(Handler(), str(root), recursive=True)
    observer.start()
    logger.info("Watching %s (Ctrl+C to stop)", root)
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping watch...")
    finally:
        observer.stop()
        observer.join()
        handler_obj.stop()


# ---------------------------------------------------------------------------
# Auto pipeline: source change -> npm build (if React) -> package -> sync -> restart
# ---------------------------------------------------------------------------

def run_command(cmd: list[str], cwd: Path, logger: logging.Logger) -> None:
    """Run an external command, log output, raise on failure.

    On Windows, npm/npx are .cmd shims and need shell=True. PowerShell must NOT
    use shell=True with a joined string — paths like ``C:\\sakshi folder\\...``
    break at the first space and ``-File`` only receives ``C:\\sakshi``.
    """
    is_windows = sys.platform == "win32"
    needs_shell = is_windows and cmd and cmd[0].lower() in {"npm", "npx"}
    if needs_shell:
        pretty = subprocess.list2cmdline([str(c) for c in cmd])
    else:
        pretty = " ".join(str(c) for c in cmd)
    logger.info("$ %s   (cwd=%s)", pretty, cwd.name)

    try:
        if needs_shell:
            result = subprocess.run(
                pretty,
                cwd=str(cwd),
                shell=True,
                capture_output=True,
                text=True,
            )
        else:
            result = subprocess.run(
                [str(c) for c in cmd],
                cwd=str(cwd),
                capture_output=True,
                text=True,
            )
    except FileNotFoundError as e:
        raise RuntimeError(f"Command not found: {cmd[0]} ({e})") from e

    if result.stdout:
        for line in result.stdout.splitlines()[-25:]:
            if line.strip():
                logger.info("  | %s", line.rstrip())
    if result.returncode != 0:
        if result.stderr:
            for line in result.stderr.splitlines()[-25:]:
                if line.strip():
                    logger.error("  ! %s", line.rstrip())
        raise RuntimeError(f"Command failed (exit {result.returncode}): {cmd[0]}")


def classify_source_path(repo_relative: str) -> tuple[bool, bool]:
    """Return (is_source_change, is_react_change) for a path relative to repo root."""
    norm = repo_relative.replace("\\", "/")
    if any(s in norm for s in AUTO_SKIP_SUBSTRINGS):
        return False, False
    is_react = any(norm.startswith(p) for p in REACT_TRIGGER_PREFIXES)
    return True, is_react


class AutoPipeline:
    """Debounce source-file changes, then run the full build->sync pipeline."""

    def __init__(self, cfg: dict, args: argparse.Namespace, logger: logging.Logger):
        self.cfg = cfg
        self.args = args
        self.logger = logger
        self.repo_root = REPO_ROOT
        self.package_script = SCRIPT_DIR / "package-for-server.ps1"
        self.frontend_dir = REPO_ROOT / "Attendance Management" / "frontend"
        self._lock = threading.Lock()
        self._timer: threading.Timer | None = None
        self._building = False
        self._react_pending = False
        self._any_pending = False
        self._rerun_after_build = False
        self.debounce_sec = 2.0

    def on_change(self, repo_relative: str) -> None:
        is_source, is_react = classify_source_path(repo_relative)
        if not is_source:
            return
        self.logger.info("changed: %s%s", repo_relative, "  [react]" if is_react else "")
        with self._lock:
            self._any_pending = True
            if is_react:
                self._react_pending = True
            if self._building:
                self._rerun_after_build = True
                return
            if self._timer:
                self._timer.cancel()
            self._timer = threading.Timer(self.debounce_sec, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        with self._lock:
            if not self._any_pending or self._building:
                return
            need_react = self._react_pending
            self._any_pending = False
            self._react_pending = False
            self._building = True
        try:
            self._run_pipeline(need_react)
        finally:
            with self._lock:
                self._building = False
                rerun = self._rerun_after_build
                self._rerun_after_build = False
                if rerun and self._any_pending:
                    if self._timer:
                        self._timer.cancel()
                    self._timer = threading.Timer(self.debounce_sec, self._fire)
                    self._timer.daemon = True
                    self._timer.start()

    def _run_pipeline(self, need_react: bool) -> None:
        start = time.time()
        try:
            if need_react:
                if not self.frontend_dir.is_dir():
                    raise RuntimeError(f"Frontend dir not found: {self.frontend_dir}")
                self.logger.info("--- [1/3] npm run build ---")
                run_command(["npm", "run", "build"], cwd=self.frontend_dir, logger=self.logger)
            else:
                self.logger.info("--- [1/3] skip npm (no React change) ---")

            if not self.package_script.is_file():
                raise RuntimeError(f"Package script not found: {self.package_script}")
            self.logger.info("--- [2/3] package-for-server.ps1 -SkipBuild ---")
            package_script = str(self.package_script).replace("'", "''")
            run_command(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    f"& '{package_script}' -SkipBuild",
                ],
                cwd=self.repo_root,
                logger=self.logger,
            )

            self.logger.info("--- [3/3] sync to server ---")
            excludes = build_excludes(self.args)
            uploaded = sync_once(
                self.cfg,
                excludes,
                self.args.dry_run,
                None,
                self.logger,
            )

            if uploaded and not self.args.no_restart and not self.args.dry_run:
                smart = self.cfg.get("restart_on_backend_change", True)
                must_migrate = needs_migration(uploaded)
                must_install = needs_pip_install(uploaded)
                if (
                    must_migrate
                    or must_install
                    or self.args.restart
                    or should_restart(uploaded, smart)
                ):
                    ssh, _ = open_ssh(self.cfg)
                    try:
                        remote_path = self.cfg.get("remote_path", "C:/SoftwizApp")
                        if must_install:
                            run_remote_pip_install(ssh, remote_path, self.logger)
                        if must_migrate:
                            run_remote_migrations(ssh, remote_path, self.logger)
                        restart_service(
                            ssh,
                            self.cfg.get("service_name", "SoftwizApp"),
                            self.logger,
                            remote_path=remote_path,
                            port=uvicorn_port(self.cfg),
                        )
                    finally:
                        ssh.close()

            elapsed = time.time() - start
            self.logger.info("Auto pipeline OK in %.1fs", elapsed)
        except Exception as e:
            self.logger.error("Auto pipeline failed: %s", e)

    def stop(self) -> None:
        with self._lock:
            if self._timer:
                self._timer.cancel()


def run_auto(cfg: dict, args: argparse.Namespace, logger: logging.Logger) -> None:
    """Watch source folders and run npm build + package + sync on every change."""
    try:
        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer
    except ImportError:
        print("Missing dependency: pip install watchdog", file=sys.stderr)
        sys.exit(1)

    pipeline = AutoPipeline(cfg, args, logger)

    class Handler(FileSystemEventHandler):
        def _push(self, event):
            if event.is_directory:
                return
            try:
                rel = Path(event.src_path).resolve().relative_to(REPO_ROOT).as_posix()
            except ValueError:
                return
            pipeline.on_change(rel)

        def on_modified(self, event):
            self._push(event)

        def on_created(self, event):
            self._push(event)

        def on_moved(self, event):
            # treat as create on dest
            if event.is_directory:
                return
            try:
                rel = Path(event.dest_path).resolve().relative_to(REPO_ROOT).as_posix()
            except (ValueError, AttributeError):
                return
            pipeline.on_change(rel)

    observer = Observer()
    watched = 0
    for rel in AUTO_SOURCE_DIRS:
        d = REPO_ROOT / rel
        if d.is_dir():
            observer.schedule(Handler(), str(d), recursive=True)
            logger.info("Watching: %s", rel)
            watched += 1
        else:
            logger.warning("Skip (not found): %s", rel)
    if watched == 0:
        logger.error("No source folders found. Are you running from the repo root?")
        return

    observer.start()
    logger.info("Auto pipeline ready. Edit source, save, watch it deploy.  (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Stopping auto pipeline...")
    finally:
        observer.stop()
        observer.join()
        pipeline.stop()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sync deploy/SoftwizApp to remote server over SSH/SFTP")
    p.add_argument("--config", type=Path, default=CONFIG_PATH, help="Path to JSON config")
    p.add_argument("--watch", action="store_true", help="Watch deploy folder and sync on change")
    p.add_argument(
        "--auto",
        action="store_true",
        help="Full pipeline: watch source -> npm build (if React) -> package -> sync -> restart",
    )
    p.add_argument("--dry-run", action="store_true", help="Show what would upload without copying")
    p.add_argument("--no-restart", action="store_true", help="Never restart the Windows service")
    p.add_argument("--restart", action="store_true", help="Always restart after sync")
    p.add_argument("--push-env", action="store_true", help="Include .env files in sync")
    p.add_argument("--include-chroma", action="store_true", help="Include chromadb/ in sync")
    p.add_argument("--include", metavar="REL_PATH", help="Sync a single file relative to SoftwizApp")
    p.add_argument(
        "--migrate-now",
        action="store_true",
        help="Run alembic upgrade head + restart on the server without uploading anything.",
    )
    p.add_argument(
        "--install-deps",
        action="store_true",
        help=(
            "Run `pip install -r requirements-all.txt` on the server and "
            "restart. Use after adding new Python dependencies "
            "(e.g. apscheduler, pywebpush). Combine with --migrate-now to do "
            "deps + migration in one shot."
        ),
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    logger = setup_logging()
    try:
        cfg = load_config(args.config)
    except (FileNotFoundError, ValueError) as e:
        logger.error("%s", e)
        return 1

    local = Path(cfg["local_path"])
    if not local.is_dir():
        logger.error("Local path does not exist: %s\nRun package-for-server.ps1 first.", local)
        return 1

    excludes = build_excludes(args)

    if args.migrate_now or args.install_deps:
        actions = []
        if args.install_deps:
            actions.append("pip install -r requirements-all.txt")
        if args.migrate_now:
            actions.append("alembic upgrade head")
        actions.append("restart")
        logger.info(
            "Server maintenance on %s: %s",
            cfg.get("host", "?"),
            " + ".join(actions),
        )
        try:
            ssh, _ = open_ssh(cfg)
            try:
                remote_path = cfg.get("remote_path", "C:/SoftwizApp")
                if args.install_deps:
                    run_remote_pip_install(ssh, remote_path, logger)
                if args.migrate_now:
                    run_remote_migrations(ssh, remote_path, logger)
                restart_service(
                    ssh,
                    cfg.get("service_name", "SoftwizApp"),
                    logger,
                    remote_path=remote_path,
                    port=uvicorn_port(cfg),
                )
            finally:
                ssh.close()
        except Exception as e:
            logger.error("Server maintenance failed: %s", e)
            return 1
        return 0

    if args.auto:
        run_auto(cfg, args, logger)
        return 0

    if args.watch:
        run_watch(cfg, args, logger)
        return 0

    try:
        uploaded = sync_once(cfg, excludes, args.dry_run, args.include, logger)
    except Exception as e:
        logger.error("Sync failed: %s", e)
        return 1

    if not args.no_restart and not args.dry_run:
        smart = cfg.get("restart_on_backend_change", True)
        must_migrate = bool(uploaded) and needs_migration(uploaded)
        must_install = bool(uploaded) and needs_pip_install(uploaded)
        do_restart = (
            must_migrate
            or must_install
            or args.restart
            or (uploaded and should_restart(uploaded, smart))
        )
        if do_restart:
            try:
                ssh, _ = open_ssh(cfg)
                try:
                    remote_path = cfg.get("remote_path", "C:/SoftwizApp")
                    if must_install:
                        run_remote_pip_install(ssh, remote_path, logger)
                    if must_migrate:
                        run_remote_migrations(ssh, remote_path, logger)
                    restart_service(
                        ssh,
                        cfg.get("service_name", "SoftwizApp"),
                        logger,
                        remote_path=remote_path,
                        port=uvicorn_port(cfg),
                    )
                finally:
                    ssh.close()
            except Exception as e:
                logger.error("Restart failed: %s", e)
                return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
