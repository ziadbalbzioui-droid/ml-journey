"""
Delete data/cmi-sleep/ RETENTION_DAYS after it was downloaded. Stdlib only, runs with the
system /usr/bin/python3 (3.9), so it does not depend on the project venv.

    /usr/bin/python3 projects/sleep-states/cleanup.py --install    # launchd job, daily + at login
    /usr/bin/python3 projects/sleep-states/cleanup.py --status     # deadline and days left
    /usr/bin/python3 projects/sleep-states/cleanup.py              # what the job runs
    /usr/bin/python3 projects/sleep-states/cleanup.py --uninstall

The job notifies daily during the last 7 days, deletes on the deadline, then removes itself.
Download date = data/cmi-sleep/.downloaded_at, or the folder's mtime if that file is missing.
"""
import datetime as dt
import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

RETENTION_DAYS = 60
DATA = Path(__file__).resolve().parents[2] / "data" / "cmi-sleep"
CACHE = Path.home() / ".cache" / "kagglehub" / "competitions" / "child-mind-institute-detect-sleep-states"
LABEL = "local.mlj.cmi-sleep-cleanup"
PLIST = Path.home() / "Library" / "LaunchAgents" / (LABEL + ".plist")
LOG = Path.home() / "Library" / "Logs" / (LABEL + ".log")


def today():
    return dt.datetime.now().astimezone().date()


def deadline():
    stamp = DATA / ".downloaded_at"
    if stamp.exists():
        start = dt.date.fromisoformat(stamp.read_text().strip())
    else:
        start = dt.datetime.fromtimestamp(DATA.stat().st_mtime).astimezone().date()
    return start + dt.timedelta(days=RETENTION_DAYS)


def notify(msg):
    subprocess.run(["osascript", "-e",
                    f'display notification "{msg}" with title "CMI sleep data"'], check=False)


def install():
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    plist = {
        "Label": LABEL,
        "ProgramArguments": ["/usr/bin/python3", str(Path(__file__).resolve())],
        "StartCalendarInterval": {"Hour": 10, "Minute": 0},
        "RunAtLoad": True,
        "StandardOutPath": str(LOG),
        "StandardErrorPath": str(LOG),
    }
    with open(PLIST, "wb") as f:
        plistlib.dump(plist, f)
    domain = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "bootout", domain, str(PLIST)], capture_output=True, check=False)
    subprocess.run(["launchctl", "bootstrap", domain, str(PLIST)], check=True)
    print("installed", PLIST)


def uninstall():
    subprocess.run(["launchctl", "bootout", f"gui/{os.getuid()}", str(PLIST)],
                   capture_output=True, check=False)
    PLIST.unlink(missing_ok=True)
    print("uninstalled", LABEL)


def status():
    job = "installed" if PLIST.exists() else "NOT installed"
    if not DATA.exists():
        print(f"no data at {DATA} · job {job}")
        return
    d = deadline()
    print(f"data at {DATA} · delete on {d} ({(d - today()).days} days left) · job {job}")


def run():
    now = today()
    if not DATA.exists():
        print(now, "no data")
        return
    d = deadline()
    left = (d - now).days
    if left <= 0:
        for p in (DATA, CACHE):
            if p.exists():
                shutil.rmtree(p)
                print(now, "deleted", p)
        notify(f"Deleted data/cmi-sleep ({RETENTION_DAYS}-day retention reached).")
        uninstall()
    elif left <= 7:
        notify(f"data/cmi-sleep will be deleted on {d} ({left} days).")
        print(now, "warned,", left, "days left")
    else:
        print(now, left, "days left")


if __name__ == "__main__":
    arg = sys.argv[1] if len(sys.argv) > 1 else ""
    {"--install": install, "--uninstall": uninstall, "--status": status}.get(arg, run)()
