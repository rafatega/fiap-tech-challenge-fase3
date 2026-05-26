from __future__ import annotations

import subprocess
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DASHBOARD_PATH = ROOT_DIR / "app" / "dashboard" / "analytics_airport_delay.py"


def main() -> None:
    if not DASHBOARD_PATH.exists():
        raise FileNotFoundError(f"Dashboard file not found: {DASHBOARD_PATH}")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            str(DASHBOARD_PATH),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
