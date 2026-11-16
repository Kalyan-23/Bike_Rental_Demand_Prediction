"""One-click runner script for the Seoul Bike Sharing application."""

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

if __name__ == "__main__":
    app_path = ROOT_DIR / "app.py"
    print(f"Starting Seoul Bike Sharing Dashboard from: {app_path}")
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\nApplication stopped.")
    except Exception as exc:
        print(f"Error launching application: {exc}")

