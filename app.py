"""Hugging Face Spaces / Streamlit entry point for GlucoSight."""
from __future__ import annotations

from pathlib import Path

APP_FILE = Path(__file__).resolve().parent / "app" / "streamlit_app.py"

globals_dict = {
    "__file__": str(APP_FILE),
    "__name__": "__main__",
}
exec(compile(APP_FILE.read_text(encoding="utf-8"), str(APP_FILE), "exec"), globals_dict)