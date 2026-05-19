"""
Verify ChatGPT's claims about the HUPA dataset against the actual files.

This script empirically checks every claim listed by the user and prints a
verdict table (CORRECT / INCORRECT / PARTIAL) with supporting numbers.

Run from project root on Windows PowerShell:
    $env:PYTHONIOENCODING='utf-8'; python src/verify_hupa.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent.parent
PREP_DIR = ROOT / "data" / "data_hupa" / "Preprocessed"

if not PREP_DIR.exists():
    sys.exit(f"[FATAL] Preprocessed directory not found: {PREP_DIR}")


# --------------------------------------------------------------------------- #
# Load every participant file
# --------------------------------------------------------------------------- #
def load_participant(path: Path) -> pd.DataFrame:
    """Load a single HUPA preprocessed CSV with semicolon separator."""
    df = pd.read_csv(path, sep=";", parse_dates=["time"])
    return df


files = sorted(PREP_DIR.glob("HUPA*P.csv"))
pid_of = lambda p: p.name.replace("P.csv", "")
data = {pid_of(p): load_participant(p) for p in files}

print(f"Total participant files found: {len(files)}")
print(f"Participants: {sorted(data.keys())}")
print()


# --------------------------------------------------------------------------- #
# CLAIM 1: 25 participants
# --------------------------------------------------------------------------- #
n_participants = len(data)
claim1 = n_participants == 25
print(f"CLAIM 1: 25 participants")
print(f"  Found: {n_participants} participants")
print(f"  Verdict: {'CORRECT' if claim1 else 'INCORRECT'}")
print()


# --------------------------------------------------------------------------- #
# CLAIM 2: all columns 100% non-null AND timestamps are exact 5-minute grid
# --------------------------------------------------------------------------- #
print("CLAIM 2: All columns 100% non-null, timestamps strictly 5-min grid")
all_non_null = True
all_strict_5min = True
per_pid_summary = []
for pid, df in data.items():
    null_counts = df.isna().sum()
    null_total = int(null_counts.sum())
    if null_total > 0:
        all_non_null = False
    # Compute interval distribution
    dt = df["time"].diff().dt.total_seconds().dropna()
    n_5min = int((dt == 300).sum())
    n_other = int((dt != 300).sum())
    if n_other > 0:
        all_strict_5min = False
    per_pid_summary.append(
        (pid, len(df), null_total, n_5min, n_other,
         float(dt.min()) if len(dt) else float("nan"),
         float(dt.max()) if len(dt) else float("nan"))
    )

print(f"  All columns non-null across all participants: {all_non_null}")
print(f"  All intervals strictly 5-min across all participants: {all_strict_5min}")
print(f"  Verdict: {'CORRECT' if (all_non_null and all_strict_5min) else 'PARTIAL/INCORRECT'}")
print()
# Print per-participant interval audit table
print("  Per-participant audit (rows / nulls / 5-min intervals / other intervals / min_dt / max_dt):")
for row in per_pid_summary:
    print(f"    {row[0]:>10}  rows={row[1]:>6}  nulls={row[2]:>4}  "
          f"5min={row[3]:>6}  other={row[4]:>4}  "
          f"dt_min={row[5]:.0f}s  dt_max={row[6]:.0f}s")
print()


# --------------------------------------------------------------------------- #
# CLAIM 3: glucose is mg/dL, range [40, 444]
# --------------------------------------------------------------------------- #
all_glucose = pd.concat([df["glucose"] for df in data.values()], ignore_index=True)
g_min = float(all_glucose.min())
g_max = float(all_glucose.max())
g_mean = float(all_glucose.mean())
likely_mgdl = g_max > 30  # mmol/L would never exceed ~30
print(f"CLAIM 3: Glucose is mg/dL, range [40, 444]")
print(f"  Global min={g_min:.1f}, max={g_max:.1f}, mean={g_mean:.2f}")
print(f"  Unit inference: {'mg/dL (correct)' if likely_mgdl else 'mmol/L'}")
claim3 = likely_mgdl and abs(g_min - 40) < 1 and abs(g_max - 444) < 1
print(f"  Verdict: {'CORRECT' if claim3 else 'PARTIAL — check exact bounds'}")
print()


# --------------------------------------------------------------------------- #
# CLAIM 4: 40 mg/dL low-cap counts per participant
# --------------------------------------------------------------------------- #
print("CLAIM 4: Counts of glucose == 40 (low cap)")
expected = {"HUPA0002": 185, "HUPA0018": 162, "HUPA0026": 501}
low_cap_table = {}
for pid, df in data.items():
    n40 = int((df["glucose"] == 40).sum())
    low_cap_table[pid] = n40
# Print all with non-zero counts
print("  All participants with glucose == 40 (sorted desc):")
for pid, n in sorted(low_cap_table.items(), key=lambda x: -x[1]):
    if n > 0:
        print(f"    {pid}: {n}")
print()
print("  Verification of ChatGPT's specific claims:")
all_correct = True
for pid, expected_n in expected.items():
    actual_n = low_cap_table.get(pid, 0)
    ok = actual_n == expected_n
    if not ok:
        all_correct = False
    print(f"    {pid}: expected={expected_n}, actual={actual_n}  -> {'OK' if ok else 'MISMATCH'}")
print(f"  Verdict: {'CORRECT' if all_correct else 'NUMBERS DIFFER'}")
print()


# --------------------------------------------------------------------------- #
# CLAIM 4b: high-extreme values (> 400)
# --------------------------------------------------------------------------- #
print("  Additional check: counts of glucose > 400")
for pid, df in data.items():
    n_hi = int((df["glucose"] > 400).sum())
    if n_hi > 0:
        print(f"    {pid}: {n_hi}")
print()


# --------------------------------------------------------------------------- #
# CLAIM 5: Median duration ~ 13.3 days
# --------------------------------------------------------------------------- #
print("CLAIM 5: Median duration ~13.3 days; <10 days for 0006/0020/0021; >30 for 0026/0027/0028")
durations = {}
for pid, df in data.items():
    span = (df["time"].max() - df["time"].min()).total_seconds() / 86400.0
    durations[pid] = span
median_dur = float(np.median(list(durations.values())))
print(f"  Median duration: {median_dur:.2f} days")
print(f"  All durations (sorted asc):")
for pid, d in sorted(durations.items(), key=lambda x: x[1]):
    flag = ""
    if d < 10:
        flag = "  <-- SHORT (<10 days)"
    elif d > 30:
        flag = "  <-- LONG (>30 days)"
    print(f"    {pid}: {d:6.2f} days{flag}")
print()


# --------------------------------------------------------------------------- #
# CLAIM 6: No-bolus participants 0011, 0015, 0018; no-carb 0015, 0018, 0020
# --------------------------------------------------------------------------- #
print("CLAIM 6: No-bolus participants 0011/0015/0018; no-carb 0015/0018/0020")
no_bolus_observed = []
no_carb_observed = []
for pid, df in data.items():
    bolus_total = float(df["bolus_volume_delivered"].sum())
    carb_total = float(df["carb_input"].sum())
    if bolus_total == 0.0:
        no_bolus_observed.append(pid)
    if carb_total == 0.0:
        no_carb_observed.append(pid)
print(f"  Observed no-bolus: {sorted(no_bolus_observed)}")
print(f"  Observed no-carb:  {sorted(no_carb_observed)}")
expected_no_bolus = {"HUPA0011", "HUPA0015", "HUPA0018"}
expected_no_carb = {"HUPA0015", "HUPA0018", "HUPA0020"}
bolus_match = set(no_bolus_observed) == expected_no_bolus
carb_match = set(no_carb_observed) == expected_no_carb
print(f"  No-bolus match: {bolus_match}")
print(f"  No-carb match:  {carb_match}")
print()


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #
print("=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  1. 25 participants:               {'CORRECT' if claim1 else 'INCORRECT'}")
print(f"  2. All non-null + strict 5-min:    "
      f"{'CORRECT' if (all_non_null and all_strict_5min) else 'PARTIAL'}")
print(f"  3. Glucose mg/dL [40, 444]:        {'CORRECT' if claim3 else 'PARTIAL'}")
print(f"  4. Low-cap counts (0002/0018/0026): "
      f"{'CORRECT' if all_correct else 'NUMBERS DIFFER'}")
print(f"  5. Median duration ~13.3 days:     "
      f"{'CORRECT' if abs(median_dur - 13.3) < 1.0 else 'CHECK NUMBERS'}")
print(f"  6. No-bolus / No-carb sets:        "
      f"{'CORRECT' if (bolus_match and carb_match) else 'PARTIAL'}")
print()
print("Done.")
