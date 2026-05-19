"""Check UOM data: can each HUPA column be reconstructed?
Inspects sample patient files for content, time resolution, missingness.
"""
import pandas as pd
import os

BASE = r'E:\claude-co-work\data\raw'
OUT = r'E:\claude-co-work\outputs\uom_to_hupa_feasibility.txt'
os.makedirs(os.path.dirname(OUT), exist_ok=True)
log = open(OUT, 'w', encoding='utf-8')

def w(s=''):
    log.write(str(s) + '\n')

# ---------- 1. Glucose: 2301 (5-min) and 2302 (15-min) ----------
w('=' * 80)
w('1. GLUCOSE (target column for HUPA)')
w('=' * 80)
for pid in ['2301', '2302']:
    f = os.path.join(BASE, 'Glucose Data', f'UoMGlucose{pid}.csv')
    df = pd.read_csv(f, parse_dates=['bg_ts'], dayfirst=True)
    w(f'\nPatient {pid}: rows={len(df)}, cols={list(df.columns)}')
    w(f'  dtypes: {df.dtypes.to_dict()}')
    w(f'  first 3 timestamps: {df["bg_ts"].iloc[:3].tolist()}')
    w(f'  value range: {df["value"].min()}-{df["value"].max()} mmol/L')
    # sampling interval
    deltas = df['bg_ts'].diff().dt.total_seconds().value_counts().head(3)
    w(f'  top 3 inter-record intervals (sec): {deltas.to_dict()}')

# ---------- 2. Activity: calories + steps + HR? ----------
w('\n' + '=' * 80)
w('2. ACTIVITY (source for calories, steps; check HR)')
w('=' * 80)
f = os.path.join(BASE, 'Activity Data', 'UoMActivity2301.csv')
df = pd.read_csv(f, parse_dates=['activity_ts'], dayfirst=True)
w(f'\nP2301 Activity: rows={len(df)}, cols={list(df.columns)}')
w(f'  dtypes: {df.dtypes.to_dict()}')
w(f'  time range: {df["activity_ts"].min()} to {df["activity_ts"].max()}')
deltas = df['activity_ts'].diff().dt.total_seconds().value_counts().head(5)
w(f'  top 5 inter-record intervals (sec): {deltas.to_dict()}')
w(f'  active_Kcal stats: mean={df["active_Kcal"].mean():.2f}, sum={df["active_Kcal"].sum():.0f}, %zero={100*(df["active_Kcal"]==0).mean():.1f}')
w(f'  step_count stats: mean={df["step_count"].mean():.2f}, sum={df["step_count"].sum():.0f}, %zero={100*(df["step_count"]==0).mean():.1f}')
w(f'  activity_type counts: {df["activity_type"].value_counts().to_dict()}')
w(f'  intensity counts: {df["intensity"].value_counts().to_dict()}')
w(f'  NO heart_rate column? {"heart_rate" not in df.columns}')

# ---------- 3. Sleep: does it contain HR all-day or only sleeping? ----------
w('\n' + '=' * 80)
w('3. SLEEP DATA (the only file containing heart_rate)')
w('=' * 80)
sleep_files = sorted(os.listdir(os.path.join(BASE, 'Sleep Data')))
w(f'Sleep folder has {len(sleep_files)} files. Naming patterns: {sleep_files[:6]}')

# UoMIDsleeptime files are sleep summaries (start/end)
# Look for the actual per-record sleep file. From README: UoMsleepID.csv has heart_rate continuous
# but actual files are UoMIDsleeptime.csv format. Verify both.
for fname in sleep_files[:10]:
    if 'sleeptime' not in fname.lower():
        continue
    fp = os.path.join(BASE, 'Sleep Data', fname)
    try:
        df = pd.read_csv(fp, nrows=3)
        w(f'\n{fname}: cols={list(df.columns)}')
        w(f'  first row: {df.iloc[0].to_dict()}')
        break
    except Exception as e:
        w(f'  {fname}: error {e}')

# Try to find ANY file that has heart_rate column
w('\nSearching for any sleep file with heart_rate column...')
for fname in sleep_files:
    fp = os.path.join(BASE, 'Sleep Data', fname)
    try:
        head = pd.read_csv(fp, nrows=1)
        if 'heart_rate' in head.columns:
            df = pd.read_csv(fp, parse_dates=[c for c in head.columns if 'ts' in c.lower()], dayfirst=True)
            w(f'  FOUND: {fname}, rows={len(df)}, cols={list(df.columns)}')
            ts_col = [c for c in df.columns if 'ts' in c.lower()][0]
            w(f'    time range: {df[ts_col].min()} to {df[ts_col].max()}')
            # is HR continuous 24h or only sleep window?
            df['hour'] = pd.to_datetime(df[ts_col]).dt.hour
            w(f'    HR records by hour: {df["hour"].value_counts().sort_index().to_dict()}')
            w(f'    HR non-null: {df["heart_rate"].notna().sum()} / {len(df)}')
            w(f'    HR stats: min={df["heart_rate"].min()}, max={df["heart_rate"].max()}, mean={df["heart_rate"].mean():.1f}')
            break
    except Exception as e:
        pass
else:
    w('  No sleep file with heart_rate column found.')

# ---------- 4. Basal (pump vs MDI distinction) ----------
w('\n' + '=' * 80)
w('4. BASAL INSULIN')
w('=' * 80)
for pid in ['2301', '2302']:
    f = os.path.join(BASE, 'Insulin Data', 'Basal Data', f'UoMBasal{pid}.csv')
    if not os.path.exists(f):
        w(f'  P{pid}: file not found')
        continue
    df = pd.read_csv(f, encoding='utf-8-sig')
    # Drop unnamed columns from BOM/trailing
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    w(f'\nP{pid} Basal: rows={len(df)}, cols={list(df.columns)}')
    if 'basal_ts' in df.columns:
        df['basal_ts'] = pd.to_datetime(df['basal_ts'], dayfirst=True, errors='coerce')
        w(f'  time range: {df["basal_ts"].min()} to {df["basal_ts"].max()}')
        deltas = df['basal_ts'].diff().dt.total_seconds().value_counts().head(3)
        w(f'  top 3 intervals (sec): {deltas.to_dict()}')
    if 'insulin_kind' in df.columns:
        w(f'  insulin_kind counts: {df["insulin_kind"].value_counts().to_dict()}')
    if 'basal_dose' in df.columns:
        w(f'  basal_dose stats: min={df["basal_dose"].min()}, max={df["basal_dose"].max()}, mean={df["basal_dose"].mean():.3f}')

# ---------- 5. Bolus ----------
w('\n' + '=' * 80)
w('5. BOLUS INSULIN')
w('=' * 80)
for pid in ['2301', '2302']:
    f = os.path.join(BASE, 'Insulin Data', 'Bolus Data', f'UoMBolus{pid}.csv')
    if not os.path.exists(f):
        w(f'  P{pid}: file not found')
        continue
    df = pd.read_csv(f, encoding='utf-8-sig')
    df = df.loc[:, ~df.columns.str.startswith('Unnamed')]
    w(f'\nP{pid} Bolus: rows={len(df)}, cols={list(df.columns)}')
    if 'bolus_ts' in df.columns:
        df['bolus_ts'] = pd.to_datetime(df['bolus_ts'], dayfirst=True, errors='coerce')
        w(f'  time range: {df["bolus_ts"].min()} to {df["bolus_ts"].max()}')
    if 'bolus_dose' in df.columns:
        w(f'  bolus_dose: min={df["bolus_dose"].min()}, max={df["bolus_dose"].max()}, mean={df["bolus_dose"].mean():.2f}')

# ---------- 6. Nutrition ----------
w('\n' + '=' * 80)
w('6. NUTRITION (source for carb_input)')
w('=' * 80)
f = os.path.join(BASE, 'Nutrition Data', 'UoMNutrition2301.csv')
df = pd.read_csv(f)
w(f'\nP2301 Nutrition: rows={len(df)}, cols={list(df.columns)}')
w(f'  carbs_g stats: min={df["carbs_g"].min()}, max={df["carbs_g"].max()}, mean={df["carbs_g"].mean():.1f}')
w(f'  has prot_g, fat_g, fibre_g? {all(c in df.columns for c in ["prot_g","fat_g","fibre_g"])}')
w(f'  meal_type counts: {df["meal_type"].value_counts().to_dict() if "meal_type" in df.columns else "N/A"}')

log.close()
print(f'Done — see {OUT}')
