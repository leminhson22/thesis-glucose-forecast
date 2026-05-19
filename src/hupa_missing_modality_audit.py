"""Audit missing-modality patterns in HUPA: which patients, which modalities, how much."""
import pandas as pd
import glob, os

FILES = sorted(glob.glob(r'E:\claude-co-work\data\data_hupa\Preprocessed\HUPA*.xlsx'))
OUT = r'E:\claude-co-work\outputs\hupa_missing_modality_audit.txt'
os.makedirs(os.path.dirname(OUT), exist_ok=True)
log = open(OUT, 'w', encoding='utf-8')
def w(s=''): log.write(str(s) + '\n')

# Load patient characteristics
char = pd.read_excel(r'E:\claude-co-work\data\data_hupa\patient_data_characteristic.xlsx')
char['Patient ID'] = char['Patient ID'].str.strip()

rows = []
for f in FILES:
    pid = os.path.basename(f).replace('.xlsx', '')
    df = pd.read_excel(f)
    n = len(df)
    rows.append({
        'pid': pid,
        'n': n,
        'days': round(n*5/1440, 1),
        'basal_nonzero': int((df['basal_rate'] > 0).sum()),
        'basal_nz_pct': 100*(df['basal_rate'] > 0).mean(),
        'bolus_events': int((df['bolus_volume_delivered'] > 0).sum()),
        'bolus_per_day': (df['bolus_volume_delivered'] > 0).sum() / (n*5/1440),
        'carb_events': int((df['carb_input'] > 0).sum()),
        'carb_per_day': (df['carb_input'] > 0).sum() / (n*5/1440),
    })
df_audit = pd.DataFrame(rows)
df_audit['treatment'] = df_audit['pid'].map(char.set_index('Patient ID')['Treatment (CSII/MDI)'])

w('=' * 110)
w('HUPA missing-modality audit')
w('=' * 110)
w(f'{"PID":<12} {"Treat":>5} {"n":>7} {"days":>5} {"basal_nz":>9} {"basal%":>7} {"bolus":>6} {"bol/d":>7} {"carb":>5} {"crb/d":>7}')
w('-'*110)
for _, r in df_audit.iterrows():
    w(f'{r["pid"]:<12} {str(r["treatment"]):>5} {r["n"]:>7} {r["days"]:>5} {r["basal_nonzero"]:>9} {r["basal_nz_pct"]:>6.1f}% {r["bolus_events"]:>6} {r["bolus_per_day"]:>7.2f} {r["carb_events"]:>5} {r["carb_per_day"]:>7.2f}')

w('\n' + '=' * 110)
w('Patients with FULLY MISSING modalities (all-zero column)')
w('=' * 110)

total_rows = df_audit['n'].sum()
missing_basal = df_audit[df_audit['basal_nonzero'] == 0]
missing_bolus = df_audit[df_audit['bolus_events'] == 0]
missing_carb  = df_audit[df_audit['carb_events'] == 0]

w(f'\nNo BASAL recorded: {len(missing_basal)} patients')
for _, r in missing_basal.iterrows():
    w(f'  {r["pid"]} ({r["treatment"]}): {r["n"]} rows ({100*r["n"]/total_rows:.2f}% of total)')
w(f'  Total rows affected: {missing_basal["n"].sum()} ({100*missing_basal["n"].sum()/total_rows:.2f}%)')

w(f'\nNo BOLUS recorded: {len(missing_bolus)} patients')
for _, r in missing_bolus.iterrows():
    w(f'  {r["pid"]} ({r["treatment"]}): {r["n"]} rows ({100*r["n"]/total_rows:.2f}% of total)')
w(f'  Total rows affected: {missing_bolus["n"].sum()} ({100*missing_bolus["n"].sum()/total_rows:.2f}%)')

w(f'\nNo CARB recorded: {len(missing_carb)} patients')
for _, r in missing_carb.iterrows():
    w(f'  {r["pid"]} ({r["treatment"]}): {r["n"]} rows ({100*r["n"]/total_rows:.2f}% of total)')
w(f'  Total rows affected: {missing_carb["n"].sum()} ({100*missing_carb["n"].sum()/total_rows:.2f}%)')

# Patients missing 2+ modalities
missing_2plus = df_audit[
    ((df_audit['basal_nonzero']==0).astype(int) +
     (df_audit['bolus_events']==0).astype(int) +
     (df_audit['carb_events']==0).astype(int)) >= 2
]
w(f'\nPatients missing 2+ modalities ({len(missing_2plus)}):')
for _, r in missing_2plus.iterrows():
    flags = []
    if r['basal_nonzero']==0: flags.append('basal')
    if r['bolus_events']==0: flags.append('bolus')
    if r['carb_events']==0: flags.append('carb')
    w(f'  {r["pid"]} ({r["treatment"]}): missing {flags}, {r["n"]} rows ({r["days"]} days)')

# Union
union = df_audit[
    (df_audit['basal_nonzero']==0) |
    (df_audit['bolus_events']==0) |
    (df_audit['carb_events']==0)
]
w(f'\nUnion (patients missing ANY of basal/bolus/carb): {len(union)} patients')
w(f'  Total rows: {union["n"].sum()} ({100*union["n"].sum()/total_rows:.2f}% of dataset)')

# Treatment vs missing pattern
w(f'\n' + '=' * 110)
w('Treatment cohort breakdown:')
w('=' * 110)
csii = df_audit[df_audit['treatment'] == 'CSII']
mdi  = df_audit[df_audit['treatment'] == 'MDI']
w(f'CSII patients: {len(csii)}, total rows {csii["n"].sum()}')
w(f'  Missing basal: {len(csii[csii["basal_nonzero"]==0])} patients (CSII SHOULD have basal!)')
w(f'  Missing bolus: {len(csii[csii["bolus_events"]==0])}')
w(f'  Missing carb:  {len(csii[csii["carb_events"]==0])}')
w(f'MDI patients: {len(mdi)}, total rows {mdi["n"].sum()}')
w(f'  Missing basal: {len(mdi[mdi["basal_nonzero"]==0])} patients (MDI may not record long-acting)')
w(f'  Missing bolus: {len(mdi[mdi["bolus_events"]==0])}')
w(f'  Missing carb:  {len(mdi[mdi["carb_events"]==0])}')

log.close()
print(f'Done — {OUT}')
