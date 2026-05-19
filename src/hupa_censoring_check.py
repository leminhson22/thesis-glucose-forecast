import pandas as pd
import glob, os

files = sorted(glob.glob(r'E:\claude-co-work\data\data_hupa\Preprocessed\HUPA*.xlsx'))
out = open(r'E:\claude-co-work\outputs\hupa_censoring_check.txt', 'w', encoding='utf-8')

header = f'{"Patient":<12} {"N":>7} {"days":>5} {"g=40":>6} {"%=40":>7} {"g>=400":>7} {"%>=400":>8} {"hypo<70":>8} {"%hypo":>7} {"hyper>180":>10} {"%hyper":>8} {"mean":>7}'
out.write(header + '\n')
out.write('-' * len(header) + '\n')

total_n = total_40 = total_400 = total_hypo = total_hyper = 0
rows = []
out.write(f'Found {len(files)} files\n')
for f in files:
    pid = os.path.basename(f).replace('.xlsx', '')
    df = pd.read_excel(f)
    n = len(df)
    g40 = int((df['glucose'] == 40).sum())
    g400 = int((df['glucose'] >= 400).sum())
    hypo = int((df['glucose'] < 70).sum())
    hyper = int((df['glucose'] > 180).sum())
    days = round(n * 5 / 1440, 1)
    rows.append((pid, n, days, g40, g400, hypo, hyper, float(df['glucose'].mean())))
    total_n += n
    total_40 += g40
    total_400 += g400
    total_hypo += hypo
    total_hyper += hyper
    out.write(f'{pid:<12} {n:>7} {days:>5} {g40:>6} {100*g40/n:>6.2f}% {g400:>7} {100*g400/n:>7.2f}% {hypo:>8} {100*hypo/n:>6.2f}% {hyper:>10} {100*hyper/n:>7.2f}% {float(df["glucose"].mean()):>7.1f}\n')

out.write('-' * len(header) + '\n')
if total_n > 0:
    out.write(f'TOTAL        {total_n:>7}       {total_40:>6} {100*total_40/total_n:>6.2f}% {total_400:>7} {100*total_400/total_n:>7.2f}% {total_hypo:>8} {100*total_hypo/total_n:>6.2f}% {total_hyper:>10} {100*total_hyper/total_n:>7.2f}%\n')

out.write('\n--- Patient share of total dataset rows ---\n')
for pid, n, days, *_ in sorted(rows, key=lambda r: -r[1]):
    out.write(f'{pid:<12} {n:>7} rows ({100*n/total_n:5.2f}%)  {days} days\n')

out.close()
print('Done — written to outputs/hupa_censoring_check.txt')
