"""Does window overlap between water years inflate fallow persistence?
Recomputes annual fallow status and year-to-year transitions from subsets of the frozen monthly states whose
imagery windows (90 d before to 180 d after each month) overlap the neighbouring water years less or not at all.
No retuning: the frozen monthly states are reused as produced. usage: 33_persistence_sensitivity.py OUTPUTS"""
import sys
import numpy as np, pandas as pd
out = sys.argv[1]; YEARS = list(range(2018, 2025)); rng = np.random.default_rng(7); N = 2000
M = pd.read_csv(f'{out}/maps/monthly_state.csv.gz', parse_dates=['date']); M = M[M.wy.isin(YEARS)]
VARIANTS = {  # name: (months used, minimum valid months, imagery window span, shared imagery with the next year)
    'All months (frozen map)': (list(range(1, 13)), 8, '3 Jul (y-1) to 29 Mar (y+1)', 'about 9 months'),
    'October to March': ([10, 11, 12, 1, 2, 3], 4, '3 Jul (y-1) to 28 Aug (y)', 'about 2 months'),
    'January to March': ([1, 2, 3], 2, '3 Oct (y-1) to 28 Aug (y)', 'none'),
}
def trans(X):
    a, b = X[:, :-1].ravel(), X[:, 1:].ravel()
    return ((a == 1) & (b == 1)).sum(), ((a == 1) & (b == 0)).sum(), ((a == 0) & (b == 1)).sum(), ((a == 0) & (b == 0)).sum()
full_map = None; rows = []
for name, (mons, minv, span, shared) in VARIANTS.items():
    d = M[M.date.dt.month.isin(mons)].dropna(subset=['state'])
    I = d.groupby(['field_id', 'wy']).state.agg(['size', 'mean']).reset_index(); I = I[I['size'] >= minv]
    I['fallow'] = (I['mean'] >= .5).astype(int)
    W = I.pivot(index='field_id', columns='wy', values='fallow').reindex(columns=YEARS).dropna(); X = W.values.astype(int)
    if full_map is None: full_map = W
    common = W.index.intersection(full_map.index)
    agree = (W.loc[common].values == full_map.loc[common].values).mean()
    ff, fc, cf, cc = trans(X); ps, pe = ff / (ff + fc), cf / (cf + cc)
    nul = []
    for _ in range(N):
        Xp = np.column_stack([rng.permutation(X[:, j]) for j in range(X.shape[1])]); a, b, c, e = trans(Xp); nul.append(a / (a + b))
    # bootstrap over fields for a CI on P(stay fallow)
    bs = []
    for _ in range(N):
        Xb = X[rng.integers(0, len(X), len(X))]; a, b, c, e = trans(Xb); bs.append(a / (a + b))
    rows.append(dict(variant=name, months='-'.join(map(str, mons)), min_valid=minv, window_span=span, shared_with_next_year=shared,
                     fields=len(X), fallow_field_years=int(X.sum()), agreement_with_frozen=round(agree, 3),
                     F_to_F=ff, F_to_C=fc, C_to_F=cf, C_to_C=cc, P_stay=round(ps, 3), P_stay_lo=round(np.percentile(bs, 2.5), 3),
                     P_stay_hi=round(np.percentile(bs, 97.5), 3), P_enter=round(pe, 3), null_stay=round(np.mean(nul), 3)))
R = pd.DataFrame(rows); R.to_csv(f'{out}/analysis/persistence_sensitivity.csv', index=False); print(R.to_string(index=False))
