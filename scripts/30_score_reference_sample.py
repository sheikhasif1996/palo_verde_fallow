"""Score the frozen annual fallow maps against the blind manual reference labels.
usage: 30_score_reference_sample.py REF_DIR OUT_DIR [LABEL_FILE]
Design: stratified random sample by water year x mapped-intensity stratum (12_build_reference_sample.py).
Strata are not the map classes (the uncertain stratum holds both), so accuracy and area use the
stratified estimators of Stehman (2014): each sample unit carries weight N_h / n_h, with n_h the units
labelled usably in that stratum. CIs come from a stratified bootstrap (resampling within strata).
Reference definitions: A = 'Fallow' only; B = 'Fallow' + 'Part-year fallow'; C = B with the 6-month rule enforced
(Part-year labels with a recorded bare span under 6 months become Cropped).
'Cannot tell', 'Not cropland' and blank labels are excluded; they are reported by stratum."""
import sys, os
import numpy as np, pandas as pd
ref, out = sys.argv[1:3]; LF = sys.argv[3] if len(sys.argv) > 3 else 'reference_labels.xlsx'; os.makedirs(out, exist_ok=True)
K = pd.read_csv(f'{ref}/key/DO_NOT_OPEN_key.csv')
L = pd.read_excel(f'{ref}/{LF}', sheet_name='Labels')
L = L.rename(columns={'Sample ID': 'sample_id', 'Label': 'label', 'Confidence': 'conf', 'Longest bare stretch (months)': 'span'})[['sample_id', 'label', 'conf', 'span']]
D = K.merge(L, on='sample_id', how='left', validate='1:1'); assert len(D) == len(K) == 450
D['label'] = D.label.fillna('Blank'); D['label_raw'] = D.label
D['label_C'] = np.where((D.label == 'Part-year fallow') & (D.span < 6), 'Cropped', D.label); D['map_fallow'] = (D.p >= 0.5).astype(int); D['h'] = D.wy.astype(str) + '_' + D.stratum
USE = ['Fallow', 'Part-year fallow', 'Cropped']
DEF = {'A_fallow_only': ('label', ['Fallow']), 'B_fallow_plus_part_year': ('label', ['Fallow', 'Part-year fallow']), 'C_part_year_6mo_rule': ('label_C', ['Fallow', 'Part-year fallow'])}
NH = D.groupby('h').stratum_N.first()

def est(d):
    """Stehman (2014) stratified estimators from usable units d (needs columns h, m, r)."""
    g = d.groupby('h'); n = g.size(); N = NH.reindex(n.index)
    mean = lambda v: (d.assign(v=v).groupby('h').v.mean() * N).sum()
    T = N.sum(); tp = mean(d.m * d.r); mp = mean(d.m); rp = mean(d.r); agree = mean((d.m == d.r).astype(float))
    ua = tp / mp if mp else np.nan; pa = tp / rp if rp else np.nan
    ua0 = mean((1 - d.m) * (1 - d.r)) / mean(1 - d.m); pa0 = mean((1 - d.m) * (1 - d.r)) / mean(1 - d.r)
    return dict(OA=agree / T, UA_fallow=ua, PA_fallow=pa, F1=2 * ua * pa / (ua + pa), UA_notfallow=ua0, PA_notfallow=pa0,
                ref_fallow_share=rp / T, map_fallow_share=mp / T, n=len(d), N=T)

def boot(d, B=2000, seed=7):
    rng = np.random.default_rng(seed); R = []
    idx = {k: v.index.values for k, v in d.groupby('h')}
    for _ in range(B):
        s = np.concatenate([rng.choice(v, len(v), replace=True) for v in idx.values()])
        R.append(est(d.loc[s].reset_index(drop=True)))
    return pd.DataFrame(R)

rows = []
for name, (col, pos) in DEF.items():
    for subset in ['all usable', 'excluding Low confidence']:
        for yr in ['pooled', 2020, 2022, 2023]:
            d = D[D[col].isin(USE)].copy()
            if subset != 'all usable': d = d[d.conf != 'Low']
            if yr != 'pooled': d = d[d.wy == yr]
            d['m'] = d.map_fallow; d['r'] = d[col].isin(pos).astype(int); d = d.reset_index(drop=True)
            e = est(d); b = boot(d)
            for k in ['OA', 'UA_fallow', 'PA_fallow', 'F1', 'ref_fallow_share', 'map_fallow_share']:
                e[k + '_lo'], e[k + '_hi'] = np.nanpercentile(b[k], [2.5, 97.5])
            rows.append(dict(definition=name, subset=subset, wy=yr, **e))
    # worst case: every 'Cannot tell' counted as a map error (reference set opposite to the map)
    d = D[D[col].isin(USE + ['Cannot tell'])].copy(); d['m'] = d.map_fallow
    d['r'] = np.where(d[col] == 'Cannot tell', 1 - d.m, d[col].isin(pos).astype(int)); d = d.reset_index(drop=True)
    e = est(d); b = boot(d)
    for k in ['OA', 'UA_fallow', 'PA_fallow', 'F1', 'ref_fallow_share', 'map_fallow_share']:
        e[k + '_lo'], e[k + '_hi'] = np.nanpercentile(b[k], [2.5, 97.5])
    rows.append(dict(definition=name, subset='worst case: Cannot tell = error', wy='pooled', **e))
R = pd.DataFrame(rows); R.to_csv(f'{out}/reference_sample_accuracy.csv', index=False)

# what was excluded, and the raw (unweighted) confusion
ex = pd.crosstab([D.wy, D.stratum], D.label); ex.to_csv(f'{out}/labels_by_stratum.csv')
D[['sample_id', 'wy', 'stratum', 'p', 'map_fallow', 'label', 'label_C', 'conf', 'span']].to_csv(f'{out}/labels_used.csv', index=False)
cm = pd.crosstab(D.label, D.map_fallow.map({1: 'map fallow', 0: 'map not fallow'})); cm.to_csv(f'{out}/raw_confusion.csv')
cc = pd.crosstab([D.label, D.conf.fillna('none')], D.map_fallow.map({1: 'map fallow', 0: 'map not fallow'})); cc.to_csv(f'{out}/raw_confusion_by_confidence.csv')
pd.set_option('display.width', 200)
print('Labels by stratum:\n', ex.to_string(), '\n')
print('Raw confusion (sample counts, unweighted):\n', cm.to_string(), '\n')
print('By confidence:\n', cc.to_string(), '\n')
show = ['definition', 'subset', 'wy', 'n', 'OA', 'OA_lo', 'OA_hi', 'UA_fallow', 'UA_fallow_lo', 'UA_fallow_hi', 'PA_fallow', 'PA_fallow_lo', 'PA_fallow_hi', 'F1', 'F1_lo', 'F1_hi', 'ref_fallow_share', 'map_fallow_share']
print(R[show].round(3).to_string(index=False))
# part-year fields: map call and mapped intensity by recorded bare span
P = D[D.label == 'Part-year fallow'].copy(); P['span_bin'] = pd.cut(P.span, [0, 5, 7, 9, 12], labels=['under 6', '6-7', '8-9', '10-12'])
sp = P.groupby('span_bin', observed=False).agg(n=('p', 'size'), mapped_fallow=('map_fallow', 'sum'), median_intensity=('p', 'median')).assign(share_mapped=lambda x: x.mapped_fallow / x.n)
sp.to_csv(f'{out}/part_year_by_span.csv'); print('\nPart-year fallow by recorded bare span:\n', sp.round(2).to_string())
print('Spearman span vs mapped intensity: %.2f' % P[['span', 'p']].corr('spearman').iloc[0, 1])
