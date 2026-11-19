"""Shared loading for per-field time series. Only scenes listed in scene_manifest.csv are used."""
import os
import numpy as np, pandas as pd
MINV = 0.8
def _load(parts, manifest, tag, start, end):
    m = pd.read_csv(manifest); m = m[m.sensor==tag]
    t = pd.to_datetime(m.datetime, format='ISO8601', utc=True).dt.tz_localize(None); m = m[(t>=start)&(t<=pd.Timestamp(end)+pd.Timedelta('1D'))]
    fs = [f'{parts}/{tag}_{s}.csv.gz' for s in m.scene]; have = [f for f in fs if os.path.exists(f)]
    if len(have) < len(fs): print(f'WARNING {tag}: {len(fs)-len(have)} manifest scenes have no part file', flush=True)
    d = pd.concat([pd.read_csv(f) for f in have], ignore_index=True); print(tag, 'scenes', len(have), 'rows', len(d), flush=True)
    d['date'] = pd.to_datetime(d.datetime, format='ISO8601', utc=True).dt.tz_localize(None).dt.normalize()
    return d[d.n_valid/d.n_total >= MINV].copy()
def _wmean(d, keys, cols):
    x = d[keys].copy(); w = d.n_valid.values.astype(float)
    for c in cols: x[c] = d[c].values.astype(float)*w
    x['w'] = w; g = x.groupby(keys, as_index=False).sum()
    for c in cols: g[c] = (g[c]/g.w).astype(float)
    return g.drop(columns='w')
def load_series(parts, manifest, start, end):
    s2 = _wmean(_load(parts,manifest,'s2',start,end), ['field_id','date'], ['B04','B08','B11','B12'])
    s2['ndvi']=(s2.B08-s2.B04)/(s2.B08+s2.B04); s2['ndmi']=(s2.B08-s2.B11)/(s2.B08+s2.B11); s2['str']=(1-s2.B12)**2/(2*s2.B12)
    s1 = _load(parts,manifest,'s1',start,end)
    s1['vv_db']=10*np.log10(s1.vv.astype(float)); s1['vh_db']=10*np.log10(s1.vh.astype(float)); s1['vvvh']=s1.vv_db-s1.vh_db
    for c in ['vv_db','vh_db']: s1[c+'_rel']=(s1[c]-s1.groupby('rel_orbit')[c].transform('mean')).astype(float)
    ls = _wmean(_load(parts,manifest,'ls',start,end), ['field_id','date'], ['lst_c'])
    ls['lst_anom']=(ls.lst_c-ls.groupby('date').lst_c.transform('median')).astype(float)
    return s2, s1, ls
def window_features(s2, s1, ls, t, days):
    """Features for each field from the trailing window (t-days, t]."""
    lo = t - pd.Timedelta(days=days)
    a = s2[(s2.date>lo)&(s2.date<=t)]; b = s1[(s1.date>lo)&(s1.date<=t)]; c = ls[(ls.date>lo)&(ls.date<=t)]
    O = a.groupby('field_id').agg(n_s2=('ndvi','size'), ndvi_mean=('ndvi','mean'), ndvi_max=('ndvi','max'), ndvi_min=('ndvi','min'), ndvi_std=('ndvi','std'), bare_frac=('ndvi',lambda x:(x<.2).mean()), ndmi_mean=('ndmi','mean'), ndmi_max=('ndmi','max'), str_mean=('str','mean'), str_max=('str','max'), b11_mean=('B11','mean'), b12_mean=('B12','mean'))
    S = b.groupby('field_id').agg(n_s1=('vv_db','size'), vv_mean=('vv_db_rel','mean'), vh_mean=('vh_db_rel','mean'), vv_std=('vv_db_rel','std'), vh_std=('vh_db_rel','std'), vv_min=('vv_db_rel','min'), vv_max=('vv_db_rel','max'), vh_min=('vh_db_rel','min'), vh_max=('vh_db_rel','max'), vvvh_mean=('vvvh','mean'))
    S['vv_range']=S.vv_max-S.vv_min; S['vh_range']=S.vh_max-S.vh_min
    L = c.groupby('field_id').agg(n_ls=('lst_anom','size'), lst_anom_mean=('lst_anom','mean'), lst_anom_max=('lst_anom','max'))
    return O.join(S,how='inner').join(L,how='left')
OPT = ['ndvi_mean','ndvi_max','ndvi_min','ndvi_std','bare_frac','ndmi_mean','ndmi_max','str_mean','str_max','b11_mean','b12_mean']
SAR = ['vv_mean','vh_mean','vv_std','vh_std','vv_range','vh_range','vvvh_mean']
THM = ['lst_anom_mean','lst_anom_max']
