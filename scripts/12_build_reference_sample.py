"""Stratified random reference sample from the frozen maps. usage: 12_build_reference_sample.py FIELDS OUTPUTS
Strata by mapped intensity within each water year: fallow (>=0.75), not fallow (<=0.25), uncertain (between).
The strata, predictions and field IDs are written only to the key file, so labelling can be blind."""
import sys, warnings
import numpy as np, pandas as pd, geopandas as gpd
warnings.filterwarnings('ignore')
fg, out = sys.argv[1:3]; N_PER=50; YEARS=[2020,2022,2023]; rng=np.random.default_rng(20261109)
dev=pd.read_csv(f'{out}/after180/fallow_intensity_wy2018_2022.csv.gz').rename(columns={'RF optical':'p'})[['field_id','wy','p']]
h=pd.read_csv(f'{out}/holdout_2023_field_results.csv.gz').rename(columns={'frozen':'p'}).assign(wy=2023)[['field_id','wy','p']]
A=pd.concat([dev,h]); A=A[A.wy.isin(YEARS)].copy()
A['stratum']=np.where(A.p>=.75,'mapped_fallow',np.where(A.p<=.25,'mapped_not_fallow','mapped_uncertain'))
g=gpd.read_file(fg); g=g[g.label!='non_crop'].set_index('field_id'); ll=g.to_crs(4326).centroid; c=g.centroid
rows=[]
for (y,s),q in A.groupby(['wy','stratum']):
    n=min(N_PER,len(q)); pick=rng.choice(q.index.values,n,replace=False); t=A.loc[pick].copy() if A.index.is_unique else None; t=q.loc[pick].copy()
    t['stratum_N']=len(q); t['stratum_n']=n; t['weight']=len(q)/n; rows.append(t)
S=pd.concat(rows,ignore_index=True); S=S.sample(frac=1,random_state=20261109).reset_index(drop=True)
S['sample_id']=[f'S{i+1:03d}' for i in range(len(S))]; S['lon']=S.field_id.map(ll.x).round(6); S['lat']=S.field_id.map(ll.y).round(6); S['acres']=S.field_id.map(g.area_ac).round(1); S['x']=S.field_id.map(c.x); S['y']=S.field_id.map(c.y)
S[['sample_id','field_id','wy','stratum','stratum_N','stratum_n','weight','p','lat','lon','acres','x','y']].to_csv(f'{out}/reference_sample/key/DO_NOT_OPEN_key.csv',index=False)
print(S.groupby(['wy','stratum']).agg(n=('p','size'),N=('stratum_N','first'),weight=('weight','first')).round(2).to_string()); print('total',len(S),'distinct fields',S.field_id.nunique())
