"""Final field-level fallow maps for water years 2018-2024 using the frozen method in outputs/frozen_method.json.
usage: 17_produce_maps.py FIELDS PARTS MANIFEST OUTPUTS
Each field is predicted by a model that excluded its 3 km block, matching the validated setup.
Writes monthly states, field-year intensity, and a GeoPackage of one row per field."""
import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
import numpy as np, pandas as pd, geopandas as gpd
from sklearn.ensemble import RandomForestClassifier as RF
import tslib
warnings.filterwarnings('ignore')
fg, parts, mf, out = sys.argv[1:5]
C=json.load(open(f'{out}/frozen_method.json')); AFTER,BEFORE=C['window_days_after'],C['window_days_before']; COLS=C['features']; assert COLS==tslib.OPT
YEARS=list(range(2018,2025)); od=f'{out}/maps'
g=gpd.read_file(fg); g=g[g.label!='non_crop'].copy(); c=g.centroid
g['block']=(np.floor(c.x/3000).astype(int).astype(str)+'_'+np.floor(c.y/3000).astype(int).astype(str))
ub=g.block.drop_duplicates().sample(frac=1,random_state=7).values; g['fold']=g.block.map({b:i%5 for i,b in enumerate(ub)})
g=g.set_index('field_id'); crop21=g.SYMB_CLASS.isin(list('PFTGCD'))
s2,s1,ls=tslib.load_series(parts,mf,'2017-08-01','2025-03-31')
print('imagery through',s2.date.max().date(),flush=True)
def feats(t):
    X=tslib.window_features(s2,s1,ls,t+pd.Timedelta(days=AFTER),BEFORE+AFTER); return X[(X.n_s2>=6)&(X.n_s1>=4)].copy()
tr=[]
for k,t in {'jan':pd.Timestamp('2021-01-01'),'dec':pd.Timestamp('2021-12-31')}.items():
    X=feats(t); f=g.loc[X.index,f'f_{k}']; X['y']=np.where(f>=.7,1,np.where((f<=.05)&crop21.loc[X.index],0,-1)); X['fold']=g.loc[X.index,'fold'].values; tr.append(X[X.y>=0])
tr=pd.concat(tr); print('training',tr.y.value_counts().to_dict(),flush=True)
M=[RF(n_estimators=500,min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=-1,random_state=7).fit(tr.loc[tr.fold!=k,COLS],tr.loc[tr.fold!=k,'y']) for k in range(5)]
rows=[]
for t in pd.date_range('2017-10-01','2024-09-01',freq='MS'):
    X=feats(t); X['date']=t; X['fold']=g.loc[X.index,'fold'].values; X=X.reset_index(); p=np.full(len(X),np.nan)
    for k in range(5):
        b=(X.fold==k).values
        if b.sum(): p[b]=M[k].predict(X.loc[b,COLS])
    X['state']=p; X['ndvi_max']=X.ndvi_max; rows.append(X[['field_id','date','state','n_s2','ndvi_max']])
D=pd.concat(rows,ignore_index=True); D['wy']=D.date.dt.year+(D.date.dt.month>=10).astype(int); D.to_csv(f'{od}/monthly_state.csv.gz',index=False); print('field-months',len(D),'months',D.date.nunique(),'fields/month median',int(D.groupby('date').size().median()),flush=True)
I=D.groupby(['field_id','wy']).agg(months=('state','size'),intensity=('state','mean')).reset_index(); I=I[I.months>=8].copy(); I['fallow']=(I.intensity>=.5).astype(int); I['area_ac']=I.field_id.map(g.area_ac)
I.to_csv(f'{od}/field_year.csv.gz',index=False)
w=I.pivot(index='field_id',columns='wy',values='fallow').rename(columns=lambda y:f'fallow_{y}'); v=I.pivot(index='field_id',columns='wy',values='intensity').rename(columns=lambda y:f'intensity_{y}').round(3)
G=g[['area_ac','geometry']].join(w).join(v); G=G.reset_index(); G.to_file(f'{od}/fallow_maps_wy2018_2024.gpkg',driver='GPKG')
S=I.groupby('wy').agg(fields=('field_id','size'),fallow_fields=('fallow','sum'),fallow_ac=('area_ac',lambda x:x[I.loc[x.index,'fallow']==1].sum()),mean_intensity=('intensity','mean'),missing=('months',lambda x:int((x<12).sum()))).round(3); S.to_csv(f'{od}/summary_by_year.csv'); print(S.to_string())
