"""One-time test of the frozen method on water year 2023. Settings come from outputs/frozen_method.json.
usage: 10_score_holdout_2023.py FIELDS PARTS MANIFEST DWR_BASE OUT"""
import sys, os, glob, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
import numpy as np, pandas as pd, geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, recall_score, precision_score, confusion_matrix
import tslib
warnings.filterwarnings('ignore')
fg, parts, mf, base, out = sys.argv[1:6]
C=json.load(open(f'{out}/frozen_method.json')); AFTER,BEFORE=C['window_days_after'],C['window_days_before']; COLS=C['features']; assert COLS==tslib.OPT
IDLE=['I','X']; CROP=list('PFTGCD')+['YP']; Y=2023
g=gpd.read_file(fg); g=g[g.label!='non_crop'].copy(); c=g.centroid
g['block']=(np.floor(c.x/3000).astype(int).astype(str)+'_'+np.floor(c.y/3000).astype(int).astype(str))
ub=g.block.drop_duplicates().sample(frac=1,random_state=7).values; g['fold']=g.block.map({b:i%5 for i,b in enumerate(ub)})
g=g.set_index('field_id'); crop21=g.SYMB_CLASS.isin(list('PFTGCD'))
x0,y0,x1,y1=g.total_bounds; res=10; W=int((x1-x0)/res)+2; H=int((y1-y0)/res)+2; T=from_origin(x0-res,y1+res,res,res)
ids=rasterize(zip(g.geometry,g.index),(H,W),transform=T,dtype='int32'); N=int(ids.max())+1; n=np.bincount(ids.ravel(),minlength=N).astype(float)
s2,s1,ls=tslib.load_series(parts,mf,'2020-08-01','2024-03-31')
def feats(t):
    X=tslib.window_features(s2,s1,ls,t+pd.Timedelta(days=AFTER),BEFORE+AFTER); return X[(X.n_s2>=6)&(X.n_s1>=4)].copy()
tr=[]
for k,t in {'jan':pd.Timestamp('2021-01-01'),'dec':pd.Timestamp('2021-12-31')}.items():
    X=feats(t); f=g.loc[X.index,f'f_{k}']; X['y']=np.where(f>=.7,1,np.where((f<=.05)&crop21.loc[X.index],0,-1)); X['fold']=g.loc[X.index,'fold'].values; tr.append(X[X.y>=0])
tr=pd.concat(tr); print('training',tr.y.value_counts().to_dict(),flush=True)
rows=[]
for t in pd.date_range(f'{Y-1}-10-01',f'{Y}-09-01',freq='MS'):
    X=feats(t); X['date']=t; X['fold']=g.loc[X.index,'fold'].values; rows.append(X.reset_index())
D=pd.concat(rows,ignore_index=True); print('field-months',len(D),'months',D.date.nunique(),flush=True)
D['rule']=(D.ndvi_max<.3).astype(float); p=np.full(len(D),np.nan)
for k in range(5):
    a=tr[tr.fold!=k]; b=(D.fold==k).values
    p[b]=RF(n_estimators=500,min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=-1,random_state=7).fit(a[COLS],a.y).predict(D.loc[b,COLS])
D['frozen']=p
I=D.groupby('field_id').agg(months=('date','size'),frozen=('frozen','mean'),rule=('rule','mean')).reset_index(); I=I[I.months>=8]
s=glob.glob(f'{base}/dwr_{Y}/shp/**/*.shp',recursive=True)[0]; crs=gpd.read_file(s,rows=1).crs
d=gpd.read_file(s,bbox=tuple(gpd.GeoSeries.from_xy([x0-500,x1+500],[y0-500,y1+500],crs=32611).to_crs(crs).total_bounds)).to_crs(32611)
k=next(c for c in d.columns if c.upper()=='SYMB_CLASS'); d=d[d.geometry.notna()&~d.geometry.is_empty]; print('DWR classes',d[k].value_counts().to_dict(),flush=True)
mk=lambda v: rasterize(((a,1) for a in d[d[k].isin(v)].geometry),(H,W),transform=T,dtype='uint8')
fi=np.bincount(ids.ravel(),weights=mk(IDLE).ravel(),minlength=N)/np.maximum(n,1); fc=np.bincount(ids.ravel(),weights=mk(CROP).ravel(),minlength=N)/np.maximum(n,1)
I['idle_frac']=fi[I.field_id]; I['crop_frac']=fc[I.field_id]; I['dwr']=np.where(I.idle_frac>=.7,1,np.where(I.crop_frac>=.7,0,-1)); I['area_ac']=I.field_id.map(g.area_ac); I['block']=I.field_id.map(g.block)
I.to_csv(f'{out}/holdout_2023_field_results.csv.gz',index=False)
v=I[I.dwr>=0].copy(); print('scored fields',len(v),'DWR idle',int(v.dwr.sum()),'of',len(I),'fields',flush=True)
res=[]; bl=v.block.values; ub2=np.unique(bl); idx=[np.where(bl==b)[0] for b in ub2]; y=v.dwr.values; rng=np.random.default_rng(7); NN=len(ub2); ii=rng.integers(0,NN,(2000,NN))
F=lambda c:2*c[:,0]/np.maximum(2*c[:,0]+c[:,1]+c[:,2],1)
for m in ['frozen','rule']:
    h=(v[m].values>=.5).astype(int); cc=np.array([[((h[i]==1)&(y[i]==1)).sum(),((h[i]==1)&(y[i]==0)).sum(),((h[i]==0)&(y[i]==1)).sum()] for i in idx]); lo,hi=np.percentile(F(cc[ii].sum(1)),[2.5,97.5])
    res.append(dict(model=m,n=len(y),OA=accuracy_score(y,h),kappa=cohen_kappa_score(y,h),F1=f1_score(y,h),F1_lo=lo,F1_hi=hi,recall=recall_score(y,h),precision=precision_score(y,h,zero_division=0),pred_area_ac=I.loc[I[m]>=.5,'area_ac'].sum()))
    print(m,'confusion [[TN FP],[FN TP]]',confusion_matrix(y,h).tolist())
R=pd.DataFrame(res); R['dwr_idle_area_ac']=(I.area_ac*I.idle_frac).sum(); R.round(4).to_csv(f'{out}/holdout_2023_metrics.csv',index=False); print(R.round(3).T.to_string())
