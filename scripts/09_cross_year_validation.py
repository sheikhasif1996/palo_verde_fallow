"""Cross-year validation against DWR for water years 2018-2022. 2023 is sealed and 2024 is reserved: neither is scored here.
usage: 09_cross_year_validation.py FIELDS PARTS MANIFEST DWR_BASE OUT
Each month gets a fallow state from the window 90 d before to 120 d after it; a water year's fallow intensity is the share of its
months in that state. A field counts as predicted fallow if intensity >= 0.5, fixed before scoring. RF models train on the 2021 PVID
labels and predict only fields from spatial blocks they did not train on. Idle = DWR I or X; cropped = P,F,T,G,C,D,YP; field needs >=70% of one."""
import sys, os, glob, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
import numpy as np, pandas as pd, geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, recall_score, precision_score
import tslib
warnings.filterwarnings('ignore')
fg, parts, mf, base, out = sys.argv[1:6]
YEARS=[2018,2019,2020,2021,2022]; REPORT={2018:25948,2019:13876,2020:14880,2021:20039,2022:24045}
AFTER,BEFORE=int(os.environ.get('AFTER',120)),90; AC=4046.8564; IDLE=['I','X']; CROP=list('PFTGCD')+['YP']
g=gpd.read_file(fg); g=g[g.label!='non_crop'].copy(); c=g.centroid
g['block']=(np.floor(c.x/3000).astype(int).astype(str)+'_'+np.floor(c.y/3000).astype(int).astype(str))
ub=g.block.drop_duplicates().sample(frac=1,random_state=7).values; g['fold']=g.block.map({b:i%5 for i,b in enumerate(ub)})
g=g.set_index('field_id'); crop21=g.SYMB_CLASS.isin(list('PFTGCD'))
x0,y0,x1,y1=g.total_bounds; res=10; W=int((x1-x0)/res)+2; H=int((y1-y0)/res)+2; T=from_origin(x0-res,y1+res,res,res)
ids=rasterize(zip(g.geometry,g.index),(H,W),transform=T,dtype='int32'); N=int(ids.max())+1; n=np.bincount(ids.ravel(),minlength=N).astype(float)
def dwr(y):
    s=glob.glob(f'{base}/dwr_{y}/shp/**/*.shp',recursive=True)[0]; crs=gpd.read_file(s,rows=1).crs
    d=gpd.read_file(s,bbox=tuple(gpd.GeoSeries.from_xy([x0-500,x1+500],[y0-500,y1+500],crs=32611).to_crs(crs).total_bounds)).to_crs(32611)
    k=next(c for c in d.columns if c.upper()=='SYMB_CLASS'); d=d[d.geometry.notna()&~d.geometry.is_empty]
    mk=lambda v: rasterize(((a,1) for a in d[d[k].isin(v)].geometry),(H,W),transform=T,dtype='uint8')
    fi=np.bincount(ids.ravel(),weights=mk(IDLE).ravel(),minlength=N)/np.maximum(n,1); fc=np.bincount(ids.ravel(),weights=mk(CROP).ravel(),minlength=N)/np.maximum(n,1)
    r=pd.DataFrame({'idle_frac':fi[g.index],'crop_frac':fc[g.index]},index=g.index); r['dwr']=np.where(r.idle_frac>=.7,1,np.where(r.crop_frac>=.7,0,-1)); return r
s2,s1,ls=tslib.load_series(parts,mf,'2017-08-01','2023-03-31')
def feats(t):
    X=tslib.window_features(s2,s1,ls,t+pd.Timedelta(days=AFTER),BEFORE+AFTER); return X[(X.n_s2>=6)&(X.n_s1>=4)].copy()
tr=[]
for k,t in {'jan':pd.Timestamp('2021-01-01'),'dec':pd.Timestamp('2021-12-31')}.items():
    X=feats(t); f=g.loc[X.index,f'f_{k}']; X['y']=np.where(f>=.7,1,np.where((f<=.05)&crop21.loc[X.index],0,-1)); X['fold']=g.loc[X.index,'fold'].values; tr.append(X[X.y>=0])
tr=pd.concat(tr); print('training samples',tr.y.value_counts().to_dict(),flush=True)
dates=[t for y in YEARS for t in pd.date_range(f'{y-1}-10-01',f'{y}-09-01',freq='MS')]
rows=[]
for t in dates:
    X=feats(t); X['date']=t; X['wy']=t.year+(t.month>=10); X['fold']=g.loc[X.index,'fold'].values; rows.append(X.reset_index())
D=pd.concat(rows,ignore_index=True); print('field-months',len(D),'fields per month min/median',D.groupby('date').size().min(),int(D.groupby('date').size().median()),flush=True)
OPT,SAR=tslib.OPT,tslib.SAR; MODS={'RF optical':OPT,'RF optical+SAR':OPT+SAR}
D['rule NDVI<0.3']=(D.ndvi_max<.3).astype(float)
for m,cols in MODS.items():
    p=np.full(len(D),np.nan)
    for k in range(5):
        a=tr[tr.fold!=k]; b=(D.fold==k).values; p[b]=RF(n_estimators=500,min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=-1,random_state=7).fit(a[cols],a.y).predict(D.loc[b,cols])
    D[m]=p
names=['rule NDVI<0.3']+list(MODS)
I=D.groupby(['field_id','wy']).agg(months=('date','size'),**{n:(n,'mean') for n in names}).reset_index(); I=I[I.months>=8]
L=[]
for y in YEARS:
    d=dwr(y); d['wy']=y; L.append(d.reset_index())
L=pd.concat(L); I=I.merge(L,on=['field_id','wy']); I['area_ac']=I.field_id.map(g.area_ac); I['block']=I.field_id.map(g.block)
I.to_csv(f'{out}/fallow_intensity_wy2018_2022.csv.gz',index=False)
res=[]; acr=[]
for y in YEARS:
    q=I[I.wy==y]; v=q.dwr>=0
    for n in names:
        h=(q[n]>=.5).astype(int); t=q.dwr
        res.append(dict(wy=y,model=n,n=int(v.sum()),n_dwr_idle=int(t[v].sum()),OA=accuracy_score(t[v],h[v]),kappa=cohen_kappa_score(t[v],h[v]),F1=f1_score(t[v],h[v]),recall=recall_score(t[v],h[v]),precision=precision_score(t[v],h[v],zero_division=0)))
        acr.append(dict(wy=y,model=n,pred_ac=q.loc[h==1,'area_ac'].sum()))
R=pd.DataFrame(res); R.round(4).to_csv(f'{out}/cross_year_metrics.csv',index=False); print(R.pivot_table(index='wy',columns='model',values='F1').round(3).to_string()); print(R.pivot_table(index='wy',columns='model',values='kappa').round(3).to_string())
print(R[R.model=='rule NDVI<0.3'].drop(columns='model').round(3).to_string(index=False))
A=pd.DataFrame(acr).pivot(index='wy',columns='model',values='pred_ac').round(0); A['DWR I+X in fields']=[I[(I.wy==y)&(I.idle_frac>0)].eval('area_ac*idle_frac').sum().round(0) for y in A.index]; A['PVID report fallow (WT ac, calendar yr)']=[REPORT[y] for y in A.index]; A.to_csv(f'{out}/cross_year_area.csv'); print(A.to_string())
print('Correlation with report, 5 years:',A.drop(columns='PVID report fallow (WT ac, calendar yr)').corrwith(A['PVID report fallow (WT ac, calendar yr)']).round(3).to_dict())
v=I[I.dwr>=0]; bl=v.block.values; ub2=np.unique(bl); idx=[np.where(bl==b)[0] for b in ub2]; y_=v.dwr.values; rng=np.random.default_rng(7)
def cnt(n):
    h=(v[n].values>=.5).astype(int); return np.array([[((h[i]==1)&(y_[i]==1)).sum(),((h[i]==1)&(y_[i]==0)).sum(),((h[i]==0)&(y_[i]==1)).sum()] for i in idx])
F=lambda c:2*c[:,0]/np.maximum(2*c[:,0]+c[:,1]+c[:,2],1); N=len(ub2); ii=rng.integers(0,N,(2000,N))
print('pooled F1:',{n:(round(float(F(cnt(n).sum(0,keepdims=True))[0]),3),[round(float(z),3) for z in np.percentile(F(cnt(n)[ii].sum(1)),[2.5,97.5])]) for n in names})
for a,b in [('RF optical','rule NDVI<0.3'),('RF optical+SAR','RF optical')]:
    da=F(cnt(a)[ii].sum(1))-F(cnt(b)[ii].sum(1)); print('F1 diff',a,'-',b,round(float(F(cnt(a).sum(0,keepdims=True))[0]-F(cnt(b).sum(0,keepdims=True))[0]),3),[round(float(z),3) for z in np.percentile(da,[2.5,97.5])])
