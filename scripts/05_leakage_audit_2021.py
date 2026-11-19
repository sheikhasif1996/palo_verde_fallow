"""Leakage audit of the 2021 classifier. usage: 05_leakage_audit_2021.py FIELDS PARTS MANIFEST OUT"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
import numpy as np, pandas as pd, geopandas as gpd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, recall_score, precision_score
import tslib
warnings.filterwarnings('ignore')
fg, parts, mf, out = sys.argv[1:5]
SN = {'jan': pd.Timestamp('2021-01-01'), 'dec': pd.Timestamp('2021-12-31')}
COLS = tslib.OPT + tslib.SAR
g = gpd.read_file(fg); g = g[g.label!='non_crop'].copy(); c = g.centroid; g['x']=c.x; g['y0']=c.y; g = g.set_index('field_id')
crop = g.SYMB_CLASS.isin(list('PFTGCD'))
s2,s1,ls = tslib.load_series(parts, mf, '2020-08-01', '2021-12-31')
for k,t in SN.items():
    assert s2[(s2.date>t-pd.Timedelta(days=150))&(s2.date<=t)].date.max()<=t
def table(days):
    r=[]
    for k,t in SN.items():
        X = tslib.window_features(s2,s1,ls,t,days); X = X[(X.n_s2>=4)&(X.n_s1>=3)].copy(); f=g.loc[X.index,f'f_{k}']
        X['snap']=k; X['y']=np.where(f>=.7,1,np.where((f<=.05)&crop.loc[X.index],0,-1)); X['x']=g.loc[X.index,'x'].values; X['y0']=g.loc[X.index,'y0'].values; r.append(X.reset_index())
    return pd.concat(r,ignore_index=True)
def folds(size, seed=7):
    b = np.floor(g.x/size).astype(int).astype(str)+'_'+np.floor(g.y0/size).astype(int).astype(str)
    u = b.drop_duplicates().sample(frac=1,random_state=seed).values; m={v:i%5 for i,v in enumerate(u)}; return b.map(m), len(u)
def model(s=7): return RF(n_estimators=300,min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=-1,random_state=s)
def met(y,h,**k): return dict(**k,n=len(y),OA=accuracy_score(y,h),kappa=cohen_kappa_score(y,h),F1=f1_score(y,h),recall=recall_score(y,h),precision=precision_score(y,h,zero_division=0))
def run(d, fo, buf=0, yy=None):
    y = d.y.values if yy is None else yy; f = d.field_id.map(fo).values; L = y>=0; P=np.full(len(d),-1); ntr=[]; nn=[]
    for k in range(5):
        te = L&(f==k); tr = L&(f!=k)
        assert not set(d.field_id[te]) & set(d.field_id[tr]), 'field in both train and test'
        T = cKDTree(d.loc[te,['x','y0']].values); dist = T.query(d.loc[tr,['x','y0']].values)[0]; nn.append(np.median(dist))
        if buf: idx=np.where(tr)[0][dist>=buf]; tr=np.zeros(len(d),bool); tr[idx]=True
        ntr.append(tr.sum()); P[te]=model().fit(d.loc[tr,COLS],y[tr]).predict(d.loc[te,COLS])
    return met(y[L],P[L]), np.mean(ntr), np.mean(nn)
R=[]; D={w:table(w) for w in [60,90,120,150]}; d=D[90]; f3,nb=folds(3000)
m,tr,nn=run(d,f3); R.append(dict(test='baseline: 3 km blocks, no buffer',train_n=round(tr),**m)); print('median distance train field to nearest test field (m):',round(nn),flush=True)
for b in [500,1000,2000,5000]:
    m,tr,_=run(d,f3,buf=b); R.append(dict(test=f'3 km blocks, {b} m buffer',train_n=round(tr),**m))
for s in [6000,10000,15000]:
    fo,n=folds(s); print('block',s,'n blocks',n,flush=True); m,tr,_=run(d,fo); R.append(dict(test=f'{s//1000} km blocks, no buffer',train_n=round(tr),**m))
for sd in [1,2,3]:
    fo,_=folds(3000,seed=sd); m,tr,_=run(d,fo); R.append(dict(test=f'3 km blocks, fold seed {sd}',train_n=round(tr),**m))
# nested window choice: window picked on training blocks only
P=[];Y=[];ch=[]
for k in range(5):
    sc={}
    for w,x in D.items():
        fx=x.field_id.map(f3).values; yv=x.y.values; tr=np.where((yv>=0)&(fx!=k))[0]; inner=x.field_id.map(folds(3000,seed=11)[0]).values
        pr=[];yt=[]
        for i in range(5):
            if i==k: continue
            a=tr[inner[tr]!=i]; b=tr[inner[tr]==i]; pr+=list(model().fit(x.loc[a,COLS],yv[a]).predict(x.loc[b,COLS])); yt+=list(yv[b])
        sc[w]=f1_score(yt,pr)
    w=max(sc,key=sc.get); ch.append(w); x=D[w]; fx=x.field_id.map(f3).values; yv=x.y.values; a=(yv>=0)&(fx!=k); b=(yv>=0)&(fx==k)
    P+=list(model().fit(x.loc[a,COLS],yv[a]).predict(x.loc[b,COLS])); Y+=list(yv[b])
print('windows chosen in nested CV',ch,flush=True); R.append(dict(test=f'nested window selection {ch}',train_n=np.nan,**met(np.array(Y),np.array(P))))
rng=np.random.default_rng(7)
for i in range(3):
    yy=d.y.values.copy(); L=yy>=0; yy[L]=rng.permutation(yy[L]); m,tr,_=run(d,f3,yy=yy); R.append(dict(test=f'labels shuffled, run {i+1} (should collapse)',train_n=round(tr),**m))
A=pd.DataFrame(R); A.round(3).to_csv(f'{out}/leakage_audit_2021.csv',index=False); print(A.drop(columns=['n']).round(3).to_string(index=False))
# preprocessing that uses all fields: compare with training-only constants
fo=s1.field_id.map(f3); dv=[]
for k in range(5):
    x=s1[fo!=k]; dv.append((x.groupby('rel_orbit').vv_db.mean()-s1.groupby('rel_orbit').vv_db.mean()).abs().max())
print('largest change in orbit-centring constant if test fold removed (dB):',round(max(dv),4))
pos=d[d.y==1]; print('fallow share of labelled samples',round((d.y==1).sum()/(d.y>=0).sum(),3))
