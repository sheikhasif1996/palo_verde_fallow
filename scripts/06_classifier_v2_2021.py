"""Adds field-history features to the 2021 classifier and compares feature sets and algorithms.
usage: 06_classifier_v2_2021.py FIELDS PARTS MANIFEST OUT
History features use only observations up to the label date. Folds match script 04 (3 km blocks, seed 7).
Differences in F1 get 95% block-bootstrap intervals, resampling the spatial blocks."""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
import numpy as np, pandas as pd, geopandas as gpd
from sklearn.ensemble import RandomForestClassifier as RF, HistGradientBoostingClassifier as GB
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, recall_score, precision_score
import tslib
warnings.filterwarnings('ignore')
fg, parts, mf, out = sys.argv[1:5]
SN = {'jan': pd.Timestamp('2021-01-01'), 'dec': pd.Timestamp('2021-12-31')}
g = gpd.read_file(fg); g = g[g.label!='non_crop'].copy(); c = g.centroid
g['block'] = (np.floor(c.x/3000).astype(int).astype(str)+'_'+np.floor(c.y/3000).astype(int).astype(str))
ub = g.block.drop_duplicates().sample(frac=1, random_state=7).values; fm = {b:i%5 for i,b in enumerate(ub)}; g['fold']=g.block.map(fm)
g = g.set_index('field_id'); crop = g.SYMB_CLASS.isin(list('PFTGCD'))
s2,s1,ls = tslib.load_series(parts, mf, '2019-12-01', '2021-12-31')
HIST = ['days_since_green','ndvi_max_180','ndvi_max_365','ndvi_mean_365','ndvi_std_365','bare_frac_180','bare_frac_365']
SARH = ['vv_std_180','vh_std_180','vv_range_180','vh_range_180','vh_mean_180']
def history(t):
    H = {}
    a = s2[(s2.date<=t)&(s2.date>t-pd.Timedelta(days=400))]
    last = a[a.ndvi>=.3].groupby('field_id').date.max(); 
    for d in (180,365):
        x = s2[(s2.date<=t)&(s2.date>t-pd.Timedelta(days=d))].groupby('field_id').ndvi
        H[f'ndvi_max_{d}']=x.max(); H[f'bare_frac_{d}']=x.apply(lambda v:(v<.2).mean())
        if d==365: H['ndvi_mean_365']=x.mean(); H['ndvi_std_365']=x.std(); H['n_365']=x.size()
    out_ = pd.DataFrame(H); out_['days_since_green']=(t-last.reindex(out_.index)).dt.days.fillna(400).astype(float)
    b = s1[(s1.date<=t)&(s1.date>t-pd.Timedelta(days=180))].groupby('field_id')
    out_['vv_std_180']=b.vv_db_rel.std(); out_['vh_std_180']=b.vh_db_rel.std(); out_['vh_mean_180']=b.vh_db_rel.mean()
    out_['vv_range_180']=b.vv_db_rel.max()-b.vv_db_rel.min(); out_['vh_range_180']=b.vh_db_rel.max()-b.vh_db_rel.min()
    return out_
r=[]
for k,t in SN.items():
    X = tslib.window_features(s2,s1,ls,t,90).join(history(t),how='inner'); X = X[(X.n_s2>=4)&(X.n_s1>=3)&(X.n_365>=10)].copy(); f=g.loc[X.index,f'f_{k}']
    X['snap']=k; X['f']=f.values; X['fold']=g.loc[X.index,'fold'].values; X['block']=g.loc[X.index,'block'].values; X['label']=g.loc[X.index,'label'].values
    X['y']=np.where(f>=.7,1,np.where((f<=.05)&crop.loc[X.index],0,-1)); r.append(X.reset_index())
d = pd.concat(r,ignore_index=True); L=d.y.values>=0; y=d.y.values
print('samples', pd.Series(y).value_counts().to_dict(), flush=True)
S = {'optical (v1)':tslib.OPT, 'optical+SAR (v1)':tslib.OPT+tslib.SAR, 'optical+history':tslib.OPT+HIST, 'optical+history+SAR':tslib.OPT+HIST+tslib.SAR+SARH, 'history only':HIST}
M = {'RF':lambda:RF(n_estimators=500,min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=-1,random_state=7),
     'GB':lambda:GB(max_iter=300,learning_rate=.05,max_leaf_nodes=15,class_weight='balanced',random_state=7)}
def oof(cols,mk,tr_snap=None,te_snap=None):
    p=np.full(len(d),np.nan)
    for k in range(5):
        tr=L&(d.fold.values!=k); te=(d.fold.values==k)&L
        if tr_snap: tr&=d.snap.values==tr_snap; te&=d.snap.values==te_snap
        if te.sum(): p[te]=mk().fit(d.loc[tr,cols],y[tr]).predict_proba(d.loc[te,cols])[:,1]
    return p
def met(v,p,**k):
    h=(p[v]>=.5).astype(int); yy=y[v]; return dict(**k,n=int(v.sum()),OA=accuracy_score(yy,h),kappa=cohen_kappa_score(yy,h),F1=f1_score(yy,h),recall=recall_score(yy,h),precision=precision_score(yy,h,zero_division=0))
ch=d.label.isin(['fallow_jan_only','fallow_dec_only']).values
P={}; rows=[]
for sn,cols in S.items():
    for mn,mk in M.items():
        P[(sn,mn)]=oof(cols,mk)
        for sub,v in {'all':L,'changed_fields':L&ch,'jan':L&(d.snap.values=='jan'),'dec':L&(d.snap.values=='dec')}.items():
            rows.append(met(v,P[(sn,mn)],features=sn,model=mn,subset=sub))
R=pd.DataFrame(rows); R.round(4).to_csv(f'{out}/classifier_v2_metrics_2021.csv',index=False)
print(R[R.subset.isin(['all','changed_fields'])].pivot_table(index=['features','model'],columns='subset',values=['F1','recall','precision']).round(3).to_string())
rule=(d.days_since_green.values>=90).astype(float)
for sub,v in {'all':L,'changed_fields':L&ch}.items(): print('rule days_since_green>=90',sub,{k:round(x,3) for k,x in met(v,rule,a=0).items() if k in('OA','F1','recall','precision','kappa')})
# all-year fallow recall, the miss found earlier
afy=(d.label.values=='fallow_all_year')
for sn in ['optical+SAR (v1)','optical+history','optical+history+SAR']:
    print(sn,'RF: all-year fallow found', {s:round(float((P[(sn,'RF')][afy&(d.snap.values==s)&L]>=.5).mean()),3) for s in SN},flush=True)
tt=[]
for a,b in [('jan','dec'),('dec','jan')]:
    for sn,cols in S.items():
        for mn,mk in M.items():
            p=oof(cols,mk,a,b); v=L&(d.snap.values==b); tt.append(met(v,p,features=sn,model=mn,train=a,test=b))
T=pd.DataFrame(tt); T.round(4).to_csv(f'{out}/classifier_v2_transfer_2021.csv',index=False); print(T.pivot_table(index=['features','model'],columns=['train'],values='F1').round(3).to_string())
# block bootstrap of F1 differences on identical samples
bl=d.block.values; ubl=np.unique(bl[L]); rng=np.random.default_rng(7); idx={b:np.where(L&(bl==b))[0] for b in ubl}
def cnt(p):
    h=(p>=.5).astype(int); return np.array([[((h[i]==1)&(y[i]==1)).sum(),((h[i]==1)&(y[i]==0)).sum(),((h[i]==0)&(y[i]==1)).sum()] for i in idx.values()])
def diff(a,b,sub=None):
    A,B=cnt(P[a]),cnt(P[b]); F=lambda c:2*c[:,0]/(2*c[:,0]+c[:,1]+c[:,2]); N=len(ubl); ii=rng.integers(0,N,(2000,N))
    da=F(A[ii].sum(1))-F(B[ii].sum(1)); obs=F(A.sum(0,keepdims=True))[0]-F(B.sum(0,keepdims=True))[0]; return dict(a=str(a),b=str(b),F1_a_minus_b=obs,ci_lo=np.percentile(da,2.5),ci_hi=np.percentile(da,97.5))
pairs=[(('optical+history','RF'),('optical (v1)','RF')),(('optical+history+SAR','RF'),('optical+history','RF')),(('optical+SAR (v1)','RF'),('optical (v1)','RF')),(('optical+history+SAR','GB'),('optical+history+SAR','RF')),(('optical+history','GB'),('optical+history','RF'))]
Di=pd.DataFrame([diff(a,b) for a,b in pairs]); Di.round(4).to_csv(f'{out}/classifier_v2_f1_differences_2021.csv',index=False); print(Di.round(3).to_string(index=False))
