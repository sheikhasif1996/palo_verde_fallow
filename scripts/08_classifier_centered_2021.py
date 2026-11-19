"""Retrospective fallow classifier: window from 90 days before to 120 days after each PVID label date.
Post-date imagery is valid for retrospective mapping and must never be used for next-year prediction.
usage: 08_classifier_centered_2021.py FIELDS PARTS MANIFEST OUT"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
import numpy as np, pandas as pd, geopandas as gpd
from sklearn.ensemble import RandomForestClassifier as RF, HistGradientBoostingClassifier as GB
from sklearn.metrics import f1_score, cohen_kappa_score, accuracy_score, recall_score, precision_score
import tslib
warnings.filterwarnings('ignore')
fg, parts, mf, out = sys.argv[1:5]
SN = {'jan': pd.Timestamp('2021-01-01'), 'dec': pd.Timestamp('2021-12-31')}; AFTER=120; BEFORE=90
g = gpd.read_file(fg); g = g[g.label!='non_crop'].copy(); c = g.centroid
g['block'] = (np.floor(c.x/3000).astype(int).astype(str)+'_'+np.floor(c.y/3000).astype(int).astype(str))
ub = g.block.drop_duplicates().sample(frac=1, random_state=7).values; g['fold']=g.block.map({b:i%5 for i,b in enumerate(ub)})
g = g.set_index('field_id'); crop = g.SYMB_CLASS.isin(list('PFTGCD'))
s2,s1,ls = tslib.load_series(parts, mf, '2019-12-01', '2022-05-31')
EX=['days_since_green','days_to_green','bare_run_days']
r=[]
for k,t in SN.items():
    X = tslib.window_features(s2,s1,ls,t+pd.Timedelta(days=AFTER),BEFORE+AFTER)
    a=s2[(s2.date<=t)&(s2.date>t-pd.Timedelta(days=400))]; last=a[a.ndvi>=.3].groupby('field_id').date.max()
    b=s2[(s2.date>t)&(s2.date<=t+pd.Timedelta(days=AFTER))]; first=b[b.ndvi>=.3].groupby('field_id').date.min()
    X['days_since_green']=(t-last.reindex(X.index)).dt.days.fillna(400).astype(float); X['days_to_green']=(first.reindex(X.index)-t).dt.days.fillna(AFTER+10).astype(float); X['bare_run_days']=X.days_since_green+X.days_to_green
    X=X[(X.n_s2>=6)&(X.n_s1>=4)].copy(); f=g.loc[X.index,f'f_{k}']; X['snap']=k; X['block']=g.loc[X.index,'block'].values; X['fold']=g.loc[X.index,'fold'].values; X['label']=g.loc[X.index,'label'].values
    X['y']=np.where(f>=.7,1,np.where((f<=.05)&crop.loc[X.index],0,-1)); r.append(X.reset_index())
d=pd.concat(r,ignore_index=True); y=d.y.values; L=y>=0; print('samples',pd.Series(y).value_counts().to_dict(),flush=True)
S={'optical':tslib.OPT,'optical+timing':tslib.OPT+EX,'SAR':tslib.SAR,'optical+SAR':tslib.OPT+tslib.SAR,'optical+timing+SAR':tslib.OPT+EX+tslib.SAR}
M={'RF':lambda:RF(n_estimators=500,min_samples_leaf=2,class_weight='balanced_subsample',n_jobs=-1,random_state=7),'GB':lambda:GB(max_iter=300,learning_rate=.05,max_leaf_nodes=15,class_weight='balanced',random_state=7)}
fo=d.fold.values; sn=d.snap.values
def oof(cols,mk,a=None,b=None):
    p=np.full(len(d),np.nan)
    for k in range(5):
        tr=L&(fo!=k); te=L&(fo==k)
        if a: tr&=sn==a; te&=sn==b
        if te.sum(): p[te]=mk().fit(d.loc[tr,cols],y[tr]).predict_proba(d.loc[te,cols])[:,1]
    return p
def met(v,p,**k):
    h=(p[v]>=.5).astype(int); t=y[v]; return dict(**k,n=int(v.sum()),OA=accuracy_score(t,h),kappa=cohen_kappa_score(t,h),F1=f1_score(t,h),recall=recall_score(t,h),precision=precision_score(t,h,zero_division=0))
ch=d.label.isin(['fallow_jan_only','fallow_dec_only']).values; P={}; rows=[]
for s,cols in S.items():
    for m,mk in M.items():
        P[(s,m)]=oof(cols,mk)
        for sub,v in {'all':L,'changed_fields':L&ch}.items(): rows.append(met(v,P[(s,m)],features=s,model=m,subset=sub))
R=pd.DataFrame(rows); R.round(4).to_csv(f'{out}/classifier_centered_metrics_2021.csv',index=False)
print(R.pivot_table(index=['features','model'],columns='subset',values=['F1','recall','precision','kappa']).round(3).to_string())
rule=(d.ndvi_max.values<.3).astype(float)
for sub,v in {'all':L,'changed_fields':L&ch}.items(): print('rule: NDVI below 0.3 from 90 d before to 120 d after',sub,{k:round(x,3) for k,x in met(v,rule).items() if k!='n'})
afy=d.label.values=='fallow_all_year'
for key in [('optical+SAR','RF'),('optical+timing+SAR','RF')]: print(key,'all-year fallow found',{s:round(float((P[key][afy&(sn==s)&L]>=.5).mean()),3) for s in SN})
tt=[]
for a,b in [('jan','dec'),('dec','jan')]:
    for s in ['optical','optical+timing','optical+SAR','optical+timing+SAR']:
        p=oof(S[s],M['RF'],a,b); tt.append(met(L&(sn==b),p,features=s,train=a,test=b))
T=pd.DataFrame(tt); T.round(4).to_csv(f'{out}/classifier_centered_transfer_2021.csv',index=False); print(T.pivot_table(index='features',columns='train',values='F1').round(3).to_string())
bl=d.block.values; ub2=np.unique(bl[L]); rng=np.random.default_rng(7); idx=[np.where(L&(bl==b))[0] for b in ub2]
def cnt(p):
    h=(p>=.5).astype(int); return np.array([[((h[i]==1)&(y[i]==1)).sum(),((h[i]==1)&(y[i]==0)).sum(),((h[i]==0)&(y[i]==1)).sum()] for i in idx])
def diff(a,b):
    A,B=cnt(P[a]),cnt(P[b]); F=lambda c:2*c[:,0]/np.maximum(2*c[:,0]+c[:,1]+c[:,2],1); N=len(ub2); ii=rng.integers(0,N,(2000,N)); da=F(A[ii].sum(1))-F(B[ii].sum(1))
    return dict(a=' / '.join(a),b=' / '.join(b),F1_diff=F(A.sum(0,keepdims=True))[0]-F(B.sum(0,keepdims=True))[0],ci_lo=np.percentile(da,2.5),ci_hi=np.percentile(da,97.5))
Di=pd.DataFrame([diff(a,b) for a,b in [(('optical+timing','RF'),('optical','RF')),(('optical+timing+SAR','RF'),('optical+timing','RF')),(('optical+SAR','RF'),('optical','RF')),(('optical+timing+SAR','GB'),('optical+timing+SAR','RF'))]]); Di.round(4).to_csv(f'{out}/classifier_centered_f1_differences_2021.csv',index=False); print(Di.round(3).to_string(index=False))
