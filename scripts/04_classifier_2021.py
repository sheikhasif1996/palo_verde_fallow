"""Field-level fallow classifier on the two 2021 PVID snapshots. usage: 04_classifier_2021.py FIELDS PARTS MANIFEST OUT
Positive: field >=70% covered by the PVID map on that date. Negative: <=5% covered and DWR says cropped.
5-fold CV by 3 km spatial block; a field's two dates always share a fold. Every feature set is scored on the same samples."""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
import numpy as np, pandas as pd, geopandas as gpd
from scipy.stats import binomtest
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import cohen_kappa_score, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
import tslib
warnings.filterwarnings('ignore')
fg, parts, mf, out = sys.argv[1:5]; os.makedirs(out, exist_ok=True)
SN = {'jan': pd.Timestamp('2021-01-01'), 'dec': pd.Timestamp('2021-12-31')}
g = gpd.read_file(fg); g = g[g.label!='non_crop'].copy(); c = g.centroid
g['block'] = (np.floor(c.x/3000).astype(int).astype(str)+'_'+np.floor(c.y/3000).astype(int).astype(str))
ub = g.block.drop_duplicates().sample(frac=1, random_state=7).values; fold = {b:i%5 for i,b in enumerate(ub)}; g['fold']=g.block.map(fold)
g = g.set_index('field_id'); crop = g.SYMB_CLASS.isin(list('PFTGCD'))
print('blocks', len(ub), 'fields per fold', g.fold.value_counts().sort_index().tolist(), flush=True)
s2,s1,ls = tslib.load_series(parts, mf, '2020-08-01', '2021-12-31')
def table(days):
    r=[]
    for k,t in SN.items():
        X = tslib.window_features(s2,s1,ls,t,days); X = X[(X.n_s2>=4)&(X.n_s1>=3)].copy()
        f = g.loc[X.index, f'f_{k}']; X['snap']=k; X['f']=f.values; X['fold']=g.loc[X.index,'fold'].values; X['label']=g.loc[X.index,'label'].values; X['area_ac']=g.loc[X.index,'area_ac'].values
        X['y'] = np.where(f>=.7,1, np.where((f<=.05)&crop.loc[X.index],0,-1)); r.append(X.reset_index())
    d = pd.concat(r, ignore_index=True); d[tslib.THM]=d[tslib.THM].fillna(d[tslib.THM].median()); return d
SETS = {'optical':tslib.OPT,'sar':tslib.SAR,'thermal':tslib.THM,'optical+sar':tslib.OPT+tslib.SAR,'optical+thermal':tslib.OPT+tslib.THM,'optical+sar+thermal':tslib.OPT+tslib.SAR+tslib.THM}
def model(): return RF(n_estimators=500, min_samples_leaf=2, class_weight='balanced_subsample', n_jobs=-1, random_state=7)
def oof(d, cols):
    lab = d.y>=0; p = np.full(len(d), np.nan)
    for k in range(5):
        tr = lab & (d.fold!=k); te = d.fold==k
        p[te.values] = model().fit(d.loc[tr,cols], d.loc[tr,'y']).predict_proba(d.loc[te,cols])[:,1]
    return p
def score(y,p,name,**kw):
    h=(p>=.5).astype(int); return dict(name=name, **kw, n=len(y), n_fallow=int(y.sum()), OA=accuracy_score(y,h), kappa=cohen_kappa_score(y,h), F1=f1_score(y,h), recall_PA=recall_score(y,h), precision_UA=precision_score(y,h,zero_division=0), AUC=roc_auc_score(y,p))
def subsets(d):
    L = d.y>=0; ch = d.label.isin(['fallow_jan_only','fallow_dec_only'])
    return {'all':L, 'changed_fields':L&ch, 'jan':L&(d.snap=='jan'), 'dec':L&(d.snap=='dec')}
res=[]
print('--- window search, optical set', flush=True)
for w in [60,90,120,150]:
    d = table(w); p=oof(d,tslib.OPT); ss=subsets(d)
    print(w,'days', d.groupby(['snap','y']).size().to_dict(), {k:round(f1_score(d.y[v],(p[v]>=.5).astype(int)),3) for k,v in ss.items()}, flush=True)
    res.append(pd.DataFrame([score(d.y[v].values,p[v],'optical',window=w,subset=k) for k,v in ss.items()]))
pd.concat(res).round(4).to_csv(f'{out}/classifier_window_search_2021.csv',index=False)
W = int(sys.argv[5]) if len(sys.argv)>5 else 90
d = table(W); print('window used',W,'samples',d.y.value_counts().to_dict(), flush=True)
P = {n:oof(d,c) for n,c in SETS.items()}
rule = (d.ndvi_max<0.3).astype(float).values   # fixed a priori, not tuned
P['rule ndvi_max<0.3']=rule
ss=subsets(d); rows=[]
for n,p in P.items():
    for k,v in ss.items():
        y=d.y[v].values; q=p[v]; rows.append(score(y,q,n,window=W,subset=k))
M=pd.DataFrame(rows); M.round(4).to_csv(f'{out}/classifier_cv_metrics_2021.csv',index=False)
print(M[M.subset.isin(['all','changed_fields'])].drop(columns=['window']).round(3).to_string(index=False))
# temporal transfer: train one date, test the other, still excluding the test block from training
tt=[]
for a,b in [('jan','dec'),('dec','jan')]:
    for n,cols in SETS.items():
        p=np.full(len(d),np.nan); L=d.y>=0
        for k in range(5):
            tr=L&(d.snap==a)&(d.fold!=k); te=L&(d.snap==b)&(d.fold==k)
            if te.sum(): p[te.values]=model().fit(d.loc[tr,cols],d.loc[tr,'y']).predict_proba(d.loc[te,cols])[:,1]
        v=L&(d.snap==b); tt.append(score(d.y[v].values,p[v],n,train=a,test=b))
TT=pd.DataFrame(tt); TT.round(4).to_csv(f'{out}/classifier_transfer_2021.csv',index=False); print(TT.drop(columns=['n','n_fallow']).round(3).to_string(index=False))
# paired test
def mc(a,b,v):
    y=d.y[v].values; A=(P[a][v]>=.5)==y; B=(P[b][v]>=.5)==y; ab=int((A&~B).sum()); ba=int((~A&B).sum()); return dict(model_a=a,model_b=b,a_right_b_wrong=ab,b_right_a_wrong=ba,p=binomtest(ab,ab+ba,.5).pvalue if ab+ba else 1.0)
Mc=[dict(subset=k,**mc(a,b,v)) for k,v in ss.items() if k in('all','changed_fields') for a,b in [('rule ndvi_max<0.3','optical'),('optical','optical+sar'),('optical','optical+thermal')]]
pd.DataFrame(Mc).to_csv(f'{out}/classifier_mcnemar_2021.csv',index=False); print(pd.DataFrame(Mc).round(4).to_string(index=False))
# everything the model never saw: ambiguous, off-map bare, other idle
best='optical+sar'; p=P[best]; d['p']=p; d['pred']=(p>=.5).astype(int)
d[['field_id','snap','f','y','label','area_ac','fold','p','pred']].to_csv(f'{out}/classifier_oof_predictions_2021.csv',index=False)
bins=[-.01,.05,.2,.5,.7,1.01]; d['fbin']=pd.cut(d.f,bins,labels=['0-5%','5-20%','20-50%','50-70%','70-100%'])
A=d[d.label=='ambiguous'].groupby(['snap','fbin']).agg(fields=('p','size'),acres=('area_ac','sum'),pred_fallow_pct=('pred',lambda x:100*x.mean()),mean_p=('p','mean')).round(2)
A.to_csv(f'{out}/classifier_ambiguous_by_coverage_2021.csv'); print(A.to_string())
U=d.groupby(['snap','label']).agg(fields=('p','size'),pred_fallow_pct=('pred',lambda x:100*x.mean()),pred_fallow_ac=('area_ac',lambda x:x[d.loc[x.index,'pred']==1].sum())).round(1); print(U.to_string()); U.to_csv(f'{out}/classifier_prediction_by_group_2021.csv')
print('predicted fallow acres', d[d.pred==1].groupby('snap').area_ac.sum().round(0).to_dict(), 'official jan 10376 dec 16841', flush=True)
imp=model().fit(d.loc[d.y>=0,SETS[best]],d.loc[d.y>=0,'y']).feature_importances_; pd.Series(imp,SETS[best]).sort_values(ascending=False).round(3).to_csv(f'{out}/classifier_importance_2021.csv'); print(pd.Series(imp,SETS[best]).sort_values(ascending=False).round(3).head(8).to_string())
