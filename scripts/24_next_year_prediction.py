"""Next-year fallow prediction, one-time test on water year 2024. usage: 24_next_year_prediction.py FIELDS PARTS MANIFEST PROCESSED OUTPUTS
Every feature for year t is knowable by 30 Sep of year t-1:
 lag1 = share of fallow months Oct-Apr of t-1 (Apr 1 + 180 days falls inside t-1); lag2 = full-year map of t-2 (its windows end by Mar of t-1);
 trailing NDVI to 30 Sep t-1; neighbour lag1 share; DWR class of t-1; static drivers. Development: t = 2020-2023, year and 3 km block both held out.
Model, feature set and threshold are chosen on development data only, written to forecast_design.json, then 2024 is scored once."""
import sys, os, json, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
import numpy as np, pandas as pd, geopandas as gpd
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.metrics import f1_score, cohen_kappa_score, recall_score, precision_score, accuracy_score, roc_auc_score
import tslib
warnings.filterwarnings('ignore')
fg, parts, mf, p, out = sys.argv[1:6]; od=f'{out}/forecast'
R=pd.read_csv(f'{p}/field_drivers.csv',index_col=0); DR={'Excessively drained':6,'Somewhat excessively drained':5,'Well drained':4,'Moderately well drained':3,'Somewhat poorly drained':2,'Poorly drained':1,'Very poorly drained':0}; R['soil_drain_ord']=R.soil_drclassdcd.map(DR)
STAT=['soil_claytotal','soil_ec','soil_sar','soil_drain_ord','soil_icc_num','elev_rel_m','slope_deg','dist_named_canal_m','dist_river_m','area_ac','compactness_pp']
bk=np.floor(R.x_km/3).astype(int).astype(str)+'_'+np.floor(R.y_km/3).astype(int).astype(str); ub=bk.drop_duplicates().sample(frac=1,random_state=7).values; R['fold']=bk.map({b:i%5 for i,b in enumerate(ub)}); R['block']=bk
I=pd.read_csv(f'{out}/maps/field_year.csv.gz'); M=pd.read_csv(f'{out}/maps/monthly_state.csv.gz',parse_dates=['date']); C=pd.read_csv(f'{p}/field_dwr_class_by_year.csv.gz')
s2,_,_=tslib.load_series(parts,mf,'2017-08-01','2023-09-30'); assert s2.date.max()<=pd.Timestamp('2023-09-30')
ct=R.loc[I.field_id.unique(),['x_km','y_km']]; fid=ct.index.values; nb=cKDTree(ct.values*1000).query(ct.values*1000,k=9)[1][:,1:]; pos={f:i for i,f in enumerate(fid)}
crop={'P':'P','F':'F','T':'T','G':'G','C':'C','D':'C','V':'C','YP':'C','I':'I','X':'I','U':'O','UL':'O'}; C['cl']=C.dwr_class.map(crop).fillna('O')
full=I.pivot(index='field_id',columns='wy',values='fallow'); cla=C.pivot(index='field_id',columns='wy',values='cl')
def build(t):
    y1=t-1; m=M[(M.date>=f'{y1-1}-10-01')&(M.date<=f'{y1}-04-01')]; assert m.date.max()+pd.Timedelta(days=180)<=pd.Timestamp(f'{y1}-09-30')
    d=pd.DataFrame(index=fid); d.index.name='field_id'; d['lag1']=m.groupby('field_id').state.mean().reindex(fid); d['lag1_fallow']=(d.lag1>=.5).astype(float); d['lag2_fallow']=full[t-2].reindex(fid)
    a=s2[(s2.date<=f'{y1}-09-30')&(s2.date>pd.Timestamp(f'{y1}-09-30')-pd.Timedelta(days=120))].groupby('field_id').ndvi; d['ndvi_max_120']=a.max().reindex(fid); d['ndvi_mean_120']=a.mean().reindex(fid); d['bare_frac_120']=a.apply(lambda v:(v<.2).mean()).reindex(fid)
    z=s2[(s2.date<=f'{y1}-09-30')&(s2.date>pd.Timestamp(f'{y1}-09-30')-pd.Timedelta(days=400))]; z=z[z.ndvi>=.3].groupby('field_id').date.max(); d['days_since_green']=(pd.Timestamp(f'{y1}-09-30')-z.reindex(fid)).dt.days.fillna(400).astype(float)
    d['nb_lag1']=d.lag1_fallow.values[nb].mean(1); d['y']=full[t].reindex(fid); d['wy']=t; d['prior_cl']=cla[t-1].reindex(fid).fillna('O').values; return d.join(R[STAT+['fold','block']]).reset_index()
A=pd.concat([build(t) for t in range(2020,2025)],ignore_index=True); cats=pd.get_dummies(A.prior_cl,prefix='cl').astype(float); A=pd.concat([A,cats],axis=1); CC=list(cats.columns)
HIST=['lag1','lag1_fallow','lag2_fallow','ndvi_max_120','ndvi_mean_120','bare_frac_120','days_since_green','nb_lag1']; S={'history':HIST,'history+crop':HIST+CC,'history+crop+static':HIST+CC+STAT}
for c in HIST+STAT: A[c]=A[c].fillna(A[A.wy<2024][c].median())
dev=A[A.wy<2024].copy(); test=A[A.wy==2024].copy(); assert test.y.notna().all(); print('dev rows',len(dev),'test rows',len(test),'dev fallow rate',round(dev.y.mean(),3),flush=True)
def mk(s=7): return RF(n_estimators=500,min_samples_leaf=3,max_features=.5,class_weight='balanced_subsample',n_jobs=-1,random_state=s)
yrs=[2020,2021,2022,2023]; P={}
for n,cols in S.items():
    p_=np.full(len(dev),np.nan)
    for y in yrs:
        for k in range(5):
            tr=dev[(dev.wy!=y)&(dev.fold!=k)]; te=(dev.wy==y)&(dev.fold==k)
            if te.sum(): p_[te.values]=mk().fit(tr[cols],tr.y).predict_proba(dev.loc[te,cols])[:,1]
    P[n]=p_
yd=dev.y.values.astype(int); sc={}
for n,q in P.items():
    ts=np.arange(.2,.81,.01); f=[f1_score(yd,(q>=t).astype(int)) for t in ts]; b=int(np.argmax(f)); sc[n]=dict(F1=float(f[b]),thr=float(round(ts[b],2)),AUC=float(roc_auc_score(yd,q)))
print('dev by feature set',json.dumps(sc),flush=True); best=max(sc,key=lambda k:sc[k]['F1']); thr=sc[best]['thr']
per=lambda d:(d.lag1_fallow.values>=.5).astype(int)
tab=dev.groupby(['lag1_fallow','lag2_fallow']).y.mean(); print('Markov table from dev:',tab.round(3).to_dict()); mkv=lambda d:d.set_index(['lag1_fallow','lag2_fallow']).index.map(tab).fillna(dev.y.mean()).values
print('dev baselines: persistence F1 %.3f | Markov F1 %.3f AUC %.3f'%(f1_score(yd,per(dev)),f1_score(yd,(mkv(dev)>=.5).astype(int)),roc_auc_score(yd,mkv(dev))),flush=True)
design=dict(model=best,features=S[best],threshold=thr,train_years=yrs,target='frozen map fallow, water year t',decided_on='development years only'); json.dump(design,open(f'{od}/forecast_design.json','w'),indent=1); print('FROZEN',best,'threshold',thr,flush=True)
cols=S[best]; p24=np.full(len(test),np.nan)
for k in range(5):
    tr=dev[dev.fold!=k]; te=(test.fold==k).values; p24[te]=mk().fit(tr[cols],tr.y).predict_proba(test.loc[te,cols])[:,1]
test['p']=p24; test['pred']=(p24>=thr).astype(int); test['per']=per(test); test['mkv_p']=mkv(test); test['mkv']=(test.mkv_p>=.5).astype(int); yt=test.y.values.astype(int)
test[['field_id','y','p','pred','per','mkv_p','mkv','block','area_ac']].to_csv(f'{od}/prediction_2024.csv',index=False)
def row(n,h,q=None): return dict(model=n,OA=accuracy_score(yt,h),kappa=cohen_kappa_score(yt,h),F1=f1_score(yt,h),recall=recall_score(yt,h),precision=precision_score(yt,h,zero_division=0),AUC=roc_auc_score(yt,q) if q is not None else np.nan,pred_ac=test.area_ac[h==1].sum(),actual_ac=test.area_ac[yt==1].sum())
Rs=pd.DataFrame([row('RF '+best,test.pred.values,p24),row('persistence',test.per.values),row('Markov order 2',test.mkv.values,test.mkv_p.values)]); Rs.round(3).to_csv(f'{od}/metrics_2024.csv',index=False); print(Rs.round(3).T.to_string())
ch=pd.DataFrame([dict(group=g,n=int(v.sum()),fallow=int(yt[v].sum()),RF_recall=recall_score(yt[v],test.pred.values[v],zero_division=0),RF_precision=precision_score(yt[v],test.pred.values[v],zero_division=0),persist_recall=recall_score(yt[v],test.per.values[v],zero_division=0)) for g,v in {'new entrants (not fallow in lag1)':test.lag1_fallow.values==0,'previously fallow':test.lag1_fallow.values==1}.items()]); ch.round(3).to_csv(f'{od}/change_groups_2024.csv',index=False); print(ch.round(3).to_string(index=False))
bl=test.block.values; ubl=np.unique(bl); ix=[np.where(bl==b)[0] for b in ubl]; cn=lambda h:np.array([[((h[i]==1)&(yt[i]==1)).sum(),((h[i]==1)&(yt[i]==0)).sum(),((h[i]==0)&(yt[i]==1)).sum()] for i in ix]); F=lambda c:2*c[:,0]/np.maximum(2*c[:,0]+c[:,1]+c[:,2],1); ii=np.random.default_rng(7).integers(0,len(ubl),(2000,len(ubl)))
A1,B1=cn(test.pred.values),cn(test.per.values); dd=F(A1[ii].sum(1))-F(B1[ii].sum(1)); print('F1, RF minus persistence: %.3f CI %s'%(F(A1.sum(0,keepdims=True))[0]-F(B1.sum(0,keepdims=True))[0],np.percentile(dd,[2.5,97.5]).round(3).tolist()))
c=C[C.wy==2024].set_index('field_id').cl.reindex(test.field_id).values; v=np.isin(c,['P','F','T','G','C','I']); t2=(c[v]=='I').astype(int); print('secondary reference DWR 2024 provisional (n=%d): RF F1 %.3f | persistence F1 %.3f'%(int(v.sum()),f1_score(t2,test.pred.values[v]),f1_score(t2,test.per.values[v])))
imp=mk().fit(dev[cols],dev.y).feature_importances_; print(pd.Series(imp,cols).sort_values(ascending=False).round(3).head(8).to_string())
