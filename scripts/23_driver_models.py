"""Field-level drivers of fallow intensity, global RF versus geographically weighted RF. usage: 23_driver_models.py PROCESSED OUTPUTS [SMOKE]
Target years 2019-2023 only; 2024 is kept out for next-year prediction. Folds: 5 folds of 3 km blocks, seed 7, a field's years staying together.
GWRF: local RFs at k-means anchors from training fields only, bisquare weights over the nearest-k neighbourhood; each test field uses its nearest anchor.
Neighbourhood size k is picked inside the training blocks. The design was fixed before results were viewed."""
import sys, json, warnings
import numpy as np, pandas as pd, geopandas as gpd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor as RF
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
p, out = sys.argv[1:3]; SMOKE=len(sys.argv)>3; od=f'{out}/drivers'; K_A=8 if SMOKE else 60; BWS=[200,500] if SMOKE else [150,300,600,1200]; TR=30 if SMOKE else 100; PERMS=2 if SMOKE else 20; BOOT=50 if SMOKE else 2000
R=pd.read_csv(f'{p}/field_drivers.csv',index_col=0); I=pd.read_csv(f'{out}/maps/field_year.csv.gz'); I=I[I.wy.between(2018,2023)]; C=pd.read_csv(f'{p}/field_dwr_class_by_year.csv.gz')
DR={'Excessively drained':6,'Somewhat excessively drained':5,'Well drained':4,'Moderately well drained':3,'Somewhat poorly drained':2,'Poorly drained':1,'Very poorly drained':0}; R['soil_drain_ord']=R.soil_drclassdcd.map(DR)
STAT=['soil_claytotal','soil_ec','soil_sar','soil_drain_ord','soil_icc_num','elev_rel_m','slope_deg','dist_named_canal_m','dist_river_m','area_ac','compactness_pp']
R=R[[*STAT,'x_km','y_km']].copy()
g=gpd.read_file(f'{p}/fields_2021_labeled.gpkg',ignore_geometry=True).set_index('field_id'); ct=R[['x_km','y_km']]; bk=np.floor(ct.x_km*1000/3000).astype(int).astype(str)+'_'+np.floor(ct.y_km*1000/3000).astype(int).astype(str)
ub=bk.drop_duplicates().sample(frac=1,random_state=7).values; R['fold']=bk.map({b:i%5 for i,b in enumerate(ub)}); R['block']=bk
Y=I[I.wy.between(2019,2023)].groupby('field_id').intensity.mean().rename('y'); A=R.join(Y,how='inner').dropna(subset=['y']); print('fields',len(A),'blocks',A.block.nunique(),'mean intensity',round(A.y.mean(),3),flush=True)
med=lambda d,c:d[c].median()
def fill(tr,te,cols):
    tr,te=tr.copy(),te.copy(); m=tr[cols].median(); return tr.assign(**{c:tr[c].fillna(m[c]) for c in cols}),te.assign(**{c:te[c].fillna(m[c]) for c in cols})
def gmod(s=7): return RF(n_estimators=500,min_samples_leaf=3,max_features=.5,n_jobs=-1,random_state=s)
def r2block(y,p,bl):
    ub=np.unique(bl); ix={b:np.where(bl==b)[0] for b in ub}; a=np.array([[((y[i]-p[i])**2).sum(),y[i].sum(),(y[i]**2).sum(),len(i)] for i in ix.values()],float); return ub,a
def sumr2(a): ssr=a[:,0].sum(); sy,sy2,n=a[:,1].sum(),a[:,2].sum(),a[:,3].sum(); return 1-ssr/(sy2-sy**2/n)
G_p=np.full(len(A),np.nan); imp=[]; G_fold=[]
for k in range(5):
    tr=A[A.fold!=k]; te=A[A.fold==k]; tr,te=fill(tr,te,STAT); m=gmod().fit(tr[STAT],tr.y); P=m.predict(te[STAT]); G_p[(A.fold==k).values]=P; base=r2_score(te.y,P); rng=np.random.default_rng(k); G_fold.append(base)
    for c in STAT:
        d=[]
        for _ in range(PERMS):
            X=te[STAT].copy(); X[c]=rng.permutation(X[c].values); d.append(base-r2_score(te.y,m.predict(X)))
        imp.append(dict(fold=k,feature=c,dR2=float(np.mean(d))))
Imp=pd.DataFrame(imp).groupby('feature').dR2.agg(['mean','std']).sort_values('mean',ascending=False); Imp.round(4).to_csv(f'{od}/importance_where_global.csv'); print('Global RF, where: blocked R2 %.3f RMSE %.3f'%(r2_score(A.y,G_p),np.sqrt(mean_squared_error(A.y,G_p)))); print(Imp.round(4).to_string(),flush=True)
def fit_one(a,X,y,xy,bw,s):
    d=np.hypot(xy[:,0]-a[0],xy[:,1]-a[1]); b=min(bw,len(y)); idx=np.argpartition(d,b-1)[:b]; h=d[idx].max()+1e-6; w=np.clip((1-(d[idx]/h)**2)**2,1e-3,None)
    return RF(n_estimators=TR,min_samples_leaf=3,max_features=.5,n_jobs=1,random_state=s).fit(X[idx],y[idx],sample_weight=w),
def local_pred(tr,te,bw,s=7):
    xy=tr[['x_km','y_km']].values*1000; cl=KMeans(K_A,n_init=3,random_state=s).fit(xy); an=cl.cluster_centers_; X=tr[STAT].values; y=tr.y.values
    F=Parallel(n_jobs=-1)(delayed(fit_one)(a,X,y,xy,bw,s) for a in an); near=cKDTree(an).query(te[['x_km','y_km']].values*1000)[1]; out_=np.full(len(te),np.nan)
    for j in np.unique(near): m=near==j; out_[m]=F[j][0].predict(te[STAT].values[m])
    return out_
L_p=np.full(len(A),np.nan); chosen=[]
for k in range(5):
    tr=A[A.fold!=k]; te=A[A.fold==k]; tr,te=fill(tr,te,STAT); ib=tr.block.drop_duplicates().sample(frac=1,random_state=13).values; ifo=tr.block.map({b:i%3 for i,b in enumerate(ib)}); sc={}
    for bw in BWS:
        pr=[];yt=[]
        for i in range(3):
            a=tr[ifo!=i]; b=tr[ifo==i]; pr+=list(local_pred(a,b,bw)); yt+=list(b.y)
        sc[bw]=r2_score(yt,pr)
    w=max(sc,key=sc.get); chosen.append(w); L_p[(A.fold==k).values]=local_pred(tr,te,w); print('fold',k,'neighbourhood scores',{a:round(b,3) for a,b in sc.items()},'chosen',w,flush=True)
A['g']=G_p; A['l']=L_p; A[['y','g','l','fold']].to_csv(f'{od}/predictions_where.csv')
bl=A.block.values; yv=A.y.values; ub_,ag=r2block(yv,G_p,bl); _,al=r2block(yv,L_p,bl); N=len(ub_); ii=np.random.default_rng(7).integers(0,N,(BOOT,N)); dG=np.array([sumr2(ag[x])for x in ii]); dL=np.array([sumr2(al[x])for x in ii]); ci=lambda z:[round(float(v),3) for v in np.percentile(z,[2.5,97.5])]
print('where model: global R2 %.3f %s RMSE %.3f | local R2 %.3f %s RMSE %.3f | local minus global R2 %.3f %s | windows chosen %s'%(r2_score(yv,G_p),ci(dG),np.sqrt(mean_squared_error(yv,G_p)),r2_score(yv,L_p),ci(dL),np.sqrt(mean_squared_error(yv,L_p)),r2_score(yv,L_p)-r2_score(yv,G_p),ci(dL-dG),chosen),flush=True)
xy=A[['x_km','y_km']].values*1000; nb=cKDTree(xy).query(xy,k=9)[1][:,1:]; n=len(A)
def moran(z): z=z-z.mean(); return np.sum(z*z[nb].sum(1))/np.sum(z*z)/8
r=np.random.default_rng(7)
for name,z in [('target intensity',yv),('global residual',yv-G_p),('local residual',yv-L_p)]:
    o=moran(z); pm=[moran(r.permutation(z)) for _ in range(999)]; print('Moran I',name,round(o,3),'p',(np.sum(np.array(pm)>=o)+1)/1000)
q=[]
for c in ['soil_ec','dist_river_m','area_ac','elev_rel_m','soil_claytotal']:
    b=pd.qcut(A[c],4,duplicates='drop'); q.append(A.groupby(b).y.mean().round(3).rename(c).reset_index(drop=True))
print('mean intensity by quartile (low to high):',pd.concat(q,axis=1).to_string())
# Panel: this year's intensity given last year
crop={'P':'P','F':'F','T':'T','G':'G','C':'C','D':'C','V':'C','YP':'C','I':'I','X':'I','U':'O','UL':'O'}
P=I.merge(C,on=['field_id','wy'],how='left'); P['cl']=P.dwr_class.map(crop).fillna('O'); pv=P.pivot(index='field_id',columns='wy',values='fallow'); pv2=P.pivot(index='field_id',columns='wy',values='intensity'); pc=P.pivot(index='field_id',columns='wy',values='cl'); rows=[]
for y in range(2019,2024):
    t=pd.DataFrame({'y':pv2[y],'prior_fallow':pv[y-1],'prior_intensity':pv2[y-1],'prior_class':pc[y-1]}).dropna(); t['wy']=y; rows.append(t.reset_index())
B=pd.concat(rows).join(R[STAT+['fold','block']],on='field_id').dropna(subset=['fold']); cats=pd.get_dummies(B.prior_class,prefix='prior_crop').astype(float); B=pd.concat([B,cats],axis=1); CC=list(cats.columns); F2=STAT+['prior_fallow']+CC
P_b=np.full(len(B),np.nan); imp=[]
for k in range(5):
    for wy in range(2019,2024):
        tr=B[(B.fold!=k)&(B.wy!=wy)]; te=B[(B.fold==k)&(B.wy==wy)]
        if not len(te): continue
        tr,te=fill(tr,te,STAT); m=RF(n_estimators=200,min_samples_leaf=3,max_features=.5,n_jobs=-1,random_state=7).fit(tr[F2],tr.y); pr=m.predict(te[F2]); P_b[(B.fold.values==k)&(B.wy.values==wy)]=pr
        if k==0:
            base=r2_score(te.y,pr); rng=np.random.default_rng(wy)
            for c in F2:
                X=te[F2].copy(); X[c]=rng.permutation(X[c].values); imp.append(dict(feature=c,dR2=base-r2_score(te.y,m.predict(X))))
print('Year-to-year model (year and block held out): R2 %.3f RMSE %.3f | persistence baseline R2 %.3f'%(r2_score(B.y,P_b),np.sqrt(mean_squared_error(B.y,P_b)),r2_score(B.y,B.prior_intensity)),flush=True)
ImB=pd.DataFrame(imp).groupby('feature').dR2.mean().sort_values(ascending=False); ImB.round(4).to_csv(f'{od}/importance_year_to_year.csv'); print(ImB.round(4).head(8).to_string())
SURF,INK2,GR='#fcfcfb','#52514e','#e4e3df'; plt.rcParams.update({'font.size':11,'axes.edgecolor':GR,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'figure.facecolor':SURF,'axes.facecolor':SURF,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,'ytick.left':False})
fig,ax=plt.subplots(figsize=(8.5,5.2)); v=Imp.iloc[::-1]; ax.barh(range(len(v)),v['mean'],color='#2a78d6',height=.65); ax.set_yticks(range(len(v))); ax.set_yticklabels(v.index); ax.set_xlabel('Drop in held-out R² when the variable is shuffled'); ax.grid(axis='x',color=GR); fig.suptitle('Which physical drivers help predict where fallow concentrates',x=.02,ha='left',fontsize=13,weight='medium'); fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(f'{od}/fig_driver_importance.png',dpi=170)
