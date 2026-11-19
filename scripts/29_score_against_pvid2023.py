"""Independent field-level test of the frozen monthly map against PVID's 2023 verification maps. usage: 29_score_against_pvid2023.py FIELDS OUTPUTS DWR_BASE PVID_DIR
Design fixed before scoring. Fallow: >=70% of field on the PVID map. Not fallow, set A: <=5% on map and DWR cropped (same rule as training). Set B: <=5% on map, PVID only, so non-program fallow counts against the map.
Map state is the frozen monthly state: 1 Jan 2023 for the January map, and 1 Jan 2024 for the 31 Dec 2023 map. Primary uses no DWR-based registration; sensitivity applies a small rigid shift."""
import sys, glob, warnings
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.features import rasterize
from sklearn.metrics import f1_score, cohen_kappa_score, recall_score, precision_score, accuracy_score
warnings.filterwarnings('ignore')
fg,out,base,pd_=sys.argv[1:5]; od=f'{out}/validation_pvid2023'; IDLE=['I','X']; CROP=list('PFTGCD')+['YP']
g=gpd.read_file(fg); g=g[g.label!='non_crop'].set_index('field_id'); c=g.centroid; bk=np.floor(c.x/3000).astype(int).astype(str)+'_'+np.floor(c.y/3000).astype(int).astype(str)
M=pd.read_csv(f'{out}/maps/monthly_state.csv.gz',parse_dates=['date'])
def dwr(y):
    s=glob.glob(f'{base}/dwr_{y}/shp/**/*.shp',recursive=True)[0]; crs=gpd.read_file(s,rows=1).crs; q=gpd.read_file(s,bbox=tuple(g.to_crs(crs).total_bounds)).to_crs(32611); k=next(x for x in q.columns if x.upper()=='SYMB_CLASS'); q=q[q.geometry.notna()&~q.geometry.is_empty]; return q,k
def frac(tif,dx=0,dy=0,gd=None):
    with rasterio.open(tif) as r: Y=r.read(1).astype('float32'); T=r.transform; H,W=Y.shape
    if dx or dy: T=rasterio.Affine(T.a,T.b,T.c+dx,T.d,T.e,T.f+dy)
    ids=rasterize(zip(g.geometry,g.index),(H,W),transform=T,dtype='int32'); N=int(ids.max())+1; n=np.bincount(ids.ravel(),minlength=N).astype(float); f=np.bincount(ids.ravel(),weights=Y.ravel(),minlength=N)/np.maximum(n,1); return pd.Series(f[g.index],index=g.index)
def dwr_crop(y):
    q,k=dwr(y); x0,y0,x1,y1=g.total_bounds; res=10; from rasterio.transform import from_origin; W=int((x1-x0)/res)+2; H=int((y1-y0)/res)+2; T=from_origin(x0-res,y1+res,res,res); ids=rasterize(zip(g.geometry,g.index),(H,W),transform=T,dtype='int32'); a=rasterize(((x,1) for x in q[q[k].isin(CROP)].geometry),(H,W),transform=T,dtype='uint8'); N=int(ids.max())+1; n=np.bincount(ids.ravel(),minlength=N).astype(float); f=np.bincount(ids.ravel(),weights=a.ravel(),minlength=N)/np.maximum(n,1); return pd.Series(f[g.index],index=g.index)
C={2023:dwr_crop(2023),2024:dwr_crop(2024)}
rows=[]; bdiff=[]; rec=[]
def score(name,date,wy,tif,dx=0,dy=0,off=None):
    f=frac(tif,dx,dy); s=M[M.date==date].set_index('field_id'); idx=s.index; d=pd.DataFrame({'f':f.reindex(idx),'state':s.state,'rule':(s.ndvi_max<.3).astype(float),'crop':C[wy].reindex(idx),'area':g.area_ac.reindex(idx),'block':bk.reindex(idx)}).dropna()
    for st,neg in [('A: DWR-cropped negatives',(d.f<=.05)&(d.crop>=.7)),('B: PVID-only negatives',d.f<=.05)]:
        u=d[(d.f>=.7)|neg].copy(); y=(u.f>=.7).astype(int).values; h=(u.state>=.5).astype(int).values; r=(u.rule>=.5).astype(int).values
        ub=np.unique(u.block); ix=[np.where(u.block.values==b)[0] for b in ub]; ii=np.random.default_rng(7).integers(0,len(ub),(2000,len(ub))); cn=np.array([[((h[i]==1)&(y[i]==1)).sum(),((h[i]==1)&(y[i]==0)).sum(),((h[i]==0)&(y[i]==1)).sum()] for i in ix]); F=2*cn[ii].sum(1)[:,0]/np.maximum(2*cn[ii].sum(1)[:,0]+cn[ii].sum(1)[:,1]+cn[ii].sum(1)[:,2],1)
        rows.append(dict(map=name,negatives=st,fields=len(u),pvid_fallow=int(y.sum()),TP=int(((h==1)&(y==1)).sum()),FP=int(((h==1)&(y==0)).sum()),FN=int(((h==0)&(y==1)).sum()),OA=round(accuracy_score(y,h),3),kappa=round(cohen_kappa_score(y,h),3),F1=round(f1_score(y,h),3),F1_lo=round(float(np.percentile(F,2.5)),3),F1_hi=round(float(np.percentile(F,97.5)),3),recall=round(recall_score(y,h),3),precision=round(precision_score(y,h,zero_division=0),3),rule_F1=round(f1_score(y,r),3)))
    a=d.loc[d.state>=.5,'area'].sum(); rec.append(dict(map=name,official_ac=off,map_fallow_ac=round(a),pct_of_official=round(100*a/off,1) if off else None,fields_on_pvid_map_ge70=int((d.f>=.7).sum()),fields_partial_5_70=int(((d.f>.05)&(d.f<.7)).sum()),mapped_fallow_not_on_map_fields=int(((d.state>=.5)&(d.f<=.05)).sum())))
J=f'{pd_}/fallow_2023-01-01.tif'; D=f'{pd_}/fallow_2023-12-31.tif'
score('1 Jan 2023 (primary)','2023-01-01',2023,J,off=19476); score('31 Dec 2023 (primary)','2024-01-01',2024,D,off=25947); score('1 Jan 2023 (shift -121 E, -40 N)','2023-01-01',2023,J,-121,-40); score('31 Dec 2023 (shift +40 E)','2024-01-01',2024,D,40,0)
R=pd.DataFrame(rows); R.to_csv(f'{od}/scores.csv',index=False); A=pd.DataFrame(rec); A.to_csv(f'{od}/area_and_coverage.csv',index=False)
pd.set_option('display.width',250); print(R.drop(columns=['F1_lo','F1_hi']).to_string(index=False)); print(R[['map','negatives','F1_lo','F1_hi']].to_string(index=False)); print(A.to_string(index=False))
