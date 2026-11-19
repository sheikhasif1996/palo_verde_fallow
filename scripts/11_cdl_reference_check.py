"""Can CDL serve as an independent fallow reference? Field-level class shares from CDL, scored on PVID 2021 first.
usage: 11_cdl_reference_check.py FIELDS CDL_DIR OUTPUTS"""
import sys, warnings
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from sklearn.metrics import f1_score, cohen_kappa_score, recall_score, precision_score, accuracy_score
warnings.filterwarnings('ignore')
fg, cdl, out = sys.argv[1:4]
NAMES={61:'Fallow/Idle',65:'Barren(65)',131:'Barren',152:'Shrubland',64:'Shrubland(64)',176:'Grass/Pasture',36:'Alfalfa',37:'Other Hay',2:'Cotton',24:'Winter Wheat',21:'Barley',228:'Dbl',121:'Dev open',122:'Dev low',123:'Dev med',124:'Dev high',111:'Water',190:'Woody wet',195:'Herb wet',82:'Developed',83:'Water(83)'}
NONCROP=[61,63,64,65,81,82,83,87,88,92,111,112,121,122,123,124,131,141,142,143,152,190,195]   # CDL non-agricultural codes plus Fallow/Idle; Grass/Pasture tested separately
g=gpd.read_file(fg); g=g[g.label!='non_crop'].copy(); g=g.set_index('field_id')
x0,y0,x1,y1=g.total_bounds; res=30; W=int((x1-x0)/res)+2; H=int((y1-y0)/res)+2; T=from_origin(x0-res,y1+res,res,res)
b=g.copy(); b['geometry']=b.buffer(-30); b=b[~b.is_empty]
ids=rasterize(zip(b.geometry,b.index),(H,W),transform=T,dtype='int32'); m=ids>0
def shares(y):
    res_=10 if y>=2024 else 30
    with rasterio.open(f'{cdl}/{y}_{res_}m_cdls/{y}_{res_}m_cdls.tif') as s, WarpedVRT(s,crs='EPSG:32611',transform=T,width=W,height=H,resampling=Resampling.nearest) as v: a=v.read(1)
    d=pd.DataFrame({'field_id':ids[m],'c':a[m]}); t=d.groupby(['field_id','c']).size().rename('n').reset_index(); t['total']=t.groupby('field_id').n.transform('sum'); t['share']=t.n/t.total; t['year']=y; return t
Y=[2018,2020,2021,2022,2023]; S={y:shares(y) for y in Y}
s=S[2021].join(g[['label','area_ac']],on='field_id'); s['ac']=s.share*s.area_ac
print('2021 share of acres by CDL class within PVID-based group (top classes):')
top=s.groupby('c').ac.sum().sort_values(ascending=False).head(12).index
pv=s[s.c.isin(top)].pivot_table(index='label',columns='c',values='ac',aggfunc='sum',fill_value=0); tot=s.groupby('label').ac.sum(); pv=(pv.div(tot,axis=0)*100).round(1); pv.columns=[NAMES.get(c,str(c)) for c in pv.columns]; print(pv.to_string()); pv.to_csv(f'{out}/cdl_vs_pvid_groups_2021.csv')
def ref(y):
    t=S[y]; p=t.pivot_table(index='field_id',values='share',columns='c',aggfunc='sum',fill_value=0)
    f=lambda L:p[[c for c in L if c in p.columns]].sum(axis=1) if any(c in p.columns for c in L) else 0*p.iloc[:,0]
    return pd.DataFrame({'A_fallow61':f([61]),'B_noncrop':f(NONCROP),'C_noncrop_grass':f(NONCROP+[176])})
r21=ref(2021).join(g[['label']]); e=r21[r21.label.isin(['fallow_all_year','cropped'])]; y=(e.label=='fallow_all_year').astype(int).values
print('\nCDL 2021 vs PVID all-year fallow (n=%d) and cropped (n=%d), field called fallow if class share >= 0.5:'%(y.sum(),(1-y).sum()))
for c in ['A_fallow61','B_noncrop','C_noncrop_grass']:
    h=(e[c].values>=.5).astype(int); print(f'  {c:17s} recall {recall_score(y,h):.3f} precision {precision_score(y,h,zero_division=0):.3f} F1 {f1_score(y,h):.3f} predicted {int(h.sum())}')
dev=pd.read_csv(f'{out}/after180/fallow_intensity_wy2018_2022.csv.gz'); h23=pd.read_csv(f'{out}/holdout_2023_field_results.csv.gz')
P=pd.concat([dev.rename(columns={'RF optical':'p'})[['field_id','wy','p','dwr']], h23.rename(columns={'frozen':'p'}).assign(wy=2023)[['field_id','wy','p','dwr']]])
rows=[]
for y_ in Y:
    q=P[P.wy==y_].set_index('field_id').join(ref(y_)); q=q.dropna(subset=['p','B_noncrop']); h=(q.p>=.5).astype(int).values
    row=dict(year=y_,fields=len(q),frozen_fallow=int(h.sum()))
    for c in ['A_fallow61','B_noncrop','C_noncrop_grass']:
        t=(q[c].values>=.5).astype(int); row[c+'_n']=int(t.sum()); row[c+'_F1_vs_frozen']=round(f1_score(t,h,zero_division=0),3); row[c+'_kappa_vs_frozen']=round(cohen_kappa_score(t,h),3)
    d=q[q.dwr>=0]; t=(d.B_noncrop.values>=.5).astype(int); row['B_noncrop_F1_vs_DWR']=round(f1_score(d.dwr.astype(int).values,t,zero_division=0),3); rows.append(row)
R=pd.DataFrame(rows); R.to_csv(f'{out}/cdl_reference_check.csv',index=False); print('\n'+R.T.to_string())
