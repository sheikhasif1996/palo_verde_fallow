"""Field-level comparison of fallow products, water years 2018-2023: frozen map, DWR I+X, CDL Fallow/Idle (class 61), PVID report.
usage: 15_product_comparison.py FIELDS CDL_DIR OUTPUTS
CDL is matched by year to the water year of the same number. CDL 2019 onward at 30 m; field called CDL fallow if class 61 covers >=50%.
Agreement scores use fields each product can label. PVID report acres are water toll acres on a calendar year and are shown for context only."""
import sys, glob, warnings
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from sklearn.metrics import f1_score, cohen_kappa_score, recall_score, precision_score
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
fg, cdl, out = sys.argv[1:4]; od=f'{out}/product_comparison'
YEARS=[2018,2019,2020,2021,2022,2023]; REPORT={2018:25948,2019:13876,2020:14880,2021:20039,2022:24045,2023:29343}
g=gpd.read_file(fg); g=g[g.label!='non_crop'].copy().set_index('field_id')
x0,y0,x1,y1=g.total_bounds; res=30; W=int((x1-x0)/res)+2; H=int((y1-y0)/res)+2; T=from_origin(x0-res,y1+res,res,res)
b=g.copy(); b['geometry']=b.buffer(-30); b=b[~b.is_empty]; ids=rasterize(zip(b.geometry,b.index),(H,W),transform=T,dtype='int32'); m=ids>0
def f61(y):
    p=glob.glob(f'{cdl}/{y}_*m_cdls/{y}_*m_cdls.tif')[0]
    with rasterio.open(p) as s, WarpedVRT(s,crs='EPSG:32611',transform=T,width=W,height=H,resampling=Resampling.nearest) as v: a=v.read(1)
    d=pd.DataFrame({'field_id':ids[m],'c':a[m]}); return d.groupby('field_id').c.apply(lambda x:(x==61).mean()).rename('cdl61_frac')
dev=pd.read_csv(f'{out}/after180/fallow_intensity_wy2018_2022.csv.gz').rename(columns={'RF optical':'p'})[['field_id','wy','p','dwr','idle_frac']]
h=pd.read_csv(f'{out}/holdout_2023_field_results.csv.gz').rename(columns={'frozen':'p'}).assign(wy=2023)[['field_id','wy','p','dwr','idle_frac']]
P=pd.concat([dev,h]); P['area_ac']=P.field_id.map(g.area_ac); C=[]
for y in YEARS:
    s=f61(y).reset_index(); s['wy']=y; C.append(s); print('cdl',y,'fields',len(s),'field share with class 61 >=50%%: %.1f%%'%(100*(s.cdl61_frac>=.5).mean()),flush=True)
A=P.merge(pd.concat(C),on=['field_id','wy']); A['mapf']=(A.p>=.5).astype(int); A['cdl']=(A.cdl61_frac>=.5).astype(int); A.to_csv(f'{od}/field_level_products.csv.gz',index=False)
area=[];pairs=[]
for y,q in A.groupby('wy'):
    area.append(dict(wy=y,fields=len(q),map_ac=q.loc[q.mapf==1,'area_ac'].sum(),dwr_ac=(q.area_ac*q.idle_frac).sum(),cdl_ac=(q.area_ac*q.cdl61_frac).sum(),pvid_report_ac=REPORT[y]))
    d=q[q.dwr>=0]
    for n,t,h_,w in [('map vs DWR',d.dwr,d.mapf,d),('CDL vs DWR',d.dwr,d.cdl,d),('map vs CDL',q.cdl,q.mapf,q)]:
        pairs.append(dict(wy=y,pair=n,n=len(w),n_ref_fallow=int(t.sum()),n_other_fallow=int(h_.sum()),F1=f1_score(t,h_,zero_division=0),kappa=cohen_kappa_score(t,h_),recall=recall_score(t,h_,zero_division=0),precision=precision_score(t,h_,zero_division=0)))
Ar=pd.DataFrame(area).round(0).astype(int); Ar.to_csv(f'{od}/area_by_product.csv',index=False); print(Ar.to_string(index=False))
Pa=pd.DataFrame(pairs).round(3); Pa.to_csv(f'{od}/agreement_by_year.csv',index=False); print(Pa.pivot_table(index='wy',columns='pair',values='F1').round(3).to_string()); print(Pa.pivot_table(index='wy',columns='pair',values='kappa').round(3).to_string())
# PVID's administrative record is the one reference independent of all products; 2021 only
q=A[A.wy==2021].join(g[['label','f_jan','f_dec']],on='field_id'); e=q[q.label.isin(['fallow_all_year','cropped'])]; y_=(e.label=='fallow_all_year').astype(int).values
prods={'frozen map (spatial holdout)':e.mapf.values,'DWR I+X (share >=0.7)':(e.idle_frac>=.7).astype(int).values,'CDL Fallow/Idle (share >=0.5)':e.cdl.values}
T2=pd.DataFrame([dict(product=k,fallow_fields=int(y_.sum()),cropped_fields=int((1-y_).sum()),found=int(((v==1)&(y_==1)).sum()),false_fallow=int(((v==1)&(y_==0)).sum()),recall=recall_score(y_,v),precision=precision_score(y_,v,zero_division=0),F1=f1_score(y_,v,zero_division=0)) for k,v in prods.items()]).round(3); T2.to_csv(f'{od}/vs_pvid_2021.csv',index=False); print(T2.to_string(index=False))
SURF,INK,INK2,GR='#fcfcfb','#0b0b0b','#52514e','#e4e3df'; S=[('map_ac','Frozen map','#2a78d6','o'),('dwr_ac','DWR idle + unclassified fallow','#eb6834','s'),('cdl_ac','CDL Fallow/Idle','#1baf7a','^'),('pvid_report_ac','PVID report "Fallow" line (water toll acres)','#eda100','D')]
plt.rcParams.update({'font.size':11,'axes.edgecolor':GR,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'text.color':INK,'figure.facecolor':SURF,'axes.facecolor':SURF,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GR,'axes.axisbelow':True,'xtick.bottom':False,'ytick.left':False})
fig,ax=plt.subplots(figsize=(10,5.6))
for c,l,col,mk in S: ax.plot(Ar.wy,Ar[c],color=col,lw=2,marker=mk,ms=8,mec=SURF,mew=1.5,label=l)
ax.set_ylim(0); ax.set_xticks(YEARS); ax.set_ylabel('Fallow area (acres)'); ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
fig.suptitle('CDL shows little fallow after 2018, while the map follows DWR',x=.02,ha='left',fontsize=15,weight='medium')
fig.text(.02,.90,'Palo Verde fields, water years 2018 to 2023. PVID report totals use a different unit and calendar year.',color=INK2)
ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.09),ncol=2); fig.tight_layout(rect=(0,0,1,.88)); fig.savefig(f'{od}/fig_product_comparison.png',dpi=170)
