"""Why is the 2024 map 11.5% below PVID net fallow? Diagnostic only; the frozen method is not changed.
usage: 19_check_2024_gap.py FIELDS PARTS MANIFEST DWR_BASE OUTPUTS AOI_SHP"""
import sys, os, glob, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
import numpy as np, pandas as pd, geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin
from sklearn.metrics import f1_score, recall_score, precision_score
import tslib
warnings.filterwarnings('ignore')
fg, parts, mf, base, out = sys.argv[1:6]; AC=4046.8564; IDLE=['I','X']; CROP=list('PFTGCD')+['YP']
g=gpd.read_file(fg); g=g[g.label!='non_crop'].set_index('field_id')
I=pd.read_csv(f'{out}/maps/field_year.csv.gz'); M=pd.read_csv(f'{out}/maps/monthly_state.csv.gz',parse_dates=['date']); M['area_ac']=M.field_id.map(g.area_ac)
x0,y0,x1,y1=g.total_bounds; res=10; W=int((x1-x0)/res)+2; H=int((y1-y0)/res)+2; T=from_origin(x0-res,y1+res,res,res)
ids=rasterize(zip(g.geometry,g.index),(H,W),transform=T,dtype='int32'); N=int(ids.max())+1; n=np.bincount(ids.ravel(),minlength=N).astype(float)
aoi=gpd.read_file(sys.argv[6]).to_crs(32611).geometry.iloc[0].buffer(0)
print('--- A. DWR by year: idle area in mapped fields, and outside them within the AOI')
rows=[]
for y in range(2018,2025):
    s=glob.glob(f'{base}/dwr_{y}/shp/**/*.shp',recursive=True)[0]; crs=gpd.read_file(s,rows=1).crs
    d=gpd.read_file(s,bbox=tuple(gpd.GeoSeries([aoi],crs=32611).to_crs(crs).total_bounds)).to_crs(32611); k=next(c for c in d.columns if c.upper()=='SYMB_CLASS'); d=d[d.geometry.notna()&~d.geometry.is_empty]; d=d[d.intersects(aoi)]
    idle=d[d[k].isin(IDLE)]; m=rasterize(((a,1) for a in idle.geometry),(H,W),transform=T,dtype='uint8'); fi=np.bincount(ids.ravel(),weights=m.ravel(),minlength=N)/np.maximum(n,1)
    in_ac=float((m[ids>0].sum())*100/AC); tot=float(idle.area.sum()/AC); cr=rasterize(((a,1) for a in d[d[k].isin(CROP)].geometry),(H,W),transform=T,dtype='uint8'); fc=np.bincount(ids.ravel(),weights=cr.ravel(),minlength=N)/np.maximum(n,1)
    q=I[I.wy==y].set_index('field_id'); q['idle_frac']=fi[q.index]; q['crop_frac']=fc[q.index]; q['dwr']=np.where(q.idle_frac>=.7,1,np.where(q.crop_frac>=.7,0,-1)); v=q[q.dwr>=0]
    rows.append(dict(wy=y,dwr_status=d['DataStatus'].mode().iat[0] if 'DataStatus' in d else '',dwr_idle_total_ac=round(tot),dwr_idle_in_fields_ac=round(in_ac),dwr_idle_outside_fields_ac=round(tot-in_ac),map_ac=round(q.loc[q.fallow==1,'area_ac'].sum()),F1=round(f1_score(v.dwr,v.fallow),3),recall=round(recall_score(v.dwr,v.fallow),3),precision=round(precision_score(v.dwr,v.fallow),3)))
    if y==2024: Q24=q.copy()
D=pd.DataFrame(rows); D.to_csv(f'{out}/analysis/check_2024_dwr.csv',index=False); print(D.to_string(index=False))
print('\n--- B. Acres by annual intensity band'); b=pd.cut(I.intensity,[-.01,0,.25,.4999,.75,1.01],labels=['0','0-0.25','0.25-0.5','0.5-0.75','0.75-1']); t=I.assign(b=b).pivot_table(index='wy',columns='b',values='area_ac',aggfunc='sum').round(0); t['intensity_weighted_ac']=I.assign(a=I.intensity*I.area_ac).groupby('wy').a.sum().round(0); t.to_csv(f'{out}/analysis/check_2024_intensity_bands.csv'); print(t.to_string())
print('\n--- C. Mean monthly mapped fallow acres, by water year quarter'); M['q']=M.date.dt.quarter; v=M[M.state==1].groupby(['date']).area_ac.sum().rename('ac').reset_index(); v['wy']=v.date.dt.year+(v.date.dt.month>=10).astype(int); v['m']=v.date.dt.strftime('%b'); pv=v.pivot(index='wy',columns='m',values='ac')[['Oct','Nov','Dec','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep']].round(0); pv['mean']=pv.mean(1).round(0); pv.to_csv(f'{out}/analysis/check_2024_monthly_acres.csv'); print(pv.to_string())
print('\n--- D. Near-threshold fields in 2024 (intensity 0.25-0.5): DWR says')
n24=Q24[(Q24.intensity>=.25)&(Q24.intensity<.5)]; print('fields',len(n24),'acres',round(n24.area_ac.sum()),'DWR idle',int((n24.dwr==1).sum()),'DWR cropped',int((n24.dwr==0).sum()),'unclear',int((n24.dwr==-1).sum()),'mapped idle acres in those',round((n24.idle_frac*n24.area_ac).sum()))
mis=Q24[(Q24.fallow==0)&(Q24.dwr==1)]; print('Missed vs DWR 2024:',len(mis),'fields',round(mis.area_ac.sum()),'ac, median intensity',round(mis.intensity.median(),3)); mf_=M[(M.wy==2024)&(M.field_id.isin(mis.index))]; print('months those missed fields were mapped fallow',mf_.groupby('date').state.mean().round(2).to_dict())
print('\n--- E. Imagery stability'); s2,_,_=tslib.load_series(parts,mf,'2017-10-01','2025-03-31'); s2['wy']=s2.date.dt.year+(s2.date.dt.month>=10).astype(int); s2=s2[s2.wy.between(2018,2024)]
x=I.groupby('field_id').fallow.sum(); stable=x[x==7].index; crops=x[x==0].index
print('obs per field per year (median):',s2.groupby(['wy','field_id']).size().groupby('wy').median().astype(int).to_dict()); 
print('median NDVI of the fields mapped fallow all 7 years:',s2[s2.field_id.isin(stable)].groupby('wy').ndvi.median().round(3).to_dict())
print('median NDVI of the fields never mapped fallow:',s2[s2.field_id.isin(crops)].groupby('wy').ndvi.median().round(3).to_dict())
fi=(M[M.wy==2024].groupby('date').n_s2.median()); print('median S2 obs per window, WY2024 months:',fi.astype(int).to_dict())
