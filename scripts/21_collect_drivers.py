"""Collect field-level driver data. usage: 21_collect_drivers.py FIELDS RAW_DIR DWR_BASE OUT_DIR
Soils: USDA SSURGO via Soil Data Access (0-30 cm, major components weighted by component percent, then by area).
Water: USGS NHD large-scale flowlines. Elevation: USGS 3DEP 1/3 arc-second via Planetary Computer. Shape from polygons. Crop from DWR."""
import sys, os, json, glob, time, warnings
import numpy as np, pandas as pd, geopandas as gpd, requests, rasterio
from shapely import wkt
from shapely.geometry import box
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
import planetary_computer as pc, pystac_client as ps
warnings.filterwarnings('ignore')
fg, raw, base, od = sys.argv[1:5]; SDA='https://sdmdataaccess.sc.egov.usda.gov/Tabular/post.rest'; AC=4046.8564
g=gpd.read_file(fg); g=g[g.label!='non_crop'].copy(); g=g[['field_id','area_ac','geometry']].copy().set_index('field_id')
bb=g.to_crs(4326).total_bounds+np.array([-.01,-.01,.01,.01]); print('fields',len(g),'bbox',bb.round(3),flush=True)
def sda(q):
    for k in range(4):
        try:
            r=requests.post(SDA,json={'format':'JSON+COLUMNNAME','query':q},timeout=180); r.raise_for_status(); t=r.json().get('Table',[])
            return pd.DataFrame(t[1:],columns=t[0]) if t else pd.DataFrame()
        except Exception as e: err=e; time.sleep(8*(k+1))
    raise RuntimeError(err)
xs=np.linspace(bb[0],bb[2],3); ys=np.linspace(bb[1],bb[3],3); P=[]
for i in range(2):
    for j in range(2):
        w=f'POLYGON(({xs[i]} {ys[j]},{xs[i+1]} {ys[j]},{xs[i+1]} {ys[j+1]},{xs[i]} {ys[j+1]},{xs[i]} {ys[j]}))'
        P.append(sda(f"SELECT mupolygonkey, mukey, mupolygongeo.STAsText() AS w FROM mupolygon WHERE mupolygongeo.STIntersects(geometry::STGeomFromText('{w}',4326))=1")); print('soil tile',i,j,len(P[-1]),flush=True)
M=pd.concat(P).drop_duplicates('mupolygonkey'); M['geometry']=M.w.map(wkt.loads); M=gpd.GeoDataFrame(M.drop(columns='w'),geometry='geometry',crs=4326); M['mukey']=M.mukey.astype(str); M.to_file(f'{raw}/ssurgo/mupolygon.gpkg',driver='GPKG'); print('soil polygons',len(M),'map units',M.mukey.nunique(),flush=True)
ms=sorted(M.mukey.unique()); A=[];H=[]
for i in range(0,len(ms),300):
    q=','.join(f"'{m}'" for m in ms[i:i+300])
    A.append(sda(f"SELECT mukey, muname, drclassdcd, hydgrpdcd, flodfreqdcd, iccdcd, aws050wta, aws0150wta FROM muaggatt WHERE mukey IN ({q})"))
    H.append(sda(f"SELECT c.mukey, c.cokey, c.comppct_r, h.hzdept_r, h.hzdepb_r, h.claytotal_r, h.sandtotal_r, h.om_r, h.ec_r, h.sar_r, h.ksat_r, h.awc_r FROM component c INNER JOIN chorizon h ON c.cokey=h.cokey WHERE c.majcompflag='Yes' AND c.mukey IN ({q})"))
A=pd.concat(A); H=pd.concat(H); A.to_csv(f'{raw}/ssurgo/muaggatt.csv',index=False); H.to_csv(f'{raw}/ssurgo/horizons.csv',index=False)
NUM=['claytotal_r','sandtotal_r','om_r','ec_r','sar_r','ksat_r','awc_r']; H[['comppct_r','hzdept_r','hzdepb_r']+NUM]=H[['comppct_r','hzdept_r','hzdepb_r']+NUM].apply(pd.to_numeric,errors='coerce')
H['ov']=(H.hzdepb_r.clip(upper=30)-H.hzdept_r.clip(lower=0)).clip(lower=0); H=H[H.ov>0]
def agg(d):
    return pd.Series({c:np.average(d[c].dropna(),weights=d.ov[d[c].notna()]) if d[c].notna().any() else np.nan for c in NUM})
C=H.groupby(['mukey','cokey','comppct_r']).apply(agg).reset_index()
def agg2(d):
    return pd.Series({c:np.average(d[c].dropna(),weights=d.comppct_r[d[c].notna()]) if d[c].notna().any() and d.comppct_r[d[c].notna()].sum()>0 else np.nan for c in NUM})
U=C.groupby('mukey').apply(agg2).reset_index(); U['mukey']=U.mukey.astype(str); A['mukey']=A.mukey.astype(str); A[['aws050wta','aws0150wta']]=A[['aws050wta','aws0150wta']].apply(pd.to_numeric,errors='coerce'); A['icc_num']=pd.to_numeric(A.iccdcd,errors='coerce')
S=M.merge(A,on='mukey',how='left').merge(U,on='mukey',how='left').to_crs(32611)
ov=gpd.overlay(g.reset_index()[['field_id','geometry']],S[['mukey','geometry']],how='intersection',keep_geom_type=True); ov['ha']=ov.area; ov=ov.merge(S.drop(columns='geometry').drop_duplicates('mukey'),on='mukey',how='left')
soil_num=NUM+['aws050wta','aws0150wta','icc_num']; R=pd.DataFrame(index=g.index)
for c in soil_num:
    t=ov[ov[c].notna()]; R['soil_'+c.replace('_r','')]=(t[c]*t.ha).groupby(t.field_id).sum()/t.groupby('field_id').ha.sum()
for c in ['drclassdcd','hydgrpdcd','flodfreqdcd']:
    t=ov[ov[c].notna()]; R['soil_'+c]=t.loc[t.groupby(['field_id',c]).ha.transform('sum').groupby(t.field_id).idxmax(),['field_id',c]].set_index('field_id')[c]
R['soil_coverage']=(ov.groupby('field_id').ha.sum()/g.area).clip(upper=1); print('soil coverage of fields: median',round(R.soil_coverage.median(),3),'fields with no soil',int(R.soil_coverage.isna().sum()),flush=True)
# NHD flowlines
U_='https://hydro.nationalmap.gov/arcgis/rest/services/nhd/MapServer/6/query'; F=[]; off=0
while True:
    r=requests.get(U_,params={'where':'1=1','geometry':f'{bb[0]},{bb[1]},{bb[2]},{bb[3]}','geometryType':'esriGeometryEnvelope','inSR':4326,'outSR':32611,'spatialRel':'esriSpatialRelIntersects','outFields':'ftype,fcode,gnis_name','returnGeometry':'true','f':'geojson','resultOffset':off,'resultRecordCount':1000},timeout=120); r.raise_for_status(); j=r.json(); n=len(j.get('features',[]))
    if n==0: break
    F+=j['features']; off+=n
    if n<1000: break
L=gpd.GeoDataFrame.from_features(F,crs=32611); L.to_file(f'{raw}/nhd/flowlines.gpkg',driver='GPKG'); print('NHD flowlines',len(L),L.ftype.value_counts().to_dict(),'named',L.gnis_name.notna().sum(),flush=True); print(L.gnis_name.value_counts().head(15).to_dict())
d=lambda q:g.geometry.distance(q.geometry.union_all()).values
can=L[L.ftype==336]; riv=L[L.gnis_name.fillna('').str.contains('Colorado River')]; print('canal/ditch segments',len(can),'river segments',len(riv),'named Drain',int(can.gnis_name.fillna('').str.contains('Drain').sum()),flush=True)
R['dist_canal_m']=d(can); R['dist_river_m']=d(riv)
dr=can[can.gnis_name.fillna('').str.contains('Drain')]
if len(dr)>5: R['dist_named_drain_m']=d(dr)
# shape
R['area_ac']=g.area_ac; R['perimeter_m']=g.length; R['compactness_pp']=4*np.pi*g.area/g.length**2
mr=g.geometry.minimum_rotated_rectangle(); R['rectangularity']=g.area/mr.area
c=g.centroid; R['x_km']=c.x/1000; R['y_km']=c.y/1000
# elevation
it=[i for i in ps.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1').search(collections=['3dep-seamless'],bbox=list(bb)).items() if i.properties.get('gsd')==10][0]
x0,y0,x1,y1=g.total_bounds; res=30; W=int((x1-x0)/res)+2; H_=int((y1-y0)/res)+2; T=from_origin(x0-res,y1+res,res,res)
with rasterio.open(pc.sign(it).assets['data'].href) as s, WarpedVRT(s,crs='EPSG:32611',transform=T,width=W,height=H_,resampling=Resampling.bilinear) as v: Z=v.read(1,masked=True).filled(np.nan).astype('float64')
dy,dx=np.gradient(Z,res); sl=np.degrees(np.arctan(np.hypot(dx,dy))); ids=rasterize(zip(g.geometry,g.index),(H_,W),transform=T,dtype='int32'); m=(ids>0)&np.isfinite(Z); e=pd.DataFrame({'f':ids[m],'z':Z[m],'s':sl[m]}).groupby('f').agg(elev_m=('z','mean'),elev_sd_m=('z','std'),slope_deg=('s','mean'))
R=R.join(e); fit=R.dropna(subset=['elev_m']); A_=np.c_[fit.x_km,fit.y_km,np.ones(len(fit))]; co=np.linalg.lstsq(A_,fit.elev_m,rcond=None)[0]; R['elev_rel_m']=R.elev_m-(co[0]*R.x_km+co[1]*R.y_km+co[2]); print('valley plane fit',co.round(3),flush=True)
# DWR crop class per field-year
x0,y0,x1,y1=g.total_bounds; res=10; W=int((x1-x0)/res)+2; H_=int((y1-y0)/res)+2; T=from_origin(x0-res,y1+res,res,res); ids=rasterize(zip(g.geometry,g.index),(H_,W),transform=T,dtype='int32'); mk=ids>0; Crop=[]
for y in range(2018,2025):
    s=glob.glob(f'{base}/dwr_{y}/shp/**/*.shp',recursive=True)[0]; crs=gpd.read_file(s,rows=1).crs; q=gpd.read_file(s,bbox=tuple(g.to_crs(crs).total_bounds)).to_crs(32611); k=next(c for c in q.columns if c.upper()=='SYMB_CLASS'); cl=sorted(q[k].dropna().unique()); code={v:i+1 for i,v in enumerate(cl)}; q=q[q.geometry.notna()&~q.geometry.is_empty]
    arr=rasterize(((a,code[b]) for a,b in zip(q.geometry,q[k])),(H_,W),transform=T,dtype='uint8'); t=pd.DataFrame({'f':ids[mk],'c':arr[mk]}); t=t[t.c>0]; m_=t.groupby(['f','c']).size().rename('n').reset_index().sort_values('n').groupby('f').tail(1); rev={v:k_ for k_,v in code.items()}; Crop.append(pd.DataFrame({'field_id':m_.f.values,'wy':y,'dwr_class':m_.c.map(rev).values}))
Crop=pd.concat(Crop); Crop.to_csv(f'{od}/field_dwr_class_by_year.csv.gz',index=False); print('DWR crop rows',len(Crop),flush=True)
G=g.drop(columns='area_ac').join(R); G.reset_index().to_file(f'{od}/field_drivers.gpkg',driver='GPKG'); R.to_csv(f'{od}/field_drivers.csv')
print(R.describe().T[['count','mean','min','50%','max']].round(3).to_string()); print(R.isna().sum()[lambda x:x>0].to_string()); print(R[['soil_drclassdcd','soil_hydgrpdcd']].apply(lambda x:x.value_counts().to_dict()).to_string())
