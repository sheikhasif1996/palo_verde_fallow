"""Step 3: per-field 2021 time series from Planetary Computer (Sentinel-2 L2A, Sentinel-1 RTC, Landsat 8 C2 L2).
usage: 02_extract_field_timeseries.py FIELDS_GPKG OUT_DIR [--limit N] [--start 2021-01-01 --end 2021-12-31]
Field means are computed inside a 30 m inward buffer on a 20 m EPSG:32611 grid. Each scene writes one
part file, so an interrupted run resumes where it stopped."""
import sys, os, time, argparse, warnings, traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.enums import Resampling as R
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
import planetary_computer as pc, pystac_client as ps
warnings.filterwarnings('ignore')
a = argparse.ArgumentParser(); a.add_argument('fields'); a.add_argument('out'); a.add_argument('--limit',type=int); a.add_argument('--start',default='2021-01-01'); a.add_argument('--end',default='2021-12-31'); a.add_argument('--workers',type=int,default=8)
A = a.parse_args(); os.makedirs(A.out, exist_ok=True)
ENV = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_MAX_RETRY='4', GDAL_HTTP_RETRY_DELAY='3', VSI_CACHE='TRUE', GDAL_NUM_THREADS='1')
RES = 20
f = gpd.read_file(A.fields); f = f[f.label!='non_crop'].copy()
f['geometry'] = f.buffer(-30); f = f[~f.is_empty & f.geometry.notna()]
x0,y0,x1,y1 = f.total_bounds; W=int((x1-x0)/RES)+2; H=int((y1-y0)/RES)+2; T=from_origin(x0-RES,y1+RES,RES,RES)
ids = rasterize(zip(f.geometry, f.field_id), (H,W), transform=T, dtype='int32'); N = int(ids.max())+1
total = np.bincount(ids.ravel(), minlength=N)
print(f'{len(f)} fields, grid {W}x{H}', flush=True)

def read(href, rs, nd):
    with rasterio.Env(**ENV), rasterio.open(href) as s:
        with WarpedVRT(s, crs='EPSG:32611', transform=T, width=W, height=H, resampling=rs, src_nodata=nd, nodata=nd) as v:
            return v.read(1)
def zonal(values, mask):
    m = mask & (ids>0)
    n = np.bincount(ids[m], minlength=N)
    out = {k: np.bincount(ids[m], weights=v[m].astype('float64'), minlength=N)/np.maximum(n,1) for k,v in values.items()}
    keep = np.where(n>0)[0]; df = pd.DataFrame({k:v[keep] for k,v in out.items()})
    df.insert(0,'n_valid',n[keep]); df.insert(0,'n_total',total[keep]); df.insert(0,'field_id',keep); return df

def s2(item):
    s = pc.sign(item); b = {k: read(s.assets[k].href, R.average, 0) for k in ['B04','B08','B11','B12']}
    scl = read(s.assets['SCL'].href, R.nearest, 0)
    m = np.isin(scl,[4,5,6,7]) & np.all([v>0 for v in b.values()],axis=0)
    off = 1000. if float(item.properties['s2:processing_baseline']) >= 4.0 else 0.   # BOA_ADD_OFFSET from baseline 04.00
    d = zonal({k: np.clip((v.astype('float64')-off)/10000., 1e-4, None) for k,v in b.items()}, m); return d.assign(tile=item.properties['s2:mgrs_tile'])
def s1(item):
    s = pc.sign(item); vv = read(s.assets['vv'].href, R.average, -32768); vh = read(s.assets['vh'].href, R.average, -32768)
    m = (vv>0)&(vh>0)&(vv!=-32768)&(vh!=-32768)
    d = zonal({'vv':vv,'vh':vh}, m); p=item.properties
    return d.assign(orbit=p['sat:orbit_state'], rel_orbit=p['sat:relative_orbit'], platform=p['platform'])
def ls(item):
    s = pc.sign(item); t = read(s.assets['lwir11'].href, R.bilinear, 0); q = read(s.assets['qa_pixel'].href, R.nearest, 1)
    bad = (q & 0b11110) > 0   # dilated cloud, cirrus, cloud, shadow
    m = (t>0)&(q!=1)&~bad
    return zonal({'lst_c': t*0.00341802+149.0-273.15}, m).assign(wrs=f"{item.properties['landsat:wrs_path']}{item.properties['landsat:wrs_row']}")

c = ps.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1')
bb = list(gpd.GeoSeries(f.geometry, crs=32611).to_crs(4326).total_bounds); dt=f'{A.start}/{A.end}'
S2 = [i for i in c.search(collections=['sentinel-2-l2a'],bbox=bb,datetime=dt,query={'eo:cloud_cover':{'lte':80}}).items() if i.properties['s2:mgrs_tile'] in ('11SQT','11SQS')]
S2 = list({'_'.join(i.id.split('_')[:-1]): i for i in sorted(S2, key=lambda i: i.id)}.values())   # newest reprocessing of each acquisition
S1 = [i for i in c.search(collections=['sentinel-1-rtc'],bbox=bb,datetime=dt).items() if i.properties['platform']=='SENTINEL-1A' and i.properties['sat:relative_orbit'] in (100,173)]   # the only orbits available in every year
LS = [i for i in c.search(collections=['landsat-c2-l2'],bbox=bb,datetime=dt,query={'platform':{'eq':'landsat-8'}}).items()]
print('scenes', len(S2), len(S1), len(LS), flush=True)
mf = pd.DataFrame([dict(sensor=t_, scene=i.id, datetime=i.datetime.isoformat(), baseline=i.properties.get('s2:processing_baseline')) for t_,L_ in (('s2',S2),('s1',S1),('ls',LS)) for i in L_])
mp = f'{A.out}/../scene_manifest.csv'
if os.path.exists(mp): mf = pd.concat([pd.read_csv(mp), mf]).drop_duplicates('scene', keep='last')
mf.to_csv(mp, index=False)
jobs = [('s2',s2,i) for i in S2[:A.limit]] + [('s1',s1,i) for i in S1[:A.limit]] + [('ls',ls,i) for i in LS[:A.limit]]
def run(tag,fn,item):
    p = f'{A.out}/{tag}_{item.id}.csv.gz'
    if os.path.exists(p): return 'cached'
    for k in range(3):
        try:
            t0=time.time(); d = fn(item).assign(datetime=item.datetime.isoformat(), scene=item.id)
            d.to_csv(p+'.tmp', index=False, compression='gzip'); os.replace(p+'.tmp', p); return f'{len(d)} fields {time.time()-t0:.0f}s'
        except Exception as e:
            err=repr(e); time.sleep(5*(k+1))
    return 'FAILED '+err
fails=0
with ThreadPoolExecutor(A.workers) as ex:
    fut={ex.submit(run,*j):j for j in jobs}
    for n,x in enumerate(as_completed(fut),1):
        r=x.result(); fails+=r.startswith('FAILED')
        if r.startswith('FAILED') or n%25==0 or A.limit: print(n,len(jobs),fut[x][0],fut[x][2].id,r,flush=True)
print('done', len(jobs), 'failed', fails, flush=True)
