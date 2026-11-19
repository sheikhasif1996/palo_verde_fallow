"""Compare candidate PVID fallow layers against DWR 2021 field boundaries. usage: script SRC_DIR DWR_SHP OUT_CSV"""
import sys, warnings
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject, Resampling
from pyproj import datadir
warnings.filterwarnings('ignore')
src, shp, out = sys.argv[1:4]
print('PROJ', datadir.get_data_dir())
for t in ['manually_digitized.tif','pg-20_modified.tif','pg_21-21_modified.tif']:
    with rasterio.open(f'{src}/{t}') as r:
        a = r.read(1, masked=True); v, c = np.unique(a.compressed(), return_counts=True)
        print(t, r.crs, r.res, r.shape, r.count, r.dtypes[0], 'nodata', r.nodata, 'bounds', tuple(round(x) for x in r.bounds))
        print('   values', dict(list(zip(v.tolist(), c.tolist()))[:8]), 'n_uniq', len(v))
aoi = gpd.read_file(f'{src}/Palo_verde_full_AOI.shp').to_crs(32611); A = aoi.geometry.iloc[0].buffer(0)
x0,y0,x1,y1 = A.bounds; res=10; W=int((x1-x0)/res)+1; H=int((y1-y0)/res)+1; T=from_origin(x0,y1,res,res)
d = gpd.read_file(shp, bbox=tuple(gpd.GeoSeries([A],crs=32611).to_crs(4269).total_bounds)).to_crs(32611); d=d[d.intersects(A)]
idle = d[d.SYMB_CLASS.isin(['I','X'])]
dr = rasterize(((g,1) for g in idle.geometry), (H,W), transform=T, dtype='uint8')
fr = rasterize(((g,1) for g in d.geometry), (H,W), transform=T, dtype='uint8')
rows=[]
def score(name,m):
    best=(-1,0,0); s=np.arange(-60,61,4)   # +-600 m at 40 m steps
    for dx in s:
        for dy in s:
            sh=np.roll(np.roll(m,dy,0),dx,1); i=(sh&dr).sum(); u=(sh|dr).sum(); iou=i/u
            if iou>best[0]: best=(iou,dx,dy)
    base=(m&dr).sum()/(m|dr).sum()
    rows.append(dict(layer=name, acres=round(m.sum()*100/4046.86), pct_in_any_dwr_field=round(100*(m&fr).sum()/m.sum(),1), pct_on_dwr_idle=round(100*(m&dr).sum()/m.sum(),1), iou_vs_dwr_idle=round(base,3), best_iou=round(best[0],3), best_dx_m=int(best[1]*res), best_dy_m=int(best[2]*res)))
for n,f in [('att1_raw',f'{src}/pvid_fallow_shapefiles/attachment1_fallow_2021-01-01.shp'),('att1_aligned',f'{src}/attachment1_fallow_2021-01-01_aligned.gpkg'),('att2_raw',f'{src}/pvid_fallow_shapefiles/attachment2_fallow_2021-12-31.shp'),('att2_aligned',f'{src}/attachment2_fallow_2021-12-31_aligned.gpkg')]:
    g=gpd.read_file(f).to_crs(32611); score(n, rasterize(((x,1) for x in g.geometry if x is not None), (H,W), transform=T, dtype='uint8'))
for n,t in [('manually_digitized.tif','manually_digitized.tif'),('pg-20_modified.tif','pg-20_modified.tif'),('pg_21-21_modified.tif','pg_21-21_modified.tif')]:
    with rasterio.open(f'{src}/{t}') as r:
        if r.crs is None: print(t,'no CRS, skipped'); continue
        a=r.read(1); nd=r.nodata; m0=((a>0)&(a!=nd if nd is not None else True)).astype('uint8')
        dst=np.zeros((H,W),'uint8'); reproject(m0,dst,src_transform=r.transform,src_crs=r.crs,dst_transform=T,dst_crs='EPSG:32611',resampling=Resampling.nearest)
        score(n,dst)
df=pd.DataFrame(rows); df.to_csv(out,index=False); print(df.to_string(index=False)); print('DWR idle+X acres in AOI',round(idle.area.sum()/4046.86))
