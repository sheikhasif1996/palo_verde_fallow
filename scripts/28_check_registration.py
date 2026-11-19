"""Registration check: overlap of each registered PVID map with field boundaries and DWR idle land, by trial shift. Diagnostic only.
usage: 28_check_registration.py FIELDS DWR_SHP_2023 TIF..."""
import sys, glob, warnings, numpy as np, rasterio, geopandas as gpd
from rasterio.features import rasterize
warnings.filterwarnings('ignore')
fg, dw = sys.argv[1:3]; g=gpd.read_file(fg); g=g[g.label!='non_crop']; s=glob.glob(f'{dw}/**/*.shp',recursive=True)[0]; crs=gpd.read_file(s,rows=1).crs; d=gpd.read_file(s,bbox=tuple(g.to_crs(crs).total_bounds)).to_crs(32611); k=next(c for c in d.columns if c.upper()=='SYMB_CLASS'); idle=d[d[k].isin(['I','X'])&d.geometry.notna()]
for tif in sys.argv[3:]:
    with rasterio.open(tif) as r: Y=r.read(1).astype(bool); T=r.transform; H,W=Y.shape
    F=rasterize(((x,1) for x in g.geometry),(H,W),transform=T,dtype='uint8').astype(bool); D=rasterize(((x,1) for x in idle.geometry),(H,W),transform=T,dtype='uint8').astype(bool); best=(-1,0,0); bF=(-1,0,0)
    for dx in range(-30,31,2):
        for dy in range(-30,31,2):
            q=np.roll(np.roll(Y,dy,0),dx,1); v=(q&D).sum()/(q|D).sum(); w=(q&F).sum()/q.sum()
            if v>best[0]: best=(v,dx,dy)
            if w>bF[0]: bF=(w,dx,dy)
    print(tif.split('/')[-1],'pct in fields %.1f'%(100*(Y&F).sum()/Y.sum()),'best in-field %.3f at shift %s m'%(bF[0],(bF[1]*abs(T.a),bF[2]*abs(T.a))),'| IoU vs DWR idle zero-shift %.3f best %.3f at shift %s m'%((Y&D).sum()/(Y|D).sum(),best[0],(best[1]*abs(T.a),best[2]*abs(T.a))))
