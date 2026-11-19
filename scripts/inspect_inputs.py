import sys, glob, warnings
import geopandas as gpd, rasterio
from shapely.geometry import box
from shapely.ops import transform
warnings.filterwarnings('ignore')
dwr_dir, src = sys.argv[1], sys.argv[2]
shp = glob.glob(dwr_dir + '/**/*.shp', recursive=True)[0]; print(shp)
aoi = gpd.read_file(src + '/Palo_verde_full_AOI.shp').to_crs(32611); aoi_g = aoi.geometry.iloc[0].buffer(0)
b = gpd.GeoSeries([aoi_g], crs=32611).to_crs(4326).total_bounds
g = gpd.read_file(shp, bbox=tuple(gpd.GeoSeries([aoi_g], crs=32611).to_crs(gpd.read_file(shp, rows=1).crs).total_bounds))
print(g.crs, len(g)); print(list(g.columns))
g = g.to_crs(32611); g = g[g.intersects(aoi_g)]; print('in AOI', len(g), 'acres', round(g.area.sum()/4046.86))
print(g.drop(columns='geometry').head(3).T.to_string())
for c in ['MAIN_CROP','CROPTYP1','CLASS1','CROPTYP2','SYMB_CLASS','COUNTY','MULTIUSE','DWR_REVISE','EMRG_CROP','YR_PLANTED']:
    if c in g.columns: print('\n', c, g.assign(ac=g.area/4046.86).groupby(c, dropna=False).ac.agg(['size','sum']).round(0).sort_values('sum', ascending=False).head(25).to_string())
for t in ['T11SQS_2021_fallow_prob.tif','T11SQT_2021_fallow_prob.tif']:
    with rasterio.open(src + '/' + t) as r:
        fp = box(*r.bounds); print(t, r.crs, r.res, r.shape, 'AOI within tile km2', round(fp.intersection(aoi_g).area/1e6,1))
