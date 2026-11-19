import sys, glob, warnings
import geopandas as gpd
warnings.filterwarnings('ignore')
base = sys.argv[1]; bb = (-114.80,33.28,-114.45,33.78)
for y in [2018,2019,2020,2022]:
    s = glob.glob(f'{base}/dwr_{y}/shp/**/*.shp', recursive=True)[0]
    crs = gpd.read_file(s, rows=1).crs
    b = gpd.GeoSeries.from_xy([bb[0],bb[2]],[bb[1],bb[3]],crs=4326).to_crs(crs).total_bounds
    g = gpd.read_file(s, bbox=tuple(b)); A = g.to_crs(32611).area/4046.86
    c = [x for x in g.columns if x.upper() in ('SYMB_CLASS','CLASS2','MAIN_CROP','CROPTYP2','DWR_REVISE')]
    print(y, s.split('/')[-1], crs.to_epsg(), len(g), 'cols', c)
    k = next(x for x in g.columns if x.upper()=='SYMB_CLASS')
    print('   ', A.groupby(g[k]).sum().round(0).sort_values(ascending=False).to_dict())
