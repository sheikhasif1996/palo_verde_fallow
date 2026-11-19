"""Step 2: assign a 2021 fallow label to each DWR field polygon from PVID's digitized snapshots.
usage: 01_make_field_labels.py SRC_DIR DWR_SHP OUT_DIR

Fallow evidence comes only from PVID (administrative). DWR supplies field geometry and the
cropped control / other-idle split, so the fallow label stays independent of DWR's image-based class.
Thresholds are fractions of each DWR field covered by a PVID snapshot.
"""
import sys, warnings
import numpy as np, pandas as pd, geopandas as gpd
from rasterio.features import rasterize
from rasterio.transform import from_origin
warnings.filterwarnings('ignore')
src, shp, out = sys.argv[1:4]
HI, LO = 0.70, 0.05
AC = 4046.8564
A = gpd.read_file(f'{src}/Palo_verde_full_AOI.shp').to_crs(32611).geometry.iloc[0].buffer(0)
d = gpd.read_file(shp, bbox=tuple(gpd.GeoSeries([A], crs=32611).to_crs(4269).total_bounds)).to_crs(32611)
d = d[d.intersects(A)].reset_index(drop=True)
d = d[['UniqueID','SYMB_CLASS','MAIN_CROP','CROPTYP1','CROPTYP2','MULTIUSE','COUNTY','geometry']].copy()
d['field_id'] = np.arange(1, len(d)+1); d['area_ac'] = d.area/AC
x0,y0,x1,y1 = d.total_bounds; res=10; W=int((x1-x0)/res)+2; H=int((y1-y0)/res)+2; T=from_origin(x0-res,y1+res,res,res)
ids = rasterize(zip(d.geometry, d.field_id), (H,W), transform=T, dtype='int32')
n = np.bincount(ids.ravel(), minlength=len(d)+1).astype(float)
jan = gpd.read_file(f'{src}/attachment1_fallow_2021-01-01_aligned.gpkg').to_crs(32611)
dec = gpd.read_file(f'{src}/attachment2_fallow_2021-12-31_aligned.gpkg').to_crs(32611)
def frac(g):
    m = rasterize(((x,1) for x in g.geometry if x is not None and not x.is_empty), (H,W), transform=T, dtype='uint8')
    return np.bincount(ids.ravel(), weights=m.ravel(), minlength=len(d)+1)[1:] / np.maximum(n[1:],1)
d['f_jan'] = frac(jan); d['f_dec'] = frac(dec)
d['f_dec_ltf'] = frac(dec[dec['class'].str.startswith('LTF')]); d['f_dec_vfp'] = frac(dec[dec['class'].str.startswith('VFP')])
crop = d.SYMB_CLASS.isin(list('PFTGCD')); idle = d.SYMB_CLASS.isin(['I','X'])
c = np.full(len(d), 'ambiguous', dtype=object)
hj, hd, lj, ld = d.f_jan>=HI, d.f_dec>=HI, d.f_jan<=LO, d.f_dec<=LO
c[hj & hd] = 'fallow_all_year'; c[hj & ld] = 'fallow_jan_only'; c[lj & hd] = 'fallow_dec_only'
c[lj & ld & crop] = 'cropped'; c[lj & ld & idle] = 'other_idle'
c[(lj & ld) & ~(crop|idle)] = 'non_crop'
d['label'] = c
d.to_file(f'{out}/fields_2021_labeled.gpkg', driver='GPKG')
s = d.groupby('label').agg(fields=('field_id','size'), acres=('area_ac','sum'), median_ac=('area_ac','median')).round(1).sort_values('acres', ascending=False)
s.to_csv(f'{out}/label_summary_2021.csv'); print(s.to_string())
print('\nDWR class within each PVID-based label (acres):')
print(d.assign(g=d.SYMB_CLASS.where(d.SYMB_CLASS.isin(['I','X']),'cropped/other')).pivot_table(index='label',columns='g',values='area_ac',aggfunc='sum',fill_value=0).round(0).to_string())
print('\nMain DWR crop on fallow_all_year fields:'); print(d[d.label=='fallow_all_year'].groupby('MAIN_CROP').area_ac.agg(['size','sum']).round(0).sort_values('sum',ascending=False).head(10).to_string())
dw = lambda v: round(d.loc[d.MAIN_CROP.isin(v),'area_ac'].sum())
r = pd.DataFrame([
 ('PVID 2021 crop report','Gross water toll acres',119144,'WT acres, calendar 2021'),
 ('PVID 2021 crop report','Fallow, first-crop line',20039,'WT acres, calendar 2021'),
 ('PVID 2021 crop report','Fallow, second-crop line',8749,'WT acres, calendar 2021'),
 ('PVID 2021 crop report','Idle or diverted',2749,'WT acres, calendar 2021'),
 ('PVID 2021 crop report','Program, calendar average of monthly values',12710,'WT acres; column read from PDF text layout'),
 ('Verification report (README)','Attachment 1 Fallow, 1 Jan 2021',10376,'official'),
 ('Verification report (README)','Attachment 2 LTF + VFP, 31 Dec 2021',6498+10343,'official'),
 ('Digitized aligned layer','Attachment 1',round(jan.area.sum()/AC),'polygon area'),
 ('Digitized aligned layer','Attachment 2',round(dec.area.sum()/AC),'polygon area'),
 ('DWR 2021 in AOI','I1 idle',dw(['I1']),'polygon area'),('DWR 2021 in AOI','I4 long-term idle',dw(['I4']),'polygon area'),('DWR 2021 in AOI','X unclassified fallow',dw(['X']),'polygon area'),
 ('DWR 2021 in AOI','All fields',round(d.area_ac.sum()),'polygon area'),
], columns=['source','item','acres','note'])
r.to_csv(f'{out}/acreage_reconciliation_2021.csv', index=False); print('\n'+r.to_string(index=False))
