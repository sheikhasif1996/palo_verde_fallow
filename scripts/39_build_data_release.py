"""Builds the data release for Zenodo: field-level fallow maps (attributes only, keyed to DWR 2021 UniqueID),
2021 training labels, digitized PVID verification maps (2021 polygons, 2023 rasters) and the blind reference sample.
DWR field geometry is not redistributed; users join to DWR's 2021 Statewide Crop Mapping on UniqueID.
usage: 39_build_data_release.py PROJECT_DIR PVID2021_DIR OUT_DIR"""
import sys, os, shutil, warnings
import pandas as pd, geopandas as gpd
warnings.filterwarnings('ignore')
P, SRC21, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
for sub in ['fallow_maps', 'pvid_maps', 'reference_sample']:
    os.makedirs(f'{OUT}/{sub}', exist_ok=True)

L = gpd.read_file(f'{P}/data/processed/fields_2021_labeled.gpkg')
key = L[['field_id', 'UniqueID']].rename(columns={'UniqueID': 'dwr_uniqueid'})

# field lookup and 2021 training labels
lab = L[['field_id', 'UniqueID', 'area_ac', 'f_jan', 'f_dec', 'f_dec_ltf', 'f_dec_vfp', 'label']].rename(
    columns={'UniqueID': 'dwr_uniqueid', 'f_dec_ltf': 'f_dec_program', 'f_dec_vfp': 'f_dec_agreement'})
lab.round(4).to_csv(f'{OUT}/field_labels_2021.csv', index=False)

# annual maps (long form) and monthly states
fy = pd.read_csv(f'{P}/outputs/maps/field_year.csv.gz').merge(key, on='field_id', how='left')
fy = fy.rename(columns={'wy': 'water_year', 'months': 'months_in_year'})
fy = fy[['field_id', 'dwr_uniqueid', 'water_year', 'area_ac', 'intensity', 'fallow', 'months_in_year']]
assert fy.dwr_uniqueid.notna().all() and (fy.groupby('water_year').size() == 2548).all()
fy.round(4).to_csv(f'{OUT}/fallow_maps/annual_fallow_wy2018_2024.csv', index=False)
ms = pd.read_csv(f'{P}/outputs/maps/monthly_state.csv.gz').merge(key, on='field_id', how='left')
ms = ms.rename(columns={'wy': 'water_year', 'state': 'fallow_state', 'n_s2': 'n_s2_obs'})
ms = ms[['field_id', 'dwr_uniqueid', 'water_year', 'date', 'fallow_state', 'n_s2_obs', 'ndvi_max']]
ms.round(4).to_csv(f'{OUT}/fallow_maps/monthly_state_wy2018_2024.csv.gz', index=False, compression='gzip')
shutil.copy(f'{P}/outputs/maps/summary_by_year.csv', f'{OUT}/fallow_maps/summary_by_year.csv')

# digitized PVID maps (derived from public USBR verification reports; geometry is PVID's, not DWR's)
for f, n in [('attachment1_fallow_2021-01-01_aligned.gpkg', 'pvid_fallow_2021-01-01.gpkg'),
             ('attachment2_fallow_2021-12-31_aligned.gpkg', 'pvid_fallow_2021-12-31.gpkg')]:
    g = gpd.read_file(f'{SRC21}/{f}').rename(columns={'source': 'report_page'})
    g['snapshot_date'] = g.snapshot_date.astype(str).str[:10]
    g.to_file(f'{OUT}/pvid_maps/{n}', driver='GPKG')
for d in ['2023-01-01', '2023-12-31']:
    shutil.copy(f'{P}/data/processed/pvid2023/fallow_{d}.tif', f'{OUT}/pvid_maps/pvid_fallow_{d}.tif')

# reference sample: design, blind labels and map values
k = pd.read_csv(f'{P}/outputs/reference_sample/key/DO_NOT_OPEN_key.csv')
u = pd.read_csv(f'{P}/outputs/reference_sample/scores_with_span/labels_used.csv')
b = pd.read_csv(f'{P}/outputs/reference_sample/sheet_blank_months.csv')[['sample_id', 'black']]
r = (k[['sample_id', 'field_id', 'wy', 'stratum', 'stratum_N', 'stratum_n', 'weight', 'p', 'lat', 'lon', 'acres']]
     .merge(u[['sample_id', 'map_fallow', 'label', 'conf', 'span']], on='sample_id')
     .merge(b, on='sample_id', how='left').merge(key, on='field_id', how='left'))
r = r.rename(columns={'wy': 'water_year', 'p': 'map_intensity', 'conf': 'confidence', 'span': 'longest_bare_months',
                      'black': 'blank_months', 'acres': 'area_ac'})
r = r[['sample_id', 'field_id', 'dwr_uniqueid', 'water_year', 'stratum', 'stratum_N', 'stratum_n', 'weight', 'lat', 'lon',
       'area_ac', 'map_intensity', 'map_fallow', 'label', 'confidence', 'longest_bare_months', 'blank_months']]
assert len(r) == 450 and r.dwr_uniqueid.notna().all()
r.round(6).to_csv(f'{OUT}/reference_sample/reference_sample_labels.csv', index=False)
print('built', OUT); [print(' ', os.path.relpath(os.path.join(a, f), OUT), os.path.getsize(os.path.join(a, f)) // 1024, 'KB') for a, _, fs in os.walk(OUT) for f in sorted(fs)]
