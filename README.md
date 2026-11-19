# palo_verde_fallow

Analysis code for field-level mapping of rotational land fallowing in the Palo Verde Irrigation District (PVID), California, from Sentinel-2 time series, water years 2018 to 2024.

The workflow queries imagery through the Microsoft Planetary Computer STAC API, builds per-field time series for 2,548 fields from the California Department of Water Resources (DWR) 2021 crop map, trains a Random Forest on digitized 2021 PVID fallowing maps, and produces monthly fallow states and annual fallow maps. The maps are validated with spatially blocked cross-validation, a held-out year (2023), PVID's 2023 verification maps and a blind stratified reference sample, and are then used for spatiotemporal, driver and next-year prediction analyses.

## Frozen method

The settings fixed before any test are in [`config/frozen_method.json`](config/frozen_method.json): an optical-only Random Forest (500 trees, minimum leaf size 2, `class_weight='balanced_subsample'`, seed 7) on 11 Sentinel-2 features, with a window from 90 days before to 180 days after each date. A field-year is fallow when at least half of its valid monthly states are fallow. Scripts 10 and 17 read these settings from `outputs/frozen_method.json`; copy the file there before running them.

Water year 2023 was scored once after the method was frozen (script 10), and next-year prediction was scored once on 2024 (script 24). Rerunning those scripts reproduces the reported results; tuning on them would invalidate the held-out tests.

## Installation

```bash
conda env create -f environment.yml
conda activate palo_verde_fallow
```

Imagery is read directly from the Planetary Computer; no account or key is needed for the collections used (`sentinel-2-l2a`, `sentinel-1-rtc`, `landsat-c2-l2`).

## Data

Input data are not included in this repository.

| Data | Source |
|---|---|
| Field boundaries and crop classes, 2018–2024 | DWR Statewide Crop Mapping, https://data.cnra.ca.gov/dataset/statewide-crop-mapping |
| PVID fallowed land verification reports | U.S. Bureau of Reclamation, https://www.usbr.gov/lc/region/g4000/4200Rpts/DecreeRpt/ |
| PVID crop reports | https://www.pvid.org |
| Cropland Data Layer | USDA NASS |
| Soils (SSURGO), hydrography (NHD), elevation (3DEP) | USDA NRCS, USGS |

The digitized PVID fallow maps, the field-level fallow maps and the reference-sample labels are archived separately (see the Data availability statement of the paper).

## Scripts

Every script takes its input and output paths as command-line arguments; run a script with no arguments, or read its first lines, for the usage. They run in numeric order.

| Step | Scripts | Purpose |
|---|---|---|
| Labels | `check_label_alignment.py`, `01` | Align digitized PVID 2021 maps with DWR fields; assign field labels |
| Time series | `check_catalog.py`, `02`, `tslib.py` | Query the STAC catalogue and extract per-field Sentinel-2, Sentinel-1 and Landsat time series |
| Development | `03`–`08` | Signal separation, classifiers, leakage audits, window comparison |
| Validation | `09`, `10`, `11`, `27`–`29` | DWR 2018–2022 and held-out 2023; CDL check; registration of and scoring against PVID 2023 maps |
| Reference sample | `12`–`14`, `30` | Stratified sample, blinded image sheets and label workbook, Stehman stratified estimators |
| Products | `15`, `16`, `17`, `19` | Product comparison, PVID report reconciliation, final 2018–2024 maps, 2024 diagnostic |
| Spatiotemporal analysis | `18`, `20`, `33` | Trend, persistence, clustering, hot spots, spell timing, persistence sensitivity |
| Drivers and prediction | `21`–`24` | Driver collection and QA, global and geographically weighted Random Forest, next-year prediction |
| Figures | `25`, `32` | Figures for the paper |
| Utilities | `inspect_inputs.py`, `inspect_dwr_years.py` | Input inspection |

## Known limitation

`13_make_image_sheets.py` selects one Sentinel-2 scene date per month for the whole district without checking coverage of each field, so some image sheets for western fields contain blank months. This is reported in the paper; the script is kept as used.

## Citation

If you use this code, please cite the paper (citation to be added on publication).

## License

To be added.
