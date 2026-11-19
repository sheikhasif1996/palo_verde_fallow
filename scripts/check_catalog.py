"""Catalog check: scene counts per year and Sentinel-2 processing baselines. usage: check_catalog.py"""
import pystac_client as ps, pandas as pd
c=ps.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1'); bb=[-114.76,33.30,-114.48,33.76]; dt='2017-08-01/2024-12-31'
s2=[i for i in c.search(collections=['sentinel-2-l2a'],bbox=bb,datetime=dt,query={'eo:cloud_cover':{'lte':80}}).items() if i.properties['s2:mgrs_tile'] in('11SQT','11SQS')]
d=pd.DataFrame([dict(id=i.id,year=i.datetime.year,bl=i.properties.get('s2:processing_baseline'),offset=i.properties.get('s2:boa_add_offset_b04')) for i in s2])
print('S2 per year\n',d.groupby('year').size().to_string()); print('\nS2 baseline by year\n',d.pivot_table(index='year',columns='bl',values='id',aggfunc='count',fill_value=0).to_string())
print('\nprops sample',[k for k in s2[0].properties if 'offset' in k.lower() or 'baseline' in k.lower()])
print('duplicate acquisitions (same datatake+tile):',d.id.str.replace(r'_\d{8}T\d{6}$','',regex=True).duplicated().sum())
s1=list(c.search(collections=['sentinel-1-rtc'],bbox=bb,datetime=dt).items())
a=pd.DataFrame([dict(y=i.datetime.year,o=i.properties['sat:relative_orbit'],p=i.properties['platform'][-2:]) for i in s1]); print('\nS1 per year/platform\n',a.pivot_table(index='y',columns='p',values='o',aggfunc='count',fill_value=0).to_string()); print('\nS1 per year/orbit\n',a.pivot_table(index='y',columns='o',values='p',aggfunc='count',fill_value=0).to_string())
l=list(c.search(collections=['landsat-c2-l2'],bbox=bb,datetime=dt,query={'platform':{'eq':'landsat-8'}}).items()); print('\nL8 per year\n',pd.Series([i.datetime.year for i in l]).value_counts().sort_index().to_string())
