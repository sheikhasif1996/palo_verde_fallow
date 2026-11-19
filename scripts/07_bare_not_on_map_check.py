"""Diagnostic: what are the fields that look bare for 90+ days but are off the PVID fallow map?
Uses imagery after the label date only to describe those fields, never as model input. usage: script FIELDS PARTS MANIFEST"""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
import numpy as np, pandas as pd, geopandas as gpd
import tslib
warnings.filterwarnings('ignore')
fg, parts, mf = sys.argv[1:4]
g = gpd.read_file(fg, ignore_geometry=True).set_index('field_id'); crop=g.SYMB_CLASS.isin(list('PFTGCD'))
s2,_,_ = tslib.load_series(parts,mf,'2019-12-01','2022-12-31')
SN={'jan':pd.Timestamp('2021-01-01'),'dec':pd.Timestamp('2021-12-31')}
rows=[]
for k,t in SN.items():
    p=s2[(s2.date<=t)&(s2.date>t-pd.Timedelta(days=400))]; last=p[p.ndvi>=.3].groupby('field_id').date.max()
    fut=s2[(s2.date>t)&(s2.date<=t+pd.Timedelta(days=120))]; first=fut[fut.ndvi>=.3].groupby('field_id').date.min()
    f=g[f'f_{k}']; ids=p.groupby('field_id').size(); ids=ids[ids>=10].index
    x=pd.DataFrame({'dsg':(t-last.reindex(ids)).dt.days.fillna(400)},index=ids); x['snap']=k; x['f']=f.reindex(ids); x['crop']=crop.reindex(ids); x['main']=g.MAIN_CROP.reindex(ids); x['label']=g.label.reindex(ids)
    x['days_to_green']=(first.reindex(ids)-t).dt.days
    x['group']=np.where(x.f>=.7,'on PVID map',np.where((x.f<=.05)&x.crop,'off map, DWR cropped','other')); rows.append(x)
d=pd.concat(rows)
b=d[d.dsg>=90]
print('Fields bare for 90+ days before the date, by group:'); print(b.groupby(['snap','group']).size().unstack().to_string())
o=b[b.group=='off map, DWR cropped']; print('\nOff map, DWR cropped, bare 90+ days:')
print('bare duration bins:', pd.cut(o.dsg,[89,120,180,270,399,400]).value_counts().sort_index().to_dict())
print('greens up within 60 days after date: %.1f%%, within 120 days: %.1f%%' % (100*(o.days_to_green<=60).mean(),100*(o.days_to_green<=120).mean()))
print(o.groupby('snap').apply(lambda x: pd.Series({'n':len(x),'green_le60_pct':100*(x.days_to_green<=60).mean(),'green_le120_pct':100*(x.days_to_green<=120).mean(),'median_dsg':x.dsg.median()})).round(1).to_string())
print('top DWR crops:',o.main.value_counts().head(6).to_dict())
m=b[b.group=='on PVID map']; print('\nOn PVID map and bare 90+ days: green within 120 days after date: %.1f%%'%(100*(m.days_to_green<=120).mean()), 'n',len(m))
pa=d[(d.group=='on PVID map')]; print('On PVID map but NOT bare 90+ days:', int((pa.dsg<90).sum()),'of',len(pa), 'median dsg', pa[pa.dsg<90].dsg.median())
