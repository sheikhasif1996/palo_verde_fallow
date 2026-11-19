"""Step 3 analysis: can PVID's all-year fallow fields be separated from cropped fields in 2021?
usage: 03_feasibility_2021.py FIELDS_GPKG PARTS_DIR OUT_DIR
Single-feature AUC needs no model fitting, so it cannot leak. Bootstrap CIs resample fields."""
import sys, glob, os, warnings
import numpy as np, pandas as pd, geopandas as gpd
from scipy.stats import rankdata
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.dates as mdates
warnings.filterwarnings('ignore')
fg, parts, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
MINV = 0.8
SURF,INK,INK2,GRID = '#fcfcfb','#0b0b0b','#52514e','#e4e3df'
COL = dict(cropped='#2a78d6', fallow_all_year='#eb6834', fallow_jan_only='#1baf7a', fallow_dec_only='#eda100')
NAME = dict(cropped='Cropped (DWR, no PVID fallow)', fallow_all_year='Fallow all year (PVID Jan + Dec)', fallow_jan_only='Fallow 1 Jan only', fallow_dec_only='Fallow 31 Dec only')
plt.rcParams.update({'font.size':11,'axes.edgecolor':GRID,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'text.color':INK,'figure.facecolor':SURF,'axes.facecolor':SURF,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GRID,'grid.linewidth':.8,'axes.axisbelow':True,'xtick.bottom':False,'ytick.left':False})
lab = gpd.read_file(fg, ignore_geometry=True)[['field_id','label','area_ac']]
MAN = pd.read_csv(os.path.join(parts,'..','scene_manifest.csv'))
def load(p):
    fs = [f'{parts}/{p}_{s}.csv.gz' for s in MAN[MAN.sensor==p].scene if os.path.exists(f'{parts}/{p}_{s}.csv.gz')]; d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d['date'] = pd.to_datetime(d.datetime, format='ISO8601', utc=True).dt.tz_localize(None).dt.normalize(); d = d[d.date.dt.year==2021]; print(p,'scenes',d.scene.nunique(),'rows',len(d),flush=True); return d
def wmean(d, keys, cols):
    w = d.n_valid.values; x = d[keys].copy()
    for c in cols: x[c]=d[c].values*w
    x['w']=w; g=x.groupby(keys,as_index=False).sum()
    for c in cols: g[c]=g[c]/g.w
    return g.drop(columns='w')
s2 = load('s2'); s2=s2[s2.n_valid/s2.n_total>=MINV]; s2=wmean(s2,['field_id','date'],['B04','B08','B11','B12'])
s2['ndvi']=(s2.B08-s2.B04)/(s2.B08+s2.B04); s2['ndmi']=(s2.B08-s2.B11)/(s2.B08+s2.B11); s2['str']=(1-s2.B12)**2/(2*s2.B12)
s1 = load('s1'); s1=s1[s1.n_valid/s1.n_total>=MINV].copy(); s1['vv_db']=10*np.log10(s1.vv); s1['vh_db']=10*np.log10(s1.vh)
for c in ['vv_db','vh_db']: s1[c+'_rel']=s1[c]-s1.groupby('rel_orbit')[c].transform('mean')
s1['vvvh']=s1.vv_db-s1.vh_db
ls = load('ls'); ls=ls[ls.n_valid/ls.n_total>=MINV]; ls=wmean(ls,['field_id','date'],['lst_c']); ls['lst_anom']=ls.lst_c-ls.groupby('date').lst_c.transform('median')
for n,d in [('s2',s2),('s1',s1),('ls',ls)]:
    q=d.merge(lab,on='field_id').groupby(['label','field_id']).size().groupby('label').median(); print('median valid obs per field',n,q.round(0).to_dict())
    s = globals(); s[n]=d.merge(lab[['field_id','label']],on='field_id')
s2.to_csv(f'{out}/s2_field_timeseries_2021.csv.gz',index=False); s1.to_csv(f'{out}/s1_field_timeseries_2021.csv.gz',index=False); ls.to_csv(f'{out}/lst_field_timeseries_2021.csv.gz',index=False)
# annual features
def p(q): return lambda x: np.nanpercentile(x,q)
F = s2.groupby('field_id').agg(n_s2=('ndvi','size'), ndvi_p95=('ndvi',p(95)), ndvi_mean=('ndvi','mean'), ndvi_range=('ndvi',lambda x: np.nanpercentile(x,95)-np.nanpercentile(x,5)), ndvi_frac_gt03=('ndvi',lambda x:(x>.3).mean()), ndmi_mean=('ndmi','mean'), ndmi_p95=('ndmi',p(95)), str_mean=('str','mean'), str_p95=('str',p(95)))
S = s1.groupby('field_id').agg(n_s1=('vv_db','size'), vv_mean=('vv_db_rel','mean'), vh_mean=('vh_db_rel','mean'), vvvh_mean=('vvvh','mean'), vv_std=('vv_db_rel','std'), vh_std=('vh_db_rel','std'), vv_range=('vv_db_rel',lambda x: np.percentile(x,95)-np.percentile(x,5)), vh_range=('vh_db_rel',lambda x: np.percentile(x,95)-np.percentile(x,5)))
L = ls.groupby('field_id').agg(n_ls=('lst_anom','size'), lst_anom_mean=('lst_anom','mean'))
L['lst_anom_summer']=ls[ls.date.dt.month.isin([6,7,8,9])].groupby('field_id').lst_anom.mean()
X = F.join(S,how='inner').join(L,how='inner').join(lab.set_index('field_id')); X=X[(X.n_s2>=15)&(X.n_s1>=30)&(X.n_ls>=8)]
X[X.columns.difference(['label'])] = X[X.columns.difference(['label'])].astype(float)
X.to_csv(f'{out}/field_features_2021.csv')
T = X[X.label.isin(['cropped','fallow_all_year'])]; y=(T.label=='fallow_all_year').values.astype(int); print('test set',T.label.value_counts().to_dict(),flush=True)
SENS = dict(ndvi='Optical',ndmi='Optical',str_='Optical',vv_='SAR',vh_='SAR',lst='Thermal')
def sens(n): return next(v for k,v in SENS.items() if n.startswith(k.rstrip('_')) )
def auc(v,y):
    r=rankdata(v); n1=y.sum(); n0=len(y)-n1; return (r[y==1].sum()-n1*(n1+1)/2)/(n1*n0)
rng=np.random.default_rng(7); rows=[]; cols=[c for c in F.columns if c!='n_s2']+[c for c in S.columns if c!='n_s1']+['lst_anom_mean','lst_anom_summer']
for c in cols:
    v=T[c].values; ok=~np.isnan(v); a=auc(v[ok],y[ok]); sg=1 if a>=.5 else -1; A=max(a,1-a); bs=[]
    idx1=np.where(ok&(y==1))[0]; idx0=np.where(ok&(y==0))[0]
    for _ in range(1000):
        i=np.r_[rng.choice(idx1,len(idx1)),rng.choice(idx0,len(idx0))]; b=auc(v[i],y[i]); bs.append(b if sg>0 else 1-b)
    rows.append(dict(feature=c,sensor=sens(c),auc=A,ci_lo=np.percentile(bs,2.5),ci_hi=np.percentile(bs,97.5),fallow_is='higher' if sg>0 else 'lower',median_fallow=np.nanmedian(v[y==1]),median_cropped=np.nanmedian(v[y==0])))
R=pd.DataFrame(rows).sort_values('auc',ascending=False); R.to_csv(f'{out}/separability_auc_2021.csv',index=False); print(R.round(3).to_string(index=False))
# no-canopy comparison: SAR on bare-looking cropped fields vs fallow, the only case where canopy cannot explain a difference
rec=[]
for fid,g in s2.groupby('field_id'):
    g=g.sort_values('date'); rec.append(pd.DataFrame({'field_id':fid,'date':g.date.values,'ndvi_':g.ndvi.values}))
nd=pd.concat(rec).sort_values('date')
m=pd.merge_asof(s1.sort_values('date'),nd,on='date',by='field_id',tolerance=pd.Timedelta('8D'),direction='nearest').dropna(subset=['ndvi_'])
m=m[m.label.isin(['cropped','fallow_all_year'])]
m['state']=np.select([m.label=='fallow_all_year',m.ndvi_<.25,m.ndvi_>.5],['fallow','cropped, no canopy (NDVI<0.25)','cropped, canopy (NDVI>0.5)'],'other')
fm=m[m.state!='other'].groupby(['state','field_id']).agg(n=('vv_db_rel','size'),vv=('vv_db_rel','mean'),vh=('vh_db_rel','mean'),vvvh=('vvvh','mean')).reset_index(); fm=fm[fm.n>=3]
B=fm.groupby('state').agg(fields=('field_id','size'),vv_med=('vv','median'),vv_q25=('vv',lambda x:x.quantile(.25)),vv_q75=('vv',lambda x:x.quantile(.75)),vh_med=('vh','median'),vh_q25=('vh',lambda x:x.quantile(.25)),vh_q75=('vh',lambda x:x.quantile(.75)),vvvh_med=('vvvh','median')).round(2); B.to_csv(f'{out}/sar_no_canopy_comparison_2021.csv'); print('\n',B.to_string())
A0=fm[fm.state.str.contains('no canopy')]; A1=fm[fm.state=='fallow']
for c in ['vv','vh','vvvh']:
    v=np.r_[A1[c],A0[c]]; yy=np.r_[np.ones(len(A1),int),np.zeros(len(A0),int)]; a=auc(v,yy); print('fallow vs no-canopy cropped AUC',c,round(max(a,1-a),3),'fallow','higher' if a>.5 else 'lower')
# time-binned curves
BIN=pd.Timedelta('15D'); o=pd.Timestamp('2021-01-01')
def curve(d,c,l):
    x=d[d.label==l].copy(); x[c]=x[c].astype(float); x['b']=((x.date-o)/BIN).astype(int); f=x.groupby(['b','field_id'])[c].mean().reset_index(); g=f.groupby('b')[c].quantile([.25,.5,.75]).unstack(); g=g[g.index<24].copy(); g['t']=o+BIN*(g.index+.5); return g
def plot(ax,d,c,labels,ylabel):
    for l in labels:
        g=curve(d,c,l); ax.fill_between(g.t,g[.25],g[.75],color=COL[l],alpha=.16,lw=0); ax.plot(g.t,g[.5],color=COL[l],lw=2,label=NAME[l],solid_capstyle='round')
    ax.set_ylabel(ylabel); ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,3,5,7,9,11])); ax.xaxis.set_major_formatter(mdates.DateFormatter('%b')); ax.set_xlim(o,pd.Timestamp('2021-12-31'))
L2=['cropped','fallow_all_year']; panels=[(s2,'ndvi','NDVI'),(s2,'ndmi','NDMI'),(s2,'str','SWIR transform (OPTRAM input)'),(s1,'vv_db_rel','VV, orbit-centred (dB)'),(s1,'vh_db_rel','VH, orbit-centred (dB)'),(ls,'lst_anom','LST minus date median (°C)')]
fig,ax=plt.subplots(2,3,figsize=(13.5,7.4)); ax=ax.ravel()
for a,(d,c,y_) in zip(ax,panels): plot(a,d,c,L2,y_)
n1,n0=(y==1).sum(),(y==0).sum()
fig.suptitle('All-year fallow fields are bare and hot; radar separates them less clearly, Palo Verde 2021',x=.02,ha='left',fontsize=15,weight='medium')
fig.text(.02,.925,f'Median and interquartile range across fields, 15-day bins. Fallow n = {n1} fields, cropped n = {n0} fields.',color=INK2,fontsize=11)
h,l=ax[0].get_legend_handles_labels(); fig.legend(h,l,loc='lower left',bbox_to_anchor=(.02,.0),ncol=2,frameon=False)
fig.tight_layout(rect=(0,.06,1,.91)); fig.savefig(f'{out}/fig1_timeseries_fallow_vs_cropped_2021.png',dpi=170); plt.close(fig)
fig,a=plt.subplots(figsize=(9.5,6.6)); r=R.iloc[::-1].reset_index(drop=True); SC=dict(Optical='#2a78d6',SAR='#eb6834',Thermal='#1baf7a')
for i,x in r.iterrows(): a.plot([x.ci_lo,x.ci_hi],[i,i],color=SC[x.sensor],lw=2,solid_capstyle='round',zorder=2)
for s,c in SC.items():
    q=r[r.sensor==s]; a.scatter(q.auc,q.index,s=62,color=c,edgecolor=SURF,linewidth=1.5,label=s,zorder=3)
a.set_yticks(r.index); a.set_yticklabels(r.feature); a.set_xlim(.5,1.005); a.set_xlabel('AUC, fallow all year vs. cropped (0.5 = no separation)'); a.grid(axis='y',visible=False); a.legend(frameon=False,loc='lower right',title='Sensor')
fig.suptitle('Which single 2021 feature separates fallow from cropped fields',x=.02,ha='left',fontsize=15,weight='medium'); fig.text(.02,.915,'Dots show AUC; bars show 95% bootstrap interval by field.',color=INK2)
fig.tight_layout(rect=(0,0,1,.9)); fig.savefig(f'{out}/fig2_feature_separability_2021.png',dpi=170); plt.close(fig)
fig,a=plt.subplots(figsize=(10,5)); plot(a,s2,'ndvi',['cropped','fallow_all_year','fallow_jan_only','fallow_dec_only'],'NDVI')
a.legend(frameon=False,loc='upper center',ncol=2,bbox_to_anchor=(.5,-.1)); fig.suptitle('Fields only on the December map went bare by August; fields only on the January map stayed bare',x=.02,ha='left',fontsize=14,weight='medium'); fig.tight_layout(rect=(0,0,1,.93)); fig.savefig(f'{out}/fig3_ndvi_transition_fields_2021.png',dpi=170); plt.close(fig)
print('saved',flush=True)
