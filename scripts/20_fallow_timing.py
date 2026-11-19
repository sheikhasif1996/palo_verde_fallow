"""Start and end dates of fallow spells from Sentinel-2 NDVI, 2017-2025. usage: 20_fallow_timing.py FIELDS PARTS MANIFEST OUTPUTS
Spell: smoothed NDVI below 0.3 for >=180 days, with breaks of <=15 days merged. NDVI is interpolated to 5 days and smoothed with a
Savitzky-Golay filter (45 days, order 2). Centred smoothing suits this retrospective timing work only; it must not feed prediction features."""
import sys, os, warnings
sys.path.insert(0, os.path.dirname(os.path.abspath(sys.argv[0])))
import numpy as np, pandas as pd, geopandas as gpd
from scipy.signal import savgol_filter
from scipy.stats import chisquare
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import tslib
warnings.filterwarnings('ignore')
fg, parts, mf, out = sys.argv[1:5]; od=f'{out}/timing'; TH,MIN_D,MERGE=.3,180,15
g=gpd.read_file(fg,ignore_geometry=True).set_index('field_id'); g=g[g.label!='non_crop']
s2,_,_=tslib.load_series(parts,mf,'2017-08-01','2025-03-31'); print('NDVI records',len(s2),flush=True)
rows=[]
for fid,o in s2.sort_values('date').groupby('field_id'):
    if len(o)<30 or fid not in g.index: continue
    d=o.date.values; t=np.arange(d[0],d[-1]+np.timedelta64(1,'D'),np.timedelta64(5,'D')); ns=lambda a:a.astype('datetime64[D]').astype(float)
    v=np.interp(ns(t),ns(d),o.ndvi.values); v=savgol_filter(v,9,2); b=v<TH
    i=0; R=[]
    while i<len(b):
        if b[i]:
            j=i
            while j+1<len(b) and b[j+1]: j+=1
            R.append([i,j]); i=j+1
        else: i+=1
    M=[]
    for r in R:
        if M and (r[0]-M[-1][1]-1)*5<=MERGE: M[-1][1]=r[1]
        else: M.append(r[:])
    for i,j in M:
        days=int((j-i+1)*5)
        if days<MIN_D: continue
        st,en=t[i],t[j]; b4=d[d<st]; af=d[d>st]; bs=(af[0]-b4[-1]).astype('timedelta64[D]').astype(int) if len(b4) and len(af) else np.nan; b5=d[d<en]; af2=d[d>en]; be=(af2[0]-b5[-1]).astype('timedelta64[D]').astype(int) if len(b5) and len(af2) else np.nan
        rows.append(dict(field_id=fid,start=pd.Timestamp(st),end=pd.Timestamp(en),days=days,censor_start=i<=6,censor_end=j>=len(b)-7,bracket_start_d=bs,bracket_end_d=be))
S=pd.DataFrame(rows); S['area_ac']=S.field_id.map(g.area_ac); S['complete']=~(S.censor_start|S.censor_end); S.to_csv(f'{od}/fallow_spells.csv',index=False)
print('spells',len(S),'fields with a spell',S.field_id.nunique(),'of',int(S.field_id.nunique()+len(set(s2.field_id.unique())-set(S.field_id.unique()))),'| complete',int(S.complete.sum()),'| open at start',int(S.censor_start.sum()),'| open at end',int(S.censor_end.sum()))
print('median days between images around start and end:',S.bracket_start_d.median(),S.bracket_end_d.median())
C=S[S.complete].copy(); MN=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
st=C.start.dt.month.value_counts().reindex(range(1,13),fill_value=0); en=C.end.dt.month.value_counts().reindex(range(1,13),fill_value=0); stS=S[~S.censor_start].start.dt.month.value_counts().reindex(range(1,13),fill_value=0); enS=S[~S.censor_end].end.dt.month.value_counts().reindex(range(1,13),fill_value=0)
T=pd.DataFrame({'month':MN,'starts_all_observed':stS.values,'ends_all_observed':enS.values,'starts_complete':st.values,'ends_complete':en.values}); T['starts_pct']=(100*T.starts_all_observed/T.starts_all_observed.sum()).round(1); T['ends_pct']=(100*T.ends_all_observed/T.ends_all_observed.sum()).round(1); T.to_csv(f'{od}/start_end_by_month.csv',index=False); print(T.to_string(index=False))
print('uniform test, starts',chisquare(stS.values),'ends',chisquare(enS.values))
print('duration of complete spells (days): median',C.days.median(),'IQR',C.days.quantile([.25,.75]).tolist(),'share under 1 yr %.1f%%, 1-2 yr %.1f%%, 2+ yr %.1f%%'%(100*(C.days<365).mean(),100*((C.days>=365)&(C.days<730)).mean(),100*(C.days>=730).mean()))
print('starts by year (not left-open):',S[~S.censor_start].start.dt.year.value_counts().sort_index().to_dict()); print('share of observed starts falling Jun-Aug: %.1f%%'%(100*stS.iloc[5:8].sum()/stS.sum()))
def active(f,d): q=S[S.field_id==f]; return ((q.start<=d)&(q.end>=d)).any()
J,D=pd.Timestamp('2021-01-01'),pd.Timestamp('2021-12-31'); rows=[]
for lab in ['fallow_all_year','fallow_jan_only','fallow_dec_only','cropped','other_idle']:
    f=g[g.label==lab].index; a=[active(x,J) for x in f]; b=[active(x,D) for x in f]; rows.append(dict(group=lab,fields=len(f),spell_active_1_Jan_2021_pct=round(100*np.mean(a),1),spell_active_31_Dec_2021_pct=round(100*np.mean(b),1)))
V=pd.DataFrame(rows); V.to_csv(f'{od}/validation_vs_pvid_2021.csv',index=False); print(V.to_string(index=False))
de=g[g.label=='fallow_dec_only'].index; x=S[S.field_id.isin(de)&(S.start<=D)&(S.end>=D)]; print('Dec-only fields: start of spell active on 31 Dec 2021, by month:',x.start.dt.to_period('M').astype(str).value_counts().sort_index().to_dict())
je=g[g.label=='fallow_jan_only'].index; y=S[S.field_id.isin(je)&(S.start<=J)&(S.end>=J)]; print('Jan-only fields: end of spell active on 1 Jan 2021, by quarter:',y.end.dt.to_period('Q').astype(str).value_counts().sort_index().to_dict())
I=pd.read_csv(f'{out}/maps/field_year.csv.gz'); recs=[]
for r in I.itertuples():
    q=S[S.field_id==r.field_id]; a=pd.Timestamp(f'{r.wy-1}-10-01'); b=pd.Timestamp(f'{r.wy}-09-30'); ov=np.maximum(0,(np.minimum(q.end.values,np.datetime64(b))-np.maximum(q.start.values,np.datetime64(a))).astype('timedelta64[D]').astype(float)+1).sum() if len(q) else 0.0; recs.append(ov)
I['bare_days']=recs; print('bare days in the water year, median: mapped fallow',I[I.fallow==1].bare_days.median(),'| mapped not fallow',I[I.fallow==0].bare_days.median()); print('mapped fallow field-years with no spell overlapping the water year: %.1f%%'%(100*(I[I.fallow==1].bare_days==0).mean()),'| mapped not fallow with >=180 bare days: %.1f%%'%(100*(I[I.fallow==0].bare_days>=180).mean()))
SURF,INK2,GR='#fcfcfb','#52514e','#e4e3df'; plt.rcParams.update({'font.size':11,'axes.edgecolor':GR,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'figure.facecolor':SURF,'axes.facecolor':SURF,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,'xtick.bottom':False,'ytick.left':False})
fig,ax=plt.subplots(1,2,figsize=(12,4.4),sharey=True)
for a,c,ti in zip(ax,['starts_pct','ends_pct'],['Spell starts','Spell ends']): a.bar(range(12),T[c],color='#2a78d6',width=.72); a.set_xticks(range(12)); a.set_xticklabels(MN); a.grid(axis='y',color=GR); a.set_title(ti,loc='left',fontsize=12)
ax[0].set_ylabel('Percent of spells'); fig.suptitle('When fallow spells begin and end, Palo Verde 2017 to 2025',x=.02,ha='left',fontsize=15,weight='medium'); fig.tight_layout(rect=(0,0,1,.92)); fig.savefig(f'{od}/fig_fallow_timing.png',dpi=170)
