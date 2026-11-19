"""Rotation, transition, trend and clustering analyses on the frozen 2018-2024 maps. usage: 18_spatiotemporal_analysis.py OUTPUTS PVID_TOTALS_CSV
Field-level temporal analyses use fields mapped in all 7 years. Permutations preserve each year's fallow count."""
import sys, warnings
import numpy as np, pandas as pd, geopandas as gpd
from scipy.spatial import cKDTree
from scipy.stats import kendalltau, norm
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Patch
warnings.filterwarnings('ignore')
out, pv = sys.argv[1:3]; od=f'{out}/analysis'; YEARS=list(range(2018,2025)); rng=np.random.default_rng(7)
SURF,INK,INK2,GR='#fcfcfb','#0b0b0b','#52514e','#e4e3df'
plt.rcParams.update({'font.size':11,'axes.edgecolor':GR,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'text.color':INK,'figure.facecolor':SURF,'axes.facecolor':SURF,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,'xtick.bottom':False,'ytick.left':False})
G=gpd.read_file(f'{out}/maps/fallow_maps_wy2018_2024.gpkg').set_index('field_id'); I=pd.read_csv(f'{out}/maps/field_year.csv.gz')
W=I.pivot(index='field_id',columns='wy',values='fallow'); full=W.dropna(); print('fields mapped',len(W),'in all 7 years',len(full),flush=True); X=full.values.astype(int); ids=full.index.values; A=G.loc[ids,'area_ac'].values; c=G.loc[ids].centroid; xy=np.c_[c.x,c.y]; n=len(ids)
# area and trend
Ac=I[I.fallow==1].groupby('wy').area_ac.sum(); P=pd.read_csv(pv,index_col=0); T=pd.DataFrame({'map_fallow_ac':Ac.round(0),'pvid_net_fallow':P.loc[YEARS,'net_fallow']}); T['map_vs_pvid_pct']=(100*(T.map_fallow_ac/T.pvid_net_fallow-1)).round(1); T.to_csv(f'{od}/annual_area.csv'); print(T.to_string())
def sen(y,x=np.array(YEARS,float)): return np.median([(y[j]-y[i])/(x[j]-x[i]) for i in range(len(y)) for j in range(i+1,len(y))])
q=pd.qcut(c.y.values,3,labels=['South','Central','North']); rows=[]
for name,m in [('District',np.ones(n,bool))]+[(k,np.asarray(q==k)) for k in ['South','Central','North']]:
    y=(X[m]*A[m][:,None]).sum(0); t,p=kendalltau(YEARS,y); rows.append(dict(area=name,fields=int(m.sum()),acres_2018=y[0].round(0),acres_2024=y[-1].round(0),min_ac=y.min().round(0),max_ac=y.max().round(0),sen_slope_ac_per_yr=round(float(sen(y)),0),kendall_tau=round(float(t),3),p=round(float(p),3)))
Tr=pd.DataFrame(rows); Tr.to_csv(f'{od}/trend_mann_kendall.csv',index=False); print(Tr.to_string(index=False))
# frequency
freq=X.sum(1); F=pd.DataFrame({'years_fallow':freq,'ac':A}).groupby('years_fallow').agg(fields=('ac','size'),acres=('ac','sum')).reindex(range(8),fill_value=0); F['pct_acres']=(100*F.acres/F.acres.sum()).round(1); F.round(1).to_csv(f'{od}/fallow_frequency.csv'); print(F.round(1).to_string())
# transitions, pooled over 6 year pairs
def trans(M): a,b=M[:,:-1].ravel(),M[:,1:].ravel(); return ((a==1)&(b==1)).sum(),((a==1)&(b==0)).sum(),((a==0)&(b==1)).sum(),((a==0)&(b==0)).sum()
ff,fc,cf,cc=trans(X); obs_ff=ff/(ff+fc); obs_cf=cf/(cf+cc)
N=2000; nullff=[];nullcf=[]
for i in range(N):
    Mp=np.column_stack([rng.permutation(X[:,j]) for j in range(7)]); a,b,c_,d=trans(Mp); nullff.append(a/(a+b)); nullcf.append(c_/(c_+d))
pp=lambda o,v:(np.sum(np.array(v)>=o)+1)/(N+1)
Tm=pd.DataFrame([dict(pair='pooled 2018-2024',F_to_F=ff,F_to_C=fc,C_to_F=cf,C_to_C=cc,P_stay_fallow=obs_ff,P_enter_fallow=obs_cf,null_stay=np.mean(nullff),null_enter=np.mean(nullcf),p_perm_stay=pp(obs_ff,nullff))]+[dict(pair=f'{YEARS[j]}-{YEARS[j+1]}',F_to_F=int(((X[:,j]==1)&(X[:,j+1]==1)).sum()),F_to_C=int(((X[:,j]==1)&(X[:,j+1]==0)).sum()),C_to_F=int(((X[:,j]==0)&(X[:,j+1]==1)).sum()),C_to_C=int(((X[:,j]==0)&(X[:,j+1]==0)).sum())) for j in range(6)])
for k in Tm.index[1:]: t=Tm.loc[k]; Tm.loc[k,'P_stay_fallow']=t.F_to_F/(t.F_to_F+t.F_to_C); Tm.loc[k,'P_enter_fallow']=t.C_to_F/(t.C_to_F+t.C_to_C)
Tm.round(3).to_csv(f'{od}/transitions.csv',index=False); print(Tm.round(3).to_string(index=False))
# run lengths and gaps
runs=[];gaps=[]
for r in X:
    s=''.join(map(str,r)); i=0
    while i<7:
        j=i
        while j<7 and s[j]==s[i]: j+=1
        L=j-i
        if s[i]=='1': runs.append(dict(length=L,censored=(i==0 or j==7)))
        elif i>0 and j<7: gaps.append(L)
        i=j
Ru=pd.DataFrame(runs); print('fallow spells',len(Ru),'complete (not touching 2018 or 2024):',int((~Ru.censored).sum())); print('complete spell length distribution:',Ru[~Ru.censored].length.value_counts().sort_index().to_dict()); print('gap between spells (cropped years):',pd.Series(gaps).value_counts().sort_index().to_dict()); pd.DataFrame({'gap':gaps}).to_csv(f'{od}/rotation_gaps.csv',index=False); Ru.to_csv(f'{od}/fallow_spells.csv',index=False)
# clustering
K=8; tree=cKDTree(xy); d,nn=tree.query(xy,k=K+1); nb=nn[:,1:]
def moran(z): z=(z-z.mean()); return (n/(n*K))*np.sum(z*z[nb].sum(1))/np.sum(z*z)*1.0
rows=[]
for j,y in enumerate(YEARS+['frequency']):
    z=(X[:,j] if y!='frequency' else freq).astype(float); o=moran(z); pm=[moran(rng.permutation(z)) for _ in range(999)]; rows.append(dict(period=y,morans_I=o,null_mean=np.mean(pm),z=(o-np.mean(pm))/np.std(pm),p=(np.sum(np.array(pm)>=o)+1)/1000))
Mo=pd.DataFrame(rows); Mo.round(3).to_csv(f'{od}/morans_I.csv',index=False); print(Mo.round(3).to_string(index=False))
xb=freq.mean(); s=freq.std(); ws=K+1.0; nbs=nn; gi=(freq[nbs].sum(1)-xb*ws)/(s*np.sqrt((n*ws-ws*ws)/(n-1))); pg=2*(1-norm.cdf(np.abs(gi))); o=np.argsort(pg); q_=np.empty(n); q_[o]=np.minimum.accumulate((pg[o]*n/np.arange(1,n+1))[::-1])[::-1]
hot=np.where((q_<.05)&(gi>0),'Hot',np.where((q_<.05)&(gi<0),'Cold','None')); Gd=G.loc[ids,['area_ac','geometry']].copy(); Gd['freq']=freq; Gd['gi']=gi; Gd['class']=hot; Gd.to_file(f'{od}/fallow_frequency_hotspots.gpkg',driver='GPKG'); print('Hot spot fields',int((hot=='Hot').sum()),'cold',int((hot=='Cold').sum()))
ramp=['#f0efec','#b7d3f6','#86b6ef','#5598e7','#2a78d6','#1c5cab','#104281','#0d366b']; fig,ax=plt.subplots(1,2,figsize=(11.5,8.2))
for f_,col in zip(range(8),ramp): sub=Gd[Gd.freq==f_]; (sub.plot(ax=ax[0],color=col,edgecolor='none') if len(sub) else None)
for k,col in [('None','#e4e3df'),('Hot','#d95926'),('Cold','#2a78d6')]: sub=Gd[Gd['class']==k]; (sub.plot(ax=ax[1],color=col,edgecolor='none') if len(sub) else None)
for a in ax: a.set_axis_off()
ax[0].set_title('Years fallow, 2018 to 2024',fontsize=12,loc='left'); ax[1].set_title('Hot spots of fallow frequency (Gi*, FDR < 0.05)',fontsize=12,loc='left')
ax[0].legend(handles=[Patch(color=ramp[i],label=f'{i}') for i in range(8)],title='Years fallow',frameon=False,loc='lower right'); ax[1].legend(handles=[Patch(color='#d95926',label='Hot spot'),Patch(color='#e4e3df',label='Not significant')],frameon=False,loc='lower right')
fig.suptitle('Where fallowing repeats across Palo Verde fields',x=.02,ha='left',fontsize=15,weight='medium'); fig.tight_layout(rect=(0,0,1,.95)); fig.savefig(f'{od}/fig_frequency_hotspots.png',dpi=170); plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(11.5,4.4)); ax[0].bar(F.index,F.acres,color='#2a78d6',width=.72); ax[0].set_xlabel('Years fallow out of 7'); ax[0].set_ylabel('Acres'); ax[0].yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}')); ax[0].grid(axis='y',color=GR); ax[0].set_xticks(range(8))
g_=pd.Series(gaps).value_counts().sort_index(); ax[1].bar(g_.index,g_.values,color='#2a78d6',width=.72); ax[1].set_xlabel('Cropped years between fallow spells'); ax[1].set_ylabel('Spell gaps'); ax[1].grid(axis='y',color=GR); ax[1].set_xticks(g_.index)
fig.suptitle('Most fields were fallow rarely, and repeat fallowing comes after short gaps',x=.02,ha='left',fontsize=14,weight='medium'); fig.tight_layout(rect=(0,0,1,.93)); fig.savefig(f'{od}/fig_rotation.png',dpi=170)
