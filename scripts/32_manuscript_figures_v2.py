"""Second-draft figures to Springer artwork rules: Arial-metric sans serif (Liberation Sans), lettering 8 pt or larger,
width at most 174 mm. PNG at 600 dpi for the Word file; submission files as TIFF (600 dpi maps/combination, 1200 dpi line art)
and vector PDF. usage: 32_manuscript_figures_v2.py PROJECT_DIR"""
import sys, os, warnings
import numpy as np, pandas as pd, geopandas as gpd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, matplotlib.dates as mdates
from matplotlib.patches import FancyBboxPatch, Rectangle, Patch, FancyArrowPatch
warnings.filterwarnings('ignore')
P=sys.argv[1]; O=f'{P}/outputs'; F=f'{P}/manuscript/figures_v2'; SUB=f'{P}/manuscript/figures_submission'; os.makedirs(F,exist_ok=True); os.makedirs(SUB,exist_ok=True)
BLUE,OR,AQ,YL='#2a78d6','#eb6834','#1baf7a','#eda100'; SURF,INK,INK2,GR='#ffffff','#0b0b0b','#52514e','#e4e3df'
plt.rcParams.update({'font.family':['Arial','Liberation Sans'],'font.size':9,'axes.edgecolor':'#bdbcb7','axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'text.color':INK,'figure.facecolor':SURF,'axes.facecolor':SURF,'axes.spines.top':False,'axes.spines.right':False,'axes.axisbelow':True,'xtick.bottom':False,'ytick.left':False,'savefig.dpi':600})
def lab(ax,s): ax.text(-.02,1.03,s,transform=ax.transAxes,fontsize=10,weight='bold',ha='right',va='bottom')
COMBO={'fig01_study_area','fig06_annual_maps','fig07_frequency_hotspots','fig11_forecast_2024'}
def save(fig,n):
    w=fig.get_size_inches()[0]*25.4; assert w<=174.5, (n,w)
    fig.savefig(f'{F}/{n}.png',dpi=600,bbox_inches='tight',facecolor='white')
    fig.savefig(f'{SUB}/{n}.pdf',bbox_inches='tight',facecolor='white')
    fig.savefig(f'{SUB}/{n}.tif',dpi=600 if n in COMBO else 1200,bbox_inches='tight',facecolor='white',pil_kwargs={'compression':'tiff_lzw'})
    plt.close(fig); print(n, round(w), 'mm')
G=gpd.read_file(f'{O}/maps/fallow_maps_wy2018_2024.gpkg'); g84=G.to_crs(4326); s=np.cos(np.radians(33.55))
def geo(ax,gdf=None):
    ax.set_aspect(1/s); ax.tick_params(labelsize=8,length=0); ax.grid(False); [ax.spines[k].set_visible(False) for k in ['top','right','left','bottom']]
L=gpd.read_file(f'{P}/data/raw/drivers/nhd/flowlines.gpkg').to_crs(4326); can=L[L.ftype==336]; riv=L[L.gnis_name.fillna('').str.contains('Colorado River')]
st=gpd.read_file(f'{P}/data/raw/boundaries/states/cb_2023_us_state_20m.shp').to_crs(4326)
fig=plt.figure(figsize=(6.5,7.4)); ax=fig.add_axes([.08,.05,.62,.92]); g84.plot(ax=ax,color='#dcdbd5',edgecolor='white',linewidth=.15); can.plot(ax=ax,color=BLUE,linewidth=.35); riv.plot(ax=ax,color=OR,linewidth=1.1)
b=g84.total_bounds; ax.set_xlim(b[0]-.03,b[2]+.03); ax.set_ylim(b[1]-.02,b[3]+.02); geo(ax); ax.set_xlabel('Longitude (°)'); ax.set_ylabel('Latitude (°)')
x0=b[0]+.005; y0=b[1]+.005; ln=5/(111.32*s); ax.plot([x0,x0+ln],[y0,y0],color=INK,lw=2); ax.text(x0+ln/2,y0+.006,'5 km',ha='center',fontsize=8); ax.annotate('N',xy=(x0+.01,y0+.075),xytext=(x0+.01,y0+.03),ha='center',fontsize=9,arrowprops=dict(arrowstyle='-|>',color=INK,lw=1))
ax.legend(handles=[Patch(color='#dcdbd5',label='Agricultural fields (DWR 2021)'),plt.Line2D([],[],color=BLUE,lw=1,label='Canals and ditches (NHD)'),plt.Line2D([],[],color=OR,lw=1.6,label='Colorado River')],frameon=False,loc='upper left',fontsize=8)
ai=fig.add_axes([.68,.68,.3,.28]); ca=st[st.NAME.isin(['California','Arizona','Nevada','Oregon','Utah','Idaho'])]; ca.plot(ax=ai,color='#f0efec',edgecolor='#a9a8a3',linewidth=.5); st[st.NAME=='California'].plot(ax=ai,color='#d8d7d1',edgecolor='#8f8e89',linewidth=.6); ai.add_patch(Rectangle((b[0]-.05,b[1]-.05),b[2]-b[0]+.1,b[3]-b[1]+.1,fill=True,color=OR,zorder=5)); ai.set_xlim(-125,-109); ai.set_ylim(31,43); ai.set_aspect(1/np.cos(np.radians(37))); ai.set_xticks([]); ai.set_yticks([]); [i.set_visible(False) for i in ai.spines.values()]; ai.text(-120.7,36.6,'California',fontsize=8,ha='center',rotation=-40); ai.text(-112.1,34.5,'Arizona',fontsize=8,ha='center'); save(fig,'fig01_study_area')
fig,ax=plt.subplots(figsize=(6.5,5.6)); ax.axis('off'); ax.set_xlim(0,100); ax.set_ylim(0,100)
def B(x,y,w,h,t,fc='#eef4fc'): ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.4',fc=fc,ec='#7f9ec7',lw=.8)); ax.text(x+w/2,y+h/2,t,ha='center',va='center',fontsize=8)
def A(a,b): ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=10,color=INK2,lw=.9))
B(2,82,30,14,'Inputs\nSentinel-2 L2A\nSentinel-1 RTC, Landsat 8\nAug 2017 to Mar 2025'); B(36,82,28,14,'Field units\nDWR 2021 polygons\n30 m inward buffer'); B(68,82,30,14,'Labels, PVID 2021\nVerification maps\n1 Jan and 31 Dec','#fdeee7')
B(14,60,72,13,'Per-field time series, 20 m grid\nNDVI, NDMI, SWIR transform, VV, VH, LST'); A((17,82),(25,73)); A((50,82),(50,73)); A((83,82),(75,73))
B(2,38,44,13,'Model comparison, spatially blocked folds\nSensor sets, window, rule vs. RF vs. boosting'); B(54,38,44,13,'Frozen method\nOptical RF, window −90 to +180 d\nFallow if intensity ≥ 0.5','#fdeee7'); A((36,60),(24,51)); A((46,44.5),(54,44.5))
B(2,18,44,12,'Validation\nDWR WY2018-2022, held-out WY2023,\nCDL, PVID reports, reference sample'); B(54,18,44,12,'Field-level maps\nWY2018-2024: monthly state,\nannual intensity'); A((76,38),(76,30)); A((54,24),(46,24))
B(2,1,30,12,'Spatiotemporal patterns\nRotation, trends,\nclustering, timing'); B(35,1,30,12,'Drivers\nGlobal RF vs. geographically\nweighted RF'); B(68,1,30,12,'Next-year prediction\nWY2024 held out,\npast data only'); A((68,18),(17,13)); A((76,18),(50,13)); A((86,18),(84,13)); save(fig,'fig02_workflow')
s2=pd.read_csv(f'{O}/s2_field_timeseries_2021.csv.gz',parse_dates=['date']); s1=pd.read_csv(f'{O}/s1_field_timeseries_2021.csv.gz',parse_dates=['date']); ls=pd.read_csv(f'{O}/lst_field_timeseries_2021.csv.gz',parse_dates=['date']); o=pd.Timestamp('2021-01-01'); BIN=pd.Timedelta('15D')
def curve(d,c,l):
    x=d[d.label==l].copy(); x[c]=x[c].astype(float); x['b']=((x.date-o)/BIN).astype(int); f=x.groupby(['b','field_id'])[c].mean().reset_index(); g=f.groupby('b')[c].quantile([.25,.5,.75]).unstack(); g=g[g.index<24].copy(); g['t']=o+BIN*(g.index+.5); return g
fig,axs=plt.subplots(2,3,figsize=(6.85,4.5)); P_=[(s2,'ndvi','NDVI'),(s2,'ndmi','NDMI'),(s2,'str','SWIR transform'),(s1,'vv_db_rel','VV, orbit-centred (dB)'),(s1,'vh_db_rel','VH, orbit-centred (dB)'),(ls,'lst_anom','LST anomaly (°C)')]
for a,(d,c,yl),t in zip(axs.ravel(),P_,'abcdef'):
    for l,col in [('cropped',BLUE),('fallow_all_year',OR)]:
        g_=curve(d,c,l); a.fill_between(g_.t,g_[.25],g_[.75],color=col,alpha=.18,lw=0); a.plot(g_.t,g_[.5],color=col,lw=1.5)
    a.set_ylabel(yl,fontsize=8); a.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10])); a.xaxis.set_major_formatter(mdates.DateFormatter('%b')); a.set_xlim(o,pd.Timestamp('2021-12-31')); a.grid(color=GR,lw=.5); a.tick_params(labelsize=8); lab(a,t)
fig.legend(handles=[plt.Line2D([],[],color=BLUE,lw=1.6,label='Cropped'),plt.Line2D([],[],color=OR,lw=1.6,label='Fallow all year')],loc='lower center',ncol=2,frameon=False,fontsize=8.5); fig.tight_layout(rect=(0,.05,1,1)); save(fig,'fig03_timeseries')
C=pd.read_csv(f'{O}/after180/cross_year_metrics.csv'); H=pd.read_csv(f'{O}/holdout_2023_metrics.csv'); wy=[2018,2019,2020,2021,2022,2023]; rf=list(C[C.model=='RF optical'].sort_values('wy').F1)+[float(H[H.model=='frozen'].F1.iloc[0])]; ru=list(C[C.model=='rule NDVI<0.3'].sort_values('wy').F1)+[float(H[H.model=='rule'].F1.iloc[0])]
fig,ax=plt.subplots(figsize=(6.5,3)); x=np.arange(6); ax.bar(x-.19,rf,.36,color=BLUE,label='Random Forest (frozen)'); ax.bar(x+.19,ru,.36,color=OR,label='NDVI rule'); ax.set_xticks(x); ax.set_xticklabels([str(i) for i in wy[:-1]]+['2023\n(held out)']); ax.set_ylim(0,1); ax.set_ylabel('F1 vs. DWR'); ax.grid(axis='y',color=GR,lw=.5); ax.legend(frameon=False,ncol=2,loc='lower center',bbox_to_anchor=(.5,1.0),fontsize=8.5); ax.axvline(4.5,color='#a9a8a3',lw=.8,ls='--'); fig.tight_layout(); save(fig,'fig04_cross_year_f1')
Ar=pd.read_csv(f'{O}/product_comparison/area_by_product_with_pvid.csv'); fig,ax=plt.subplots(figsize=(6.5,3.4))
for c,l,col,mk in [('map_ac','Frozen map',BLUE,'o'),('dwr_ac','DWR idle + unclassified fallow',OR,'s'),('cdl_ac','CDL Fallow/Idle',AQ,'^'),('pvid_net_fallow','PVID net fallow',YL,'D')]: ax.plot(Ar.wy,Ar[c],color=col,lw=1.6,marker=mk,ms=5,mec='white',mew=.8,label=l)
ax.set_ylim(0); ax.set_xticks(Ar.wy); ax.set_ylabel('Fallow area (acres)'); ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}')); ax.grid(color=GR,lw=.5); ax.legend(frameon=False,ncol=2,loc='upper center',bbox_to_anchor=(.5,1.22),fontsize=8.5); fig.tight_layout(); save(fig,'fig05_product_comparison')
Yr=list(range(2018,2025)); fig,axs=plt.subplots(2,4,figsize=(6.85,5.4)); axs=axs.ravel(); bx=g84.total_bounds
for a,y,t in zip(axs,Yr,'abcdefg'):
    g84.plot(ax=a,color='#e7e6e1',edgecolor='none'); k=g84[g84[f'fallow_{y}']==1]; k.plot(ax=a,color=OR,edgecolor='none'); a.set_xlim(bx[0]-.01,bx[2]+.01); a.set_ylim(bx[1]-.01,bx[3]+.01); a.set_aspect(1/s); a.set_axis_off(); a.set_title(f'{t}  WY{y}\n{k.area_ac.sum():,.0f} acres',fontsize=8,loc='left')
axs[7].axis('off'); axs[7].legend(handles=[Patch(color=OR,label='Fallow'),Patch(color='#e7e6e1',label='Not fallow')],frameon=False,loc='center',fontsize=9); fig.tight_layout(); save(fig,'fig06_annual_maps')
Hs=gpd.read_file(f'{O}/analysis/fallow_frequency_hotspots.gpkg').to_crs(4326); ramp=['#f0efec','#b7d3f6','#86b6ef','#5598e7','#2a78d6','#1c5cab','#104281','#0d366b']; fig,axs=plt.subplots(1,2,figsize=(6.85,5.3))
for f_,col in zip(range(8),ramp):
    q=Hs[Hs.freq==f_]; q.plot(ax=axs[0],color=col,edgecolor='none') if len(q) else None
for k_,col in [('None','#e4e3df'),('Hot','#d95926')]: Hs[Hs['class']==k_].plot(ax=axs[1],color=col,edgecolor='none')
for a,t in zip(axs,'ab'): a.set_aspect(1/s); a.set_axis_off(); lab(a,t)
axs[0].legend(handles=[Patch(color=ramp[i],label=str(i)) for i in range(8)],title='Years fallow',frameon=False,loc='lower right',fontsize=8,title_fontsize=8); axs[1].legend(handles=[Patch(color='#d95926',label='Hot spot'),Patch(color='#e4e3df',label='Not significant')],frameon=False,loc='lower right',fontsize=8); fig.tight_layout(); save(fig,'fig07_frequency_hotspots')
Fr=pd.read_csv(f'{O}/analysis/fallow_frequency.csv',index_col=0); Gp=pd.read_csv(f'{O}/analysis/rotation_gaps.csv').gap.value_counts().sort_index(); fig,axs=plt.subplots(1,2,figsize=(6.8,2.9)); axs[0].bar(Fr.index,Fr.acres,color=BLUE,width=.72); axs[0].set_xlabel('Years fallow out of 7'); axs[0].set_ylabel('Acres'); axs[0].yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}')); axs[0].set_xticks(range(8)); axs[1].bar(Gp.index,Gp.values,color=BLUE,width=.72); axs[1].set_xlabel('Cropped years between fallow spells'); axs[1].set_ylabel('Number of gaps'); axs[1].set_xticks(Gp.index)
for a,t in zip(axs,'ab'): a.grid(axis='y',color=GR,lw=.5); lab(a,t)
fig.tight_layout(); save(fig,'fig08_rotation')
Tm=pd.read_csv(f'{O}/timing/start_end_by_month.csv'); fig,axs=plt.subplots(1,2,figsize=(6.8,2.8),sharey=True)
for a,c,t in zip(axs,['starts_pct','ends_pct'],'ab'): a.bar(range(12),Tm[c],color=BLUE,width=.72); a.set_xticks(range(12)); a.set_xticklabels(Tm.month,fontsize=8); a.grid(axis='y',color=GR,lw=.5); lab(a,t); a.set_xlabel('Spell starts' if t=='a' else 'Spell ends')
axs[0].set_ylabel('Percent of spells'); fig.tight_layout(); save(fig,'fig09_timing')
I_=pd.read_csv(f'{O}/drivers/importance_where_global.csv',index_col=0).iloc[::-1]; Pr=pd.read_csv(f'{O}/drivers/predictions_where.csv'); NM={'soil_claytotal':'Clay','dist_river_m':'Distance to river','area_ac':'Field area','soil_sar':'Sodium adsorption ratio','soil_ec':'Salinity (EC)','elev_rel_m':'Relative elevation','compactness_pp':'Compactness','soil_icc_num':'Irrigated capability','slope_deg':'Slope','soil_drain_ord':'Drainage class','dist_named_canal_m':'Distance to named canal'}
fig,axs=plt.subplots(1,2,figsize=(6.85,3.3),gridspec_kw=dict(width_ratios=[1.35,1])); axs[0].barh(range(len(I_)),I_['mean'],xerr=I_['std'],color=BLUE,height=.62,error_kw=dict(ecolor=INK2,lw=.7)); axs[0].set_yticks(range(len(I_))); axs[0].set_yticklabels([NM[i] for i in I_.index],fontsize=8); axs[0].axvline(0,color='#a9a8a3',lw=.7); axs[0].set_xlabel('Drop in held-out R² when shuffled'); axs[0].grid(axis='x',color=GR,lw=.5); lab(axs[0],'a')
axs[1].hexbin(Pr.g,Pr.y,gridsize=26,cmap=matplotlib.colors.LinearSegmentedColormap.from_list('b',plt.cm.Blues(np.linspace(.35,1,256))),mincnt=1,extent=(0,1,0,1),linewidths=.1,bins='log'); axs[1].plot([0,1],[0,1],color=OR,lw=.9); axs[1].set_xlabel('Predicted intensity'); axs[1].set_ylabel('Observed intensity'); axs[1].set_xlim(0,1); axs[1].set_ylim(0,1); axs[1].set_aspect('equal'); lab(axs[1],'b'); fig.tight_layout(); save(fig,'fig10_drivers')
Pd=pd.read_csv(f'{O}/forecast/prediction_2024.csv'); g2=g84.merge(Pd[['field_id','y','pred']],on='field_id'); fig,ax=plt.subplots(figsize=(4.4,5.6)); cat=[(1,1,BLUE,'Fallow, predicted fallow'),(1,0,OR,'Fallow, not predicted'),(0,1,YL,'Not fallow, predicted fallow'),(0,0,'#e4e3df','Not fallow, predicted not fallow')]
for a_,b_,c_,l_ in cat: q=g2[(g2.y==a_)&(g2.pred==b_)]; q.plot(ax=ax,color=c_,edgecolor='none') if len(q) else None
ax.set_aspect(1/s); ax.set_axis_off(); ax.legend(handles=[Patch(color=c,label=l) for a,b,c,l in cat],frameon=False,loc='lower right',fontsize=8); fig.tight_layout(); save(fig,'fig11_forecast_2024')
