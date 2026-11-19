"""Rebuild PVID's fallow totals from every crop report, cross-check overlapping columns, and compare with the products.
usage: 16_pvid_report_reconciliation.py REPORT_DIR OUTPUTS
Net fallow = first-crop Fallow - second-crop Fallow, following each report's own arithmetic (Gross WT - Total Second Crop = Net WT)."""
import sys, re, subprocess
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
rd, out = sys.argv[1:3]; od=f'{out}/product_comparison'; num=lambda s:int(s.replace(',',''))
V={}; rows=[]
for ry in range(2018,2025):
    t=subprocess.run(['pdftotext','-layout',f'{rd}/{ry}.pdf','-'],capture_output=True,text=True).stdout; L=[l for l in t.splitlines() if l.strip()]
    yrs=[int(x) for x in re.findall(r'\b(20\d\d)\b',next(l for l in L if re.search(r'\b20\d\d\s+20\d\d\s+20\d\d\b',l)))]
    f=[l for l in L if re.match(r'\s*Fallow\s',l)]; assert len(f)==2,(ry,f)
    gw=next(l for l in L if 'GROSS WT' in l); nw=next(l for l in L if 'NET WATER TOLL' in l); ts=next(l for l in L if 'TOTAL SECOND CROP' in l)
    i=next(i for i,l in enumerate(L) if l.strip()=='FALLOWED'); av=re.findall(r'([\d,]*)\s*Avg',L[i+1])
    get=lambda l:[num(x) for x in re.findall(r'\d[\d,]*',l)]
    for j,y in enumerate(yrs):
        rec=dict(report=ry,year=y,first=get(f[0])[j],second=get(f[1])[j],gross=get(gw)[j],net=get(nw)[j],tot2=get(ts)[j],prog=num(av[j]) if av[j] else np.nan); rows.append(rec)
R=pd.DataFrame(rows); print('arithmetic check, Gross - Total Second Crop = Net, all rows:',bool((R.gross-R.tot2==R.net).all()))
chk=R.groupby('year').agg(first_v=('first','nunique'),second_v=('second','nunique'),prog_v=('prog','nunique'),reports=('report','nunique')); print('years where reports disagree:',chk[(chk.first_v>1)|(chk.second_v>1)|(chk.prog_v>1)].to_string() if ((chk.first_v>1)|(chk.second_v>1)|(chk.prog_v>1)).any() else 'none')
P=R.sort_values('report').groupby('year').last()[['first','second','prog']]; P['net_fallow']=P['first']-P['second']; P.to_csv(f'{od}/pvid_report_totals.csv'); print(P.to_string())
A=pd.read_csv(f'{od}/area_by_product.csv').drop(columns='pvid_report_ac').set_index('wy'); Y=A.index
P=P.astype(float); A['pvid_gross_fallow']=P.loc[Y,'first']; A['pvid_net_fallow']=P.loc[Y,'net_fallow']; A['pvid_program_avg']=P.loc[Y,'prog']
A['program_avg_wy_aligned']=[.25*P.loc[y-1,'prog']+.75*P.loc[y,'prog'] if (y-1) in P.index else np.nan for y in Y]
for c in ['map_ac','dwr_ac']: A[c+'_vs_net_pct']=(100*(A[c]/A.pvid_net_fallow-1)).round(1)
A['map_vs_program_wy_pct']=(100*(A.map_ac/A.program_avg_wy_aligned-1)).round(1)
A.round(0).to_csv(f'{od}/area_by_product_with_pvid.csv'); print(A[['map_ac','dwr_ac','cdl_ac','pvid_gross_fallow','pvid_net_fallow','pvid_program_avg','program_avg_wy_aligned']].round(0).astype('Int64').to_string()); print(A[['map_ac_vs_net_pct','dwr_ac_vs_net_pct','map_vs_program_wy_pct']].to_string())
for ref in ['pvid_net_fallow','pvid_gross_fallow','pvid_program_avg']:
    print(ref,{c:(round(float(np.corrcoef(A[c],A[ref])[0,1]),3),round(float(np.mean(np.abs(A[c]-A[ref]))))) for c in ['map_ac','dwr_ac','cdl_ac']})
SURF,INK2,GR='#fcfcfb','#52514e','#e4e3df'; S=[('map_ac','Frozen map','#2a78d6','o'),('dwr_ac','DWR idle + unclassified fallow','#eb6834','s'),('cdl_ac','CDL Fallow/Idle','#1baf7a','^'),('pvid_net_fallow','PVID net fallow (first crop minus second crop)','#eda100','D')]
plt.rcParams.update({'font.size':11,'axes.edgecolor':GR,'axes.labelcolor':INK2,'xtick.color':INK2,'ytick.color':INK2,'figure.facecolor':SURF,'axes.facecolor':SURF,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.color':GR,'axes.axisbelow':True,'xtick.bottom':False,'ytick.left':False})
fig,ax=plt.subplots(figsize=(10,5.6))
for c,l,col,mk in S: ax.plot(Y,A[c],color=col,lw=2,marker=mk,ms=8,mec=SURF,mew=1.5,label=l)
ax.set_ylim(0); ax.set_xticks(list(Y)); ax.set_ylabel('Fallow area (acres)'); ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter('{x:,.0f}'))
fig.suptitle("The map closely tracks DWR and PVID's net fallow; CDL does not after 2018",x=.02,ha='left',fontsize=15,weight='medium')
fig.text(.02,.90,'Palo Verde, 2018 to 2023. Map, DWR and CDL are water-year polygon acres; PVID figures are calendar-year water toll acres.',color=INK2)
ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.09),ncol=2); fig.tight_layout(rect=(0,0,1,.88)); fig.savefig(f'{od}/fig_product_comparison.png',dpi=170)
