"""One blinded image sheet per reference sample: 12 monthly Sentinel-2 true-colour chips plus a NAIP aerial photo.
usage: 13_make_image_sheets.py FIELDS REF_DIR [--limit N] [--months M]"""
import sys, argparse, warnings, os
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd, geopandas as gpd, rasterio
from rasterio.enums import Resampling as R
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
import planetary_computer as pc, pystac_client as ps
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')
a=argparse.ArgumentParser(); a.add_argument('fields'); a.add_argument('ref'); a.add_argument('--limit',type=int); a.add_argument('--months',type=int,default=12); A=a.parse_args()
ENV=dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',GDAL_HTTP_MAX_RETRY='4',GDAL_HTTP_RETRY_DELAY='3',VSI_CACHE='TRUE')
K=pd.read_csv(f'{A.ref}/key/DO_NOT_OPEN_key.csv'); 
if A.limit: K=K.groupby('wy').head(A.limit)
g=gpd.read_file(A.fields).set_index('field_id'); RES=10; NP=120
x0,y0,x1,y1=g.total_bounds; x0-=1000;y0-=1000;x1+=1000;y1+=1000; W=int((x1-x0)/RES); H=int((y1-y0)/RES); T=from_origin(x0,y1,RES,RES)
def box_of(r):
    b=g.geometry[r.field_id].bounds; h=max(b[2]-b[0],b[3]-b[1])/2+150; h=max(h,300); cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2; return cx-h,cy-h,cx+h,cy+h
B={r.sample_id:box_of(r) for r in K.itertuples()}
c=ps.Client.open('https://planetarycomputer.microsoft.com/api/stac/v1')
bb=list(gpd.GeoSeries.from_xy([x0,x1],[y0,y1],crs=32611).to_crs(4326).total_bounds)
def rd(href,nd=0):
    with rasterio.Env(**ENV),rasterio.open(href) as s,WarpedVRT(s,crs='EPSG:32611',transform=T,width=W,height=H,resampling=R.bilinear,src_nodata=nd,nodata=nd) as v: return v.read(1).astype('float32')
def mosaic(items):
    out=np.zeros((H,W,3),'float32')
    for it in sorted(items,key=lambda i:i.properties['s2:mgrs_tile']!='11SQT'):
        s=pc.sign(it); off=1000. if float(it.properties['s2:processing_baseline'])>=4 else 0.; a=[]
        for b in ['B04','B03','B02']: a.append(rd(s.assets[b].href))
        a=np.stack(a,-1); m=(out.sum(-1)==0)&(a.sum(-1)>0); out[m]=np.clip((a[m]-off)/10000.,0,1)
    return out
def month_scenes(y):
    its=[i for i in c.search(collections=['sentinel-2-l2a'],bbox=bb,datetime=f'{y-1}-10-01/{y}-09-30',query={'eo:cloud_cover':{'lte':60}}).items() if i.properties['s2:mgrs_tile'] in('11SQT','11SQS')]
    its=list({'_'.join(i.id.split('_')[:-1]):i for i in sorted(its,key=lambda i:i.id)}.values()); df={}
    for i in its: df.setdefault(i.datetime.strftime('%Y-%m-%d'),[]).append(i)
    pick={}
    for d,L in df.items():
        tiles={i.properties['s2:mgrs_tile'] for i in L}; sc=np.mean([i.properties['eo:cloud_cover'] for i in L])+(0 if '11SQT' in tiles else 100)+(0 if len(tiles)==2 else 20)
        k=d[:7]
        if k not in pick or sc<pick[k][0]: pick[k]=(sc,d,L)
    return pick
CH={r.sample_id:{} for r in K.itertuples()}; MONTHS={}
def do_month(args):
    y,k,(sc,d,L),ids=args
    try: M=mosaic(L)
    except Exception as e: print('FAILED',y,k,repr(e),flush=True); return
    for sid in ids:
        b=B[sid]; xs=((np.linspace(b[0],b[2],NP,endpoint=False)+RES/2-x0)/RES).astype(int); ys=((y1-(np.linspace(b[3],b[1],NP,endpoint=False)-RES/2))/RES).astype(int)
        xs=np.clip(xs,0,W-1); ys=np.clip(ys,0,H-1); ch=np.clip(M[np.ix_(ys,xs)]/0.32,0,1)**0.85; CH[sid][k]=((ch*255).astype('uint8'),d)
    print('month done',y,k,d,flush=True)
jobs=[]
for y,q in K.groupby('wy'):
    P=month_scenes(int(y)); months=[m.strftime('%Y-%m') for m in pd.date_range(f'{int(y)-1}-10-01',f'{int(y)}-09-01',freq='MS')][:A.months]; MONTHS[int(y)]=months
    jobs+=[(int(y),k,P[k],list(q.sample_id)) for k in months if k in P]
print('month mosaics',len(jobs),flush=True)
with ThreadPoolExecutor(4) as ex: list(ex.map(do_month,jobs))
NA={}
def naip(r):
    try:
        b=B[r.sample_id]; ll=gpd.GeoSeries.from_xy([b[0],b[2]],[b[1],b[3]],crs=32611).to_crs(4326).total_bounds
        its=list(c.search(collections=['naip'],bbox=list(ll),datetime='2014-01-01/2024-12-31').items())
        if not its: return
        mid=pd.Timestamp(f'{r.wy}-04-01',tz='UTC'); it=min(its,key=lambda i:abs(pd.Timestamp(i.datetime)-mid)); s=pc.sign(it)
        tr=from_origin(b[0],b[3],(b[2]-b[0])/480,(b[3]-b[1])/480)
        with rasterio.Env(**ENV),rasterio.open(s.assets['image'].href) as ds,WarpedVRT(ds,crs='EPSG:32611',transform=tr,width=480,height=480,resampling=R.bilinear) as v: NA[r.sample_id]=(np.moveaxis(v.read([1,2,3]),0,-1),it.datetime.strftime('%Y-%m-%d'))
    except Exception as e: print('NAIP failed',r.sample_id,repr(e),flush=True)
with ThreadPoolExecutor(8) as ex: list(ex.map(naip,K.itertuples()))
print('naip chips',len(NA),'of',len(K),flush=True)
def outline(ax,sid,fid,n):
    b=B[sid]; sx=n/(b[2]-b[0]); sy=n/(b[3]-b[1]); ge=g.geometry[fid]
    for p in (ge.geoms if hasattr(ge,'geoms') else [ge]):
        xy=np.array(p.exterior.coords); ax.plot((xy[:,0]-b[0])*sx,(b[3]-xy[:,1])*sy,color='#ffe600',lw=1.1)
for r in K.itertuples():
    months=MONTHS[int(r.wy)]; fig=plt.figure(figsize=(14,7.6),facecolor='white'); gs=fig.add_gridspec(3,6,left=.01,right=.99,top=.9,bottom=.01,wspace=.04,hspace=.14)
    fig.suptitle(f'{r.sample_id}   |   Water year {int(r.wy)} (Oct {int(r.wy)-1} to Sep {int(r.wy)})   |   yellow line = field to label',fontsize=14,x=.01,ha='left')
    for i,k in enumerate(months):
        ax=fig.add_subplot(gs[i//4,i%4]); ax.axis('off'); nm=pd.Timestamp(k+'-01').strftime('%b %Y')
        if k in CH[r.sample_id]:
            im,d=CH[r.sample_id][k]; ax.imshow(im,interpolation='nearest'); outline(ax,r.sample_id,r.field_id,NP); ax.set_xlim(0,NP); ax.set_ylim(NP,0); ax.set_title(f'{nm}  ({d})',fontsize=9)
        else: ax.text(.5,.5,'no clear image',ha='center',va='center',transform=ax.transAxes,color='#666'); ax.set_title(nm,fontsize=9)
    ax=fig.add_subplot(gs[:,4:]); ax.axis('off')
    if r.sample_id in NA:
        im,d=NA[r.sample_id]; ax.imshow(im,interpolation='bilinear'); outline(ax,r.sample_id,r.field_id,480); ax.set_xlim(0,480); ax.set_ylim(480,0); ax.set_title(f'NAIP aerial photo, {d}',fontsize=10)
    else: ax.text(.5,.5,'no aerial photo available',ha='center',va='center',transform=ax.transAxes,color='#666')
    fig.savefig(f'{A.ref}/chips/{r.sample_id}.png',dpi=95); plt.close(fig)
print('saved',len(K),flush=True)
