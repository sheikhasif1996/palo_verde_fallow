"""Register PVID map page images to the georeferenced 2021 page and extract fallow fields, without using DWR.
usage: 27_register_pvid_maps.py REF_TIF PAGE_PNG OUT_TIF OFFICIAL_ACRES
Search: scale and small rotation by normalized cross-correlation of the grey backdrop with fallow fill masked. Yellow fill area provides a scale-independent check."""
import sys, numpy as np, rasterio
from PIL import Image
from scipy.ndimage import affine_transform, binary_closing, binary_opening, uniform_filter
ref, page, out, off = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
SMIN,SMAX = (float(sys.argv[7]),float(sys.argv[8])) if len(sys.argv)>8 else (.94,1.18)
DX,DY = float(sys.argv[5]), float(sys.argv[6])   # correction (m, east and north) moving the 2021 reference page onto the validated aligned 2021 layer
def gray(a): return a[...,:3].mean(-1).astype('float32')
def yellow(a):
    r,g,b=[a[...,i].astype('float32') for i in range(3)]; return (r>170)&(g>165)&(b<120)&(np.abs(r-g)<70)
with rasterio.open(ref) as s: Rf=np.moveaxis(s.read(),0,-1); T=s.transform; crs=s.crs
Sp=np.array(Image.open(page).convert('RGB'))
def prep(a):
    g=gray(a); m=yellow(a); g[m]=np.nan; hi=g-uniform_filter(np.nan_to_num(g,nan=np.nanmean(g)),25); hi[~np.isfinite(g)]=0; return hi
R0=prep(Rf); S0=prep(Sp); print('ref',R0.shape,'page',S0.shape,flush=True)
k=4; R=R0[::k,::k]; Sx=S0[::k,::k]; H,W=R.shape
FR=np.fft.rfft2(R-R.mean(),s=(H*2,W*2))
def ncc(s,th):
    c,sn=np.cos(th),np.sin(th); A=np.array([[c,-sn],[sn,c]])/s; w=affine_transform(Sx,A,offset=0,output_shape=(H,W),order=1,mode='constant'); f=np.fft.rfft2(w-w.mean(),s=(H*2,W*2)); x=np.fft.irfft2(FR*np.conj(f),s=(H*2,W*2)); i=np.unravel_index(np.argmax(x),x.shape); d=(i[0] if i[0]<H else i[0]-2*H, i[1] if i[1]<W else i[1]-2*W); return x[i]/(np.linalg.norm(R-R.mean())*np.linalg.norm(w-w.mean())),d
best=(-1,None)
for s in np.arange(SMIN,SMAX,.004):
    for deg in np.arange(-.8,.81,.2):
        v,d=ncc(s,np.radians(deg))
        if v>best[0]: best=(v,(s,deg,d))
print('coarse',best,flush=True); v,(s,deg,d)=best
# fine search at full resolution over small window: scale and rotation jitter, translation by correlation on a 2x-decimated grid
k=2; R=R0[::k,::k]; Sx=S0[::k,::k]; H,W=R.shape; FR=np.fft.rfft2(R-R.mean(),s=(H*2,W*2)); b2=(-1,None)
for s2 in np.arange(s-.006,s+.0061,.001):
    for d2 in np.arange(deg-.2,deg+.201,.05):
        v,dd=ncc(s2,np.radians(d2))
        if v>b2[0]: b2=(v,(s2,d2,dd))
v,(s,deg,dd)=b2; print('fine ncc %.4f scale %.4f rot %.3f shift(px in ref) %s'%(v,s,deg,(dd[0]*k,dd[1]*k)),flush=True)
c,sn=np.cos(np.radians(deg)),np.sin(np.radians(deg)); A=np.array([[c,-sn],[sn,c]])/s
tgt=np.zeros((R0.shape[0],R0.shape[1])); Y=yellow(Sp).astype('float32')
offs=np.array([-dd[0]*k,-dd[1]*k],float); offs=-A@np.array([dd[0]*k,dd[1]*k],float)
yw=affine_transform(Y,A,offset=offs,output_shape=R0.shape,order=1,mode='constant')>.5; yw=binary_opening(binary_closing(yw,iterations=2),iterations=1)
a=abs(T.a*T.e); ac=yw.sum()*a/4046.8564; print('yellow acres from registered raster %.0f | official %.0f | ratio %.3f'%(ac,off,ac/off),flush=True)
T2=rasterio.Affine(T.a,T.b,T.c+DX,T.d,T.e,T.f+DY)
rgb=np.stack([affine_transform(Sp[...,i].astype('float32'),A,offset=offs,output_shape=R0.shape,order=1,mode='constant') for i in range(3)],0).astype('uint8')
with rasterio.open(out.replace('.tif','_rgb.tif'),'w',driver='GTiff',width=R0.shape[1],height=R0.shape[0],count=3,dtype='uint8',crs=crs,transform=T2,compress='deflate') as d_: d_.write(rgb)
with rasterio.open(out,'w',driver='GTiff',width=R0.shape[1],height=R0.shape[0],count=1,dtype='uint8',crs=crs,transform=T2,compress='deflate') as d_: d_.write(yw.astype('uint8'),1)
