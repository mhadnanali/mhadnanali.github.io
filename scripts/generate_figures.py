"""Reproduce the static SVG figures used on the CV and 3DGS fundamentals pages.

Usage: python scripts/generate_figures.py
Requirements: pip install numpy matplotlib
Runs without COLMAP, CUDA, or internet access.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse, Rectangle, FancyArrowPatch, Polygon
from matplotlib.colors import to_rgba

ROOT = Path(__file__).resolve().parents[1] / 'docs' / 'notes'
CV = ROOT / 'cv-assets'
GS = ROOT / '3dgs' / 'assets'
CV.mkdir(parents=True,exist_ok=True)
GS.mkdir(parents=True,exist_ok=True)
INK='#243247'; BLUE='#3478B7'; ORANGE='#E19742'; GREEN='#3F9878'; PURPLE='#9268A8'; GREY='#64748B'; BG='#FAFBFD'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'svg.fonttype':'none'})

def canvas(w=9,h=3.9,xlim=(0,10),ylim=(0,5)):
    fig,ax=plt.subplots(figsize=(w,h));fig.patch.set_facecolor('white');ax.set_facecolor('white')
    ax.set(xlim=xlim,ylim=ylim);ax.set_aspect('equal',adjustable='box');ax.axis('off')
    return fig,ax

def save(fig,folder,name):
    fig.savefig(folder/name,format='svg',bbox_inches='tight',pad_inches=.18,facecolor='white')
    plt.close(fig)

def box(ax,x,y,w,h,text,color=BLUE,fs=10):
    ax.add_patch(Rectangle((x,y),w,h,edgecolor=color,facecolor=to_rgba(color,.07),linewidth=1.5,zorder=2))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=fs,color=INK,zorder=3)

def arrow(ax,p,q,color=GREY,ls='-',lw=1.6,head=10):
    a=FancyArrowPatch(p,q,arrowstyle='-|>',mutation_scale=head,color=color,linewidth=lw,linestyle=ls)
    ax.add_patch(a)

def label(ax,x,y,s,size=9,ha='center',color=INK,va='center'):
    ax.text(x,y,s,fontsize=size,ha=ha,va=va,color=color)

# 1 radiance pipeline
fig,ax=canvas(xlim=(0,12),ylim=(0,4))
box(ax,.3,1.4,2.3,1.15,'Scene surface\n(material + lighting)',ORANGE)
box(ax,3.4,1.4,2.2,1.15,'Outgoing radiance\n(direction-dependent)',BLUE)
box(ax,6.45,1.4,2.15,1.15,'Camera optics\n+ sensor',GREEN)
box(ax,9.25,1.4,2.35,1.15,'Pixel measurements\n(RGB image)',PURPLE)
for a,b in [(2.6,3.4),(5.6,6.45),(8.6,9.25)]:arrow(ax,(a,1.97),(b,1.97))
label(ax,5.95,3.45,'Radiance is not identical to a stored RGB value',12)
label(ax,5.95,.45,'Exposure, spectral response, white balance and tone mapping affect observed pixels',9,color=GREY)
save(fig,CV,'radiance-to-rgb.svg')

# 2 coordinate frames
fig,ax=canvas(xlim=(0,11),ylim=(0,5.6))
origin=(2.3,1.45); cam=(6.7,2.05); point=(9.2,4.55)
for s,c in [((2.3,1.45),(4,1.45)),((2.3,1.45),(2.3,3.5)),((6.7,2.05),(8.65,2.05)),((6.7,2.05),(6.7,4.15))]:arrow(ax,s,c,BLUE if s[0]==2.3 else GREEN)
label(ax,4.15,1.42,'Xw',9);label(ax,2.2,3.65,'Zw',9);label(ax,8.85,2.04,'Xc',9);label(ax,6.62,4.3,'Zc',9)
ax.add_patch(Circle(point,.10,color=ORANGE));label(ax,9.3,4.9,'World point P',10)
arrow(ax,cam,point,ORANGE);arrow(ax,origin,cam,GREY,ls='--')
label(ax,2.4,.8,'World coordinate frame',10);label(ax,7.05,1.35,'Camera coordinate frame',10)
label(ax,5.6,5.25,'Same 3D point, different coordinates in each frame',11)
save(fig,CV,'camera-frames.svg')

# 3 depth vs Euclidean range
fig,ax=canvas(xlim=(0,9),ylim=(0,5))
cam=(1.4,1.2);p=(6.1,4)
arrow(ax,cam,(7.6,1.2),BLUE)
ax.plot([p[0],p[0]],[1.2,p[1]],linestyle='--',color=GREY)
arrow(ax,cam,p,ORANGE)
ax.scatter(*p,c=ORANGE,s=58)
label(ax,4.0,.83,'Depth Z (along optical axis)',10,color=BLUE)
label(ax,3.5,3.13,'Range = Euclidean distance',10,color=ORANGE)
label(ax,6.4,3.95,'P = (X,Y,Z)',10,ha='left');label(ax,1.3,.7,'Camera',10)
label(ax,4.5,4.7,'Depth and distance are not the same off-axis',12)
save(fig,CV,'depth-vs-range.svg')

# 4 triangulation
fig,ax=canvas(xlim=(0,11),ylim=(0,6))
ca=(2,1);cb=(8.8,1);p=(5.4,5)
for c,lab in [(ca,'Camera A'),(cb,'Camera B')]:
 ax.add_patch(Polygon([(c[0]-.24,c[1]-.24),(c[0]+.24,c[1]-.24),(c[0],c[1]+.24)],closed=True,color=BLUE))
 label(ax,c[0],.53,lab,10)
 ax.plot([c[0],p[0]],[c[1],p[1]],color=ORANGE,lw=2)
ax.scatter(*p,c=ORANGE,s=65)
label(ax,5.4,5.35,'3D point estimated from matching observations',10)
arrow(ax,(2.3,1),(8.5,1),GREEN)
label(ax,5.4,1.28,'Camera baseline',9,color=GREEN)
label(ax,5.5,5.75,'Triangulation uses camera poses + corresponding image rays',12)
save(fig,CV,'two-view-triangulation.svg')

# 5 SfM chain
fig,ax=canvas(xlim=(0,12),ylim=(0,3.6))
steps=[('Overlapping\nimages',.1),('Keypoints +\nmatches',2.55),('Estimate camera\nposes',5.0),('Triangulate\n3D points',7.45),('Bundle\nadjustment',9.9)]
for i,(s,x) in enumerate(steps):
 box(ax,x,1.3,2.0,1.07,s,[ORANGE,BLUE,GREEN,BLUE,PURPLE][i]);
 if i<4:arrow(ax,(x+2.0,1.82),(steps[i+1][1],1.82))
label(ax,5.95,3.1,'Typical incremental Structure-from-Motion workflow',12)
label(ax,5.95,.65,'Result: calibrated/registered views and sparse reconstructed 3D points',9,color=GREY)
save(fig,CV,'sfm-workflow.svg')

# 6 scene representations
fig,ax=canvas(w=10,h=4.2,xlim=(0,12),ylim=(0,4.8))
centers=[(1.55,2.4),(4.5,2.4),(7.5,2.4),(10.5,2.4)]
for cx,cy in centers:ax.add_patch(Rectangle((cx-1.15,cy-1.15),2.3,2.3,fill=False,edgecolor='#CBD5E1'))
rng=np.random.default_rng(12)
xy=rng.normal(size=(115,2))*.5;xy=xy[np.sum(xy**2,axis=1)<1]
ax.scatter(xy[:,0]+1.55,xy[:,1]+2.4,s=9,c=BLUE)
for dx in np.arange(-.8,.9,.38):
 for dy in np.arange(-.8,.9,.38):
  if dx**2+dy**2<.65:ax.add_patch(Rectangle((4.5+dx-.16,2.4+dy-.16),.32,.32,facecolor=to_rgba(ORANGE,.35),edgecolor=ORANGE,lw=.5))
poly=np.array([[6.7,1.85],[7.7,1.5],[8.25,2.45],[7.8,3.3],[6.7,3.05],[6.7,1.85]])
ax.plot(poly[:,0],poly[:,1],color=GREEN,lw=2)
for i,j in [(0,3),(1,3),(1,4)]:ax.plot([poly[i,0],poly[j,0]],[poly[i,1],poly[j,1]],color=GREEN,lw=.7)
for x,y,sx,sy,ang in [(10.1,2.6,.65,.25,32),(10.9,2.1,.53,.34,-24),(10.6,2.9,.3,.65,60)]:
 ax.add_patch(Ellipse((x,y),sx*2,sy*2,angle=ang,facecolor=to_rgba(PURPLE,.19),edgecolor=to_rgba(PURPLE,.65)))
for (cx,cy),txt in zip(centers,['Point cloud','Voxel grid','Triangle mesh','Gaussian primitives']):label(ax,cx,.85,txt,10)
label(ax,6,4.3,'Different representations of the same conceptual 3D structure',12)
save(fig,CV,'scene-representations.svg')

# 7 nerf vs 3dgs
fig,ax=canvas(xlim=(0,12),ylim=(0,4.5))
box(ax,.65,2.35,2.3,1.1,'3D coordinate\n+ direction',BLUE)
box(ax,3.8,2.35,2.1,1.1,'MLP / feature\nencoding',BLUE)
box(ax,6.78,2.35,2.05,1.1,'Density + color\nsamples',BLUE)
box(ax,9.6,2.35,1.95,1.1,'Ray integration',BLUE)
for s,e in [(2.95,3.8),(5.9,6.78),(8.83,9.6)]:arrow(ax,(s,2.9),(e,2.9))
box(ax,.65,.55,2.3,1.0,'Stored 3D\nGaussians',ORANGE)
box(ax,3.8,.55,2.1,1.0,'Project onto\nimage plane',ORANGE)
box(ax,6.78,.55,2.05,1.0,'Rasterized\nsplats',ORANGE)
box(ax,9.6,.55,1.95,1.0,'Alpha composite',ORANGE)
for s,e in [(2.95,3.8),(5.9,6.78),(8.83,9.6)]:arrow(ax,(s,1.05),(e,1.05))
label(ax,.30,2.9,'NeRF',9,ha='left',color=BLUE);label(ax,.30,1.05,'3DGS',9,ha='left',color=ORANGE)
label(ax,6.0,4.17,'Classic NeRF (implicit) vs. vanilla 3DGS (explicit)',12)
save(fig,CV,'nerf-vs-3dgs.svg')

# 8 raymarch vs raster
fig,ax=canvas(xlim=(0,12),ylim=(0,4.9))
for x,heading in [(0,'Ray marching'),(6.15,'Rasterization / splatting')]:
 ax.add_patch(Rectangle((x+.18,.45),5.35,3.7,fill=False,edgecolor='#CBD5E1'))
 label(ax,x+2.85,4.5,heading,11)
# ray
cam=(.75,1.2);tgt=(4.85,3.42)
arrow(ax,cam,tgt,BLUE,lw=2)
for a in np.linspace(.2,.88,8):
 p=np.array(cam)*(1-a)+np.array(tgt)*a
 ax.add_patch(Circle(p,.10,facecolor=ORANGE,edgecolor='white',lw=.4))
label(ax,3,.75,'Samples along a ray',9)
# projection
ax.add_patch(Ellipse((8.1,2.4),1.95,1.0,angle=28,edgecolor=ORANGE,facecolor=to_rgba(ORANGE,.16),lw=2))
ax.plot([9.55,9.55],[.85,3.75],color=BLUE,lw=3)
arrow(ax,(8.8,2.5),(10.2,2.5),ORANGE)
ax.add_patch(Ellipse((10.55,2.5),.55,1.4,edgecolor=PURPLE,facecolor=to_rgba(PURPLE,.25)))
label(ax,10.55,1,'2D footprint',9)
save(fig,CV,'raymarch-vs-splat.svg')

# 9 Gauss anatomy
fig,ax=canvas(xlim=(0,12),ylim=(0,4.7))
for idx,(cx,sx,sy,ang,txt) in enumerate([(2,.9,.9,0,'Isotropic'),(6,1.45,.53,20,'Anisotropic'),(10,1.25,.4,-50,'Rotated anisotropic')]):
 ax.add_patch(Ellipse((cx,2.4),2*sx,2*sy,angle=ang,edgecolor=ORANGE,facecolor=to_rgba(ORANGE,.14),linewidth=2))
 ax.scatter([cx],[2.4],c=BLUE,s=30,zorder=4)
 label(ax,cx,.75,txt,10)
label(ax,6,4.22,'Equal-value ellipses are 2D illustrations of 3D covariance',12)
label(ax,6,3.72,'Center is identical; shape/orientation may differ',9,color=GREY)
save(fig,GS,'isotropic-anisotropic.svg')

# 10 initialization
fig,ax=canvas(xlim=(0,12),ylim=(0,4))
xy=np.array([(1.15,1.9),(1.7,2.3),(2.2,1.65),(2.6,2.1),(1.2,2.75),(2.6,2.85)])
ax.scatter(xy[:,0],xy[:,1],color=BLUE,s=35)
box(ax,4.12,1.62,3.3,1.1,'Initialize centers, scale,\nrotation, opacity, SH',GREEN)
arrow(ax,(2.95,2.15),(4.06,2.15));arrow(ax,(7.45,2.15),(8.15,2.15))
for (x,y),a in zip(xy+np.array([7.3,0]),[10,-15,30,0,-20,45]):
 ax.add_patch(Ellipse((x,y),.45,.2,angle=a,facecolor=to_rgba(ORANGE,.2),edgecolor=ORANGE,lw=1))
label(ax,1.9,.9,'COLMAP 3D points',10);label(ax,9.7,.9,'Initial Gaussian primitives',10)
label(ax,6,3.68,'Each reconstructed point seeds one trainable primitive',12)
save(fig,GS,'point-to-gaussians.svg')

# 11 3D-to-2D approximation
fig,ax=canvas(xlim=(0,12),ylim=(0,4.9))
ax.add_patch(Ellipse((2.8,2.6),2.5,1.0,angle=25,facecolor=to_rgba(ORANGE,.15),edgecolor=ORANGE,lw=2))
ax.scatter([2.8],[2.6],c=BLUE,s=30)
ax.plot([7.3,7.3],[.6,4.2],color=BLUE,lw=3)
for y in np.arange(.8,4.1,.4):ax.plot([7.28,7.33],[y,y],color=BLUE,lw=.4)
arrow(ax,(4.2,2.6),(6.6,2.6),GREEN)
ax.add_patch(Ellipse((9.55,2.5),2.15,1.05,angle=-12,edgecolor=ORANGE,facecolor=to_rgba(ORANGE,.2),lw=2))
label(ax,2.9,1.35,'3D Gaussian\n(world space)',10)
label(ax,5.5,3.15,'Project with camera\nand local Jacobian',9,color=GREEN)
label(ax,7.28,4.5,'Image plane',10,color=BLUE)
label(ax,9.55,1.3,'2D elliptical footprint\n(screen space)',10)
label(ax,6,4.8,'A 3D Gaussian contributes to an area of image pixels',12)
save(fig,GS,'gaussian-projection.svg')

# 12 alpha color contribution layers
fig,ax=canvas(xlim=(0,12),ylim=(0,4.6))
centers=[(2.5,2.3),(5.5,2.3),(8.5,2.3)]
for (cx,cy),col,alpha,num in zip(centers,[BLUE,ORANGE,GREEN],[.6,.5,.4],[1,2,3]):
 ax.add_patch(Ellipse((cx,cy),2,1.3,facecolor=to_rgba(col,alpha*.5),edgecolor=col,lw=2))
 label(ax,cx,.96,f'Layer {num} : alpha={alpha}',9)
for x in [4.0,7.0]:arrow(ax,(x,2.3),(x+.2,2.3),GREY)
label(ax,6,4,'Front-to-back accumulation of overlapping contributions',12)
label(ax,6,.35,'Visible contribution = remaining transmittance × current alpha × current color',9,color=GREY)
save(fig,GS,'alpha-layers.svg')

# 13 training
fig,ax=canvas(xlim=(0,12),ylim=(5.8,-.4))
steps=[('Current Gaussian\nparameters',1.3,0.45),('Render predicted\nimage',4.75,0.45),('Compare with\nground truth',8.3,.45),('Compute image\nloss',8.3,3.25),('Backpropagate\ngradients',4.75,3.25),('Optimizer updates\nGaussian params',1.3,3.25)]
for i,(txt,x,y) in enumerate(steps):box(ax,x,y,2.55,1.13,txt,[BLUE,ORANGE,GREEN,PURPLE,PURPLE,BLUE][i],fs=8.6)
for p,q in [((3.85,1),(4.75,1)),((7.3,1),(8.3,1)),((9.575,1.58),(9.575,3.25)),((8.3,3.8),(7.3,3.8)),((4.75,3.8),(3.85,3.8)),((2.575,3.25),(2.575,1.58))]:arrow(ax,p,q)
label(ax,6,5.1,'Repeated optimization loop (one sampled camera view per iteration in standard training)',11)
save(fig,GS,'training-cycle.svg')

# 14 clone/split/prune
fig,ax=canvas(xlim=(0,12),ylim=(0,4.8))
for c,title in [(2,'Clone: add similar primitive'),(6,'Split: replace large primitive'),(10,'Prune: remove weak primitive')]:
 ax.add_patch(Rectangle((c-1.7,.45),3.4,3.55,fill=False,edgecolor='#CBD5E1'))
 label(ax,c,4.35,title,10)
if True:
 ax.add_patch(Ellipse((1.3,2.35),.75,.4,facecolor=to_rgba(BLUE,.2),edgecolor=BLUE))
 arrow(ax,(1.75,2.35),(2.15,2.35))
 for x,y in [(2.5,2.55),(2.95,2.17)]:ax.add_patch(Ellipse((x,y),.6,.33,facecolor=to_rgba(BLUE,.2),edgecolor=BLUE))
 ax.add_patch(Ellipse((5.28,2.3),1.55,.88,facecolor=to_rgba(ORANGE,.22),edgecolor=ORANGE))
 arrow(ax,(6.05,2.3),(6.45,2.3))
 for x,y in [(6.9,2.65),(7.1,2.1)]:ax.add_patch(Ellipse((x,y),.55,.33,facecolor=to_rgba(ORANGE,.3),edgecolor=ORANGE))
 ax.add_patch(Ellipse((9.43,2.3),.83,.46,facecolor=to_rgba(GREEN,.08),edgecolor=GREEN,linestyle='--'))
 arrow(ax,(9.95,2.3),(10.45,2.3));ax.text(11.1,2.3,'×',fontsize=26,color='#A53A3A',ha='center',va='center')
label(ax,6,.13,'Schematic only: actual clone/split/prune decisions depend on training statistics and thresholds',9,color=GREY)
save(fig,GS,'densification.svg')

# 15 multiview
fig,ax=canvas(xlim=(0,12),ylim=(0,4.9))
p=(6,3.5);ax.scatter(*p,c=ORANGE,s=70);label(ax,6,4,'Shared 3D Gaussian',11)
for cx,lab in [(2.1,'Camera A'),(9.9,'Camera B')]:
 c=(cx,.8);ax.add_patch(Polygon([(cx-.2,.55),(cx+.2,.55),(cx,.92)],closed=True,color=BLUE));
 arrow(ax,c,p,GREEN,lw=1.5)
 label(ax,cx,.27,lab,10)
label(ax,6,1.45,'Same learned parameters\nused for both rendered views',9)
label(ax,6,4.55,'Multiple image views constrain a shared 3D representation',12)
save(fig,GS,'multi-view.svg')

# 16 output separation
fig,ax=canvas(xlim=(0,12),ylim=(0,4.3))
box(ax,.2,1.25,2.4,1.55,'Trained Gaussians\n(in memory)',BLUE)
box(ax,3.45,1.25,2.4,1.55,'PLY file\n(stored attributes)',GREEN)
box(ax,6.7,1.25,2.4,1.55,'Gaussian renderer\n+ selected camera',ORANGE)
box(ax,9.65,1.25,2.15,1.55,'Rendered RGB\nimage / viewer',PURPLE)
arrow(ax,(2.6,2.0),(3.45,2.0));arrow(ax,(5.85,2.0),(6.7,2.0));arrow(ax,(9.1,2.0),(9.65,2.0))
label(ax,6,3.7,'Representation, stored model and rendered image are different objects',12)
label(ax,6,.6,'A viewer projects the Gaussian model; the PLY itself is not an RGB image',9,color=GREY)
save(fig,GS,'ply-render-view.svg')

# 17 SH
fig,ax=canvas(xlim=(0,12),ylim=(0,4.8))
center=(5.9,2.6);ax.add_patch(Circle(center,.43,facecolor=to_rgba(ORANGE,.25),edgecolor=ORANGE,lw=2))
for ang,col in [(10,BLUE),(70,GREEN),(145,PURPLE),(210,BLUE),(290,ORANGE)]:
 t=np.deg2rad(ang);dx=1.7*np.cos(t);dy=1.7*np.sin(t)
 arrow(ax,center,(center[0]+dx,center[1]+dy),col,lw=2)
label(ax,6,.28,'View direction changes the color evaluated from fixed SH coefficients',10)
label(ax,6,4.67,'Spherical harmonics encode directional appearance, not extra spatial points',12)
save(fig,GS,'sh-directions.svg')
# 18 overall 3DGS pipeline diagram
fig,ax=canvas(xlim=(0,12),ylim=(0,3.75))
steps=[('Input images',.15,ORANGE),('COLMAP pose\n+ sparse XYZ',2.58,BLUE),('Initialize\nGaussians',5.0,GREEN),('Render +\noptimize',7.42,PURPLE),('PLY / novel\nview render',9.85,ORANGE)]
for i,(s,x,c) in enumerate(steps):
    box(ax,x,1.35,1.95,1.1,s,c,fs=9)
    if i<4:arrow(ax,(x+1.95,1.9),(steps[i+1][1],1.9))
label(ax,6,3.25,'Input → cameras & points → trainable representation → rendering',12)
label(ax,6,.65,'Densification and pruning happen during the optimization loop',9,color=GREY)
save(fig,GS,'full-pipeline.svg')

print('Generated SVG figures into:',CV,'and',GS)
