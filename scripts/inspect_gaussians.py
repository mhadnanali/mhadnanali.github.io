"""Inspect a standard official 3DGS point_cloud.ply without requiring CUDA.

pip install numpy matplotlib plyfile
python scripts/inspect_gaussians.py --ply "C:\\path\\to\\point_cloud.ply" --out "local_figures"

Outputs plots and a text summary; does not change the source PLY.
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_cloud(a,b,title,xlab,ylab,out):
    fig,ax=plt.subplots(figsize=(7,5))
    ax.scatter(a,b,s=.35,alpha=.6,rasterized=True)
    ax.set(xlabel=xlab,ylabel=ylab,title=title)
    ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out,dpi=170);plt.close(fig)

def hist(values,label,out):
    finite=values[np.isfinite(values)]
    if len(finite)==0:return
    fig,ax=plt.subplots(figsize=(7,4));ax.hist(finite,bins=65)
    ax.set(title=f'Distribution of {label}',xlabel=label,ylabel='Gaussians (sample)')
    fig.tight_layout();fig.savefig(out,dpi=170);plt.close(fig)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--ply',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path('local_figures'))
    ap.add_argument('--max-points',type=int,default=30000)
    args=ap.parse_args()
    from plyfile import PlyData
    if not args.ply.is_file():ap.error(f'PLY not found: {args.ply}')
    args.out.mkdir(parents=True,exist_ok=True)
    ply=PlyData.read(str(args.ply))
    if 'vertex' not in ply:ap.error('PLY has no vertex element')
    v=ply['vertex'].data;n=len(v);names=set(v.dtype.names or [])
    report=[f'File: {args.ply}',f'Number of vertices/Gaussians: {n:,}',
            'Properties: '+', '.join(v.dtype.names or ())]
    print('\n'.join(report))
    if not {'x','y','z'}<=names:ap.error('PLY lacks x,y,z')
    rng=np.random.default_rng(2026)
    ids=rng.choice(n,min(n,max(1,args.max_points)),replace=False) if n else np.array([],dtype=int)
    if len(ids):
        x=np.asarray(v['x'][ids],float);y=np.asarray(v['y'][ids],float);z=np.asarray(v['z'][ids],float)
        plot_cloud(x,y,'Sampled Gaussian centers — XY','X','Y',args.out/'gaussians_xy.png')
        plot_cloud(x,z,'Sampled Gaussian centers — XZ','X','Z',args.out/'gaussians_xz.png')
    if 'opacity' in names and len(ids):
        raw=np.asarray(v['opacity'][ids],float)
        opacity=1/(1+np.exp(-np.clip(raw,-80,80)))
        report.append('Activated opacity sampled min/mean/max: '+str(np.round([opacity.min(),opacity.mean(),opacity.max()],4)))
        hist(opacity,'activated opacity (sigmoid(raw opacity))',args.out/'gaussians_opacity.png')
    scale_names=[f'scale_{j}' for j in range(3)]
    if set(scale_names)<=names and len(ids):
        scales=np.stack([np.exp(np.clip(np.asarray(v[k][ids],float),-30,30)) for k in scale_names],axis=1)
        report.append('Activated scale sampled median per axis: '+str(np.round(np.median(scales,axis=0),6)))
        hist(scales.max(axis=1),'maximum activated axis scale',args.out/'gaussians_scale.png')
    if {'f_dc_0','f_dc_1','f_dc_2'}<=names:
        report.append('Found RGB spherical-harmonic DC coefficients (not ordinary RGB vertex colors).')
    if {'red','green','blue'}<=names:
        report.append('Found ordinary vertex colors; this may be a generic point cloud rather than standard Gaussian PLY.')
    (args.out/'summary.txt').write_text('\n'.join(report)+'\n',encoding='utf8')
    print('\nGenerated figures and summary in:',args.out.resolve())

if __name__=='__main__':main()
