"""Inspect text-format COLMAP sparse model: cameras.txt, images.txt, points3D.txt.

Convert COLMAP binary sparse model to TXT in COLMAP GUI (Export model as text)
or using the COLMAP `model_converter` command, before running this script.

pip install numpy matplotlib
python scripts/inspect_colmap.py --model "C:\\path\\to\\sparse_text" --out "local_colmap_figures"
"""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def data_lines(path):
    if not path.is_file():raise FileNotFoundError(path)
    with path.open(encoding='utf8') as f:
        return [line.strip() for line in f if line.strip() and not line.lstrip().startswith('#')]

def quat_rot(q):
    w,x,y,z=np.asarray(q,float);norm=np.linalg.norm([w,x,y,z])
    if norm==0:raise ValueError('Invalid quaternion')
    w,x,y,z=np.array([w,x,y,z])/norm
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                     [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                     [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--out',type=Path,default=Path('local_colmap_figures'))
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    cams=data_lines(args.model/'cameras.txt')
    poses=[line.strip() for line in (args.model/'images.txt').read_text(encoding='utf8').splitlines() if not line.lstrip().startswith('#')]
    pts=data_lines(args.model/'points3D.txt')
    # images.txt records alternate: pose/image information line, followed by 2D keypoint observations line.
    centers=[];image_names=[]
    for line in poses[::2]:
        p=line.split(maxsplit=9)
        if len(p)<10:continue
        q=np.array(list(map(float,p[1:5])));t=np.array(list(map(float,p[5:8])))
        try:centers.append(-quat_rot(q).T@t);image_names.append(p[9])
        except ValueError:pass
    xyz=[];rgb=[]
    for line in pts:
        p=line.split()
        if len(p)>=7:
            try:
                xyz.append(list(map(float,p[1:4])));rgb.append(list(map(int,p[4:7])))
            except ValueError:pass
    xyz=np.asarray(xyz,dtype=float).reshape(-1,3);rgb=np.asarray(rgb,dtype=float).reshape(-1,3)/255
    centers=np.asarray(centers,dtype=float).reshape(-1,3)
    report=f'COLMAP models: {len(cams)} camera intrinsics records, {len(centers)} registered images, {len(xyz)} sparse 3D points.'
    print(report)
    fig,ax=plt.subplots(figsize=(8,6))
    if len(xyz):
        idx=np.random.default_rng(2026).choice(len(xyz),min(25000,len(xyz)),replace=False)
        ax.scatter(xyz[idx,0],xyz[idx,1],c=rgb[idx],s=.6,alpha=.7,rasterized=True,label='COLMAP sparse 3D points')
    if len(centers):ax.scatter(centers[:,0],centers[:,1],marker='^',s=35,label='Registered camera centers')
    ax.set(title='COLMAP reconstruction — XY overview',xlabel='X',ylabel='Y')
    ax.set_aspect('equal',adjustable='datalim');ax.grid(alpha=.15);ax.legend(loc='best')
    fig.tight_layout();fig.savefig(args.out/'colmap_xy.png',dpi=180);plt.close(fig)
    (args.out/'summary.txt').write_text(report+'\n',encoding='utf8')
    print('Output:',args.out.resolve())

if __name__=='__main__':main()
