"""Plot recorded fabric positions in the actual gripper frame for grasp diagnosis."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

OUTPUT=OUTPUT_ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path)
    parser.add_argument('--time',type=float,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert args.episode.resolve().is_relative_to(OUTPUT) and args.out.resolve().is_relative_to(OUTPUT)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    os.environ['MPLCONFIGDIR']=str(OUTPUT/'plot-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    rows=[json.loads(s) for s in (args.episode/'trace.jsonl').read_text().splitlines()]
    index=min(range(len(rows)),key=lambda i:abs(rows[i]['time_s']-args.time))
    row=rows[index]
    topology=json.loads((args.episode/'topology.json').read_text())
    key='upper' if 'upper' in topology['cloth'] else 'coupon'
    field='cloth_'+key if key!='coupon' else 'cloth_nodes'
    with np.load(args.episode/'control-steps.npz') as arrays:
        nodes=arrays[field][index]
    pose=row['left_link_poses_xyzw'][row['left_link_names'].index('gripper')]
    x,y,z,w=np.asarray(pose[3:],dtype=np.float64)
    rotation=np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                       [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                       [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    local=(nodes-np.asarray(row['measured_tcp_m']))@rotation*1000
    half_gap=float(row['joint_positions'][6])*.04695*1000
    fig,axes=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
    for ax,side,label,lo,hi in ((axes[0],2,'Approach axis, mm',-15,13),
                               (axes[1],1,'Across pad, mm',-10,10)):
        ax.scatter(local[:,side],local[:,0],s=8,c=local[:,2],cmap='viridis',alpha=.75)
        ax.add_patch(Rectangle((lo,-max(half_gap,.15)),hi-lo,2*max(half_gap,.15),
                              fill=False,color='red',linestyle='--',label='Nominal open gap / pad extent'))
        ax.axhline(0,color='gray',linewidth=.5)
        ax.axvline(0,color='gray',linewidth=.5)
        ax.set(xlabel=label,ylabel='Closing axis, mm',xlim=(-30,40),ylim=(-40,40))
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8,loc='lower right')
    fig.suptitle(f'{args.episode.parent.name}: {row["time_s"]:.1f} s, {row["control"]["phase"]}')
    fig.savefig(args.out.with_suffix('.png'),dpi=170)
    fig.savefig(args.out.with_suffix('.pdf'))
    plt.close(fig)
    args.out.with_suffix('.json').write_text(json.dumps({
        'episode':str(args.episode),'frame':index,'time_s':row['time_s'],
        'actual_normalized_gripper':row['joint_positions'][6],
        'trace_sha256':hashlib.sha256((args.episode/'trace.jsonl').read_bytes()).hexdigest(),
        'interpretation':'Diagnostic projection, not a collision or successful-grasp certificate. Dashed box is nominal; scorer uses actual collision hulls.'
    },indent=2)+'\n')


if __name__=='__main__':
    main()
