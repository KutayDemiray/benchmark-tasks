"""Plot measured snap insertion, alignment and contact loads from robot diagnostics."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from robot_geometry import rotation

OUTPUT=OUTPUT_ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episodes',type=Path,nargs='+')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert args.out.resolve().is_relative_to(OUTPUT)
    os.environ['MPLCONFIGDIR']=str(OUTPUT/'plot-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    evidence=[]
    handles=[]
    for episode in args.episodes:
        assert episode.resolve().is_relative_to(OUTPUT)
        path=episode/'physics.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        setup=json.loads((episode/'setup.json').read_text());cfg=setup['parameters']
        measured=[]
        for row in rows:
            male=np.array(row['hardware']['male']);socket=np.array(row['hardware']['socket'])
            R=rotation(socket[3:]);local=R.T@(male[:3]-socket[:3])
            angle=np.degrees(np.arccos(np.clip((R.T@rotation(male[3:]))[2,2],-1,1)))
            forces=row.get('male_robot_link_normal_forces_N',row['male_robot_normal_forces_N'])
            load=np.linalg.norm(forces,axis=1).sum()
            measured.append([row['physics_step']/cfg['simulation']['physics_hz'],
                             local[2]*1000,np.linalg.norm(local[:2])*1000,angle,load])
        data=np.array(measured)
        label=episode.parent.name+' / '+episode.name
        if not (episode/'result.json').exists():label+=' (interrupted)'
        for column,axis in enumerate(axes.flat,1):
            line,=axis.plot(data[:,0],data[:,column],lw=.8,label=label)
            if column==1:handles.append(line)
        evidence.append({'episode':str(episode),'physics_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                         'physics_samples':len(rows),'full_task_success':False})
    axes[0,0].axhline(7.2,color='black',ls='--',lw=.8,label='seating upper depth bound')
    axes[0,1].axhline(.75,color='black',ls='--',lw=.8)
    axes[1,0].axhline(12,color='black',ls='--',lw=.8)
    axes[1,1].axhline(25,color='black',ls='--',lw=.8)
    for axis,label in zip(axes.flat,('Axial cap/socket separation (mm)','Lateral misalignment (mm)',
                                   'Relative normal angle (degrees)','Robot rigid normal contact (N)')):
        axis.set_ylabel(label);axis.set_xlabel('Simulation time (s)');axis.grid(alpha=.2)
    fig.suptitle('Ordinary YAM pressing: measured outcomes, isolated diagnostics')
    fig.legend(handles,[h.get_label() for h in handles],loc='lower center',fontsize=8)
    fig.tight_layout(rect=(0,.10,1,.95))
    args.out.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(args.out,dpi=160)
    fig.savefig(args.out.with_suffix('.pdf'))
    args.out.with_suffix('.json').write_text(json.dumps({'inputs':evidence,
        'note':'Dashed physical limits are unchanged. No curve establishes full garment success.'},indent=2)+'\n')
    plt.close(fig)


if __name__=='__main__':
    main()
