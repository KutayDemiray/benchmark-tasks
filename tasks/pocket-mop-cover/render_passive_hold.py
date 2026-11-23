"""Render a complete online passive hold with audited native measurements."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from matplotlib.patches import Polygon
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from audit_mechanism import capsule_proxy
from render_passive_contact import load


def render(run,capture,output):
    read=lambda name:json.loads((run/name).read_text())
    result=read('result.json');analysis=read('passive-analysis.json')
    cloth=read('passive-geometry-stride-1.json');rigid=read('passive-rigid-audit.json')
    comparison=json.loads((capture/'native-property-comparison.json').read_text())
    assert result['termination_reason']=='diagnostic_settled_hold'
    assert comparison['matched']==258 and comparison['different']==comparison['unrecorded']==0
    assert (run/'scene.usda').read_bytes()==(capture/'scene.usda').read_bytes()
    for folder in (run,capture):assert json.loads((folder/'gpu-verification.json').read_text())['verified']
    assert analysis['final_two_seconds']['speed_threshold_passed']
    assert cloth['complete_native_coverage'] and cloth['selected_states_passed']
    assert rigid['complete_native_coverage'] and rigid['rigid_checks_passed']
    data=load(run);hz=data['request']['physics_hz'];stride=hz//20;frames=len(data['poses'])
    assert frames*stride==result['physics_steps']
    rows=[json.loads(line) for line in (run/'steps.jsonl').read_text().splitlines()]
    assert len(rows)==result['physics_steps'] and len(rows)==result['first_passive_hold_control_step']*stride
    speed=np.array([[row['max_cloth_speed_m_s'],row['max_cloth_difference_speed_m_s']] for row in rows])
    speed=speed.reshape(frames,stride,2).max(axis=1)
    times=np.arange(1,frames+1)/20
    fig=plt.figure(figsize=(12.8,7.2),dpi=100,facecolor='#f5f7fa')
    grid=fig.add_gridspec(2,2,left=.07,right=.97,bottom=.13,top=.79,height_ratios=[2.,1.],wspace=.25,hspace=.45)
    axes=[fig.add_subplot(grid[0,0]),fig.add_subplot(grid[0,1])]
    trace=fig.add_subplot(grid[1,:]);panels=[]
    fig.suptitle('Passive release: a complete two-second settling hold',fontsize=19,y=.97)
    fig.text(.5,.91,'Same geometry and material; 258 selected native properties match the Kit reference',ha='center',fontsize=11)
    fig.text(.5,.06,'Measured native states at actual speed. Initialized diagnostic; no robot or full-task success.',ha='center',fontsize=10)
    colors=dict(Left='#8a929d',Right='#8a929d',Bridge='#5d6673',Handle='#4e5968',Rocker='#d67242',Clip='#dfad25')
    xy_min=data['nodes'][:,:,:2].min(axis=(0,1));xy_max=data['nodes'][:,:,:2].max(axis=(0,1))
    for shape in data['shapes']:
        if not shape['enabled']:continue
        points=capsule_proxy(shape['capsule'])[0] if 'capsule' in shape else np.asarray(shape['points'])
        body=data['names'].index(shape['body'])
        for pose in data['poses'][:,body]:
            world=Rotation.from_quat(pose[3:]).apply(points)+pose[:3]
            xy_min=np.minimum(xy_min,world[:,:2].min(axis=0));xy_max=np.maximum(xy_max,world[:,:2].max(axis=0))
    for ax,projection,title in zip(axes,([1,2],[0,1]),('Side detail: head and textile','Overhead: complete assembly')):
        ax.set_title(title,fontsize=12);ax.set_aspect('equal');ax.grid(alpha=.15)
        if projection==[1,2]:
            ax.set(xlim=(-.16,.10),ylim=(-.005,.07),xlabel='Width (m)',ylabel='Height (m)')
            ax.axhline(0,color='#5d6673',lw=2)
        else:ax.set(xlim=(xy_min[0]-.015,xy_max[0]+.015),ylim=(xy_min[1]-.015,xy_max[1]+.015),xlabel='Length (m)',ylabel='Width (m)')
        patches=[]
        for shape in data['shapes']:
            if not shape['enabled']:continue
            points=capsule_proxy(shape['capsule'])[0] if 'capsule' in shape else np.asarray(shape['points'])
            patch=Polygon(np.zeros((3,2)),facecolor=colors.get(shape['body'],'#929ca8'),edgecolor='#303945',lw=.4)
            ax.add_patch(patch);patches.append((patch,points,data['names'].index(shape['body'])))
        textile=ax.scatter([],[],s=.65,color='#2379d1',alpha=.38,linewidths=0,zorder=4)
        panels.append((projection,patches,textile))
    trace.set(xlim=(0,frames/20),ylim=(0,max(.14,float(speed.max())*1.05)),xlabel='Elapsed time (s)',ylabel='Max node speed (m/s)')
    trace.grid(alpha=.18);trace.axhline(.03,color='#b64d35',ls='--',lw=1,label='Required limit')
    trace.axvspan(frames/20-2,frames/20,color='#2b976e',alpha=.12,label='Final two seconds')
    native,=trace.plot([],[],color='#2459a9',lw=1,label='Native velocity')
    difference,=trace.plot([],[],color='#b67b22',lw=1,label='Position differences')
    trace.legend(loc='upper right',fontsize=8,ncol=4)
    status=fig.text(.5,.84,'',ha='center',fontsize=12)
    writer=FFMpegWriter(fps=20,codec='libx264',extra_args=['-threads','2','-crf','20','-pix_fmt','yuv420p','-movflags','+faststart'])
    assert not output.exists();output.parent.mkdir(parents=True,exist_ok=True)
    with writer.saving(fig,str(output),dpi=100):
        for i in range(frames):
            for projection,patches,textile in panels:
                for patch,points,body in patches:
                    pose=data['poses'][i,body]
                    world=Rotation.from_quat(pose[3:]).apply(points)+pose[:3]
                    projected=world[:,projection];patch.set_xy(projected[ConvexHull(projected).vertices])
                textile.set_offsets(data['nodes'][i][:,projection])
            native.set_data(times[:i+1],speed[:i+1,0])
            difference.set_data(times[:i+1],speed[:i+1,1])
            row=rows[(i+1)*stride-1]
            status.set_text(f"{times[i]:.2f} / {frames/20:.2f} s   |   Head {row['head_inclination_deg']:.3f}°   |   Continuous hold {row['passive_hold_seconds']:.2f} / 2.00 s")
            status.set_color('#217b58' if row['passive_hold_seconds']>=2 else '#263747')
            if i==frames-1:fig.savefig(output.with_suffix('.preview.png'))
            writer.grab_frame()
    plt.close(fig)
    decoded=subprocess.run(['ffmpeg','-v','error','-i',str(output),'-f','null','-','-progress','pipe:1','-nostats'],capture_output=True,text=True)
    counts=[int(line.split('=',1)[1]) for line in decoded.stdout.splitlines() if line.startswith('frame=')]
    assert decoded.returncode==0 and counts[-1]==frames
    inputs=dict(data['inputs'])
    for name in ('result.json','steps.jsonl','passive-analysis.json','passive-geometry-stride-1.json','passive-rigid-audit.json'):
        inputs[str(run/name)]=hashlib.sha256((run/name).read_bytes()).hexdigest()
    inputs[str(capture/'native-property-comparison.json')]=hashlib.sha256((capture/'native-property-comparison.json').read_bytes()).hexdigest()
    output.with_suffix('.json').write_text(json.dumps(dict(diagnostic_only=True,benchmark_eligible=False,
        source=run.name,all_recorded_control_frames=True,decoded_frames=frames,duration_s=frames/20,playback_speed=1,
        speed_trace_note='Per-control-interval maxima over all native physics steps; complete unclipped range.',
        source_sha256=inputs,renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        video_sha256=hashlib.sha256(output.read_bytes()).hexdigest()),indent=2))
    print(output,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('capture',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();assert args.output.resolve().is_relative_to(OUTPUT_ROOT)
    render(args.run,args.capture,args.output)
