"""Search complete native recordings for continuous holds; never assign a task score."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from passive_contact import native_physics_errors
from passive_settling import PassiveHold


def analyze(run):
    request=json.loads((run/'request.json').read_text())
    result=json.loads((run/'result.json').read_text())
    assert request['experiment']=='passive_contact' and result['status']=='completed'
    assert not native_physics_errors((run/'kit.log').read_text())
    assert not request.get('passive_free_space')
    hz=request['physics_hz']; monitor=PassiveHold(hz)
    proof=json.loads((run/'passive-initialization.json').read_text())
    scene_path=run.parent/proof['source_full_run']/'scene-audit.json'
    scene=json.loads(scene_path.read_text())
    table_top=scene['table_center_m'][2]+scene['table_size_m'][2]/2
    with np.load(run/'geometry.npz') as geometry:
        previous=geometry['initial'].astype(float)
    peaks=[]; hashes={}; longest=0
    for path in sorted((run/'physics-chunks').glob('*.npz')):
        hashes[str(path.relative_to(run))]=hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as chunk:
            positions=chunk['positions'];velocities=chunk['velocities']
            normals=Rotation.from_quat(chunk['frame_body_poses'][:,0,3:]).apply([0.,0.,1.])
            inclination=np.rad2deg(np.arccos(np.clip(normals[:,2],-1,1)))
            wings=np.max(abs(np.rad2deg(chunk['frame_joint_positions'][:,:2])),axis=1)
            height=chunk['frame_body_poses'][:,0,2]-table_top
            for i in range(len(positions)):
                points=positions[i].astype(float)
                native=float(np.linalg.norm(velocities[i],axis=1).max())
                difference=float(np.linalg.norm(points-previous,axis=1).max()*hz)
                previous=points
                monitor.observe(native,difference,float(inclination[i]),float(wings[i]),float(height[i]))
                longest=max(longest,monitor.contiguous_steps)
                peaks.append((native,difference))
    assert monitor.steps==result['physics_steps']
    first=monitor.first_stop_step
    first_window=None
    if first is not None:
        window=np.asarray(peaks[first-2*hz:first])
        first_window=dict(start_s=(first-2*hz)/hz,end_s=first/hz,
            control_step=first//(hz//20),native_speed_m_s=float(window[:,0].max()),
            difference_speed_m_s=float(window[:,1].max()))
    for name in ('request.json','result.json','geometry.npz','passive-initialization.json'):
        hashes[name]=hashlib.sha256((run/name).read_bytes()).hexdigest()
    report=dict(diagnostic_only=True,benchmark_eligible=False,run=run.name,
        all_native_steps=monitor.steps,physics_hz=hz,
        maximum_continuous_hold_s=longest/hz,first_complete_control_boundary_window=first_window,
        criteria=dict(native_speed_m_s=.03,difference_speed_m_s=.03,head_inclination_deg=3.,
            wing_angle_deg=3.,head_height_m=[-.001,.01],minimum_duration_s=2.,control_hz=20),
        limitations=['Offline observation is not a fresh online stopping test.',
            'No cloth/rigid penetration, seam, strain or full-task qualification is implied.'],
        source_sha256=hashes,table_source_sha256=hashlib.sha256(scene_path.read_bytes()).hexdigest(),
        assessor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        hold_observer_sha256=hashlib.sha256(Path(__file__).with_name('passive_settling.py').read_bytes()).hexdigest())
    (run/'passive-hold-search.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    args=parser.parse_args()
    assert args.run.resolve().is_relative_to(OUTPUT_ROOT)
    analyze(args.run)
