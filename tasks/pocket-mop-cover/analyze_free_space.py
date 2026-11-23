"""Measure force transfer in initialized zero-gravity cloth/frame controls."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from audit_mechanism import capsule_proxy
from diagnose_release_energy import SurfaceEnergy
from passive_contact import native_physics_errors


def analyze(run):
    """Record native frame momentum and analytical cloth momentum without scoring."""
    request = json.loads((run/'request.json').read_text())
    result = json.loads((run/'result.json').read_text())
    proof = json.loads((run/'passive-initialization.json').read_text())
    assert request['experiment'] == 'passive_contact' and request['passive_free_space']
    assert result['status'] == 'completed' and proof['free_space_counterfactual']
    assert not proof['robots_present'] and not proof['controllers_present']
    assert proof['all_initial_velocities_zero'] and proof['state_writes_only_before_recording']
    assert not native_physics_errors((run/'kit.log').read_text())
    assert json.loads((run/'process.json').read_text())['returncode'] == 0
    dynamics = json.loads((run.parent/request['passive_source']/'frame-dynamics.json').read_text())
    masses = np.asarray(dynamics['masses_kg'][0])
    with np.load(run/'geometry.npz') as geometry:
        material = json.loads((run/'effective-parameters.json').read_text())
        cloth = SurfaceEnergy(geometry['rest'], geometry['faces'], material)
    names = json.loads((run/'frame-names.json').read_text())['body_names']
    assert names == dynamics['body_names']
    shapes = []
    for shape in json.loads((run/'frame-geometry.json').read_text())['colliders']:
        if not shape['enabled']:
            continue
        vertices = capsule_proxy(shape['capsule'])[0] if 'capsule' in shape else np.asarray(shape['points'])
        shapes.append((names.index(shape['body']), vertices))
    frame_momentum = []
    cloth_momentum = []
    min_frame_z = min_cloth_z = float('inf')
    hashes = {}
    for path in sorted((run/'physics-chunks').glob('*.npz')):
        hashes[str(path.relative_to(run))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as chunk:
            velocities = chunk['frame_body_velocities']
            frame_momentum.extend(np.sum(velocities[:, :, :3]*masses[None, :, None], axis=1))
            cloth_momentum.extend(np.sum(chunk['velocities']*cloth.node_mass[None, :, None], axis=1))
            min_cloth_z = min(min_cloth_z, float(chunk['positions'][:, :, 2].min()))
            for poses in chunk['frame_body_poses']:
                for index, vertices in shapes:
                    points = Rotation.from_quat(poses[index, 3:]).apply(vertices)+poses[index, :3]
                    min_frame_z = min(min_frame_z, float(points[:, 2].min()))
    frame_momentum = np.asarray(frame_momentum)
    cloth_momentum = np.asarray(cloth_momentum)
    assert len(frame_momentum) == result['physics_steps']
    peak = int(np.argmax(np.linalg.norm(frame_momentum, axis=1)))
    early_peak = int(np.argmax(np.linalg.norm(frame_momentum[:round(.1*request['physics_hz'])], axis=1)))
    report = dict(diagnostic_only=True, benchmark_eligible=False, run=run.name,
        physics_steps=len(frame_momentum), frame_mass_kg=float(masses.sum()),
        analytical_cloth_mass_kg=float(cloth.node_mass.sum()),
        maximum_frame_linear_momentum_kg_m_s=float(np.linalg.norm(frame_momentum, axis=1).max()),
        maximum_cloth_linear_momentum_proxy_kg_m_s=float(np.linalg.norm(cloth_momentum, axis=1).max()),
        maximum_combined_linear_momentum_proxy_kg_m_s=float(np.linalg.norm(frame_momentum+cloth_momentum, axis=1).max()),
        peak_frame_momentum_time_s=(peak+1)/request['physics_hz'],
        frame_momentum_at_peak=frame_momentum[peak].tolist(),
        first_100ms_peak=dict(time_s=(early_peak+1)/request['physics_hz'],
            frame_momentum_kg_m_s=frame_momentum[early_peak].tolist(),
            cloth_momentum_proxy_kg_m_s=cloth_momentum[early_peak].tolist()),
        cloth_momentum_proxy_at_peak=cloth_momentum[peak].tolist(),
        minimum_frame_collider_z_m=min_frame_z, minimum_cloth_z_m=min_cloth_z,
        source_sha256=hashes,
        limitations=['Cloth mass distribution is an analytical rest-area estimate, not native inverse-mass readback.',
            'Momentum is diagnostic evidence, not a full collision/contact or mechanism qualification.'],
        assessor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    for name in ('request.json','result.json','passive-initialization.json','geometry.npz','frame-names.json','frame-geometry.json'):
        hashes[name] = hashlib.sha256((run/name).read_bytes()).hexdigest()
    (run/'free-space-momentum.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key:value for key,value in report.items() if key!='source_sha256'}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path, nargs='+')
    args = parser.parse_args()
    for run in args.runs:
        assert run.resolve().is_relative_to(OUTPUT_ROOT)
        analyze(run)
