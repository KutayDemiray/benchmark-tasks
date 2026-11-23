"""Measure passive diagnostic settling and analytical energy without scoring it."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from diagnose_release_energy import SurfaceEnergy, rigid_energy
from passive_contact import native_physics_errors


def analyze(run):
    def read(name):
        return json.loads((run/name).read_text())
    request, result = read('request.json'), read('result.json')
    assert request['experiment'] == 'passive_contact' and result['status'] == 'completed'
    assert not (run/'native-constraint-rejection.json').exists(), 'Rejected native constraint'
    assert not native_physics_errors((run/'kit.log').read_text()), 'Rejected native physics'
    proof = read('passive-initialization.json')
    paths = sorted((run/'physics-chunks').glob('*.npz'))
    hz = request['physics_hz']; total = result['physics_steps']
    assert len(paths) == int(np.ceil(total/hz))
    first_index = max(0, total-2*hz)
    speed = difference = 0.
    minimum_z = float('inf')
    hashes = {}
    previous = None
    minimum_position = maximum_position = first_position = final_position = None
    geometry = np.load(run/'geometry.npz')
    has_cloth = request['passive_mode'] != 'bare'
    if has_cloth:
        for path in paths[max(0, first_index//hz-1):]:
            hashes[str(path.relative_to(run))] = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path) as chunk:
                x = chunk['positions']; v = chunk['velocities']
                start = int(path.stem)*hz
                if previous is None:
                    previous = geometry['initial'] if start == 0 else x[0]
                for offset in range(len(x)):
                    if start+offset >= first_index:
                        speed = max(speed, float(np.linalg.norm(v[offset], axis=1).max()))
                        difference = max(difference, float(np.linalg.norm(x[offset].astype(float)-previous, axis=1).max()*hz))
                        minimum_z = min(minimum_z, float(x[offset, :, 2].min()))
                        points=x[offset].astype(float)
                        if minimum_position is None:
                            minimum_position=points.copy();maximum_position=points.copy();first_position=points.copy()
                        minimum_position=np.minimum(minimum_position,points)
                        maximum_position=np.maximum(maximum_position,points)
                        final_position=points
                    previous = x[offset].astype(float)
    energy = None
    if has_cloth and request['passive_mode'] not in ('compound', 'compound_articulation') and not request.get('passive_free_space'):
        source = run.parent/proof['source_run']
        original = run.parent/proof['source_full_run']
        dynamics = json.loads((source/'frame-dynamics.json').read_text())
        source_hz = json.loads((original/'request.json').read_text())['physics_hz']
        index = (proof['source_control_step']+1)*(source_hz//20)-1
        with np.load(original/'physics-chunks'/f'{index//source_hz:05d}.npz') as initial_chunk:
            poses = initial_chunk['frame_body_poses'][index % source_hz].copy()
            poses[:, 2] += request.get('passive_release_height') or 0.
            joints = initial_chunk['frame_joint_positions'][index % source_hz]
        if (run/'initial-frame-state.npz').exists():
            with np.load(run/'initial-frame-state.npz') as state:
                poses, joints = state['poses'].copy(), state['joints'].copy()
            hashes['initial-frame-state.npz'] = hashlib.sha256((run/'initial-frame-state.npz').read_bytes()).hexdigest()
        surface = SurfaceEnergy(geometry['rest'], geometry['faces'], read('effective-parameters.json'))
        initial = dict(time_s=0., **surface.measure(geometry['initial'], np.zeros_like(geometry['initial'])),
            **rigid_energy(poses, np.zeros((len(poses), 6)), joints, dynamics))
        rows = [initial]
        path = paths[0]
        hashes[str(path.relative_to(run))] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as chunk:
            for offset in range(0, min(len(chunk['positions']), hz), max(1, hz//80)):
                rows.append(dict(time_s=(offset+1)/hz,
                    **surface.measure(chunk['positions'][offset], chunk['velocities'][offset]),
                    **rigid_energy(chunk['frame_body_poses'][offset], chunk['frame_body_velocities'][offset],
                        chunk['frame_joint_positions'][offset], dynamics)))
        for row in rows:
            row['total_proxy_j'] = sum(value for key, value in row.items() if key.endswith('_j'))
        energy = dict(initial=initial, sampled_first_second=rows,
            maximum_total_proxy_increase_j=max(row['total_proxy_j'] for row in rows)-initial['total_proxy_j'],
            analytical_cloth_mass_kg=float(surface.node_mass.sum()),
            limitation='Membrane/bending terms are analytical proxies, not exact native energy or direct contact work. Cloth mass is derived from rest area, thickness and density, not native mass readback.')
    for name in ('request.json', 'result.json', 'passive-initialization.json', 'geometry.npz', 'effective-parameters.json'):
        hashes[name] = hashlib.sha256((run/name).read_bytes()).hexdigest()
    report = dict(diagnostic_only=True, benchmark_eligible=False, run=run.name,
        mode=request['passive_mode'], solver=request.get('passive_solver', 'TGS'), physics_hz=hz,
        final_inclination_deg=result['final']['head_inclination_deg'],
        final_bridge_height_m=result['final']['bridge_height_m'],
        final_two_seconds=None if not has_cloth else dict(native_speed_m_s=speed, difference_speed_m_s=difference,
            min_cloth_z_m=minimum_z, measured_intervals=total-first_index,
            complete_two_second_window=total-first_index == 2*hz,
            speed_threshold_passed=total-first_index == 2*hz and speed <= .03 and difference <= .03,
            max_node_displacement_bound_m=float(np.linalg.norm(maximum_position-minimum_position,axis=1).max()),
            max_node_net_displacement_m=float(np.linalg.norm(final_position-first_position,axis=1).max()),
            displacement_note='Bounding-box diagonal over each node trajectory; diagnostic only, no change to velocity scoring.'),
        energy=energy, source_sha256=hashes,
        limitations=['No complete penetration/strain audit is provided by this diagnostic.',
            'Cold initialized contact; no full-task or qualification score.'],
        assessor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (run/'passive-analysis.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key not in ('energy', 'source_sha256')}, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', nargs='+', type=Path)
    args = parser.parse_args()
    for run in args.runs:
        assert run.resolve().is_relative_to(OUTPUT_ROOT)
        analyze(run)
