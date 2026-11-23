"""Independently inspect a native strain event without starting the simulator."""

import argparse
import json
from pathlib import Path
from task_paths import OUTPUT_ROOT

import numpy as np

from strain_witness import StrainWitness


def verify(episode):
    setup = json.loads((episode/'setup.json').read_text())
    cfg = setup['parameters']
    meta = json.loads((episode/'topology.json').read_text())
    state = dict(np.load(episode/'worst-strain-state.npz',allow_pickle=False))
    history = np.load(episode/'strain-event-history.npz',allow_pickle=False)
    witness = StrainWitness(meta['cloth']['edges'],meta['cloth']['rest_lengths'],meta['cloth']['faces'])
    witness.peak = state
    witness.frames = [dict(physics_step=int(i)) for i in history['physics_step']]
    audit = witness.audit()
    assert audit['cpu_agreement'], 'CPU and native strain disagree'
    peak = -1.
    matched = None
    with (episode/'physics.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            if np.isfinite(row['strain']):
                peak = max(peak,row['strain'])
            if row['physics_step'] == audit['physics_step']:
                matched = row
    assert matched is not None, 'Witness missing from native trace'
    assert abs(peak-audit['cpu_absolute_strain']) < 1e-9, 'Witness is not global maximum'
    assert abs(matched['strain']-audit['cpu_absolute_strain']) < 1e-9
    assert matched['worst_strain_edge_index'] == audit['edge_index']
    edges = np.asarray(meta['cloth']['edges'])
    endpoints = edges[audit['edge_index']]
    positions = history['cloth_positions_m'][:,endpoints].astype(float)
    velocities = history['cloth_velocities_m_s'][:,endpoints].astype(float)
    signed = np.linalg.norm(positions[:,0]-positions[:,1],axis=1)/audit['rest_length_m']-1.
    table_h = cfg['tabletop']['table_top_m']
    audit['history'] = [dict(physics_step=int(step),time_s=float(step/cfg['simulation']['physics_hz']),
        signed_strain=float(signed[i]),edge_positions_m=positions[i].tolist(),
        edge_velocities_m_s=velocities[i].tolist(),minimum_edge_table_gap_m=float(positions[i,:,2].min()-table_h))
        for i,step in enumerate(history['physics_step'])]
    audit['absolute_strain_guard'] = cfg['gate']['max_edge_strain']
    audit['violates_guard'] = audit['cpu_absolute_strain'] > cfg['gate']['max_edge_strain']
    audit['positions_finite'] = bool(np.isfinite(state['cloth_positions_m']).all())
    audit['velocities_finite'] = bool(np.isfinite(state['cloth_velocities_m_s']).all())
    audit['rivet_node'] = bool(any(np.isin(endpoints,attachment['ids']).any()
                                   for attachment in meta['cloth']['attachments']))
    audit['table_penetration_at_event_m'] = matched['table_penetration_m']
    audit['rigid_normal_force_at_event_N'] = max(matched['snap_robot_contact_norm_N'].values())
    audit['validity_rule_changed'] = False
    audit['cause'] = 'Unresolved; compression or extension classification does not establish a solver cause.'
    return audit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    assert args.out.resolve().is_relative_to(OUTPUT_ROOT)
    result = verify(args.episode)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('physics_step','classification','signed_strain','cpu_agreement','violates_guard')}))


if __name__ == '__main__':
    main()
