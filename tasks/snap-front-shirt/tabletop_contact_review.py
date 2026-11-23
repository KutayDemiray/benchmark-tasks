"""Compare passive tabletop contact records without changing any simulation."""

from task_paths import OUTPUT_ROOT

import argparse
import json
from pathlib import Path

import numpy as np

OUTPUT=OUTPUT_ROOT


def review(run):
    """Report penetration witnesses and resting gaps of the initially flat back."""
    records=[]
    for episode in sorted(run.glob('episode-*')):
        cfg=json.loads((episode/'setup.json').read_text())['parameters']
        meta=json.loads((episode/'topology.json').read_text());c=meta['cloth']
        outcome=json.loads((episode/'result.json').read_text())
        initial=np.asarray(c['initial']);panel=np.asarray(c['panel_ids'])
        h=cfg['tabletop']['table_top_m']
        mask=(panel==0)&(initial[:,2]<h+cfg['tabletop']['back_initial_gap_m']+1e-7)
        nodes=np.load(episode/'control-steps.npz')['cloth_nodes'].astype(float)
        tail=nodes[-20:]
        gaps=tail[:,mask,2]-h
        item=dict(episode=episode.name,case=outcome['case'],seed=outcome['seed'],
            physics_hz=cfg['simulation']['physics_hz'],
            contact_generation_offset_m=cfg['fabric']['contact_offset_m'],
            physical_rest_offset_m=cfg['fabric']['rest_offset_m'],
            control_steps=outcome['control_steps'],diagnostic_pass=outcome.get('diagnostic_pass'),
            invalid_reasons=outcome['invalid_reasons'],
            max_table_penetration_m=outcome.get('max_table_penetration_m'),
            flat_back_nodes=int(mask.sum()),
            terminal_flat_back_gap_quantiles_m=dict(zip(('minimum','p01','median','p99','maximum'),
                np.quantile(gaps,[0,.01,.5,.99,1]).tolist())),
            gap_scope='Last up to two seconds, initially flat back-panel nodes only. Short failed episodes are not settled comparisons.')
        path=episode/'worst-table-contact.npz'
        if path.exists():
            w=np.load(path);points=w['nodes'];i=int(np.argmin(points[:,2]))
            item['worst_contact']=dict(physics_step=int(w['physics_step']),
                time_s=int(w['physics_step'])/cfg['simulation']['physics_hz'],
                node=i,panel=int(panel[i]),initial_position_m=initial[i].tolist(),
                position_m=points[i].tolist(),penetration_m=float(w['penetration_m']))
        records.append(item)
    return dict(run=str(run),episodes=records)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs',nargs='+',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert args.out.resolve().is_relative_to(OUTPUT)
    assert all(p.resolve().is_relative_to(OUTPUT) for p in args.runs)
    result=dict(runs=[review(p) for p in args.runs],
        note='Diagnostic comparison; does not award task credit, establish statistical reliability, or change a frozen outcome.')
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(runs=len(result['runs']),output=str(args.out))))
