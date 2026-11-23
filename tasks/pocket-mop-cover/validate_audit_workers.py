"""Check spawned audit workers against an already checked native interval."""

from task_paths import OUTPUT_ROOT

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np

from audit_full_cover import audit


def validate(run,start,split,stop):
    """Verify partition bounds, global contact indices and boundary velocities."""
    assert 0<=start<split<stop and stop-split>=2
    reference_path=run/f'full-cloth-audit-part-{start:05d}-{stop:05d}.json'
    reference=json.loads(reference_path.read_text())
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs=[pool.submit(audit,run,False,interval) for interval in ((start,split),(split,stop))]
        first,last=[job.result() for job in jobs]
    assert first['physics_steps']+last['physics_steps']==reference['physics_steps']
    assert first['violations']+last['violations']==reference['violations']
    for field in ('max_abs_edge_strain','self_crossing_penetration_upper_bound_m','max_seam_gap_m'):
        assert max(first[field],last[field])==reference[field],field
    for field in ('native_final_two_second_speed_m_s','difference_final_two_second_speed_m_s'):
        assert last[field]==reference[field],field
    assert first['finite'] and last['finite'] and reference['finite']
    hz=json.loads((run/'request.json').read_text())['physics_hz']
    with np.load(run/'physics-chunks'/f'{split-1:05d}.npz') as chunk:previous=chunk['positions'][-1]
    with np.load(run/'physics-chunks'/f'{split:05d}.npz') as chunk:current=chunk['positions'][0]
    boundary=float(np.linalg.norm(current-previous,axis=1).max()*hz)
    assert boundary==last['first_difference_speed_m_s']
    paths=[reference_path,run/f'full-cloth-audit-part-{start:05d}-{split:05d}.json',
           run/f'full-cloth-audit-part-{split:05d}-{stop:05d}.json']
    report=dict(passed=True,interval_chunks=[start,stop],split_chunk=split,
        physics_steps=reference['physics_steps'],partition_boundary_difference_speed_m_s=boundary,
        exact_global_contact_indices=True,exact_bounds_and_final_speeds=True,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        scope='Spawned CPU worker equivalence on a recorded interval; no full-run or benchmark certificate.')
    (run/'spawned-audit-validation.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--start',type=int,required=True)
    parser.add_argument('--split',type=int,required=True)
    parser.add_argument('--stop',type=int,required=True)
    args=parser.parse_args()
    assert args.run.resolve().parent==OUTPUT_ROOT
    print(json.dumps(validate(args.run,args.start,args.split,args.stop),indent=2))
