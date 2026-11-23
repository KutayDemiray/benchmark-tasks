"""Run the unchanged material checks on disjoint completed recording chunks."""

from task_paths import OUTPUT_ROOT

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np

from audit_full_cover import audit


def merge_parts(parts, expected_ranges, total_steps, control_steps, hz):
    """Require exact coverage and consistent source hashes before merging bounds."""
    assert len(parts)==len(expected_ranges) and expected_ranges[0][0]==0
    hashes={}
    expected_start=0
    for part,(start,stop) in zip(parts,expected_ranges):
        assert start==expected_start and list(part['chunk_range'])==[start,stop]
        assert part['checked_chunks']==stop-start and part['physics_step_offset']==start*hz
        expected_start=stop
        expected_steps=min(stop*hz,total_steps)-start*hz
        assert part['physics_steps']==expected_steps
        for name,value in part['source_sha256'].items():
            if name in hashes:assert hashes[name]==value, f'Inputs changed between workers: {name}'
            hashes[name]=value
    expected_files={f'physics-chunks/{i:05d}.npz' for i in range(expected_start)}
    assert {name for name in hashes if name.startswith('physics-chunks/')}==expected_files
    assert sum(part['physics_steps'] for part in parts)==total_steps==control_steps*(hz//20)
    assert parts[-1]['physics_steps']>=2*hz, 'Final worker must contain the entire final two seconds'
    report=parts[-1].copy()
    completed=all(part.get('recording_status','completed')=='completed' for part in parts)
    report.update(complete=completed,diagnostic_only=not completed or any(part.get('diagnostic_only',False) for part in parts),
        complete_recorded_step_coverage=True,physics_steps=total_steps,checked_chunks=expected_start,
        chunk_range=None,physics_step_offset=0,
        finite=all(part['finite'] for part in parts),
        first_difference_speed_m_s=parts[0].get('first_difference_speed_m_s'),
        max_abs_edge_strain=max(part['max_abs_edge_strain'] for part in parts),
        self_crossing_penetration_upper_bound_m=max(part['self_crossing_penetration_upper_bound_m'] for part in parts),
        max_seam_gap_m=max(part['max_seam_gap_m'] for part in parts),
        no_attachment=all(part['no_attachment'] for part in parts),
        violations=[entry for part in parts for entry in part['violations']],source_sha256=hashes,
        partition_ranges=expected_ranges,
        parallel_scope='Identical per-substep checks on disjoint consecutive chunks; boundary differences include the preceding native state.')
    report['recorded_numerical_checks_passed']=(report['finite'] and report['max_abs_edge_strain']<=.2
        and report['max_seam_gap_m']<=.001 and not report['violations'] and report['no_attachment'])
    report['numerical_checks_passed']=completed and report['recorded_numerical_checks_passed']
    return report


def parallel_audit(run,workers,diagnostic_error_run=False,output_prefix=''):
    """Audit a finished run using bounded independent CPU worker processes."""
    result=json.loads((run/'result.json').read_text())
    request=json.loads((run/'request.json').read_text())
    assert request.get('robot_cover')
    assert result['status']=='completed' or (diagnostic_error_run and result['status']=='error')
    paths=sorted((run/'physics-chunks').glob('[0-9][0-9][0-9][0-9][0-9].npz'))
    assert [int(path.stem) for path in paths]==list(range(len(paths)))
    workers=min(workers,max(1,len(paths)//3))
    assert 1<=workers<=32
    ranges=[]
    for indices in np.array_split(np.arange(len(paths)),workers):
        ranges.append((int(indices[0]),int(indices[-1])+1))
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn')) as pool:
        futures=[pool.submit(audit,run,False,interval,diagnostic_error_run) for interval in ranges]
        parts=[future.result() for future in futures]
    report=merge_parts(parts,ranges,result['physics_steps'],result['control_steps'],request['physics_hz'])
    report['partition_report_sha256']={f'full-cloth-audit-part-{start:05d}-{stop:05d}.json':
        hashlib.sha256((run/f'full-cloth-audit-part-{start:05d}-{stop:05d}.json').read_bytes()).hexdigest()
        for start,stop in ranges}
    report['audit_source_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('audit_cover_parallel.py','audit_full_cover.py','robot_contacts.py','self_contact.py','triangle_broadphase.py')}
    (run/(output_prefix+'full-cloth-audit.json')).write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--output-prefix',choices=('', 'parallel32-check-'),default='')
    parser.add_argument('--diagnostic-error-run',action='store_true',help='Inspect a stopped recording without qualifying it for a score')
    args=parser.parse_args()
    assert args.run.resolve().parent==OUTPUT_ROOT
    result=parallel_audit(args.run,args.workers,args.diagnostic_error_run,args.output_prefix)
    result['violations']=len(result['violations']);result.pop('source_sha256')
    print(json.dumps(result,indent=2))
