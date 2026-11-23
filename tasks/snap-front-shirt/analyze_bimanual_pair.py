"""Compare supported and unsupported native trials with identical pressing actions."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from task_paths import OUTPUT_ROOT


def benefit(supported_drift,unsupported_drift,supported_closed,unsupported_closed,both_valid):
    """Apply the predeclared usefulness gate without modifying task scoring."""
    absolute=unsupported_drift-supported_drift
    relative=absolute/unsupported_drift if unsupported_drift>0 else 0.
    drift=bool(absolute>=.0005 and relative>=.5)
    closure=bool(supported_closed and not unsupported_closed)
    return dict(absolute_drift_reduction_m=absolute,relative_drift_reduction=relative,
        drift_benefit=drift,closure_benefit=closure,
        qualifies_for_replication=bool(both_valid and supported_closed and (drift or closure)))


def load(worker):
    episode=worker/'episode-00'
    return dict(path=str(worker),setup=json.loads((episode/'setup.json').read_text()),
        result=json.loads((episode/'result.json').read_text()),
        rows=[json.loads(line) for line in (episode/'trace.jsonl').open()],
        actions=np.load(episode/'control-steps.npz')['actions'],
        initial=np.load(episode/'initial-measured-state.npz'),
        manifest=json.loads((worker/'manifest.json').read_text()),
        exit=json.loads((worker/'exit.json').read_text()),
        verification=json.loads((worker/'verification.json').read_text()))


def compare(supported,unsupported):
    """Require matched seeds, physics, actions and intact released closures."""
    a,b=load(supported),load(unsupported)
    assert a['verification']['verification_pass'] and b['verification']['verification_pass'], 'Independent audit is required'
    assert a['exit']['returncode']==b['exit']['returncode']==0, 'Worker runtime failure'
    assert a['exit']['worker_pid']!=b['exit']['worker_pid'], 'Use distinct native processes'
    assert a['setup']['seed']==b['setup']['seed'], 'Unmatched seed'
    assert a['setup']['parameters']==b['setup']['parameters'], 'Different physics or scoring parameters'
    assert a['setup']['controller'].startswith('bimanual_') and b['setup']['controller']=='none'
    assert a['setup']['control_hz']==b['setup']['control_hz'], 'Different action clocks'
    assert a['manifest']['bundle_sha256']==b['manifest']['bundle_sha256'], 'Different source bundles'
    assert a['manifest']['versions']==b['manifest']['versions'], 'Different runtime versions'
    assert set(a['initial'].files)==set(b['initial'].files), 'Different initial-state records'
    differences={key:float(np.abs(a['initial'][key]-b['initial'][key]).max()) for key in a['initial'].files}
    assert all(np.isfinite(v) and v<1e-7 for v in differences.values()), 'Initial native states differ'
    full_length=len(a['actions'])==len(b['actions'])
    count=min(len(a['actions']),len(b['actions']))
    assert count>0 and np.array_equal(a['actions'][:count,7:],b['actions'][:count,7:]), 'Pressing-arm actions differ'
    assert np.array_equal(b['actions'][:,:7],np.repeat(b['actions'][:1,:7],len(b['actions']),axis=0)), 'Ablated support arm moved'
    indices=[i for i,row in enumerate(a['rows']) if row['control']['phase'] in ('align','press')]
    assert indices, 'No alignment/pressing interval was reached'
    start,end=min(indices),max(indices)
    def drift(rows):
        if len(rows)<=end:return None
        xy=np.array([r['physics']['hardware']['socket0'][:2] for r in rows[start:end+1]])
        return float(np.linalg.norm(xy-xy[0],axis=1).max())
    drifts=[drift(run['rows']) for run in (a,b)]
    valid=[not run['result']['invalid_reasons'] and run['result'].get('pilot_rigid_contact_guard_passed') is True
           for run in (a,b)]
    closed=[0 in run['result']['retained_pairs'] for run in (a,b)]
    both_valid=all(valid) and full_length and all(d is not None for d in drifts)
    decision=benefit(*[0. if d is None else d for d in drifts],*closed,both_valid)
    return dict(scope='One matched causal pilot comparison; a second qualifying comparison is required',
        supported=str(supported),unsupported=str(unsupported),seed=a['setup']['seed'],
        revision=a['setup']['controller'],bundle_sha256=a['manifest']['bundle_sha256'],
        actions_identical_for_pressing_arm=True,full_matching_duration=full_length,
        drift_window_control_steps=[start,end],drift_reference='socket0 XY at first supported alignment step',
        supported_drift_m=drifts[0],unsupported_drift_m=drifts[1],supported_valid=valid[0],unsupported_valid=valid[1],
        supported_retained=closed[0],unsupported_retained=closed[1],
        supported_task_score=a['result']['task_score'],unsupported_task_score=b['result']['task_score'],
        worker_pids=[a['exit']['worker_pid'],b['exit']['worker_pid']],
        initial_maximum_absolute_differences=differences,**decision)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('supported',type=Path);parser.add_argument('unsupported',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    for path in (args.supported,args.unsupported,args.out):assert path.resolve().is_relative_to(OUTPUT_ROOT)
    result=compare(args.supported,args.unsupported)
    result['analyzer_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('qualifies_for_replication','drift_benefit','closure_benefit')}))


if __name__=='__main__':main()
