"""Verify fresh-process feedback-controller reproduction of a tabletop result."""

import argparse
import json
from pathlib import Path

import numpy as np

from task_paths import OUTPUT_ROOT as OUTPUT


def compare(first,second):
    """Compare frozen sources, initial states and independently audited outcomes."""
    runs=(first,second);manifests=[];setups=[];outcomes=[];initial=[];actions=[]
    for run in runs:
        audit=json.loads((run/'verification.json').read_text())
        assert audit['verification_pass'] and audit['episodes'][0]['full_task_success']
        manifests.append(json.loads((run/'manifest.json').read_text()))
        episode=run/'episode-00'
        setups.append(json.loads((episode/'setup.json').read_text()))
        outcomes.append(json.loads((episode/'result.json').read_text()))
        initial.append(np.load(episode/'initial-measured-state.npz'))
        actions.append(np.load(episode/'control-steps.npz')['actions'])
    assert first.resolve()!=second.resolve() and manifests[0]['pid']!=manifests[1]['pid'],'same_process'
    assert manifests[0]['sources']==manifests[1]['sources'],'different_source_bundle'
    assert manifests[0]['versions']==manifests[1]['versions'],'different_runtime_versions'
    assert setups[0]['parameters']==setups[1]['parameters'],'different_parameters'
    assert setups[0]['seed']==setups[1]['seed'],'different_seed'
    differences={k:float(np.max(np.abs(initial[0][k]-initial[1][k]))) for k in initial[0].files}
    assert max(differences.values())<1e-7,'different_initial_state'
    assert all(r['stage']=='tabletop_task' and r['task_score']==1. and r['retained_pairs']==[0,1,2]
        and not r['invalid_reasons'] and not r['original_hanger_task_success'] for r in outcomes)
    return dict(passed=True,kind='fresh_process_feedback_controller_reproduction',runs=list(map(str,runs)),
        source_bundle_sha256=manifests[0]['bundle_sha256'],seed=setups[0]['seed'],versions=manifests[0]['versions'],
        process_ids=[m['pid'] for m in manifests],initial_maximum_absolute_differences=differences,
        control_steps=[r['control_steps'] for r in outcomes],
        action_sequences_identical=actions[0].shape==actions[1].shape and np.array_equal(*actions),
        outcomes=outcomes,
        scope='Feedback runs from the same reset; this is not a statistical success-rate claim or an exact-action trajectory replay.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first',type=Path);parser.add_argument('second',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert all(p.resolve().is_relative_to(OUTPUT) for p in (args.first,args.second,args.out))
    result=compare(args.first,args.second)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'],output=str(args.out))))
