"""Adversarial offline checks using copies of a measured successful record."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from robot_geometry import rotation
from tabletop_scoring import task_score
from episode_timing import physics_stride
from task_paths import OUTPUT_ROOT

OUTPUT=OUTPUT_ROOT
HERE=Path(__file__).resolve().parent


def check(run):
    """Replay the current scorer and reject deliberate state/record shortcuts."""
    episode=run/'episode-00'
    original_rows=[json.loads(s) for s in (episode/'trace.jsonl').read_text().splitlines()]
    original_physics=[json.loads(s) for s in (episode/'physics.jsonl').read_text().splitlines()]
    setup=json.loads((episode/'setup.json').read_text());cfg=setup['parameters']
    original_meta=json.loads((episode/'topology.json').read_text())
    archived=json.loads((episode/'result.json').read_text())
    stride=physics_stride(cfg)
    results=[]
    cases=('unaltered','robot_support','wrong_pair','proximity_only','cap_through_rim',
           'damage_then_rest','short_retention','initially_closed','nonfinite_contact',
           'hidden_attachment','one_correct_pair','two_correct_pairs','controller_labels_irrelevant')
    for case in cases:
        physics=list(original_physics);meta=dict(original_meta)
        rows=[dict(r) for r in original_rows]
        expected=0.;invalid=None
        def replace_pose(step,male_index,socket_index,axial):
            if physics[step] is original_physics[step]:physics[step]=copy.deepcopy(physics[step])
            socket=np.array(physics[step]['hardware'][f'socket{socket_index}'])
            male=socket.copy();male[:3]+=rotation(socket[3:])@np.array([0.,0.,axial])
            physics[step]['hardware'][f'male{male_index}']=male.tolist()
        if case=='unaltered':expected=1.
        elif case=='robot_support':rows[-1]['robot_clearance_m']=dict(fabric_m=0.,hardware_m=.1)
        elif case=='wrong_pair':replace_pose(-1,0,1,.0068)
        elif case=='proximity_only':
            for i in range(3):replace_pose(-1,i,i,.009)
        elif case=='cap_through_rim':
            replace_pose(-1,0,0,.003);invalid='cap_through_socket_rim'
        elif case=='damage_then_rest':
            physics[3]=dict(physics[3],strain=.16);invalid='cloth_strain_limit'
        elif case=='short_retention':
            step=len(physics)-1-round(2.5*cfg['simulation']['physics_hz'])
            for i in range(3):replace_pose(step,i,i,.009)
        elif case=='initially_closed':
            replace_pose(0,0,0,.0068);invalid='initially_closed_snap'
        elif case=='nonfinite_contact':
            physics[-1]=copy.deepcopy(physics[-1]);physics[-1]['snap_robot_contact_norm_N']['socket6']=float('nan')
            invalid='invalid_contact_measurement'
        elif case=='hidden_attachment':meta['grasp_attachments']=True;invalid='hidden_attachment'
        elif case=='one_correct_pair':
            for i in (1,2):replace_pose(-1,i,i,.009)
            expected=1/3
        elif case=='two_correct_pairs':replace_pose(-1,2,2,.009);expected=2/3
        elif case=='controller_labels_irrelevant':
            for row in rows:row['control']={'phase':'unrelated_label','completed':False}
            expected=1.
        for i,row in enumerate(rows):row['physics']=physics[(i+1)*stride]
        result=task_score(rows,physics,cfg,meta,archived['termination_reason'])
        assert result['task_score']==expected,(case,result['task_score'],expected,result['invalid_reasons'])
        if invalid is not None:assert invalid in result['invalid_reasons'],(case,result['invalid_reasons'])
        if case=='unaltered':
            assert result['task_score']==archived['task_score'] and result['retained_pairs']==archived['retained_pairs']
            assert result['per_control_scores']==[r['task_score'] for r in original_rows]
        results.append(dict(case=case,expected_score=expected,actual_score=result['task_score'],
            invalid_reasons=result['invalid_reasons'],passed=True))
        print(json.dumps(results[-1]),flush=True)
    hashes={name:hashlib.sha256((HERE/name).read_bytes()).hexdigest()
        for name in ('tabletop_scoring.py','robot_scoring.py','robot_geometry.py','episode_timing.py','check_tabletop_shortcuts.py')}
    return dict(passed=True,run=str(run),runtime_bundle_sha256=setup['bundle_sha256'],
        reviewed_scorer_source_hashes=hashes,cases=results,
        note='Only unaltered is a measured simulation record. Other cases deliberately modify copies offline; they are scorer checks, not physically executed robot trials. Frozen native data is unchanged.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert args.run.resolve().is_relative_to(OUTPUT) and args.out.resolve().is_relative_to(OUTPUT)
    result=check(args.run);args.out.write_text(json.dumps(result,indent=2)+'\n')
