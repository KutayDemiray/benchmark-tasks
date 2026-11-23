"""Summarize the bounded support decision without promoting diagnostic credit."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
from pathlib import Path


def summarize(output):
    """Combine archived outcomes with independent geometric failure evidence."""
    runs=[]
    for name in ('p10-fixed-loaded-loop','p10-gentle-distributed-shirt','p10-wooden-hanger-shirt'):
        root=output/name
        hashes={}
        def read(relative):
            path=root/relative
            raw=path.read_bytes()
            hashes[relative]=hashlib.sha256(raw).hexdigest()
            return json.loads(raw)
        result=read('episode-00/result.json')
        setup=read('episode-00/setup.json')
        verification=read('verification.json')
        audits=[read('contact-substeps-audit.json')]
        if (root/'contact-controls-audit.json').exists():
            audits.append(read('contact-controls-audit.json'))
        cfg=setup['parameters']
        maximum_depth=max(a['max_vertex_penetration_m'] for a in audits)
        crossed=sum(a['centreline_crossing_frames'] for a in audits)
        deep_box=sum(bool(b.get('deeply_intersecting_triangles'))
                     for a in audits for row in a['frames'] for b in row['bars'])
        hook_depth=max(a.get('max_hook_rail_penetration_lower_bound_m',0.) for a in audits)
        # 1 mm is the existing diagnostic deep-penetration threshold. Do not
        # interpret absence of sampled witnesses as proof of continuous contact.
        penetration_failure=bool(maximum_depth>.001 or crossed or deep_box or hook_depth>.001)
        assert verification['verification_pass'],'Evidence verification failed'
        assert result['full_task_score']==0. and not result['full_task_success']
        runs.append(dict(run=name,seed=result['seed'],source_bundle_sha256=result['bundle_sha256'],
            control_steps=result['control_steps'],physics_steps=result['physics_steps_observed'],
            termination_reason=result['termination_reason'],diagnostic_pass=result['diagnostic_pass'],
            maximum_strain=result['max_cloth_strain'],strain_limit=cfg['gate']['max_edge_strain'],
            maximum_rivet_error_m=result['max_rivet_error_m'],rivet_limit_m=cfg['gate']['max_attachment_error_m'],
            maximum_bar_vertex_penetration_m=maximum_depth,penetration_failure=penetration_failure,
            maximum_hook_rail_penetration_witness_m=hook_depth,
            observed_contact_sampling=[a['sampling'] for a in audits],
            declared_fixed_support=cfg['garment_test'].get('fixed_support',False),
            complete_task_success=False,evidence_verified=True,
            videos=verification['episodes'][0]['videos'],input_sha256=hashes))
    all_failed=all(not r['diagnostic_pass'] and r['penetration_failure'] for r in runs)
    assert all_failed,'New evidence needs a fresh decision, not an automatic shelve verdict'
    return dict(date='2026-11-21',decision='shelve_hanger_candidate',comparisons=runs,
        full_task_complete=False,statistical_success_rate_claimed=False,
        reason='All three targeted comparisons fail support and show cloth/bar penetration. No credible free garment support established.',
        preserved_result='Isolated p08 ordinary-action snap press/grasp/lift/release gate passed.',
        potential_salvage='Tabletop one-snap shirt is a separate, unvalidated scope; do not relabel the coupon as a full garment success.',
        limitations=['No hardware calibration; physical values remain estimates.',
                     'Geometric witnesses establish failure, not its unique implementation or solver cause.',
                     'No full task controller, scorer, complete reset, or successful shirt demonstration exists.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root',type=Path,default=OUTPUT_ROOT)
    args=parser.parse_args()
    assert args.output_root.resolve()==OUTPUT_ROOT
    report=summarize(args.output_root)
    path=args.output_root/'p10-support-decision.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(decision=report['decision'],comparisons=len(report['comparisons']),full_task_complete=False)))
