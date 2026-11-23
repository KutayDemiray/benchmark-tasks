"""Measure a release-only diagnostic without granting benchmark credit."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from parallel_outcomes import observations


def assess(run, workers=4):
    """Report native clearance and settling using the unchanged outcome measurements."""
    request = json.loads((run / 'request.json').read_text())
    assert request.get('release_probe_run') and not request.get('release_probe_without_cover')
    result = json.loads((run / 'result.json').read_text())
    assert result['status'] == 'completed'
    rows = [json.loads(line) for line in (run / 'steps.jsonl').read_text().splitlines()]
    numerical = dict(finite=True, max_edge_strain=0., max_penetration_m=0., max_seam_gap_m=0.,
                     unexpected_constraint=False, diagnostic_fixture=True)
    records = []
    with (run / 'release-measurements.jsonl').open('w') as output:
        for observation, measured in observations(run, numerical, workers, 'release-'):
            row = rows[observation.step]
            normal = Rotation.from_quat(row['bridge_pose_xyzw'][3:]).apply([0., 0., 1.])
            inclination = float(np.rad2deg(np.arccos(np.clip(normal[2], -1., 1.))))
            record = dict(time_s=row['time_s'], head_inclination_deg=inclination,
                          observation=asdict(observation), **measured)
            records.append(record)
            output.write(json.dumps(record) + '\n')
    assert len(records) == result['control_steps'] and len(records) >= 40
    final = records[-40:]
    checks = dict(
        both_pockets_seated=all(all(p['seated_samples'] for p in r['pockets']) for r in final),
        passive_flat_wings=all(abs(r['observation']['wing_angle_deg']) <= 3.
                               and not r['observation']['wings_externally_supported'] for r in final),
        head_level=all(r['head_inclination_deg'] <= 5. for r in final),
        head_on_table=all(-.001 <= r['observation']['head_height_over_support_m'] <= .010
                          and r['observation']['over_support_surface'] for r in final),
        hands_clear=all(r['observation']['all_grippers_clearance_m'] >= .020
                        and not r['observation']['textile_grasped']
                        and not r['observation']['handle_grasped'] for r in final),
        native_settled=all(r['observation']['max_cloth_speed_m_s'] <= .03 for r in final),
        difference_settled=all(r['observation']['max_cloth_difference_speed_m_s'] <= .03 for r in final))
    audit_checks = {}
    for name, field in [('full-cloth-audit.json', 'recorded_numerical_checks_passed'),
                         ('frame-robot-audit.json', 'recorded_contact_checks_passed')]:
        path = run / name
        audit_checks[name] = False
        if path.exists():
            audit = json.loads(path.read_text())
            for source, digest in audit['source_sha256'].items():
                assert hashlib.sha256((run / source).read_bytes()).hexdigest() == digest, 'Audit inputs changed'
            audit_checks[name] = bool(audit[field] and audit.get('complete_recorded_step_coverage', audit.get('complete')))
    all_clear = all(r['observation']['all_grippers_clearance_m'] >= .020
                    and r['cloth_robot_clearance_lower_bound_m'] >= .002
                    and r['wings_robot_clearance_lower_bound_m'] >= .002
                    and r['handle_robot_clearance_lower_bound_m'] >= .002 for r in records)
    report = dict(diagnostic_only=True, benchmark_eligible=False, full_task_success=False,
                  source_run=run.name, final_window_s=[final[0]['time_s']-.05, final[-1]['time_s']],
                  final_conditions=checks, complete_audits=audit_checks,
                  release_diagnostic_passed=all(checks.values()) and all(audit_checks.values()),
                  hands_clear_entire_recording=all_clear,
                  minimum_hand_clearance_m=min(r['observation']['all_grippers_clearance_m'] for r in records),
                  maximum_head_inclination_deg=max(r['head_inclination_deg'] for r in records),
                  final_native_speed_m_s=max(r['observation']['max_cloth_speed_m_s'] for r in final),
                  final_difference_speed_m_s=max(r['observation']['max_cloth_difference_speed_m_s'] for r in final),
                  scope='Same native measurement and final two-second conditions as the task, plus a five-degree head-level diagnostic. Installation and retention history are absent; no benchmark score is issued.',
                  source_sha256={name: hashlib.sha256((run / name).read_bytes()).hexdigest()
                                 for name in ('request.json', 'result.json', 'release-measurements.jsonl',
                                              'release-observation-parallel-provenance.json')},
                  assessor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (run / 'release-assessment.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    assess(args.run, args.workers)
