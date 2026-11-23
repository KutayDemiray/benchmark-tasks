"""Audit aggregate single-snap evidence without awarding garment-task credit."""

from task_paths import OUTPUT_ROOT

import argparse
import json
from pathlib import Path

from audit import audit

OUTPUT = OUTPUT_ROOT


def review(coupled_run, reproduction_run=None):
    """Require positive mechanics, negative controls and reproduced robot coupling."""
    sources = {}
    checks = {}

    def load(name):
        path = OUTPUT/name
        assert path.resolve().is_relative_to(OUTPUT)
        verified = audit(path)
        assert all(e.get('independent_rescore_matches') for e in verified['episodes']), name
        assert json.loads((path/'exit.json').read_text())['returncode'] == 0, name
        sources[name] = verified
        return json.loads((path/'results.json').read_text())

    fixture = load('p07-reset-and-command-test')
    checks['aligned_press_and_1N_retention'] = all(
        r['seated_during_press'] and r['retained_under_1N'] and r['cloth_integrity']
        and not r['invalid_reasons'] for r in fixture)
    checks['axial_release_and_peel_with_deforming_fabric'] = all(
        r['released_under_limit'] and r['fabric_deformed'] for r in fixture)
    negative = load('p07-clearance-and-controls')
    rejects = {r['case']: r for r in negative if r['case'] in ('offset','tilt','wrong_side','no_press')}
    checks['misalignment_reversed_and_no_press_rejected'] = len(rejects) == 4 and all(
        not r['seated_during_press'] and not r['retained_under_1N']
        and r['cloth_integrity'] and not r['invalid_reasons'] for r in rejects.values())
    grasp = load('p08-controlled-release')[0]
    slip = load('p08-frictionless-control')[0]
    checks['ordinary_robot_cloth_grasp_and_controlled_release'] = bool(
        grasp['lifted_15mm'] and grasp['released_after_opening'] and grasp['cloth_integrity'])
    checks['zero_friction_does_not_lift'] = bool(not slip['lifted_15mm'] and slip['cloth_integrity'])
    press = load('p08-feedback-press-reset')
    checks['free_pair_robot_press_with_repeated_reset'] = len(press) == 2 and all(
        r['seated_after_robot_press'] and r['cloth_integrity'] and not r['invalid_reasons'] for r in press)
    no_press = load('p08-no-press-reset')
    checks['free_pair_does_not_self_close'] = len(no_press) == 2 and all(
        not r['seated_after_robot_press'] and not r['uncommanded_seating_detected']
        and r['peak_robot_normal_contact_N'] == 0 and r['cloth_integrity'] for r in no_press)
    def upper_only(name):
        from robot_scoring import upper_hardware_clearance
        valid = True
        evidence = []
        for episode in sorted((OUTPUT/name).glob('episode-*')):
            topology = json.loads((episode/'topology.json').read_text())
            held = [json.loads(line) for line in (episode/'trace.jsonl').read_text().splitlines()
                    if json.loads(line)['control']['phase']=='hold']
            values = [upper_hardware_clearance(row, topology) for row in held]
            valid &= bool(values) and min(values)>.001
            evidence.append({'episode': episode.name, 'hold_samples': len(values),
                             'minimum_upper_hardware_clearance_m': min(values) if values else None})
        sources[name]['supplemental_upper_hardware_exclusion'] = evidence
        return valid

    coupled = load(coupled_run)
    checks['upper_tab_carries_retained_lower_coupon_and_releases'] = bool(coupled) and all(
        r['lifted_pair_by_upper_fabric'] and r['released_pair_on_support']
        and r['cloth_integrity'] and not r['invalid_reasons'] for r in coupled)
    checks['upper_metal_cap_not_directly_grasped'] = upper_only(coupled_run)
    checks['fresh_process_coupled_controller_reproduction'] = False
    if reproduction_run is not None:
        assert reproduction_run != coupled_run, 'A run cannot reproduce itself'
        repeated = load(reproduction_run)
        same_source = sources[coupled_run]['bundle_sha256'] == sources[reproduction_run]['bundle_sha256']
        original_manifest = json.loads((OUTPUT/coupled_run/'manifest.json').read_text())
        repeated_manifest = json.loads((OUTPUT/reproduction_run/'manifest.json').read_text())
        distinct_process = original_manifest['pid'] != repeated_manifest['pid']
        original_setup = json.loads((OUTPUT/coupled_run/'episode-00/setup.json').read_text())
        repeated_setup = json.loads((OUTPUT/reproduction_run/'episode-00/setup.json').read_text())
        same_setup = original_setup['parameters'] == repeated_setup['parameters']
        checks['fresh_process_coupled_controller_reproduction'] = bool(repeated) and same_source and same_setup and distinct_process and all(
            r['lifted_pair_by_upper_fabric'] and r['released_pair_on_support']
            and r['cloth_integrity'] and not r['invalid_reasons'] for r in repeated) and upper_only(reproduction_run)
    return {
        'scope': 'Aggregate isolated functional mechanism gate; not garment-task success',
        'passed': all(checks.values()), 'checks': checks,
        'pending_or_failed': [name for name, passed in checks.items() if not passed],
        'audited_sources': sources, 'full_task_score': 0., 'full_task_success': False,
        'limits': [
            'Hardware and fabric laws are estimates, not measured Wrangler calibration.',
            'Fixture alignment probes validate capture/rejection; robot pressing starts with a documented aligned pair.',
            'Robot alignment of separate garment plackets remains an integration challenge.',
            'Peel trajectory replay fails local coordinate/orientation tolerances even where functional outcomes agree.',
            'Coupled lift excludes direct lower-piece support at recorded states; rigid socket force is checked each physics step.',
            'Multiple fasteners, wrong pairings, hanger support and full-task shortcuts require subsequent validation.',
            'This gate is a local development decision, not RoboCurve review approval.'
        ]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--coupled-run', required=True)
    parser.add_argument('--reproduction-run')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.out.resolve().is_relative_to(OUTPUT)
    result = review(args.coupled_run, args.reproduction_run)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'passed': result['passed'], 'pending_or_failed': result['pending_or_failed'],
                      'full_task_success': False}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
