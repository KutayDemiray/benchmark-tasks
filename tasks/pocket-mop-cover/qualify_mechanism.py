"""Qualify the isolated contact mechanism from positive and negative evidence."""

from task_paths import OUTPUT_ROOT

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

OUT = OUTPUT_ROOT


def canonical_geometry(metadata):
    """Remove the deliberately changed catch collision flags and test fixture."""
    value = deepcopy(metadata)
    value.pop('latch_contacts')
    value.pop('pedal_pusher', None)
    for collider in value['colliders']:
        collider.pop('enabled')
    return value


def qualify(positive_name, negative_name, output_name="mechanism-gate.json"):
    assert Path(output_name).name == output_name and output_name.endswith(".json")
    positive, negative = OUT / positive_name, OUT / negative_name
    for path in (positive, negative):
        assert path.resolve().parent == OUT
    def read(path, name):
        return json.loads((path / name).read_text())
    pa, na = [read(path, 'mechanism-audit.json') for path in (positive, negative)]
    pr, nr = [read(path, 'request.json') for path in (positive, negative)]
    pm, nm = [read(path, 'mechanism.json') for path in (positive, negative)]
    source_hashes = [hashlib.sha256((path / 'source/frame_native.py').read_bytes()).hexdigest() for path in (positive, negative)]
    from frame_solver import configuration
    matched_solver = configuration(pr) == configuration(nr)
    if pr.get('frame_contact_correction_speed') is not None:
        solver_hashes = [hashlib.sha256((path/'source/frame_solver.py').read_bytes()).hexdigest() for path in (positive,negative)]
        matched_solver = matched_solver and solver_hashes[0] == solver_hashes[1]
        for path, request in ((positive,pr),(negative,nr)):
            authored = read(path,'frame-numerics.json')
            matched_solver = matched_solver and authored['configuration'] == configuration(request)
            matched_solver = matched_solver and authored['authored_before_simulation'] and authored['constant_throughout_episode']
            matched_solver = matched_solver and len(authored['changes']) == 7
            matched_solver = matched_solver and all(abs(change['after']-request['frame_contact_correction_speed']) < 1e-7 for change in authored['changes'])
    backends = ['installed_ovphysx' if r.get('mechanism_sdk_source') else 'kit_physx' for r in (pr,nr)]
    from passive_contact import native_physics_errors
    native_ok = all(not native_physics_errors((path/'kit.log').read_text()) for path in (positive,negative))
    sdk_matches = True
    if 'installed_ovphysx' in backends:
        sdk_matches = backends[0] == backends[1]
        if sdk_matches:
            native_records = [read(path,'sdk-native-properties.json') for path in (positive,negative)]
            keys = ('body_names','joint_names','is_fixed_base','stiffness','damping','maximum_forces',
                    'masses_kg','joint_limits_rad_or_m','body_gravity_disabled','max_joint_velocities_rad_s_or_m_s')
            sdk_matches = all(native_records[0][key] == native_records[1][key] for key in keys)
            sdk_matches = sdk_matches and all(read(path,'native-initialization-check.json')['passed'] for path in (positive,negative))
            versions = [read(path,'versions.json') for path in (positive,negative)]
            sdk_matches = sdk_matches and versions[0].get('ovphysx') == versions[1].get('ovphysx')
            changes = [[c for c in read(path,'sdk-source-preservation.json')['changes'] if c['path']=='/World/Frame'] for path in (positive,negative)]
            sdk_matches = sdk_matches and changes[0] == changes[1]
    checks = {
        'matched_frame_contact_solver': matched_solver,
        'matched_physics_backend': backends[0] == backends[1],
        'native_error_free': native_ok,
        'matched_sdk_native_properties': sdk_matches,
        'completed_processes': all(read(path, 'result.json')['status'] == 'completed' and read(path, 'process.json')['returncode'] == 0 for path in (positive, negative)),
        'identical_frame_source': source_hashes[0] == source_hashes[1],
        'identical_frame_geometry_and_materials': canonical_geometry(pm) == canonical_geometry(nm),
        'matched_press_and_physics': all(pr[key] == nr[key] for key in ('physics_hz', 'iterations', 'press_overtravel')),
        'matched_retention_duration': pr.get('retention_extension',0.)==nr.get('retention_extension',0.),
        'positive_contacts_enabled': pm['latch_contacts'] and not pr['disable_latch_contact'],
        'negative_contacts_disabled': not nm['latch_contacts'] and nr['disable_latch_contact'],
        'both_rigid_audits_pass': pa['rigid_checks_passed'] and na['rigid_checks_passed'],
        'flat_retention_entire_observed_window': pa['retained'] and pa['final_two_seconds_wing_angle_deg'] <= 3.,
        'retained_lift_over_100mm': pa['actual_press_lift_m'] >= .1,
        'negative_folds_without_catch': not na['retained'] and na['final_two_seconds_wing_angle_deg'] > 30. and na['actual_press_lift_m'] >= .1,
        'physical_pedal_unlock': pr['mechanism_unlock'] and pa['physical_unlock_observed'] and pa['final_unlocked_min_wing_angle_deg'] > 30.,
    }
    hashes = {}
    for folder in (positive, negative):
        names = ['request.json', 'mechanism.json', 'mechanism-audit.json', 'manifest.json', 'gpu-verification.json', 'process.json']
        if pr.get('frame_contact_correction_speed') is not None:
            names.append('frame-numerics.json')
        if backends[0] == backends[1] == 'installed_ovphysx':
            names.extend(('sdk-native-properties.json','sdk-source-preservation.json','native-initialization-check.json','versions.json'))
        for name in names:
            path = folder / name
            hashes[str(path.relative_to(OUT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = dict(passed=all(checks.values()), full_task_success=False, physics_backend=backends[0],
        scope='Isolated estimated Freudenberg-family folding frame: contact lock, lifted retention, disabled-catch control and physical pedal unlock.',
        positive_run=positive_name, negative_run=negative_name, checks=checks,
        evidence_sha256=hashes, retained_angle_deg=pa['final_two_seconds_wing_angle_deg'],
        retention_window_s=pa['retention_window_s'],retained_lift_m=pa['actual_press_lift_m'], unlocked_angle_deg=pa['final_unlocked_min_wing_angle_deg'],
        peak_audited_penetration_m=max(pa['floor_penetration_m'], pa['nonadjacent_convex_penetration_m'], na['floor_penetration_m'], na['nonadjacent_convex_penetration_m']),
        limitations=['Laboratory force-limited press and pedal pusher; no robot or textile.',
            'Dimensions and polypropylene compliance estimated from a related mechanism family, not measured CombiSpeed internals.',
            'Adjacent hinge-bearing contacts excluded; handle proxy is a conservative convex outer bound.',
            'This does not qualify the cloth, assembled task, reset or independent physics replay.'])
    (OUT / output_name).write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--positive', required=True)
    parser.add_argument('--negative', required=True)
    parser.add_argument('--output', default='mechanism-gate.json')
    args = parser.parse_args()
    result = qualify(args.positive, args.negative, args.output)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result['passed'] else 1)
