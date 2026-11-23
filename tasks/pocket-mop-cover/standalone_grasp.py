"""Re-execute the qualified stock-YAM grasp actions in an isolated SDK scene."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import shutil
import traceback

import numpy as np


def validate_source(request, output):
    source_name = request.get('grasp_sdk_source')
    assert source_name and Path(source_name).name == source_name and source_name not in ('.', '..')
    assert request['experiment'] == 'robot_grasp'
    source = output/source_name
    gate = json.loads((output/'product-gate.json').read_text())
    assert gate['passed'] and gate['grasp_run'] == source_name
    original = json.loads((source/'request.json').read_text())
    assert not any(request.get(key) for key in ('reinforced', 'clamp_tail', 'finite_seam', 'flat_rest_panels',
        'flat_rest_bending', 'detach_lip', 'cloth_contact_correction_speed', 'bend_damping_rate', 'cooked_collider_probe'))
    keys = ('physics_hz', 'iterations', 'spacing', 'seconds', 'repeats', 'contact_updates',
            'contact_subiterations', 'ccd', 'product_profile', 'mouth_height', 'table_origin',
            'grasp_margin_clearance', 'grasp_opening', 'close_seconds', 'placement', 'two_hands',
            'placement_tilt', 'release_opening', 'left_grasp_height', 'right_grasp_height', 'seed')
    for key in keys:
        assert request[key] == original[key], 'SDK grasp must preserve ' + key
    assert json.loads((source/'grasp-audit.json').read_text())['grasp_gate_passed']
    for name in ('native.py', 'mesh.py', 'parameters.json'):
        assert (source/'source'/name).read_bytes() == Path(__file__).with_name(name).read_bytes(), 'Original source changed: ' + name
    old_tree = ast.parse((source/'source/robot_gate.py').read_text())
    new_tree = ast.parse(Path(__file__).with_name('robot_gate.py').read_text())
    for name in ('fabric', 'surface_asset', 'solve_pose'):
        old = next(node for node in old_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        new = next(node for node in new_tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
        assert ast.dump(old) == ast.dump(new), 'Original physical authoring function changed: ' + name
    reference_name = request.get('grasp_sdk_reference')
    assert reference_name and Path(reference_name).name == reference_name and reference_name not in ('.', '..')
    reference = output/reference_name
    reference_request = json.loads((reference/'request.json').read_text())
    assert reference_request['cooked_collider_probe'] and reference_request['state_run'] == source_name
    assert json.loads((reference/'result.json').read_text())['status'] == 'completed'
    assert json.loads((reference/'process.json').read_text())['returncode'] == 0
    assert (reference/'native-yam-contract.json').is_file()
    for key in keys:
        assert request[key] == reference_request[key], 'Kit robot reference must preserve ' + key
    return source


def verify_station_copy(source, out):
    """Independently re-read the preserved authored world coordinates."""
    from pxr import Usd, UsdGeom
    stage = Usd.Stage.Open(str(out/'scene.usda'))
    expected = json.loads((source/'station-pose-check.json').read_text())
    cache = UsdGeom.XformCache()
    measured = {}
    for name, record in expected.items():
        prim = stage.GetPrimAtPath('/World/envs/env_0/' + name)
        assert prim
        transform = cache.GetLocalToWorldTransform(prim)
        p = np.array(transform.ExtractTranslation())
        q = transform.ExtractRotationQuat()
        q = np.array([*q.GetImaginary(), q.GetReal()])
        target = np.array(record['quaternion_xyzw'])
        distance = float(np.linalg.norm(p-np.array(record['position_m'])))
        angle = float(2*np.arccos(np.clip(abs(q@target)/(np.linalg.norm(q)*np.linalg.norm(target)), 0., 1.)))
        assert distance < 2e-6 and angle < 2e-5
        measured[name] = dict(position_m=p.tolist(), quaternion_xyzw=q.tolist(),
                              position_error_m=distance, orientation_error_rad=angle)
    (out/'station-pose-check.json').write_text(json.dumps(measured, indent=2))


def run(request, out):
    source = validate_source(request, out.parent)
    report = dict(status='initializing', full_task_success=False, product_gate_passed=False,
                  diagnostic_only=True, backend='installed_ovphysx')
    sim = cameras = None
    chunks = []
    physics_count = chunk_count = 0
    def flush():
        nonlocal chunk_count
        if chunks:
            data = {key: np.stack([r[key] for r in chunks]) for key in chunks[0]}
            np.savez_compressed(out/'physics-chunks'/f'{chunk_count:05d}.npz',
                                physics_end_step=physics_count, **data)
            chunks.clear()
            chunk_count += 1
    try:
        from sdk_session import SDKSession, author_defaults, surface_bindings
        from sdk_yam import SDKYam
        from mesh import surface_edges, edge_strain
        from yam_arena import ik, station
        from scipy.spatial.transform import Rotation
        for name in ('scene.usda', 'scene-audit.json', 'geometry.npz', 'mesh.json',
                     'effective-parameters.json', 'gripper-colliders.json', 'controller-plan.json', 'grasp-observation.json'):
            shutil.copyfile(source/name, out/name)
        (out/'physics-chunks').mkdir()
        material = json.loads((out/'effective-parameters.json').read_text())
        geometry = np.load(out/'geometry.npz')
        rest, initial, faces = [geometry[name] for name in ('rest', 'initial', 'faces')]
        root = '/World/envs/env_0'
        author_defaults(out/'scene.usda', out, root+'/Pocket', [root+'/LeftArm', root+'/RightArm'])
        verify_station_copy(source, out)
        scene = json.loads((out/'scene-audit.json').read_text())
        shift = np.asarray(scene['translation_S_W_m'])
        from sdk_cooked_geometry import capture
        capture(out, out.parent/request['grasp_sdk_reference'], geometry_kind='gripper')
        sim = SDKSession(out, enable_rendering=True)
        hz = request['physics_hz']
        dt = 1/hz
        decimation = hz//20
        sim.step(dt)
        x, v = surface_bindings(sim, root+'/Pocket', material, rest, faces)
        robot = SDKYam(sim)
        from capture_yam_contract import compare as compare_yam_contract
        compare_yam_contract(out.parent/request['grasp_sdk_reference'],
                             json.loads((out/'sdk-yam-contract.json').read_text()), out)
        points = sim.read(x)
        points[0, :len(initial)] = initial
        x.write(points)
        v.write(np.zeros(v.shape, dtype=np.float32))
        sim.engine.update_articulations_kinematic()
        rigid_geometry = json.loads((out/'gripper-colliders.json').read_text())
        gripper_ids = {}
        fk_checks = {}
        for arm in robot.arms:
            side = arm['side']
            selected_names = rigid_geometry[side]['body_names']
            gripper_ids[side] = [arm['poses'].body_names.index(name) for name in selected_names]
            poses = sim.read(arm['poses'])[0]
            gripper = poses[arm['poses'].body_names.index('gripper')]
            tcp = gripper[:3]+Rotation.from_quat(gripper[3:]).apply(robot.constants.TCP_OFFSET)
            expected, rotation = ik.tcp_pose(np.asarray(robot.constants.READY), station.BASES[side])
            error = float(np.linalg.norm(tcp-(expected+shift)))
            assert error < 1e-5, 'SDK reset differs from stock YAM forward kinematics'
            fk_checks[side] = dict(tcp_error_m=error, native_tcp_m=tcp.tolist())
        from sdk_cameras import SDKCameras
        cameras = SDKCameras(sim, robot, root+'/Pocket')
        cameras.warmup_and_verify(sim.read(x)[0, :len(initial)], x)
        (out/'native-initialization-check.json').write_text(json.dumps(dict(passed=not sim.errors, native_errors=sim.errors), indent=2))
        reset = dict(position_error_m=float(abs(sim.read(x)[0, :len(initial)]-initial).max()),
                     speed_m_s=float(np.linalg.norm(sim.read(v)[0, :len(initial)], axis=1).max()),
                     robot_fk=fk_checks, state_writes_only_at_initialization=True)
        assert reset['position_error_m'] < 1e-6 and reset['speed_m_s'] == 0.
        (out/'sdk-reset-check.json').write_text(json.dumps(reset, indent=2))
        edges = surface_edges(faces)
        source_rows = [json.loads(line) for line in (source/'steps.jsonl').read_text().splitlines()]
        assert len(source_rows) <= request['seconds']*20
        patch = json.loads((out/'grasp-observation.json').read_text())['patch_ids']
        max_strain = 0.
        lift_origin = None
        max_action_error = 0.
        with (out/'steps.jsonl').open('w', buffering=1) as log:
            for index, source_row in enumerate(source_rows):
                action = np.asarray(source_row['robot_action'])
                phase = source_row['phase']
                if phase == 'lift' and lift_origin is None:
                    lift_origin = sim.read(x)[0, patch].mean(0)
                robot.command(action)
                max_action_error = max(max_action_error, float(abs(robot.action-action).max()))
                for _ in range(decimation):
                    robot.apply_gravity_compensation()
                    sim.step(dt)
                    points = sim.read(x)[0, :len(initial)]
                    velocity = sim.read(v)[0, :len(initial)]
                    assert np.isfinite(points).all() and np.isfinite(velocity).all()
                    chunk = dict(positions=points, velocities=velocity)
                    for arm in robot.arms:
                        poses = sim.read(arm['poses'])[0]
                        chunk[arm['side']+'_grasp_poses'] = poses[gripper_ids[arm['side']]]
                        chunk[arm['side']+'_all_poses'] = poses
                    chunks.append(chunk)
                    physics_count += 1
                    max_strain = max(max_strain, float(abs(edge_strain(rest, points, faces, edges=edges)).max()))
                    if len(chunks) >= hz:
                        flush()
                    assert max_strain < 2., 'Catastrophic cloth distortion'
                if phase == 'hold_grasp':
                    lifted = points[patch].mean(0)
                    report['patch_lift_m'] = float(lifted[2]-lift_origin[2])
                    support_top = scene['support_center_m'][2]+scene['support_size_m'][2]/2
                    report['cloth_min_height_over_block_m'] = float(points[:, 2].min()-support_top)
                row = dict(step=index, time_s=(index+1)/20, phase=phase, robot_action=robot.action.tolist(),
                    measured_joint_pos=robot.read14().tolist(), max_abs_edge_strain=max_strain,
                    min_cloth_z_m=float(points[:, 2].min()), cloth_center_m=points.mean(0).tolist(),
                    score=0., full_task_success=False)
                log.write(json.dumps(row)+'\n')
                cameras.record(points)
                if (index+1)%20 == 0:
                    (out/'heartbeat.json').write_text(json.dumps(row, indent=2))
                    print(json.dumps(row), flush=True)
        flush()
        (out/'sdk-source-preservation.json').write_text(json.dumps(dict(source_run=source.name,
            source_sha256={name: hashlib.sha256((source/name).read_bytes()).hexdigest()
                           for name in ('scene.usda', 'steps.jsonl', 'geometry.npz', 'manifest.json')},
            replayed_ordinary_actions=True, maximum_action_difference=max_action_error,
            original_geometry_material=True, state_writes_only_at_initialization=True,
            cameras_not_yet_qualified=True, camera_frames=cameras.count), indent=2))
        report.update(status='completed', termination_reason='diagnostic_complete', control_steps=len(source_rows),
            physics_steps=physics_count, max_abs_edge_strain=max_strain,
            grasp_observed=report.get('patch_lift_m', 0.) > .08, numerical_gate_passed=max_strain <= .2,
            limitations=['Native SDK repeat of a single-pocket robot grasp; not a full task.',
                'Requires complete independent cloth/contact audits and current-runtime camera videos.',
                'Recorded robot actions are replayed; no recorded object/robot poses drive the simulation.'])
    except Exception:
        report.update(status='error', termination_reason='runtime_error', traceback=traceback.format_exc())
        print(report['traceback'], flush=True)
    finally:
        flush()
        (out/'result.json').write_text(json.dumps(report, indent=2))
        try:
            if cameras is not None:
                cameras.close()
        finally:
            if sim is not None:
                sim.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    args = parser.parse_args()
    run(json.loads(args.request.read_text()), args.request.parent)
