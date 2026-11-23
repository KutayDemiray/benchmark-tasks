"""Isolated snap diagnostics. Fixture wrenches are never robot actions."""

from task_paths import OUTPUT_ROOT

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import numpy as np

HERE = Path(__file__).resolve().parent
CASES = ('press', 'no_press', 'offset', 'tilt', 'wrong_side', 'peel', 'peel_pull', 'peel_controlled',
         'grasp', 'frictionless', 'torque_free')


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write(path, value):
    path.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + '\n')


def snapshot(out):
    bundle = out / 'task-bundle'
    bundle.mkdir(exist_ok=HERE.resolve() == bundle.resolve())
    hashes = {}
    for source in sorted(HERE.iterdir()):
        if source.is_file():
            if source.resolve() != (bundle / source.name).resolve():
                shutil.copy2(source, bundle / source.name)
            hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    digest = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    versions = {}
    for name in ('isaacsim', 'isaaclab', 'isaaclab-arena', 'yam-arena', 'numpy', 'torch', 'warp-lang'):
        versions[name] = importlib.metadata.version(name)
    for name in ('inspect-robots','inspect-robots-yam','inspect-robots-dream','dreamlabs-client',
                 'imageio','imageio-ffmpeg','scipy'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]='not installed'
    versions['python']=__import__('sys').version
    write(out / 'manifest.json', {'sources': hashes, 'bundle_sha256': digest, 'versions': versions,
                                 'pid': os.getpid(), 'command': __import__('sys').argv})
    return digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case', choices=CASES, default='press')
    parser.add_argument('--cases', help='Comma-separated diagnostic cases in one process')
    parser.add_argument('--robust', action='store_true')
    parser.add_argument('--physics-hz', type=int)
    parser.add_argument('--snap-max-angular-speed', type=float,
                        help='Diagnostic numerical limit in rad/s, for snap hardware only')
    parser.add_argument('--free-torque', type=float, default=1e-7,
                        help='Free-body actuator-check torque in Nm')
    parser.add_argument('--replay', type=Path, help='Prior episode directory containing per-step fixture inputs')
    parser.add_argument('--replay-mode', choices=('forces', 'commands'), default='commands')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--no-cloth', action='store_true')
    parser.add_argument('--steps', type=int, default=4800)
    parser.add_argument('--measure-every', type=int, default=1, help='Physics-step spacing of cloth observations; validation requires 1')
    args = parser.parse_args()
    assert args.measure_every >= 1 and args.steps > 0 and args.episodes > 0
    from gpu_runtime import inherited_fds
    inherited_fds()
    out = args.out.resolve()
    assert out.is_relative_to(OUTPUT_ROOT)
    digest = snapshot(out)
    cfg = json.loads((HERE / 'parameters.json').read_text())
    if args.replay:
        assert not args.robust and not args.cases and args.episodes == 1, 'Replay is one frozen episode'
        replay_setup = json.loads((args.replay / 'setup.json').read_text())
        cfg = replay_setup['parameters']
        args.case = replay_setup['case']
        args.seed = replay_setup['seed']
        args.no_cloth = replay_setup['no_cloth']
        args.steps = len(np.load(args.replay / 'per-step.npz')['forces'])
    if args.physics_hz:
        cfg['simulation']['physics_hz'] = args.physics_hz
    if args.snap_max_angular_speed is not None:
        assert args.snap_max_angular_speed > 0.
        cfg['simulation']['snap_max_angular_speed_rad_s'] = args.snap_max_angular_speed
    if not args.replay:
        cfg['fixture']['free_torque_Nm'] = args.free_torque
    dt = 1 / cfg['simulation']['physics_hz']
    privacy = out / 'runtime/privacy.toml'
    privacy.write_text('[privacy]\nperformance=false\nusage=false\npersonalization=false\n')
    # SimulationApp otherwise forces package-local portable caches. Pass an
    # owned portable root through its documented command-line mechanism.
    import sys
    sys.argv = [sys.argv[0], '--portable-root', str(out.parent / 'kit-portable')]
    from network_guard import deny_internet_sockets
    deny_internet_sockets()
    from isaacsim import SimulationApp
    from gpu_runtime import assignment, kit_arguments, verify_native
    binding_args = kit_arguments(out/'kit.log').split()
    app = SimulationApp({'headless': True, 'active_gpu': assignment()['index'], 'physics_gpu': 0, 'multi_gpu': False,
                         'enable_crashreporter': False,
                         'extra_args': [*binding_args, '--ext-folder', str(Path(os.environ['ISAAC_PATH']) / 'extscache'),
                                        f'--/log/file={out}/kit.log', '--/app/settings/persistent=false',
                                        '--/telemetry/enableAnonymousData=false', '--/telemetry/enableSentry=false',
                                        '--/telemetry/enableNVDF=false', '--/telemetry/useOpenEndpoint=false',
                                        '--/exts/omni.kit.telemetry/skipDeferredStartup=true',
                                        f'--/structuredLog/privacySettingsFile={privacy}',
                                        f'--/structuredLog/logDirectory={out}/runtime/structured',
                                        f'--/telemetry/log/file={out}/runtime/telemetry.log',
                                        f'--/app/userConfigPath={out}/runtime/kit.config.json',
                                        f'--/app/tokens/cache={out.parent}/runtime-cache/kit',
                                        f'--/app/tokens/data={out}/runtime/kit-data',
                                        f'--/app/tokens/logs={out}/runtime/kit-logs']}, experience=str(HERE / 'bench.kit'))
    import carb
    import omni.usd
    import omni.physx
    import omni.physics.tensors as tensors
    import torch
    from pxr import UsdGeom, UsdPhysics, PhysxSchema, Gf, UsdLux
    from geometry import build, build_grasp
    from scoring import analyze

    settings = carb.settings.get_settings()
    settings.set('/physics/cudaDevice', 0)
    settings.set('/physics/updateToUsd', True)
    settings.set('/physics/updateVelocitiesToUsd', True)
    settings.set('/physics/suppressReadback', True)
    results = []
    cases = args.cases.split(',') if args.cases else [args.case]
    cases *= args.episodes
    base_cfg = copy.deepcopy(cfg)
    try:
        for episode, case in enumerate(cases):
            assert case in CASES
            cfg = copy.deepcopy(base_cfg)
            if case == 'torque_free':
                assert args.no_cloth, 'Free-torque actuator check uses no cloth'
                cfg['simulation']['gravity_m_s2'] = 0.
            rng = np.random.default_rng(args.seed + episode)
            dest = out / f'episode-{episode:02d}'
            dest.mkdir()
            omni.usd.get_context().new_stage()
            stage = omni.usd.get_context().get_stage()
            world = UsdGeom.Xform.Define(stage, '/World')
            stage.SetDefaultPrim(world.GetPrim())
            UsdGeom.SetStageMetersPerUnit(stage, 1.)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            scene = UsdPhysics.Scene.Define(stage, '/World/physics')
            scene.CreateGravityDirectionAttr(Gf.Vec3f(0, 0, -1))
            scene.CreateGravityMagnitudeAttr(cfg['simulation']['gravity_m_s2'])
            phys = PhysxSchema.PhysxSceneAPI.Apply(scene.GetPrim())
            phys.CreateEnableGPUDynamicsAttr(True)
            phys.CreateBroadphaseTypeAttr('GPU')
            phys.CreateSolverTypeAttr('TGS')
            phys.CreateTimeStepsPerSecondAttr(cfg['simulation']['physics_hz'])
            scene.GetPrim().GetAttribute('physxScene:enableExternalForcesEveryIteration').Set(False)
            light = UsdLux.DomeLight.Define(stage, '/World/light')
            light.CreateIntensityAttr(800.)
            offset = .004 if case == 'offset' else 0.
            angle = np.deg2rad(35) if case == 'tilt' else 0.
            if case == 'wrong_side':
                angle = np.pi
            if args.robust:
                offset += float(rng.uniform(-.00015, .00015))
                cfg['snap']['spring_stiffness_N_m'] *= float(rng.uniform(.9, 1.1))
                cfg['snap']['friction'] *= float(rng.uniform(.8, 1.2))
                cfg['fabric']['friction'] *= float(rng.uniform(.85, 1.15))
                cfg['fabric']['youngs_modulus_Pa'] *= float(rng.uniform(.9, 1.1))
            grasp = case in ('grasp', 'frictionless')
            assert not (grasp and args.no_cloth), 'Pinch cases require the fabric coupon'
            if case == 'frictionless':
                cfg['fabric']['friction'] = 0.
            replay_inputs = None
            if args.replay:
                setup = json.loads((args.replay / 'setup.json').read_text())
                cfg, case, offset, angle = setup['parameters'], setup['case'], setup['offset_m'], setup['angle_rad']
                assert cfg['simulation']['physics_hz'] == 1/dt
                replay_inputs = np.load(args.replay / 'per-step.npz')
                assert args.replay_mode != 'commands' or grasp or 'fixture_commands' in replay_inputs, (
                    'Legacy archive has no actuator-command stream; use --replay-mode forces')
            grasp = case in ('grasp', 'frictionless')
            write(dest / 'setup.json', {'parameters': cfg, 'case': case, 'offset_m': offset, 'angle_rad': angle,
                                       'seed': args.seed + episode, 'no_cloth': args.no_cloth})
            data = build_grasp(stage, cfg) if grasp else build(stage, cfg, offset=offset, angle=angle, guided=case not in ('peel', 'peel_pull', 'peel_controlled', 'torque_free'), cloth=not args.no_cloth)
            male_api = PhysxSchema.PhysxRigidBodyAPI(stage.GetPrimAtPath('/World/male'))
            max_angular_speed = cfg['simulation'].get('snap_max_angular_speed_rad_s')
            if max_angular_speed is not None:
                # USD stores this limit in degrees/s; tensor velocities use rad/s.
                male_api.CreateMaxAngularVelocityAttr(float(np.rad2deg(max_angular_speed)))
            stage.GetRootLayer().Export(str(dest / 'initial.usda'))
            write(dest / 'topology.json', data)
            interface = omni.physx.get_physx_simulation_interface()
            interface.attach_stage(omni.usd.get_context().get_stage_id())
            interface.simulate(dt, 0.)
            interface.fetch_results()
            sim = tensors.create_simulation_view('torch')
            sim.set_subspace_roots('/')
            assert sim.device_ordinal == 0, f'Unexpected physics device {sim.device_ordinal}'
            male = sim.create_rigid_body_view('/World/male')
            assert male.count == 1, 'Reset left unexpected rigid bodies'
            gpu_record = verify_native(sim, male.get_transforms().device, dest, out/'kit.log')
            gpu_record.update(physics_device_ordinal=sim.device_ordinal, torch_device=str(sim.device),
                              cuda_name=torch.cuda.get_device_name(0),
                              cuda_uuid=str(torch.cuda.get_device_properties(0).uuid),
                              renderer_activeGpu=gpu_record['renderer_index'], multi_gpu=False)
            write(dest/'gpu.json', gpu_record)
            write(dest / 'native-mass-inertia.json', {
                'mass': male.get_masses().cpu().numpy(),
                'inertia': male.get_inertias().cpu().numpy(),
                'max_angular_speed_deg_s': male_api.GetMaxAngularVelocityAttr().Get(),
                'angular_damping_s_inv': male_api.GetAngularDampingAttr().Get(),
            })
            balls = None if grasp else sim.create_rigid_body_view('/World/ball*')
            pads = sim.create_articulation_view('/World/pinch/base') if grasp else None
            if pads is not None:
                print('FIXTURE_DOFS ' + str(pads.shared_metatype.dof_names), flush=True)
                write(dest / 'pinch-dofs.json', {'dofs': pads.shared_metatype.dof_names, 'links': pads.shared_metatype.link_names})
            cloth_views = {name: sim.create_surface_deformable_body_view('/World/' + name) for name in data['cloth']}
            ids = torch.tensor([0], dtype=torch.int32, device='cuda:0')
            force = torch.zeros((1, 3), device='cuda:0')
            torque = torch.zeros((1, 3), device='cuda:0')
            records = []
            trace_file = (dest / 'trace.jsonl').open('w', buffering=1)
            per_step = {'forces': [], 'torques': [], 'male_poses': [], 'male_velocities': [], 'ball_poses': [],
                        'pad_targets': [], 'fixture_joint_positions': [], 'fixture_link_poses': [],
                        'fixture_commands': []}
            write(dest / 'command-format.json', {
                'columns': ['z_position_mode', 'target_z_m', 'force_z_N',
                            'roll_position_mode', 'target_roll_rad', 'torque_x_Nm'],
                'position_mode': '1 = bounded PD actuator using frozen fixture gains; 0 = prescribed wrench',
                'grasp': 'pinch articulation commands are in pad_targets instead',
                'robot_actions': False,
            })
            cloth_frames = {name: [] for name in cloth_views}
            start = time.monotonic()
            release_origin = None
            for step in range(args.steps):
                t = step * dt
                pose = male.get_transforms().cpu().numpy()[0].copy()
                velocity = male.get_velocities().cpu().numpy()[0].copy()
                z = pose[2] - .1
                fz = 0.
                tx = 0.
                z_position_mode, roll_position_mode = False, False
                target_z, target_roll = 0., 0.
                pad_target = np.empty((0, 7), dtype=np.float32)
                if case == 'torque_free':
                    phase = 'free_torque_actuator_check'
                    tx = cfg['fixture'].get('free_torque_Nm', 1e-7)
                elif grasp:
                    fixture = cfg['pinch_fixture']
                    phase = 'pinch_close' if t < fixture['close_seconds'] else 'pinch_hold' if t < .7 else 'pinch_lift' if t < 1.7 else 'pinch_hold_raised' if t < 2.2 else 'pinch_open'
                    closed_gap = fixture['closed_gap_m']
                    gap = .004 - (.004-closed_gap) * min(t/fixture['close_seconds'], 1.) if t < 2.2 else closed_gap + .024 * min((t-2.2)/.2, 1.)
                    lift = fixture['lift_m'] * np.clip((t-.7)/1., 0, 1)
                    pad_height = fixture['pad_center_height_m']
                    pad_target = np.array([[-(.003+gap/2), 0, pad_height+lift, 0, 0, 0, 1],
                                           [(.003+gap/2), 0, pad_height+lift, 0, 0, 0, 1]], dtype=np.float32)
                    if replay_inputs is not None:
                        pad_target = replay_inputs['pad_targets'][step]
                    target_by_name = {'lift0': float(pad_target[0, 2] - pad_height), 'lift1': float(pad_target[1, 2] - pad_height),
                                      'close0': float(pad_target[0, 0] + .005), 'close1': float(pad_target[1, 0] - .005)}
                    targets = [[target_by_name[name] for name in pads.shared_metatype.dof_names]]
                    pads.set_dof_position_targets(torch.tensor(targets, device='cuda:0'), ids)
                elif case == 'no_press':
                    phase = 'no_press'
                elif t < .3:
                    phase = 'settle'
                    z_position_mode, target_z = True, .1 + cfg['fixture']['initial_height_m']
                    fz = cfg['fixture']['kp_N_m'] * (target_z - pose[2]) - cfg['fixture']['kd_Ns_m'] * velocity[2]
                elif t < 2.3:
                    phase = 'press'
                    initial_z = .1 + cfg['fixture']['initial_height_m']
                    press_z = .1 + cfg['fixture']['press_target_m']
                    target = initial_z - min((t - .3) / 2., 1.) * (initial_z - press_z)
                    z_position_mode, target_z = True, target
                    fz = cfg['fixture']['kp_N_m'] * (target - pose[2]) - cfg['fixture']['kd_Ns_m'] * velocity[2]
                elif t < 2.8:
                    phase = 'retention'
                    fz = 1.
                elif t < 3.3:
                    phase = 'unloaded'
                else:
                    phase = 'release'
                    if case == 'peel_controlled':
                        if release_origin is None:
                            release_origin = float(pose[2])
                        target = release_origin + cfg['fixture']['release_speed_m_s'] * (t - 3.3)
                        z_position_mode, target_z = True, target
                        fz = cfg['fixture']['kp_N_m'] * (target - pose[2]) - cfg['fixture']['kd_Ns_m'] * velocity[2]
                        # Diagnostic torque actuator, not a pose write or robot gain.
                        roll = np.arctan2(2*(pose[6]*pose[3]+pose[4]*pose[5]),
                                          1-2*(pose[3]**2+pose[4]**2))
                        fixture = cfg['fixture']
                        target_roll = min((t-3.3)*fixture['peel_angular_speed_rad_s'], fixture['peel_target_angle_rad'])
                        roll_position_mode = True
                        tx = float(np.clip(fixture['peel_kp_Nm_rad']*(target_roll-roll)
                                           - fixture['peel_kd_Nms_rad']*velocity[3], -.05, .05))
                    elif case in ('peel', 'peel_pull'):
                        tx = min((t - 3.3) * .04, .05)
                        if case == 'peel_pull':
                            fz = 1.
                    else:
                        if cfg['fixture'].get('release_protocol') == 'displacement':
                            if release_origin is None:
                                release_origin = float(pose[2])
                            target = release_origin + cfg['fixture']['release_speed_m_s'] * (t - 3.3)
                            z_position_mode, target_z = True, target
                            fz = cfg['fixture']['kp_N_m'] * (target - pose[2]) - cfg['fixture']['kd_Ns_m'] * velocity[2]
                        else:
                            fz = min((t - 3.3) * 15, 25)
                command = np.array([int(z_position_mode), target_z, 0. if z_position_mode else fz,
                                    int(roll_position_mode), target_roll, 0. if roll_position_mode else tx], dtype=np.float64)
                if replay_inputs is not None and args.replay_mode == 'commands' and not grasp:
                    command = replay_inputs['fixture_commands'][step].copy()
                    # Restore Python scalars to preserve the original float32
                    # sensor arithmetic, rather than silently promoting to float64.
                    fz = (cfg['fixture']['kp_N_m'] * (float(command[1]) - pose[2])
                          - cfg['fixture']['kd_Ns_m'] * velocity[2]) if command[0] else float(command[2])
                    if command[3]:
                        roll = np.arctan2(2*(pose[6]*pose[3]+pose[4]*pose[5]),
                                          1-2*(pose[3]**2+pose[4]**2))
                        tx = (cfg['fixture']['peel_kp_Nm_rad']*(float(command[4])-roll)
                              - cfg['fixture']['peel_kd_Nms_rad']*velocity[3])
                    else:
                        tx = float(command[5])
                fz = np.clip(fz, -25, 25)
                tx = np.clip(tx, -.05, .05)
                force[0, 2] = float(fz)
                torque[0, 0] = float(tx)
                if replay_inputs is not None and args.replay_mode == 'forces':
                    force[:] = torch.as_tensor(replay_inputs['forces'][step], device='cuda:0')
                    torque[:] = torch.as_tensor(replay_inputs['torques'][step], device='cuda:0')
                male.apply_forces_and_torques_at_position(force, torque, None, ids, True)
                interface.simulate(dt, (step + 1) * dt)
                interface.fetch_results()
                pose = male.get_transforms().cpu().numpy()[0].copy()
                vel = male.get_velocities().cpu().numpy()[0].copy()
                ballposes = balls.get_transforms().cpu().numpy().copy() if balls is not None else np.empty((0, 7))
                per_step['forces'].append(force.cpu().numpy()[0].copy())
                per_step['fixture_commands'].append(command)
                per_step['torques'].append(torque.cpu().numpy()[0].copy())
                per_step['male_poses'].append(pose)
                per_step['male_velocities'].append(vel)
                per_step['ball_poses'].append(ballposes)
                per_step['pad_targets'].append(pad_target)
                per_step['fixture_joint_positions'].append(pads.get_dof_positions().cpu().numpy()[0].copy() if grasp else np.empty(0))
                per_step['fixture_link_poses'].append(pads.get_link_transforms().cpu().numpy()[0].copy() if grasp else np.empty((0, 7)))
                if step % args.measure_every == 0:
                    record = {'step': step, 'time_s': t, 'phase': phase, 'robot_action': None,
                              'diagnostic_force_N': per_step['forces'][-1], 'diagnostic_torque_Nm': per_step['torques'][-1],
                              'male_pose_xyzw': pose, 'male_velocity': vel,
                              'ball_poses_xyzw': ballposes, 'full_task_score': 0.0, 'cloth': {}}
                    for name, view in cloth_views.items():
                        points = view.get_simulation_nodal_positions().cpu().numpy()[0].copy()
                        initial = data['cloth'][name]['initial']
                        points = points[:len(initial)]
                        cloth_frames[name].append(points)
                        faces = data['cloth'][name]['faces']
                        edge = np.vstack((faces[:, :2], faces[:, 1:], faces[:, [2, 0]]))
                        length = np.linalg.norm(points[edge[:, 0]] - points[edge[:, 1]], axis=1)
                        rest = np.linalg.norm(initial[edge[:, 0]] - initial[edge[:, 1]], axis=1)
                        local = data['cloth'][name]['local_rivet']
                        if name == 'upper':
                            u, w = pose[3:6], pose[6]
                            target = local + 2 * (w * np.cross(u, local) + np.cross(u, np.cross(u, local))) + pose[:3]
                        else:
                            target = local + data['socket_origin']
                        error = np.linalg.norm(points[data['cloth'][name]['rivet_ids']] - target, axis=1)
                        record['cloth'][name] = {'max_abs_edge_strain': float(np.max(np.abs(length / rest - 1))),
                                                 'max_rivet_error_m': float(error.max()),
                                                 'z_range_m': float(np.ptp(points[:, 2])), 'finite': bool(np.isfinite(points).all())}
                    records.append(jsonable(record))
                    trace_file.write(json.dumps(records[-1], allow_nan=False) + '\n')
                    if not np.isfinite(pose).all():
                        break
                if step % 960 == 0:
                    print(json.dumps({'episode': episode, 'step': step, 'phase': phase, 'cap_height_m': float(pose[2] - .1), 'force_N': float(per_step['forces'][-1][2])}), flush=True)
            trace_file.close()
            np.savez_compressed(dest / 'per-step.npz', **{k: np.array(v) for k, v in per_step.items()}, dt=dt)
            for name, frames in cloth_frames.items():
                np.savez_compressed(dest / f'{name}-nodes.npz', positions=np.array(frames), initial=data['cloth'][name]['initial'], faces=data['cloth'][name]['faces'])
            result = analyze(records, cfg)
            if case == 'torque_free':
                result['free_torque_final_angle_rad'] = float(2 * np.arccos(np.clip(abs(pose[6]), 0, 1)))
                result['free_torque_final_omega_x_rad_s'] = float(vel[3])
            if grasp:
                from scoring import analyze_grasp
                result.update(analyze_grasp(records))
            if replay_inputs is not None:
                result['replay_mode'] = args.replay_mode
                result['replay_max_position_error_m'] = float(np.max(np.abs(np.array(per_step['male_poses'])[:, :3] - replay_inputs['male_poses'][:, :3])))
                if not grasp:
                    result['replay_max_ball_position_error_m'] = float(np.max(np.abs(np.array(per_step['ball_poses'])[:, :, :3] - replay_inputs['ball_poses'][:, :, :3])))
                result['replay_initial_usd_identical'] = (dest / 'initial.usda').read_bytes() == (args.replay / 'initial.usda').read_bytes()
                result['replay_source_bundle_sha256'] = json.loads((args.replay / 'result.json').read_text())['bundle_sha256']
                result['replay_position_tolerance_m'] = .001
                result['replay_rigid_positions_match'] = result['replay_max_position_error_m'] <= .001
                source_pose = replay_inputs['male_poses']
                replay_pose = np.array(per_step['male_poses'])
                dots = np.sum(replay_pose[:, 3:] * source_pose[:, 3:], axis=1)
                angles = 2*np.arccos(np.clip(np.abs(dots), 0, 1))
                result['replay_max_angle_error_rad'] = float(np.max(angles))
                result['replay_angle_tolerance_rad'] = float(np.deg2rad(5.))
                result['replay_angles_match'] = result['replay_max_angle_error_rad'] <= result['replay_angle_tolerance_rad']
                cloth_errors = {}
                for name, frames in cloth_frames.items():
                    reference = np.load(args.replay / f'{name}-nodes.npz')['positions']
                    measured = np.array(frames)
                    if reference.shape == measured.shape:
                        cloth_errors[name] = float(np.max(np.abs(measured-reference)))
                result['replay_cloth_max_coordinate_error_m'] = cloth_errors
                result['replay_cloth_position_tolerance_m'] = .001
                result['replay_cloth_positions_match'] = (
                    len(cloth_errors) == len(cloth_frames) and all(error <= .001 for error in cloth_errors.values()))
            result.update(case=case, seed=args.seed + episode, bundle_sha256=digest,
                          cloth_measure_every_physics_steps=args.measure_every,
                          runtime_seconds=time.monotonic() - start, cloth=not args.no_cloth,
                          termination_reason='diagnostic_step_limit', stage='isolated_fixture_not_robot', full_task_success=False)
            write(dest / 'result.json', result)
            results.append(result)
            print('RESULT ' + json.dumps(result), flush=True)
            sim.invalidate()
            interface.detach_stage()
        write(out / 'results.json', results)
    except BaseException as exc:
        import traceback
        traceback.print_exc()
        write(out / 'failure.json', {'error': repr(exc), 'traceback': traceback.format_exc(), 'full_task_success': False})
        raise
    finally:
        app.close(skip_cleanup=False)



if __name__ == '__main__':
    main()
