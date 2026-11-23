"""Native Arena/YAM isolated control; no fixture wrenches or automatic grasps."""

import argparse
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import numpy as np

from bench import snapshot, write, jsonable

HERE = Path(__file__).resolve().parent


def start_app(out, record):
    """Start the native application with owned caches and explicit GPU selection."""
    runtime = out/'runtime'
    privacy = runtime/'privacy.toml'
    privacy.write_text('[privacy]\nperformance=false\nusage=false\npersonalization=false\n')
    sys.argv = [sys.argv[0], '--portable-root', str(out.parent/'kit-portable')]
    from network_guard import deny_internet_sockets
    deny_internet_sockets()
    from isaaclab.app import AppLauncher
    from gpu_runtime import kit_arguments
    kit = [
        kit_arguments(out/'kit.log'),
        '--/app/settings/persistent=false', '--/telemetry/enableAnonymousData=false',
        '--/telemetry/enableSentry=false', '--/telemetry/enableNVDF=false',
        '--/telemetry/useOpenEndpoint=false', '--/exts/omni.kit.telemetry/skipDeferredStartup=true',
        f'--/log/file={out}/kit.log', f'--/structuredLog/privacySettingsFile={privacy}',
        f'--/structuredLog/logDirectory={runtime}/structured', f'--/telemetry/log/file={runtime}/telemetry.log',
        f'--/app/userConfigPath={runtime}/kit.config.json', f'--/app/tokens/data={runtime}/kit-data',
        f'--/app/tokens/cache={out.parent}/runtime-cache/kit', f'--/app/tokens/logs={runtime}/kit-logs',
    ]
    return AppLauncher(headless=True, enable_cameras=record, device='cuda:0', multi_gpu=False,
                       kit_args=' '.join(kit)).app


def make_environment(cfg, record, metadata, out, scene_builder=None, rig=None):
    """Build stock YAM with an owned procedural cloth asset."""
    import omni.usd
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab_arena.assets.object import Object
    from isaaclab_arena.assets.object_type import ObjectType
    from isaaclab_arena.assets.object_library import DomeLight
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
    from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
    from yam_arena.embodiment import BimanualYamEmbodiment
    from yam_arena.scene import env_cfg_callback
    from robot_geometry import grasp_scene, press_scene

    @clone
    def spawn(prim_path, _cfg, translation=None, orientation=None, **kwargs):
        stage = omni.usd.get_context().get_stage()
        builder = scene_builder or (press_scene if cfg['robot_test'].get('case') in ('press', 'no_press', 'press_lift') else grasp_scene)
        metadata.update(builder(stage, prim_path, cfg))
        return stage.GetPrimAtPath(prim_path)

    def configure(env_cfg):
        from episode_timing import control_hz,physics_stride
        env_cfg = env_cfg_callback(control_hz=control_hz(cfg))(env_cfg)
        env_cfg.seed = cfg.get('episode_seed', 100)
        env_cfg.log_dir = str(out/'runtime/arena')
        env_cfg.sim.log_dir = str(out/'runtime/isaaclab')
        env_cfg.recorders.dataset_export_dir_path = str(out/'runtime/recorders')
        env_cfg.sim.dt = 1/cfg['simulation']['physics_hz']
        env_cfg.decimation = physics_stride(cfg)
        env_cfg.sim.render_interval = env_cfg.decimation
        return env_cfg

    from dataclasses import replace
    from yam_arena.rig import TABLETOP
    if rig is None:
        rig = replace(TABLETOP, name='snap_coupon_bench', top_cam=replace(
            TABLETOP.top_cam, pos=(.30, 0., 1.65), rot_xyzw=(0., .7071067811865475, 0., .7071067811865476)))
    emb = BimanualYamEmbodiment(enable_cameras=record, contact_sensors=True, rig=rig)
    if scene_builder is not None and 'parked_robot_joints_rad' in cfg.get('garment', {}):
        from yam_arena.constants import ARM_JOINTS
        for side, joints in cfg['garment']['parked_robot_joints_rad'].items():
            getattr(emb.scene_config, side+'_robot').init_state.joint_pos = {
                **dict(zip(ARM_JOINTS, map(float, joints))), 'joint[7-8]': -.04695}
    if cfg['robot_test'].get('press_start') == 'hover':
        from yam_arena import ik
        from yam_arena.constants import ARM_JOINTS, LEFT_BASE_POS, READY
        from yam_arena.demo import DOWN_GRIP
        initial_tcp = [.30, .12, cfg['robot_test']['press_support_top_m']+.035]
        initial_q, error, angle = ik.solve(initial_tcp, DOWN_GRIP, READY,
                                          base_pos=LEFT_BASE_POS, iters=300)
        assert error<.0001 and angle<.001
        emb.scene_config.left_robot.init_state.joint_pos = {
            **dict(zip(ARM_JOINTS, map(float, initial_q))), 'joint[7-8]': 0.}
        cfg['robot_test'].update(initial_left_joint_positions_rad=initial_q.tolist(),
            initial_left_gripper_normalized=0., initial_left_tcp_target_m=initial_tcp,
            initial_pose_note='Isolated hover start, clear above the open snap; no object grasp or attachment.')

    if cfg['robot_test'].get('zero_friction', False):
        from isaaclab_physx.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
        for name in ('left_robot', 'right_robot'):
            spawn_cfg = getattr(emb.scene_config, name).spawn
            spawn_cfg.make_uninstanceable = True
            spawn_cfg.physics_material = RigidBodyMaterialCfg(
                static_friction=0., dynamic_friction=0., friction_combine_mode='min')

    asset = Object(name='snap_test', object_type=ObjectType.BASE, spawner_cfg=SpawnerCfg(func=spawn))
    asset.disable_reset_pose()
    scene = Scene([DomeLight(), asset])
    spec = IsaacLabArenaEnvironment(name='snap_front_single', scene=scene, embodiment=emb,
                                    env_cfg_callback=configure)
    return ArenaEnvBuilder(spec, ArenaEnvBuilderCfg(num_envs=1, solve_relations=False, device='cuda:0')).make_registered()


class Measurements:
    """Read native physics states without applying object forces or poses."""

    def __init__(self, base, meta, dest):
        import omni.physics.tensors as tensors
        self.sim = tensors.create_simulation_view('torch')
        self.sim.set_subspace_roots('/')
        assert self.sim.device_ordinal == 0
        self.meta = meta
        self.bodies = {k: self.sim.create_rigid_body_view(p) for k, p in meta['hardware'].items()}
        self.cloths = {k: self.sim.create_surface_deformable_body_view(d['path'])
                       for k, d in meta['cloth'].items()}
        # Batch readback preserves every physics-step observation while avoiding
        # a GPU synchronization for each tiny body. Check names and contents
        # against the separate views before using the indices.
        self.body_batch = self.sim.create_rigid_body_view(list(meta['hardware'].values()))
        self.body_indices = {k: list(self.body_batch.prim_paths).index(p)
                             for k, p in meta['hardware'].items()}
        batch = self.body_batch.get_transforms().cpu().numpy()
        assert self.body_batch.count == len(self.bodies)
        for name, view in self.bodies.items():
            assert np.array_equal(batch[self.body_indices[name]], view.get_transforms().cpu().numpy()[0])
        self.cloth_batch = self.sim.create_surface_deformable_body_view(
            [d['path'] for d in meta['cloth'].values()])
        self.cloth_indices = {k: list(self.cloth_batch.prim_paths).index(d['path'])
                              for k, d in meta['cloth'].items()}
        batch = self.cloth_batch.get_simulation_nodal_positions().cpu().numpy()
        assert self.cloth_batch.count == len(self.cloths)
        for name, view in self.cloths.items():
            count = len(meta['cloth'][name]['initial'])
            assert np.array_equal(batch[self.cloth_indices[name], :count],
                                  view.get_simulation_nodal_positions().cpu().numpy()[0, :count])
        write(dest/'measurement-views.json', {'body_indices': self.body_indices,
              'cloth_indices': self.cloth_indices, 'initial_matches_individual_views': True})
        env_root = meta['root'].rsplit('/', 1)[0]
        from yam_arena.robot import asset_link_paths
        link_paths = asset_link_paths()
        self.contact_links_per_arm = len(link_paths)
        filters = [env_root+'/'+arm+'/'+suffix.lstrip('/')
                   for arm in ('LeftArm', 'RightArm') for suffix in link_paths.values()]
        self.robot_contacts = self.sim.create_rigid_contact_view(
            [meta['hardware']['male']], filter_patterns=[filters])
        self.socket_contacts = self.sim.create_rigid_contact_view(
            [meta['hardware']['socket']], filter_patterns=[filters]) if 'socket' in meta['hardware'] else None
        assert self.robot_contacts.filter_count == len(filters)
        write(dest/'contact-sensors.json', {'male_sensor': meta['hardware']['male'], 'robot_link_filters': filters})
        self.dt = base.physics_dt
        self.max_robot_force = 0.
        self.log = (dest/'physics.jsonl').open('w', buffering=1)
        self.index = 0
        self.max_strain = self.max_rivet = 0.
        self.finite = True
        self.last = None
        self.nodes = {}
        self.original_step = base.sim.step
        def measured_step(*args, **kwargs):
            self.original_step(*args, **kwargs)
            self.observe()
        base.sim.step = measured_step

    def observe(self):
        from robot_geometry import rotation
        body_array = self.body_batch.get_transforms().cpu().numpy()
        cloth_array = self.cloth_batch.get_simulation_nodal_positions().cpu().numpy()
        poses = {k: body_array[i].copy() for k, i in self.body_indices.items()}
        link_force = self.robot_contacts.get_contact_force_matrix(self.dt).cpu().numpy()[0].copy()
        force = link_force.reshape(2, self.contact_links_per_arm, 3).sum(axis=1)
        self.max_robot_force = max(self.max_robot_force, float(np.linalg.norm(link_force, axis=-1).sum()))
        record = {'physics_step': self.index, 'hardware': poses, 'cloth': {},
                  'male_robot_normal_forces_N': force, 'male_robot_link_normal_forces_N': link_force}
        if self.socket_contacts is not None:
            socket_force = self.socket_contacts.get_contact_force_matrix(self.dt).cpu().numpy()[0]
            record['socket_robot_normal_forces_N'] = socket_force.reshape(2, self.contact_links_per_arm, 3).sum(axis=1)
        for name, view in self.cloths.items():
            d = self.meta['cloth'][name]
            nodes = cloth_array[self.cloth_indices[name], :len(d['initial'])].copy()
            self.nodes[name] = nodes
            lengths = np.linalg.norm(nodes[d['edges'][:, 0]]-nodes[d['edges'][:, 1]], axis=1)
            strain = float(np.max(np.abs(lengths/d['rest_lengths']-1)))
            key = next(k for k, v in self.meta['hardware'].items() if v == d['hardware_path'])
            pose = poses[key]
            wanted = d['local_rivet']@rotation(pose[3:]).T+pose[:3]
            error = float(np.linalg.norm(nodes[d['rivet_ids']]-wanted, axis=1).max())
            finite = bool(np.isfinite(nodes).all() and np.isfinite(pose).all())
            self.max_strain = max(self.max_strain, strain)
            self.max_rivet = max(self.max_rivet, error)
            self.finite &= finite
            record['cloth'][name] = dict(strain=strain, rivet_error_m=error, finite=finite)
        self.last = jsonable(record)
        self.log.write(json.dumps(self.last, allow_nan=False)+'\n')
        self.index += 1
        return self.last

    def close(self):
        self.log.close()
        self.sim.invalidate()


def run_episode(env, cfg, args, meta, dest, bundle):
    import carb
    import torch
    import omni.usd
    from yam_arena.constants import READY
    from yam_arena.runtime import arm_q, tcp_world
    from robot_control import CouponController, PressController, PressLiftController

    env.reset()
    base = env.unwrapped
    assert base.action_manager.total_action_dim == 14
    assert list(base.action_manager.active_terms) == ['left_arm', 'left_gripper', 'right_arm', 'right_gripper']
    assert abs(base.step_dt-.1) < 1e-9
    from gpu_runtime import verify_environment
    verify_environment(base, meta, dest, args.record)
    stage = omni.usd.get_context().get_stage()
    from pxr import UsdPhysics
    from robot_geometry import rotation
    attachments = []
    for prim in stage.Traverse():
        if not str(prim.GetPath()).startswith(meta['root']):
            continue
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            assert joint.GetBody0Rel().GetTargets() and joint.GetBody1Rel().GetTargets(), 'World-mounted test hardware'
        if prim.GetTypeName() == 'OmniPhysicsVtxXformAttachment':
            target = [str(p) for p in prim.GetRelationship('omniphysics:src1').GetTargets()]
            assert target and all(p in meta['hardware'].values() for p in target), 'Non-rivet attachment'
            attachments.append({'path': str(prim.GetPath()), 'target': target})
    write(dest/'attachment-audit.json', {'local_rivets': attachments, 'world_mounts': False,
                                        'robot_grasp_attachments': False})
    stage.GetRootLayer().Export(str(dest/'initial.usda'))
    write(dest/'setup.json', {'parameters': cfg, 'seed': args.seed, 'case': args.case,
                              'stage': 'single_snap', 'recorded_cameras': bool(args.record),
                              'control_hz': 10., 'action_dimension': 14, 'bundle_sha256': bundle})
    if args.case=='press_lift':
        from yam_arena.analysis import link_hulls
        meta['robot_collision_planes'] = {name: [h.planes for h in hulls] for name, hulls in link_hulls().items()}
    write(dest/'topology.json', meta)
    measure = Measurements(base, meta, dest)
    measure.observe()
    left = base.scene['left_robot']
    right = base.scene['right_robot']
    initial_state = {'joint_positions': base.observation_manager.compute()['policy']['joint_pos'][0].cpu().numpy()}
    for name, view in measure.bodies.items():
        initial_state['pose_'+name] = view.get_transforms().cpu().numpy().copy()
        initial_state['velocity_'+name] = view.get_velocities().cpu().numpy().copy()
    for name, view in measure.cloths.items():
        initial_state['cloth_'+name] = measure.nodes[name].copy()
        initial_state['cloth_velocity_'+name] = view.get_simulation_nodal_velocities().cpu().numpy().copy()
    initial_state['left_joint_velocities'] = left.data.joint_vel.torch.cpu().numpy().copy()
    initial_state['right_joint_velocities'] = right.data.joint_vel.torch.cpu().numpy().copy()
    np.savez_compressed(dest/'initial-measured-state.npz', **initial_state)
    controller_cls = PressLiftController if args.case=='press_lift' else PressController if args.case in ('press', 'no_press') else CouponController
    controller = controller_cls(arm_q(left), tcp_world(left), cfg, args.case)
    writers = {}
    if args.record:
        import imageio.v2 as imageio
        for name in ('top_cam', 'left_cam', 'right_cam'):
            writers[name] = imageio.get_writer(str(dest/(name+'.mp4')), fps=10,
                                               codec='libx264', pixelformat='yuv420p', macro_block_size=1)
    rows, node_frames = [], {k: [] for k in meta['cloth']}
    reason = 'diagnostic_time_limit'
    start = time.monotonic()
    try:
        with (dest/'trace.jsonl').open('w', buffering=1) as trace:
            for step in range(round(args.seconds*10)):
                t = step*.1
                if args.case == 'press_lift':
                    initial = meta['cloth']['upper']['initial']
                    mask = initial[:, 0] < meta['grasp_tab_cutoff_x_m']
                    if args.grasp_mode in ('side', 'side_long'):
                        mask = (initial[:, 0]<initial[:, 0].min()+.005) & (np.abs(initial[:, 1]-.12)<.0061)
                    front = measure.nodes['upper'][mask]
                    poses = measure.last['hardware']
                elif args.case in ('press', 'no_press'):
                    front, poses = None, measure.last['hardware']
                else:
                    initial = meta['cloth']['coupon']['initial']
                    front = measure.nodes['coupon'][initial[:, 0] < .303]
                    poses = measure.last['hardware']['male']
                action, control = controller.action(t, tcp_world(left), front, poses)
                env.step(torch.tensor(action[None], dtype=torch.float32, device=base.device))
                row = {'step': step, 'time_s': (step+1)*.1, 'robot_action': action,
                       'joint_positions': base.observation_manager.compute()['policy']['joint_pos'][0].cpu().numpy(),
                       'measured_tcp_m': tcp_world(left),
                       'left_link_poses_xyzw': left.data.body_link_pose_w.torch[0].cpu().numpy(),
                       'left_link_names': list(left.body_names),
                       'right_link_poses_xyzw': right.data.body_link_pose_w.torch[0].cpu().numpy(),
                       'right_link_names': list(right.body_names), 'control': control, 'physics': measure.last,
                       'full_task_score': 0., 'full_task_success': False}
                if args.case=='press_lift':
                    from robot_scoring import lower_clearance, upper_hardware_clearance, upper_cloth_clearance
                    row['lower_robot_clearance_m'] = lower_clearance(row, measure.nodes['lower'], meta)
                    row['upper_hardware_clearance_m'] = upper_hardware_clearance(row, meta)
                    if control['phase'] in ('open', 'retreat'):
                        row['upper_cloth_robot_clearance_m'] = upper_cloth_clearance(row, measure.nodes['upper'], meta)
                rows.append(jsonable(row))
                trace.write(json.dumps(rows[-1], allow_nan=False)+'\n')
                for name in node_frames:
                    node_frames[name].append(measure.nodes[name].copy())
                for name, writer in writers.items():
                    pixels = base.scene[name].data.output['rgb']
                    pixels = pixels.torch if hasattr(pixels, 'torch') else pixels
                    writer.append_data(pixels[0, ..., :3].cpu().numpy().astype(np.uint8))
                if step % 10 == 0:
                    print(json.dumps({'step': step, 'phase': control['phase'],
                                      'cap_z': measure.last['hardware']['male'][2],
                                      'max_strain': measure.max_strain, 'ik_error_m': control['ik_error_m']}), flush=True)
                if measure.max_robot_force > cfg['gate']['max_press_N']:
                    reason = 'robot_press_force_limit_exceeded'
                    break
                if not measure.finite or measure.max_strain > cfg['gate']['max_edge_strain'] or measure.max_rivet > cfg['gate']['max_attachment_error_m']:
                    reason = 'invalid_fabric_state'
                    break
                if control.get('abort', False):
                    reason = control.get('abort_reason', 'controller_aborted')
                    break
                if control.get('completed', False):
                    reason = 'diagnostic_completed'
                    break
    finally:
        for writer in writers.values():
            writer.close()
        measure.close()
    from robot_scoring import analyze_grasp, analyze_press, analyze_press_lift
    with (dest/'physics.jsonl').open() as stream:
        physical_records = [json.loads(line) for line in stream]
    scorer = analyze_press_lift if args.case=='press_lift' else analyze_press if args.case in ('press', 'no_press') else analyze_grasp
    diagnostic_score = scorer(rows, physical_records, cfg, meta)
    np.savez_compressed(dest/'control-steps.npz',
                        actions=np.array([r['robot_action'] for r in rows]),
                        **{('cloth_nodes' if name=='coupon' else 'cloth_'+name): np.array(frames) for name, frames in node_frames.items()})
    z0 = meta['initial_hardware']['male'][2]
    held = [r['physics']['hardware']['male'][2] for r in rows if r['control']['phase']=='hold']
    released = [r['physics']['hardware']['male'][2] for r in rows if r['time_s']>11.5]
    valid = measure.finite and measure.max_strain <= cfg['gate']['max_edge_strain'] and measure.max_rivet <= cfg['gate']['max_attachment_error_m']
    result = {'stage': 'single_snap_robot_press_lift_diagnostic' if args.case=='press_lift' else 'single_snap_robot_press_diagnostic' if args.case in ('press', 'no_press') else 'single_snap_robot_grasp_diagnostic', 'case': args.case, 'seed': args.seed,
              'bundle_sha256': bundle, 'termination_reason': reason, 'runtime_seconds': time.monotonic()-start,
              'physics_steps_observed': measure.index, 'control_steps': len(rows),
              'cloth_integrity': valid, 'max_cloth_strain': measure.max_strain,
              'max_rivet_error_m': measure.max_rivet, 'peak_robot_normal_contact_N': measure.max_robot_force,
              'lifted_15mm': bool(valid and held and min(held)>z0+.015),
              'released_after_opening': bool(valid and released and max(released)<z0+.015),
              'peak_ik_position_error_m': controller.max_ik_error, 'peak_ik_angle_rad': controller.max_ik_angle,
              'full_task_score': 0., 'full_task_success': False, 'isolated_gate_pass': False,
              'note': 'Actual stock YAM actions; isolated diagnostic only. No complete snap cycle or garment task.'}
    if args.case in ('press', 'no_press', 'press_lift'):
        result.pop('lifted_15mm')
        result.pop('released_after_opening')
    result.update(diagnostic_score)
    write(dest/'result.json', result)
    print('RESULT '+json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--case', choices=['reach', 'grasp', 'frictionless', 'press', 'no_press', 'press_lift'], default='grasp')
    parser.add_argument('--seconds', type=float, help='Bounded simulation time; controller may finish earlier')
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--seed', type=int, default=100)
    parser.add_argument('--record', action='store_true')
    parser.add_argument('--grasp-mode', choices=('pinch', 'bunch', 'side', 'side_long'), default='pinch')
    parser.add_argument('--support-thickness', type=float, default=.04, help='Physical support plate thickness in metres')
    parser.add_argument('--support-width', type=float, default=.35, help='Physical support width along Y, metres')
    parser.add_argument('--press-start', choices=('ready', 'hover'), default='ready')
    parser.add_argument('--upper-width', type=float, default=.04, help='Physical upper fabric width along X, metres')
    args = parser.parse_args()
    if args.press_start == 'hover':
        assert args.case in ('press', 'no_press', 'press_lift')
    assert .06 <= args.support_width <= .35
    assert .04 <= args.upper_width <= .12
    if args.grasp_mode == 'side_long':
        assert args.upper_width >= .10 and args.support_thickness <= .006
    assert .004 <= args.support_thickness <= .04
    if args.seconds is None:
        args.seconds = {'press': 22., 'no_press': 4., 'press_lift': 40.}.get(args.case, 17.)
    assert 0 < args.seconds <= 300 and args.episodes >= 1
    from gpu_runtime import inherited_fds
    inherited_fds()
    cfg = json.loads((HERE/'parameters.json').read_text())
    cfg['robot_test']['case'] = args.case
    cfg['robot_test']['grasp_mode'] = args.grasp_mode
    cfg['robot_test']['support_thickness_m'] = args.support_thickness
    cfg['robot_test']['support_width_m'] = args.support_width
    cfg['robot_test']['upper_width_m'] = args.upper_width
    cfg['robot_test']['press_start'] = args.press_start
    if args.case == 'frictionless':
        cfg['fabric']['friction'] = 0.
        cfg['robot_test']['zero_friction'] = True
    bundle = snapshot(args.out)
    app = start_app(args.out, args.record)
    env = None
    try:
        results = []
        for index in range(args.episodes):
            import omni.usd
            omni.usd.get_context().new_stage()
            metadata = {}
            dest = args.out/f'episode-{index:02d}'
            dest.mkdir()
            cfg['episode_seed'] = args.seed
            env = make_environment(cfg, args.record, metadata, args.out)
            results.append(run_episode(env, cfg, args, metadata, dest, bundle))
            env.close()
            env = None
        write(args.out/'results.json', results)
    except BaseException as exc:
        write(args.out/'failure.json', {'error': repr(exc), 'traceback': traceback.format_exc(),
                                       'full_task_success': False})
        traceback.print_exc()
        raise
    finally:
        if env is not None:
            env.close()
        app.close()


if __name__ == '__main__':
    main()
