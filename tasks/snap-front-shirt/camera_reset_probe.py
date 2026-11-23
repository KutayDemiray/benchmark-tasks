"""Diagnose native reset-camera freshness without changing physics or robot state."""

import argparse
import json
from pathlib import Path
import traceback

import numpy as np

from bench import snapshot, write
from evaluation_runner import selected_configuration
from gpu_runtime import inherited_fds, scheduled_assignment
from network_guard import deny_internet_sockets
from physics_provenance import verify_physical_sources
from policy_contract import CAMERAS
from robot_runner import start_app, make_environment
from tabletop_scene import configure, build, camera_rig
from task_paths import OUTPUT_ROOT

HERE = Path(__file__).resolve().parent


def array(value):
    """Copy a native tensor or proxy to CPU for the diagnostic record."""
    if hasattr(value, 'torch'):
        value = value.torch
    return value.cpu().numpy().copy()


def physical_state(episode):
    """Read authoritative positions, velocities and clocks without simulation."""
    result = dict(cloth=episode.measure.nodes(),
        cloth_velocity=array(episode.measure.cloth.get_simulation_nodal_velocities()),
        hardware=array(episode.measure.bodies.get_transforms()),
        hardware_velocity=array(episode.measure.bodies.get_velocities()),
        measured_joints=episode.joint_positions(),
        physics_index=np.array(episode.measure.index),
        simulator_index=np.array(episode.base.sim.get_physics_step_count()))
    for side in ('left', 'right'):
        robot = getattr(episode, side)
        result[side + '_joints'] = array(robot.data.joint_pos)
        result[side + '_joint_velocities'] = array(robot.data.joint_vel)
    return result


def capture(episode, output, label):
    """Save actual RGB, camera poses and robot link poses for visual inspection."""
    from PIL import Image
    images = episode.images()
    poses = {}
    for name in CAMERAS:
        Image.fromarray(images[name]).save(output / (label + '-' + name + '.png'))
        camera = episode.base.scene[name]
        poses[name] = dict(position=array(camera.data.pos_w),
                          quaternion_world_xyzw=array(camera.data.quat_w_world),
                          frame=array(camera.frame))
    for side in ('left', 'right'):
        robot = getattr(episode, side)
        poses[side + '_robot'] = dict(names=list(robot.body_names),
                                    poses=array(robot.data.body_link_pose_w))
    write(output / (label + '-poses.json'), poses)
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stability-run', type=Path, required=True)
    parser.add_argument('--resync-fabric', action='store_true', help='Probe the installed render-Fabric resynchronization routine')
    args = parser.parse_args()
    assert args.out.resolve().is_relative_to(OUTPUT_ROOT)
    assert args.stability_run.resolve().is_relative_to(OUTPUT_ROOT)
    assert scheduled_assignment() is not None
    inherited_fds()
    deny_internet_sockets()
    gate = json.loads((args.stability_run / 'results.json').read_text())
    selected = selected_configuration(gate)
    proof = verify_physical_sources(HERE, args.stability_run / 'task-bundle',
        json.loads((args.stability_run / 'manifest.json').read_text()))
    write(args.out / 'physical-source-verification.json', proof)
    bundle = snapshot(args.out)
    cfg = json.loads((HERE / 'parameters.json').read_text())
    configure(cfg)
    cfg['revision'] = 'p12-tabletop-evaluation'
    cfg['completion']['control_hz'] = 20
    cfg['simulation']['physics_hz'] = 960
    cfg['fabric']['contact_offset_m'] = .001
    cfg['fabric']['solver_iterations'] = selected['iterations']
    cfg['garment']['body_mesh_pitch_m'] = selected['pitch']
    cfg['episode_seed'] = 100
    cfg['tabletop_test'] = dict(case='fasten_three', contact_substeps_seconds=0., right_camera_observer=False)
    args.seed = 100
    args.record = True
    args.strain_witness = True
    args.controller = 'none'
    args.seconds = 1.
    app = env = episode = None
    try:
        app = start_app(args.out, True)
        import omni.usd
        from tabletop_backend import TabletopEpisode
        omni.usd.get_context().new_stage()
        meta = {}
        env = make_environment(cfg, True, meta, args.out, scene_builder=build, rig=camera_rig())
        destination = args.out / 'episode-00'
        destination.mkdir()
        episode = TabletopEpisode(env, cfg, args, meta, destination, bundle)
        before = physical_state(episode)
        np.savez_compressed(args.out / 'before-refresh.npz', **before)
        captures = {'original': capture(episode, args.out, 'original')}
        for name in CAMERAS:
            episode.base.scene[name].update(0., force_recompute=True)
        captures['sensor_only'] = capture(episode, args.out, 'sensor_only')
        for index in range(4):
            episode.base.sim.render()
            for name in CAMERAS:
                episode.base.scene[name].update(0., force_recompute=True)
            label = 'render_refresh_' + str(index + 1)
            captures[label] = capture(episode, args.out, label)
        if args.resync_fabric:
            manager = episode.base.sim.physics_manager
            write(args.out / 'render-backend.json', dict(use_fabric=episode.base.sim.cfg.use_fabric,
                fabric_loaded=manager._fabric is not None,
                routine='PhysxManager._sync_fabric_after_resume; render-Fabric only'))
            manager._sync_fabric_after_resume()
            for index in range(4):
                episode.base.sim.render()
                for name in CAMERAS:
                    episode.base.scene[name].update(0., force_recompute=True)
                label = 'fabric_resync_' + str(index + 1)
                captures[label] = capture(episode, args.out, label)
        after = physical_state(episode)
        np.savez_compressed(args.out / 'after-refresh.npz', **after)
        unchanged = {name: bool(np.array_equal(before[name], after[name])) for name in before}
        write(args.out / 'refresh-state-check.json', dict(unchanged=unchanged, passed=all(unchanged.values()),
            scope='Rendering and sensor refresh only; no state writes, action or physics step'))
        assert all(unchanged.values()), 'Camera refresh changed measured physics'
        held = episode.joint_positions()
        episode.step(held)
        captures['first_hold'] = capture(episode, args.out, 'first_hold')
        differences = {}
        for label, images in captures.items():
            differences[label] = {name: float(np.abs(images[name].astype(float) - captures['first_hold'][name]).mean())
                                  for name in CAMERAS}
        write(args.out / 'image-differences.json', dict(mean_absolute_pixel_difference_to_first_hold=differences,
            scope='Diagnostic, not a task scoring threshold'))
        for _ in range(19):
            if episode.reason is not None:
                break
            episode.step(held)
        result = episode.finish('camera_diagnostic_complete')
        write(args.out / 'results.json', [result])
        write(args.out / 'camera-diagnostic.json', dict(physics_unchanged_during_refresh=True,
            saved_stages=list(captures), visual_review_pending=True, task_success_claim=False,
            original_state_sha_scope='before-refresh.npz and after-refresh.npz contain every checked array'))
    except BaseException as exc:
        write(args.out / 'failure.json', dict(error=repr(exc), traceback=traceback.format_exc()))
        raise
    finally:
        if episode is not None and not episode.closed:
            episode.finish('camera_diagnostic_failure')
        if env is not None:
            env.close()
        if app is not None:
            app.close()


if __name__ == '__main__':
    main()
