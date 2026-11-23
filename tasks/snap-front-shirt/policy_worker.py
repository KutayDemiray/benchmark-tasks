"""Native worker accepting local observation/action requests over inherited IPC."""

import argparse
import json
from multiprocessing.connection import Connection
import os
from pathlib import Path
from task_paths import OUTPUT_ROOT
import traceback

from bench import snapshot,write
from fresh_workers import inherited_lock
from policy_contract import INSTRUCTION
from robot_runner import start_app,make_environment
from tabletop_scene import configure,build,camera_rig

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--ipc-fd',type=int,required=True)
    parser.add_argument('--seed',type=int,default=100)
    parser.add_argument('--control-hz',type=int,choices=(10,20),default=20)
    parser.add_argument('--seconds',type=float,default=300.)
    parser.add_argument('--cloth-solver-iterations',type=int,choices=(64,128),default=64)
    parser.add_argument('--body-mesh-pitch',type=float,choices=(.01,.005),default=.01)
    parser.add_argument('--controller',choices=('none','baseline','bimanual_1','bimanual_2','bimanual_3'),default='none')
    parser.add_argument('--instruction',default=INSTRUCTION)
    parser.add_argument('--pilot-diagnostics',action='store_true')
    args=parser.parse_args()
    assert 0<args.seconds<=300
    inherited_lock()
    assert args.out.resolve().is_relative_to(OUTPUT_ROOT)
    args.record=True;args.strain_witness=True
    conn=Connection(args.ipc_fd)
    cfg=json.loads((HERE/'parameters.json').read_text());configure(cfg)
    cfg['revision']='p12-tabletop-evaluation'
    cfg['completion']['control_hz']=args.control_hz
    cfg['simulation']['physics_hz']=960
    cfg['fabric']['contact_offset_m']=.001
    cfg['fabric']['solver_iterations']=args.cloth_solver_iterations
    cfg['garment']['body_mesh_pitch_m']=args.body_mesh_pitch
    cfg['episode_seed']=args.seed
    cfg['tabletop_test']=dict(case='fasten_three',contact_substeps_seconds=0.,
                             right_camera_observer=args.controller=='baseline')
    bundle=snapshot(args.out)
    app=env=episode=None
    try:
        app=start_app(args.out,True)
        import omni.usd
        from tabletop_backend import TabletopEpisode
        omni.usd.get_context().new_stage();meta={}
        env=make_environment(cfg,True,meta,args.out,scene_builder=build,rig=camera_rig())
        destination=args.out/'episode-00';destination.mkdir()
        episode=TabletopEpisode(env,cfg,args,meta,destination,bundle,instruction=args.instruction)
        controller=None;target=control=None
        if args.controller=='baseline':
            from tabletop_control import TablePressController
            from yam_arena.runtime import arm_q,tcp_world
            controller=TablePressController(arm_q(episode.left),tcp_world(episode.left),cfg,
                                           targets=(0,1,2),observer=True)
        elif args.controller.startswith('bimanual_'):
            from bimanual_control import BimanualPressController
            from yam_arena.runtime import arm_q,tcp_world
            controller=BimanualPressController(arm_q(episode.left),tcp_world(episode.left),
                arm_q(episode.right),tcp_world(episode.right),cfg,revision=int(args.controller[-1]))
        conn.send(dict(ok=True,pid=os.getpid(),observation=episode.observe(),native_physics_index=episode.measure.index,bundle_sha256=bundle))
        while True:
            request=conn.recv()
            before=episode.measure.index
            op=None
            try:
                op=request['op']
                if op=='observe':
                    response=dict(ok=True,observation=episode.observe(),native_physics_index=episode.measure.index)
                elif op in ('step','script_step'):
                    if op=='script_step':
                        assert controller is not None, 'Scripted assistance is disabled in policy mode'
                        if len(episode.rows)%(args.control_hz//10)==0:
                            if args.controller.startswith('bimanual_'):
                                target,control=controller.action(episode.time_s,episode.tcp_world(episode.left),
                                    episode.tcp_world(episode.right),episode.measure.last)
                            else:
                                target,control=controller.action(episode.time_s,episode.tcp_world(episode.left),episode.measure.last)
                        result=episode.step(target,control)
                    else:
                        assert controller is None, 'External policy actions require controller=none'
                        result=episode.step(request['action'])
                    terminal=request.get('terminal')
                    if terminal is not None and episode.reason is None:
                        episode.reason=terminal['kind']
                    elif (controller is not None and control['completed'] and episode.reason is None
                          and len(episode.rows)%(args.control_hz//10)==0):
                        episode.reason='controller_aborted' if control['abort_reason'] else 'controller_completed'
                    if episode.reason is not None:
                        from dataclasses import replace
                        result=replace(result,terminated=episode.reason!='time_limit',
                            truncated=episode.reason=='time_limit',termination_reason=episode.reason)
                    response=dict(ok=True,result=result)
                elif op=='finish':
                    result=episode.finish(request.get('reason','closed'))
                    write(args.out/'results.json',[result])
                    conn.send(dict(ok=True,result=result))
                    break
                else:
                    raise ValueError('Unknown native worker operation')
                conn.send(response)
            except (AssertionError,ValueError,KeyError,TypeError) as exc:
                if episode.measure.index!=before:
                    raise  # An unrecorded partial physics step is a failed episode.
                conn.send(dict(ok=False,error=repr(exc),physics_advanced=False))
    except BaseException as exc:
        write(args.out/'failure.json',dict(error=repr(exc),traceback=traceback.format_exc(),full_task_success=False))
        traceback.print_exc()
        if episode is not None and not episode.closed:
            episode.meta['native_execution_error']=True
            write(episode.destination/'topology.json',episode.meta)
        try:conn.send(dict(ok=False,error=repr(exc)))
        except (OSError,EOFError):pass
        raise
    finally:
        if episode is not None and not episode.closed:
            try:episode.finish('worker_failure')
            except BaseException:traceback.print_exc()
        if env is not None:env.close()
        if app is not None:app.close()
        conn.close()


if __name__=='__main__':main()
