"""Native reset checks and ordinary-action replay in a fresh simulation."""

import hashlib
import json

import numpy as np

from robot_gate import array


def snapshot(base,frame,cover):
    """Read every articulated state and, when present, every cloth node."""
    values={}
    for name,rig in [('frame',frame),('left',base.scene['left_robot']),('right',base.scene['right_robot'])]:
        values[name+'_poses']=array(rig.root_view.get_link_transforms())
        values[name+'_velocities']=array(rig.root_view.get_link_velocities())
        values[name+'_joints']=array(rig.root_view.get_dof_positions())
        values[name+'_joint_velocities']=array(rig.root_view.get_dof_velocities())
    if cover is not None:
        values['cloth_positions'],values['cloth_velocities']=cover.nodes()
    return values


def verify_resets(env,frame,cover,request,out):
    """Perturb with ordinary actions, then check two resets in the same process."""
    import torch
    from yam_arena.constants import READY
    base=env.unwrapped
    initial=snapshot(base,frame,cover)
    records=[]
    for repeat in range(2):
        action=np.r_[READY,.8,READY,.8]
        action[1]+=.02;action[8]-=.02
        for _ in range(5):
            env.step(torch.tensor(action[None],device=base.device,dtype=torch.float32))
        changed=snapshot(base,frame,cover)
        env.reset(seed=request['seed'])
        if cover is not None:cover.bind(base)
        reset=snapshot(base,frame,cover)
        errors={name:float(abs(reset[name]-value).max()) for name,value in initial.items()}
        perturbation=max(float(abs(changed[name]-value).max()) for name,value in initial.items())
        records.append(dict(repeat=repeat,perturbation_observed=perturbation>1e-4,reset_max_abs_errors=errors))
    passed=all(row['perturbation_observed'] and max(row['reset_max_abs_errors'].values())<2e-6 for row in records)
    (out/'reset-verification.json').write_text(json.dumps(dict(passed=passed,cycles=records,
        scope='Native nodes, link poses, joint states and all velocities; each history recorder starts after the final reset.',
        diagnostic_state_tolerance=2e-6),indent=2))
    np.savez_compressed(out/'reset-initial-state.npz',**initial)
    assert passed, 'Complete native reset did not reconstruct the starting state'


def replay(session,source,out):
    """Execute recorded 14D actions from reset without writing object states."""
    assert json.loads((source/'result.json').read_text())['status']=='completed'
    request=json.loads((source/'request.json').read_text())
    assert request.get('completion_mode', 'released') == session.request.get('completion_mode', 'released'), 'Replay completion mode differs'
    assert request.get('cover_offset_x',0.)==session.request.get('cover_offset_x',0.), 'Replay cover layout differs'
    for key in ('physics_hz','iterations','spacing','contact_updates','contact_subiterations','ccd',
                'product_profile','table_origin','snap_throat','robot_cover','frame_contact_correction_speed','cloth_contact_correction_speed',
                'camera_width','camera_height','full_sdk_source'):
        assert request.get(key)==session.request.get(key), f'Replay configuration differs: {key}'
    for name in ('versions.json','frame-geometry.json','scene-audit.json'):
        assert json.loads((source/name).read_text())==json.loads((out/name).read_text()),name
    if request.get('robot_cover'):
        with np.load(source/'geometry.npz') as original,np.load(out/'geometry.npz') as current:
            assert set(original.files)==set(current.files)
            assert all(np.array_equal(original[key],current[key]) for key in original.files)
        assert json.loads((source/'effective-parameters.json').read_text())==json.loads((out/'effective-parameters.json').read_text())
    original=json.loads((source/'manifest.json').read_text())['files']
    current=json.loads((out/'manifest.json').read_text())['files']
    for name in ('frame_native.py','frame_asset.py','native.py','mesh.py','cover_mesh.py','station_coordinates.py',
                 'full_observation.py','full_sdk.py','sdk_session.py','sdk_yam.py','sdk_cameras.py','sdk_cooked_geometry.py'):
        assert original[name]==current[name], f'Physics authoring changed: {name}'
    if request.get('frame_contact_correction_speed') is not None:
        assert original['frame_solver.py']==current['frame_solver.py'], 'Frame solver authoring changed'
        assert json.loads((source/'frame-numerics.json').read_text())==json.loads((out/'frame-numerics.json').read_text()), 'Frame solver readback differs'
    if request.get('cloth_contact_correction_speed') is not None:
        assert original['cloth_solver.py']==current['cloth_solver.py'], 'Cloth solver authoring changed'
        assert json.loads((source/'cloth-numerics.json').read_text())==json.loads((out/'cloth-numerics.json').read_text()), 'Cloth solver readback differs'
    actions=source/'steps.jsonl'
    rows=[json.loads(line) for line in actions.read_text().splitlines()]
    assert len(rows)<=round(session.request['seconds']*20)
    for row in rows:session.step(np.asarray(row['robot_action']),row['phase'])
    (out/'replay-provenance.json').write_text(json.dumps(dict(source_run=source.name,
        actions_sha256=hashlib.sha256(actions.read_bytes()).hexdigest(),control_steps=len(rows),
        fresh_simulation=True,from_reset=True,object_state_writes_during_replay=False,
        outcome='Independent audits and scoring are still required; matching commands do not imply success.'),indent=2))
