"""Gate an installed-SDK full scene behind native material, mechanism and stock-grasp proofs."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import traceback
from types import SimpleNamespace

import numpy as np

from robot_gate import array


def read(path):
    return json.loads(path.read_text())


def validate_source(request, output):
    """Require a declared-reset scene and matching, separately qualified SDK physics."""
    name=request.get('full_sdk_source')
    assert name and Path(name).name==name and name not in ('.','..')
    assert request['experiment']=='frame_robot' and request['robot_cover'] and request['robot_unlock'] and request['robot_reopen']
    assert request.get('completion_mode','released')=='released' and request['reset_check']
    assert not any(request.get(key) for key in ('release_probe_run','frame_press','upright_handle',
        'frame_contact_correction_speed','cloth_contact_correction_speed','reinforced'))
    assert request['product_gate']=='product-sdk-gate.json' and request['mechanism_gate']=='mechanism-sdk-gate.json'
    for filename in ('product-sdk-gate.json','mechanism-sdk-gate.json'):
        gate=read(output/filename)
        assert gate['passed'] and gate['physics_backend']=='installed_ovphysx', 'SDK qualification is incomplete'
        for path,digest in gate['evidence_sha256'].items():
            assert hashlib.sha256((output/path).read_bytes()).hexdigest()==digest, 'Stale SDK gate: '+path
    source=output/name
    original=read(source/'request.json')
    assert read(source/'result.json')['status']=='completed' and read(source/'process.json')['returncode']==0
    assert not original.get('release_probe_run') and not original.get('full_sdk_source')
    assert original['robot_cover'] and read(source/'reset-verification.json')['passed']
    assert not read(source/'scene-audit.json')['diagnostic_initialization']
    for key in ('physics_hz','iterations','spacing','seconds','contact_updates','contact_subiterations','ccd',
                'product_profile','mouth_height','table_origin','snap_throat','cover_offset_x','frame_lift',
                'camera_width','camera_height','seed'):
        assert request[key]==original[key], 'Declared source configuration differs: '+key
    for file in ('native.py','mesh.py','cover_mesh.py','full_cover.py','frame_native.py','frame_asset.py',
                 'parameters.json','station_coordinates.py'):
        assert (source/'source'/file).read_bytes()==Path(__file__).with_name(file).read_bytes(), 'Physical authoring changed: '+file
    return source


class NativeView:
    """Expose read-only native values in the recorder's canonical body/joint order."""

    TYPES={
        'get_link_transforms':('ARTICULATION_LINK_POSE','body'),
        'get_link_velocities':('ARTICULATION_LINK_VELOCITY','body'),
        'get_dof_positions':('ARTICULATION_DOF_POSITION','joint'),
        'get_dof_velocities':('ARTICULATION_DOF_VELOCITY','joint'),
        'get_masses':('ARTICULATION_BODY_MASS','body'),
        'get_coms':('ARTICULATION_BODY_COM_POSE','body'),
        'get_inertias':('ARTICULATION_BODY_INERTIA','body'),
        'get_dof_stiffnesses':('ARTICULATION_DOF_STIFFNESS','joint'),
        'get_dof_dampings':('ARTICULATION_DOF_DAMPING','joint'),
    }

    def __init__(self,sim,path,body_names,joint_names):
        self.sim,self.path=sim,path
        self.bindings={}
        for kind,_ in self.TYPES.values():
            if kind not in self.bindings:self.bindings[kind]=sim.bind(kind,path)
        poses=self.bindings['ARTICULATION_LINK_POSE'];q=self.bindings['ARTICULATION_DOF_POSITION']
        assert len(poses.body_names)==len(body_names) and len(q.dof_names)==len(joint_names)
        assert set(poses.body_names)==set(body_names) and set(q.dof_names)==set(joint_names)
        self.body_order=[poses.body_names.index(name) for name in body_names]
        self.joint_order=[q.dof_names.index(name) for name in joint_names]

    def __getattr__(self,name):
        if name not in self.TYPES:raise AttributeError(name)
        kind,domain=self.TYPES[name]
        order=self.body_order if domain=='body' else self.joint_order
        return lambda:self.sim.read(self.bindings[kind])[:,order]


class SDKEnvironment:
    """Implement reset/step/sensor boundaries for the shared RobotSession recorder."""

    def __init__(self,request,out,source):
        from sdk_session import SDKSession,author_defaults,surface_bindings
        from sdk_yam import SDKYam
        from full_cover import FullCover
        from mesh import surface_edges
        from capture_yam_contract import compare
        from sdk_cameras import SDKCameras
        from standalone_grasp import verify_station_copy
        self.request,self.out,self.source=request,out,source
        self.unwrapped=self;self.device='cuda:0';self.recording_started=False
        self.sdk=self.cameras=None
        for name in ('scene.usda','scene-audit.json','geometry.npz','mesh.json','cover-mesh.json',
                     'effective-parameters.json','frame-geometry.json','frame-names.json','frame-dynamics.json',
                     'gripper-colliders.json','robot-colliders.json','robot-cooked-colliders.json'):
            shutil.copyfile(source/name,out/name)
        self.metadata=read(out/'frame-geometry.json');self.names=read(out/'frame-names.json')
        self.robot_geometry=read(out/'robot-colliders.json');self.colliders=read(out/'gripper-colliders.json')
        with np.load(source/'reset-initial-state.npz') as archive:
            self.initial={name:archive[name].copy() for name in archive.files}
        root='/World/envs/env_0';self.frame_path=root+'/Frame'
        author_defaults(out/'scene.usda',out,root+'/Pocket',[root+'/LeftArm',root+'/RightArm',self.frame_path])
        verify_station_copy(source,out)
        # Read the SDK's actual cooked hulls in a short independent process.
        # Its scene is byte-identical; its transient states never enter the episode.
        from sdk_cooked_geometry import capture
        capture(out,source)
        self.sdk=SDKSession(out,enable_rendering=True)
        self.sdk.step(1/request['physics_hz'])
        self.robot=SDKYam(self.sdk)
        gate=read(out.parent/'product-sdk-gate.json')
        qualified=out.parent/gate['grasp_run']
        reference=read(qualified/'sdk-yam-reference-comparison.json')['reference_run']
        compare(out.parent/reference,read(out/'sdk-yam-contract.json'),out)
        class Scene(dict):
            pass
        self.scene=Scene();self.scene.sensors={}
        for side,arm in zip(('left','right'),self.robot.arms):
            body_names=self.robot_geometry[side]['body_names']
            original=read(out.parent/reference/'native-yam-contract.json')[side]
            view=NativeView(self.sdk,arm['root'],body_names,original['joint_names'])
            self.scene[side+'_robot']=SimpleNamespace(root_view=view,body_names=body_names,joint_names=original['joint_names'])
        view=NativeView(self.sdk,self.frame_path,self.names['body_names'],self.names['joint_names'])
        self.frame=SimpleNamespace(root_view=view,**self.names);self.scene['frame']=self.frame
        assert not view.bindings['ARTICULATION_LINK_POSE'].is_fixed_base, 'The task frame must remain free'
        source_dynamics=read(out/'frame-dynamics.json');native={**self.names}
        for key,method in (('masses_kg','get_masses'),('local_com_poses_xyzw','get_coms'),
            ('inertias_kg_m2','get_inertias'),('joint_stiffnesses','get_dof_stiffnesses'),('joint_dampings','get_dof_dampings')):
            actual=getattr(view,method)();expected=np.asarray(source_dynamics[key])
            assert actual.shape==expected.shape and np.allclose(actual,expected,atol=1e-7,rtol=1e-5), key
            native[key]=actual.tolist()
        native['scope']='Read-only matched SDK native mass, inertia and unconditional passive drive properties.'
        (out/'frame-dynamics.json').write_text(json.dumps(native,indent=2))
        speed=self.sdk.bind('ARTICULATION_DOF_MAX_VELOCITY',self.frame_path)
        speed.write(np.full(speed.shape,5.939047009418473e36,dtype=np.float32))
        assert np.all(self.sdk.read(speed)==np.float32(5.939047009418473e36))
        self.root_pose=self.sdk.bind('ARTICULATION_ROOT_POSE',self.frame_path)
        self.root_velocity=self.sdk.bind('ARTICULATION_ROOT_VELOCITY',self.frame_path)
        self.frame_target=self.sdk.bind('ARTICULATION_DOF_POSITION_TARGET',self.frame_path)
        self.frame_velocity_target=self.sdk.bind('ARTICULATION_DOF_VELOCITY_TARGET',self.frame_path)
        self.frame_effort=self.sdk.bind('ARTICULATION_DOF_ACTUATION_FORCE',self.frame_path)
        geometry=np.load(out/'geometry.npz');cover=FullCover.__new__(FullCover)
        cover.initial,cover.rest,cover.faces=[geometry[name].copy() for name in ('initial','rest','faces')]
        cover.geometry=read(out/'cover-mesh.json');cover.material=read(out/'effective-parameters.json')
        cover.center=np.array(read(out/'scene-audit.json')['cover_center_m'])
        cover.edges=surface_edges(cover.faces)
        cover.rest_lengths=np.linalg.norm(cover.rest[cover.edges[:,1]]-cover.rest[cover.edges[:,0]],axis=1)
        cover.max_strain=0.
        self.x,self.v=surface_bindings(self.sdk,root+'/Pocket',cover.material,cover.rest,cover.faces)
        rest=self.sdk.bind('SURFACE_DEFORMABLE_REST_POSITION',root+'/Pocket')
        cover.obj=SimpleNamespace(root_view=SimpleNamespace(
            get_simulation_nodal_positions=lambda:self.sdk.read(self.x),
            get_simulation_nodal_velocities=lambda:self.sdk.read(self.v),
            get_rest_nodal_positions=lambda:self.sdk.read(rest)))
        self.cover=cover;self.scene['pocket']=cover.obj
        self.sim=SimpleNamespace(step=self.native_step)
        self.reset(seed=request['seed'])
        from pxr import Usd,UsdPhysics
        stage=Usd.Stage.Open(str(out/'scene.usda'));poses=view.bindings['ARTICULATION_LINK_POSE']
        paths={prim.GetName():str(prim.GetPath()) for prim in stage.Traverse()
               if str(prim.GetPath()).startswith(self.frame_path+'/') and prim.HasAPI(UsdPhysics.RigidBodyAPI)}
        assert set(paths)==set(poses.body_names)
        self.cameras=SDKCameras(self.sdk,self.robot,root+'/Pocket',width=request['camera_width'],
            height=request['camera_height'],rigid_sources=[dict(poses=poses,paths=paths)])
        self.cameras.warmup_and_verify(self.cover.nodes()[0],self.x)
        (out/'native-initialization-check.json').write_text(json.dumps(dict(passed=not self.sdk.errors,native_errors=self.sdk.errors),indent=2))
        self.render()
        for filename in ('native-default-override.json','sdk-yam-contract.json','sdk-cloth-properties.json'):
            assert (out/filename).exists()
        (out/'sdk-source-preservation.json').write_text(json.dumps(dict(source_run=source.name,
            from_declared_reset=True,source_scene_sha256=hashlib.sha256((source/'scene.usda').read_bytes()).hexdigest(),
            scene_sha256=hashlib.sha256((out/'scene.usda').read_bytes()).hexdigest(),
            physical_geometry_materials_and_stock_robot_preserved=True,
            initialization_writes_only_before_recording=True,qualification_gate_sha256={name:hashlib.sha256((out.parent/name).read_bytes()).hexdigest()
                for name in ('product-sdk-gate.json','mechanism-sdk-gate.json')}),indent=2))

    def native_step(self,*args,**kwargs):
        self.robot.apply_gravity_compensation()
        self.sdk.step(1/self.request['physics_hz'])

    def step(self,actions):
        values=array(actions)
        assert values.shape==(1,14)
        self.robot.command(values[0])
        for _ in range(self.request['physics_hz']//20):self.sim.step()
        self.render()

    def render(self):
        if self.cameras is not None:
            for name,image in self.cameras.snapshot(self.cover.nodes()[0]).items():
                self.scene.sensors[name]=SimpleNamespace(data=SimpleNamespace(output={'rgb':image[None]}))

    def reset(self,seed):
        assert not self.recording_started, 'No native state writes after the episode starts'
        assert seed==self.request['seed']
        view=self.frame.root_view
        q=view.bindings['ARTICULATION_DOF_POSITION'];v=view.bindings['ARTICULATION_DOF_VELOCITY']
        order=[self.names['joint_names'].index(name) for name in q.dof_names]
        q.write(self.initial['frame_joints'][:,order].astype(np.float32))
        v.write(self.initial['frame_joint_velocities'][:,order].astype(np.float32))
        bridge=self.names['body_names'].index('Bridge')
        self.root_pose.write(self.initial['frame_poses'][:,bridge].astype(np.float32))
        self.root_velocity.write(np.zeros(self.root_velocity.shape,dtype=np.float32))
        for binding in (self.frame_target,self.frame_velocity_target,self.frame_effort):
            binding.write(np.zeros(binding.shape,dtype=np.float32))
        points=np.zeros(self.x.shape,dtype=np.float32);points[0,:len(self.cover.initial)]=self.initial['cloth_positions']
        velocity=np.zeros(self.v.shape,dtype=np.float32);velocity[0,:len(self.cover.initial)]=self.initial['cloth_velocities']
        self.x.write(points);self.v.write(velocity)
        self.robot.reset();self.cover.max_strain=0.
        self.sdk.engine.update_articulations_kinematic()
        self.cover.bind(self)
        self.render()

    def close(self):
        if self.cameras is not None:self.cameras.close();self.cameras=None
        if self.sdk is not None:self.sdk.close();self.sdk=None


def run(request,out):
    """Execute script, local interface diagnostic or Dream through the same SDK scene."""
    from robot_session import RobotSession
    from full_observation import make_observer
    from episode_checks import verify_resets,snapshot,replay
    from frame_robot import scripted_episode
    from station_coordinates import TRANSLATION_S_W
    source=validate_source(request,out.parent)
    env=session=None
    report=dict(status='initializing',backend='installed_ovphysx',full_task_success=False,
        diagnostic_only=request['policy']=='interface-check',score_status='Pending independent native outcome audit')
    try:
        env=SDKEnvironment.__new__(SDKEnvironment)
        env.sdk=env.cameras=None
        env.__init__(request,out,source)
        verify_resets(env,env.frame,env.cover,request,out)
        initial=snapshot(env,env.frame,env.cover)
        errors={name:float(abs(initial[name]-expected).max()) for name,expected in env.initial.items()}
        assert max(errors.values())<2e-6, 'SDK scene does not reconstruct the declared native starting state'
        (out/'sdk-reset-check.json').write_text(json.dumps(dict(passed=True,source_initial_max_abs_errors=errors,
            state_writes_only_before_recording=True),indent=2))
        bodies,observe=make_observer(env,env.frame,env.cover,env.colliders,env.robot_geometry,request)
        class Session(RobotSession):
            def measured_joint_pos(self):return self.env.robot.read14()
        session=Session(env,out,request,observe);session.cover_center=env.cover.center.copy()
        env.recording_started=True
        if request.get('replay_run'):
            replay(session,out.parent/request['replay_run'],out)
            report.update(status='completed',termination_reason='action_replay_complete')
        elif request['policy'] in ('dream','interface-check'):
            from dream_evaluation import run_policy
            report.update(run_policy(session))
        else:
            scene=read(out/'scene-audit.json');top=scene['table_center_m'][2]+scene['table_size_m'][2]/2
            scripted_episode(session,env.frame,bodies,TRANSLATION_S_W,top,env.metadata,env.robot_geometry,env.cover,report)
    except Exception:
        report.update(status='error',termination_reason='runtime_error',traceback=traceback.format_exc())
        print(report['traceback'],flush=True)
    finally:
        if session:
            session.close();report.update(control_steps=len(session.rows),physics_steps=session.physics_steps)
            env.cameras.provenance.update(recording_owner='RobotSession',shared_recorded_frames=len(session.rows))
        (out/'result.json').write_text(json.dumps(report,indent=2))
        if env:env.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--request',type=Path,required=True)
    path=parser.parse_args().request
    run(read(path),path.parent)
