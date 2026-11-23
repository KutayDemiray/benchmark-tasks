"""Stock YAM implicit drives and 14-D actions for explicit SDK diagnostics."""

import hashlib
import json
from pathlib import Path

import numpy as np


class SDKYam:
    """Apply the installed YAM action contract through documented native bindings."""

    def __init__(self, session):
        from yam_arena import constants, ik
        from pxr import Usd, UsdPhysics
        self.sim = session
        self.constants = constants
        self.ik = ik
        self.arms = []
        stage = Usd.Stage.Open(str(session.out / 'scene.usda'))
        evidence = {}
        for side, name in (('left', 'LeftArm'), ('right', 'RightArm')):
            root = '/World/envs/env_0/' + name
            q = session.bind('ARTICULATION_DOF_POSITION', root)
            poses = session.bind('ARTICULATION_LINK_POSE', root)
            assert poses.is_fixed_base and set(q.dof_names) == set(constants.ALL_JOINTS)
            assert set(poses.body_names) == set(constants.LINKS)
            arm_ids = [q.dof_names.index(name) for name in constants.ARM_JOINTS]
            finger_ids = [q.dof_names.index(name) for name in constants.FINGER_JOINTS]
            config = {}
            for joint in q.dof_names:
                config[joint] = (constants.SHOULDER if joint in constants.ARM_JOINTS[:3]
                                 else constants.WRIST if joint in constants.ARM_JOINTS[3:]
                                 else constants.GRIPPER)
            native = {}
            for tensor, key in (('ARTICULATION_DOF_STIFFNESS', 'stiffness'),
                                ('ARTICULATION_DOF_DAMPING', 'damping'),
                                ('ARTICULATION_DOF_MAX_FORCE', 'effort'),
                                ('ARTICULATION_DOF_MAX_VELOCITY', 'velocity')):
                binding = session.bind(tensor, root)
                expected = np.array([[config[name][key] for name in q.dof_names]], dtype=np.float32)
                assert binding.shape == expected.shape
                binding.write(expected)
                actual = session.read(binding)
                assert np.array_equal(actual, expected), 'Stock YAM native property mismatch: ' + key
                native[key] = actual.tolist()
            friction = session.bind('ARTICULATION_DOF_FRICTION_PROPERTIES', root)
            value = session.read(friction)
            assert value.shape == (1, 8, 3)
            # Installed Isaac Lab 3.0 writes static/dynamic/viscous properties.
            # YAM overrides static friction only; its USD dynamic/viscous defaults are zero.
            assert np.all(value[:, :, 1:] == 0), 'Unexpected stock dynamic or viscous friction'
            value[0, arm_ids, 0] = [constants.SHOULDER['friction']] * 3 + [constants.WRIST['friction']] * 3
            friction.write(value)
            assert np.array_equal(session.read(friction), value)
            native['friction_static_dynamic_viscous'] = value.tolist()
            limits = session.read(session.bind('ARTICULATION_DOF_LIMIT', root))
            assert np.allclose(limits[0, arm_ids, 0], ik.LOWER, atol=1e-6, rtol=0)
            assert np.allclose(limits[0, arm_ids, 1], ik.UPPER, atol=1e-6, rtol=0)
            assert np.allclose(limits[0, finger_ids], [[constants.FINGER_OPEN, 0.]] * 2, atol=1e-7, rtol=0)
            native['limits_rad_or_m'] = limits.tolist()
            for tensor in ('ARTICULATION_BODY_MASS', 'ARTICULATION_BODY_INERTIA',
                           'ARTICULATION_BODY_COM_POSE', 'ARTICULATION_BODY_DISABLE_GRAVITY',
                           'ARTICULATION_DOF_ARMATURE', 'ARTICULATION_DOF_DRIVE_TYPE'):
                native[tensor] = session.read(session.bind(tensor, root),
                    np.uint8 if tensor.endswith(('DISABLE_GRAVITY', 'DRIVE_TYPE')) else np.float32).tolist()
            assert not np.asarray(native['ARTICULATION_BODY_DISABLE_GRAVITY']).any()
            assert np.all(np.asarray(native['ARTICULATION_DOF_DRIVE_TYPE']) == 1), 'Stock force drives required'
            paths = {}
            for prim in stage.Traverse():
                if str(prim.GetPath()).startswith(root + '/') and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                    assert prim.GetName() not in paths
                    paths[prim.GetName()] = str(prim.GetPath())
            assert set(paths) == set(poses.body_names)
            initial = np.zeros(q.shape, dtype=np.float32)
            initial[0, arm_ids] = constants.READY
            initial[0, finger_ids] = constants.FINGER_OPEN
            q.write(initial)
            for kind in ('ARTICULATION_DOF_VELOCITY', 'ARTICULATION_DOF_VELOCITY_TARGET', 'ARTICULATION_DOF_ACTUATION_FORCE'):
                binding = session.bind(kind, root)
                binding.write(np.zeros(binding.shape, dtype=np.float32))
            target = session.bind('ARTICULATION_DOF_POSITION_TARGET', root)
            target.write(initial)
            effort = session.bind('ARTICULATION_DOF_ACTUATION_FORCE', root)
            gravity = session.bind('ARTICULATION_GRAVITY_FORCE', root)
            self.arms.append(dict(side=side, root=root, q=q, poses=poses, arm_ids=arm_ids,
                finger_ids=finger_ids, target=target, effort=effort, gravity=gravity, paths=paths))
            evidence[side] = dict(joint_names=q.dof_names, body_names=poses.body_names,
                                 native_properties=native, body_paths=paths)
        session.engine.update_articulations_kinematic()
        for arm in self.arms:
            evidence[arm['side']]['initial_native_q'] = session.read(arm['q']).tolist()
            evidence[arm['side']]['initial_gravity_compensation'] = session.read(arm['gravity']).tolist()
            evidence[arm['side']]['initial_link_poses'] = session.read(arm['poses']).tolist()
        self.action = np.r_[constants.READY, 1., constants.READY, 1.]
        assert np.allclose(self.read14(), self.action, atol=1e-7, rtol=0)
        source_root = Path(constants.__file__).parent
        evidence['contract'] = dict(stock_constants_verified=True, action_dimension=14, control_hz=20,
            joint_target_pd='Native implicit force drives, unchanged stock gains/limits.',
            gravity_compensation='Native generalized gravity forces on six arm joints every physics step.',
            finger_command='Normalized opening times stock FINGER_OPEN on both finger joints.',
            state_writes_only_at_initialization=True,
            installed_source_sha256={name: hashlib.sha256((source_root/name).read_bytes()).hexdigest()
                                    for name in ('constants.py', 'robot.py', 'actions.py')})
        (session.out/'sdk-yam-contract.json').write_text(json.dumps(evidence, indent=2))

    def reset(self):
        """Restore the stock ready pose before the episode recorder is attached."""
        for arm in self.arms:
            value=np.zeros(arm['q'].shape,dtype=np.float32)
            value[0,arm['arm_ids']]=self.constants.READY
            value[0,arm['finger_ids']]=self.constants.FINGER_OPEN
            arm['q'].write(value)
            arm['target'].write(value)
            for kind in ('ARTICULATION_DOF_VELOCITY','ARTICULATION_DOF_VELOCITY_TARGET'):
                if kind not in arm:arm[kind]=self.sim.bind(kind,arm['root'])
                arm[kind].write(np.zeros(arm[kind].shape,dtype=np.float32))
            arm['effort'].write(np.zeros(arm['effort'].shape,dtype=np.float32))
        self.action=np.r_[self.constants.READY,1.,self.constants.READY,1.]
        self.sim.engine.update_articulations_kinematic()

    def read14(self):
        result = []
        for arm in self.arms:
            q = self.sim.read(arm['q'])[0]
            result.extend(q[arm['arm_ids']].tolist())
            result.append(float(np.clip(q[arm['finger_ids']].mean()/self.constants.FINGER_OPEN, 0., 1.)))
        return np.asarray(result)

    def command(self, action):
        from evaluation_contract import process_action
        action = process_action(action)
        c = self.constants
        for offset, arm in zip((0, 7), self.arms):
            value = np.zeros(arm['target'].shape, dtype=np.float32)
            value[0, arm['arm_ids']] = action[offset:offset+6]
            value[0, arm['finger_ids']] = action[offset+6]*c.FINGER_OPEN
            arm['target'].write(value)
        self.action = action.copy()

    def apply_gravity_compensation(self):
        for arm in self.arms:
            gravity = self.sim.read(arm['gravity'])
            assert gravity.shape == arm['q'].shape and np.isfinite(gravity).all()
            effort = np.zeros(arm['effort'].shape, dtype=np.float32)
            effort[0, arm['arm_ids']] = gravity[0, arm['arm_ids']]
            arm['effort'].write(effort)
