"""Prefix scoring from native states, independently checked against offline scoring."""

from collections import deque

import numpy as np

from episode_timing import control_hz, max_actions, physics_stride, retention_samples
from robot_scoring import snap_seated
from tabletop_scoring import seating_matrix


class StreamingScore:
    """Track validity and retained closures without retaining an entire episode."""

    def __init__(self, cfg, meta):
        self.cfg, self.meta = cfg, meta
        self.count = len(cfg['garment']['snap_heights_from_hem_m'])
        self.required = cfg['tabletop']['required_snap_indices']
        self.stride, self.hz = physics_stride(cfg), control_hz(cfg)
        self.index = -1
        self.control_steps = 0
        self.last = None
        self.last_bad_pair = np.full((self.count,self.count),-1,dtype=np.int64)
        self.last_wrong = self.last_bad_release = -1
        self.invalid = []
        self.window = deque(maxlen=retention_samples(cfg))
        self.score = 0.
        self.retained = []
        self.maximum = np.zeros(4)
        try:
            assert self.required == [0,1,2], 'unexpected_task_definition'
            assert not meta.get('native_execution_error',False), 'native_execution_error'
            assert not meta['diagnostic_world_mounts'] and not meta['grasp_attachments'], 'hidden_attachment'
            assert not meta.get('hanger_attachments',False), 'hidden_attachment'
            assert len(meta['cloth']['attachments']) == 2*self.count, 'missing_local_rivets'
            for i in self.required:
                assert not snap_seated(meta['initial_hardware'][f'male{i}'],
                                       meta['initial_hardware'][f'socket{i}'],cfg), 'initially_closed_snap'
        except (AssertionError,KeyError,ValueError,TypeError,IndexError) as exc:
            self.invalidate(str(exc))

    def invalidate(self, reason):
        if reason not in self.invalid:
            self.invalid.append(reason)
        self.score = 0.
        self.retained = []

    def observe_physics(self, state):
        expected = self.index+1
        self.index = expected
        self.last = state
        try:
            assert state['physics_step'] == expected, 'missing_physics_state'
            poses = np.asarray(list(state['hardware'].values()),dtype=float)
            assert poses.shape[-1] == 7 and np.isfinite(poses).all(), 'nonfinite_hardware'
            assert (np.abs((poses[:,3:]**2).sum(axis=1)-1)<.01).all(), 'invalid_hardware_quaternion'
            values = np.array([state[k] for k in ('strain','rivet_error_m','table_penetration_m','table_coverage_fraction')])
            assert np.isfinite(values).all() and (values>=0).all(), 'nonfinite_measurement'
            assert values[3]<=1, 'invalid_table_coverage'
            assert state['finite'], 'nonfinite_state'
            assert values[0]<=self.cfg['gate']['max_edge_strain'], 'cloth_strain_limit'
            assert values[1]<=self.cfg['gate']['max_attachment_error_m'], 'rivet_error_limit'
            assert values[2]<=self.cfg['tabletop']['max_table_penetration_m'], 'table_penetration'
            forces=[]
            for kind in ('male','socket'):
                for i in range(self.count):
                    forces.append(state['snap_robot_contact_norm_N'][f'{kind}{i}'])
            forces=np.asarray(forces)
            assert np.isfinite(forces).all() and (forces>=0).all(), 'invalid_contact_measurement'
            peak=float(forces.max())
            self.maximum=np.maximum(self.maximum,[*values[:3],peak])
            assert peak<=25., 'robot_normal_force_limit'
            seated,rim=seating_matrix([state],self.count,self.cfg,with_invalid=True)
            assert not rim.any(), 'cap_through_socket_rim'
            assert expected!=0 or not seated.any(), 'initially_closed_snap'
            self.last_bad_pair[~seated[0]]=expected
            if seated[0][~np.eye(self.count,dtype=bool)].any():
                self.last_wrong=expected
            if values[3]<.98 or peak>.05:
                self.last_bad_release=expected
        except (AssertionError,KeyError,ValueError,TypeError,IndexError) as exc:
            self.invalidate(str(exc))

    def observe_control(self, row):
        step=self.control_steps
        self.control_steps+=1
        try:
            assert self.control_steps<=max_actions(self.cfg), 'incomplete_episode'
            assert self.index==self.control_steps*self.stride, 'incomplete_episode'
            assert row['step']==step and abs(row['time_s']-self.control_steps/self.hz)<1e-7, 'invalid_control_time'
            assert row['physics']==self.last, 'control_physics_mismatch'
            action=np.asarray(row['robot_action'])
            assert action.shape==(14,) and np.isfinite(action).all(), 'invalid_action'
            assert np.all((action[[6,13]]>=0)&(action[[6,13]]<=1)), 'invalid_gripper_action'
            clearance=np.array([row['robot_clearance_m'][k] for k in ('fabric_m','hardware_m')])
            assert np.isfinite(clearance).all(), 'invalid_clearance'
            center=np.asarray(row['physics']['mean_cloth_position_m'])
            assert center.shape==(3,) and np.isfinite(center).all(), 'invalid_cloth_position'
            self.window.append(dict(clearance=float(min(row['robot_clearance_m'].values())),
                                    center=center.copy(),physics_step=self.index))
        except (AssertionError,KeyError,ValueError,TypeError,IndexError) as exc:
            self.invalidate(str(exc))
        self.score=0.
        self.retained=[]
        if self.invalid or len(self.window)<self.window.maxlen:
            return self.result()
        p=self.cfg['tabletop']
        first=self.window[0]['physics_step']
        if self.last_bad_release>=first or self.last_wrong>=first:
            return self.result()
        if any(row['clearance']<p['released_robot_clearance_m'] for row in self.window):
            return self.result()
        centers=np.array([row['center'] for row in self.window])
        if np.linalg.norm(centers-centers[-1],axis=1).max()>p['max_released_drift_m']:
            return self.result()
        self.retained=[i for i in self.required if self.last_bad_pair[i,i]<first]
        self.score=len(self.retained)/len(self.required)
        return self.result()

    def finalize(self):
        if self.control_steps==0 or self.index!=self.control_steps*self.stride:
            self.invalidate('incomplete_episode')
        if self.meta.get('native_execution_error',False):
            self.invalidate('native_execution_error')
        return self.result()

    def result(self):
        return dict(task_score=self.score,full_task_success=self.score==1.,
                    retained_pairs=list(self.retained),invalid_reasons=list(self.invalid))
