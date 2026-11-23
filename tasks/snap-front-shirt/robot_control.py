"""Local Cartesian feedback converted exclusively to stock 14D YAM actions."""

import numpy as np

from yam_arena import ik
from yam_arena.constants import LEFT_BASE_POS, READY

GRASP_ROT = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
BUNCH_ROT = GRASP_ROT @ np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
SIDE_ANGLE = np.deg2rad(-70.)
SIDE_ROT = np.column_stack([[np.sin(SIDE_ANGLE), -np.cos(SIDE_ANGLE), 0.],
                            [0., 0., -1.],
                            [np.cos(SIDE_ANGLE), np.sin(SIDE_ANGLE), 0.]])
LONG_SIDE_PITCH = np.deg2rad(-20.)
LONG_SIDE_ROT = SIDE_ROT @ np.array([[1., 0., 0.],
    [0., np.cos(LONG_SIDE_PITCH), -np.sin(LONG_SIDE_PITCH)],
    [0., np.sin(LONG_SIDE_PITCH), np.cos(LONG_SIDE_PITCH)]])
SIDE_HOVER = np.array([.23, .15, .99])


class CouponController:
    """Approach a cloth overhang, pinch, lift, hold and release."""

    def __init__(self, initial_q, initial_tcp, cfg, case):
        self.q = np.asarray(initial_q).copy()
        self.start = np.asarray(initial_tcp).copy()
        self.cfg = cfg
        self.case = case
        self.grasp_mode = cfg['robot_test'].get('grasp_mode', 'pinch')
        self.rotation = BUNCH_ROT if self.grasp_mode == 'bunch' else GRASP_ROT
        self.pick = np.array([cfg['robot_test'].get('pick_x_m', .295), .12, cfg['robot_test']['support_top_m']])
        self.max_ik_error = 0.
        self.max_ik_angle = 0.
        self.phase = 'approach'
        self.release_tcp = None

    def action(self, time_s, measured_tcp, front_nodes, hardware_pose):
        """Return the ordinary action and the current stage."""
        hover = self.pick + [-.055, 0., .05]
        if time_s < 2.5:
            phase, goal, grip = 'approach', hover, 1.
            s = ik.smoothstep(time_s/2.5)
            target = self.start*(1-s)+goal*s
        elif time_s < 5.:
            phase, grip = 'approach_cloth', 1. if self.grasp_mode == 'bunch' else .3
            # Track the actual unheld cloth, then hold the target while closing.
            if len(front_nodes):
                self.pick[2] = float(np.mean(front_nodes[:, 2]))
            s = ik.smoothstep((time_s-2.5)/2.5)
            target = hover*(1-s)+self.pick*s
        elif time_s < 6.:
            phase, target = 'close', self.pick.copy()
            grip = 1.-ik.smoothstep(time_s-5.) if self.grasp_mode == 'bunch' else 0.
        elif time_s < 8.:
            phase, grip = 'lift', 0.
            target = self.pick + [0., 0., .06*ik.smoothstep((time_s-6.)/2.)]
        elif time_s < 10.:
            phase, target, grip = 'hold', self.pick+[0., 0., .06], 0.
        elif time_s < 12.:
            phase, grip = 'carry_over_support', 0.
            target = self.pick + [.10*ik.smoothstep((time_s-10.)/2.), 0., .06]
        elif time_s < 14.:
            phase, grip = 'lower_to_support', 0.
            h = self.cfg['robot_test']['support_top_m']
            offset = float(measured_tcp[2]-hardware_pose[2])
            desired_z = max(h+.020, min(h+.055, h+self.cfg['robot_test'].get('release_cap_height_m', .006)+offset))
            if self.release_tcp is None:
                self.release_tcp = self.pick + [.10, 0., .06]
            self.release_tcp[2] += float(np.clip(desired_z-self.release_tcp[2], -.003, .003))
            target = self.release_tcp.copy()
        elif time_s < 15.:
            phase, target, grip = 'open', self.release_tcp.copy(), 1.
        else:
            phase, grip = 'retreat', 1.
            target = self.release_tcp + [-.055*ik.smoothstep((time_s-15.)/2.), 0., .07*ik.smoothstep((time_s-15.)/2.)]
        if self.case == 'reach':
            phase, target, grip = 'reach', hover, 1.
        self.q, err, angle = ik.solve(target, ik.matrix_to_quat_xyzw(self.rotation),
                                     self.q, base_pos=LEFT_BASE_POS, iters=80)
        self.max_ik_error = max(self.max_ik_error, err)
        self.max_ik_angle = max(self.max_ik_angle, angle)
        self.phase = phase
        return np.r_[self.q, grip, READY, 1.], {'phase': phase, 'target_tcp_m': target,
                                               'ik_error_m': err, 'ik_angle_rad': angle}


class SideCouponController:
    """Pinch the accessible free tab from its sides below a thin physical plate."""

    def __init__(self, initial_q, initial_tcp, cfg, case):
        self.q = np.asarray(initial_q).copy()
        self.start = np.asarray(initial_tcp).copy()
        self.h = cfg['robot_test']['support_top_m']
        self.long_tab = cfg['robot_test'].get('grasp_mode') == 'side_long'
        self.rotation = LONG_SIDE_ROT if self.long_tab else SIDE_ROT
        self.pick = np.array([.261, .124, self.h-.011])
        self.release_tcp = None
        self.max_ik_error = self.max_ik_angle = 0.

    def action(self, time_s, measured_tcp, front_nodes, hardware_pose):
        if time_s < 4. and len(front_nodes):
            if self.long_tab:
                # Put the observed free-edge patch 5 mm inside the finger pads.
                self.pick = np.mean(front_nodes, axis=0)-.005*self.rotation[:, 2]
            else:
                self.pick[1] = float(np.mean(front_nodes[:, 1]))+.001
                self.pick[2] = float(np.clip(np.min(front_nodes[:, 2])-.003, self.h-.025, self.h-.011))
        # Approach along the pad axis after straddling the thin vertical fabric.
        # The old diagonal path put the patch against one finger before closing.
        outside = self.pick-.045*self.rotation[:, 2] if self.long_tab else self.pick+[-.025, .025, 0.]
        opening = .65 if self.long_tab else .35
        extraction = np.array([-.05, 0., .08]) if self.long_tab else np.array([-.035, 0., 0.])
        lift_height = .17 if self.long_tab else .10
        extracted = self.pick+extraction
        raised = extracted+[0., 0., lift_height]
        carried = raised + [.14, 0., 0.]
        if time_s < 2.:
            phase, grip = 'approach', opening
            f = ik.smoothstep(time_s/2.)
            target = self.start*(1-f)+outside*f
        elif time_s < 4.:
            phase, grip = 'approach_cloth', opening
            f = ik.smoothstep((time_s-2.)/2.)
            target = outside*(1-f)+self.pick*f
        elif time_s < 5.:
            phase, target = 'close', self.pick.copy()
            grip = opening*(1.-ik.smoothstep(time_s-4.)) if self.long_tab else 0.
        elif time_s < 7.:
            phase, grip = 'extract_tab', 0.
            target = self.pick+extraction*ik.smoothstep((time_s-5.)/2.)
        elif time_s < 9.:
            phase, grip = 'lift', 0.
            target = extracted + [0., 0., lift_height*ik.smoothstep((time_s-7.)/2.)]
        elif time_s < 11.:
            phase, target, grip = 'hold', raised, 0.
        elif time_s < 13.:
            phase, grip = 'carry_over_support', 0.
            target = raised + [.14*ik.smoothstep((time_s-11.)/2.), 0., 0.]
        elif time_s < 15.:
            phase, grip = 'lower_to_support', 0.
            if self.release_tcp is None:
                self.release_tcp = carried.copy()
            offset = float(measured_tcp[2]-hardware_pose[2])
            desired = max(self.h+.06, min(self.h+(.20 if self.long_tab else .10), self.h+.014+offset))
            self.release_tcp[2] += float(np.clip(desired-self.release_tcp[2], -.003, .003))
            target = self.release_tcp.copy()
        elif time_s < 16.:
            phase, target, grip = 'open', self.release_tcp.copy(), 1.
        else:
            phase, grip = 'retreat', 1.
            f = ik.smoothstep((time_s-16.)/3.)
            target = self.release_tcp + [-.08*f, .02*f, .06*f]
        self.q, error, angle = ik.solve(target, ik.matrix_to_quat_xyzw(self.rotation), self.q,
                                        base_pos=LEFT_BASE_POS, iters=150)
        self.max_ik_error = max(self.max_ik_error, error)
        self.max_ik_angle = max(self.max_ik_angle, angle)
        info = {'phase': phase, 'target_tcp_m': target, 'ik_error_m': error, 'ik_angle_rad': angle,
                'observed_grasp_target_m': self.pick.copy()}
        if error>.003 or angle>.02:
            info.update(abort=True, abort_reason='side_grasp_target_unreachable')
        return np.r_[self.q, grip, READY, 1.], info


class PressController:
    """Press an initially aligned pair using stock finger-pad collision contacts."""

    def __init__(self, initial_q, initial_tcp, cfg, case):
        from yam_arena.demo import DOWN_GRIP
        from yam_arena.urdf import quat_xyzw_to_matrix
        self.q = np.asarray(initial_q).copy()
        self.start = np.asarray(initial_tcp).copy()
        self.start_rotation = ik.tcp_pose(self.q, LEFT_BASE_POS)[1]
        self.rotation = quat_xyzw_to_matrix(np.asarray(DOWN_GRIP))
        self.cfg = cfg
        self.case = case
        self.early_contact = cfg['robot_test'].get('press_start') == 'hover'
        self.approach_end = 0. if self.early_contact else 3.
        self.contact_end = .6 if self.early_contact else 5.
        self.h = cfg['robot_test'].get('press_support_top_m', .90)
        self.hover = np.array([.30, .12, self.h+.045])
        self.press_z = None
        self.press_end = None
        self.seated_samples = 0
        self.max_ik_error = self.max_ik_angle = 0.

    def action(self, time_s, measured_tcp, front_nodes, hardware_pose):
        from robot_scoring import snap_seated
        male_pose, socket_pose = hardware_pose['male'], hardware_pose['socket']
        rot = self.rotation
        if time_s < self.approach_end:
            phase = 'approach'
            s = ik.smoothstep(time_s/self.approach_end)
            target = self.start*(1-s)+self.hover*s
            rot = ik.slerp_matrix(self.start_rotation, self.rotation, s)
        elif time_s < self.contact_end:
            phase = 'contact_approach'
            s = ik.smoothstep((time_s-self.approach_end)/(self.contact_end-self.approach_end))
            xy = .5*(np.asarray(male_pose[:2])+socket_pose[:2]) if self.early_contact else male_pose[:2]
            contact = np.array([*xy, male_pose[2]+(.0125 if self.early_contact else .017)])
            origin = self.start if self.early_contact else self.hover
            target = origin*(1-s)+contact*s
            self.press_z = float(target[2])
        elif self.press_end is None:
            phase = 'press'
            seated = snap_seated(male_pose, socket_pose, self.cfg)
            self.seated_samples = self.seated_samples+1 if seated else 0
            if not seated:
                # Compensate for stock actuator compliance using ordinary targets.
                # No object force/pose write or mechanism parameter change.
                self.press_z = max(self.h-.005, self.press_z-.00025)
            xy = .5*(np.asarray(male_pose[:2])+socket_pose[:2]) if self.early_contact else male_pose[:2]
            target = np.array([*xy, self.press_z])
            if self.seated_samples>=4 or time_s>=14.:
                self.press_end = time_s
        elif time_s < self.press_end+2.:
            phase = 'withdraw'
            fraction = ik.smoothstep((time_s-self.press_end)/2.)
            target = np.array([male_pose[0], male_pose[1],
                               self.press_z*(1-fraction)+self.hover[2]*fraction])
        else:
            phase, target = 'released_observation', self.hover.copy()
        if self.case == 'no_press':
            phase, target = 'no_press', self.hover.copy()
            rot = self.rotation
        self.q, err, angle = ik.solve(target, ik.matrix_to_quat_xyzw(rot), self.q,
                                     base_pos=LEFT_BASE_POS, iters=80)
        self.max_ik_error = max(self.max_ik_error, err)
        self.max_ik_angle = max(self.max_ik_angle, angle)
        completed = self.case!='no_press' and self.press_end is not None and time_s>=self.press_end+4.
        return np.r_[self.q, 0., READY, 1.], {'phase': phase, 'target_tcp_m': target,
                'ik_error_m': err, 'ik_angle_rad': angle, 'completed': completed,
                'press_end_s': self.press_end, 'consecutive_seated_controls': self.seated_samples,
                'snap_lateral_error_m': float(np.linalg.norm(np.asarray(male_pose[:2])-socket_pose[:2]))}

class PressLiftController:
    """Press a free pair, then grasp only the upper tab and lift both coupons."""

    def __init__(self, initial_q, initial_tcp, cfg, case):
        import copy
        self.press = PressController(initial_q, initial_tcp, cfg, 'press')
        self.q = np.asarray(initial_q).copy()
        self.cfg = copy.deepcopy(cfg)
        self.cfg['robot_test'].update(support_top_m=cfg['robot_test']['press_support_top_m'],
                                     pick_x_m=.27, release_cap_height_m=.014)
        self.grasp = None
        self.transition_at = None
        self.mode = self.cfg['robot_test'].get('grasp_mode', 'pinch')
        self.rotation = LONG_SIDE_ROT if self.mode == 'side_long' else SIDE_ROT if self.mode == 'side' else BUNCH_ROT if self.mode == 'bunch' else GRASP_ROT
        self.lift_baseline = None
        self.max_ik_error = self.max_ik_angle = 0.

    def action(self, time_s, measured_tcp, front_nodes, hardware_pose):
        from robot_scoring import snap_seated
        if self.transition_at is None:
            action, info = self.press.action(time_s, measured_tcp, None, hardware_pose)
            self.q = action[:6].copy()
            if info.get('completed'):
                if not snap_seated(hardware_pose['male'], hardware_pose['socket'], self.cfg):
                    info.update(phase='press_failed', abort=True, abort_reason='single_snap_did_not_seat')
                else:
                    self.transition_at = time_s
                    self.lift_baseline = {name: hardware_pose[name][2] for name in ('male', 'socket')}
                    self.transition_start = np.asarray(measured_tcp).copy()
                    self.transition_rotation = ik.tcp_pose(self.q, LEFT_BASE_POS)[1]
                info['completed'] = False
        elif time_s < self.transition_at+3.:
            fraction = ik.smoothstep((time_s-self.transition_at)/3.)
            goal = SIDE_HOVER if self.mode in ('side', 'side_long') else np.array([.215, .12, self.cfg['robot_test']['support_top_m']+.05])
            target = self.transition_start*(1-fraction)+goal*fraction
            rot = ik.slerp_matrix(self.transition_rotation, self.rotation, fraction)
            self.q, err, angle = ik.solve(target, ik.matrix_to_quat_xyzw(rot), self.q,
                                         base_pos=LEFT_BASE_POS, iters=100)
            action = np.r_[self.q, .35 if self.mode in ('side', 'side_long') else 1., READY, 1.]
            info = {'phase': 'change_to_cloth_grasp', 'target_tcp_m': target,
                    'ik_error_m': err, 'ik_angle_rad': angle}
        else:
            if self.grasp is None:
                cls = SideCouponController if self.mode in ('side', 'side_long') else CouponController
                self.grasp = cls(self.q, measured_tcp, self.cfg, 'grasp')
            elapsed = time_s-self.transition_at-3.
            action, info = self.grasp.action(elapsed, measured_tcp, front_nodes, hardware_pose['male'])
            self.q = action[:6].copy()
            info['completed'] = elapsed >= (19. if self.mode in ('side', 'side_long') else 17.)
            hold_start = 9. if self.mode in ('side', 'side_long') else 8.
            if hold_start <= elapsed < hold_start+2. and any(
                    hardware_pose[name][2] < self.lift_baseline[name]+.015
                    for name in ('male', 'socket')):
                info.update(abort=True, abort_reason='upper_tab_grasp_failed', completed=False)
        self.max_ik_error = max(self.max_ik_error, info['ik_error_m'])
        self.max_ik_angle = max(self.max_ik_angle, info['ik_angle_rad'])
        return action, info
