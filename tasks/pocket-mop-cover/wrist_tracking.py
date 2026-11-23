"""Measured wrist tracking through ordinary position commands and stock gains."""

import json

import numpy as np
from scipy.spatial.transform import Rotation


def limited(vector, maximum):
    """Limit a correction vector's norm without changing its direction."""
    return vector * min(1., maximum / max(float(np.linalg.norm(vector)), 1e-12))


class MeasuredWristGuide:
    """Correct measured grasp pose error without driving an object or passive joint.

    Measurements can be robot forward kinematics or markers fixed to held bodies.
    A missing measurement callback is an ideal-tracking CPU preflight, which
    cannot establish loaded contact feasibility.
    """

    def __init__(self, session, measure=None, measure_robot=None, measurement='robot_joint_fk'):
        self.session = session
        self.measure = measure
        self.measure_robot = measure_robot
        self.measurement = measurement
        self.offset = {side: np.zeros(3) for side in ('left', 'right')}
        self.rotation = {side: np.eye(3) for side in ('left', 'right')}
        self.records = []
        self.last_targets = None

    def move(self, targets, seconds, phase):
        """Execute a bounded correction, then measure error for the next motion."""
        corrected = {}
        for side, (position, rotation, opening) in targets.items():
            corrected[side] = (np.asarray(position) + self.offset[side],
                               self.rotation[side] @ rotation, opening)
        self.session.move_both(corrected, seconds, phase)
        self.last_targets = corrected
        if self.measure is None:
            return
        record = dict(phase=phase, ordinary_actions_only=True, hands={})
        for side, (position, rotation, _) in targets.items():
            measured = self.measure(side)
            error = np.asarray(position) - measured[:3, 3]
            angular = Rotation.from_matrix(rotation @ measured[:3, :3].T).as_rotvec()
            record['hands'][side] = dict(position_error_m=error.tolist(),
                rotation_error_rad=angular.tolist(), applied_position_bias_m=self.offset[side].tolist(),
                applied_rotation_bias_rad=Rotation.from_matrix(self.rotation[side]).as_rotvec().tolist())
            if self.measure_robot is not None:
                robot = self.measure_robot(side)
                record['hands'][side].update(
                    robot_position_error_m=(np.asarray(position)-robot[:3, 3]).tolist(),
                    robot_rotation_error_rad=Rotation.from_matrix(rotation@robot[:3, :3].T).as_rotvec().tolist())
            self.offset[side] = limited(self.offset[side] + limited(.7 * error, .006), .045)
            update = Rotation.from_rotvec(limited(.7 * angular, np.deg2rad(3.))).as_matrix()
            proposed = Rotation.from_matrix(update @ self.rotation[side]).as_rotvec()
            self.rotation[side] = Rotation.from_rotvec(limited(proposed, np.deg2rad(15.))).as_matrix()
        self.records.append(record)
        (self.session.out / 'wrist-tracking.json').write_text(json.dumps(dict(
            scope='Ordinary wrist position targets corrected from measured poses; unchanged actuator gains/limits. No object or passive-joint writes.',
            measurement=self.measurement,
            records=self.records), indent=2))

    def settle(self, targets, phase):
        """Correct steady load error while both ends remain clear of the cover."""
        # Reserve direct bridge pressure and upright release within 120 seconds.
        for _ in range(10):
            self.move(targets, .75, phase)
            if self.measure is not None:
                errors=self.records[-1]['hands'].values()
                if all(np.linalg.norm(value['position_error_m'])<.001 and
                       np.linalg.norm(value['rotation_error_rad'])<.01 for value in errors):
                    return
        assert self.measure is None, 'Loaded wrist tracking did not converge within the bounded alignment period'
