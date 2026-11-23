"""Screen stock YAM reach for prospective garment poses; no physics or task success."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from robot_control import SIDE_ROT, GRASP_ROT
from yam_arena import ik
from yam_arena.constants import LEFT_BASE_POS, RIGHT_BASE_POS, READY

OUTPUT = OUTPUT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    assert args.out.resolve().is_relative_to(OUTPUT)
    records = []
    reflection = np.diag([1., -1., 1.])
    for pitch in (-20., -40., -60.):
        a = np.deg2rad(pitch)
        rotation = SIDE_ROT @ np.array([[1., 0., 0.], [0., np.cos(a), -np.sin(a)],
                                        [0., np.sin(a), np.cos(a)]])
        for side, base, y in (('left', LEFT_BASE_POS, .055), ('right', RIGHT_BASE_POS, -.055)):
            R = rotation if side == 'left' else reflection@rotation@np.diag([-1., 1., 1.])
            q = np.asarray(READY).copy()
            for z in np.arange(1.24, .59, -.08):
                target = np.array([.27, y, z])
                q, error, angle = ik.solve(target, ik.matrix_to_quat_xyzw(R), q, base_pos=base, iters=300)
                records.append({'kind': 'placket_reach', 'side': side, 'pitch_deg': pitch,
                                'target_m': target.tolist(), 'joints_rad': q.tolist(),
                                'error_m': error, 'angle_error_rad': angle,
                                'within_pose_tolerance': error < .001 and angle < .01})
    for side, base, y in (('left', LEFT_BASE_POS, .22), ('right', RIGHT_BASE_POS, -.22)):
        R = GRASP_ROT if side == 'left' else reflection@GRASP_ROT@np.diag([-1., 1., 1.])
        q = np.asarray(READY).copy()
        for z in (.95, 1.10, 1.18, 1.26, 1.32):
            target = np.array([.30, y, z])
            q, error, angle = ik.solve(target, ik.matrix_to_quat_xyzw(R), q, base_pos=base, iters=300)
            records.append({'kind': 'shoulder_lift', 'side': side, 'target_m': target.tolist(),
                            'joints_rad': q.tolist(), 'error_m': error, 'angle_error_rad': angle,
                            'within_pose_tolerance': error < .001 and angle < .01})
    result = {'scope': 'CPU reach screen only. No robot-object collision, cloth, hanger or success validation.',
              'method': 'Installed stock YAM IK, unmodified joint limits, deterministic continuation from READY.',
              'local_solver_limit': 'A failed solve is not a proof of global unreachability.',
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'records': records}
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    for kind in ('placket_reach', 'shoulder_lift'):
        selected = [r for r in records if r['kind'] == kind]
        print(json.dumps({'kind': kind, 'sampled': len(selected),
                          'within_pose_tolerance': sum(r['within_pose_tolerance'] for r in selected)}))


if __name__ == '__main__':
    main()
