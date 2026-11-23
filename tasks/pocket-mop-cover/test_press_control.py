"""Bounded ordinary pressing when a passive frame closes or remains stuck."""

from task_paths import OUTPUT_ROOT

from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy.spatial.transform import Rotation

from robot_reopen import finish_press
from bridge_press import press_bridge


class PressTests(unittest.TestCase):
    def test_bridge_pressure_follows_horizontal_motion_with_bounded_depth(self):
        self.run_bridge_case(stuck=True)

    def test_bridge_pressure_stops_when_passive_wings_close(self):
        self.run_bridge_case(stuck=False)

    def run_bridge_case(self, stuck):
        with tempfile.TemporaryDirectory(dir=OUTPUT_ROOT) as tmp:
            class Session:
                out = Path(tmp)
                moves = []

                def move(self, *args, **kwargs):
                    self.moves.append(args)

                def hold(self, *args):
                    pass

            session = Session()
            bridge = np.eye(4)
            bridge[:3, 3] = [.1, .2, .02]

            def measured_frame():
                count = sum(move[-1] == 'press_bridge_rail' for move in session.moves)
                current = bridge.copy()
                current[0, 3] += count * .001
                angle = 6. if stuck else max(0., 6. - 1.1 * count)
                poses = {'Bridge': current}
                for name, sign in (('Left', -1.), ('Right', 1.)):
                    wing = current.copy()
                    wing[:3, :3] = Rotation.from_euler('y', sign * angle, degrees=True).as_matrix()
                    poses[name] = wing
                return poses

            press_bridge(session, bridge, measured_frame)
            presses = [move for move in session.moves if move[-1] == 'press_bridge_rail']
            self.assertEqual(len(presses), 18 if stuck else 5)
            self.assertLessEqual(sum(move[4] for move in session.moves) + .5, 13.2000001)
            touch = session.moves[1][1]
            for index, move in enumerate(presses):
                self.assertAlmostEqual(move[1][0], touch[0] + index * .001)
                self.assertGreaterEqual(move[1][2], touch[2] - .065000001)
                self.assertEqual((move[0], move[3]), ('left', 0.))
            self.assertEqual(session.moves[-1][-1], 'clear_bridge_workspace')
            self.assertGreater(session.moves[-1][1][2], touch[2] + .099)
            self.assertGreater(session.moves[-1][1][1], touch[1] + .199)

    def test_stuck_frame_cannot_request_unbounded_depth(self):
        self.run_case(stuck=True)

    def test_press_stops_on_measured_flat_frame(self):
        self.run_case(stuck=False)

    def run_case(self,stuck):
        with tempfile.TemporaryDirectory(dir=OUTPUT_ROOT) as tmp:
            class Session:
                out=Path(tmp)
                moves=[]
                def move(self,*args):
                    self.moves.append(args)
            session=Session()
            def measured_frame():
                angle=6. if stuck else max(0.,6.-1.1*len(session.moves))
                poses={name:np.eye(4) for name in ('Bridge','Left','Right')}
                for name,sign in (('Left',-1.),('Right',1.)):
                    poses[name][:3,:3]=Rotation.from_euler('y',sign*angle,degrees=True).as_matrix()
                return poses
            target=np.eye(4);target[:3,3]=[.1,.2,.3]
            finish_press(session,target,measured_frame)
            self.assertEqual(len(session.moves),7 if stuck else 5)
            for side,position,rotation,opening,seconds,phase in session.moves:
                self.assertEqual(side,'right')
                np.testing.assert_allclose(position[:2],target[:2,3])
                np.testing.assert_array_equal(rotation,np.eye(3))
                self.assertLessEqual(target[2,3]-position[2],.035000001)
                self.assertEqual((opening,seconds,phase),(.08,.4,'finish_frame_press'))


if __name__=='__main__':
    unittest.main()
