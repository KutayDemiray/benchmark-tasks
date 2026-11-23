"""CPU checks for bounded robot feedback and refusal to advance after failed seating."""

import json
from pathlib import Path
import unittest

import numpy as np

from robot_control import PressController, PressLiftController
from yam_arena import ik
from yam_arena.constants import LEFT_BASE_POS, READY


class RobotControlTests(unittest.TestCase):
    def make_controller(self, cls):
        cfg=json.loads((Path(__file__).parent/'parameters.json').read_text())
        tcp=ik.tcp_pose(READY,LEFT_BASE_POS)[0]
        return cls(READY,tcp,cfg,'press'),tcp

    def test_late_seating_delays_withdrawal_within_stock_action_limits(self):
        controller,tcp=self.make_controller(PressController)
        withdrew=False
        for time_s in np.arange(0,22,.1):
            poses={'male':[.30,.12,.9078 if time_s>=11.3 else .915,0,0,0,1],
                   'socket':[.30,.12,.901,0,0,0,1]}
            action,info=controller.action(time_s,tcp,None,poses)
            self.assertEqual(action.shape,(14,))
            self.assertTrue(np.isfinite(action).all())
            self.assertTrue(np.all(action[:6]>=ik.LOWER) and np.all(action[:6]<=ik.UPPER))
            if info['phase']=='withdraw':
                self.assertGreaterEqual(time_s,11.3)
                withdrew=True
            tcp=ik.tcp_pose(action[:6],LEFT_BASE_POS)[0]
            if info.get('completed'):
                self.assertTrue(withdrew)
                return
        self.fail('Bounded press controller did not finish')

    def test_unseated_pair_never_advances_to_cloth_lift(self):
        controller,tcp=self.make_controller(PressLiftController)
        for time_s in np.arange(0,22,.1):
            poses={'male':[.30,.12,.915,0,0,0,1],'socket':[.30,.12,.901,0,0,0,1]}
            action,info=controller.action(time_s,tcp,None,poses)
            self.assertNotIn(info['phase'],('change_to_cloth_grasp','close','lift','hold'))
            tcp=ik.tcp_pose(action[:6],LEFT_BASE_POS)[0]
            if info.get('abort'):
                self.assertEqual(info['phase'],'press_failed')
                return
        self.fail('Failed snap did not terminate before cloth lifting')

    def test_missed_upper_tab_aborts_before_carry(self):
        controller,tcp=self.make_controller(PressLiftController)
        poses={'male':[.30,.12,.9075,0,0,0,1],'socket':[.30,.12,.901,0,0,0,1]}
        controller.transition_at=0.
        controller.lift_baseline={k:v[2] for k,v in poses.items()}
        action,info=controller.action(11.1,tcp,np.array([[.27,.12,.895]]),poses)
        self.assertTrue(info['abort'])
        self.assertEqual(info['abort_reason'],'upper_tab_grasp_failed')
        self.assertFalse(info['completed'])
        self.assertEqual(info['phase'],'hold')
        self.assertEqual(action.shape,(14,))

    def test_long_tab_path_respects_stock_kinematics(self):
        from robot_control import SideCouponController, LONG_SIDE_ROT, SIDE_HOVER
        cfg=json.loads((Path(__file__).parent/'parameters.json').read_text())
        cfg['robot_test'].update(grasp_mode='side_long', support_top_m=.9)
        q,error,angle=ik.solve(SIDE_HOVER,ik.matrix_to_quat_xyzw(LONG_SIDE_ROT),READY,
                               base_pos=LEFT_BASE_POS,iters=300)
        tcp=ik.tcp_pose(q,LEFT_BASE_POS)[0]
        controller=SideCouponController(q,tcp,cfg,'grasp')
        front=np.array([[.282,.119,.804],[.284,.121,.806]])
        for t in np.arange(0,19.1,.1):
            # Synthetic observations check reachability only, not physics success.
            hardware=[.30,.12,max(.907,tcp[2]-.10),0,0,0,1]
            action,info=controller.action(t,tcp,front,hardware)
            self.assertFalse(info.get('abort'),info)
            self.assertEqual(action.shape,(14,))
            self.assertTrue(np.all(action[:6]>=ik.LOWER) and np.all(action[:6]<=ik.UPPER))
            self.assertLess(info['ik_error_m'],.0002)
            tcp=ik.tcp_pose(action[:6],LEFT_BASE_POS)[0]


if __name__=='__main__':
    unittest.main()
