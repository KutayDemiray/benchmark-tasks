"""CPU regression checks for SDK qualification and stock target semantics."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from sdk_yam import SDKYam
from standalone_grasp import validate_source


class SDKContractTests(unittest.TestCase):
    def test_source_reference_uses_run_name_after_authoring_function_checks(self):
        source_code = Path(__file__).resolve().parent
        keys = ('physics_hz', 'iterations', 'spacing', 'seconds', 'repeats', 'contact_updates',
                'contact_subiterations', 'ccd', 'product_profile', 'mouth_height', 'table_origin',
                'grasp_margin_clearance', 'grasp_opening', 'close_seconds', 'placement', 'two_hands',
                'placement_tilt', 'release_opening', 'left_grasp_height', 'right_grasp_height', 'seed')
        request = {key: None for key in keys}
        request.update(grasp_sdk_source='qualified-grasp', grasp_sdk_reference='reference', experiment='robot_grasp')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run, reference = root/'qualified-grasp', root/'reference'
            (run/'source').mkdir(parents=True)
            reference.mkdir()
            for name in ('native.py', 'mesh.py', 'parameters.json', 'robot_gate.py'):
                (run/'source'/name).write_bytes((source_code/name).read_bytes())
            (run/'request.json').write_text(json.dumps(request))
            (run/'grasp-audit.json').write_text('{"grasp_gate_passed": true}')
            (root/'product-gate.json').write_text('{"passed": true, "grasp_run": "qualified-grasp"}')
            (reference/'request.json').write_text(json.dumps({**request, 'cooked_collider_probe': True, 'state_run': run.name}))
            (reference/'result.json').write_text('{"status": "completed"}')
            (reference/'process.json').write_text('{"returncode": 0}')
            (reference/'native-yam-contract.json').write_text('{}')
            self.assertEqual(validate_source(request, root), run)
            wrong = json.loads((reference/'request.json').read_text())
            wrong['state_run'] = 'another-grasp'
            (reference/'request.json').write_text(json.dumps(wrong))
            with self.assertRaises(AssertionError):
                validate_source(request, root)

    def test_target_execution_matches_stock_clamping_without_command_slew_filter(self):
        from yam_arena import constants
        from evaluation_contract import process_action
        class Target:
            shape = (1, 8)
            def write(self, value):
                self.value = value.copy()
        robot = SDKYam.__new__(SDKYam)
        robot.constants = constants
        robot.action = np.r_[constants.READY, 1., constants.READY, 1.]
        robot.arms = [dict(target=Target(), arm_ids=list(range(6)), finger_ids=[6, 7]) for _ in range(2)]
        raw = np.array([3., -9., .4, .5, .6, .7, -1., -3., 9., -.4, -.5, -.6, -.7, 2.])
        robot.command(raw)
        expected = process_action(raw)
        np.testing.assert_array_equal(robot.action, expected)
        for offset, arm in zip((0, 7), robot.arms):
            np.testing.assert_allclose(arm['target'].value[0, :6], expected[offset:offset+6])
            np.testing.assert_allclose(arm['target'].value[0, 6:], expected[offset+6]*constants.FINGER_OPEN)


if __name__ == '__main__':
    unittest.main()
