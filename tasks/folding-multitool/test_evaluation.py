"""CPU contract tests. These do not replace the scheduled live interface check."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from inspect_robots import Observation, Scene, EmbodimentInfo, Box, ActionSemantics, ObservationSpace, CameraSpec
from episode_config import INSTRUCTION, CAMERAS, WIDTH, HEIGHT, CONTROL_HZ
from evaluation import DIM_LABELS, make_policy, observation, vector


def sample(step=0, state=None):
    images = {name: np.full((HEIGHT, WIDTH, 3), 30+20*i, np.uint8) for i, name in enumerate(CAMERAS)}
    return Observation(images=images,
        state={"joint_pos": np.array([0., 1., 1., -.6, 0., 0., 1.]*2) if state is None else state,
               "privileged_object_pose": np.ones(7)},
        extra={"contacts": "must never reach model"}, instruction="discard this untrusted override",
        state_time=step/CONTROL_HZ, image_times=dict.fromkeys(CAMERAS, step/CONTROL_HZ))


def info():
    return EmbodimentInfo(name="yam", control_hz=20,
        action_space=Box(shape=(14,), low=np.full(14, -4.), high=np.full(14, 4.),
                         semantics=ActionSemantics(control_mode="joint_pos", gripper="continuous", dim_labels=DIM_LABELS)),
        observation_space=ObservationSpace(cameras=tuple(CameraSpec(n, HEIGHT, WIDTH) for n in CAMERAS)))


class EvaluationTests(unittest.TestCase):
    def test_privileged_state_is_discarded_and_images_are_copied(self):
        raw = sample()
        safe = observation(raw, 0)
        self.assertEqual(set(safe.state), {"joint_pos"})
        self.assertEqual(safe.extra, {})
        self.assertEqual(safe.instruction, INSTRUCTION)
        self.assertEqual(tuple(safe.images), CAMERAS)
        safe.images["top_cam"][0, 0] = 255
        self.assertFalse(np.array_equal(safe.images["top_cam"], raw.images["top_cam"]))

    def test_bad_shape_finite_and_time_contracts_are_rejected(self):
        for values in (np.ones(13), np.ones((1, 14)), [np.nan]*14):
            with self.assertRaises(ValueError):
                vector(values)
        with self.assertRaises(ValueError):
            observation(sample(), 1)
        raw = sample()
        raw.images["right_cam"] = np.zeros((176, 320, 4), np.uint8)
        with self.assertRaises(ValueError):
            observation(raw, 0)

    def test_installed_policy_and_session_wire_and_chunk_order_without_network(self):
        with tempfile.TemporaryDirectory() as tmp, patch("dream.DreamClient", side_effect=AssertionError("network forbidden")):
            policy, fixture = make_policy(tmp, offline=True)
            policy.bind(info())
            policy.reset(Scene(id="offline", instruction=INSTRUCTION))
            state = sample().state["joint_pos"].copy()
            rows = []
            for k in range(128):
                chunk = policy.act(observation(sample(k, state), k))
                self.assertEqual(chunk.control_hz, 20)
                self.assertEqual(len(chunk.actions), 1)
                state = vector(chunk.actions[0].data)
                rows.append(state)
            transcript = policy.transcript()
            policy.close()
            self.assertEqual([r["step"] for r in transcript["requests"]], [32, 80])
            np.testing.assert_allclose(rows[:32], np.tile(sample().state["joint_pos"], (32, 1)))
            self.assertAlmostEqual(rows[32][0], .10)
            self.assertAlmostEqual(rows[32][7], -.07)
            np.testing.assert_allclose(rows[32][[6, 13]], [.60, .80])
            np.testing.assert_allclose(rows[80][[6, 13]], [.85, .55])
            self.assertEqual(fixture.calls, 2)
            wire = [json.loads(l) for l in (Path(tmp)/"wire.jsonl").read_text().splitlines()]
            obs = [r for r in wire if r["event"] == "observation"]
            self.assertEqual(len(obs), 128)
            self.assertEqual(obs[80]["timestamp"], 4.)
            # This is measured feedback from the previous tick, not the next target.
            self.assertEqual(obs[80]["state"]["state.arm_l.gripper_pos"], [.60])
            self.assertEqual(obs[80]["state"]["state.arm_r.gripper_pos"], [.80])
            self.assertEqual(len(list((Path(tmp)/"frames").glob("*.jpg"))), 384)
            self.assertNotIn("contacts", json.dumps(wire))
            self.assertNotIn("privileged", json.dumps(wire))


if __name__ == "__main__":
    unittest.main()
