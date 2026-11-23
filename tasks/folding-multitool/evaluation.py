"""Restricted Dream boundary and an offline test of the installed Dream adapter.

The offline client never constructs a DreamClient or accesses the network. It
exercises the installed DreamPolicy AND DreamSession using asymmetric arm and
gripper commands. It is an interface test, never a model evaluation.
"""
import base64
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from inspect_robots import Observation
from episode_config import CAMERAS, WIDTH, HEIGHT, INSTRUCTION, CONTROL_HZ, MODEL

DIM_LABELS = tuple(f"{side}_{part}" for side in ("left", "right")
                   for part in (*(f"j{i}" for i in range(6)), "gripper"))
WIRE_STATE_KEYS = {"prompt", "robot.arm", "robot.exec_head", "robot.max_joint_vel_rad_s",
                   "robot.gripper_open", "state.arm_l.joint_pos", "state.arm_l.gripper_pos",
                   "state.arm_r.joint_pos", "state.arm_r.gripper_pos"}
WIRE_CAMERAS = {"image_window.top", "image_window.left", "image_window.right"}


def vector(values):
    a = np.asarray(values, dtype=np.float64)
    if a.shape != (14,) or not np.isfinite(a).all():
        raise ValueError("Expected exactly 14 finite joint/gripper values")
    return a.copy()


def observation(raw, step):
    """Build a fresh allowlisted observation; discard all privileged extras."""
    images = {}
    t = step / CONTROL_HZ
    if not np.isclose(raw.state_time, t, atol=1e-7):
        raise ValueError("Measured state timestamp is not the 20 Hz episode clock")
    for name in CAMERAS:
        frame = np.asarray(raw.images[name])
        if frame.shape != (HEIGHT, WIDTH, 3) or frame.dtype != np.uint8:
            raise ValueError(f"{name}: expected RGB uint8 {WIDTH}x{HEIGHT}")
        if not np.isclose(raw.image_times[name], t, atol=1e-7):
            raise ValueError(f"{name}: stale camera timestamp")
        images[name] = np.ascontiguousarray(frame).copy()
    state = vector(raw.state["joint_pos"])
    if (state[[6, 13]] < 0).any() or (state[[6, 13]] > 1).any():
        raise ValueError("Measured gripper opening is not normalized to 0..1")
    return Observation(images=images, state={"joint_pos": state}, instruction=INSTRUCTION,
                       image_times=dict.fromkeys(CAMERAS, t), state_time=t, extra={})


def verify_measured_state(obs, measurement):
    """Diagnostics only: independently reconstruct ordering from native names."""
    from yam_arena.constants import ARM_JOINTS, FINGER_JOINTS, FINGER_OPEN
    expected = []
    for side in ("left", "right"):
        joint_map = dict(zip(measurement[side+"_joint_names"], measurement[side+"_joint_positions"]))
        expected.extend(joint_map[name] for name in ARM_JOINTS)
        expected.append(float(np.clip(np.mean([joint_map[n] for n in FINGER_JOINTS])/FINGER_OPEN, 0., 1.)))
    np.testing.assert_allclose(obs.state["joint_pos"], expected, atol=1e-6, rtol=0)


def verify_action_execution(emb, applied):
    """Read actual native action terms, independently of the requested vector."""
    manager = emb._base.action_manager
    names = ("left_arm", "left_gripper", "right_arm", "right_gripper")
    if tuple(manager.active_terms) != names:
        raise ValueError("Native action terms are not in the declared 14D order")
    actual = np.concatenate([manager.get_term(n).processed_actions[0].detach().cpu().numpy()
                             for n in names]).astype(float)
    np.testing.assert_allclose(actual, applied, atol=1e-6, rtol=0)
    return actual


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


class LoggedClient:
    """Capture the precise station inputs/replies, without credentials or headers."""
    def __init__(self, client, directory):
        self.client = client
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.frames = self.directory / "frames"
        self.frames.mkdir(exist_ok=True)
        self.log = (self.directory / "wire.jsonl").open("w")
        self.tick = 0

    def write(self, kind, **data):
        self.log.write(json.dumps({"event": kind, **data}, default=json_value)+"\n")
        self.log.flush()

    @property
    def last_timing(self):
        return self.client.last_timing

    @property
    def session_id(self):
        return self.client.session_id

    def get_models(self):
        result = self.client.get_models()
        self.write("models", ids=[m.id for m in result])
        return result

    def policy(self, config):
        self.write("policy", config=config)
        return self.client.policy(config)

    def reset_episode(self):
        self.write("reset_episode")
        self.tick = 0
        return self.client.reset_episode()

    def set_observation(self, frames, state, *, timestamp=None):
        if set(frames) != WIRE_CAMERAS or set(state) != WIRE_STATE_KEYS:
            raise ValueError("Unexpected data in the Dream wire observation")
        if state["prompt"] != INSTRUCTION:
            raise ValueError("Dream instruction differs from the shared episode instruction")
        files = {}
        for name, encoded in frames.items():
            data = base64.b64decode(encoded, validate=True)
            with Image.open(io.BytesIO(data)) as im:
                if im.size != (WIDTH, HEIGHT) or im.mode != "RGB":
                    raise ValueError("Dream JPEG has wrong dimensions or colour mode")
            path = self.frames / f"{self.tick:04d}-{name.removeprefix('image_window.')}.jpg"
            path.write_bytes(data)
            files[name] = {"path": str(path.relative_to(self.directory)),
                           "sha256": hashlib.sha256(data).hexdigest()}
        self.write("observation", tick=self.tick, timestamp=timestamp, state=state, frames=files)
        self.tick += 1
        return self.client.set_observation(frames, state, timestamp=timestamp)

    def get_action(self, *, commanded=None):
        self.write("request", after_tick=self.tick-1, commanded=commanded)
        try:
            reply = self.client.get_action(commanded=commanded)
        except Exception as exc:
            # No headers, credentials, or provider exception bodies in this log.
            self.write("request_error", error_type=type(exc).__name__)
            raise
        self.write("reply", reply=reply, timing=self.last_timing, session_id=self.session_id)
        return reply

    def close(self):
        try:
            self.client.close()
        finally:
            if not self.log.closed:
                self.log.close()


class OfflineClient:
    """No-network fixture at the actual DreamClient protocol boundary."""
    last_timing = {"serve_infer_ms": 0.0}
    session_id = "OFFLINE-INTERFACE-TEST-NOT-A-MODEL"

    def __init__(self):
        self.calls = 0
        self.observed = 0
        self.initial = None
        self.target = None

    def get_models(self):
        return [SimpleNamespace(id=MODEL)]

    def policy(self, config):
        assert config["checkpoint"] == MODEL and config["_record"] is False
        assert config["obs_window"] == 9 and config["obs_fps"] == 5
        self.reset_episode()

    def reset_episode(self):
        self.calls = self.observed = 0
        self.initial = None

    def set_observation(self, frames, state, *, timestamp=None):
        assert set(frames) == WIRE_CAMERAS and set(state) == WIRE_STATE_KEYS
        assert np.isclose(timestamp, self.observed/CONTROL_HZ)
        self.measured = vector([*state["state.arm_l.joint_pos"], *state["state.arm_l.gripper_pos"],
                                *state["state.arm_r.joint_pos"], *state["state.arm_r.gripper_pos"]])
        if self.initial is None:
            self.initial = self.measured.copy()
        self.observed += 1

    def get_action(self, *, commanded=None):
        self.calls += 1
        # Distinct signs/magnitudes expose arm swaps and gripper indexing errors.
        row = self.initial.copy()
        sign = 1 if self.calls % 2 else -1
        row[0] += sign*.10
        row[7] -= sign*.07
        row[[6, 13]] = [.60, .80] if sign == 1 else [.85, .55]
        self.target = row.copy()
        return {"q_l_chunk": np.tile(row[:7], (48, 1)).tolist(),
                "q_r_chunk": np.tile(row[7:], (48, 1)).tolist()}

    def close(self):
        pass


def make_policy(directory, offline=False):
    from inspect_robots_dream.policy import DreamPolicy
    from inspect_robots_dream.session import DreamSession, _dream_client
    fixture = OfflineClient() if offline else None
    session = DreamSession(make_client=lambda: LoggedClient(fixture or _dream_client(), directory))
    return DreamPolicy(model=MODEL, session=session, record=False, dry_run=False,
                       rows_per_request=48, timeout_s=120.), fixture
