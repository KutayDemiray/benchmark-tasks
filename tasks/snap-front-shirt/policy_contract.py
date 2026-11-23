"""Local policy boundary: measured RGB/proprioception and ordinary 14D targets."""

import numpy as np

CAMERAS = ('top_cam','left_cam','right_cam')
INSTRUCTION = ('Fasten the three upper front snaps of the shirt on the table. '
               'Withdraw both arms at least 3 cm and leave the shirt fastened for 3 seconds.')


def validate_action(data):
    """Validate shape and finiteness without changing stock clipping behavior."""
    value = np.asarray(data,dtype=np.float64)
    if value.shape != (14,) or not np.isfinite(value).all():
        raise ValueError('Action must contain 14 finite absolute joint/gripper targets')
    return value.copy()


def joint_bounds():
    from yam_arena.ik import LOWER, UPPER
    return np.r_[LOWER,0.,LOWER,0.],np.r_[UPPER,1.,UPPER,1.]


def observation(images, joints, timestamp, instruction=INSTRUCTION):
    """Construct a policy observation with an explicit whitelist of public fields."""
    import cv2
    from inspect_robots.types import Observation
    if set(images) != set(CAMERAS):
        raise ValueError('All three stock cameras are required')
    state = validate_action(joints)
    if not np.isfinite(timestamp) or timestamp < 0:
        raise ValueError('Invalid observation timestamp')
    rgb = {}
    for name in CAMERAS:
        pixels = np.asarray(images[name])
        if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError('Policy images must be HWC RGB uint8')
        rgb[name] = np.ascontiguousarray(cv2.resize(pixels,(320,176)))
    return Observation(images=rgb,state={'joint_pos':state},instruction=instruction,
                       image_times={name:float(timestamp) for name in CAMERAS},
                       state_time=float(timestamp),extra={})


def embodiment_info(control_hz=20, revision=None):
    from inspect_robots.embodiment import EmbodimentInfo,RESETTABLE,SEEDABLE,PRIVILEGED_SUCCESS
    from inspect_robots_yam.config import action_box,observation_space,DEFAULT_CAMERAS
    assert control_hz in (10,20)
    low,high = joint_bounds()
    return EmbodimentInfo(name='snap_front_shirt',action_space=action_box(low,high),
        observation_space=observation_space(176,320,DEFAULT_CAMERAS),control_hz=float(control_hz),
        is_simulated=True,capabilities=frozenset({RESETTABLE,SEEDABLE,PRIVILEGED_SUCCESS}),
        supported_target_kinds=frozenset(),environment_id='snap-front-shirt-tabletop',
        environment_revision=revision,docs=INSTRUCTION)
