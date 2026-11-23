"""Public Dream/YAM contract shared by the task's script and policy recorder."""

from task_paths import ENV_FILE

import hashlib
import json

import numpy as np

CONTROL_HZ = 20
ACTION_BUDGET = 2400
CAMERAS = ('top_cam', 'left_cam', 'right_cam')
WIDTH, HEIGHT = 320, 176
LABELS = tuple(f'{side}_{part}' for side in ('left', 'right')
               for part in (*(f'j{i}' for i in range(6)), 'gripper'))


def action_bounds():
    from yam_arena import ik
    return np.tile(np.r_[ik.LOWER, 0.], 2), np.tile(np.r_[ik.UPPER, 1.], 2)


def vector(value):
    result = np.asarray(value, dtype=np.float64)
    assert result.shape == (14,) and np.isfinite(result).all(), 'Expected a finite 14D YAM vector'
    return result.copy()


def process_action(value):
    """Match installed YamArmAction/YamGripperAction position clamping exactly."""
    value = vector(value)
    lower, upper = action_bounds()
    return np.clip(value, lower, upper)


def public_observation(images, measured, instruction, control_step):
    """Create the complete allowlisted policy input, excluding diagnostic fields."""
    from inspect_robots import Observation
    assert set(images) == set(CAMERAS), 'Exactly the three stock RGB cameras are required'
    assert isinstance(instruction, str) and instruction.strip()
    assert isinstance(control_step, int) and 0 <= control_step <= ACTION_BUDGET
    copied = {}
    for name in CAMERAS:
        value = np.asarray(images[name])
        assert value.dtype == np.uint8 and value.shape == (HEIGHT, WIDTH, 3), name
        copied[name] = value.copy()
        copied[name].setflags(write=False)
    state = vector(measured)
    state.setflags(write=False)
    timestamp = control_step / CONTROL_HZ
    return Observation(images=copied, state={'joint_pos': state}, instruction=instruction,
                       image_times={name: timestamp for name in CAMERAS}, state_time=timestamp)


def embodiment_info():
    from inspect_robots import ActionSemantics, Box, CameraSpec, EmbodimentInfo, ObservationSpace
    lower, upper = action_bounds()
    return EmbodimentInfo(name='pocket_mop_cover', control_hz=CONTROL_HZ, is_simulated=True,
        morphology='yam-bimanual', environment_id='pocket-mop-cover',
        action_space=Box(shape=(14,), low=lower, high=upper,
            semantics=ActionSemantics(control_mode='joint_pos', gripper='continuous', dim_labels=LABELS)),
        observation_space=ObservationSpace(cameras=tuple(CameraSpec(name, HEIGHT, WIDTH) for name in CAMERAS),
                                           state_keys=frozenset({'joint_pos'})))


def assessment_paths(request):
    """Select assessment artifacts for the endpoint declared before an episode."""
    mode = request.get('completion_mode', 'released')
    assert mode in ('released', 'handle-held'), 'Unknown completion mode'
    if mode == 'handle-held':
        return 'handle-held-v2', 'handle-held-benchmark-result.json', 'handle-held-outcomes.jsonl'
    return 'released-v1', 'benchmark-result.json', 'outcomes.jsonl'


def validate_task_revision(request, source):
    """Keep the declared instruction, scorer and execution endpoint consistent."""
    revision = assessment_paths(request)[0]
    metadata = json.loads((source / 'task.json').read_text())
    assert metadata.get('rubric_revision', 'released-v1') == revision, 'Task instruction and completion mode differ'
    return revision


def configuration_identity(request):
    """Bind model eligibility to the exact declared physics and task configuration."""
    execution = {'name', 'policy', 'replay_run', 'reset_check', 'policy_check_steps',
                 'demo_run', 'replay_evidence', 'interface_evidence', 'dream_model'}
    configuration = {name: value for name, value in request.items() if name not in execution}
    return hashlib.sha256(json.dumps(configuration, sort_keys=True, allow_nan=False).encode()).hexdigest()


def task_specification(source):
    """Return benchmark-defining text, excluding mutable submission evidence."""
    task = json.loads((source / 'task.json').read_text())
    return {key: task[key] for key in ('instruction', 'setup', 'scoring', 'run_limit', 'reset_notes', 'fidelity')}


def preserved_runtime_paths():
    """Identify installed YAM control, calibration and policy implementation sources."""
    from pathlib import Path
    import importlib.util
    yam = Path(importlib.util.find_spec('yam_arena').origin).parent
    dream = Path(importlib.util.find_spec('inspect_robots_dream').origin).parent
    names = ('constants.py', 'robot.py', 'actions.py', 'embodiment.py', 'station.py', 'scene.py', 'ik.py')
    return [*(yam / name for name in names),
            *(dream / name for name in ('policy.py', 'session.py', '_station.py', 'bridge.py')),
            ENV_FILE]
