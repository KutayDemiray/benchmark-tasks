"""Version 2: verify installation while one ordinary gripper holds only the handle."""

from dataclasses import asdict, dataclass, replace
import math

from outcome_native import cloth_clearance, prepare_swept_geometry, shape_clearance, swept_shapes
from score_logic import OutcomeScorer

REVISION = 'handle-held-v2'


@dataclass(frozen=True)
class HandleSupport:
    """Conservative geometry measured over every native step of one control interval."""
    side: str | None = None
    non_handle_robot_clearance_m: float = 0.
    other_robot_clearance_m: float = 0.
    finger_handle_clearance_m: tuple[float, float] = (1., 1.)

    def valid(self):
        return (self.side in ('left', 'right')
                and self.non_handle_robot_clearance_m >= .002
                and self.other_robot_clearance_m >= .02
                and max(self.finger_handle_clearance_m) < .002)


class HandleHeldScorer:
    """Retain the original pocket and functional checks with an explicit new endpoint."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.base = OutcomeScorer()
        self.side = None

    def update(self, observation, support):
        values = (support.non_handle_robot_clearance_m, support.other_robot_clearance_m,
                  *support.finger_handle_clearance_m)
        finite = all(math.isfinite(value) for value in values)
        exclusive = finite and support.valid()
        side_changed = self.side is not None and self.side != support.side
        self.side = support.side if exclusive else None
        lifted = observation.all_cloth_clearance_m >= .1
        eligible = exclusive and lifted and not side_changed
        # Invalid support resets the ORIGINAL lift/translation/return history.
        # Pocket partial credit and all numerical invalidity remain unchanged.
        checked = replace(observation, finite=observation.finite and finite,
            wings_externally_supported=observation.wings_externally_supported or not eligible)
        result = self.base.update(checked)
        success = eligible and result['retention_verified'] and not result['invalid_reasons']
        result.update(score=1. if success else min(result['score'], .5), success=bool(success),
            termination_reason='invalid' if result['invalid_reasons'] else 'success' if success else 'running',
            rubric_revision=REVISION, handle_only_support_valid=exclusive,
            holding_side=support.side if exclusive else None,
            returned_hold_seconds=self.base.return_hold_s)
        result.pop('released_hold_seconds')
        return result


class HandleSupportObserver:
    """Certify exclusive handle support from actual swept robot/frame geometry."""

    def __init__(self, frame, frame_names, robots, faces):
        self.frame = prepare_swept_geometry(frame['colliders'])
        self.frame_names = frame_names['body_names']
        self.handle = [shape for shape in self.frame if shape['body'] == 'Handle']
        self.robots = robots
        self.robot_shapes = {side: prepare_swept_geometry(robots[side]['colliders'])
                             for side in ('left', 'right')}
        self.fingers = {side: {name: [shape for shape in self.robot_shapes[side] if shape['body'] == name]
                              for name in ('tip_left', 'tip_right')} for side in ('left', 'right')}
        assert self.handle and all(self.fingers[side][name] for side in self.fingers for name in self.fingers[side])
        self.faces = faces

    def measure(self, state):
        frame = swept_shapes(self.frame, state['frame_body_poses'], self.frame_names)
        handle = [shape for shape in frame if shape['body'] == 'Handle']
        non_handle = [shape for shape in frame if shape['body'] != 'Handle']
        robots = {side: swept_shapes(self.robot_shapes[side], state[side+'_body_poses'],
                                     self.robots[side]['body_names']) for side in ('left', 'right')}
        all_robots = robots['left'] + robots['right']
        non_handle_gap = shape_clearance(non_handle, all_robots, .002)
        details = {}
        candidates = []
        for side, other in (('left', 'right'), ('right', 'left')):
            finger_gaps = []
            for name in ('tip_left', 'tip_right'):
                fingers = [shape for shape in robots[side] if shape['body'] == name]
                finger_gaps.append(shape_clearance(handle, fingers, .002))
            other_gap = min(shape_clearance(frame, robots[other], .02),
                            cloth_clearance(state['positions'], self.faces, robots[other]))
            if non_handle_gap >= .002 and other_gap >= .02 and max(finger_gaps) < .002:
                # Swept proximity alone can be satisfied by a single passing
                # contact. Require both pads to remain possible contacts at
                # EACH native state, including the interior of the interval.
                for index in range(len(state['frame_body_poses'])):
                    native_handle = swept_shapes(self.handle,
                        state['frame_body_poses'][index:index + 1], self.frame_names)
                    for finger_index, name in enumerate(('tip_left', 'tip_right')):
                        native_finger = swept_shapes(self.fingers[side][name],
                            state[side + '_body_poses'][index:index + 1], self.robots[side]['body_names'])
                        finger_gaps[finger_index] = max(finger_gaps[finger_index],
                            shape_clearance(native_handle, native_finger, .002))
            candidate = HandleSupport(side, non_handle_gap, other_gap, tuple(finger_gaps))
            details[side] = asdict(candidate)
            if candidate.valid():
                candidates.append(candidate)
        # Two simultaneous candidate holders cannot satisfy the free-arm clearance.
        assert len(candidates) <= 1
        selected = candidates[0] if candidates else HandleSupport(None, non_handle_gap)
        return selected, details
