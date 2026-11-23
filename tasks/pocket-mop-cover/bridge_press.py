"""Close the passive frame by pressing its existing rigid bridge with a free hand."""

import json

import numpy as np


def bridge_press_target(bridge):
    """Place downward-facing fingertips above the crossbar nearer its center."""
    point = bridge[:3, 3] + bridge[:3, :3] @ np.array([.030, .025, .035])
    # Stock cooked fingertip extent is 12.8425 mm beyond the TCP. Rounding
    # upward leaves at least 1 mm of initial clearance above this rigid rail.
    point[2] += .014
    quarter_turn = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    return point, quarter_turn @ np.diag([-1., 1., -1.])


def press_bridge(session, bridge, read_frame=None):
    """Use bounded ordinary left-wrist pressure, then clear the frame again."""
    from robot_reopen import frame_fold_degrees

    point, rotation = bridge_press_target(bridge)
    session.move('left', point + [0., 0., .06], rotation, 0., 2., 'approach_bridge_press', joint=True)
    session.move('left', point, rotation, 0., 1.5, 'touch_bridge_rail')
    records = []
    depths = [.005 * index for index in range(1, 9)] + [.04 + .0025 * index for index in range(1, 11)]
    for depth in depths:
        current = None if read_frame is None else read_frame()
        angle = None if current is None else frame_fold_degrees(current)
        if angle is not None and angle <= 1.:
            break
        target = point - [0., 0., depth]
        if current is not None:
            # Follow only the measured horizontal rail position. Vertical
            # overtravel remains bounded relative to the initial touch pose.
            target[:2] = bridge_press_target(current['Bridge'])[0][:2]
        # The central contact rolls the head less, but r117 remained 3.48
        # degrees folded at 50 mm overtravel. Continue in small increments
        # to 65 mm with the same stock gains, force limits and fold stop.
        seconds = .4 if depth <= .04 else .3
        session.move('left', target, rotation, 0., seconds, 'press_bridge_rail')
        records.append(dict(extra_command_depth_m=depth, duration_s=seconds, fold_before_deg=angle,
            fold_after_deg=None if read_frame is None else frame_fold_degrees(read_frame())))
    session.hold(.5, 'hold_bridge_press')
    session.move('left', point + [0., 0., .06], rotation, 0., 1.5, 'clear_bridge_press')
    # Vertical clearance alone leaves the closed hand in the subsequent lift
    # path. Move outside the head's swept area before the handle raises it.
    session.move('left', point + [0., .20, .10], rotation, 0., 1.5, 'clear_bridge_workspace')
    (session.out / 'bridge-press.json').write_text(json.dumps(dict(
        physical_contact='Closed stock fingertips on existing Bridge/Crossbar at local [30,25,35] mm; jaws rotated 90 degrees to clear Connector/Cross',
        ordinary_wrist_commands_only=True, unchanged_stock_actuator_limits=True,
        initial_touch_tcp_m=point.tolist(), maximum_extra_depth_m=.065,
        maximum_duration_s=13.2, records=records), indent=2))
