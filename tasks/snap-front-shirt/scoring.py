"""Fail-closed analysis of isolated diagnostics; never a full-task scorer."""

import math


def analyze(records, cfg):
    """Evaluate measured rigid geometry and a sustained diagnostic pull.

    Args:
        records: Ordered measured state records, not controller stage assertions.
        cfg: Frozen physical parameters from the run bundle.

    Returns:
        Diagnostic predicates and zero full-task credit.
    """
    result = {'full_task_score': 0.0, 'full_task_success': False,
              'isolated_gate_pass': False, 'invalid_reasons': [],
              'seated_during_press': False, 'retained_under_1N': False,
              'released_under_limit': False, 'cloth_integrity': False,
              'gate_pending': ['aggregate_functional_assessment', 'ordinary_robot_manipulation'],
              'scope': 'one fixture diagnostic; not a robot or garment outcome'}
    if not records:
        result['invalid_reasons'].append('empty_trace')
        return result
    try:
        previous_time = -math.inf
        max_strain, max_drape, max_speed, max_force, max_rivet, max_torque = 0., 0., 0., 0., 0., 0.
        retained = []
        cloth_seen = False
        seated_limit = cfg['snap'].get('ring_height_m', .003) + cfg['snap']['cap_thickness_m']/2 + .0007
        # The stud sphere must be on the inner side of the socket constriction.
        barrier = cfg['snap']['socket_ball_height_m'] + cfg['snap']['head_from_cap_m']
        for row in records:
            t = float(row['time_s'])
            pose = row['male_pose_xyzw']
            velocity = row['male_velocity']
            wrench = row['diagnostic_force_N'] + row['diagnostic_torque_Nm']
            values = [t, *pose, *velocity, *wrench]
            if not all(math.isfinite(float(v)) for v in values):
                raise ValueError('nonfinite_state')
            if t <= previous_time:
                raise ValueError('nonmonotonic_time')
            previous_time = t
            if abs(sum(float(q)**2 for q in pose[3:]) - 1.) > .01:
                raise ValueError('invalid_quaternion')
            height = float(pose[2]) - .1
            aligned = math.hypot(pose[0], pose[1]) <= .00075
            normal_z = 1 - 2 * (pose[3]**2 + pose[4]**2)
            seated = aligned and normal_z > math.cos(math.radians(12)) and .001 < height < min(seated_limit, barrier - .0006)
            max_speed = max(max_speed, math.sqrt(sum(float(v)**2 for v in velocity[:3])))
            max_force = max(max_force, math.sqrt(sum(float(v)**2 for v in wrench[:3])))
            max_torque = max(max_torque, math.sqrt(sum(float(v)**2 for v in wrench[3:])))
            if row['phase'] == 'press' and seated and row['diagnostic_force_N'][2] < -.05:
                result['seated_during_press'] = True
            if row['phase'] == 'retention':
                retained.append((t, seated and .95 <= row['diagnostic_force_N'][2] <= 1.05))
            if row['phase'] == 'release' and height > .012 and max_force <= cfg['gate']['max_release_N'] + .001:
                result['released_under_limit'] = True
            for cloth in row['cloth'].values():
                cloth_seen = True
                if not cloth['finite'] or not all(math.isfinite(cloth[key]) for key in ('max_abs_edge_strain', 'z_range_m', 'max_rivet_error_m')):
                    raise ValueError('nonfinite_cloth')
                max_strain = max(max_strain, cloth['max_abs_edge_strain'])
                max_drape = max(max_drape, cloth['z_range_m'])
                max_rivet = max(max_rivet, cloth['max_rivet_error_m'])
        # Require the final 0.2 seconds of the fixed 0.5-second pull test.
        if retained:
            tail = [sample for sample in retained if sample[0] >= retained[-1][0] - .2 - 1e-7]
            result['retained_under_1N'] = (result['seated_during_press'] and len(tail) >= 10
                                          and retained[-1][0] - retained[0][0] >= .48
                                          and tail[-1][0] - tail[0][0] >= .19 and all(ok for _, ok in tail))
        if max_force > cfg['gate']['max_press_N'] + .001:
            result['invalid_reasons'].append('diagnostic_force_limit_exceeded')
        if max_torque > cfg['gate']['max_release_torque_Nm'] + 1e-7:
            result['invalid_reasons'].append('diagnostic_torque_limit_exceeded')
        if max_strain > cfg['gate']['max_edge_strain']:
            result['invalid_reasons'].append('cloth_strain_limit_exceeded')
        if max_rivet > cfg['gate']['max_attachment_error_m']:
            result['invalid_reasons'].append('rivet_error_limit_exceeded')
        # Separation of a never-captured stud is not a release test success.
        result['released_under_limit'] = result['released_under_limit'] and result['retained_under_1N']
        result.update(max_cloth_strain=max_strain, max_cloth_z_range_m=max_drape,
                      max_speed_m_s=max_speed, max_fixture_force_N=max_force,
                      max_fixture_torque_Nm=max_torque,
                      max_rivet_error_m=max_rivet,
                      cloth_integrity=(cloth_seen and max_strain <= cfg['gate']['max_edge_strain']
                                       and max_rivet <= cfg['gate']['max_attachment_error_m']),
                      fabric_deformed=cloth_seen and max_drape >= cfg['gate']['min_visible_drape_m'])
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        result['invalid_reasons'].append(str(exc))
    if result['invalid_reasons']:
        result['cloth_integrity'] = False
        result['seated_during_press'] = False
        result['retained_under_1N'] = False
        result['released_under_limit'] = False
    return result


def analyze_grasp(records):
    """Measure load lifting and opening/slip in the pinch fixture."""
    raised = [r['male_pose_xyzw'][2] for r in records if r['phase'] == 'pinch_hold_raised']
    released = [r['male_pose_xyzw'][2] for r in records if r['phase'] == 'pinch_open' and r['time_s'] > 2.7]
    return {'pinch_lifted_15mm': bool(raised and min(raised[-10:]) > .115),
            'fell_after_opening': bool(released and max(released[-10:]) < .09),
            'raised_height_min_m': min(raised[-10:], default=None),
            'released_height_max_m': max(released[-10:], default=None),
            'robot_grasp_validated': False}
