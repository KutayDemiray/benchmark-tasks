"""Offline measured-state scoring for isolated robot cloth manipulation."""

import math


def analyze_grasp(control, physics, cfg, topology):
    """Score a robot grasp diagnostic without granting any garment-task credit."""
    result = {'cloth_integrity': False, 'lifted_15mm': False, 'released_after_opening': False,
              'max_cloth_strain': 0., 'max_rivet_error_m': 0., 'invalid_reasons': [],
              'full_task_score': 0., 'full_task_success': False, 'isolated_gate_pass': False}
    try:
        assert control and physics, 'empty_trace'
        last = -math.inf
        held, released = [], []
        for row in control:
            t = float(row['time_s'])
            assert math.isfinite(t) and t > last, 'invalid_control_time'
            last = t
            action = row['robot_action']
            assert len(action) == 14 and all(math.isfinite(float(v)) for v in action), 'invalid_robot_action'
            assert 0 <= action[6] <= 1 and 0 <= action[13] <= 1, 'invalid_gripper_action'
            q = row['joint_positions']
            assert len(q) == 14 and all(math.isfinite(float(v)) for v in q), 'invalid_robot_state'
            z = float(row['physics']['hardware']['male'][2])
            assert math.isfinite(z), 'invalid_hardware_state'
            phase = row['control']['phase']
            if phase == 'hold' and q[6] < .05:
                held.append((t, z))
            if t > 11.5 and phase == 'retreat' and q[6] > .9:
                released.append((t, z))
        previous = -1
        for row in physics:
            assert row['physics_step'] == previous+1, 'missing_physics_steps'
            previous = row['physics_step']
            for pose in row['hardware'].values():
                assert len(pose) == 7 and all(math.isfinite(float(v)) for v in pose), 'invalid_hardware_state'
                assert abs(sum(v*v for v in pose[3:])-1) < .01, 'invalid_quaternion'
            assert row['cloth'], 'missing_fabric'
            for cloth in row['cloth'].values():
                assert cloth['finite'], 'nonfinite_fabric'
                strain, error = float(cloth['strain']), float(cloth['rivet_error_m'])
                assert math.isfinite(strain) and math.isfinite(error) and strain >= 0 and error >= 0, 'invalid_fabric_measurements'
                result['max_cloth_strain'] = max(result['max_cloth_strain'], strain)
                result['max_rivet_error_m'] = max(result['max_rivet_error_m'], error)
        assert result['max_cloth_strain'] <= cfg['gate']['max_edge_strain'], 'cloth_strain_limit_exceeded'
        assert result['max_rivet_error_m'] <= cfg['gate']['max_attachment_error_m'], 'rivet_error_limit_exceeded'
        expected = round(control[-1]['time_s']*cfg['simulation']['physics_hz'])+1
        assert len(physics) == expected, 'incomplete_physics_coverage'
        assert topology['diagnostic_world_mounts'] is False and topology['grasp_attachments'] is False, 'hidden_attachment'
        result['cloth_integrity'] = True
        initial_z = topology['initial_hardware']['male'][2]
        result['lifted_15mm'] = bool(held and held[-1][0]-held[0][0] >= 1.5 and min(z for _, z in held)>initial_z+.015)
        result['released_after_opening'] = bool(result['lifted_15mm'] and released
                                                and released[-1][0]-released[0][0] >= .5
                                                and max(z for _, z in released)<min(z for _, z in held)-.015)
    except (AssertionError, ValueError, TypeError, KeyError, IndexError) as exc:
        result['invalid_reasons'].append(str(exc))
        result['cloth_integrity'] = False
        result['lifted_15mm'] = result['released_after_opening'] = False
    return result


def snap_seated(male_pose, socket_pose, cfg):
    """Check insertion in the moving socket frame, independent of world coordinates."""
    import numpy as np
    def rotation(q):
        x, y, z, w = q
        return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])
    m, s = np.asarray(male_pose), np.asarray(socket_pose)
    if m.shape != (7,) or s.shape != (7,) or not np.isfinite(np.r_[m, s]).all():
        return False
    if abs(float(m[3:]@m[3:])-1)>.01 or abs(float(s[3:]@s[3:])-1)>.01:
        return False
    r = rotation(s[3:])
    relative = r.T@(m[:3]-s[:3])
    normal = float((r.T@rotation(m[3:]))[2, 2])
    p = cfg['snap']
    limit = min(p['ring_height_m']+p['cap_thickness_m']/2+.0007,
                p['socket_ball_height_m']+p['head_from_cap_m']-.0006)
    minimum=.001
    tabletop=cfg.get('tabletop',{})
    if 'seat_rim_penetration_allowance_m' in tabletop:
        minimum=p['ring_height_m']+p['cap_thickness_m']/2-tabletop['seat_rim_penetration_allowance_m']
    return bool(np.linalg.norm(relative[:2]) <= .00075 and minimum < relative[2] < limit
                and normal > math.cos(math.radians(12)))


def analyze_press(control, physics, cfg, topology):
    """Score measured pressing and released seating independently of controller labels."""
    result = {'cloth_integrity': False, 'seated_after_robot_press': False,
              'robot_press_contact_detected': False, 'uncommanded_seating_detected': False,
              'peak_robot_normal_contact_N': 0., 'max_cloth_strain': 0.,
              'max_rivet_error_m': 0., 'invalid_reasons': [],
              'full_task_score': 0., 'full_task_success': False, 'isolated_gate_pass': False}
    try:
        assert control and physics, 'empty_trace'
        assert not topology['diagnostic_world_mounts'] and not topology['grasp_attachments'], 'hidden_attachment'
        assert topology.get('snap_pair_present'), 'missing_snap_pair'
        initial = topology['initial_hardware']
        assert not snap_seated(initial['male'], initial['socket'], cfg), 'initially_seated_pair'
        first_contact = first_seated = None
        for index, row in enumerate(physics):
            assert row['physics_step']==index, 'missing_physics_steps'
            for pose in row['hardware'].values():
                assert len(pose)==7 and all(math.isfinite(float(x)) for x in pose), 'nonfinite_hardware'
                assert abs(sum(v*v for v in pose[3:])-1)<.01, 'invalid_quaternion'
            force = row['male_robot_normal_forces_N']
            assert len(force)==2 and all(len(v)==3 and all(math.isfinite(x) for x in v) for v in force), 'invalid_robot_force'
            link_force = row.get('male_robot_link_normal_forces_N', force)
            assert all(len(v)==3 and all(math.isfinite(x) for x in v) for v in link_force), 'invalid_link_force'
            magnitude = sum(math.sqrt(sum(x*x for x in v)) for v in link_force)
            result['peak_robot_normal_contact_N'] = max(result['peak_robot_normal_contact_N'], magnitude)
            if first_contact is None and sum(v[2] for v in force)<-.05:
                first_contact = index
            poses = row['hardware']
            if first_seated is None and snap_seated(poses['male'], poses['socket'], cfg):
                first_seated = index
            assert row['cloth'], 'missing_fabric'
            for cloth in row['cloth'].values():
                assert cloth['finite'] and math.isfinite(cloth['strain']) and math.isfinite(cloth['rivet_error_m']) and cloth['strain']>=0 and cloth['rivet_error_m']>=0, 'invalid_fabric'
                result['max_cloth_strain'] = max(result['max_cloth_strain'], cloth['strain'])
                result['max_rivet_error_m'] = max(result['max_rivet_error_m'], cloth['rivet_error_m'])
        assert result['max_cloth_strain']<=cfg['gate']['max_edge_strain'], 'cloth_strain_limit_exceeded'
        assert result['max_rivet_error_m']<=cfg['gate']['max_attachment_error_m'], 'rivet_error_limit_exceeded'
        assert result['peak_robot_normal_contact_N']<=cfg['gate']['max_press_N'], 'robot_press_force_limit_exceeded'
        previous = -math.inf
        for row in control:
            t = row['time_s']
            assert math.isfinite(t) and t>previous, 'invalid_control_time'
            previous = t
            assert len(row['robot_action'])==14 and all(math.isfinite(float(x)) for x in row['robot_action']), 'invalid_robot_action'
            assert 0<=row['robot_action'][6]<=1 and 0<=row['robot_action'][13]<=1, 'invalid_gripper_action'
            index = round(t*cfg['simulation']['physics_hz'])
            assert row['physics']==physics[index], 'mismatched_control_and_physics'
        assert len(physics)==round(control[-1]['time_s']*cfg['simulation']['physics_hz'])+1, 'incomplete_physics_coverage'
        result['cloth_integrity'] = True
        result['robot_press_contact_detected'] = first_contact is not None
        result['uncommanded_seating_detected'] = first_seated is not None and (first_contact is None or first_seated<first_contact)
        # Final released outcome, regardless of which controller phase produced it.
        tail = [r for r in control if r['time_s']>=control[-1]['time_s']-1.5-1e-8]
        clear = len(tail)>=2 and tail[-1]['time_s']-tail[0]['time_s']>=1.5-1e-8
        for row in tail:
            clear &= row['measured_tcp_m'][2]-row['physics']['hardware']['male'][2]>.030
        start_index = round(tail[0]['time_s']*cfg['simulation']['physics_hz'])
        retained = True
        for row in physics[start_index:]:
            poses = row['hardware']
            retained &= snap_seated(poses['male'],poses['socket'],cfg)
            retained &= sum(math.sqrt(sum(x*x for x in v)) for v in row['male_robot_normal_forces_N'])<.05
        result['seated_after_robot_press'] = bool(clear and retained and first_contact is not None
            and first_seated is not None and not result['uncommanded_seating_detected'])
    except (AssertionError, ValueError, TypeError, KeyError, IndexError) as exc:
        result['invalid_reasons'].append(str(exc))
        result['cloth_integrity'] = False
        result['seated_after_robot_press'] = False
    return result

def lower_clearance(row, lower_nodes, topology):
    """Conservative lower-coupon and socket distances from both robots at a recorded control state."""
    import numpy as np
    def rotation(q):
        x,y,z,w=np.asarray(q,dtype=np.float64)
        return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
                         [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
                         [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
    assert all(topology['robot_collision_planes'].get(k) for k in ('gripper','tip_left','tip_right')), 'missing_robot_collision_hulls'
    nodes=np.asarray(lower_nodes,dtype=np.float64)
    edges=np.asarray(topology['cloth']['lower']['edges'])
    cover=float(np.linalg.norm(nodes[edges[:,0]]-nodes[edges[:,1]],axis=1).max())
    center=np.asarray(row['physics']['hardware']['socket'][:3],dtype=np.float64)
    cloth_bound=rigid_bound=math.inf
    for side in ('left','right'):
        for name,pose in zip(row[side+'_link_names'],row[side+'_link_poses_xyzw']):
            R=rotation(pose[3:])
            local=(nodes-np.asarray(pose[:3],dtype=np.float64))@R
            cap_local=(center-np.asarray(pose[:3],dtype=np.float64))@R
            for planes in topology['robot_collision_planes'].get(name,[]):
                planes=np.asarray(planes)
                cloth_bound=min(cloth_bound,float(np.max(local@planes[:,:3].T-planes[:,3],axis=1).min())-cover)
                rigid_bound=min(rigid_bound,float(np.max(planes[:,:3]@cap_local-planes[:,3]))-.009)
    return {'fabric_m':cloth_bound,'socket_m':rigid_bound}


def upper_hardware_clearance(row, topology):
    """Bound clearance of an enclosing male-hardware sphere from both robots."""
    import numpy as np
    x = np.asarray(row['physics']['hardware']['male'][:3], dtype=np.float64)
    bound = math.inf
    assert all(topology['robot_collision_planes'].get(k) for k in ('gripper','tip_left','tip_right'))
    for side in ('left','right'):
        for name, pose in zip(row[side+'_link_names'], row[side+'_link_poses_xyzw']):
            qx,qy,qz,qw = np.asarray(pose[3:], dtype=np.float64)
            rotation = np.array([[1-2*(qy*qy+qz*qz),2*(qx*qy-qz*qw),2*(qx*qz+qy*qw)],
                                 [2*(qx*qy+qz*qw),1-2*(qx*qx+qz*qz),2*(qy*qz-qx*qw)],
                                 [2*(qx*qz-qy*qw),2*(qy*qz+qx*qw),1-2*(qx*qx+qy*qy)]])
            local = (x-np.asarray(pose[:3], dtype=np.float64))@rotation
            for planes in topology['robot_collision_planes'].get(name, []):
                planes = np.asarray(planes, dtype=np.float64)
                bound = min(bound, float(np.max(planes[:, :3]@local-planes[:, 3]))-.009)
    return bound


def upper_cloth_clearance(row, upper_nodes, topology):
    """Apply the same conservative face-cover bound to the upper fabric."""
    proxy_topology = {**topology, 'cloth': {**topology['cloth'], 'lower': topology['cloth']['upper']}}
    proxy_row = {**row, 'physics': {**row['physics'], 'hardware': {
        **row['physics']['hardware'], 'socket': row['physics']['hardware']['male']}}}
    return lower_clearance(proxy_row, upper_nodes, proxy_topology)['fabric_m']


def analyze_press_lift(control, physics, cfg, topology):
    """Require the snapped upper tab to carry the lower coupon without direct robot support."""
    result=analyze_press(control,physics,cfg,topology)
    result.update(lifted_pair_by_upper_fabric=False, released_pair_on_support=False)
    try:
        assert result['cloth_integrity'], 'invalid_press_episode'
        held=[r for r in control if r['control']['phase']=='hold']
        baseline=[r for r in control if r['control']['phase']=='change_to_cloth_grasp']
        assert baseline, 'missing_postpress_baseline'
        initial=baseline[0]['physics']['hardware']
        assert held and held[-1]['time_s']-held[0]['time_s']>=1.5, 'insufficient_hold'
        for row in held:
            poses=row['physics']['hardware']
            assert snap_seated(poses['male'],poses['socket'],cfg), 'snap_open_during_lift'
            assert min(poses[k][2]-initial[k][2] for k in ('male','socket'))>.015, 'lower_coupon_not_lifted'
            assert row['joint_positions'][6]<.05, 'gripper_not_closed'
            assert math.isfinite(row.get('upper_hardware_clearance_m', math.nan)), 'missing_upper_hardware_clearance'
            assert row['upper_hardware_clearance_m']>.001, 'upper_hardware_directly_grasped'
            clearance=row['lower_robot_clearance_m']
            assert all(math.isfinite(v) for v in clearance.values()), 'nonfinite_robot_clearance'
            assert min(clearance.values())>.001, 'lower_coupon_directly_supported_by_robot'
        first=round(held[0]['time_s']*cfg['simulation']['physics_hz'])
        last=round(held[-1]['time_s']*cfg['simulation']['physics_hz'])
        for row in physics[first:last+1]:
            assert snap_seated(row['hardware']['male'],row['hardware']['socket'],cfg), 'snap_open_between_actions'
            force=row['socket_robot_normal_forces_N']
            assert sum(math.sqrt(sum(x*x for x in v)) for v in force)<.05, 'robot_contacts_lower_hardware'
        result['lifted_pair_by_upper_fabric']=True
        tail=[r for r in control if r['time_s']>=control[-1]['time_s']-1.5-1e-8]
        h=topology['support_top_m']
        released=result['seated_after_robot_press']
        for row in tail:
            released &= row['joint_positions'][6]>.9
            released &= min(row['lower_robot_clearance_m'].values())>.001
            released &= row.get('upper_hardware_clearance_m', -math.inf)>.001
            released &= row.get('upper_cloth_robot_clearance_m', -math.inf)>.001
            for key in ('male','socket'):
                pose=row['physics']['hardware'][key]
                released &= h-.005 < pose[2] < h+.025
                radius = cfg['snap']['cap_diameter_m']/2
                released &= topology['support_front_x_m']+radius < pose[0] < topology['support_front_x_m']+.4-radius
                cy = topology.get('support_center_y_m', .12)
                half_width = topology.get('support_width_m', .35)/2
                released &= cy-half_width+radius < pose[1] < cy+half_width-radius
        result['released_pair_on_support']=bool(released)
    except (AssertionError,ValueError,KeyError,IndexError,TypeError) as exc:
        result['invalid_reasons'].append(str(exc))
        result['lifted_pair_by_upper_fabric']=False
        result['released_pair_on_support']=False
    return result
