"""CPU reach and conservative robot separation screens, without native physics."""

import argparse
import json
from pathlib import Path
from task_paths import OUTPUT_ROOT

import numpy as np


def separation(first, second):
    """Return a separating-axis lower bound for two enclosing link boxes."""
    _, ca, ra, ha = first
    _, cb, rb, hb = second
    axes = [*ra.T, *rb.T]
    for a in ra.T:
        for b in rb.T:
            axis = np.cross(a,b)
            length = np.linalg.norm(axis)
            if length > 1e-9:
                axes.append(axis/length)
    return max(abs(axis@(cb-ca))-np.abs(axis@ra)@ha-np.abs(axis@rb)@hb for axis in axes)


def arm_boxes(q, base, opening=0.):
    from yam_arena.analysis import fk_link_poses_world
    from yam_arena.constants import ARM_JOINTS,FINGER_JOINTS,FINGER_OPEN
    from tabletop_clearance import stock_collision_bounds
    joints = dict(zip(ARM_JOINTS,q,strict=True))
    joints.update({name:opening*FINGER_OPEN for name in FINGER_JOINTS})
    poses = fk_link_poses_world(joints,base)
    boxes=[]
    for name,bounds in stock_collision_bounds().items():
        if name not in poses:
            continue
        bounds=np.asarray(bounds);pose=poses[name];r=pose[:3,:3]
        boxes.append((name,pose[:3,3]+r@bounds.mean(axis=0),r,(bounds[1]-bounds[0])/2))
    return boxes


def screen():
    from yam_arena import ik
    from yam_arena.constants import LEFT_BASE_POS,RIGHT_BASE_POS,READY
    from yam_arena.demo import DOWN_GRIP
    from yam_arena.urdf import quat_xyzw_to_matrix
    left_q,left_error,left_angle=ik.solve([.28,.12,.768],DOWN_GRIP,READY,base_pos=LEFT_BASE_POS,iters=300)
    left=arm_boxes(left_q,LEFT_BASE_POS)
    records=[]
    for x,y,tilt in ((.24,.18,60),(.24,.20,60),(.27,.20,75)):
        angle=np.deg2rad(tilt)
        rotation=np.array([[1.,0.,0.],[0.,np.cos(angle),-np.sin(angle)],[0.,np.sin(angle),np.cos(angle)]])
        rotation=rotation@quat_xyzw_to_matrix(np.asarray(DOWN_GRIP))
        target=[x,y,.775]
        q,error,orientation=ik.solve(target,ik.matrix_to_quat_xyzw(rotation),READY,base_pos=RIGHT_BASE_POS,iters=300)
        right=arm_boxes(q,RIGHT_BASE_POS)
        gap,link_a,link_b=min((separation(a,b),a[0],b[0]) for a in left for b in right)
        records.append(dict(support_tcp_m=target,support_tilt_deg=tilt,right_joints_rad=q.tolist(),
            left_joints_rad=left_q.tolist(),ik_error_m=error,ik_angle_rad=orientation,
            left_ik_error_m=left_error,left_ik_angle_rad=left_angle,
            interarm_separation_lower_bound_m=gap,closest_links=[link_a,link_b],
            endpoint_screen_passed=bool(max(error,left_error)<.0001 and max(orientation,left_angle)<.001 and gap>=.002)))
    return dict(scope='CPU endpoint screen only; no cloth/table contact or trajectory validation',
        method='Installed stock YAM IK and conservative collision-hull enclosing boxes; unmodified limits',
        limitation='A rejected local IK/box screen does not establish global unreachability or impossibility',
        records=records)


def controller_screen(revision):
    """Sample the full controller path against rigid geometry using synthetic contacts.

    This checks nominal commanded reach and separation. Fabric response and
    dynamic tracking still require native validation and contact guards.
    """
    from bimanual_control import BimanualPressController,PILOTS
    from tabletop_scene import configure
    from yam_arena import ik
    from yam_arena.constants import LEFT_BASE_POS,RIGHT_BASE_POS,READY
    cfg=json.loads((Path(__file__).parent/'parameters.json').read_text());configure(cfg)
    left_q=right_q=np.asarray(READY)
    left_tcp=ik.tcp_pose(left_q,LEFT_BASE_POS)[0]
    right_tcp=ik.tcp_pose(right_q,RIGHT_BASE_POS)[0]
    controller=BimanualPressController(left_q,left_tcp,right_q,right_tcp,cfg,revision)
    h=cfg['tabletop']['table_top_m'];hardware={}
    for i,height in enumerate(cfg['garment']['snap_heights_from_hem_m']):
        x=cfg['tabletop']['hem_x_m']-height
        hardware[f'male{i}']=[x,.12,h+.011,0.,0.,0.,1.]
        hardware[f'socket{i}']=[x,.12,h+.002,0.,0.,0.,1.]
    table=('table',np.array([.605,.12,h-.02]),np.eye(3),np.array([.5,.9,.02]))
    records=[];seated=False
    for index in range(400):
        time_s=index*.1
        if controller.press is not None and controller.press.phase=='press':
            if time_s-6.-controller.press.phase_started>=.6:seated=True
        if seated:hardware['male0'][2]=h+.0085
        force=1. if time_s>=6. and right_tcp[2]<h+.028 else 0.
        state=dict(hardware=hardware,snap_robot_contact_norm_N={'male0':force})
        action,control=controller.action(time_s,left_tcp,right_tcp,state)
        left_q,right_q=action[:6],action[7:13]
        left_tcp=ik.tcp_pose(left_q,LEFT_BASE_POS)[0]
        right_tcp=ik.tcp_pose(right_q,RIGHT_BASE_POS)[0]
        left=arm_boxes(left_q,LEFT_BASE_POS,action[6])
        right=arm_boxes(right_q,RIGHT_BASE_POS,action[13])
        gap,a,b=min((separation(a,b),a[0],b[0]) for a in left for b in right)
        body_gap=min(separation(box,table) for box in left+right if not box[0].startswith('tip_'))
        finger_gap=min(separation(box,table) for box in left+right if box[0].startswith('tip_'))
        records.append(dict(time_s=time_s,phase=control['phase'],support_phase=control['support_phase'],
            interarm_box_gap_m=gap,closest_links=[a,b],nonfinger_table_box_gap_m=body_gap,
            finger_table_box_gap_m=finger_gap,support_ik_error_m=control['support_ik_error_m'],
            pressing_ik_error_m=control.get('ik_error_m',0.),actions=action.tolist()))
        if control['completed']:break
    gaps=np.array([r['interarm_box_gap_m'] for r in records])
    reach=max(max(r['support_ik_error_m'],r['pressing_ik_error_m']) for r in records)
    return dict(scope='CPU nominal commanded paths with synthetic contact; not a native feasibility result',
        revision=revision,settings=PILOTS[revision],completed=controller.completed,
        minimum_interarm_box_gap_m=float(gaps.min()),maximum_ik_error_m=reach,
        screen_passed=bool(controller.completed and gaps.min()>=.002 and reach<=.005),
        limitation='Bounding-box table gaps can reject safe concave hull geometry; native contact guards remain required.',
        records=records)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--controller-revision',type=int,choices=(1,2,3))
    args=parser.parse_args()
    assert args.out.resolve().is_relative_to(OUTPUT_ROOT)
    result=screen() if args.controller_revision is None else controller_screen(args.controller_revision)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    if args.controller_revision is not None:
        print(json.dumps({k:v for k,v in result.items() if k!='records'}))
    else:
        for record in result['records']:
            print(json.dumps({k:record[k] for k in ('support_tcp_m','support_tilt_deg','interarm_separation_lower_bound_m','endpoint_screen_passed')}))


if __name__=='__main__':main()
