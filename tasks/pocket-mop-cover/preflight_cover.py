"""Check full-cover command reach and stock speed limits without running physics."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from robot_reopen import guide_and_place,handle_guidance_target,transform
from robot_session import RobotSession


class KinematicSession(RobotSession):
    """Use the actual motion interpolation while replacing physics with limit checks."""

    def __init__(self,action,shift,out,center):
        from yam_arena.constants import SHOULDER,WRIST,GRIPPER,FINGER_OPEN
        self.action=action.copy();self.shift=shift;self.out=out;self.cover_center=center
        self.plan=[];self.steps=0;self.phase=None
        self.limits=np.tile([SHOULDER['velocity']/20]*3+[WRIST['velocity']/20]*3+
                           [GRIPPER['velocity']/abs(FINGER_OPEN)/20],2)

    def step(self,target,phase):
        from yam_arena import ik
        self.phase=phase
        ratio=abs(target-self.action)/self.limits
        assert np.all(ratio<=1.001), f'Joint speed limit: index {ratio.argmax()}, ratio {ratio.max():.3f}'
        for offset in (0,7):
            assert np.all(target[offset:offset+6]>=ik.LOWER) and np.all(target[offset:offset+6]<=ik.UPPER)
            assert 0<=target[offset+6]<=1
        self.action=target.copy();self.steps+=1


def preflight(source,out,centers,lifts):
    """Screen proposed cover locations using an actual recorded rail grasp."""
    from station_coordinates import TRANSLATION_S_W
    from yam_arena import ik,station
    request=json.loads((source/'request.json').read_text())
    shift=TRANSLATION_S_W if request['table_origin'] else np.zeros(3)
    names=json.loads((source/'frame-names.json').read_text())
    rows=[json.loads(line) for line in (source/'steps.jsonl').read_text().splitlines()]
    indices=[i for i,row in enumerate(rows) if row['phase']=='verify_inner_rail_grasp']
    index=indices[-1];row=rows[index]
    physics_index=(index+1)*(request['physics_hz']//20)-1
    with np.load(source/'physics-chunks'/f'{physics_index//request["physics_hz"]:05d}.npz') as chunk:
        poses=chunk['frame_body_poses'][physics_index%request['physics_hz']]
        joints=chunk['frame_joint_positions'][physics_index%request['physics_hz']]
    B=transform(poses[names['body_names'].index('Bridge')])
    L=transform(poses[names['body_names'].index('Left')])
    wrists=[]
    for side,offset in (('left',0),('right',7)):
        position,rotation=ik.tcp_pose(np.array(row['measured_joint_pos'])[offset:offset+6],station.BASES[side])
        value=np.eye(4);value[:3,3]=position+shift;value[:3,:3]=rotation;wrists.append(value)
    LT=np.linalg.inv(L)@wrists[0];BT=np.linalg.inv(B)@wrists[1]
    angle=-float(joints[names['joint_names'].index('HingeLeft')])
    rail_y=json.loads((source/'rail-grasp-plan.json').read_text())['rail_y_m']
    results=[]
    for center in centers:
        for lift in lifts:
            folder=out/f'candidate-{len(results):02d}';folder.mkdir(parents=True,exist_ok=False)
            center_array=None if center is None else np.array(center)
            session=KinematicSession(np.array(row['robot_action']),shift,folder,center_array)
            report=dict(center_m=center,lift_m=lift)
            try:
                right_target=BT
                handle_grasp=None
                if center is not None:
                    H=transform(poses[names['body_names'].index('Handle')])
                    handle_grasp=np.linalg.inv(H)@wrists[1]
                    right_target=handle_guidance_target(B,H,wrists[1],
                        float(joints[names['joint_names'].index('HandleY')]))
                guide_and_place(session,B,LT,right_target,angle,rail_y,lift,shift,handle_grasp=handle_grasp)
                report['passed']=True
                if center is not None:
                    heights=[]
                    for target in session.plan:
                        if target['phase']!='guide_frame_open' or target['side']!='left':continue
                        wrist=np.eye(4);wrist[:3,:3]=target['rotation'];wrist[:3,3]=target['position']
                        left=wrist@np.linalg.inv(LT)
                        nose=left[:3,3]+left[:3,:3]@[-.192,0.,-.003]
                        heights.append(float(nose[2]-station.TABLE_TOP-shift[2]))
                    report['planned_lower_nose_heights_m']=heights
                    assert len(heights)==12 and np.allclose(heights,.001,atol=1e-7,rtol=0.), 'Insertion path leaves the qualified nose height'
            except AssertionError as error:
                report.update(passed=False,error=str(error),phase=session.phase,
                    target_phase=session.plan[-1]['phase'] if session.plan else None)
            report['control_steps']=session.steps
            results.append(report)
            (folder/'result.json').write_text(json.dumps(report,indent=2))
            print(json.dumps(report),flush=True)
    proof=dict(source_run=source.name,source_control_step=index,results=results,
        controller_source_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('preflight_cover.py','robot_reopen.py','robot_session.py','wrist_tracking.py','bridge_press.py','supported_release.py')},
        grasp_state_sha256=hashlib.sha256((source/'physics-chunks'/f'{physics_index//request["physics_hz"]:05d}.npz').read_bytes()).hexdigest(),
        scope='Ordinary command reach/speed checks only. No contact or simulated feasibility is certified.')
    (out/'result.json').write_text(json.dumps(proof,indent=2))
    return proof


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('out',type=Path)
    args=parser.parse_args()
    root=OUTPUT_ROOT
    assert args.source.resolve().parent==root and args.out.resolve().parent==root
    preflight(args.source,args.out,[[-.1,-.04,0.],[-.15,-.03,0.],[-.15,0.,0.],[-.2,0.,0.]],(.4,.22))
