"""Read-only native diagnostics shared by every full-scene controller backend."""

import numpy as np
from scipy.spatial.transform import Rotation

from robot_gate import array


def make_observer(base,frame,cover,colliders,robot_geometry,req):
    """Return frame readback and per-substep/control diagnostics without writing state."""
    def bodies():return array(frame.root_view.get_link_transforms())[0].reshape(-1,7)
    def observe(native):
        poses=bodies()
        q=array(frame.root_view.get_dof_positions())[0]
        if native:
            values=dict(frame_body_poses=poses,frame_joint_positions=q)
            if req.get('release_probe_run'):
                values['frame_body_velocities']=array(frame.root_view.get_link_velocities())[0].reshape(-1,6)
            if cover is not None:values.update(cover.observe_native())
            for side in ('left','right'):
                robot=base.scene[side+'_robot']
                robot_poses=array(robot.root_view.get_link_transforms())[0].reshape(-1,7)
                values[side+'_grasp_poses']=robot_poses[colliders[side]['indices']]
                if robot_geometry:
                    values[side+'_body_poses']=robot_poses
            return values
        observation=dict(bridge_pose_xyzw=poses[frame.body_names.index('Bridge')].tolist(),
            handle_pose_xyzw=poses[frame.body_names.index('Handle')].tolist(),
            frame_joint_angles_deg=dict(zip(frame.joint_names,np.rad2deg(q).tolist())),
            wing_angles_deg=np.rad2deg(q[[frame.joint_names.index('HingeLeft'),frame.joint_names.index('HingeRight')]]).tolist())
        if cover is not None:observation.update(cover.observe_control(poses,frame.body_names))
        return observation
    return bodies,observe
