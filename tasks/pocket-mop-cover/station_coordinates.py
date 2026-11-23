"""Rigid translation of the stock station for floating-point conditioning.

S is simulation space and W is the published YAM station frame. T_S_W is a
translation only; cameras, robot bases, fixtures and task objects move together.
No dimensions, relative poses, gains, limits or camera intrinsics change.
"""
from dataclasses import replace
import numpy as np

TRANSLATION_S_W=np.array([-.45,-.10,-.7475])


def translated_rig(rig,translation):
    """Return the same calibrated rig expressed in a translated world frame."""
    shift=np.asarray(translation)
    bases={side:tuple(np.asarray(pos)+shift) for side,pos in rig.bases.items()}
    camera=replace(rig.top_cam,pos=tuple(np.asarray(rig.top_cam.pos)+shift))
    return replace(rig,bases=bases,top_cam=camera)


def translated_room(rig,translation):
    """Translate each stock room pose, including lights and the camera stand."""
    from yam_arena.scene import task_room
    from isaaclab_arena.utils.pose import Pose
    assets=task_room('station',rig.top_cam)
    for asset in assets:
        pose=getattr(asset,'initial_pose',None)
        if pose is not None:
            asset.set_initial_pose(Pose(position_xyz=tuple(np.asarray(pose.position_xyz)+translation),
                rotation_xyzw=pose.rotation_xyzw))
    return assets


def verify_station(stage,rig,translation,extra_positions=None):
    """Check actual authored world poses against the unchanged stock calibration.

    Run after reset and before any control action. This reads USD transforms;
    checking only Python initial_pose metadata misses cached construction errors.
    """
    from pxr import Usd,UsdGeom
    from yam_arena.scene import task_room
    expected={}
    for asset in task_room('station',rig.top_cam):
        pose=getattr(asset,'initial_pose',None)
        if pose is not None:
            expected[asset.name]=(np.asarray(pose.position_xyz)+translation,pose.rotation_xyzw)
    for side,name in [('left','LeftArm'),('right','RightArm')]:
        expected[name]=(np.asarray(rig.bases[side])+translation,(0.,0.,0.,1.))
    expected['TopCam']=(np.asarray(rig.top_cam.pos)+translation,rig.top_cam.rot_xyzw)
    for name,position in (extra_positions or {}).items():
        expected[name]=(position,(0.,0.,0.,1.))
    cache=UsdGeom.XformCache(Usd.TimeCode.Default())
    measured={}
    for name,(position,rotation) in expected.items():
        path='/World/envs/env_0/'+name
        prim=stage.GetPrimAtPath(path)
        assert prim.IsValid(),f'Missing station object {path}'
        transform=cache.GetLocalToWorldTransform(prim)
        actual=np.asarray(transform.ExtractTranslation())
        quaternion=transform.ExtractRotationQuat()
        actual_q=np.array([*quaternion.GetImaginary(),quaternion.GetReal()])
        target_q=np.asarray(rotation)
        delta=float(np.linalg.norm(actual-np.asarray(position)))
        angle=float(2*np.arccos(np.clip(abs(np.dot(actual_q,target_q))/(np.linalg.norm(actual_q)*np.linalg.norm(target_q)),0.,1.)))
        assert delta<2e-6,f'{name}: native pose differs from translated calibration by {delta} m'
        assert angle<2e-5,f'{name}: native orientation differs from calibration by {angle} rad'
        measured[name]=dict(position_m=actual.tolist(),quaternion_xyzw=actual_q.tolist(),
            position_error_m=delta,orientation_error_rad=angle)
    return measured
