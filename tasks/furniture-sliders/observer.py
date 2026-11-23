"""Rendering-only rear view for human contact inspection; not a policy input."""
import os
import numpy as np


def add_camera(scene_config):
    if os.environ.get('FURNITURE_SLIDERS_OBSERVER')!='1':return
    from isaaclab.sensors import CameraCfg
    from isaaclab.sim import PinholeCameraCfg
    from scipy.spatial.transform import Rotation
    from yam_arena.embodiment import RENDERER
    from copy import deepcopy
    position=np.array((.96,.54,.96))
    target=np.array((.37,0,.825))
    forward=target-position;forward/=np.linalg.norm(forward)
    left=np.cross((0,0,1),forward);left/=np.linalg.norm(left)
    up=np.cross(forward,left)
    quaternion=tuple(Rotation.from_matrix(np.column_stack((forward,left,up))).as_quat())
    width,height=map(int,os.environ['FURNITURE_SLIDERS_OBSERVER_SIZE'].split('x'))
    scene_config.furniture_observer=CameraCfg(
        prim_path='{ENV_REGEX_NS}/FurnitureObserver',width=width,height=height,data_types=['rgb'],
        update_period=0.,spawn=PinholeCameraCfg(focal_length=18.,clipping_range=(.02,20.)),
        offset=CameraCfg.OffsetCfg(pos=tuple(position),rot=quaternion,convention='world'),
        renderer_cfg=deepcopy(RENDERER))


def recording_images(embodiment,observation):
    images=dict(observation.images)
    if os.environ.get('FURNITURE_SLIDERS_OBSERVER')=='1':
        value=embodiment._base.scene['furniture_observer'].data.output['rgb']
        value=value.torch if hasattr(value,'torch') else value
        images['observer_cam']=np.ascontiguousarray(value[0,...,:3].cpu().numpy(),dtype=np.uint8)
    return images
