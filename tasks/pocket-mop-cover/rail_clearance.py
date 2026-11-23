"""Kinematic collision checks for approaching a real inner frame rail."""

import numpy as np

from audit_frame_robot import compare,prepare
from robot_clearance import link_transforms


class RailClearance:
    """Reject proposed open-finger approaches that pass through frame hardware."""

    def __init__(self,frame_geometry,frame_poses,frame_names,robot_geometry,shift):
        self.shift=shift
        self.frame=[]
        for shape in frame_geometry['colliders']:
            if shape['enabled']:
                prepared=prepare(shape,frame_poses[None,frame_names.index(shape['body'])])
                prepared['name']=shape['body']+'/'+shape['name']
                self.frame.append(prepared)
        self.robot=[]
        identity=np.array([[0.,0.,0.,0.,0.,0.,1.]])
        for shape in robot_geometry['colliders']:
            prepared=prepare(shape,identity)
            self.robot.append(prepared)

    def contacts(self,joints,opening):
        """Report intersections of a proposed arm pose with the measured frame."""
        links=link_transforms(np.r_[joints,opening],'left',self.shift)
        hits=[]
        for source in self.robot:
            T_W_B=links[source['body']]
            position=T_W_B[:3,3][None]
            rotation=T_W_B[None,:3,:3]
            _,_,vertices,normals,edges=source['data']
            world=vertices@T_W_B[:3,:3].T+position
            target={**source,'data':(position,rotation,vertices,normals,edges),
                'low':world.min(0)[None],'high':world.max(0)[None],
                'global_low':world.min(0),'global_high':world.max(0)}
            for frame in self.frame:
                depth,_,_=compare(frame,target)
                if depth>.0001:
                    hits.append(dict(frame=frame['name'],robot=source['name'],depth_bound_m=depth))
        return hits
