"""Conservative outcome observations from recorded native substep geometry.

No command, controller phase or intended grasp is used. Swept bounds certify
clearance throughout each recorded control interval; a possible contact is
treated as support until separation is proved. Numerical validity is supplied
by the separate complete contact audit, never inferred from successful motion.
"""

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from audit_mechanism import capsule_proxy
from measurements import pocket_seating
from score_logic import Observation,OutcomeScorer


def gap(low_a,high_a,low_b,high_b):
    """Return the Euclidean distance between axis-aligned bounding boxes."""
    delta=np.maximum(np.maximum(low_a-high_b,low_b-high_a),0.)
    return np.linalg.norm(delta,axis=-1)


def prepare_swept_geometry(geometry):
    """Cache actual convex vertices and unit projection axes for each collider."""
    result=[]
    for shape in geometry:
        if not shape.get('enabled',True):continue
        vertices=capsule_proxy(shape['capsule'])[0] if 'capsule' in shape else np.asarray(shape['points'])
        normals=ConvexHull(vertices).equations[:,:3]
        normals=np.unique(np.round(normals,12),axis=0)
        normals/=np.linalg.norm(normals,axis=1)[:,None]
        result.append(dict(body=shape['body'],vertices=vertices,projection_axes=normals))
    return result


def swept_shapes(geometry,poses,body_names):
    """Enclose actual convex geometry across every recorded native substep."""
    if geometry and 'projection_axes' not in geometry[0]:
        geometry=prepare_swept_geometry(geometry)
    result=[]
    for shape in geometry:
        body=poses[:,body_names.index(shape['body'])]
        rotation=Rotation.from_quat(body[:,3:]).as_matrix()
        world=np.einsum('tij,pj->tpi',rotation,shape['vertices'])+body[:,None,:3]
        cloud=world.reshape(-1,3)
        # Axes select projections only. Extrema include ALL native poses, so
        # using endpoint orientations here never omits midinterval contact.
        axes=np.concatenate([shape['projection_axes']@rotation[0].T,
                             shape['projection_axes']@rotation[-1].T])
        result.append(dict(body=shape['body'],low=cloud.min(0),high=cloud.max(0),
                           cloud=cloud,axes=axes))
    return result


def projection_clearance(first,second):
    """Certify separation of swept convex supersets along unit projection axes."""
    coarse=float(gap(first['low'],first['high'],second['low'],second['high']))
    if 'cloud' not in first or 'cloud' not in second:return coarse
    axes=np.concatenate([first['axes'],second['axes']])
    axes/=np.linalg.norm(axes,axis=1)[:,None]
    a=first['cloud']@axes.T
    b=second['cloud']@axes.T
    separation=np.maximum(a.min(0)-b.max(0),b.min(0)-a.max(0)).max()
    # Every projection gap is a lower bound on Euclidean separation. The
    # capsule proxy is an outer hull; unknown separation remains zero.
    return max(coarse,float(separation)-1e-9)


def shape_clearance(first,second,refine_below=.02):
    """Certify a lower clearance bound between two groups of swept colliders."""
    assert first and second
    minimum=float('inf')
    for a in first:
        for b in second:
            bound=float(gap(a['low'],a['high'],b['low'],b['high']))
            if bound>=minimum:continue
            if bound<refine_below:bound=projection_clearance(a,b)
            minimum=min(minimum,bound)
    return minimum


def cloth_clearance(positions,faces,shapes):
    """Bound cloth/rigid separation using every swept material triangle."""
    low=np.minimum.reduce([positions[:,faces[:,i]].min(0) for i in range(3)])
    high=np.maximum.reduce([positions[:,faces[:,i]].max(0) for i in range(3)])
    cloth_low,cloth_high=low.min(0),high.max(0)
    minimum=float('inf')
    for shape in shapes:
        if gap(cloth_low,cloth_high,shape['low'],shape['high'])>=minimum:continue
        minimum=min(minimum,float(gap(low,high,shape['low'],shape['high']).min()))
    return minimum


class NativeOutcome:
    """Convert complete recorded intervals into phase-independent outcome scores."""

    def __init__(self,rest,faces,cover,frame,frame_names,robots,table_top,physics_hz,numerical,table_bounds):
        self.faces=faces
        self.cover=cover
        self.frame=frame
        self.frame_names=frame_names
        self.robots=robots
        self.frame_shapes=prepare_swept_geometry(frame['colliders'])
        self.robot_shapes={side:prepare_swept_geometry(robots[side]['colliders'])
                           for side in ('left','right')}
        self.table_top=table_top
        self.table_bounds=table_bounds
        self.physics_hz=physics_hz
        self.previous=np.asarray(rest).copy()
        self.numerical=numerical
        self.scorer=OutcomeScorer()

    def update(self,step,state):
        """Score one complete native control interval after numerical auditing."""
        positions=state['positions'];velocities=state['velocities']
        poses=state['frame_body_poses'];joints=state['frame_joint_positions']
        assert len(positions)==len(velocities)==len(poses)==len(joints)==self.physics_hz//20
        bodies=self.frame_names['body_names']
        joint_names=self.frame_names['joint_names']
        seats=[]
        for index,name in enumerate(('Left','Right')):
            pose=poses[-1,bodies.index(name)]
            rotation=Rotation.from_quat(pose[3:]).as_matrix()
            center=pose[:3]+rotation@np.array([-.142 if index==0 else .142,0.,0.])
            if index==1:rotation=rotation@np.diag([-1.,-1.,1.])
            pocket=self.cover['pockets'][index]
            seats.append(pocket_seating(positions[-1],np.asarray(pocket['closed_faces']),center,[.1,.1,.006],
                markers=pocket['markers'],faces=np.asarray(pocket['faces']),profile='beveled',rotation=rotation))
        frame_bounds=swept_shapes(self.frame_shapes,poses,bodies)
        head_bounds=[b for b in frame_bounds if b['body'] in ('Left','Right')]
        on_table=all(np.all(b['low'][:2]>=self.table_bounds[0]) and
                     np.all(b['high'][:2]<=self.table_bounds[1]) for b in head_bounds)
        robot_bounds=[]
        hand_bounds=[]
        for side in ('left','right'):
            geometry=self.robots[side]
            bounds=swept_shapes(self.robot_shapes[side],state[side+'_body_poses'],geometry['body_names'])
            robot_bounds.extend(bounds)
            hand_bounds.extend(b for b in bounds if b['body'] in ('gripper','tip_left','tip_right'))
        cloth_gap=cloth_clearance(positions,self.faces,robot_bounds)
        wing_gap=shape_clearance(head_bounds,robot_bounds,.002)
        handle_gap=shape_clearance([b for b in frame_bounds if b['body']=='Handle'],robot_bounds,.002)
        hand_gap=min(shape_clearance(frame_bounds,hand_bounds),cloth_clearance(positions,self.faces,hand_bounds))
        differences=np.diff(np.concatenate((self.previous[None],positions)),axis=0)*self.physics_hz
        self.previous=positions[-1].copy()
        angles=np.rad2deg(joints[:,[joint_names.index('HingeLeft'),joint_names.index('HingeRight')]])
        bridge=poses[:,bodies.index('Bridge')]
        observation=Observation(step=step,dt=1/20,
            pocket_depth_m=tuple(.04 if seat['seated_samples'] else 0. for seat in seats),
            pocket_width_fraction=tuple(.8 if seat['seated_samples'] else 0. for seat in seats),
            orientation_correct=all(seat['orientation_correct'] for seat in seats),
            pocket_orientation_correct=tuple(seat['orientation_correct'] for seat in seats),
            wing_angle_deg=float(abs(angles).max()),
            textile_grasped=cloth_gap<.002,wings_externally_supported=wing_gap<.002,
            all_cloth_clearance_m=float(positions[:,:,2].min()-self.table_top),
            head_xy_m=tuple(float(v) for v in bridge[-1,:2]),head_height_over_support_m=float(bridge[:,2].max()-self.table_top),
            all_grippers_clearance_m=hand_gap,handle_grasped=handle_gap<.002,
            over_support_surface=on_table,
            max_cloth_speed_m_s=float(np.linalg.norm(velocities,axis=2).max()),
            max_cloth_difference_speed_m_s=float(np.linalg.norm(differences,axis=2).max()),
            **self.numerical)
        result=self.scorer.update(observation)
        return observation,{**result,'pockets':seats,'cloth_robot_clearance_lower_bound_m':cloth_gap,
            'wings_robot_clearance_lower_bound_m':wing_gap,'handle_robot_clearance_lower_bound_m':handle_gap}
