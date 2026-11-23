"""Whole snap-front shirt on a broad work table, without hanger or fixtures."""

import numpy as np

from garment_mesh import pattern
from garment_scene import surface
from geometry import shape,set_attr
from robot_geometry import male,socket,rotation


def configure(cfg):
    """Configure the distinct tabletop candidate without changing material laws."""
    cfg['original_hanger_status']=cfg['status']
    cfg['revision']='p11-tabletop-three-snaps-r2'
    cfg['status']='Tabletop adaptation: support and one-snap development pass; three-snap task validation pending.'
    cfg['completion'].update(front_snap_count=3,required_snap_indices=[0,1,2],
        shirt_start='front-up on a broad table',hanger_start=None,scope='tabletop_adaptation')
    cfg['garment']['extra_transverse_grid_m']=[]
    cfg['garment']['extra_height_grid_m']=[]
    cfg['garment']['hanging_origin_m']=[0.,0.,0.]
    cfg['garment']['parked_robot_joints_rad']={s:[0.,1.,1.,-.6,0.,0.] for s in ('left','right')}
    cfg['tabletop']={
        'revision':'p11-tabletop-three-snaps-r2',
        'table_center_xy_m':[.605,.12], 'table_size_xy_m':[1.,1.8],
        'table_top_m':.75, 'table_thickness_m':.04,
        'hem_x_m':.92, 'center_y_m':.12, 'back_initial_gap_m':.0005,
        'required_snap_indices':[0,1,2],
        'max_table_penetration_m':.001,
        'released_observation_seconds':3., 'released_robot_clearance_m':.03,
        'max_released_drift_m':.02,
        'seat_rim_penetration_allowance_m':.0005,
        'initial_alignment_supplied':True,
        'exterior_plate_thickness_m':.0004,'exterior_plate_center_local_m':.00115,
        'exterior_plate_mass_kg':.0004,
        'hardware_revision_note':'Add physical exterior rivet plates to both cloth-mounted halves. The original head/socket/springs remain unchanged. Plates sandwich the local fabric, make the top hardware visible, and provide direct pressing contact; dimensions/mass estimated.',
        'hardware_construction_reference':'https://www.scovill.com/about-us/faqs/',
        'scope':'Separate tabletop candidate. One-snap work is development toward three upper closures. Original hanger task remains incomplete.',
        'mesh_note':'10 mm body grid with the existing 2 mm placket/rivet refinement. Hanger-specific shoulder/neck refinement removed; cloth material unchanged.',
    }
    cfg['robot_test']['press_start']='ready'


def layout(cfg):
    """Rotate the full sewn shirt front-up; preserve all intrinsic edge lengths."""
    data=pattern(cfg)
    p=cfg['tabletop']
    # S is the previous standing pattern frame, W the tabletop world.
    # Front normal -S.X becomes +W.Z; the neck points toward the robot bases.
    R_W_S=np.array([[0.,0.,-1.],[0.,-1.,0.],[-1.,0.,0.]])
    t_W_S=np.array([p['hem_x_m'],p['center_y_m'],
                   p['table_top_m']+cfg['garment']['back_depth_m']+p['back_initial_gap_m']])
    for key in ('initial','rest_positions'):
        data[key]=data[key]@R_W_S.T+t_W_S
    for rivet in data['rivets']:
        rivet['center']=rivet['center']@R_W_S.T+t_W_S
    assert data['initial'][:,2].min()>p['table_top_m']+cfg['fabric']['rest_offset_m']
    return data,R_W_S


def build(stage,root,cfg):
    """Author static furniture and free fabric-mounted compatible snap halves."""
    from pxr import Gf,UsdGeom,UsdPhysics
    from scipy.spatial.transform import Rotation
    UsdGeom.Xform.Define(stage,root)
    p=cfg['tabletop']
    h=p['table_top_m'];thickness=p['table_thickness_m']
    center=p['table_center_xy_m'];size=p['table_size_xy_m']
    shape(stage,root+'/floor','box',(0.,0.,-.025),(4.,4.,.05),cfg['snap'],(.28,.28,.28))
    shape(stage,root+'/table','box',(*center,h-thickness/2),(*size,thickness),cfg['snap'],(.46,.37,.27))
    for index,(x,y) in enumerate(((.15,-.60),(.15,.84),(1.06,-.60),(1.06,.84))):
        shape(stage,root+f'/leg{index}','box',(x,y,(h-thickness)/2),(.04,.04,h-thickness),cfg['snap'],(.35,.32,.28))
    data,R_W_S=layout(cfg)
    cloth_path=root+'/shirt'
    surface(stage,cloth_path,data,cfg)
    R_S_H=np.array([[0.,0.,-1.],[1.,0.,0.],[0.,-1.,0.]])
    q=Rotation.from_matrix(R_W_S@R_S_H).as_quat()
    hardware={};initial={};attachments=[]
    for rivet in data.pop('rivets'):
        name,kind=rivet['name'],rivet['kind']
        offset=R_W_S@np.array([.0007 if kind=='male' else -.0007,0.,0.])
        center=rivet['center']+offset
        path=root+'/'+name
        if kind=='male':
            male(stage,path,center,q,cfg)
        else:
            _,balls=socket(stage,path,center,q,cfg)
            hardware.update({f'{name}_ball{i}':s for i,s in enumerate(balls)})
        sign=1. if kind=='male' else -1.
        shape(stage,path+'/exterior_plate','cylinder',(0.,0.,sign*p['exterior_plate_center_local_m']),
              (cfg['snap']['cap_diameter_m']/2,p['exterior_plate_thickness_m']),cfg['snap'],(.86,.84,.77))
        UsdPhysics.MassAPI(stage.GetPrimAtPath(path)).CreateMassAttr(
            cfg['snap'][kind+'_mass_kg']+p['exterior_plate_mass_kg'])
        hardware[name]=path
        initial[name]=[*center,*q]
        ids=rivet['ids'];local=(data['initial'][ids]-center)@rotation(q)
        att=stage.DefinePrim(root+'/'+name+'_rivet','OmniPhysicsVtxXformAttachment')
        att.GetRelationship('omniphysics:src0').SetTargets([cloth_path])
        att.GetRelationship('omniphysics:src1').SetTargets([path])
        set_attr(att,'omniphysics:attachmentEnabled',True)
        set_attr(att,'omniphysics:vtxIndicesSrc0',ids.tolist())
        set_attr(att,'omniphysics:localPositionsSrc1',[Gf.Vec3f(*map(float,v)) for v in local])
        attachments.append(dict(hardware=name,ids=ids,local=local))
    data.update(path=cloth_path,attachments=attachments)
    return dict(root=root,hardware=hardware,initial_hardware=initial,cloth=data,
        diagnostic_world_mounts=False,grasp_attachments=False,hanger_attachments=False,
        table_top_m=h,table_center_xy_m=center,table_size_xy_m=size,
        required_snap_indices=p['required_snap_indices'],initial_alignment_supplied=True,
        initial_condition='Whole unfastened shirt, front-up on table; no hanger or external attachments.',
        original_hanger_task_success=False)


def camera_rig():
    """Show the whole tabletop shirt while preserving stock wrist cameras/bases."""
    from dataclasses import replace
    from yam_arena.rig import TABLETOP,Lens
    return replace(TABLETOP,name='snap_shirt_tabletop',top_cam=replace(TABLETOP.top_cam,
        pos=(.60,.12,2.10),rot_xyzw=(0.,2**-.5,0.,2**-.5),convention='world',lens=Lens.from_focal(13.)))
