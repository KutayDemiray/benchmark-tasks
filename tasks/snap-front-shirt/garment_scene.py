"""Native sewn garment, passive press-studs and freely supported lightweight hanger."""

import math
import numpy as np

from garment_mesh import pattern
from geometry import body, shape, set_attr
from robot_geometry import male, socket, rotation


def segment(stage,path,start,end,radius,material,color,kind='cylinder'):
    """Create a cylinder between endpoints, including its physical collision."""
    from pxr import Gf
    start,end = np.asarray(start),np.asarray(end)
    direction = end-start
    geom = shape(stage,path,kind,(start+end)/2,(radius,float(np.linalg.norm(direction))),material,color)
    q = Gf.Rotation(Gf.Vec3d(0,0,1),Gf.Vec3d(*map(float,direction))).GetQuat()
    geom.AddOrientOp().Set(Gf.Quatf(q))


def surface(stage,path,data,cfg):
    """Author one sewn FEM surface with the validated coupon material law."""
    from pxr import Gf,UsdGeom,UsdPhysics,UsdShade,Vt,PhysxSchema
    p=cfg['fabric']
    points,faces=data['initial'],data['faces']
    tri=UsdGeom.Mesh.Define(stage,path)
    tri.CreatePointsAttr(Vt.Vec3fArray(points.tolist()))
    tri.CreateFaceVertexCountsAttr([3]*len(faces))
    tri.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
    tri.CreateSubdivisionSchemeAttr('none')
    tri.CreateDoubleSidedAttr(True)
    tri.CreateDisplayColorAttr([Gf.Vec3f(.22,.42,.65)])
    prim=tri.GetPrim()
    for schema in ('OmniPhysicsDeformableBodyAPI','OmniPhysicsSurfaceDeformableSimAPI','PhysxSurfaceDeformableBodyAPI'):
        assert prim.ApplyAPI(schema),schema
    set_attr(prim,'omniphysics:restShapePoints',Vt.Vec3fArray(data.get('rest_positions',points).tolist()))
    set_attr(prim,'omniphysics:restTriVtxIndices',[Gf.Vec3i(*map(int,f)) for f in faces])
    set_attr(prim,'omniphysics:restBendAnglesDefault','restShapeDefault')
    set_attr(prim,'omniphysics:mass',data['audit']['fabric_mass_kg'])
    set_attr(prim,'physxDeformableBody:selfCollision',True)
    set_attr(prim,'physxDeformableBody:enableSpeculativeCCD',True)
    set_attr(prim,'physxDeformableBody:collisionPairUpdateFrequency',cfg['garment'].get('collision_pair_updates',4))
    set_attr(prim,'physxDeformableBody:collisionIterationMultiplier',cfg['garment'].get('collision_iteration_multiplier',1))
    set_attr(prim,'physxDeformableBody:solverPositionIterationCount',p['solver_iterations'])
    UsdPhysics.CollisionAPI.Apply(prim)
    cp=PhysxSchema.PhysxCollisionAPI.Apply(prim)
    cp.CreateContactOffsetAttr(p['contact_offset_m'])
    cp.CreateRestOffsetAttr(p['rest_offset_m'])
    mat=UsdShade.Material.Define(stage,path+'/fabric')
    for schema in ('OmniPhysicsBaseMaterialAPI','OmniPhysicsDeformableMaterialAPI',
                   'OmniPhysicsSurfaceDeformableMaterialAPI','PhysxSurfaceDeformableMaterialAPI'):
        assert mat.GetPrim().ApplyAPI(schema),schema
    for name,value in {
        'omniphysics:density':p['areal_density_kg_m2']/p['thickness_m'],
        'omniphysics:dynamicFriction':p['friction'],
        'omniphysics:youngsModulus':p['youngs_modulus_Pa'],
        'omniphysics:poissonsRatio':p['poisson_ratio'],
        'omniphysics:surfaceThickness':p['thickness_m'],
        'omniphysics:surfaceBendStiffness':p['bend_modulus_Pa'],
        'physxDeformableMaterial:elasticityDamping':.005,
        'physxDeformableMaterial:bendDamping':.005,
    }.items():
        set_attr(mat.GetPrim(),name,value)
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat,UsdShade.Tokens.weakerThanDescendants,'physics')
    visual=UsdShade.Material.Define(stage,path+'/appearance')
    shader=UsdShade.Shader.Define(stage,path+'/appearance/surface')
    shader.CreateIdAttr('UsdPreviewSurface')
    from pxr import Sdf
    shader.CreateInput('diffuseColor',Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(.12,.32,.58))
    shader.CreateInput('roughness',Sdf.ValueTypeNames.Float).Set(.85)
    visual.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(),'surface')
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(visual)


def build(stage,root,cfg):
    """Create the already-hanging diagnostic; reset poses are never servoed."""
    from pxr import Gf,UsdGeom
    from scipy.spatial.transform import Rotation
    UsdGeom.Xform.Define(stage,root)
    shape(stage,root+'/floor','box',(0,0,-.025),(4,4,.05),cfg['snap'],(.24,.24,.24))
    h=cfg['hanger']
    material=dict(cfg['snap'],friction=h['friction'],
                  contact_offset_m=h.get('contact_offset_m',cfg['snap']['contact_offset_m']))
    rail=np.asarray(h['rail_center_m'])
    half=h['rail_length_m']/2
    segment(stage,root+'/rail',rail-[0,half,0],rail+[0,half,0],h['rail_radius_m'],material,(.55,.55,.58),h.get('rail_geometry','cylinder'))
    for side in (-1,1):
        foot=rail+[0,side*half,-rail[2]]
        segment(stage,root+f'/post_{"left" if side>0 else "right"}',foot,rail+[0,side*half,0],.012,material,(.45,.45,.48))
        shape(stage,root+f'/foot_{"left" if side>0 else "right"}','box',foot+[0,0,.01],(.35,.12,.02),material,(.35,.35,.38))
    hanger=root+'/hanger'
    fixed_support=cfg.get('garment_test',{}).get('fixed_support',False)
    body(stage,hanger,h['origin_m'],h['mass_kg'],fixed=fixed_support)
    wing=h['width_m']/2
    bottom=h['arm_drop_m']
    if h.get('body_style')=='wooden':
        from hanger_shapes import wooden_beams
        for i,beam in enumerate(wooden_beams(h)):
            geom=shape(stage,hanger+f'/bar{i}','box',beam['center'],beam['size'],material,(.72,.58,.35))
            ops=geom.GetOrderedXformOps()
            q=Rotation.from_matrix(beam['rotation']).as_quat()
            orient=geom.AddOrientOp()
            orient.Set(Gf.Quatf(float(q[3]),Gf.Vec3f(*map(float,q[:3]))))
            geom.SetXformOpOrder([ops[0],orient,ops[1]])
        end=wing-h['crossbar_radius_m']
        segment(stage,hanger+'/bar2',[0,-end,-bottom],[0,end,-bottom],h['crossbar_radius_m'],material,(.72,.58,.35),'capsule')
        segment(stage,hanger+'/bar3',[0,0,0],[0,0,.04],h['hook_wire_radius_m'],material,(.70,.70,.72))
    else:
        for i,(a,b) in enumerate((([0,0,0],[0,wing,-bottom]),([0,0,0],[0,-wing,-bottom]),
                                 ([0,-wing,-bottom],[0,wing,-bottom]),([0,0,0],[0,0,.04]))):
            segment(stage,hanger+f'/bar{i}',a,b,h['body_radius_m'],material,(.72,.58,.35),h.get('bar_geometry','cylinder'))
    from garment_contact_audit import hook_polyline,initial_hook_rail_gap
    hook=hook_polyline(h)
    assert initial_hook_rail_gap(h)>0.,'Initial hanger hook intersects rail'
    for i,(a,b) in enumerate(zip(hook[:-1],hook[1:])):
        segment(stage,hanger+f'/hook{i}',a,b,h['hook_wire_radius_m'],material,(.70,.70,.72))
    case=cfg.get('garment_test',{}).get('case')
    probe=case in ('support_probe','sleeve_probe','elongated_sleeve_probe','loaded_sleeve_probe')
    if probe:
        # Diagnostic patch over the existing free hanger crossbar. It isolates
        # surface/cylinder contact; it is not a garment or a fastening trial.
        grid=np.arange(-.05,.0501,.002)
        points=np.array([[h['origin_m'][0]+x,y,h['origin_m'][2]-bottom+.025] for y in grid for x in grid])
        faces=[]
        n=len(grid)
        for j in range(n-1):
            for i in range(n-1):
                a=j*n+i
                faces.extend([[a,a+1,a+n],[a+1,a+n+1,a+n]])
        faces=np.array(faces,dtype=np.int32)
        edges=np.unique(np.sort(np.vstack((faces[:,:2],faces[:,1:],faces[:,[2,0]])),axis=1),axis=0)
        from garment_mesh import topology_audit
        audit=topology_audit(points,faces)
        audit['fabric_mass_kg']=.01*cfg['fabric']['areal_density_kg_m2']
        data=dict(initial=points,faces=faces,edges=edges,
            rest_lengths=np.linalg.norm(points[edges[:,0]]-points[edges[:,1]],axis=1),
            uv=points[:,:2],panel_ids=np.zeros(len(points),dtype=np.int32),rivets=[],audit=audit)
        if case in ('sleeve_probe','elongated_sleeve_probe','loaded_sleeve_probe'):
            # A closed textile sleeve cannot leave the middle of this long bar
            # through an open shirt front. This isolates two-sided shell contact.
            angles=np.arange(64)*2*np.pi/64
            contour=np.column_stack((.02*np.sin(angles),.02*np.cos(angles)))
            if case in ('elongated_sleeve_probe','loaded_sleeve_probe'):
                # A 50 mm longer loop contains an ordinary freely falling
                # steel test rod. Its weight tests garment-scale load without
                # attachments, imposed wrenches or the full shirt topology.
                top=np.linspace(-np.pi/2,np.pi/2,33)
                bottom_arc=np.linspace(np.pi/2,3*np.pi/2,33)
                contour=np.vstack((
                    np.column_stack((.02*np.sin(top),.02*np.cos(top))),
                    np.column_stack((np.full(24,.02),-np.arange(1,25)*.002)),
                    np.column_stack((.02*np.sin(bottom_arc),-.05+.02*np.cos(bottom_arc))),
                    np.column_stack((np.full(24,-.02),-.05+np.arange(1,25)*.002)),
                ))
            ys=np.arange(-.05,.0501,.002)
            points=np.array([[h['origin_m'][0]+x,y,
                              h['origin_m'][2]-bottom+z] for y in ys for x,z in contour])
            faces=[]
            nring=len(contour)
            for j in range(len(ys)-1):
                for i in range(nring):
                    a=j*nring+i
                    b=j*nring+(i+1)%nring
                    faces.extend([[a,b,a+nring],[b,b+nring,a+nring]])
            faces=np.asarray(faces,dtype=np.int32)
            edges=np.unique(np.sort(np.vstack((faces[:,:2],faces[:,1:],faces[:,[2,0]])),axis=1),axis=0)
            area=np.linalg.norm(np.cross(points[faces[:,1]]-points[faces[:,0]],points[faces[:,2]]-points[faces[:,0]]),axis=1).sum()/2
            audit=topology_audit(points,faces)
            assert audit['boundary_loops']==2 and audit['oriented']
            audit['fabric_mass_kg']=float(area*cfg['fabric']['areal_density_kg_m2'])
            data=dict(initial=points,faces=faces,edges=edges,
                rest_lengths=np.linalg.norm(points[edges[:,0]]-points[edges[:,1]],axis=1),
                uv=points[:,:2],panel_ids=np.zeros(len(points),dtype=np.int32),rivets=[],audit=audit)
    else:
        data=pattern(cfg)
    # A reset must not start with rail/posts penetrating a sleeve. Surface
    # topology alone cannot detect overlaps with separate physical objects.
    clearance={}
    for name,a,b,radius in (
        ('rail',rail-[0,half,0],rail+[0,half,0],h['rail_radius_m']),
        ('left_post',rail+[0,half,-rail[2]],rail+[0,half,0],.012),
        ('right_post',rail+[0,-half,-rail[2]],rail+[0,-half,0],.012),
    ):
        direction=b-a
        relative=data['initial']-a
        f=np.clip(relative@direction/(direction@direction),0.,1.)
        # Conservatively enclose the actual cylinder with a capsule.
        gap=float(np.linalg.norm(relative-f[:,None]*direction,axis=1).min()-radius)
        # Cover complete triangles as well as their sampled vertices.
        clearance[name]=gap-float(data['rest_lengths'].max())
        assert clearance[name]>.001,f'Initial shirt intersects support: {name}, bound {clearance[name]}'
    data['initial_static_clearance_bounds_m']=clearance
    if h.get('body_style')=='wooden':
        from hanger_shapes import wooden_initial_clearance
        clearance=wooden_initial_clearance(data['initial'],data['faces'],[*h['origin_m'],0,0,0,1],h,.0003)
        assert clearance['separated'],'Initial cloth intersects wooden shoulders'
        from garment_contact_audit import segment_triangle_distances,bar_contacts
        local=data['initial']-np.asarray(h['origin_m'])
        end=wing-h['crossbar_radius_m']
        distances=segment_triangle_distances(local[data['faces']],[0,-end,-bottom],[0,end,-bottom])
        crossings=bar_contacts(data['initial'],data['faces'],[*h['origin_m'],0,0,0,1],h)[2]['crossed_triangles']
        distances[crossings]=0.
        clearance['crossbar_capsule_gap_m']=float(distances.min()-h['crossbar_radius_m'])
        assert clearance['crossbar_capsule_gap_m']>cfg['fabric']['rest_offset_m']
        data['initial_wooden_shoulder_clearance']=clearance
    elif h.get('bar_geometry')=='capsule':
        from garment_contact_audit import initial_capsule_clearances
        gaps=initial_capsule_clearances(data['initial'],data['faces'],[*h['origin_m'],0,0,0,1],h)
        assert min(gaps)>cfg['fabric']['rest_offset_m'],'Initial cloth intersects hanger body'
        data['initial_hanger_triangle_clearance_m']=gaps
    path=root+'/shirt'
    surface(stage,path,data,cfg)
    # Local +Z is the outward front normal (-world X). Male heads point inwards.
    q=Rotation.from_matrix(np.array([[0,0,-1],[1,0,0],[0,-1,0]])).as_quat()
    hardware={'hanger':hanger}
    initial_hardware={'hanger':[*h['origin_m'],0,0,0,1]}
    if case=='loaded_sleeve_probe':
        payload=root+'/payload'
        center=np.asarray(h['origin_m'])+[0,0,-bottom-.05]
        radius,length,density=.01,.08,7850.
        mass=math.pi*radius**2*length*density
        body(stage,payload,center,mass)
        segment(stage,payload+'/rod',[0,-length/2,0],[0,length/2,0],radius,
                dict(cfg['snap'],friction=.4),(.3,.3,.32))
        hardware['payload']=payload
        initial_hardware['payload']=[*center,0,0,0,1]
        data['payload_assumptions']=dict(material='steel test rod',radius_m=radius,length_m=length,
            assumed_density_kg_m3=density,computed_mass_kg=mass,measured=False,
            attachments=False,external_forces=False)
    attachments=[]
    for rivet in data.pop('rivets'):
        name,kind=rivet['name'],rivet['kind']
        center=rivet['center']+([.0007,0,0] if kind=='male' else [-.0007,0,0])
        hp=root+'/'+name
        if kind=='male':
            male(stage,hp,center,q,cfg)
        else:
            _,balls=socket(stage,hp,center,q,cfg)
            hardware.update({f'{name}_ball{i}':p for i,p in enumerate(balls)})
        hardware[name]=hp
        initial_hardware[name]=[*center,*q]
        ids=rivet['ids']
        local=(data['initial'][ids]-center)@rotation(q)
        att=stage.DefinePrim(root+'/'+name+'_rivet','OmniPhysicsVtxXformAttachment')
        att.GetRelationship('omniphysics:src0').SetTargets([path])
        att.GetRelationship('omniphysics:src1').SetTargets([hp])
        set_attr(att,'omniphysics:attachmentEnabled',True)
        set_attr(att,'omniphysics:vtxIndicesSrc0',ids.tolist())
        set_attr(att,'omniphysics:localPositionsSrc1',[Gf.Vec3f(*map(float,v)) for v in local])
        attachments.append(dict(hardware=name,ids=ids,local=local))
    data.update(path=path,attachments=attachments)
    return dict(root=root,hardware=hardware,initial_hardware=initial_hardware,cloth=data,
                diagnostic_world_mounts=fixed_support,grasp_attachments=False,hanger_attachments=False,
                allowed_diagnostic_world_joint=(hanger+'/diagnostic_mount' if fixed_support else None),
                target_snap_index=cfg['completion']['target_snap_index'],required_snap_count=1,
                initial_condition=(f'{case} on hanger crossbar; contact diagnostic only' if probe else
                                   'top snap initially closed; final-state control only, no robot success' if case=='closed_target' else
                                   'already hanging, all snaps open; integration diagnostic only'))


def camera_rig():
    """Keep the stock bases/wrist cameras and use an oblique overhead view."""
    from dataclasses import replace
    from scipy.spatial.transform import Rotation
    from yam_arena.rig import TABLETOP,Lens
    position=np.array([-1.05,-.65,2.15])
    target=np.array([.28,0,.95])
    forward=(target-position)/np.linalg.norm(target-position)
    left=np.cross([0,0,1],forward)
    left/=np.linalg.norm(left)
    up=np.cross(forward,left)
    q=Rotation.from_matrix(np.column_stack((forward,left,up))).as_quat()
    return replace(TABLETOP,name='snap_shirt',top_cam=replace(TABLETOP.top_cam,
                   pos=tuple(position),rot_xyzw=tuple(q),convention='world',lens=Lens.from_focal(13.)))
