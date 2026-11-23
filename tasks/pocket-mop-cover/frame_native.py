"""Estimated folding-frame mechanism based on Freudenberg EP2838411B1 figs. 3–5.

The product's exact internal dimensions are unmeasured. The model preserves
geared wings, a ground-contacting pedal foot, a retaining groove and a universal
handle connector. A constant elastic flexure approximates the plastic groove
lip; no state-dependent capture, lock or fabric constraints exist.
"""
import numpy as np

CLIP_LENGTH=.024
CLIP_WIDTH=.036
CLIP_THICKNESS=.0015
CLIP_MODULUS=1.3e9
CLIP_DENSITY=900.
CLIP_MASS=CLIP_LENGTH*CLIP_WIDTH*CLIP_THICKNESS*CLIP_DENSITY
CLIP_STIFFNESS=3*CLIP_MODULUS*(CLIP_WIDTH*CLIP_THICKNESS**3/12)/CLIP_LENGTH
CLIP_DAMPING=.2*2*np.sqrt((CLIP_MASS*CLIP_LENGTH**2/3)*CLIP_STIFFNESS)


def passive_actuators():
    """Return the unconditional zero-rest-angle elastic clip model."""
    from isaaclab.actuators import ImplicitActuatorCfg
    return {"plastic_flexure":ImplicitActuatorCfg(joint_names_expr=["ClipFlexure"],
        effort_limit_sim=10.,stiffness=CLIP_STIFFNESS,damping=CLIP_DAMPING)}


def author_frame(stage,path="/World/Frame",origin=(0.,0.,.160),*,latch_contacts=True,snap_throat=False):
    from pxr import Gf,PhysxSchema,Sdf,UsdGeom,UsdPhysics,UsdShade,Vt
    from scipy.spatial import ConvexHull
    from blade_geometry import shape
    origin=np.asarray(origin)
    root=UsdGeom.Xform.Define(stage,path)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    articulation=PhysxSchema.PhysxArticulationAPI.Apply(root.GetPrim())
    articulation.CreateEnabledSelfCollisionsAttr(True)
    articulation.CreateSolverPositionIterationCountAttr(64)
    articulation.CreateSolverVelocityIterationCountAttr(4)
    material=UsdShade.Material.Define(stage,path+"/Plastic")
    api=UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(.55)
    api.CreateDynamicFrictionAttr(.55)
    api.CreateRestitutionAttr(0.)
    positions={}
    colliders=[]
    masses={}
    def body(name,position,mass):
        positions[name]=np.asarray(position)
        masses[name]=mass
        prim=UsdGeom.Xform.Define(stage,path+"/"+name)
        prim.AddTranslateOp().Set(Gf.Vec3d(*(origin+position)))
        UsdPhysics.RigidBodyAPI.Apply(prim.GetPrim())
        UsdPhysics.MassAPI.Apply(prim.GetPrim()).CreateMassAttr(mass)
        return prim.GetPrim()
    def collision(prim,enabled=True):
        UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(enabled)
        api=PhysxSchema.PhysxCollisionAPI.Apply(prim)
        api.CreateContactOffsetAttr(.0005)
        api.CreateRestOffsetAttr(0.)
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material,UsdShade.Tokens.weakerThanDescendants,"physics")
    def box(parent,name,size,center,color=(.7,.72,.73),enabled=True):
        cube=UsdGeom.Cube.Define(stage,path+"/"+parent+"/"+name)
        cube.CreateSizeAttr(1.)
        cube.AddTranslateOp().Set(Gf.Vec3d(*center))
        cube.AddScaleOp().Set(Gf.Vec3d(*size))
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        collision(cube.GetPrim(),enabled)
        corners=np.asarray([[x,y,z] for x in (-.5,.5) for y in (-.5,.5) for z in (-.5,.5)])
        colliders.append(dict(body=parent,name=name,points=(corners*np.asarray(size)+center).tolist(),enabled=enabled))
    def convex(parent,name,points,color=(.7,.72,.73),enabled=True):
        points=np.asarray(points)
        hull=ConvexHull(points)
        faces=hull.simplices.copy()
        for i,face in enumerate(faces):
            normal=np.cross(points[face[1]]-points[face[0]],points[face[2]]-points[face[0]])
            if normal@hull.equations[i,:3]<0:
                faces[i]=face[[0,2,1]]
        mesh=UsdGeom.Mesh.Define(stage,path+"/"+parent+"/"+name)
        mesh.CreatePointsAttr(Vt.Vec3fArray(points.tolist()))
        mesh.CreateFaceVertexCountsAttr([3]*len(faces))
        mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
        mesh.CreateSubdivisionSchemeAttr("none")
        mesh.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        collision(mesh.GetPrim(),enabled)
        UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr("convexHull")
        colliders.append(dict(body=parent,name=name,points=points.tolist(),enabled=enabled))
    def hinge(name,parent,child,anchor,limits,axis="Y",friction=0.):
        joint=UsdPhysics.RevoluteJoint.Define(stage,path+"/"+name)
        joint.CreateBody0Rel().SetTargets([Sdf.Path(path+"/"+parent)])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(path+"/"+child)])
        joint.CreateLocalPos0Attr(Gf.Vec3f(*(np.asarray(anchor)-positions[parent])))
        joint.CreateLocalPos1Attr(Gf.Vec3f(*(np.asarray(anchor)-positions[child])))
        joint.CreateAxisAttr(axis)
        joint.CreateLowerLimitAttr(limits[0])
        joint.CreateUpperLimitAttr(limits[1])
        PhysxSchema.PhysxJointAPI.Apply(joint.GetPrim()).CreateJointFrictionAttr(friction)
        return joint

    body("Bridge",(0.,0.,0.),.060)
    for sign,label in [(-1,"L"),(1,"R")]:
        box("Bridge","Rail"+label,(.053,.014,.015),(.0105,sign*.043,.0075))
        box("Bridge","Upright"+label,(.014,.014,.025),(.030,sign*.043,.0175))
    box("Bridge","Crossbar",(.014,.10,.010),(.030,0.,.030))
    box("Bridge","HandleMount",(.014,.024,.022),(.030,0.,.040))

    body("Left",(-.008,0.,0.),.110)
    points,_=shape((.100,.10,.006))
    convex("Left","End",points+(-.142,0.,0.))
    for sign,label in [(-1,"L"),(1,"R")]:
        box("Left","InnerRail"+label,(.092,.020,.006),(-.046,sign*.040,0.))
    body("Right",(.008,0.,0.),.130)
    points,_=shape((.192,.10,.006))
    points[:,0]*=-1
    convex("Right","End",points+(.096,0.,0.))
    box("Right","TongueStem",(.012,.028,.007),(-.003,0.,.0065))
    box("Right","Tongue",(.047,.032,.003),(-.0235,0.,.0115))
    box("Left","ClosedStripClamp",(.030,.090,.001),(-.155,0.,.0025),(.12,.13,.14))
    box("Right","ClosedStripClamp",(.030,.090,.001),(.155,0.,.0025),(.12,.13,.14))
    left=hinge("HingeLeft","Bridge","Left",(-.008,0.,0.),(-90.,0.))
    right=hinge("HingeRight","Bridge","Right",(.008,0.,0.),(0.,90.))
    assert right.GetPrim().ApplyAPI("NewtonMimicAPI")
    right.GetPrim().GetAttribute("newton:mimicEnabled").Set(True)
    right.GetPrim().GetAttribute("newton:mimicCoef1").Set(-1.)
    right.GetPrim().GetAttribute("newton:mimicCoef0").Set(0.)
    right.GetPrim().GetRelationship("newton:mimicJoint").SetTargets([left.GetPath()])

    rocker_origin=np.array([-.045,0.,.004])
    body("Rocker",rocker_origin,.020)
    dark=(.10,.12,.13)
    # The published pedal has triangular feet that touch the floor when tilted.
    # These are working contact geometry, not hidden support constraints.
    foot_section=[(-.040,.025),(-.030,-.006),(0.,.002),(0.,.013)]
    for y,label in [(-.023,"L"),(.023,"R")]:
        convex("Rocker","Foot"+label,[(x,yy,z) for yy in (y-.0015,y+.0015) for x,z in foot_section],
            dark,latch_contacts)
    pedal_section=[(-.045,.026),(-.045,.029),(-.006,.016),(-.006,.013)]
    convex("Rocker","Pedal",[(x,y,z) for y in (-.0275,.0275) for x,z in pedal_section],dark,latch_contacts)
    box("Rocker","GrooveBack",(.003,.036,.010),(-.0015,0.,.008),dark,latch_contacts)
    box("Rocker","LowerBar",(.014,.036,.002),(.007,0.,.005),dark,latch_contacts)
    box("Rocker","FlexureRoot",(.008,.036,.004),(-.008,0.,.0106),dark,latch_contacts)
    hinge("RockerHinge","Left","Rocker",rocker_origin,(-30.,0.))
    clip_origin=rocker_origin+np.array([-.010,0.,.0101])
    body("Clip",clip_origin,CLIP_MASS)
    box("Clip","UpperBar",(CLIP_LENGTH,CLIP_WIDTH,CLIP_THICKNESS),
        (CLIP_LENGTH/2,0.,0.),dark,latch_contacts)
    if snap_throat:
        # Estimated rounded snap-groove mouth, represented by two ramps.
        # EP2838411B1 [0046] explicitly describes a snapped-in wing, and Fig. 5
        # shows the projecting groove lip. The previous parallel, clearance-only
        # bars omitted this contact feature and creeped open in the long lift.
        # Throat: 2.60 mm for the 3 mm tongue; interference flexes this same
        # constant PP leaf. No capture or angle-dependent force is introduced.
        section=[(.019,-.00075),(.021,-.0015),(.022,-.0015),(.024,-.00075),(.024,.00075),(.019,.00075)]
        convex("Clip","SnapThroat",[(x,y,z) for y in (-CLIP_WIDTH/2,CLIP_WIDTH/2) for x,z in section],dark,latch_contacts)
    clip_joint=hinge("ClipFlexure","Rocker","Clip",clip_origin,(-15.,15.))
    spring=UsdPhysics.DriveAPI.Apply(clip_joint.GetPrim(),"angular")
    spring.CreateTypeAttr("force")
    spring.CreateStiffnessAttr(CLIP_STIFFNESS*np.pi/180)
    spring.CreateDampingAttr(CLIP_DAMPING*np.pi/180)
    spring.CreateMaxForceAttr(10.)
    spring.CreateTargetPositionAttr(0.)
    # The actuator configuration provides a constant, unconditional elastic
    # restoring torque. It is not changed on seating, contact or wing angle.

    body("Connector",(.030,0.,.052),.015)
    box("Connector","Cross",(.025,.025,.020),(0.,0.,0.))
    hinge("HandleX","Bridge","Connector",(.030,0.,.052),(-85.,85.),axis="X",friction=.03)
    body("Handle",(.030,0.,.062),.180)
    shaft=UsdGeom.Capsule.Define(stage,path+"/Handle/Shaft")
    shaft.CreateRadiusAttr(.011)
    shaft.CreateHeightAttr(.478)
    shaft.CreateAxisAttr("Z")
    shaft.AddTranslateOp().Set(Gf.Vec3d(0.,0.,.25))
    shaft.CreateDisplayColorAttr([Gf.Vec3f(.70,.72,.75)])
    collision(shaft.GetPrim())
    colliders.append(dict(body="Handle",name="Shaft",capsule=dict(radius=.011,height=.478,center=[0.,0.,.25]),enabled=True))
    box("Handle","Grip",(.026,.026,.10),(0.,0.,.44),(.12,.2,.35))
    hinge("HandleY","Connector","Handle",(.030,0.,.062),(-85.,85.),friction=.03)
    return dict(path=path,origin=origin.tolist(),body_origins={k:v.tolist() for k,v in positions.items()},
        body_masses_kg=masses,latch_contacts=latch_contacts,colliders=colliders,
        gear_equation="HingeRight = -HingeLeft",handle_length_m=.5,estimated_dimensions=True,
        snap_throat_estimate=dict(enabled=snap_throat,throat_m=.0026,tongue_m=.003,interference_m=.0004),
        mechanism_reference="https://data.epo.org/publication-server/rest/v1.0/publication-dates/20161221/patents/EP2838411NWB1/document.pdf",
        mechanism_figures="3–5; related manufacturer mechanism family, not exact product metrology",
        clip_model=dict(length_m=CLIP_LENGTH,width_m=CLIP_WIDTH,thickness_m=CLIP_THICKNESS,
            estimated_modulus_pa=CLIP_MODULUS,density_kg_m3=CLIP_DENSITY,
            rotational_stiffness_nm_rad=CLIP_STIFFNESS,damping_nms_rad=CLIP_DAMPING,
            groove_clearance_m=.00035,tongue_overlap_m=.008,
            approximation="Linear equivalent cantilever compliance; dimensions and polypropylene properties estimated. Groove clearance is relative to the 3 mm tongue; earlier prototypes allowed excessive free motion."))


def author_press_guide(stage,path="/World/Frame",origin=(0.,0.,.160)):
    """Attach only the bridge to the disclosed force-limited laboratory press."""
    from pxr import Gf,Sdf,UsdGeom,UsdPhysics
    base=UsdGeom.Xform.Define(stage,path+"/LaboratoryBase")
    UsdPhysics.RigidBodyAPI.Apply(base.GetPrim())
    UsdPhysics.MassAPI.Apply(base.GetPrim()).CreateMassAttr(1.)
    fixed=UsdPhysics.FixedJoint.Define(stage,path+"/LaboratoryBaseFixed")
    fixed.CreateBody1Rel().SetTargets([base.GetPath()])
    joint=UsdPhysics.PrismaticJoint.Define(stage,path+"/LaboratoryPress")
    joint.CreateBody0Rel().SetTargets([base.GetPath()])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(path+"/Bridge")])
    joint.CreateLocalPos0Attr(Gf.Vec3f(*origin))
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-origin[2])
    joint.CreateUpperLimitAttr(.10)
    drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),"linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(500.)
    drive.CreateDampingAttr(8.)
    drive.CreateMaxForceAttr(15.)
    drive.CreateTargetPositionAttr(0.)
    return joint


def author_pedal_pusher(stage, path="/World/PedalPusher", offset=(0.,0.,0.)):
    """Create a visible, force-limited laboratory finger for unlock testing."""
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics
    from native import box
    offset=np.asarray(offset)
    base_position=np.array([-.085,0.,.30])+offset
    finger_position=np.array([-.085,0.,.20])+offset
    root=UsdGeom.Xform.Define(stage,path)
    UsdPhysics.ArticulationRootAPI.Apply(root.GetPrim())
    box(stage,path+"/Base",(.02,.02,.03),base_position,dynamic=True,color=(.75,.2,.15))
    box(stage,path+"/Finger",(.020,.025,.010),finger_position,dynamic=True,mass=.05,color=(.75,.2,.15))
    fixed=UsdPhysics.FixedJoint.Define(stage,path+"/Fixed")
    fixed.CreateBody1Rel().SetTargets([Sdf.Path(path+"/Base")])
    fixed.CreateLocalPos0Attr(Gf.Vec3f(*base_position))
    joint=UsdPhysics.PrismaticJoint.Define(stage,path+"/Press")
    joint.CreateBody0Rel().SetTargets([Sdf.Path(path+"/Base")])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(path+"/Finger")])
    joint.CreateLocalPos0Attr(Gf.Vec3f(0.,0.,-.10))
    joint.CreateAxisAttr("Z")
    joint.CreateLowerLimitAttr(-.19)
    joint.CreateUpperLimitAttr(.01)
    drive=UsdPhysics.DriveAPI.Apply(joint.GetPrim(),"linear")
    drive.CreateTypeAttr("force")
    drive.CreateStiffnessAttr(800.)
    drive.CreateDampingAttr(10.)
    drive.CreateMaxForceAttr(20.)
    drive.CreateTargetPositionAttr(0.)
    return dict(path=path,finger_size_m=[.020,.025,.010],initial_center_m=finger_position.tolist(),
                force_limit_n=20.,laboratory_only=True)
