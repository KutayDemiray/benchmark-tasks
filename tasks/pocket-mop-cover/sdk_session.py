"""Task-local lifecycle and readback helpers for isolated installed-SDK tests."""

import hashlib
import json
import logging
from pathlib import Path

import numpy as np


def author_defaults(scene_path,out,cloth_path,articulation_paths):
    """Match the Kit defaults recorded independently by the R158b native capture."""
    from pxr import Usd,Sdf
    stage=Usd.Stage.Open(str(scene_path));before=stage.GetRootLayer().ExportToString()
    overrides={cloth_path:{'physxDeformableBody:maxDepenetrationVelocity':3.,
                          'physxDeformableBody:linearDamping':.005,'physxDeformableBody:sleepThreshold':.005}}
    metadata={}
    for path in articulation_paths:
        prim=stage.GetPrimAtPath(path);assert prim
        metadata[path]=stage.GetRootLayer().GetPrimAtPath(path).GetInfo('apiSchemas')
        schemas=prim.GetMetadata('apiSchemas').GetAppliedItems()
        if 'PhysxArticulationAPI' not in schemas:
            prim.SetMetadata('apiSchemas',Sdf.TokenListOp.CreateExplicit([*schemas,'PhysxArticulationAPI']))
        overrides[path]={'physxArticulation:sleepThreshold':.005,'physxArticulation:stabilizationThreshold':1.}
    for path,values in overrides.items():
        prim=stage.GetPrimAtPath(path);assert prim
        for name,value in values.items():
            attr=prim.GetAttribute(name);assert not attr or not attr.HasAuthoredValueOpinion()
            prim.CreateAttribute(name,Sdf.ValueTypeNames.Float,custom=False).Set(value)
    after=stage.GetRootLayer().ExportToString()
    for path,values in overrides.items():
        for name in values:assert stage.GetPrimAtPath(path).RemoveProperty(name)
    for path,value in metadata.items():stage.GetPrimAtPath(path).SetMetadata('apiSchemas',value)
    assert stage.GetRootLayer().ExportToString()==before, 'Unexplained physical edits'
    stage.GetRootLayer().ImportFromString(after);stage.GetRootLayer().Save()
    report=dict(diagnostic_only=True,overrides=overrides,only_listed_attributes_and_required_api_metadata_changed=True,
        source_native_capture='passive-r158b-kit-properties',
        before_layer_sha256=hashlib.sha256(before.encode()).hexdigest(),after_layer_sha256=hashlib.sha256(after.encode()).hexdigest(),
        material_geometry_unchanged=True,ordinary_backend_changed=False)
    (out/'native-default-override.json').write_text(json.dumps(report,indent=2))


class SDKSession:
    """Own one scheduled physics-only SDK instance and all native bindings."""

    def __init__(self,out,enable_rendering=False,record_geometry=False,scene_path=None):
        from gpu_coordination import require_reservation
        assignment=require_reservation()
        from ovphysx import PhysX,PhysXConfig,enable_python_logging,set_log_level
        from ovphysx.types import LogLevel
        import ovstage
        self.out=out;self.bindings=[];self.errors=[];self.engine=None;self.stage=None
        logger=logging.getLogger('pocket_sdk_'+out.name);logger.setLevel(logging.INFO)
        handler=logging.FileHandler(out/'kit.log')
        handler.setFormatter(logging.Formatter('[%(levelname)s] [omni.physx.sdk] %(message)s'))
        logger.addHandler(handler)
        errors=self.errors
        class NativeErrors(logging.Handler):
            def emit(self,record):
                if record.levelno>=logging.ERROR:errors.append(record.getMessage())
        logger.addHandler(NativeErrors());enable_python_logging(logger.name);set_log_level(LogLevel.INFO)
        settings={'/physics/suppressReadback':True}
        if enable_rendering:
            settings.update({'/renderer/activeGpu':assignment['index'], '/renderer/multiGpu/enabled':False,
                             '/renderer/multiGpu/activeCudaGpus':'0,'})
        try:
            recording=dict(omnipvd_output_enabled=True,omnipvd_ovd_recording_directory=str(out/'omnipvd')) if record_geometry else {}
            self.engine=PhysX(active_cuda_gpus='0',config=PhysXConfig(cooked_collider_cache_dir=str(out/'cooked-cache'),
                carbonite_overrides=settings,**recording))
            self.stage=ovstage.Stage('pocket-sdk-'+out.name)
            ovstage.population.open_usd(self.stage,str(out/'scene.usda' if scene_path is None else scene_path),ordinal=1,domains=ovstage.PopulationDomain.PHYSICS)
            self.stage.advance_write_floor(ordinal=1).wait();self.engine.attach_ovstage(self.stage,read_ordinal=1)
        except Exception:
            self.close();raise

    def bind(self,name,path):
        from ovphysx.types import TensorType
        binding=self.engine.create_tensor_binding(pattern=path,tensor_type=getattr(TensorType,name),raise_if_empty=True)
        self.bindings.append(binding);return binding

    @staticmethod
    def read(binding,dtype=np.float32):
        value=np.empty(binding.shape,dtype=dtype);binding.read(value);return value

    def step(self,dt):
        self.engine.step_sync(dt)
        assert not self.errors, 'Native SDK error: '+str(self.errors)

    def verify(self,binding):
        import torch
        from standalone_contact import verify_device
        value=torch.empty(binding.shape,dtype=torch.float32,device='cuda:0')
        binding.read(value);verify_device(self.out,value)
        assert not self.errors
        (self.out/'native-initialization-check.json').write_text(json.dumps(dict(passed=True,native_errors=self.errors),indent=2))

    def close(self):
        for binding in reversed(self.bindings):binding.destroy()
        self.bindings.clear()
        if self.engine is not None:
            if self.stage is not None:self.engine.detach_ovstage();self.stage.destroy();self.stage=None
            self.engine.release();self.engine=None


def surface_bindings(session,path,material,rest,faces):
    """Check native cloth topology/material and expose its recorded state tensors."""
    native={}
    for kind,key in (('DEFORMABLE_MATERIAL_YOUNGS_MODULUS','young_modulus_pa'),
        ('DEFORMABLE_MATERIAL_POISSONS_RATIO','poisson_ratio'),('DEFORMABLE_MATERIAL_DYNAMIC_FRICTION','dynamic_friction'),
        ('DEFORMABLE_MATERIAL_THICKNESS','shell_thickness_m'),('DEFORMABLE_MATERIAL_BENDING_STIFFNESS','bend_modulus_pa'),
        ('DEFORMABLE_MATERIAL_ELASTICITY_DAMPING','elasticity_damping'),('DEFORMABLE_MATERIAL_BENDING_DAMPING','bend_damping')):
        value=session.read(session.bind(kind,path+'/Fabric'));native[key]=value.tolist()
        assert np.allclose(value,material[key],atol=1e-8,rtol=1e-5), 'Native cloth property changed: '+key
    x=session.bind('SURFACE_DEFORMABLE_SIM_POSITION',path);v=session.bind('SURFACE_DEFORMABLE_SIM_VELOCITY',path)
    actual_rest=session.read(session.bind('SURFACE_DEFORMABLE_REST_POSITION',path))
    actual_faces=session.read(session.bind('SURFACE_DEFORMABLE_SIM_ELEMENT_INDICES',path),np.int32)
    assert x.shape[0]==1 and x.shape[1]>=len(rest) and x.shape[2]==3
    assert np.allclose(actual_rest[0,:len(rest)],rest,atol=1e-7,rtol=0)
    assert np.array_equal(actual_faces[0,:len(faces)],faces)
    assert not actual_rest[0,len(rest):].any()
    tail=actual_faces[0,len(faces):];assert np.all((tail==0)|(tail==-1))
    (session.out/'sdk-cloth-properties.json').write_text(json.dumps(dict(material=native,nodes=len(rest),faces=len(faces),padding_verified=True),indent=2))
    return x,v
