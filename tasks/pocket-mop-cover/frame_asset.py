"""Free physical folding frame for isolated stock-robot handling tests."""
import numpy as np


def frame_asset(origin, metadata_sink, folded=False,snap_throat=False,fold_angle_deg=55.):
    """Author the frame and reset its free root and every passive joint."""
    from isaaclab.assets import ArticulationCfg
    from isaaclab.managers import EventTermCfg
    from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
    from isaaclab.sim.utils import clone
    from isaaclab.utils import configclass
    from isaaclab_arena.assets.asset import Asset
    from frame_native import author_frame,passive_actuators

    def spawn(path,cfg,translation=None,orientation=None,**kwargs):
        from isaaclab.sim.utils.stage import get_current_stage
        stage=get_current_stage()
        metadata_sink.update(author_frame(stage,path=path,origin=origin,snap_throat=snap_throat))
        return stage.GetPrimAtPath(path)

    def reset(env,env_ids):
        import torch
        rig=env.scene['frame']
        root=rig.data.default_root_state.torch[env_ids].clone()
        root[:,:3]=torch.as_tensor(origin,device=env.device)
        origins=getattr(env.scene.env_origins,'torch',env.scene.env_origins)
        root[:,:3]+=origins[env_ids]
        root[:,3:7]=torch.tensor([0.,0.,0.,1.],device=env.device)
        root[:,7:]=0.
        q=rig.data.default_joint_pos.torch[env_ids].clone()
        q[:]=0.
        if folded:
            q[:,rig.joint_names.index('HingeLeft')]=-np.deg2rad(fold_angle_deg)
            q[:,rig.joint_names.index('HingeRight')]=np.deg2rad(fold_angle_deg)
            q[:,rig.joint_names.index('RockerHinge')]=-np.deg2rad(20.)
        else:
            q[:,rig.joint_names.index('HandleY')]=np.deg2rad(70.)
        rig.write_root_state_to_sim(root,env_ids=env_ids)
        rig.write_joint_state_to_sim(q,torch.zeros_like(q),env_ids=env_ids)
        rig.reset(env_ids=env_ids)

    @configclass
    class SpawnCfg(SpawnerCfg):
        func: object=clone(spawn)

    class Frame(Asset):
        def __init__(self):
            super().__init__('frame',tags=['folding-mop-frame','passive-contact-lock'])
        def get_object_cfg(self):
            return self.name,ArticulationCfg(prim_path='{ENV_REGEX_NS}/Frame',spawn=SpawnCfg(),
                init_state=ArticulationCfg.InitialStateCfg(pos=tuple(origin)),actuators=passive_actuators())
        def get_event_cfg(self):
            return self.name,EventTermCfg(func=reset,mode='reset')
    return Frame()
