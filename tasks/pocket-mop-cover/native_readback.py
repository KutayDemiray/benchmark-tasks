"""Minimal public-tensor access for passive host-readback diagnostics only.

These adapters have no controller, actuator, runtime force or integration code.
Their state-write methods are used exclusively for diagnostic initialization.
"""


class RigidReadback:
    def __init__(self, simulation_view, path):
        import torch
        self.root_view = simulation_view.create_rigid_body_view(path)
        assert self.root_view.count == 1
        self.device = str(self.root_view.get_transforms().device)
        self.indices = torch.tensor([0], dtype=torch.int32, device=self.device)

    def write_root_link_pose_to_sim_index(self, *, root_pose):
        self.root_view.set_transforms(root_pose.to(self.device).contiguous(), self.indices)

    def write_root_link_velocity_to_sim_index(self, *, root_velocity):
        self.root_view.set_velocities(root_velocity.to(self.device).contiguous(), self.indices)

    def update(self, dt):
        pass  # Native views read simulation state directly; there is no cache.


class SurfaceReadback:
    def __init__(self, simulation_view, path):
        import torch
        self.root_view = simulation_view.create_surface_deformable_body_view(path)
        assert self.root_view.count == 1
        self.device = str(self.root_view.get_simulation_nodal_positions().device)
        self.indices = torch.tensor([0], dtype=torch.int32, device=self.device)

    def write_nodal_state_to_sim_index(self, state):
        positions = self.root_view.get_simulation_nodal_positions().clone()
        velocities = self.root_view.get_simulation_nodal_velocities().clone()
        positions[:, :state.shape[1]] = state[:, :, :3].to(positions.device)
        velocities[:, :state.shape[1]] = state[:, :, 3:].to(velocities.device)
        self.root_view.set_simulation_nodal_positions(positions, self.indices)
        self.root_view.set_simulation_nodal_velocities(velocities, self.indices)

    def update(self, dt):
        pass  # No cached data and no state writes.

    def reset(self):
        pass  # Initialization writes are explicit; no per-step reset exists.


class UsdSurfaceReadback:
    """Read native USD mesh exports when CPU tensor surface access is unavailable."""

    def __init__(self, stage, path):
        from pxr import UsdGeom
        import numpy as np
        self.mesh = UsdGeom.Mesh(stage.GetPrimAtPath(path))
        self.root_view = self
        self.device = 'cpu'
        self.initial = np.array(self.mesh.GetPointsAttr().Get(), dtype=np.float32)
        self.native_updates = 0
        self.maximum_displacement = 0.
        self.maximum_speed = 0.

    def get_simulation_nodal_positions(self):
        import numpy as np
        return np.array(self.mesh.GetPointsAttr().Get(), dtype=np.float32)[None]

    def get_simulation_nodal_velocities(self):
        import numpy as np
        value = self.mesh.GetVelocitiesAttr().Get()
        assert value is not None and len(value) == len(self.initial), 'Missing native surface velocities'
        return np.array(value, dtype=np.float32)[None]

    def get_rest_nodal_positions(self):
        import numpy as np
        return np.array(self.mesh.GetPrim().GetAttribute('omniphysics:restShapePoints').Get(), dtype=np.float32)[None]

    def write_nodal_state_to_sim_index(self, state):
        # State is authored before cooking; this entry point only verifies it.
        import numpy as np
        from robot_gate import array
        assert self.native_updates == 0
        expected = array(state)
        error = float(np.max(abs(self.get_simulation_nodal_positions()-expected[:, :, :3])))
        assert error < 1e-7, f'Initial USD surface state differs from source by {error} m'
        assert np.max(abs(self.get_simulation_nodal_velocities()-expected[:, :, 3:])) < 1e-7

    def update(self, dt):
        import numpy as np
        from omni.physx import get_physx_interface
        # Native readback updates USD. No user-authored pose/velocity writes occur.
        get_physx_interface().update_transformations(False, True, True, False)
        self.native_updates += 1
        self.maximum_displacement = max(self.maximum_displacement,
            float(np.linalg.norm(self.get_simulation_nodal_positions()[0]-self.initial, axis=1).max()))
        self.maximum_speed = max(self.maximum_speed,
            float(np.linalg.norm(self.get_simulation_nodal_velocities()[0], axis=1).max()))

    def reset(self):
        pass  # All state is authored before cooking; no reset writes occur.


class ArticulationReadback:
    """Read a native articulation without framework asset callbacks or actuators."""

    def __init__(self, simulation_view, path):
        import torch
        self.simulation_view = simulation_view
        self.root_view = simulation_view.create_articulation_view(path)
        assert self.root_view.count == 1
        self.device = str(self.root_view.get_root_transforms().device)
        self.indices = torch.tensor([0], dtype=torch.int32, device=self.device)
        self.body_names = list(self.root_view.shared_metatype.link_names)
        self.joint_names = list(self.root_view.shared_metatype.dof_names)
        self.data = self

    def write_root_link_pose_to_sim_index(self, *, root_pose):
        self.root_view.set_root_transforms(root_pose.to(self.device).contiguous(), self.indices)

    def write_root_link_velocity_to_sim_index(self, *, root_velocity):
        self.root_view.set_root_velocities(root_velocity.to(self.device).contiguous(), self.indices)

    def write_joint_position_to_sim_index(self, *, position):
        self.root_view.set_dof_positions(position.to(self.device).contiguous(), self.indices)

    def write_joint_velocity_to_sim_index(self, *, velocity):
        self.root_view.set_dof_velocities(velocity.to(self.device).contiguous(), self.indices)

    @property
    def body_link_pose_w(self):
        self.simulation_view.update_articulations_kinematic()
        return self.root_view.get_link_transforms()

    def update(self, dt):
        pass  # Native views have no framework state or actuator updates.


class _SingleLinkView:
    """Present a rigid body's readback in the same shape as a one-link group."""

    def __init__(self, view):
        self.view = view

    def get_link_transforms(self):
        return self.view.get_transforms()[:, None, :]

    def get_link_velocities(self):
        return self.view.get_velocities()[:, None, :]

    def get_masses(self):
        return self.view.get_masses().reshape(1, 1)

    def get_coms(self):
        return self.view.get_coms().reshape(1, 1, 7)

    def get_inertias(self):
        return self.view.get_inertias().reshape(1, 1, 9)

    def get_dof_positions(self):
        return self.view.get_transforms().new_empty((1, 0))

    def __getattr__(self, name):
        return getattr(self.view, name)


class RigidLinkReadback(RigidReadback):
    """Represent an ordinary rigid link in a mixed native mechanism readback."""

    def __init__(self, simulation_view, path):
        super().__init__(simulation_view, path)
        self.root_view = _SingleLinkView(self.root_view)
        self.body_names = [path.rsplit('/', 1)[1]]
        self.joint_names = []
        self.data = self

    def write_joint_position_to_sim_index(self, *, position):
        assert position.numel() == 0

    def write_joint_velocity_to_sim_index(self, *, velocity):
        assert velocity.numel() == 0

    @property
    def body_link_pose_w(self):
        return self.root_view.get_link_transforms()
