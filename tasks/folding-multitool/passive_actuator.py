"""Tool-local passive force law. Import only after AppLauncher starts.

The native build defaults to Newton-authored actuators even with PhysX. The
conservative detent force therefore runs at this asset's write_data_to_sim
boundary, with zero PD gains. Friction uses the native implicit joint solver.
The robot's installed actuator path is untouched.
"""
import torch
from isaaclab_physx.assets.articulation.articulation import Articulation as PhysxArticulation


class PassiveToolArticulation(PhysxArticulation):
    def write_data_to_sim(self):
        if not hasattr(self, "_passive_curves"):
            self._passive_curves = {name: torch.tensor(self.cfg.mechanics[name]["potential_knots"],
                           device=self.device, dtype=torch.float32) for name in self.joint_names}
        q_all = self.data.joint_pos.torch
        efforts = torch.zeros_like(q_all)
        for i, name in enumerate(self.joint_names):
            j, k = self.cfg.mechanics[name], self._passive_curves[name]
            q = q_all[:, i]
            ix = torch.clamp(torch.searchsorted(k[:, 0].contiguous(), q.contiguous(), right=True)-1, 0, len(k)-2)
            width = k[ix+1, 0]-k[ix, 0]
            x = torch.clamp((q-k[ix, 0])/width, 0, 1)
            du = (k[ix+1, 1]-k[ix, 1])*6*x*(1-x)/width
            efforts[:, i] = -du
        self.actuators.target_command.set_effort_index(value=efforts, full_data=True)
        self.last_passive_effort = efforts
        self.passive_update_count = getattr(self, "passive_update_count", 0)+1
        super().write_data_to_sim()
