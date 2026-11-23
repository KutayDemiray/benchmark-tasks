"""Independent gravity-load gate; no controller-stage input or extra task score."""
import numpy as np
from collections import deque
from yam_arena.urdf import quat_xyzw_to_matrix
from mechanism import load_spec
from load_gauge import BASE_HEIGHT, STEM_HEIGHT, MASS, GRAVITY
from jaw_contact import inner_pinch


class LoadGate:
    def __init__(self):
        self.count = 0
        self.passed = False
        self.last_valid = False
        self.poses = deque(maxlen=40)
        self.translation_span = self.rotation_span = None

    def update(self, m, functional_handles):
        payload = m.get("payload")
        if payload is None or payload.get("name") != "load_gauge":
            self.count = 0
            self.last_valid = False
            self.poses.clear()
            return
        forces = np.asarray(payload["contact_forces_n"])
        points = np.asarray(payload["contact_points_w"], float)
        valid = (functional_handles and forces.shape[0] >= 6 and
                 np.linalg.norm(forces[2:], axis=1).max() < .1 and
                 inner_pinch(load_spec(), m["link_poses"], forces, points) and
                 payload["low"][2] >= .7475+.040 and
                 np.linalg.norm(payload["velocity"][:3]) < .05 and
                 np.linalg.norm(payload["velocity"][3:]) < 1. and
                 abs(payload["mass_kg"]-MASS) < 1e-6 and payload["mass_kg"]*GRAVITY >= .9999)
        if valid:
            pose = np.asarray(payload["pose"])
            rot = quat_xyzw_to_matrix(pose[3:])
            local = (points[:2]-pose[:3])@rot
            # Both contacts must lie on the stem above its shoulder.
            valid = bool(np.all(local[:, 2] >= BASE_HEIGHT+.001) and
                         np.all(local[:, 2] <= BASE_HEIGHT+STEM_HEIGHT+.001))
        self.last_valid = bool(valid)
        self.count = self.count+1 if valid else 0
        if valid:
            self.poses.append(np.asarray(payload["pose"], float))
        else:
            self.poses.clear()
        self.translation_span = self.rotation_span = None
        if len(self.poses) == 40:
            a = np.array(self.poses)
            self.translation_span = float(np.linalg.norm(a[:, None, :3]-a[None, :, :3], axis=-1).max())
            q = a[:, 3:]/np.linalg.norm(a[:, 3:], axis=1)[:, None]
            self.rotation_span = float(np.degrees(2*np.arccos(np.clip(abs(q@q.T), 0., 1.))).max())
            # Last-substep angular velocities can oscillate in a stiff contact
            # even when the observed body pose is stationary. Use the same
            # finite-window displacement a real-world video can establish.
            if self.translation_span <= .002 and self.rotation_span <= 3.:
                self.passed = True
