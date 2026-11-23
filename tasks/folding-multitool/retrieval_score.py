"""Independent sequential jaw-extraction, transport, and released-placement score."""
from collections import deque
import numpy as np
from jaw_contact import inner_pinch
from mechanism import load_spec
from retrieval_scene import (POCKET_XY, POCKET_HALF, POCKET_FLOOR, POCKET_RIM,
                             TRAY_XY, TRAY_HALF, TRAY_FLOOR, TRAY_TOP)


class RetrievalGate:
    def __init__(self):
        self.initial_in_pocket = False
        self.chain = False
        self.clear_count = 0
        self.extracted_at = None
        self.carried = False
        self.chain_carried = False
        self.extracted_seen = False
        self.release_authorized = False
        self.positions = deque(maxlen=40)
        self.in_tray = False
        self.passed = False
        self.last_pinch = False

    def update(self, m, functional_handles):
        p = m.get("payload")
        if p is None or p.get("name") != "nut":
            return
        low, high = np.asarray(p["low"]), np.asarray(p["high"])
        center = (low+high)/2
        over_pocket = (np.all(low[:2] >= POCKET_XY-POCKET_HALF+.0003) and
                       np.all(high[:2] <= POCKET_XY+POCKET_HALF-.0003))
        at_reset_floor = (over_pocket and abs(low[2]-POCKET_FLOOR) < .002 and
                          high[2] <= POCKET_RIM-.002)
        inside_recess = (over_pocket and low[2] >= POCKET_FLOOR-.001 and
                         high[2] <= POCKET_RIM-.002)
        if m["step"] == 0:
            self.initial_in_pocket = bool(at_reset_floor)
        f = np.asarray(p["contact_forces_n"], float)
        # The coupling nut weighs about 0.0939 N. Keep total non-jaw support
        # below the existing 0.003 N bound, not the deployment gate's 0.1 N
        # tool-contact threshold. Summing norms prevents splitting support
        # across multiple fingers or cancelling opposing normals.
        other_contact_n = float(np.linalg.norm(f[2:], axis=1).sum())
        pinch = (functional_handles and self.initial_in_pocket and len(f) >= 6 and
                 other_contact_n < .003 and
                 inner_pinch(load_spec(), m["link_poses"], f, p["contact_points_w"]))
        self.last_pinch = bool(pinch)
        self.in_tray = bool(np.all(low[:2] >= TRAY_XY-TRAY_HALF+.003) and
                            np.all(high[:2] <= TRAY_XY+TRAY_HALF-.003) and
                            abs(low[2]-TRAY_FLOOR) < .001 and high[2] < TRAY_TOP)
        if pinch:
            if inside_recess:
                self.chain = True
            if self.chain:
                unsupported = np.linalg.norm(p.get("unattributed_normal_n", [np.inf]*3)) < .003
                self.clear_count = self.clear_count+1 if low[2] >= POCKET_RIM+.010 and unsupported else 0
                if self.clear_count >= 20 and self.extracted_at is None:
                    self.extracted_at = center[:2].copy()
                    self.extracted_seen = True
                if self.extracted_at is not None and unsupported and np.linalg.norm(center[:2]-self.extracted_at) >= .030:
                    self.carried = self.chain_carried = True
                # Only loss of a qualified carry above the tray authorizes a
                # release; losing the nut elsewhere cannot be repaired by a push.
                self.release_authorized = bool(self.chain_carried and
                    np.all(low[:2] > TRAY_XY-TRAY_HALF+.003) and
                    np.all(high[:2] < TRAY_XY+TRAY_HALF-.003) and low[2] >= TRAY_FLOOR-.001)
        else:
            self.chain = False
            self.chain_carried = False
            self.clear_count = 0
            self.extracted_at = None
        if (other_contact_n >= .003 or
                np.any(low[:2] < TRAY_XY-TRAY_HALF+.003) or
                np.any(high[:2] > TRAY_XY+TRAY_HALF-.003)):
            self.release_authorized = False
        # Carry credit is a historical achievement; full credit still needs a
        # qualified release, no robot/tool contacts, and geometric separation.
        released = (self.in_tray and self.release_authorized and
                    np.linalg.norm(f, axis=1).max() < .02 and
                    p.get("tool_robot_clearance_m", -1.) >= .020 and
                    np.linalg.norm(p["velocity"][:3]) < .005)
        if released:
            self.positions.append(center.copy())
        else:
            self.positions.clear()
        self.passed = False
        if len(self.positions) == 40:
            a = np.array(self.positions)
            if np.max(np.linalg.norm(a[:, None]-a[None, :], axis=-1)) <= .002:
                self.passed = True

    def snapshot(self):
        return dict(nut_initial_in_pocket=self.initial_in_pocket,
                    jaw_pinch_valid=self.last_pinch, jaw_extraction_seen=self.extracted_seen,
                    jaw_carry_pass=self.carried, qualified_release=self.release_authorized,
                    nut_in_tray=self.in_tray, released_hold_steps=len(self.positions),
                    retrieval_pass=self.passed)
