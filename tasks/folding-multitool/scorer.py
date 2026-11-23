"""Independent observed deployment gate. Never reads the controller's stage."""
import numpy as np
from load_score import LoadGate
from retrieval_score import RetrievalGate


class DeploymentScorer:
    def __init__(self):
        self.folded_seen = False
        self.open_seen = False
        self.cycle_seen = False
        self.tip_down_seen = False
        self.open_count = self.close_count = self.down_count = 0
        self.peak_gap = 0.
        self.score = 0.
        self.invalid_reason = None
        self.load = LoadGate()
        self.retrieval = RetrievalGate()

    def update(self, m):
        q = m["q"]
        values = np.array(list(q.values())+m["root_pos"])
        if not np.isfinite(values).all():
            self.invalid_reason = "nonfinite_state"
            self.score = 0.
            return
        if m["step"] == 0:
            self.folded_seen = (abs(q["fold_a"]-np.pi) < .1 and abs(q["fold_b"]+np.pi) < .1)
        self.peak_gap = max(self.peak_gap, m["gap_m"])
        deployed = max(abs(q["fold_a"]), abs(q["fold_b"])) < np.deg2rad(5)
        # Each arm must pinch its own handle between both original fingers.
        p = np.asarray(m["pads_n"])
        opposite_handles = (min(p[0, 2], p[1, 2], p[2, 3], p[3, 3]) > .1 or
                            min(p[0, 3], p[1, 3], p[2, 2], p[3, 2]) > .1)
        head_clear = p[:, :2].max() < .1
        # During functional use only the original fingers may load the tool.
        # A forearm supporting the head must not count as handle-mediated use.
        links_clear = True
        for side in ("left", "right"):
            names = m.get(side+"_contact_names", [])
            forces = m.get(side+"_contact_forces_n", [])
            if not names or len(names) != len(forces):
                links_clear = False
                continue
            for name, force in zip(names, forces):
                if name not in ("tip_left", "tip_right") and np.linalg.norm(force) >= .1:
                    links_clear = False
        valid = self.folded_seen and deployed and opposite_handles and head_clear and links_clear
        self.open_count = self.open_count+1 if valid and m["gap_m"] >= .014 else 0
        if self.open_count >= 10:
            self.open_seen = True
        self.close_count = self.close_count+1 if valid and self.open_seen and m["gap_m"] <= .001 else 0
        if self.close_count >= 10:
            self.cycle_seen = True
        self.down_count = self.down_count+1 if valid and self.cycle_seen and m["tip_axis"][2] < -.94 else 0
        if self.down_count >= 20:
            self.tip_down_seen = True
        self.score = .25 if self.cycle_seen and not self.invalid_reason else 0.
        self.load.update(m, valid and self.cycle_seen and self.tip_down_seen and not self.invalid_reason)
        self.retrieval.update(m, valid and self.cycle_seen and self.tip_down_seen and not self.invalid_reason)
        if not self.invalid_reason:
            if self.retrieval.carried:
                self.score = .5
            if self.retrieval.passed:
                self.score = 1.

    def snapshot(self):
        return {"score": self.score, "folded_reset_seen": self.folded_seen,
                "open_14mm_seen": self.open_seen, "closed_1mm_seen": self.cycle_seen,
                "tip_down_seen": self.tip_down_seen, "peak_gap_m": self.peak_gap,
                "deployment_cycle_pass": self.cycle_seen and self.tip_down_seen and not self.invalid_reason,
                "loaded_gauge_pass": self.load.passed and not self.invalid_reason,
                "loaded_hold_steps": self.load.count, "loaded_hold_valid": self.load.last_valid,
                "loaded_translation_span_m": self.load.translation_span,
                "loaded_rotation_span_deg": self.load.rotation_span,
                **self.retrieval.snapshot(),
                "retrieval_pass": self.retrieval.passed and not self.invalid_reason,
                "invalid_reason": self.invalid_reason}
