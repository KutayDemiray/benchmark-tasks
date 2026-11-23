"""Deterministic outcome rubric; not yet connected to a qualified full scene.

Inputs must come from recorded native geometry/contact measurements. Controller
phase names and commands are deliberately absent. Diagnostic rigs cannot score.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Observation:
    step: int
    dt: float
    pocket_depth_m: tuple[float, float]
    pocket_width_fraction: tuple[float, float]
    orientation_correct: bool
    wing_angle_deg: float
    textile_grasped: bool
    wings_externally_supported: bool
    all_cloth_clearance_m: float
    head_xy_m: tuple[float, float]
    head_height_over_support_m: float
    all_grippers_clearance_m: float
    handle_grasped: bool = False
    max_cloth_speed_m_s: float = 0.0
    max_cloth_difference_speed_m_s: float = 0.0
    finite: bool = True
    max_edge_strain: float = 0.0
    max_penetration_m: float = 0.0
    max_seam_gap_m: float = 0.0
    unexpected_constraint: bool = False
    diagnostic_fixture: bool = False
    pocket_orientation_correct: tuple[bool, bool] | None = None
    over_support_surface: bool = True


class OutcomeScorer:
    """Track the physical retention sequence and current partial installation."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Clear all episode history, including latched invalidity."""
        self.last_step = -1
        self.pocket_seconds = [0.0, 0.0]
        self.lift_xy = None
        self.translated = False
        self.return_hold_s = 0.0
        self.retention_verified = False
        self.release_hold_s = 0.0
        self.invalid_reasons = set()

    def update(self, obs):
        """Consume one control interval whose numerical bounds cover its substeps."""
        assert obs.step == self.last_step + 1, "Missing, repeated or reordered observations"
        assert 0 < obs.dt <= .05 + 1e-10, "Expected observations at least at 20 Hz"
        self.last_step = obs.step
        values = (*obs.pocket_depth_m, *obs.pocket_width_fraction,
                  obs.wing_angle_deg, obs.all_cloth_clearance_m, *obs.head_xy_m,
                  obs.head_height_over_support_m, obs.all_grippers_clearance_m,
                  obs.max_edge_strain, obs.max_penetration_m, obs.max_seam_gap_m,
                  obs.max_cloth_speed_m_s, obs.max_cloth_difference_speed_m_s)
        checks = {
            "nonfinite_state": not obs.finite or not all(math.isfinite(x) for x in values),
            "material_strain": obs.max_edge_strain > .2,
            "penetration": obs.max_penetration_m > .001,
            "seam_failure": obs.max_seam_gap_m > .001,
            "unexpected_constraint": obs.unexpected_constraint,
            "diagnostic_fixture": obs.diagnostic_fixture,
            "run_limit": obs.step >= 2400,
        }
        self.invalid_reasons.update(k for k, failed in checks.items() if failed)
        seated = []
        for i in range(2):
            orientation = obs.orientation_correct if obs.pocket_orientation_correct is None else obs.pocket_orientation_correct[i]
            valid = (orientation and obs.pocket_depth_m[i] >= .04
                     and obs.pocket_width_fraction[i] >= .8 and not obs.textile_grasped)
            self.pocket_seconds[i] = self.pocket_seconds[i] + obs.dt if valid else 0.0
            seated.append(valid)
        both = all(seated)
        free_flat = (both and abs(obs.wing_angle_deg) <= 3
                     and not obs.wings_externally_supported)
        if not free_flat:
            self.lift_xy = None
            self.translated = False
            self.return_hold_s = 0.0
            self.retention_verified = False
            self.release_hold_s = 0.0
        elif min(self.pocket_seconds) >= 2 - 1e-9:
            lifted = obs.all_cloth_clearance_m >= .1
            if lifted and not self.retention_verified:
                if self.lift_xy is None:
                    self.lift_xy = obs.head_xy_m
                distance = math.dist(self.lift_xy, obs.head_xy_m)
                self.translated |= distance >= .1
                returned = self.translated and distance <= .01
                self.return_hold_s = self.return_hold_s + obs.dt if returned else 0.0
                self.retention_verified = self.return_hold_s >= 2 - 1e-9
            elif not lifted and not self.retention_verified:
                # An interrupted lift cannot accumulate hold time across contacts.
                self.lift_xy = None
                self.translated = False
                self.return_hold_s = 0.0
            released = (self.retention_verified and -.001 <= obs.head_height_over_support_m <= .01
                        and obs.over_support_surface
                        and not obs.handle_grasped
                        and obs.all_grippers_clearance_m >= .02
                        and obs.max_cloth_speed_m_s <= .03
                        and obs.max_cloth_difference_speed_m_s <= .03)
            self.release_hold_s = self.release_hold_s + obs.dt if released else 0.0
        score = .25 * sum(t >= 2 - 1e-9 for t in self.pocket_seconds)
        complete = self.release_hold_s >= 2 - 1e-9
        if complete:
            score = 1.0
        if self.invalid_reasons:
            score, complete = 0.0, False
        return dict(step=obs.step, score=score, success=complete,
                    invalid_reasons=sorted(self.invalid_reasons),
                    pocket_hold_seconds=self.pocket_seconds.copy(),
                    retention_verified=self.retention_verified,
                    released_hold_seconds=self.release_hold_s,
                    termination_reason="invalid" if self.invalid_reasons else
                    "success" if complete else "running")
