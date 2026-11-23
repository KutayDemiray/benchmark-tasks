"""Stock-URDF IK and 14-dimensional robot commands. Never writes tool state."""
import numpy as np
import copy
from yam_arena import ik
from yam_arena.constants import READY

HZ = 20


def orientation(closing, approach):
    x = np.asarray(closing, float)
    x /= np.linalg.norm(x)
    z = np.asarray(approach, float)
    z -= x*np.dot(z, x)
    z /= np.linalg.norm(z)
    return np.column_stack([x, np.cross(z, x), z])


class Arm:
    def __init__(self, side):
        self.side = side
        self.base = np.array([0., .305 if side == "left" else -.305, .75])
        self.q = np.asarray(READY, float)
        self.grip = 1.
        self.pos, self.rot = ik.tcp_pose(self.q, self.base)
        self.error = (0., 0.)
        self.free_joint_fallbacks = []

    def aim(self, pos, rot):
        q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), self.q, base_pos=self.base, iters=180)
        self.error = (pe, re)
        if pe > .001 or re > np.deg2rad(1):
            raise RuntimeError(f"{self.side} IK failure at {np.round(pos, 4)}: {pe*1000:.2f}mm {np.degrees(re):.2f}deg")
        self.q, self.pos, self.rot = q, np.asarray(pos).copy(), np.asarray(rot).copy()

    def move(self, pos, rot, seconds=2., grip=None):
        p0, r0, g0 = self.pos.copy(), self.rot.copy(), self.grip
        g1 = self.grip if grip is None else grip
        n = max(1, round(seconds*HZ))
        for i in range(1, n+1):
            t = ik.smoothstep(i/n)
            self.aim(p0+t*(np.asarray(pos)-p0), ik.slerp_matrix(r0, rot, t))
            self.grip = g0+t*(g1-g0)
            yield

    def set_grip(self, grip, seconds=1.):
        yield from self.move(self.pos, self.rot, seconds, grip)

    def move_free(self, pos, rot, seconds=3., grip=.35):
        """Plan a free-arm move before executing it.

        Prefer the Cartesian path. If that crosses a wrist singularity/limit,
        use a bounded joint-space path to the same reachable clearance pose.
        This method is only called after release or before initial acquisition.
        Held-tool paths always use strict Cartesian IK without this fallback.
        """
        trial, commands = copy.deepcopy(self), []
        try:
            for _ in trial.move(pos, rot, seconds, grip):
                commands.append((trial.q.copy(), trial.grip))
        except RuntimeError as exc:
            choices = []
            for seed in (self.q, np.asarray(READY, float)):
                q, pe, re = ik.solve(pos, ik.matrix_to_quat_xyzw(rot), seed, base_pos=self.base, iters=500)
                if pe <= .001 and re <= np.deg2rad(1):
                    choices.append((np.linalg.norm(q-self.q), q))
            if not choices:
                raise RuntimeError(f"{self.side} free clearance pose unreachable: {pos}") from exc
            goal = min(choices, key=lambda pair: pair[0])[1]
            self.free_joint_fallbacks.append({"target": np.asarray(pos).tolist(), "reason": str(exc)})
            commands = []
            for i in range(1, max(1, round(seconds*HZ))+1):
                s = ik.smoothstep(i/max(1, round(seconds*HZ)))
                commands.append((self.q+s*(goal-self.q), self.grip+s*(grip-self.grip)))
        for q, g in commands:
            self.q, self.grip = q, g
            self.pos, self.rot = ik.tcp_pose(q, self.base)
            yield

    def command(self):
        return np.r_[self.q, self.grip]


def together(*motions):
    pending = [iter(m) for m in motions]
    while pending:
        remaining = []
        for motion in pending:
            try:
                next(motion)
                remaining.append(motion)
            except StopIteration:
                pass
        pending = remaining
        if pending:
            yield


def hold(seconds):
    for _ in range(round(HZ*seconds)):
        yield


def checked_action(left, right):
    from inspect_robots.types import Action
    a = np.r_[left.command(), right.command()]
    if a.shape != (14,) or not np.isfinite(a).all():
        raise ValueError("Action must contain 14 finite values")
    for offset in (0, 7):
        if np.any(a[offset:offset+6] < ik.LOWER) or np.any(a[offset:offset+6] > ik.UPPER):
            raise ValueError("Joint target outside stock URDF limits")
        if not 0 <= a[offset+6] <= 1:
            raise ValueError("Gripper outside stock command range")
    return Action(data=a.tolist())
