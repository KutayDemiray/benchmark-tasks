"""Bounded, local scripted deployment through stock robot contacts only.

Desired tool angles below generate *robot TCP paths*. They are never written
to the tool. Each transition checks measured tool state and opposing contacts.
"""
import numpy as np
from yam_arena import ik
from yam_arena.urdf import quat_xyzw_to_matrix
from mechanism import load_spec, link_transforms, transform, jaw_angle_for_gap, rz
from motion import Arm, orientation, together, hold, checked_action, HZ
import adapter


class Context:
    def __init__(self):
        self.left, self.right = Arm("left"), Arm("right")
        self.spec = load_spec()
        self.stage = "reset"
        self.events = []

    def action(self):
        return checked_action(self.left, self.right)

    def body(self, name):
        pose = adapter.last_record["link_poses"][name]
        return transform(pose[:3], quat_xyzw_to_matrix(pose[3:]))

    def set_stage(self, stage):
        self.stage = stage
        self.events.append({"step": adapter.last_record["step"], "stage": stage})

    def require_grasp(self, side, body):
        index = ("jaw_a", "jaw_b", "handle_a", "handle_b").index(body)
        start = 0 if side == "left" else 2
        forces = np.asarray(adapter.last_record["pads_n"])[start:start+2, index]
        if np.min(forces) < .1:
            raise RuntimeError(f"No opposing {side} contacts on {body}: {forces}")

    def require_fold(self, name, angle, tolerance=.12):
        observed = adapter.last_record["q"][name]
        if abs(observed-angle) > tolerance:
            raise RuntimeError(f"{name} did not follow contact deployment: wanted {angle:.3f}, observed {observed:.3f}")

    def local_grasp(self, arm, body):
        return np.linalg.inv(self.body(body))@transform(arm.pos, arm.rot)

    def arc(self, grasps, targets, seconds=5., root_target=None):
        """Move held robot TCPs along a consistent linkage path; observe afterward."""
        root0 = self.body("jaw_a")
        root1 = root0 if root_target is None else root_target
        q0 = adapter.last_record["q"].copy()
        q1 = {**q0, **targets}
        n = round(seconds*HZ)
        for i in range(1, n+1):
            s = ik.smoothstep(i/n)
            root = transform(root0[:3, 3]+s*(root1[:3, 3]-root0[:3, 3]),
                             ik.slerp_matrix(root0[:3, :3], root1[:3, :3], s))
            q = {k: q0[k]+s*(q1[k]-q0[k]) for k in q0}
            links = link_transforms(self.spec, q)
            for arm, body, local in grasps:
                tcp = root@links[body]@local
                arm.aim(tcp[:3, 3], tcp[:3, :3])
            yield


def grasp_pose(ctx, body, band, side):
    """Shallow broad-face grasp: 8 mm outboard TCP, 13 mm stock fingertip overhang."""
    tr = ctx.body(body)
    # Orientation comes from the measured body, position from its physical band.
    sign = 1 if body == "handle_a" else -1
    local = np.array([band, sign*.008, 0.])
    p = tr[:3, 3]+tr[:3, :3]@local
    # World-outboard approach is used at acquisition/regrasp.
    r = orientation((0, 0, 1 if side == "left" else -1),
                    (0, -1 if side == "left" else 1, 0))
    return p, r


def approach(arm, p, r):
    pre = p-r[:, 2]*.07
    high = pre.copy(); high[2] += .08
    # A 33 mm aperture clears the handle but avoids full-open fingers striking
    # the table on horizontal approach; stock fingers/limits are untouched.
    yield from arm.move_free(high, r, 3., grip=.35)
    yield from arm.move(pre, r, 1.5)
    yield from arm.move(p, r, 2.)


def acquire_head(ctx, arm):
    # Handle A is open, exposing jaw B. Grip that actual jaw body, which is
    # directly connected to fold B; do not rely on a nonexistent head lock.
    assert arm.side == "right"
    tr = ctx.body("jaw_b")
    p = tr[:3, 3]+tr[:3, :3]@np.array([.020, -.016, 0.])
    r = orientation(-tr[:3, 2], tr[:3, 1])
    yield from approach(arm, p, r)
    yield from arm.set_grip(0.)
    yield from hold(.5)
    ctx.require_grasp(arm.side, "jaw_b")
    if np.asarray(adapter.last_record["pads_n"])[2:4, 2:].max() > .1:
        raise RuntimeError("Jaw-support grip also contacted a handle")


def acquire_width(ctx, arm, body, band):
    """At half fold, pinch the 15 mm handle width from above and outboard.

    This regrasp changes the torque path and keeps the wrist within its stock
    limits during the remaining 90 degrees. The U-channel plate edges are real
    colliders; no hidden solid fills the hollow handle.
    """
    def pose():
        tr = ctx.body(body)
        # Insert far enough for both fingers to straddle the real shell edges.
        # Eight mm above the shell center left the sloping finger tips too high
        # after normal stock-robot compliance and the one-arm tool's sag.
        p = tr[:3, 3]+tr[:3, :3]@np.array([band, 0., .003])
        return p, orientation(-tr[:3, 1], .6*tr[:3, 0]-.8*tr[:3, 2])

    p, rot = pose()
    high = p-.04*rot[:, 2]+[0., .08 if arm.side == "left" else -.12, .10]
    high[2] = min(high[2], .96 if arm.side == "left" else .98)
    # Rotate the empty gripper outside the assembly. A direct joint-space
    # fallback to the old near pose swept a finger through the free handle.
    for _ in arm.move_free(high, rot, 3., grip=.35):
        start = 0 if arm.side == "left" else 2
        if np.asarray(adapter.last_record["pads_n"])[start:start+2].max() > .1:
            raise RuntimeError("Empty width-regrasp transfer touched the tool")
        yield
    p, rot = pose()
    yield from arm.move(p-.04*rot[:, 2], rot, 2.)
    # Refresh after the unheld head has settled, before the contact approach.
    p, rot = pose()
    yield from arm.move(p, rot, 2.)
    yield from arm.set_grip((ctx.spec["geometry"]["handle_width"]-.007)/.0939)
    yield from hold(.5)
    ctx.require_grasp(arm.side, body)


def release(arm):
    yield from arm.set_grip(.5, .8)
    if arm.rot[2, 2] < -.4:
        yield from arm.move(arm.pos-arm.rot[:, 2]*.06, arm.rot, 1.5)
    else:
        # A long horizontal retreat folds the wrist against its limit here.
        # Clear the tool first, then rise; all motion still uses stock actions.
        yield from arm.move(arm.pos-arm.rot[:, 2]*.02, arm.rot, .7)
        yield from arm.move(arm.pos-arm.rot[:, 2]*.02+[0., 0., .04], arm.rot, 1.2)


def working_grasp(body):
    """A shallow diagonal pinch with room to open the jaws when tip-down.

    The original purely lateral pinch put the left wrist at its bend limit
    after turning tip-down. A 25-degree approach along the handle gives it
    travel. Twelve mm outboard keeps the diagonal fingertips off the other
    handle; checked against the actual stock collision meshes.
    """
    sign = 1 if body == "handle_a" else -1
    band = -.064 if body == "handle_a" else -.030
    tilt = np.deg2rad(25.)
    return transform([band, -sign*.012, 0.],
                     orientation([0, 0, sign], [np.sin(tilt), sign*np.cos(tilt), 0.]))


def script(ctx, mode):
    l, r = ctx.left, ctx.right
    # Nominal 12.5 N per finger at the stock 5000 N/m gain. The 5 mm
    # aperture undershoot provides torsional grip for the finite end detent.
    grip = (.01511-.005)/.0939
    ctx.set_stage("check_stationary_folded_reset")
    rest = ctx.body("jaw_a")
    payload_start = (np.asarray(adapter.last_record["payload"]["pose"][:3])
                     if mode == "retrieval" else None)
    yield from hold(2.)
    drift = np.linalg.norm(ctx.body("jaw_a")[:3, 3]-rest[:3, 3])
    if drift > .002:
        raise RuntimeError(f"Unheld tool drifted {drift*1000:.2f} mm in two seconds")
    if max(abs(v) for v in adapter.last_record["qvel"].values()) > .05:
        raise RuntimeError("Unheld passive joints have not settled below 0.05 rad/s")
    if payload_start is not None:
        from retrieval_scene import POCKET_FLOOR, POCKET_RIM
        payload = adapter.last_record["payload"]
        if (np.linalg.norm(np.asarray(payload["pose"][:3])-payload_start) > .002 or
                abs(payload["low"][2]-POCKET_FLOOR) > .001 or payload["high"][2] > POCKET_RIM-.002):
            raise RuntimeError("Unheld coupling nut did not remain settled in its recess")
    ctx.set_stage("acquire_folded_handles")
    lp, lr = grasp_pose(ctx, "handle_b", -.030, "left")
    rp, rr = grasp_pose(ctx, "handle_a", -.064, "right")
    yield from together(approach(l, lp, lr), approach(r, rp, rr))
    yield from together(l.set_grip(grip), r.set_grip(grip))
    yield from hold(.75)
    ctx.require_grasp("left", "handle_b")
    ctx.require_grasp("right", "handle_a")
    grasps = [(l, "handle_b", ctx.local_grasp(l, "handle_b")),
              (r, "handle_a", ctx.local_grasp(r, "handle_a"))]
    ctx.set_stage("lift_folded")
    root = ctx.body("jaw_a"); root[:3, 3] += [.05, 0., .06]
    yield from ctx.arc(grasps, {}, 3., root)
    yield from hold(1.)
    ctx.require_grasp("left", "handle_b")
    ctx.require_grasp("right", "handle_a")
    if mode == "grasp":
        return

    grasps = [(l, "handle_b", ctx.local_grasp(l, "handle_b")),
              (r, "handle_a", ctx.local_grasp(r, "handle_a"))]

    # The closed B stop provides the reaction while A opens. Rotate the head
    # 90 degrees so neither wrist must rotate through the full 180-degree fold.
    ctx.set_stage("deploy_a_against_closed_b_stop")
    root = ctx.body("jaw_a"); root[:3, :3] = rz(np.pi/2)
    yield from ctx.arc(grasps, {"fold_a": 0., "fold_b": -np.pi, "jaw": 0.}, 6., root)
    yield from hold(.5)
    ctx.require_fold("fold_a", 0., np.deg2rad(5))
    ctx.require_fold("fold_b", -np.pi)
    ctx.require_grasp("left", "handle_b")
    ctx.require_grasp("right", "handle_a")

    ctx.set_stage("bring_one_open_handle_to_handover_pose")
    root = ctx.body("jaw_a")
    root[:3, 3] = [.34, 0., .85]
    root[:3, :3] = rz(np.pi/2)
    grasps = [(l, "handle_b", ctx.local_grasp(l, "handle_b")),
              (r, "handle_a", ctx.local_grasp(r, "handle_a"))]
    yield from ctx.arc(grasps, {}, 3., root)

    ctx.set_stage("right_regrasp_across_handle_width")
    yield from release(r)
    yield from acquire_width(ctx, r, "handle_a", -.064)
    ctx.set_stage("orient_exposed_jaw_for_right_grasp")
    root = ctx.body("jaw_a"); root[:3, :3] = np.eye(3)
    # Translate while turning to keep the held left wrist away from its limit.
    root[:3, 3] = [.38, 0., .90]
    yield from ctx.arc([(l, "handle_b", ctx.local_grasp(l, "handle_b")),
                       (r, "handle_a", ctx.local_grasp(r, "handle_a"))], {}, 4., root)
    yield from hold(.5)
    ctx.require_fold("fold_a", 0.)
    ctx.require_grasp("right", "handle_a")

    # Keep the original broad-face B grasp. A sole width grip on A cannot
    # resist the cantilever's gravity torque about its closing axis: the tool
    # rolled ~25 degrees in that grip during the abandoned tail handover.
    # Stock collision meshes permit the head grip and B's first 90 degrees
    # with this 30 mm band, so no tail handover is needed.
    ctx.set_stage("right_handover_to_exposed_jaw_b")
    yield from release(r)
    yield from acquire_head(ctx, r)
    ctx.set_stage("open_b_to_width_regrasp")
    yield from ctx.arc([(l, "handle_b", ctx.local_grasp(l, "handle_b"))], {"fold_b": -np.pi/2}, 6.)
    yield from hold(.5)
    ctx.require_fold("fold_b", -np.pi/2)
    ctx.set_stage("left_regrasp_across_handle_width")
    yield from release(l)
    yield from acquire_width(ctx, l, "handle_b", -.030)
    ctx.set_stage("deploy_handle_b")
    yield from ctx.arc([(l, "handle_b", ctx.local_grasp(l, "handle_b"))], {"fold_b": 0.}, 6.)
    yield from hold(.5)
    ctx.require_fold("fold_b", 0.)

    ctx.set_stage("bring_deployed_tool_to_working_regrasp")
    root = ctx.body("jaw_a"); root[0, 3] = .40
    yield from ctx.arc([(l, "handle_b", ctx.local_grasp(l, "handle_b")),
                       (r, "jaw_b", ctx.local_grasp(r, "jaw_b"))], {}, 3., root)

    ctx.set_stage("left_regrasp_working_handle")
    yield from release(l)
    tr = ctx.body("handle_b")@working_grasp("handle_b")
    yield from approach(l, tr[:3, 3], tr[:3, :3])
    yield from l.set_grip(grip)
    yield from hold(.5)
    ctx.require_grasp("left", "handle_b")
    ctx.set_stage("right_regrasp_working_handle")
    yield from release(r)
    tr = ctx.body("handle_a")@working_grasp("handle_a")
    yield from approach(r, tr[:3, 3], tr[:3, :3])
    yield from r.set_grip(grip)
    yield from hold(.5)
    ctx.require_grasp("right", "handle_a")
    grasps = [(l, "handle_b", ctx.local_grasp(l, "handle_b")),
              (r, "handle_a", ctx.local_grasp(r, "handle_a"))]
    ctx.set_stage("open_working_jaws")
    yield from ctx.arc(grasps, {"jaw": jaw_angle_for_gap(ctx.spec, .016)}, 3.)
    yield from hold(1.)
    ctx.require_fold("fold_a", 0., np.deg2rad(5))
    ctx.require_fold("fold_b", 0., np.deg2rad(5))
    ctx.set_stage("close_working_jaws")
    yield from ctx.arc(grasps, {"jaw": 0.}, 3.)
    yield from hold(1.)
    ctx.set_stage("orient_tip_down")
    root = ctx.body("jaw_a")
    root[:3, :3] = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])
    root[2, 3] = .90
    yield from ctx.arc(grasps, {}, 5., root)
    yield from hold(1.5)
    if not adapter.monitor.snapshot()["deployment_cycle_pass"]:
        raise RuntimeError("Observed deployment/jaw-cycle gate failed")
    if mode == "load-gate":
        from load_controller import script_load
        yield from script_load(ctx)
    elif mode == "retrieval":
        from retrieval_controller import script_retrieval
        yield from script_retrieval(ctx)
