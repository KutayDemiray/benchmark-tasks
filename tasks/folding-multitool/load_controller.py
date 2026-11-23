"""Local jaw-mediated gravity-load test, after the complete deployment gate."""
import numpy as np
from yam_arena.urdf import quat_xyzw_to_matrix
from mechanism import transform, rz, jaw_angle_for_gap
from motion import hold
from load_gauge import BITE_HEIGHT
from jaw_contact import inner_pinch
import adapter

TIP_DOWN = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])


def handles(ctx):
    return [(ctx.left, "handle_b", ctx.local_grasp(ctx.left, "handle_b")),
            (ctx.right, "handle_a", ctx.local_grasp(ctx.right, "handle_a"))]


def jaw_mid_local(spec, angle):
    g = spec["geometry"]
    a = np.array([g["jaw_tip_x"]-g["bite_from_tip"], g["jaw_tip_half_gap"], 0.])
    b = rz(angle)@(a*[1., -1., 1.])
    return (a+b)/2


def gauge_bite():
    pose = np.asarray(adapter.last_record["payload"]["pose"])
    return pose[:3]+quat_xyzw_to_matrix(pose[3:])@np.array([0., 0., BITE_HEIGHT])


def actual_bite(ctx):
    root = ctx.body("jaw_a")
    return root[:3, 3]+root[:3, :3]@jaw_mid_local(ctx.spec, adapter.last_record["q"]["jaw"])


def root_for_bite(ctx, bite, angle):
    # Level the two needle tips by bisecting the jaw angle. A fixed jaw-A
    # orientation left the opposite contact at the gauge's top corner.
    rot = TIP_DOWN@rz(-angle/2)
    return transform(np.asarray(bite)-rot@jaw_mid_local(ctx.spec, angle), rot)


def position_bite(ctx, point, seconds=3.):
    """Bounded observed-position correction using robot TCP actions only."""
    for attempt in range(4):
        q = adapter.last_record["q"]["jaw"]
        target = root_for_bite(ctx, point, q)
        yield from ctx.arc(handles(ctx), {}, seconds if attempt == 0 else 1.5, target)
        yield from hold(.5)
        if np.linalg.norm(actual_bite(ctx)-point) <= .001:
            return
    raise RuntimeError(f"Jaw bite-point alignment did not converge: error {np.linalg.norm(actual_bite(ctx)-point)*1000:.2f} mm")


def script_load(ctx):
    if not adapter.monitor.snapshot()["deployment_cycle_pass"]:
        raise RuntimeError("Load test cannot precede the complete deployment cycle")
    ctx.set_stage("open_jaws_above_load_gauge")
    angle = jaw_angle_for_gap(ctx.spec, .016)
    above = gauge_bite()+[0., 0., .060]
    yield from ctx.arc(handles(ctx), {"jaw": angle}, 4., root_for_bite(ctx, above, angle))
    yield from hold(.5)
    if adapter.last_record["gap_m"] < .014:
        raise RuntimeError("Working jaws did not open for the 10 mm gauge")
    ctx.set_stage("align_open_jaws_with_gauge")
    yield from position_bite(ctx, above, 2.)
    ctx.set_stage("lower_to_gauge_stem")
    yield from position_bite(ctx, gauge_bite(), 3.)
    ctx.set_stage("pinch_gauge_with_inner_jaws")
    references = handles(ctx)
    # Reduce the commanded gap while measuring actual jaw forces. The gauge
    # stops actual closure; native robot compliance supplies the pinch load.
    # These angles generate wrist targets and are never written to the tool.
    for gap in (.006, .003, .0007):
        angle = jaw_angle_for_gap(ctx.spec, gap)
        yield from ctx.arc(references, {"jaw": angle}, 2., root_for_bite(ctx, gauge_bite(), angle))
        yield from hold(.75)
        payload = adapter.last_record["payload"]
        force = np.linalg.norm(np.asarray(payload["contact_forces_n"])[:2], axis=1)
        if inner_pinch(ctx.spec, adapter.last_record["link_poses"], payload["contact_forces_n"], payload["contact_points_w"]) and force.min() >= 5.:
            break
    else:
        raise RuntimeError("Could not establish two inner-jaw contacts with at least 5 N normal force each")
    ctx.set_stage("seat_loaded_gauge_pinch")
    # Motor compliance displaced the first trial's actual bite about 3 mm up
    # the stem while squeezing. Correct that observed displacement before
    # lifting; retain the loaded grasp offsets, not an unloaded TCP pose.
    for attempt in range(4):
        bite, center = actual_bite(ctx), gauge_bite()
        dz = center[2]-bite[2]
        # The clamped gauge may slide sideways on the table. Its measured
        # inner-face contacts establish engagement; do not demand an arbitrary
        # absolute XY centering while friction resists that loaded motion.
        if abs(dz) <= .001:
            break
        if attempt == 3:
            raise RuntimeError("Loaded bite depth did not converge within 1 mm")
        angle = adapter.last_record["q"]["jaw"]
        yield from ctx.arc(handles(ctx), {}, 2., root_for_bite(ctx, bite+[0., 0., dz], angle))
        yield from hold(.5)
    for attempt in range(4):
        payload = adapter.last_record["payload"]
        force = np.linalg.norm(np.asarray(payload["contact_forces_n"])[:2], axis=1)
        if inner_pinch(ctx.spec, adapter.last_record["link_poses"], payload["contact_forces_n"], payload["contact_points_w"]) and force.min() >= 5.:
            break
        if attempt == 3:
            raise RuntimeError("Seated gauge pinch did not retain 5 N on both jaws")
        angle = min(0., adapter.last_record["q"]["jaw"]+.02)
        yield from ctx.arc(handles(ctx), {"jaw": angle}, 1., root_for_bite(ctx, gauge_bite(), angle))
        yield from hold(.5)
    ctx.require_fold("fold_a", 0., np.deg2rad(5))
    ctx.require_fold("fold_b", 0., np.deg2rad(5))
    ctx.require_grasp("left", "handle_b")
    ctx.require_grasp("right", "handle_a")
    ctx.set_stage("lift_one_newton_gauge")
    root = ctx.body("jaw_a"); root[2, 3] += .060
    # Capture the loaded motor deflection so the lift preserves the pinch.
    yield from ctx.arc(handles(ctx), {}, 4., root)
    ctx.set_stage("hold_one_newton_gauge")
    yield from hold(4.)
    if not adapter.monitor.snapshot()["loaded_gauge_pass"]:
        raise RuntimeError("Observed 1 N / 2 s inner-jaw load gate failed")
