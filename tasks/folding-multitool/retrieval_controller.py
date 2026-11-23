"""Contact-only extraction and release after the demonstrated deployment cycle."""
import numpy as np
from yam_arena.urdf import quat_xyzw_to_matrix
from mechanism import transform, rz, jaw_angle_for_gap
from motion import hold
from load_controller import handles, jaw_mid_local, actual_bite, TIP_DOWN
from jaw_contact import inner_pinch
from retrieval_scene import NUT_HEIGHT, TRAY_XY, TRAY_FLOOR
import adapter


def nut_center():
    p = np.asarray(adapter.last_record["payload"]["pose"])
    return p[:3]+quat_xyzw_to_matrix(p[3:])@np.array([0., 0., NUT_HEIGHT/2])


def root_for_bite(ctx, bite, angle):
    # Bisect the central-jaw angle so the two needle tips stay level. The 18 mm
    # coupling nut gives 7 mm nominal floor clearance at its mid-height bite.
    rot = TIP_DOWN@rz(-angle/2)
    return transform(np.asarray(bite)-rot@jaw_mid_local(ctx.spec, angle), rot)


def position_bite(ctx, point, seconds=3.):
    for attempt in range(4):
        angle = adapter.last_record["q"]["jaw"]
        yield from ctx.arc(handles(ctx), {}, seconds if attempt == 0 else 1.5,
                           root_for_bite(ctx, point, angle))
        yield from hold(.5)
        if np.linalg.norm(actual_bite(ctx)-point) <= .0008:
            return
    raise RuntimeError("Nut approach did not align the observed bite point within 0.8 mm")


def script_retrieval(ctx):
    if not adapter.monitor.snapshot()["deployment_cycle_pass"]:
        raise RuntimeError("Nut retrieval requires complete functional deployment")
    ctx.set_stage("open_jaws_above_recess")
    angle = jaw_angle_for_gap(ctx.spec, .016)
    above = nut_center()+[0., 0., .065]
    yield from ctx.arc(handles(ctx), {"jaw": angle}, 4., root_for_bite(ctx, above, angle))
    yield from hold(.5)
    if adapter.last_record["gap_m"] < .014:
        raise RuntimeError("Jaws did not open sufficiently for insertion")
    ctx.set_stage("align_jaws_with_recess")
    yield from position_bite(ctx, above, 2.)
    ctx.set_stage("insert_needles_around_nut")
    yield from position_bite(ctx, nut_center(), 3.)
    ctx.set_stage("pinch_nut_between_inner_faces")
    references = handles(ctx)
    for gap in (.006, .003, .0007):
        angle = jaw_angle_for_gap(ctx.spec, gap)
        yield from ctx.arc(references, {"jaw": angle}, 2., root_for_bite(ctx, nut_center(), angle))
        yield from hold(.5)
        p = adapter.last_record["payload"]
        if inner_pinch(ctx.spec, adapter.last_record["link_poses"], p["contact_forces_n"], p["contact_points_w"]):
            break
    else:
        raise RuntimeError("Nut pinch lacks opposing inner-jaw contact")
    for name in ("fold_a", "fold_b"):
        ctx.require_fold(name, 0., np.deg2rad(5))
    ctx.require_grasp("left", "handle_b")
    ctx.require_grasp("right", "handle_a")
    ctx.set_stage("extract_nut_from_pocket")
    root = ctx.body("jaw_a"); root[2, 3] += .065
    yield from ctx.arc(handles(ctx), {}, 4., root)
    yield from hold(1.5)
    if not adapter.monitor.snapshot()["jaw_extraction_seen"]:
        raise RuntimeError("Independent scorer did not observe jaw-mediated extraction")
    ctx.set_stage("carry_nut_over_tray")
    root = ctx.body("jaw_a")
    root[:2, 3] += TRAY_XY-nut_center()[:2]
    yield from ctx.arc(handles(ctx), {}, 5., root)
    yield from hold(.5)
    if not adapter.monitor.snapshot()["jaw_carry_pass"]:
        raise RuntimeError("Independent scorer did not observe a continuous jaw carry")
    ctx.set_stage("lower_nut_into_tray")
    root = ctx.body("jaw_a")
    root[2, 3] += TRAY_FLOOR+NUT_HEIGHT/2+.001-nut_center()[2]
    yield from ctx.arc(handles(ctx), {}, 3., root)
    yield from hold(.5)
    ctx.set_stage("release_nut_by_opening_jaws")
    angle = jaw_angle_for_gap(ctx.spec, .016)
    yield from ctx.arc(handles(ctx), {"jaw": angle}, 3., root_for_bite(ctx, actual_bite(ctx), angle))
    yield from hold(.5)
    ctx.set_stage("withdraw_empty_jaws")
    root = ctx.body("jaw_a"); root[2, 3] += .065
    yield from ctx.arc(handles(ctx), {}, 3., root)
    ctx.set_stage("verify_released_nut_in_tray")
    yield from hold(2.5)
    if not adapter.monitor.snapshot()["retrieval_pass"]:
        raise RuntimeError("Released-placement scorer did not award full success")
