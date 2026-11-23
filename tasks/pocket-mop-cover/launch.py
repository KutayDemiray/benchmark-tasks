"""Supervise one native task tree under cooperative or exclusive GPU admission."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
OUTPUT = OUTPUT_ROOT
from gpu_coordination import reservation, snapshot


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--name", required=True)
    p.add_argument("--experiment", choices=("probe", "robot_grasp", "mechanism", "relax", "frame_robot", "material_decay", "passive_contact"), default="probe")
    p.add_argument('--full-sdk-source', help='Use the declared-reset full scene through the installed SDK after separate SDK material/mechanism/stock-grasp qualification')
    p.add_argument("--diagnostic-output-root", help="Passive diagnostics only: nested task-output folder containing archived source inputs")
    p.add_argument("--passive-stop-when-settled", action="store_true", help="Diagnostic only: end at the first complete two-second native settling hold, checked at 20 Hz boundaries")
    p.add_argument("--passive-match-kit-defaults", action="store_true", help="Diagnostic only: explicitly match the original Kit native defaults in the installed SDK")
    p.add_argument("--passive-cloth-correction", choices=("bounded", "unbounded"), help="Diagnostic only: reciprocal native-default comparison of 3 m/s versus unbounded cloth contact correction")
    p.add_argument("--passive-pvd-record", action="store_true", help="Diagnostic only: record native PhysX objects for a short internal-property comparison")
    p.add_argument("--passive-canonical-order", action="store_true", help="Diagnostic only: reorder USD frame children without changing any physical attribute")
    p.add_argument("--passive-sdk-scene", help="Diagnostic only: compare an accepted original scene through the installed standalone PhysX SDK")
    p.add_argument("--passive-source", help="Diagnostic only: completed hands-clear release recording")
    p.add_argument("--passive-native-pipeline", action="store_true", help="Diagnostic only: attach and step native PhysX directly, with public tensor readback and no asset callbacks")
    p.add_argument("--passive-usd-readback", action="store_true", help="Diagnostic only: read native exported surface points/velocities through USD for the host-readback rigid-joint comparison")
    p.add_argument("--passive-host-readback", action="store_true", help="Diagnostic only: retain GPU physics with host readback for CPU-compatible gear constraints")
    p.add_argument("--passive-external-forces-once", action="store_true", help="Diagnostic only: compare once-per-step gravity application in TGS")
    p.add_argument("--passive-partitions", type=int, choices=(1, 32), help="Diagnostic only: change native GPU constraint partition count, preserving physical constraints")
    p.add_argument("--passive-solver", choices=("TGS", "PGS"), default="TGS", help="Diagnostic only: native rigid/articulation solver algorithm")
    p.add_argument("--passive-no-self-contact", action="store_true", help="Counterfactual only: isolate sewn-cloth self-contact forces; never a valid task configuration")
    p.add_argument("--passive-contact-subiterations", type=int, choices=(1,), help="Diagnostic only: compare the native default one contact solve per material iteration")
    p.add_argument("--passive-no-joint-friction", action="store_true", help="Counterfactual only: isolate the legacy universal-joint friction model")
    p.add_argument("--passive-no-frame-self-contact", action="store_true", help="Counterfactual only: isolate internal rigid collision constraints; never qualify a latch with its contacts disabled")
    p.add_argument("--passive-unit-scale", type=float, choices=(10., 100.), help="Diagnostic only: equivalent decimetre/centimetre units to test numerical conditioning")
    p.add_argument("--passive-freeze-joints", choices=("handle", "head", "all"), help="Counterfactual only: isolate handle versus folding-head degrees of freedom")
    p.add_argument("--passive-no-contact-friction", action="store_true", help="Counterfactual only: remove all tangential contact friction to isolate normal-contact response")
    p.add_argument("--passive-release-height", type=float, choices=(.002, .01), help="Diagnostic only: release the unchanged assembly just above the support under normal gravity")
    p.add_argument("--passive-head-correction", action="store_true", help="Diagnostic only: apply 0.1 m/s contact-position correction to Bridge/Left/Right, retaining default gimbal and elastic-latch correction")
    p.add_argument("--passive-position-iterations", type=int, choices=(4, 16), help="Diagnostic only: compare native articulation position solves with the baseline 64")
    p.add_argument("--passive-center-body-frames", action="store_true", help="Diagnostic only: preserve physical geometry, joint frames and inertia while centering native link origins at their COM")
    p.add_argument("--passive-zero-bending", action="store_true", help="Counterfactual only: remove cloth bend resistance to isolate elastic forces from contact")
    p.add_argument("--passive-handle-roll", type=float, choices=(0.,), help="Diagnostic only: initialize the same lowered handle centered along the head, preserving all material and mechanism properties")
    p.add_argument("--passive-free-space", action="store_true", help="Counterfactual only: translate frame and cloth 0.2 m above the support with zero gravity to isolate internal contact")
    p.add_argument("--passive-mimic-frequency", type=float, choices=(0., 480., 960.), help="Diagnostic only: explicit PhysX gear compliance in rad/s; zero is the rigid constraint control")
    p.add_argument("--passive-deduplicate-wing-limits", action="store_true", help="Diagnostic only: remove limits already implied by the opposite wing and rigid 1:1 gear")
    p.add_argument("--passive-mode", choices=("articulated", "compound", "bare", "maximal", "split_latch", "reroot_handle", "compound_articulation", "hybrid"), default="articulated")
    p.add_argument("--cloth-contact-correction-speed", type=float, choices=(.1,), help="Explicit constant cloth contact correction [m/s]; requires a matched fresh product gate")
    p.add_argument("--product-gate", default="product-gate.json", help="Qualified material evidence filename within this task output folder")
    p.add_argument("--frame-contact-correction-speed", type=float, choices=(.1,), help="Explicit constant frame contact correction [m/s]; requires a matched fresh mechanism gate")
    p.add_argument("--grasp-sdk-reference", help="Independent stock Kit native property capture required by the SDK grasp diagnostic")
    p.add_argument("--grasp-sdk-source", help="Diagnostic only: repeat qualified stock YAM grasp actions in the installed SDK")
    p.add_argument("--probe-sdk-source", help="Diagnostic only: repeat a qualified contact insertion/withdrawal coupon in the installed SDK")
    p.add_argument("--mechanism-sdk-source", help="Diagnostic only: repeat a qualified laboratory latch test in the installed SDK")
    p.add_argument("--mechanism-gate", default="mechanism-gate.json", help="Qualified mechanism evidence filename within this task output folder")
    p.add_argument("--snap-throat",action="store_true",help="Estimated physical snap-groove lip; requires fresh mechanism qualification")
    p.add_argument("--retention-extension",type=float,choices=(0.,12.),default=0.,help="Additional seconds of isolated passive retention before physical unlock")
    p.add_argument("--upright-handle",action="store_true",help="Ordinary robot motion to turn the grasped handle upright")
    p.add_argument("--frame-lift",type=float,choices=(.14,.22,.4),default=None,help="Ordinary handle lift; the whole head must clear the table for retention")
    p.add_argument("--frame-press",action="store_true",help="Isolated robot press from a freely supported folded frame; no cloth")
    p.add_argument("--robot-unlock",action="store_true",help="Stock robot holds the handle and physically presses the pedal from a stable flat start")
    p.add_argument("--robot-reopen",action="store_true",help="After physical unlocking, grasp an existing frame rail and guide the frame open")
    p.add_argument("--robot-cover",action="store_true",help="Gated native two-pocket robot integration, with the separate cover initially on the table")
    p.add_argument("--policy", choices=("scripted", "interface-check", "dream"), default="scripted")
    p.add_argument("--camera-width", type=int, choices=(320,640), default=640)
    p.add_argument("--camera-height", type=int, choices=(176,360), default=360)
    p.add_argument("--policy-check-steps", type=int, default=64)
    p.add_argument("--demo-run", help="Matched audited full-score demonstration under the declared rubric required before hosted calls")
    p.add_argument("--replay-evidence", help="Matched independently audited fresh-process replay")
    p.add_argument("--interface-evidence", help="Matched native no-network interface validation")
    p.add_argument("--completion-mode", choices=("released", "handle-held"), default="released", help="Explicit task revision: finish the retention test holding only the handle; original release rubric remains available")
    p.add_argument("--cover-offset-x",type=float,choices=(0.,.02),default=0.,help="Disclosed tabletop layout variant for wrist reach; moves only the initial cover")
    p.add_argument("--reset-check",action="store_true",help="Check two perturbed native resets in one process before recording robot actions")
    p.add_argument("--replay-run",help="Replay a complete recorded frame/cover run from reset using its ordinary 14D actions")
    p.add_argument("--release-probe-run",help="Diagnostic only: initialize the held release pose from a completed full recording")
    p.add_argument("--release-probe-baseline",action="store_true",help="In a release probe, play the original remaining commands after a one-second held warmup")
    p.add_argument("--release-probe-without-cover",action="store_true",help="Diagnostic control only: omit the textile when restoring the recorded held frame and robot poses")
    p.add_argument("--release-probe-refine-contact",action="store_true",help="Diagnostic convergence test only: permit increased timestep resolution, solver iterations or contact updates")
    p.add_argument("--release-probe-hands-clear",action="store_true",help="Diagnostic only: initialize both arms at their recorded final clear poses and hold them still")
    p.add_argument("--release-probe-depenetration",choices=('frame','cloth','both'),help="Diagnostic only: limit selected contact correction to 0.1 m/s before simulation")
    p.add_argument("--release-probe-depenetration-speed",type=float,choices=(.1,.2,3.),default=.1,help="Diagnostic contact-correction speed; 3 m/s reproduces the rigid schema default")
    p.add_argument("--release-probe-deduplicate-colliders",action="store_true",help="Diagnostic only: remove redundant clamp collision boxes after proving containment in the unchanged end solids")
    p.add_argument("--release-probe-velocity-iterations",type=int,choices=(0,1,32),help="Diagnostic only: compare frame velocity solves against the baseline four iterations")
    p.add_argument("--release-probe-contact-last",action="store_true",help="Diagnostic only: resolve articulation contacts later in the native solver")
    p.add_argument("--release-probe-no-speculative-ccd",action="store_true",help="Diagnostic only: test whether speculative contact margins cause the resting instability")
    p.add_argument("--table-origin",action="store_true",help="Express the unchanged stock station with its tabletop near zero for FEM precision")
    p.add_argument("--diagnostic-origin-shift",type=float,nargs=3,default=None,help="Translate every object together for cold-restart numerical conditioning only")
    p.add_argument("--bend-damping-rate",type=float,choices=(1.,),default=None,help="Explicit estimated surface bending decay rate [1/s], independently characterized; baseline unchanged")
    p.add_argument("--with-cover",action="store_true",help="Gated isolated two-pocket frame installation; no full robot task")
    p.add_argument("--lab-camera",action="store_true",help="Experimental raw-scene camera; rejects stale native-to-rendered transforms")
    p.add_argument("--pocket-start-angle",type=float,choices=(45.,55.),default=55.,help="Initial folded-frame placement angle in a laboratory pocket test")
    p.add_argument("--retention-raise",type=float,choices=(0.,.04),default=0.,help="Additional ordinary laboratory lift target for all-cloth clearance")
    p.add_argument("--with-pocket",action="store_true",help="Single coupon under the isolated physical folding-frame press; no full scene")
    p.add_argument("--finite-seam",action="store_true",help="Estimated 2 mm finite sewn edge and flat panel metric; needs fresh qualification")
    p.add_argument("--flat-rest-bending",action="store_true",help="Flat neutral panel bending with explicit sewn folds; diagnostic model revision")
    p.add_argument("--flat-rest-panels",action="store_true",help="Estimated flat sewn-panel rest geometry; requires fresh physical qualification")
    p.add_argument("--state-run", help="Source recorded grasp for cold-start contact convergence only")
    p.add_argument("--contact-updates", type=int, choices=(1,4,8,16), default=1)
    p.add_argument("--contact-subiterations", type=int, choices=(1,2,4), default=1)
    p.add_argument("--release-opening",type=float,choices=(.25,1.),default=.25)
    p.add_argument("--right-grasp-height",type=float,choices=(0.,.006,.007,.008,.009,.010),default=0.)
    p.add_argument("--left-grasp-height",type=float,choices=(-.001,0.,.001),default=0.)
    p.add_argument("--grasp-margin-clearance",type=float,choices=(0.,.006),default=0.,help="Move each cloth grasp outward to clear the visible support edge")
    p.add_argument("--cooked-collider-probe",action="store_true",help="Read actual stock PhysX convex geometry after reset; no control actions or task score")
    p.add_argument("--grasp-opening",type=float,choices=(0.,.008,.012),default=.008,help="Ordinary normalized gripper command; zero requests fully closed pads with unchanged stock gains")
    p.add_argument("--close-seconds",type=float,choices=(1.,3.),default=1.,help="Ordinary closing duration; the slow variant adds a one-second closed hold")
    p.add_argument("--early-feed",action="store_true",help="Slower forward lay-down beginning before the pocket reaches the table")
    p.add_argument("--two-hands",action="store_true",help="Optional two-margin grasp diagnostic, not a task restriction")
    p.add_argument("--placement-tilt",type=float,choices=(0.,30.),default=0.,help="Ordinary gripper pitch during cloth placement")
    p.add_argument("--placement", choices=("bench", "table", "forward_table"), default="table", help="Robot grasp diagnostic placement surface")
    p.add_argument("--pedal-overtravel",type=float,choices=(0.,.01),default=0.,help="Extra laboratory pedal travel with the same 20 N cap")
    p.add_argument("--unlock-lift",type=float,choices=(0.,.025,.04),default=0.,help="Concurrent force-limited handle-side lift during physical pedal press")
    p.add_argument("--mechanism-unlock",action="store_true",help="Extend isolated mechanism test with physical pedal contact and a second lift")
    p.add_argument("--press-overtravel",type=float,choices=(0.,.03),default=0.,help="Laboratory compliant-press command only; force cap stays 15 N")
    p.add_argument("--disable-latch-contact", action="store_true", help="Isolated mechanism negative control only")
    p.add_argument("--blade-profile", choices=("square", "beveled"), default="square")
    p.add_argument("--mode", choices=("settle", "insert", "cycle"), default="settle")
    p.add_argument("--physics-hz", type=int, choices=(120, 240, 480, 960, 1920), default=240)
    p.add_argument("--iterations", type=int, choices=(32, 64, 128), default=32)
    p.add_argument("--spacing", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seconds", type=float, default=6)
    p.add_argument("--mouth-height", type=float, default=0.016)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--ccd", action="store_true")
    p.add_argument("--reinforced", action="store_true")
    p.add_argument("--product-profile", choices=("sprint", "base_micro"), default="sprint")
    p.add_argument("--lip-model", choices=("surface", "volume"), default="surface")
    p.add_argument("--seam-filter", choices=("element", "none", "pair"), default="element")
    p.add_argument("--filter-grouping", choices=("pairs", "vertices"), default="pairs")
    p.add_argument("--detach-lip", action="store_true", help="Diagnostic attachment negative control")
    p.add_argument("--clamp-tail", action="store_true", help="Two visible 100 g gravity weights, laboratory coupon characterization only")
    args = p.parse_args()
    diagnostic_output = OUTPUT
    if args.diagnostic_output_root is not None:
        assert args.experiment == "passive_contact", "Archived output roots are diagnostic-only"
        diagnostic_output = Path(args.diagnostic_output_root).resolve()
        assert diagnostic_output.is_relative_to(OUTPUT.resolve()), "Diagnostic output must remain inside this task output folder"
    assert Path(args.name).name == args.name and args.name not in (".", "..")
    assert Path(args.mechanism_gate).name == args.mechanism_gate and args.mechanism_gate.endswith(".json")
    from frame_solver import configuration
    from cloth_solver import configuration as cloth_configuration
    configuration(vars(args))
    cloth_configuration(vars(args))
    assert Path(args.product_gate).name == args.product_gate and args.product_gate.endswith(".json")
    assert 0 < args.seconds <= (120 if args.robot_reopen else 40) and 1 <= args.repeats <= 5
    assert args.spacing in (0.005, 0.0025) and 0.004 <= args.mouth_height <= 0.025
    assert not args.early_feed or (args.experiment=="robot_grasp" and args.placement=="forward_table")
    assert not args.left_grasp_height or args.experiment=="robot_grasp"
    assert not args.grasp_margin_clearance or args.experiment=="robot_grasp"
    assert not args.cooked_collider_probe or args.experiment=="robot_grasp"
    assert not args.pedal_overtravel or (args.experiment=="mechanism" and args.mechanism_unlock)
    assert not args.snap_throat or args.experiment in ("mechanism","frame_robot")
    assert not args.retention_extension or args.experiment=="mechanism"
    assert not args.upright_handle or args.experiment=="frame_robot"
    assert args.frame_lift is None or args.experiment=="frame_robot"
    assert not args.frame_press or (args.experiment=="frame_robot" and not args.upright_handle)
    assert not args.robot_unlock or (args.experiment=='frame_robot' and not args.frame_press and not args.upright_handle)
    assert not args.robot_reopen or args.robot_unlock
    assert not args.robot_cover or (args.robot_reopen and args.product_profile=='base_micro' and not args.reinforced and not args.with_cover)
    assert not args.cover_offset_x or args.robot_cover
    assert (args.camera_width,args.camera_height) in ((320,176),(640,360))
    if args.policy != 'scripted':
        assert args.experiment == 'frame_robot' and args.robot_cover
        assert not args.release_probe_run and not args.replay_run and args.seconds == 120
        assert (args.camera_width,args.camera_height) == (320,176)
        assert 40 <= args.policy_check_steps <= 80
    if args.robot_cover and not args.release_probe_run:
        from evaluation_contract import validate_task_revision
        validate_task_revision(vars(args), HERE)
    if args.policy == 'dream':
        from dream_evaluation import require_hosted_evidence
        require_hosted_evidence(vars(args),HERE)
    assert args.completion_mode == "released" or (args.experiment == "frame_robot" and args.robot_cover and not args.release_probe_run), "Handle-held completion is only valid for ordinary full-cover episodes"
    assert not args.reset_check or args.experiment=='frame_robot'
    if args.replay_run:
        assert args.experiment=='frame_robot' and Path(args.replay_run).name==args.replay_run
        assert (OUTPUT/args.replay_run/'result.json').is_file()
    if args.release_probe_run:
        assert args.experiment=='frame_robot' and args.robot_cover and args.table_origin
        assert not args.replay_run and not args.reset_check and args.seconds <= 12
        assert Path(args.release_probe_run).name == args.release_probe_run
        assert (OUTPUT/args.release_probe_run/'result.json').is_file()
    assert not args.release_probe_baseline or args.release_probe_run
    assert not args.release_probe_without_cover or args.release_probe_run
    assert not args.release_probe_refine_contact or (args.release_probe_run and not args.release_probe_without_cover)
    assert not args.release_probe_hands_clear or (args.release_probe_run and not args.release_probe_baseline)
    assert not args.release_probe_depenetration or (args.release_probe_run and args.release_probe_hands_clear and not args.release_probe_without_cover)
    assert args.release_probe_depenetration_speed==.1 or args.release_probe_depenetration
    assert not args.release_probe_deduplicate_colliders or (args.release_probe_run and args.release_probe_hands_clear and not args.release_probe_without_cover)
    assert not args.release_probe_contact_last or (args.release_probe_run and args.release_probe_hands_clear and not args.release_probe_without_cover)
    assert not (args.release_probe_velocity_iterations is not None or args.release_probe_no_speculative_ccd) or (args.release_probe_run and args.release_probe_hands_clear and not args.release_probe_without_cover)
    assert not args.table_origin or args.experiment in ("robot_grasp","frame_robot")
    assert args.diagnostic_origin_shift is None or args.experiment=="relax"
    assert args.bend_damping_rate is None or (args.product_profile=="base_micro" and not args.reinforced)
    assert not args.with_cover or (args.experiment=="mechanism" and args.product_profile=="base_micro" and not args.reinforced and not args.with_pocket)
    assert not args.lab_camera or args.with_cover
    assert args.pocket_start_angle==55. or (args.experiment=='mechanism' and (args.with_cover or args.with_pocket))
    assert not args.retention_raise or (args.experiment=='mechanism' and args.with_cover and not args.mechanism_unlock)
    if args.with_cover or args.robot_cover:
        product_gate=json.loads((OUTPUT/args.product_gate).read_text())
        mechanism_gate=json.loads((OUTPUT/args.mechanism_gate).read_text())
        assert product_gate["passed"] and mechanism_gate["passed"], "Isolated material and mechanism gates must pass first"
        assert mechanism_gate.get("retention_window_s",[10,12])[1]>=24, "Long retention must be qualified"
        positive=OUTPUT/mechanism_gate["positive_run"]
        assert hashlib.sha256((HERE/"frame_native.py").read_bytes()).hexdigest()==hashlib.sha256((positive/"source/frame_native.py").read_bytes()).hexdigest()
        positive_request=json.loads((positive/"request.json").read_text())
        assert args.snap_throat==positive_request.get("snap_throat",False)
        assert configuration(vars(args))==configuration(positive_request), "Frame solver must match its isolated qualification"
        if args.frame_contact_correction_speed is not None:
            assert mechanism_gate["checks"]["matched_frame_contact_solver"]
            assert args.physics_hz==positive_request["physics_hz"], "Frame timestep must match its fresh qualification"
            assert hashlib.sha256((HERE/"frame_solver.py").read_bytes()).hexdigest()==hashlib.sha256((positive/"source/frame_solver.py").read_bytes()).hexdigest(), "Frame solver changed after qualification"
        assert not args.finite_seam and not args.flat_rest_panels and not args.flat_rest_bending and args.bend_damping_rate is None
        cycle=OUTPUT/product_gate["cycle_run"]
        qualified_request=json.loads((cycle/"request.json").read_text())
        assert cloth_configuration(vars(args))==cloth_configuration(qualified_request), "Cloth solver must match its isolated qualification"
        if args.cloth_contact_correction_speed is not None:
            assert product_gate["checks"]["same_cloth_contact_solver"]
            assert hashlib.sha256((HERE/"cloth_solver.py").read_bytes()).hexdigest()==hashlib.sha256((cycle/"source/cloth_solver.py").read_bytes()).hexdigest(), "Cloth solver changed after qualification"
        for key in ("physics_hz","iterations","contact_updates","contact_subiterations","ccd","spacing","mouth_height"):
            if args.release_probe_refine_contact and key in ("physics_hz","iterations","contact_updates","contact_subiterations"):
                assert getattr(args,key)>=qualified_request[key], f"Diagnostic refinement cannot reduce {key}"
            else:
                assert getattr(args,key)==qualified_request[key], f"Two-pocket test must use the qualified {key}"
        assert hashlib.sha256((HERE/"parameters.json").read_bytes()).hexdigest()==hashlib.sha256((cycle/"source/parameters.json").read_bytes()).hexdigest(), "Material configuration changed after qualification"
    assert not args.with_pocket or (args.experiment=="mechanism" and args.product_profile=="base_micro" and not args.reinforced)
    assert not args.finite_seam or (args.product_profile=="base_micro" and not args.reinforced and (args.experiment!="mechanism" or args.with_pocket))
    assert not (args.flat_rest_panels and args.flat_rest_bending)
    assert not (args.flat_rest_panels or args.flat_rest_bending) or args.experiment in ("relax","robot_grasp")
    assert not args.flat_rest_panels or (args.product_profile=="base_micro" and not args.reinforced)
    assert args.product_profile != "base_micro" or not args.reinforced, "BaseMicro profile has no rubber strip"
    assert args.experiment != "robot_grasp" or args.repeats == 1, "Robot repeated resets are not yet validated"
    assert not args.clamp_tail or args.repeats == 1, "Tail-weight resets are not yet supported"
    assert args.physics_hz <= 960 or (args.experiment == 'passive_contact' and args.passive_sdk_scene), '1920 Hz is restricted to the standalone convergence diagnostic'
    assert sum(bool(value) for value in (args.probe_sdk_source, args.mechanism_sdk_source, args.grasp_sdk_source, args.full_sdk_source)) <= 1
    assert not args.grasp_sdk_reference or args.grasp_sdk_source
    if args.grasp_sdk_source:
        from standalone_grasp import validate_source
        validate_source(vars(args), diagnostic_output)
    if args.probe_sdk_source:
        from standalone_probe import validate_source
        validate_source(vars(args), OUTPUT)
    if args.mechanism_sdk_source:
        from standalone_mechanism import validate_source
        validate_source(vars(args), OUTPUT)
    if args.full_sdk_source:
        from full_sdk import validate_source
        validate_source(vars(args), OUTPUT)
    from passive_contact import check_request
    check_request(vars(args), diagnostic_output)
    out = diagnostic_output / args.name
    out.mkdir(parents=True, exist_ok=False)
    (out / "request.json").write_text(json.dumps(vars(args), indent=2) + "\n")
    try:
        with reservation(out) as (gpu, descriptors, legacy):
            return run_reserved(args, out, gpu, descriptors, legacy)
    except (AssertionError, RuntimeError, BlockingIOError, KeyError, ValueError, OSError) as exc:
        attempted = (out / "console.log").exists()
        artifact = 'launcher-error.json' if attempted else 'blocked.json'
        (out / artifact).write_text(json.dumps({"reason": str(exc), "child_launch_attempted": attempted}))
        print(str(exc), flush=True)
        return 1 if attempted else 75


def run_reserved(args, out, gpu, descriptors, legacy):
    bundle = out / "source"
    bundle.mkdir()
    hashes = {}
    for file in sorted(HERE.iterdir()):
        if file.is_file():
            data = file.read_bytes()
            (bundle / file.name).write_bytes(data)
            hashes[file.name] = hashlib.sha256(data).hexdigest()
    manifest = {"files": hashes, "bundle_sha256": hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    versions = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    (out / "versions.json").write_text(json.dumps(versions, indent=2, sort_keys=True))
    preserved_sources = {}
    from evaluation_contract import preserved_runtime_paths
    for source in preserved_runtime_paths():
        preserved_sources[str(source)] = hashlib.sha256(source.read_bytes()).hexdigest()
    (out / "preserved-sources.json").write_text(json.dumps(preserved_sources, indent=2))
    env = os.environ.copy()
    env.update(CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=gpu['uuid'],
        ROBOCURVE_GPU_INDEX=str(gpu['index']), ROBOCURVE_GPU_UUID=gpu['uuid'],
        ROBOCURVE_CUDA_DEVICE='cuda:0', PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1")
    if legacy:
        env['POCKET_GPU_LOCK_FD'] = str(descriptors[0])
    else:
        env.pop('POCKET_GPU_LOCK_FD', None)
    for variable, name in (("XDG_CACHE_HOME", "xdg-cache"), ("CUDA_CACHE_PATH", "cuda-cache"), ("WARP_CACHE_PATH", "warp-cache"), ("TMPDIR", "tmp")):
        if legacy:
            path = OUTPUT / "runtime" / name
            path.mkdir(parents=True, exist_ok=True)
            env[variable] = str(path)
    env["POCKET_OUT"] = str(out)
    started = time.monotonic()
    # Parent and child retain the same open file description for the whole job.
    with (out / "console.log").open("w") as log:
        result = subprocess.run([sys.executable, "-B", str(bundle / ("full_sdk.py" if args.full_sdk_source else "standalone_grasp.py" if args.grasp_sdk_source else "standalone_probe.py" if args.probe_sdk_source else "standalone_mechanism.py" if args.mechanism_sdk_source else {"passive_contact": "passive_contact.py", "material_decay": "material_decay.py", "frame_robot": "frame_robot.py", "robot_grasp": "robot_gate.py", "mechanism": "mechanism_probe.py", "relax": "relax_probe.py", "probe": "probe.py"}[args.experiment])), "--request", str(out / "request.json")],
                                stdout=log, stderr=subprocess.STDOUT, env=env, pass_fds=descriptors, cwd=bundle)
    (out / "after-gpu.json").write_text(json.dumps(snapshot(), indent=2))
    (out / "process.json").write_text(json.dumps({"returncode": result.returncode, "wall_seconds": time.monotonic() - started}, indent=2))
    print(out, "returncode=", result.returncode, flush=True)
    recorded = json.loads((out / "result.json").read_text()) if (out / "result.json").exists() else {}
    return result.returncode or (0 if recorded.get("status") == "completed" else 1)


if __name__ == "__main__":
    sys.exit(main())
