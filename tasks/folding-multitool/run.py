"""Scheduler-only shared scripted / Dream-1 retrieval episode runner."""
import argparse
import atexit
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parent


def dump(path, value):
    path.write_text(json.dumps(value, indent=2)+"\n")


def run(args):
    global ROOT
    if os.environ.get("ROBOCURVE_TASK_ID") != "folding-multitool":
        raise RuntimeError("Launch through gpu-scheduler with --task folding-multitool")
    sys.path.insert(0, os.environ.get("ROBOCURVE_SCHEDULER_ROOT", str(Path.home()/"robocurve-gpu-scheduler")))
    from isaac_binding import assigned_gpu, kit_args, verify
    assigned_gpu()
    job = Path(os.environ["ROBOCURVE_JOB_DIR"])
    out = Path(args.out).resolve()/os.environ["ROBOCURVE_JOB_ID"]
    out.mkdir(parents=True, exist_ok=False)
    bundle = out/"source"; bundle.mkdir()
    hashes = {}
    for source in ROOT.iterdir():
        if source.is_file():
            shutil.copy2(source, bundle/source.name)
            hashes[source.name] = hashlib.sha256(source.read_bytes()).hexdigest()
    # Import the frozen snapshot so later local edits cannot change an admitted job.
    ROOT = bundle
    sys.path.insert(0, str(bundle))
    from episode_config import INSTRUCTION, CAMERAS, WIDTH, HEIGHT, CONTROL_HZ, SEED, TOP_CAMERA_VARIANT, contract
    versions = {}
    for name in ("inspect-robots", "inspect-robots-dream", "dreamlabs-client", "torch", "numpy", "isaacsim"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    dump(out/"metadata.json", {"command": sys.argv, "task": "folding-multitool", "mode": args.mode,
         "controller": args.controller, "max_steps": args.max_steps, "contract": contract(ROOT),
         "versions": versions, "seed": SEED, "hosted": args.controller == "dream",
         "reset_diagnostic": args.capture_reset,
         "job_dir": str(job), "physical_gpu": os.environ["ROBOCURVE_GPU_INDEX"], "hashes": hashes,
         "physics_dt": 1/240, "control_hz": 20, "mechanics": "approximate_unmeasured",
         "no_tool_state_writes_after_reset": True})
    print("ARTIFACTS "+str(out), flush=True)
    app, emb, writer, policy = None, None, None, None
    began = time.monotonic()
    try:
        if args.controller == "dream":
            from preflight import ready
            readiness = ready(ROOT, args.demo_run, args.interface_run, args.out)
            dump(out/"hosted-preflight.json", readiness)
            if not readiness["ready"]:
                raise RuntimeError("Hosted evaluation prerequisites: "+"; ".join(readiness["blockers"]))
        from isaaclab.app import AppLauncher
        app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0", multi_gpu=False,
                          kit_args=kit_args()).app
        import numpy as np
        import imageio.v2 as imageio
        from inspect_robots import Scene
        from inspect_robots.types import Action
        from yam_arena.constants import READY
        import yam_arena.inspect_embodiment as native
        native_sources = {}
        native_root = Path(native.__file__).resolve().parent
        for name in ("inspect_embodiment.py", "scene.py", "embodiment.py", "actions.py", "constants.py", "asset.py", "station.py"):
            path = native_root/name
            if path.is_file():
                native_sources[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        dump(out/"native-source.json", native_sources)
        native._APP = app
        atexit.register(native._close_app_at_exit)
        import adapter
        adapter.configure(args.mode)
        adapter.install()
        emb = native.YamArenaEmbodiment(task=str(ROOT/"task.py"), object_jitter=0,
                  control_hz=CONTROL_HZ, cam_width=WIDTH, cam_height=HEIGHT, device="cuda:0",
                  top_cam=TOP_CAMERA_VARIANT)
        scene = Scene(id="folding-multitool", instruction=INSTRUCTION)
        if args.capture_reset:
            from reset_diagnostic import capture_reset
            obs = capture_reset(emb, scene, SEED, out)
        else:
            obs = emb.reset(scene, seed=SEED)
        adapter.reset(emb._env)
        interface_checks = {"passed": False, "network_called": False}
        if args.controller == "interface-check":
            # Exercise the native reset a second time, before the test episode.
            first = adapter.last_record
            obs = emb.reset(scene, seed=SEED)
            adapter.reset(emb._env)
            np.testing.assert_allclose(list(first["q"].values()), list(adapter.last_record["q"].values()), atol=1e-3)
            np.testing.assert_allclose(first["payload"]["pose"][:3], adapter.last_record["payload"]["pose"][:3], atol=1e-3)
            assert adapter.monitor.snapshot()["folded_reset_seen"]
            assert adapter.monitor.score == 0. and adapter.last_record["step"] == 0
            interface_checks["repeat_folded_reset"] = True
        sim = emb._base.sim
        native_tensor = sim.physics_sim_view.create_rigid_body_view(
            "/World/envs/env_0/multitool/jaw_a").get_transforms()
        if native_tensor.shape[0] != 1:
            raise RuntimeError("Native verification tensor must contain the actual tool root")
        evidence = verify(sim, native_tensor.device)
        shutil.copy2(job/"device-verification.json", out/"device-verification.json")
        print("DEVICE_VERIFIED "+json.dumps(evidence), flush=True)
        from audit import check
        dump(out/"reset-measurement.json", adapter.last_record)
        check(emb._env, out/"native-audit.json")
        if args.mode == "retrieval" and not adapter.monitor.snapshot()["nut_initial_in_pocket"]:
            imageio.imwrite(out/"invalid-reset.png", np.concatenate([obs.images[n] for n in CAMERAS], axis=1))
            raise RuntimeError("Invalid retrieval reset: nut is not on the declared pocket floor")
        imageio.imwrite(out/"setup.png", obs.images["top_cam"])
        imageio.imwrite(out/"setup-all.png", np.concatenate([obs.images[n] for n in CAMERAS], axis=1))
        if args.video:
            writer = imageio.get_writer(out/"diagnostic.mp4", fps=20, codec="libx264",
                     macro_block_size=1, ffmpeg_params=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
            writer.append_data(np.concatenate([obs.images[n] for n in CAMERAS], axis=1))
        # The model branch never constructs the scripted controller or a Context.
        ctx, gen, fixture = None, None, None
        if args.controller == "scripted" and args.mode != "preview":
            from controller import Context, script
            ctx = Context()
            gen = script(ctx, args.mode)
        from evaluation import observation, vector, verify_measured_state, verify_action_execution
        verify_measured_state(obs, adapter.last_record)
        observation(obs, 0)
        initial_passive_updates = adapter.last_record["passive_updates"]
        camera_checks = {n: {"shape": list(obs.images[n].shape), "dtype": str(obs.images[n].dtype),
                              "rgb_sha256": hashlib.sha256(obs.images[n].tobytes()).hexdigest(),
                              "std": float(obs.images[n].std())} for n in CAMERAS}
        if any(c["std"] < 1. for c in camera_checks.values()):
            raise RuntimeError("A reset camera is blank")
        if len({c["rgb_sha256"] for c in camera_checks.values()}) != 3:
            raise RuntimeError("Reset camera streams are duplicated")
        dump(out/"observation-contract.json", {"cameras": camera_checks,
             "top_camera_variant": TOP_CAMERA_VARIANT, "top_camera": asdict(emb.rig.top_cam),
             "state_order": list(emb.info.action_space.semantics.dim_labels),
             "state": list(obs.state["joint_pos"]), "instruction": INSTRUCTION,
             "state_from_native_measurements": True})
        reason, error, steps, previous = "step_limit", None, 0, "reset"
        with (out/"trace.jsonl").open("w") as log:
            log.write(json.dumps({**adapter.last_record, "stage": "reset", "action": None,
                                  "robot_state": list(obs.state["joint_pos"]), "simulation_time": 0.})+"\n")
            try:
                if args.controller != "scripted":
                    from evaluation import make_policy
                    from dataclasses import replace
                    if args.controller == "interface-check":
                        # The installed facade explicitly constructs without
                        # opening a session or sending a request. Check it also
                        # imports inside this Isaac process, where conflicting
                        # bundled libraries can otherwise require a bridge.
                        from dream import DreamClient
                        with DreamClient() as local_client:
                            assert local_client._policy is None
                            for name in ("get_models", "policy", "set_observation", "get_action", "close"):
                                assert callable(getattr(local_client, name))
                        interface_checks["dream_client_import_and_constructor_no_session"] = True
                    policy, fixture = make_policy(out/"inference", offline=args.controller == "interface-check")
                    # Binding needs only robot/action/camera specifications. Do
                    # not give it task docs, target descriptions, or capabilities.
                    info = replace(emb.info, docs="", environment_id=None, environment_revision=None,
                                   capabilities=frozenset(), supported_target_kinds=frozenset())
                    policy.bind(info)
                    policy.reset(scene)
                for k in range(args.max_steps):
                    control_began = time.monotonic()
                    if ctx:
                        try:
                            next(gen)
                        except StopIteration:
                            reason = "script_complete"; break
                        action = ctx.action()
                    elif policy:
                        before = (emb._t, emb._base.scene["multitool"].passive_update_count)
                        chunk = policy.act(observation(obs, k))
                        after = (emb._t, emb._base.scene["multitool"].passive_update_count)
                        assert before == after, "Physics advanced during policy inference"
                        assert chunk.control_hz == CONTROL_HZ and len(chunk.actions) == 1
                        action = chunk.actions[0]
                    else:
                        action = Action(data=[*READY, 1., *READY, 1.])
                    raw_action = vector(action.data)
                    applied = np.clip(raw_action, emb._low, emb._high)
                    decision_seconds = time.monotonic()-control_began
                    final = emb.step(action)
                    native_action = verify_action_execution(emb, applied)
                    obs = final.observation
                    steps += 1
                    verify_measured_state(obs, adapter.last_record)
                    observation(obs, steps)
                    assert adapter.last_record["step"] == steps
                    assert adapter.last_record["passive_updates"]-initial_passive_updates == steps*12
                    stage = ctx.stage if ctx else args.controller if policy else "preview"
                    record = {**adapter.last_record, "stage": stage, "action": raw_action.tolist(),
                              "applied_action": applied.tolist(), "simulation_time": float(emb._t),
                              "native_processed_action": native_action.tolist(),
                              "decision_wall_seconds": decision_seconds,
                              "robot_state": np.asarray(obs.state["joint_pos"]).tolist()}
                    log.write(json.dumps(record)+"\n")
                    frame = np.concatenate([obs.images[n] for n in CAMERAS], axis=1)
                    if writer:
                        writer.append_data(frame)
                    if stage != previous:
                        imageio.imwrite(out/f"{steps:04d}-{stage}.png", frame)
                        previous = stage
                    if steps % 20 == 0:
                        log.flush()
                        print(json.dumps({"step": steps, "stage": stage, "q": record["q"],
                              "gap_m": record["gap_m"], "pads_n": record["pads_n"]}), flush=True)
                    if fixture and steps in (80, 128):
                        measured = np.asarray(obs.state["joint_pos"])
                        joint_ids = [i for i in range(14) if i not in (6, 13)]
                        np.testing.assert_allclose(measured[joint_ids], fixture.target[joint_ids], atol=.04, rtol=0)
                        np.testing.assert_allclose(measured[[6, 13]], fixture.target[[6, 13]], atol=.04, rtol=0)
                        interface_checks[f"asymmetric_target_{steps}"] = {
                            "requested": fixture.target.tolist(), "measured": measured.tolist()}
                    if final.terminated or final.truncated:
                        reason = final.termination_reason or ("environment_truncated" if final.truncated else "environment_terminated")
                        break
                    if action.terminal:
                        reason = "policy_"+action.terminal.kind
                        break
                if fixture:
                    assert steps == 128 and fixture.calls == 2 and fixture.observed == 128
                    assert reason == "step_limit" and adapter.monitor.score == 0.
                    interface_checks.update(passed=True, actions=steps, requests=fixture.calls,
                        measured_state_order=True, no_physics_during_inference=True,
                        native_action_term_order=True, native_processed_targets_match=True,
                        twelve_physics_steps_per_action=True, action_budget_termination=True,
                        scorer_score=adapter.monitor.score, scripted_assistance=False)
            except Exception as exc:
                reason = "stage_failed" if ctx else "policy_or_interface_error"
                error = repr(exc)
                traceback.print_exc()
            imageio.imwrite(out/"final.png", np.concatenate([obs.images[n] for n in CAMERAS], axis=1))
        if writer:
            writer.close(); writer = None
        if policy:
            dump(out/"inference-transcript.json", policy.transcript())
        if args.controller == "interface-check":
            dump(out/"interface-check.json", interface_checks)
        result = {"mode": args.mode, "controller": args.controller, "steps": steps, "reason": reason, "error": error,
                  "scoring": adapter.monitor.snapshot(), "device_verified": True,
                  "simulation_seconds": steps/CONTROL_HZ, "seed": SEED,
                  "wall_seconds": time.monotonic()-began, "events": ctx.events if ctx else [],
                  "simulation_only": True, "hardware_calibrated": False}
        result["free_motion_fallbacks"] = {a.side: a.free_joint_fallbacks for a in (ctx.left, ctx.right)} if ctx else {}
        dump(out/"result.json", result)
        dump(job/"folding-multitool-result.json", result)
        print("RESULT "+json.dumps(result), flush=True)
    except BaseException:
        failure = {"traceback": traceback.format_exc(), "wall_seconds": time.monotonic()-began}
        dump(out/"failure.json", failure)
        dump(job/"folding-multitool-failure.json", failure)
        raise
    finally:
        if policy:
            policy.close()
        if writer:
            writer.close()
        if emb:
            emb.close()
        # Native atexit guard bounds Kit shutdown and runs only after artifacts save.
        if app is not None and "native" not in locals():
            app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("preview", "grasp", "deployment-gate", "load-gate", "retrieval"), default="deployment-gate")
    parser.add_argument("--out", default=str(Path.home()/"yam-eval/folding-multitool/runs"))
    parser.add_argument("--max-steps", type=int, default=3600)
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--capture-reset", action="store_true", help="Read-only native substep trace during reset settling")
    parser.add_argument("--controller", choices=("scripted", "dream", "interface-check"), default="scripted")
    parser.add_argument("--demo-run", help="Verified 1.0 demo directory; otherwise discover matching evidence in --out")
    parser.add_argument("--interface-run", help="Passed offline interface directory; otherwise discover matching evidence in --out")
    args = parser.parse_args()
    if not 1 <= args.max_steps <= 3600:
        parser.error("--max-steps must be in 1..3600")
    if args.controller != "scripted" and args.mode != "retrieval":
        parser.error("Dream and interface checks require the complete --mode retrieval scene")
    if args.capture_reset and (args.mode != "retrieval" or args.controller != "scripted"):
        parser.error("--capture-reset is restricted to scripted retrieval diagnostics")
    if args.controller == "interface-check" and args.max_steps != 128:
        parser.error("--controller interface-check requires --max-steps 128")
    if args.controller == "dream" and (args.max_steps != 3600 or not args.video):
        parser.error("Dream requires --max-steps 3600 --video")
    run(args)
