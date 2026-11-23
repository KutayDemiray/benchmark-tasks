# Install reusable furniture sliders and move a side table

Fit one slider under each of the four feet, slide the table into the marked destination, and release it. The scene uses a full-size 350 mm household side table and four separate 88.9 mm sliders. `task.json` specifies the layout and score; `spec.json` distinguishes published dimensions from estimates. The foam is a rigid high-friction approximation, not a calibrated model of the named product.

The robot is the installed native two-arm YAM. Its meshes, joint limits, actuator gains and 14-dimensional action interface are unchanged. The feasibility controller reads privileged state and issues only those actions. Furniture and sliders move through contact; no attachments, snapping or post-reset pose writes are used. Carrying, upending and different installation strategies remain legal. A separate audit lifted the whole table with all feet at least 18.3 mm clear for two seconds, then lowered and released it. Partial lifting is therefore not required to lift this object; prepositioning pads before a whole-table lift remains a plausible alternative installation strategy, not a demonstrated complete alternative solution.

## Run and score

Use the existing native environment and cooperative scheduler; versions and external runtime requirements are documented in [DEPENDENCIES.md](DEPENDENCIES.md). Set `ROBOCURVE_SCHEDULER_DIR` if the scheduler is not at `$HOME/robocurve-gpu-scheduler`. Run from this task folder, with output directories outside the source checkout: the runner copies its source into each run, so output must not be nested inside that source directory.

```bash
source /path/to/IsaacLab-Arena/yam.env
export ROBOCURVE_SCHEDULER_DIR=/path/to/robocurve-gpu-scheduler
export FURNITURE_OUTPUT=/path/to/furniture-sliders-recordings

"$ROBOCURVE_SCHEDULER_DIR/gpu-scheduler" launch \
  --task furniture-sliders --mode exclusive --gpu auto --wait 300 --detach -- \
  python -B run.py --mode one-pad --max-steps 1600 --video --out "$FURNITURE_OUTPUT/one-pad-NEW"

# After that run completes and its explicit one_pad_gate_pass is true:
"$ROBOCURVE_SCHEDULER_DIR/gpu-scheduler" launch \
  --task furniture-sliders --mode exclusive --gpu auto --wait 300 --detach -- \
  python -B run.py --mode full --max-steps 6000 --video \
  --gate-evidence "$FURNITURE_OUTPUT/one-pad-NEW/result.json" --out "$FURNITURE_OUTPUT/full-NEW"

# CPU-only checks; these do not initialize the simulator:
python -B -m unittest discover -s . -p 'test_*.py' -v
python -B audit_run.py "$FURNITURE_OUTPUT/full-NEW"
python -B replay_score.py "$FURNITURE_OUTPUT/full-NEW"
# Audit an already recorded model episode without inference or simulation:
python -B tools/audit_model.py /path/to/recorded-model-run
```

`--wait 300` bounds admission, not execution. The job ID printed by launch can be passed to the scheduler's `wait` and `logs` commands. Check `result.json` and any `failure.json` after completion; simulator shutdown can return zero following an application exception. The runner rejects unscheduled simulation, binds physics to masked `cuda:0`, disables multi-GPU rendering, and verifies the actual physics tensor, CUDA UUID and active physical renderer after warmup. Scheduling is separate from task physics and scoring.

The full limit is 6000 actions at 20 Hz, or 300 simulated seconds, including preparation. G1 has a 1600-action limit and tests insertion plus at least 100 mm of released-pad retention. G1 took 803 wall seconds on the development machine. The successful full demo used 5838 actions (291.9 simulated seconds) and took 3188 wall seconds with four 640 × 352 recorded views. This is one observed runtime, not a throughput benchmark. The fixed initial layout uses seed 0 and no object jitter. Reset restores the bare-footed table, four loose foam-up pads and native ready arm positions. No materials are consumed.

`turn-debug`, `continuation-debug` and `rear-debug` initialize installed pads for diagnosis and cannot count as full demonstrations. `rear-debug` additionally requires a saved state via `--reset-state`; that option is rejected in full mode. `lift` tests a legitimate whole-object lift. Among scripted modes, only `full` starts from the specified four-loose-pad task and can report `full_task_demonstration`. Live Dream evaluation uses the same bare-table, four-loose-pad reset and public scorer. Changed physical specifications require a new G1 result. The full scripted demo reaches score 1.0 from the specified bare start; broader robustness is not established.

The three policy cameras are native. Video runs add a static rear observer only for recording and visual inspection; it is not supplied to the policy. Logs retain every action and measured state for independent replay. Geometry, solver settings and materials are in `assets.py`, `floor_geometry.py`, `partkit.py` and `spec.json`; scoring is in `scoring.py`; scripted control and collision screening are in `controller.py` and `motion_plan.py`.

## Practical real-world scoring

Use the stated real-size table, four separate reusable sliders, a secured finished panel, the 400 mm goal outline and a ruler. Measure actual assembled geometry and loaded foam height when reproducing the task; estimated simulation values are not product measurements. Use overhead video for table displacement and opposing low views for the foot-pad interfaces. Apply small visible reference dots to the table and pads if needed; these add no physical retention.

1. Record the bare initial layout. No pad begins under a foot. Start the clock and record the complete trial.
2. At the final release, count distinct upright pads supporting feet. Each of the four projected corners of a counted foot must lie at least 2 mm inside the usable foam boundary. Inspect the lowest bearing edge using a 0.5 mm feeler gauge; the permitted gap is at most 0.5 mm to the actual loaded foam surface and between the pad's lowest rim and the floor. A 500 g corner load can reveal rocking or a floating foot; it is a post-trial support check, not part of the robot's solution.
3. Require the table within 3 degrees and pads within 5 degrees of upright, all feet on the panel, both grippers at least 50 mm clear, and no other robot support. Observe 2 seconds with at most 2 mm translation and 1 degree rotation. Award 0.1 per qualifying distinct support, up to 0.4; this is current-state credit.
4. Award 1.0 only if all four qualify, the tabletop’s vertical projection is entirely inside the goal (check its corners with a square or plumb line), and video shows at least 150 mm net displacement during one continuous segment with all four feet on the same pads and the pads on the floor. Mark the table centre at the start of that segment and measure straight-line displacement, not accumulated hidden slip. Any loss of a qualified support restarts the segment. Return movement cannot repair an earlier break in support without a new qualifying segment.

A whole-object lift or upending during installation is allowed. Carrying the assembled furniture and pads into the goal alone yields at most 0.4 because it does not demonstrate floor sliding. A legitimate easier complete solution should revise the capability claim rather than be prohibited. The intended distinction is installing loose reusable supports followed by retained floor motion; four repeated placements alone do not establish novelty.

## Fidelity and evidence limits

Table size and slider diameter follow published products; leg construction, mass distribution, foam thickness and contact coefficients are estimates. At nominal four-foot loading, the flat contact pressure is about 1.8 kPa. A tilted foot concentrates load along an edge, where compression and indentation may alter retention. Neither foam stiffness nor friction has been physically measured. The fixed rigid approximation may demonstrate a simulated solution, but does not validate the real product's compliance or retention accuracy.

The floor is one static triangle mesh with actual mount cutouts; it has no artificial internal tile seams. Task-local physics runs at 480 Hz with 64 position and 8 velocity iterations, while native robot settings remain unchanged. Recorded torque and finger-contact diagnostics must be reviewed with the full demonstration. No model score, task-board media ID or external submission is implied by local scripted evidence. See the accompanying evidence manifest for completed gates and outstanding items.

A numerical comparison using the exact demonstrated controller at 960 Hz failed first-pad containment during lowering: the most distant foot corner reached 44.013 mm radius against the unchanged 41 mm limit. The pad remained in bearing contact. A separate diagnostic with one pad already installed passed 134.836 mm of retained movement and release at 960 Hz; it is not an insertion or full-task demonstration. This package preserves the demonstrated 480 Hz configuration, discloses placement sensitivity, and makes no numerical-convergence or broader robustness claim. The successful full run also had only 1.809 mm final goal clearance on its tight side.


## Live Dream-1 evaluation

The same `run.py` supports `--mode interface-check` and `--mode dream`. It uses the installed `inspect-robots-dream` integration without changing dependencies or native YAM. The model receives only the task instruction, measured 14D joint/gripper state, and `top_cam`, `left_cam`, `right_cam` RGB images resized to 320 × 176. The rear observer remains recording-only. Native recordings stay at 640 × 352 per camera.

Dream's standard 32-action preroll and 48-row queue count toward the 6000-action limit. Actions run at 20 simulated Hz, with physics paused during inference. The native driver retains its usual joint-limit clipping. No scripted controller or privileged state is used to choose model actions. Recorded raw targets, native-clipped targets, inputs and inference chunks are retained for independent audit. SDK transport retries, if any, belong to the same episode and are logged.

Before hosted calls, the offline interface check uses the installed DreamPolicy with a local session substitute. Its small free-space movements verify ordering, action execution, clocks and recording; it cannot count as a model score or a task demonstration. Live mode requires a passed check with identical executable source/specification, a continuous full-score scripted demonstration on the same scene, and a create-once attempt ledger. Configure credentials through the installed Dream client's environment or credentials file; never put a key in the task folder.

From the task directory, using explicit paths to the evidence and new output locations:

```bash
# These commands reserve the complete simulator process tree. Numerical and
# performance experiments still require exclusive mode.
"$ROBOCURVE_SCHEDULER_DIR/gpu-scheduler" launch \
  --task furniture-sliders --mode dev --gpu auto --wait 300 --detach -- \
  python -B run.py --mode interface-check --max-steps 96 --video \
  --demo-evidence /path/to/full-demo/result.json --out /path/to/new-interface-check

# One paid episode, only after the local check passes. Do not overwrite an
# attempt ledger or repeat paid episodes to select a successful outcome.
"$ROBOCURVE_SCHEDULER_DIR/gpu-scheduler" launch \
  --task furniture-sliders --mode dev --gpu auto --wait 300 --detach -- \
  python -B run.py --mode dream --max-steps 6000 --seed 0 --video \
  --demo-evidence /path/to/full-demo/result.json \
  --preflight-evidence /path/to/new-interface-check/result.json \
  --attempt-ledger /path/to/one-authorized-attempt.json --out /path/to/model-run
```

A model score is reported only from the genuine hosted run's preserved result. Uploaded media are identified by the task-specific `media` field in `task.json`.

## Recorded evaluation

The continuous scripted demonstration scored **1.0 in 5838/6000 actions** from bare feet and four loose sliders. It completed 203.966 mm of qualified supported movement and released goal settling. One genuine hosted **Dream-1 (`dreamlabs/dream-1-beta`) episode scored 0.0 in 6000/6000 actions**, terminating with `max_steps`. The environment seed was 0, no scripted assistance was loaded, and all targets were ordinary native 14D commands at 20 simulated Hz. Independent raw-state scoring and input/chunk/action reconciliation agree with the saved results.

The installed SDK returned 125 chunks through 249 transport attempts, reporting 125 distinct server session IDs and 124 expiration retries. The simulator never reset, but server memory continuity was lost across renewals; this is an observed result under those conditions, not a stable-session performance estimate. Stock joint limits clipped 4315 actions, chiefly at joint 2 of each arm and right joint 3. Raw model targets and the actual clipped commands are retained. No second paid episode was made to obtain a better score.

The model trial took 5550.315 wall seconds under GPU contention for 300 simulated action seconds; this is not a performance benchmark. It had zero qualified supports throughout. Sampled inspection covers 15 actions in all four recorded views, plus complete numeric trace replay. The rear camera remains outside the model interface. The local interface check made zero hosted requests and passed ordering, action execution, timing, termination, scoring and recording checks before the live episode.

The media are `model_run.mp4` (20,910,696 bytes; 6001 frames, 300.05 seconds), `demo.mp4` (25,741,621 bytes; 5839 frames, 291.95 seconds), and `setup.png` (144,319 bytes). Both videos are anonymous, uncut H.264/yuv420p MP4s at 1280 × 704 and 20 fps. Their extra initial frame accounts for the 0.05-second difference from action time. [provenance.json](provenance.json) records their hashes, the evaluated source/specification and the independent assessment results. Raw recordings and inference/state logs remain external to Git. Approximate rigid foam, numerical placement sensitivity and fixed-layout robustness limits remain as disclosed above.

The retained diagnostic modes and `regression-*.json` fixtures document earlier investigations and support the current controller/tests. Initialized-pad diagnostics are explicitly labeled and do not supply the full-score demonstration. The published executable task files and physical specification remain byte-identical to the evaluated source; submission changes are documentation, metadata, ignore rules and a copied CPU-only model auditor.
