# Fasten three upper snaps on a table-supported shirt

Fasten the three upper front press-studs of a laid-out shirt, withdraw both
robots at least 30 mm, and leave the closures retained for three seconds.
The declared budget is **6,000 ordinary 14D actions at 20 Hz (300 seconds)**.
The task uses the stock dual-arm YAM robot. The successful script presses with
one arm and positions the other wrist camera; no bimanual benefit is claimed.

The submitted scripted demonstration scored **1.0** in **905 actions / 45.25 s**.
A fresh-process reproduction scored **1.0** in **895 actions / 44.75 s**. Both
use seed 100 and the same native 20 Hz physics/action/scoring implementation as
the genuine Dream-1 attempt. Independent raw-state rescoring agrees.

The genuine `dreamlabs/dream-1-beta` attempt scored **0.0**, then ended with
`DreamSessionExpired` after **80 actions / 4.0 simulated seconds**: 32 standard
observation-preroll holds and 48 genuine model targets. The first inference
completed; the second reported a server-side session replacement. No scripted
assistance, fallback or episode retry was used. The recording is submitted with
its actual zero score. This interrupted attempt does not measure performance
over the full budget. See [EVIDENCE.md](EVIDENCE.md) for the error, audits,
limitations and media hashes.

## Setup and fidelity

The fixed scene contains a representative adult long-sleeve snap-front shirt,
a rigid wood-like 1.0 x 1.8 m table, and two stock YAM arms with parallel grippers.
The table top is 0.75 m high, centered at (0.605, 0.12) m. Both arms start READY.
The shirt starts front-up, with nominal placket overlap and snap alignment
supplied by the layout. Estimated dimensions are 0.76 m body length, 0.56 m flat
chest width and 1.56 m sleeve span; estimated fabric mass is 180 g. Seven mutually
compatible 12 mm metal snap pairs lie at 80 mm pitch. Targets 0, 1 and 2 start at
x = 0.28, 0.36, 0.44 m, y = 0.12 m. The other four pairs need not be closed.
The setup screenshot is the first recorded overhead frame, at 0.05 s.

The real-product reference is a [Wrangler broadcloth western snap shirt](https://www.wrangler.com/shop/wrangler-long-sleeve-solid-broadcloth-western-snap-shirt-711.html),
recorded as 55% cotton / 45% polyester, 4 oz broadcloth. The pattern, mechanical
properties and snap internals are estimates; no hardware measurements were
available. This is not a calibrated SKU replica. Collar, cuffs and pockets are
simplified. Shirt spreading, dressing and putting it on a hanger are outside
this tabletop task; the historical hanger investigation did not succeed.

Flexible surface FEM and local rivets couple the garment to passive four-ball
spring sockets. Mechanical entry, pressing, retention and release arise from
contact. There is no proximity latch, attraction, runtime fabric positioning or
hidden grasp constraint. Wrong compatible pairings remain physically possible;
correct pairing is a scoring requirement. Ordinary buttons are not modeled.

The validated configuration uses 960 Hz physics, 64 cloth solver iterations,
a 5 mm body mesh with 2 mm rivet-patch sampling, and 1 mm cloth contact offset.
The base [parameters.json](parameters.json) retains earlier defaults; the
validated stability configuration supplies these recorded overrides. Every
native episode writes the complete effective configuration to `setup.json`.
Robot gains, limits, gravity compensation and the standard 14D actions remain
stock. Material and numerical limitations are detailed in [REVISIONS.md](REVISIONS.md).

## Score and reset

Score is 0, 1/3, 2/3 or 1 for target pairs correctly seated throughout the final
three-second released interval. Both complete robots must stay at least 30 mm
from the fabric and hardware; at least 98% of cloth nodes remain over the table;
mean nodal position drifts at most 20 mm. Wrong pairing during this interval
prevents credit. Controller phases and arm roles do not earn credit.

Seating requires socket-frame axial separation strictly between 6.0 and 7.2 mm,
lateral error at most 0.75 mm, and normal misalignment below 12 degrees. Invalid
simulation episodes receive zero: nonfinite states, cap penetration through a
socket rim, absolute edge strain above 15%, rivet error or table penetration
above 1 mm, or rigid robot/snap normal contact above 25 N. These simulation guards
are separate from the physical task's retained-closure, clearance and drift
conditions. The table-depth guard also flags cloth curled under the underside.

Every evaluation episode reconstructs the native scene, cloth, hardware, rivets,
robots and controller/contact state in a fresh process. The chosen configuration
passed seed-100/101 ten-second screens and a full seed-100 300-second passive
stability check. Historical same-process reset failures are retained; no 5/5,
9/10 or broader statistical reliability claim is made.

## Environment and paths

Use the existing compatible native runtime; do not upgrade it to reproduce
these results. [DEPENDENCIES.md](DEPENDENCIES.md) records exact observed versions.
Python 3.12, Isaac Sim/Lab, Arena/YAM, Torch, NumPy, imageio/FFmpeg and the installed
Inspect Robots/Dream integration are required. GPU-free unit checks use that
same environment. Source and version identifiers are in
[submission-evidence.json](submission-evidence.json).

From the submission repository root, configure locations for your host:

```bash
export ARENA_CHECKOUT=/path/to/IsaacLab-Arena
export SNAP_TASK_DIR="$PWD/tasks/snap-front-shirt"
export SNAP_OUTPUT_ROOT=/path/to/private/snap-front-shirt
export ROBOCURVE_SCHEDULER_ROOT=/path/to/existing/robocurve-gpu-scheduler
export SNAP_GPU_LOCK=/path/to/shared/.robocurve-gpu-0.lock
source "$ARENA_CHECKOUT/yam.env"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
```

`task_paths.py` centralizes these paths. Defaults are relative to the current
user's home, not the development account. On the shared development host, point
to the existing canonical scheduler and locks; do not create another scheduler
runtime or change reservations. The stable task ID is always `snap-front-shirt`.

Inspect native jobs and GPU use before launching. Wrap the complete launcher
and all children with the scheduler; use exclusive mode for validation:

```bash
"$ROBOCURVE_SCHEDULER_ROOT/gpu-scheduler" status
nvidia-smi
python -B -m unittest discover -s "$SNAP_TASK_DIR" -p 'test_*.py'
python -B "$SNAP_TASK_DIR/validate_submission.py"
```

Each simulator validates membership through `isaac_binding.assigned_gpu()`,
binds masked physics to `cuda:0`, selects rendering with the assigned physical
`ROBOCURVE_GPU_INDEX`, and disables multi-GPU rendering. Actual tensor device,
CUDA UUID, renderer and process allocation checks are recorded per episode.
A direct unscheduled launcher retains an exclusive GPU-0 fallback and occupancy
check. Invalid scheduler membership never falls back. Dream stages require the
canonical scheduler. Put `--detach` on the scheduler, not the inner launcher.

## Reproduce and independently score

Use fresh output names. Existing completed evidence can be reused after its
source and raw-state audits pass; it need not be simulated again for submission.
For a fresh installation, generate the stability evidence first:

```bash
"$ROBOCURVE_SCHEDULER_ROOT/gpu-scheduler" launch --task snap-front-shirt --mode exclusive --gpu auto --wait 300 -- \
  python -B "$SNAP_TASK_DIR/launch.py" --stage stability --out "$SNAP_OUTPUT_ROOT/stability"
python -B "$SNAP_TASK_DIR/review_stability_round.py" "$SNAP_OUTPUT_ROOT/stability" \
  --out "$SNAP_OUTPUT_ROOT/stability/verification.json"
```

The bounded historical stability search preserves failed candidates and selects
64 iterations / 5 mm only after both screens and the complete horizon pass.
It can take several hours. It is not a task-success demonstration.

Generate two interface/reset checks and the scripted demonstration/reproduction:

```bash
"$ROBOCURVE_SCHEDULER_ROOT/gpu-scheduler" launch --task snap-front-shirt --mode exclusive --gpu auto --wait 300 -- \
  python -B "$SNAP_TASK_DIR/launch.py" --stage evaluation --out "$SNAP_OUTPUT_ROOT/validation/interface" -- \
  --stability-run "$SNAP_OUTPUT_ROOT/stability" --mode hold --control-hz 20 --seconds 1 --episodes 2 --interface-checks
"$ROBOCURVE_SCHEDULER_ROOT/gpu-scheduler" launch --task snap-front-shirt --mode exclusive --gpu auto --wait 300 -- \
  python -B "$SNAP_TASK_DIR/launch.py" --stage evaluation --out "$SNAP_OUTPUT_ROOT/validation/baseline-reproduction" -- \
  --stability-run "$SNAP_OUTPUT_ROOT/stability" --mode scripted --control-hz 20 --seconds 300 --episodes 2 --seed 100
python -B "$SNAP_TASK_DIR/verify_tabletop_run.py" "$SNAP_OUTPUT_ROOT/validation/baseline-reproduction/worker-00" \
  --out "$SNAP_OUTPUT_ROOT/demo-verification.json"
python -B "$SNAP_TASK_DIR/check_tabletop_shortcuts.py" "$SNAP_OUTPUT_ROOT/validation/baseline-reproduction/worker-00" \
  --out "$SNAP_OUTPUT_ROOT/shortcut-verification.json"
```

The submitted 45.25 s / 44.75 s episodes used 2094.51 s / 2507.41 s native wall
time, excluding startup. Control frequency refers to simulated time. Each
worker records actions, measured state, physics-rate observations, score,
termination reason, seed, full configuration, source hashes and three videos.
The scorer can be replayed without the simulator using `verify_tabletop_run.py`
and `verify_streaming_replay.py`. Exact-action native replay is a separate
experiment: the recorded 905 targets executed exactly but scored 0.0, with
valid physics. Full-score feedback reproduction is the demonstrated result.

## Dream interface and existing model evidence

`dream_live_runner.py` uses the installed `inspect-robots-dream` policy/session
and the same native worker, scene, reset, physics, instruction, action budget,
scorer and recording as the script. Input is instruction text, measured left
six joints/gripper then right six joints/gripper, and `top_cam`, `left_cam`,
`right_cam` RGB at 320 x 176. Privileged cloth/contact/object state is only for
scoring and diagnostics. External policy execution sets `controller=none` and
rejects scripted steps. Ordinary absolute joint/gripper targets execute at
20 Hz; inference waits do not advance physics.

The renderer's time-zero wrist transforms are stale. The wrapper excludes only
that observation during the installed 32-action preroll. The first real forwarded
sample is at 0.05 s. No action is added, no frame is fabricated, and all holds
count against the 6000-action budget. The native five-second adapter check
verified this, action ordering, joint responses, timing and script rejection.

Offline `hold`, `scripted`, `fake` and action-replay modes deny Internet sockets.
The separate live runner has no model fallback, episode retry or application
inference retry. A durable attempt record prevents accidentally repeating a
paid episode. Credentials follow the installed SDK (`DREAM_API_KEY` or the
private credential file); secrets are never committed. Account-side observation
recording is disabled. A new hosted episode needs its own authorization.

For an independently authorized reproduction after local checks, the command
shape is:

```bash
"$ROBOCURVE_SCHEDULER_ROOT/gpu-scheduler" launch --task snap-front-shirt --mode exclusive --gpu auto --wait 300 -- \
  python -B "$SNAP_TASK_DIR/launch.py" --stage dream_local_check --out "$SNAP_OUTPUT_ROOT/dream-local-check" -- \
  --stability-run "$SNAP_OUTPUT_ROOT/stability" --validation-queue "$SNAP_OUTPUT_ROOT/validation" \
  --demo-run "$SNAP_OUTPUT_ROOT/validation/baseline-reproduction/worker-00"
python -B "$SNAP_TASK_DIR/prepare_review_media.py" --run "$SNAP_OUTPUT_ROOT/validation/baseline-reproduction/worker-00" \
  --out "$SNAP_OUTPUT_ROOT/demo-media"
# Inspect the exported views and record their review before any hosted access.
"$ROBOCURVE_SCHEDULER_ROOT/gpu-scheduler" launch --task snap-front-shirt --mode exclusive --gpu auto --wait 300 -- \
  python -B "$SNAP_TASK_DIR/launch.py" --stage dream_live --out "$SNAP_OUTPUT_ROOT/dream-attempt" -- \
  --stability-run "$SNAP_OUTPUT_ROOT/stability" --validation-queue "$SNAP_OUTPUT_ROOT/validation" \
  --demo-run "$SNAP_OUTPUT_ROOT/validation/baseline-reproduction/worker-00" \
  --local-check "$SNAP_OUTPUT_ROOT/dream-local-check" --demo-media "$SNAP_OUTPUT_ROOT/demo-media" --seed 100
```

Existing `model_run.mp4` already records the genuine attempt; it should not be
rerun to improve the score. Session continuity remains unresolved. Approximately
106.18 wall seconds elapsed between the first response and second request;
slow simulation is a possible contributor, not an established server-side cause.

## Source layout and historical investigations

The active task is `tabletop_scene.py`, `tabletop_backend.py`, `local_episode.py`,
`policy_worker.py`, `tabletop_control.py`, the scoring modules, and the evaluation
and Dream runners. Geometry, garment mesh, robot utilities and `bench.snapshot`
are shared dependencies. They are retained at their original module names.

`bench.py`, `garment_runner.py`, hanger/contact/grasp diagnostics, plot helpers,
bimanual pilots and legacy coordination helpers preserve earlier investigations.
They are not additional claimed benchmarks. Historical queues and the legacy
GPU guard should not be attached to new scheduled jobs. Read
[REVISIONS.md](REVISIONS.md) for failed prototypes and rejected hypotheses.

Raw recordings, simulator caches, credentials and large measured-state artifacts
stay outside Git. [submission-evidence.json](submission-evidence.json) records
compact outcomes, versions and hashes. `package_revision.py` can inventory an
explicit completed evidence set; its source archive contains no raw recordings.
The p19 submission changes only paths, documentation, metadata and an exact
reviewed helper hash allowance. Native physics, robot, controller, policy and
scoring sources match the validated recordings. Submission is not acceptance.
