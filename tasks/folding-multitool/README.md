# Deploy butterfly multitool pliers and retrieve a recessed coupling nut

The robot must unfold a passive butterfly multitool and use its handles to
grip a coupling nut between the working jaws inside a narrow
recess, extract and carry it, then release it into a tray and clear the nut.
Functional deployment must transmit force through the tool's linkage.

The scene revision is `coupling-nut-rectangular-recess-v1`; the scoring revision
is `retrieval-without-empty-cycle-v2`. The scripted
demonstration scored **1.0 in 3,018 actions (150.9 simulated seconds)**. The one
genuine hosted **Dream-1 attempt scored 0.0 after 3,600 actions (180 simulated
seconds)**. Both used seed 0 and the same scene, reset, instruction, physics,
original scorer, cameras and recording path. Independent frozen-source rescoring
matched every recorded score. Separate CPU assessments with the revised scorer
also match every recorded score and preserve both outcomes. No new episode or
model call was needed. The videos include the reset and every executed action.
See [EVIDENCE.md](EVIDENCE.md) and [evidence.json](evidence.json) for provenance.

## Scene and passive mechanism

`task.json` gives the fixed layout in the native station frame. The tabletop is
at z=0.7475 m. A folded steel tool rests along the x direction on two wooden
supports. The fixed metal maintenance coupon contains the upright loose nut;
the separate blue rigid catch tray is the destination. Nut and fixture dimensions
are in `retrieval_scene.py`; the supports and reset are in `task.py` and
`geometry.py`. Colors/material contacts approximate dry rigid surfaces. The
fixed coupon/tray are modeled as attached fixtures, not movable manipulation
objects. Both have real open interiors made from five native box colliders.

The product reference is the [Leatherman Rebar](https://www.leatherman.com/products/rebar).
The published envelope is 101.6 mm folded / 158.75 mm open, 29.972 mm wide,
16.51 mm thick and 188.5 g. Internal geometry, component inertia and mechanics
are **unmeasured estimates**, not a calibrated product replica. The plier head
has **no lock**; other locking implements do not imply a locking plier head.
The [manufacturer's spring/abutment patent](https://patents.google.com/patent/US7146668B2/en)
is explanatory, not evidence of exact internals for this specimen.

The floating four-body graph is
`handle_a -- fold_a -- jaw_a -- jaw -- jaw_b -- fold_b -- handle_b`.
Fold pivots are estimated at x=-14 mm, y=+/-7.486 mm relative to the central
pivot. Jaw limits are [-0.70,0] rad; fold A [0,pi], fold B [-pi,0]. Reset folds
are +pi and -pi with the jaw closed. Deployed folds seat at unilateral zero
stops. Native implicit Coulomb/viscous friction is 0.015 Nm / 0.002 Nm s for
folds, 0.008 Nm / 0.001 Nm s for the jaw. The conservative end-detent potential
has a finite 6 mJ barrier; it never snaps joint angles. Contact static/dynamic
friction is 0.45/0.35, contact offset 0.3 mm, restitution zero. Hollow handle
channels, hinge ears, stowed-implement occupancy and self-collision are included.
There is no attachment, opening spring, tool trajectory drive or post-reset tool
state write. Only the original robot receives standard joint/gripper targets.

The nut is an [M6 DIN 6334 coupling nut](https://www.accu.co.uk/hexagonal-stud-connector-nuts/62290-HTBN-M6-A2),
10 mm across flats, 18 mm tall, catalogue mass 9.57 g. Twelve convex sectors
preserve its approximately 5 mm through-bore. Threads and chamfers are omitted.
The 24 x 24 x 28 mm recess is a credible access problem for narrow plier jaws;
its dimensions were not chosen to enlarge the actual tool or robot fingers.

## Scoring and checks in the real world

`scorer.py`, `retrieval_score.py` and `jaw_contact.py` read measured state and
contacts independently of controller stage labels. No score comes from requested
poses or desired trajectories.

- **0.25 (optional):** a folded reset; both folds within 5 degrees of deployed; opposing
  original fingers pinch separate handles; jaws open >=14 mm for 10 control
  frames, then close <=1 mm for 10 frames (0.5 s each). Tool head and non-finger
  robot contacts cannot support functional use. This credit records the cycle;
  it is not a prerequisite for retrieval credit.
- **0.5:** from the folded reset, deploy both folds within 5 degrees and begin
  an opposing inner-jaw pinch inside the recess. Original fingers must load both
  handles, without supporting the head or using other robot links. Either or
  both grippers may supply these handle contacts; no arm assignment is prescribed.
  No preceding empty-jaw cycle or tip-down dwell is required. Keep the nut
  bottom >=10 mm above the rim for 1 s, then carry >=30 mm horizontally from
  that extraction position without other support. Carry credit is historical.
- **1.0:** release from that same carry wholly over the tray with 3 mm edge
  margin. The whole nut must rest on the tray floor below the rim, move <5 mm/s,
  drift <=2 mm over 40 consecutive frames (2 s), and be >=20 mm from all robot
  and tool geometry. Losing the carry elsewhere and subsequently pushing the
  nut into the tray does not qualify. Otherwise the achieved lower score remains;
  no qualifying history means zero. Nonfinite state invalidates the score.

The simulator checks opposing inner-jaw contact directions/locations and force
>0.1 N per jaw. Total non-jaw robot/handle support must stay below 0.003 N;
fixture/unattributed support is separately excluded during extraction/carry.
Contact points are partner averages, not a proof of every individual contact
patch. Geometric clearance uses conservative bounds with measured body poses.
The final released check requires all tool/robot contacts <0.02 N and stricter
non-jaw exclusion. A nut merely placed in the tray cannot satisfy the history.

For a real-world check use a ruler/caliper, marked tray margins and synchronized
side/overhead video to observe fold angles, jaw gap, extraction height, carry,
release, drift and clearance. Verify that the opposing jaws alone retain the
nut and that the robot loads both handles without supporting the head or nut. Small
force thresholds are numerical contact-noise tolerances for the simulation;
real-world evidence needs visible separation and retention under gravity.
Before hardware trials, verify actual jaw fit and grasps in this 24 mm recess,
and measure hinge torque-angle, stop faces, collapse resistance, component
inertia and friction. Keep hands clear
of hinge pinch points. This simulation is not a hardware calibration.

## Environment and reproduction

Use the existing native `yam.env`, stock YAM URDF/meshes, gains and joint limits.
No dependency install or shared configuration edit is needed on the evaluated
workstation. See [ENVIRONMENT.md](ENVIRONMENT.md) for measured package versions
and external runtime dependencies. This package does not vendor Isaac Sim, YAM,
or the cooperative GPU scheduler.

Run from the repository root. Adjust installation roots only if installed
elsewhere. The stable scheduler task ID is **folding-multitool**, including when
this folder is renamed. Every simulator invocation must stay inside the scheduler
job and its descendants. Use exclusive mode for numerical comparisons; ordinary
episodes can use dev mode. Physics is masked `cuda:0`, rendering uses the
assigned physical GPU, and multi-GPU rendering is disabled. Warmup verifies the
actual physics tensor, CUDA UUID and active renderer, and saves that evidence.

```bash
source "$HOME/IsaacLab-Arena/yam.env"
export ROBOCURVE_SCHEDULER_ROOT="$HOME/robocurve-gpu-scheduler"
FOLDING_TASK_DIR="$PWD/tasks/folding-multitool"
FOLDING_RUNS="$HOME/yam-eval/folding-multitool/runs"

"$ROBOCURVE_SCHEDULER_ROOT/gpu-scheduler" launch \
  --task folding-multitool --mode dev --gpu auto --wait 300 --detach -- \
  python -B "$FOLDING_TASK_DIR/run.py" --mode retrieval --controller scripted \
  --max-steps 3600 --video --out "$FOLDING_RUNS"
```

The runner prints `ARTIFACTS <run-directory>`. The accepted demo took 1,314.5 s
wall time; the model episode took 782.9 s. These are elapsed observations under
cooperative scheduling, not exclusive performance measurements. Seed is 0,
physics 240 Hz, control 20 Hz, and the maximum is 3,600 actions. Success, a fatal
error or the action limit terminates the episode. The reset guard rejects a nut
outside its declared recess floor before action 0.

Pure CPU checks (no simulator or provider call):

```bash
source "$HOME/IsaacLab-Arena/yam.env"
cd tasks/folding-multitool
python -B -m unittest -v test_mechanism test_motion test_jaw_contact \
  test_load_score test_retrieval test_evaluation test_preflight \
  test_primary_score test_scoring_revision
```

The offline native interface test uses the same complete retrieval scene:

```bash
"$ROBOCURVE_SCHEDULER_ROOT/gpu-scheduler" launch \
  --task folding-multitool --mode exclusive --gpu auto --wait 300 --detach -- \
  python -B "$FOLDING_TASK_DIR/run.py" --mode retrieval \
  --controller interface-check --max-steps 128 --video --out "$FOLDING_RUNS"
```

It must report `interface-check.json` with `passed: true`. It verifies repeat
reset, three distinct RGB streams, measured/native joint and gripper ordering,
asymmetric target tracking, 12 physics steps per action, paused inference and
budget termination. It uses two local fixture replies and makes no network call.
The completed native check is preserved; no new simulation was run for packaging.

### Dream boundary and recorded command

Dream receives only the shared instruction, three 320 x 176 RGB streams
`top_cam`, `left_cam`, `right_cam`, and measured 14D state ordered
`[left_j0..5, left_gripper, right_j0..5, right_gripper]`. Joint angles are radians;
grippers are 0 closed / 1 open. Privileged observations are removed. The installed
station adapter adds static YAM metadata and its previous commanded row for
chunk joining, without task objects, contacts, stages, scores or scripted advice.
The initial 32-action measured-pose hold fills the observation window and counts
against the budget. The model's ordinary targets execute at 20 Hz, with native
joint-limit clipping recorded beside raw targets and no interpolation or scripted
completion. Physics pauses during network waits. Provider account recording is off.

This is the completed hosted invocation, expressed with portable paths. It would
make another paid attempt if executed; the actual 0.0 outcome is preserved.

```bash
FOLDING_DEMO="$FOLDING_RUNS/7a988ed297694ed1b9792ec7b06897cc"
FOLDING_INTERFACE="$FOLDING_RUNS/74bbb0f4c2464a779b1917397b65fd1b"
"$ROBOCURVE_SCHEDULER_ROOT/gpu-scheduler" launch \
  --task folding-multitool --mode dev --gpu auto --wait 300 --detach -- \
  python -B "$FOLDING_TASK_DIR/run.py" --mode retrieval --controller dream \
  --max-steps 3600 --video --demo-run "$FOLDING_DEMO" \
  --interface-run "$FOLDING_INTERFACE" --out "$FOLDING_RUNS"
```

Preflight requires a reviewed full-score demo, passed native interface check,
matching episode sources and locally configured Dream credentials. An explicitly
hashed scoring-only revision may reuse the original evidence after both original
traces are rescored; `scoring_revision.py` rejects any other episode change. The runner
and evaluation boundary must match the tested versions. The only permitted
legacy runner difference is the two explicitly checked scheduler/output path
substitutions documented in `EVIDENCE.md`; any runtime edit still blocks it.

Rescore an actual saved run and prepare media from existing recordings, on CPU:

```bash
python -B "$FOLDING_TASK_DIR/rescore.py" "$FOLDING_DEMO"
FOLDING_MODEL="$FOLDING_RUNS/49b19db344a5489b9249701a7b3e43c5"
python -B "$FOLDING_TASK_DIR/rescore.py" "$FOLDING_MODEL"
python -B "$FOLDING_TASK_DIR/prepare_media.py" --review-run "$FOLDING_MODEL"
python -B "$FOLDING_TASK_DIR/prepare_media.py" --demo-run "$FOLDING_DEMO" \
  --model-run "$FOLDING_MODEL" --out "$HOME/yam-eval/folding-multitool/evaluation"
```

Raw recordings, per-step traces, inference inputs/replies, original frozen source,
seeds, native-source hashes and device audits are preserved outside Git. This
submission includes compact provenance and independent assessment summaries;
rescore commands require the original run directories, not the uploaded videos.
The videos are anonymous H.264/yuv420p, under 50 MB; setup.png is under 10 MB.

The commands above use each run's original frozen scorer. To reproduce the
review revision, write separate reports outside the original run directories:

```bash
python -B "$FOLDING_TASK_DIR/rescore_revision.py" "$FOLDING_DEMO" \
  --role demo --out /tmp/folding-multitool-rescored-demo.json
python -B "$FOLDING_TASK_DIR/rescore_revision.py" "$FOLDING_MODEL" \
  --role model --out /tmp/folding-multitool-rescored-model.json
```

`rescored-demo.json` and `rescored-model.json` in this package contain the resulting
assessment and old/new source hashes. The original records and uploaded media
`M-JTNCGU` are unchanged. Regression tests show that the primary scorer awards
1.0 with all empty-cycle flags false. Alternative finger assignments in those
tests check scoring semantics; they are not additional physical demonstrations.

## Controller and source map

The scripted controller uses privileged local measurements for feasibility.
Left pinches folded handle B near its pivot; right grips A farther out. Together
they lift and unfold A against B's folded stop, regrip A across its width and
turn the head to expose jaw B. Right grasps that jaw while left unfolds B halfway,
regrips and completes B. Both arms regrip separate working handles from outboard,
open/close the jaws and turn the tool tip-down. They translate over the recess,
open, lower to the nut's midpoint, pinch, extract, carry, lower over the tray,
release and withdraw. All tool motion arises from robot contact and passive
mechanics. Missing contact, failed deployment or unreachable held-tool poses
stop and preserve the failure. Logged free-arm IK fallbacks do not change tool
state and are absent from the model episode.

| Files | Purpose |
| --- | --- |
| `mechanism.json`, `mechanism.py`, `geometry.py`, `tool_asset.py`, `passive_actuator.py` | Passive linkage and actual-envelope geometry |
| `task.py`, `retrieval_scene.py`, `compound_asset.py`, `adapter.py` | Native scene/reset, collision shapes and measured state |
| `controller.py`, `motion.py`, `load_controller.py`, `retrieval_controller.py` | Contact-driven scripted feasibility sequence |
| `scorer.py`, `retrieval_score.py`, `jaw_contact.py`, `clearance.py` | Independent observed scoring |
| `run.py`, `episode_config.py`, `evaluation.py`, `preflight.py`, `audit.py` | Shared episode, restricted model I/O, scheduler and native audits |
| `rescore.py`, `rescore_revision.py`, `scoring_revision.py`, `review_run.py`, `prepare_media.py` | Original/revised CPU assessments, provenance and media export |
| `load_gauge.py`, `load_score.py`, `reset_diagnostic.py`, geometry/grasp checks | Separate diagnostic investigations; retained dependencies |

### Historical diagnostics and limitations

Deployment/load scenes are not submission episodes. The earlier 1 N gauge hold
was re-evaluated with a disclosed 2 s pose-window criterion after an instantaneous
angular-velocity threshold rejected a nearly stationary pose. Its original failed
result remains preserved. Load modules remain imported by the scorer/controller;
they must not be deleted as unused historical files. Earlier short-nut scenes
had invalid resets and are not submitted as evidence. The complete revised demo
uses the published-size coupling nut and real five-box recess/tray.

CPU tests reject single-jaw carry, robot/handle/fixture support, direct placement
and drop-then-push shortcuts. A 660-sample recorded-grasp IK check stays inside
stock joint limits; 240 sampled direct-finger poses collide with the recess.
These finite checks do not exclude every tilted finger grasp, hook or scoop.
The scorer requires the complete opposing-jaw history even if another strategy
moves the nut. No exhaustive dynamic shortcut or broad robustness claim is made.

The key capability goes beyond merely unfolding an object or using a ready tool:
contact-driven reconfiguration, handovers, seating passive stops, then coordinated
handle loading for useful jaw force. Small nut images and wrist occlusion limit
video-only contact verification; the independent assessment also uses measured
geometry and contact data. Hardware friction, detents and collapse resistance
remain to be measured.
