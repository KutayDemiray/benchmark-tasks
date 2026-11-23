# Live Dream-1 evaluation

> Historical investigation: results and pending-work statements below refer to
> earlier prototypes, released-v1, or the deferred SDK path. The active submitted
> endpoint and completed evidence are in [README.md](README.md). These diagnostics
> do not change the submitted physics or scoring.

Current evidence (2026-11-22): original released-v1 scripted score **0.5** in
`base-r123-fixed-arm-opening`; its archived source hashes and recorded score were
rechecked in `current-evidence-review.json`. No released full score, audited
fresh-process replay or genuine hosted model score exists.

`base-r167-local-interface` passed a real native64-action check with the installed
DreamPolicy and a local no-network session: three320x176 camera streams, measured
14D state, action timing/termination, two complete resets and video decoding.
Actual physicscuda:0 and physical rendererGPU1 were verified after warmup.
This diagnostic is not a benchmark score. Its video is
`$HOME/yam-eval/pocket-mop-cover/progress-videos/2026-11-22-live-interface-check.mp4`.

The installed SDK passes reduced passive-settling and latch diagnostics. The fresh
insertion/withdrawal cycle `base-r163-sdk-cycle` completed and passed its complete independent native
contact audit (13,440 states, peak strain1.63%, no crossings or penetration). Its stock-grasp/camera repeat awaits GPU admission
after an installedUSD wrapper compatibility fix. All26 native stock YAM property
comparisons passed before that rendering error. The shared full SDK wrapper is
implemented but has not yet passed native validation. No physical or scoring
thresholds were relaxed. The commands below require final matched evidence.

## Resume the outstanding SDK qualification

The insertion cycle and independent Kit stock-property capture are complete.
The next native run is `base-r165b-sdk-grasp`; admission timeouts have not created
that run directory. Check `sdk-grasp-admission.json` and scheduler status first:
a bounded admission retry may already be pending. Do not launch a duplicate.
Use this command after the previous admission process has finished:

```bash
source $HOME/IsaacLab-Arena/yam.env
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
$HOME/robocurve-gpu-scheduler/gpu-scheduler launch \
  --task pocket-mop-cover --mode exclusive --gpu auto --wait 300 --detach -- \
  python -B $HOME/benchmark-tasks/tasks/pocket-mop-cover/launch.py \
  --name base-r165b-sdk-grasp --experiment robot_grasp --product-profile base_micro \
  --physics-hz 480 --iterations 64 --seconds 40 --table-origin --placement table \
  --grasp-margin-clearance .006 --grasp-opening .008 --close-seconds 3 \
  --spacing .0025 --contact-updates 8 --contact-subiterations 2 --ccd \
  --grasp-sdk-source base-r84-clear-margin-table \
  --grasp-sdk-reference base-r164-kit-yam-contract
```

After its completed result and process status are verified, run these CPU audits
with the archived source. An error requires inspection and a fresh run name;
never overwrite evidence or bypass a failed qualification.

```bash
source $HOME/IsaacLab-Arena/yam.env
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
TASK=$HOME/benchmark-tasks/tasks/pocket-mop-cover
OUT=$HOME/yam-eval/pocket-mop-cover
RUN="$OUT/base-r165b-sdk-grasp"
python -B "$RUN/source/audit_grasp.py" "$RUN"
python -B "$RUN/source/verify_videos.py" "$RUN"
python -B "$RUN/source/qualify.py" --cycle base-r163-sdk-cycle \
  --grasp base-r165b-sdk-grasp --output product-sdk-gate.json
```

Inspect all three actual camera recordings. Only a passing SDK product gate
permits the full SDK scene commands below. The failed earlier `base-r165-sdk-grasp`
is retained: stock-property comparison passed, but rendering failed before any
motion because the installed USD distribution lacks the `UsdRender` wrapper.
The task-local adapter now uses standard typed render prims and the installed
OVRTX `deviceIds` contract; no installed files were changed.

## Shared episode and public interface

`frame_robot.py` builds the same stock YAM, free folding frame and unattached
sewn cover, performs the same reset and verifies the actual physics/render GPU.
`RobotSession` records every native substep and all three cameras. The script and
model use that exact session; the model branch is selected before any scripted
movement. Scripted controller feedback remains outside the policy interface.

Dream receives the declared task instruction, exactly `top_cam`, `left_cam`,
`right_cam` RGB images at 320 x 176, and measured state in this order:
left six arm joints in radians, left gripper, right six arm joints, right gripper.
Grippers are normalized: zero closed, one open. Neither object poses, contacts,
geometry, scores, success flags nor controller plans enter model observations.

The installed `DreamPolicy` (`inspect-robots-dream` 0.1.0) uses
`dreamlabs/dream-1-beta`. The first 32 actions collect camera history while holding
the measured initial target. These count against the 2,400-action budget. Ordinary
14D targets execute at 20 Hz with 24 native substeps per action at 480 Hz. Physics
pauses during hosted inference. Stock position/gripper clamping is preserved;
physical gains, efforts, gravity compensation and velocity limits are unchanged.

`--policy interface-check` runs the actual installed policy with a local mock
session and no network. **Do not use DreamPolicy dry_run for local validation:**
it still makes hosted requests. The local native check exercises reset, all camera
streams, measured state, 14 action slots, timing, termination and recording.
It is diagnostic and cannot receive benchmark credit. CPU fixtures validate the
same wire format and enforce the absence of network or scripted-controller calls.

## Commands

Use the existing native environment. Every simulator launch must retain this
scheduler ID; do not bypass an occupied GPU or alter other reservations.

```bash
source $HOME/IsaacLab-Arena/yam.env
TASK=$HOME/benchmark-tasks/tasks/pocket-mop-cover
OUT=$HOME/yam-eval/pocket-mop-cover
SCHEDULER=$HOME/robocurve-gpu-scheduler/gpu-scheduler
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1

COMMON=(--experiment frame_robot --product-profile base_micro
  --physics-hz 480 --iterations 64 --seconds 120 --table-origin --snap-throat
  --robot-unlock --robot-reopen --robot-cover --cover-offset-x .02
  --frame-lift .4 --spacing .0025 --contact-updates 8 --contact-subiterations 2
  --ccd --reset-check --completion-mode released --camera-width 320 --camera-height 176)

# CPU only; never starts a simulator or contacts Dream.
python -B -m unittest discover -s "$TASK" -p 'test_dream_evaluation.py' -v

# Local native adapter validation with an explicitly no-network session.
"$SCHEDULER" launch --task pocket-mop-cover --mode exclusive --gpu auto --wait 300 -- \
  python -B "$TASK/launch.py" --name interface-released-new "${COMMON[@]}" \
  --policy interface-check --policy-check-steps 64
python -B "$OUT/interface-released-new/source/verify_videos.py" "$OUT/interface-released-new"

# Candidate demonstration: the existing Kit physics is still known to fail release.
# This command does not resolve or bypass that physical blocker.
"$SCHEDULER" launch --task pocket-mop-cover --mode exclusive --gpu auto --wait 300 -- \
  python -B "$TASK/launch.py" --name demo-released-new "${COMMON[@]}" --policy scripted
RUN="$OUT/demo-released-new"
python -B "$RUN/source/audit_frame_robot.py" "$RUN"
python -B "$RUN/source/audit_cover_parallel.py" "$RUN" --workers 16
python -B "$RUN/source/score_full_run.py" "$RUN" --workers 8
python -B "$RUN/source/verify_videos.py" "$RUN"

# Fresh-process replay, with the same scene/reset/physics and recorded actions.
"$SCHEDULER" launch --task pocket-mop-cover --mode exclusive --gpu auto --wait 300 -- \
  python -B "$TASK/launch.py" --name replay-released-new "${COMMON[@]}" \
  --policy scripted --replay-run demo-released-new
# Repeat all four audit/score/video commands above with RUN=replay-released-new.

# ONE hosted episode, only after both independent full scores equal 1.0 and
# the native interface check passes against these same sources and settings.
"$SCHEDULER" launch --task pocket-mop-cover --mode exclusive --gpu auto --wait 300 -- \
  python -B "$TASK/launch.py" --name dream-released-one "${COMMON[@]}" --policy dream \
  --demo-run demo-released-new --replay-evidence replay-released-new \
  --interface-evidence interface-released-new
# Audit, rescore and verify the model recording with those same four commands.
# Preserve its actual score, including zero. Do not relaunch to improve it.
```


For the installed SDK candidate, append the following three flags to **every**
interface, demo, replay and hosted command above, using fresh run names:

```bash
--full-sdk-source base-r167-local-interface \
--product-gate product-sdk-gate.json --mechanism-gate mechanism-sdk-gate.json
```

This reconstructs the declared original starting state from the archived full
scene and uses `full_sdk.py` for the same reset/step/observe/record boundary.
The script, local interface and Dream share `RobotSession`, `full_observation`
and the independent scorers. Actual SDK cooked collision hulls are captured in
a short separate initialization process under the same scheduler reservation;
its dynamic state is discarded. The full SDK launcher refuses admission until
its separate matching material/grasp and mechanism gates pass. The completed Kit
interface check is not a substitute for a fresh SDK interface check after those
gates pass.

The host gate checks immutable source hashes, installed YAM/policy sources,
physical configuration, instruction/specification, initial state, native scene
properties, complete audits, video decoding, actual GPU verification and replay
provenance. It fails before opening the model session if any prerequisite is
missing or stale. All three qualification runs must use the same camera resolution
and the final shared physics implementation. If the physical implementation
changes, revalidate all affected gates and regenerate the demo/replay/interface
records. The SDK reduced scenes cannot be substituted for a full-task demo.

Credentials are read by the installed client from its existing configuration
(`DREAM_API_KEY` or `~/.dreamlabs/credentials`); values are never copied into run
artifacts. Configured key presence is confirmed, but model authorization/network
access is untested. `hosted-dream-attempt.json` is created exclusively in the task
output root before the first hosted episode. It prevents automatic paid retries,
including after errors or a zero score. Access failure is reported as such rather
than fabricated as a measured model score.

## Recorded evidence and media

Each run archives `request.json`, the complete task source and manifest,
`versions.json`, preserved installed-source hashes, GPU assignment proof, reset
checks, initial state, every ordinary action, measured state, native physics chunks,
three H.264 camera recordings and the final termination reason. The model also
preserves exact JPEG observations in `model-observations.jsonl.gz`, inference
chunks in `inference.jsonl`, `policy-actions.jsonl` and `policy-transcript.json`.
Privileged diagnostics are recorded locally for the independent scorer.

`setup.png` and the three `*-initial.png` images are captured at the declared
reset before any action. Final `demo.mp4` and `model_run.mp4` must be generated
only from the corresponding verified native recordings, preserve full duration,
and show the actual measured score. Use `prepare_submission_media.py` after both
runs have been audited. It verifies H.264, full frame counts and the 50 MB limit;
it does not upload, create a PR, or set the media ID.

```bash
python -B "$TASK/prepare_submission_media.py" --demo demo-released-new \
  --model dream-released-one --output "$OUT/submission-media"
```

Inspect the setup image and full videos before uploading. The media manifest
leaves visual review pending until a person or agent has actually inspected them.

Final artifact paths will be populated only when genuine recordings exist.
At present `demo.mp4` and `model_run.mp4` do not exist as qualified submission
artifacts. The original `task.json` media field remains null.
