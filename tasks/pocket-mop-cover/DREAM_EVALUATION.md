# Handle-held-v2 Dream evaluation

The user authorized a revised endpoint: finish holding only the handle after the
complete lift/translation/return/hold proof. Final unsupported placement and
resting-cloth settling are removed. Scene, reset, physical model, stock robot,
material/contact invalidation thresholds, 2,400-action budget and camera contract
are unchanged. The active backend is the already-qualified Kit physics.

The fresh R168 demonstration independently scores **1.0** in 1,865 actions
(93.25 s), with complete material/contact and exclusive-handle-support audits.
R169 passes the matching native no-network interface check. Fresh-process R170
replay independently scores **1.0**. The one genuine hosted R171 episode scores
**0.0** after 2,400 actions (120 s). It installs no pocket, and its complete rigid
audit invalidates right-arm self-intersections above 1 mm (maximum 12.813 mm).
Execution terminates at `action_budget`; scoring terminates as `invalid` with
`rigid_contact_audit`. The full material audit passes.

All 7,200 public RGB images, measured observations and ordinary model targets
pass the independent trace audit. R171 contains 32 camera-history preroll actions
and 2,368 model actions, with 50 returned chunks. The installed client records
99 request attempts, including its expired-request retries within this single
continuous episode; these are not repeated task resets. Actual records are kept.
No privileged feedback or scripted assistance reaches the policy.

Historical R123 remains 0.5 under released-v1; its retrospective screen is not
qualifying evidence. Original records and metadata are preserved. Local evidence
and anonymous media preparation are complete. Media ID M-G7NS83 is present in
task.json; PR submission remains separate.

## Execute and assess

First follow [REPRODUCTION.md](REPRODUCTION.md) for environment paths and gates.
The commands below record the completed evaluations. Existing run names cannot
be overwritten; choose fresh names for local reproduction. The hosted claim is
already consumed and blocks another paid episode. Do not erase it or automatically
repeat the hosted command.

Use the native installed environment and the stable scheduler ID for every
simulator. The complete launcher/child tree keeps the scheduler reservation.
Masked physics uses `cuda:0`, the physical renderer is selected explicitly, and
actual assignments are verified after warmup. Ordinary demonstration/interface/model
episodes use shared development admission with 12 GiB initial headroom; physical
parameter and performance comparisons use exclusive reservations. Existing jobs
remain untouched.

```bash
# First set TASK, OUT, SCHEDULER and exported paths as in REPRODUCTION.md.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
COMMON=(--experiment frame_robot --product-profile base_micro
  --physics-hz 480 --iterations 64 --seconds 120 --table-origin --snap-throat
  --robot-unlock --robot-reopen --robot-cover --cover-offset-x .02
  --frame-lift .4 --spacing .0025 --contact-updates 8 --contact-subiterations 2
  --ccd --reset-check --completion-mode handle-held --camera-width 320 --camera-height 176)

# Fresh continuous scripted episode from the declared reset.
"$SCHEDULER" launch --task pocket-mop-cover --mode dev --gpu auto --min-free-mib 12288 --wait 300 --detach -- \
  python -B "$TASK/launch.py" --name base-r168-handle-held-demo "${COMMON[@]}" --policy scripted
```

Wait for the returned scheduler job and verify result.json reports completed and
process.json `returncode: 0`. An admission timeout creates no demonstration. A native
error must be inspected; use a fresh name for any repaired run. Audit on CPU with
the immutable archived sources after native completion:

```bash
RUN="$OUT/base-r168-handle-held-demo"
python -B "$TASK/audit_episode_contacts.py" "$RUN"
python -B "$RUN/source/audit_cover_parallel.py" "$RUN" --workers 16
python -B "$RUN/source/score_full_run.py" "$RUN" --workers 8
python -B "$RUN/source/score_handle_held.py" "$RUN"
python -B "$RUN/source/verify_videos.py" "$RUN"
```

The phase-independent rigid auditor retains the original collision algorithm and
archives itself in each run's assessment-source directory. This fixes the older
frame-handling diagnostic's assumption that every recording contains scripted
release phases. Native recordings, launch archives and scoring rules stay unchanged.
For an exact repeat, use the saved assessment-source/audit_episode_contacts.py;
it verifies and loads the original run/source contact helpers.

The original assessment remains in benchmark-result.json. The active revised
result is handle-held-benchmark-result.json, with handle-held-outcomes.jsonl.
The latter reuses hash-bound original native observations and independently
measures exclusive support at every native substep. Complete physics invalidity
also invalidates the revised endpoint and preserves a zero model score.
Inspect recordings before treating a numerical pass as a demonstration.

```bash
# The installed DreamPolicy with a local no-network Session; no paid requests.
"$SCHEDULER" launch --task pocket-mop-cover --mode dev --gpu auto --min-free-mib 12288 --wait 300 --detach -- \
  python -B "$TASK/launch.py" --name base-r169-handle-held-interface "${COMMON[@]}" \
  --policy interface-check --policy-check-steps 64
# Wait for completion before this verification.
python -B "$OUT/base-r169-handle-held-interface/source/verify_videos.py" "$OUT/base-r169-handle-held-interface"

# Fresh-process replay of ordinary actions, with no recorded object-state writes.
"$SCHEDULER" launch --task pocket-mop-cover --mode dev --gpu auto --min-free-mib 12288 --wait 300 --detach -- \
  python -B "$TASK/launch.py" --name base-r170-handle-held-replay "${COMMON[@]}" \
  --policy scripted --replay-run base-r168-handle-held-demo
# Repeat the five audit/scoring/video commands with RUN=base-r170-handle-held-replay.

# Exactly one hosted episode after the matched demo and replay independently score 1.0.
"$SCHEDULER" launch --task pocket-mop-cover --mode dev --gpu auto --min-free-mib 12288 --wait 300 --detach -- \
  python -B "$TASK/launch.py" --name base-r171-handle-held-dream "${COMMON[@]}" --policy dream \
  --demo-run base-r168-handle-held-demo --replay-evidence base-r170-handle-held-replay \
  --interface-evidence base-r169-handle-held-interface
# Repeat the five audit/scoring/video commands for the actual model recording.
python -B "$TASK/audit_model_trace.py" "$OUT/base-r171-handle-held-dream"
```

## Model boundary and artifacts

Dream receives the declared instruction; `top_cam`, `left_cam`, `right_cam` RGB at 320×176;
and measured left six joints + gripper, then right six joints + gripper. It receives no
object/contact/score/controller data. Grippers are normalized: 0 closed, 1 open.
Ordinary joint/gripper targets execute at 20 Hz with 24 native substeps/action at 480 Hz.
Stock clamping and physical gains/limits remain unchanged. The 32 camera-history
preroll actions count against 2,400. Physics pauses during inference. The model
branch starts before scripted movement and has no scripted assistance.

The local interface check uses the actual installed inspect-robots-dream policy
with a mock Session; DreamPolicy.dry_run still makes hosted calls and is forbidden
for local validation. The model is dreamlabs/dream-1-beta. Credentials are already
configured, and R171 completed with genuine hosted action chunks. hosted-dream-attempt.json is
created exclusively before the first attempt and prevents automatic paid retries,
including after errors or zero. Do not erase that claim to improve a result.

The CPU-only `audit_model_trace.py` decodes every input image, checks each measured
state against the native trace, and verifies every executed target against the
returned model chunk and ordinary stock clamping. It makes no hosted calls.

Each run archives its source, source hashes, package versions, seed, request,
reset checks, GPU proof, every native physics state, every executed action,
measured state, initial setup.png and three camera videos. Model runs also save
model-observations.jsonl.gz, inference.jsonl, policy-actions.jsonl and
policy-transcript.json. Termination and final measured scores remain explicit.

```bash
python -B "$TASK/prepare_submission_media.py" --demo base-r168-handle-held-demo \
  --model base-r171-handle-held-dream --output "$OUT/submission-media-handle-held-v2"
```

The exporter requires genuine audited recordings under matching task rules. It
writes anonymous full-duration H.264/yuv420p demo.mp4 and model_run.mp4 below 50 MB
each, setup.png and a hash manifest. The final exported media were decoded and
visually inspected; media-manifest.json records the review. Re-export to a new
output directory if needed; the exporter never overwrites evidence. The uploaded media ID is M-G7NS83; this command itself only exports local media.
Original released-v1 instructions are archived in DREAM_EVALUATION_RELEASED_V1.md.
