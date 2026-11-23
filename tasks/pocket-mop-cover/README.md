# Fit a pocket mop cover onto its folding frame

Install both sewn pockets, open and physically lock the frame flat, then prove
retention while holding only the handle: lift all cloth 100 mm above the table,
move the head 100 mm, return within 10 mm and hold for 2 s. Finish holding the
handle. The other entire arm stays 20 mm clear; all non-handle frame parts stay
2 mm clear of every robot link during the retention proof.

The active task is **handle-held-v2**, with stock bimanual YAM and native Kit
PhysX. [task.json](task.json) contains the submitted instruction, initial layout,
partial credit, physical invalidation limits and 2,400-action budget at 20 Hz.
[HANDLE_HELD_V2.md](HANDLE_HELD_V2.md) explains the exact endpoint.

## Completed evaluation

| Recording | Independently assessed result | Actions / simulated time |
|---|---|---|
| Fresh continuous scripted demo R168 | **1.0**, complete physics and handle-support audits pass | 1,865 / 93.25 s |
| Fresh-process ordinary-action replay R170 | **1.0**, independently audited | 1,865 / 93.25 s |
| Native offline Dream interface check R169 | Pass, zero hosted calls | 64 / 3.2 s |
| Genuine hosted Dream-1 episode R171 | **0.0**, contact-invalid | 2,400 / 120 s |

Dream installed no pocket. Its complete rigid-contact audit rejects right-arm
self-intersections (12.813 mm maximum against the unchanged 1 mm limit); its
material audit passes. Execution ended normally at the action budget; independent
assessment ended as `invalid`. There was one hosted episode, with no rerun to
improve the score.

Dream receives only the instruction, `top_cam`, `left_cam`, `right_cam` RGB at
320×176 and measured 14D left/right joint and gripper state. Ordinary 14D targets
execute at 20 Hz; privileged object/contact/controller data remain outside the
policy interface. All 7,200 input images and 2,400 executed states/targets pass
the offline trace audit. Script, replay and model use the same scene, reset,
physics, instruction, scorer and recording implementation. Seed is 0.

The recorded simulation wall times were approximately 23.6 minutes for the demo,
29.5 minutes for replay and 57.4 minutes for Dream, excluding separate CPU audits.
These shared-GPU observations are not controlled performance measurements.

Anonymous, full-duration H.264/yuv420p `demo.mp4`, `model_run.mp4` and `setup.png`
are prepared under `$OUT/submission-media-handle-held-v2`. Both videos are below
50 MB. The uploaded media ID is **M-G7NS83**, preserved in `task.json`.
[Evaluation evidence](evaluation-evidence.json) records original artifact hashes,
run identifiers, scores, termination and media hashes. Raw recordings, native
state chunks, inference logs and immutable launch sources remain in the external
output archive; they are not embedded in Git. The PR remains a separate action.

## Reproduce and review

Start with [REPRODUCTION.md](REPRODUCTION.md): environment paths, recorded
versions, CPU checks and exact isolated qualification commands. It works from
this task folder in any checkout; output ownership and shared GPU coordination
are explicit. [DREAM_EVALUATION.md](DREAM_EVALUATION.md) gives the full-task,
replay, interface, hosted-evaluation, independent scoring and media commands.
All simulator launches use scheduler ID `pocket-mop-cover`.

The cleanup CPU suite passes 158 tests from a separate source and output directory.
It exercises geometry, false-success cases, scoring, recording, the offline
Dream interface and scheduler failure checks. See
[PACKAGE_VALIDATION.md](PACKAGE_VALIDATION.md) for the cleanup's checks and limits.
Original qualification provenance is in
[qualification-evidence.json](qualification-evidence.json); gates can be
regenerated with the commands in the reproduction guide.

## Fidelity and limitations

[PRODUCT.md](PRODUCT.md) distinguishes published product facts from estimates.
The modeled CombiSpeed Pro / BaseMicro dimensions, internal mechanism and textile
response are not hardware-calibrated. [parameters.json](parameters.json) retains
legacy research entries; `--product-profile base_micro` and the full-cover
implementation select the evaluated model.

There are no cloth attachments, proximity seating triggers, runtime object-pose
writes or laboratory fixtures in the full task. The script uses privileged
native poses to choose ordinary robot targets. Handle support is inferred from
geometry, sustained assembly lift and exclusion of other load paths; contact
forces are not instrumented. Pocket enclosure is measured at 20 Hz, with
material/contact/support audits covering every native substep.

The original released-v1 endpoint required unsupported final table placement.
Its best complete R123 score remains 0.5 and its passive overturning issue is
unresolved. It is outside the approved held endpoint. SDK migration is deferred.
The fresh R168 demonstration establishes feasibility for the active task; a
retrospective score or success-rate claim is not used in its place.

## Source organization

- Execution: `launch.py`, `frame_robot.py`, `robot_session.py`, `full_cover.py`,
  `frame_native.py`, `frame_asset.py`, `handle_held_finish.py`.
- Public model interface: `evaluation_contract.py`, `dream_evaluation.py`.
- Assessment and media: `audit_episode_contacts.py`, `audit_cover_parallel.py`,
  `score_full_run.py`, `score_handle_held.py`, `audit_model_trace.py`,
  `verify_videos.py`, `prepare_submission_media.py`.
- Paths and GPU coordination: `task_paths.py`, `gpu_coordination.py`,
  `runtime_checks.py`; configuration is documented in the reproduction guide.
- Isolated qualification: `qualify.py`, `qualify_mechanism.py` and their audit tools.
- Historical investigations: `DEVELOPMENT_NOTES.md`, `PRODUCT_HISTORY.md`,
  `RELEASE_DIAGNOSTIC.md`, `CONTACT_SOLVER.md`, `PASSIVE_CONTACT.md`,
  `SDK_COMPARISON.md`, `DREAM_EVALUATION_RELEASED_V1.md` and related diagnostic
  scripts. They remain available because several are shared imports or test
  dependencies. Their pending-work statements do not describe the active endpoint.
