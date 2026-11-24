# Evaluation and submission provenance

| Episode | Run ID | Score | Actions | Termination | Seed |
| --- | --- | --- | --- | --- | --- |
| Scripted demonstration | `7a988ed297694ed1b9792ec7b06897cc` | 1.0 | 3,018 | success | 0 |
| Genuine Dream-1 | `49b19db344a5489b9249701a7b3e43c5` | 0.0 | 3,600 | step_limit | 0 |
| Offline native interface | `74bbb0f4c2464a779b1917397b65fd1b` | diagnostic 0.0 | 128 | step_limit | 0 |

The demonstration is continuous from the declared folded reset. Deployment,
regrasping, working-jaw cycle, inner-jaw extraction, carry, release and final
clearance all occur within the declared 3,600-action budget. Independent
frozen-source rescoring matches every action; submission preparation reran that
CPU assessment without modifying original evidence. Key score milestones are
2,211 (deployment), 2,657 (extraction), 2,722 (carry) and 3,018 (full release).

Qualified inner-jaw pinching is continuous from actions 2,560–2,898. The nut
travels 253.5 mm horizontally with zero detected robot/handle support. During
carry its bottom remains at least 32.9 mm above the recess rim, loaded folds stay
within 0.001 degrees of their stops and sampled nut/jaw penetration is <=0.138 mm.
Final clearance is >=21.4 mm with no tool/robot contact and drift <0.001 mm.
The complete one-second sampled video sequence, full-resolution retrieval frames
and measured contact/pose evidence were reviewed. Small nut size and wrist
occlusion limit video-only assessment; this is not hardware validation.

Dream-1 genuinely returned and executed actions through the installed integration.
It repeatedly approached the folded tool but achieved no deployment, extraction,
carry or release credit. Independent scoring remains 0.0. Its 79 action requests
produced 75 chunks and four `DreamSessionExpired` errors after observation ticks
224, 320, 512 and 608. The installed integration recovered across five provider
sessions within one robot episode, with no reset, scripted intervention or paid
success-seeking repeat. Native joint bounds clipped 433 executed rows; both raw
and applied targets are retained. The first 32 measured-pose hold actions count
against the 3,600-action limit. Provider recording was disabled.

The SDK warned that a human input acknowledgment was absent. No human
acknowledgment was fabricated. Agent inspection of simulated inputs and the
passed native offline interface checks are documented separately.

All three runs verified physical GPU 1, masked physics tensors `cuda:0`, CUDA
UUID `cc9ca88f-3544-a5fd-4676-362e0d647c69`, renderer [1], multi-GPU rendering off.
Native stock robot and passive tool audits passed. Each episode has frozen source,
source hashes, configuration, seed, measured trace/actions and complete video.
The model additionally preserves its exact input JPEGs, inference replies,
latencies, errors and termination. These large raw records remain outside Git;
`evidence.json` includes content hashes and compact measured results.

## Media

| File | Bytes | Frames | Duration |
| --- | --- | --- | --- |
| `demo.mp4` | 3,498,354 | 3,019 | 150.95 s |
| `model_run.mp4` | 4,841,411 | 3,601 | 180.05 s |
| `setup.png` | 96,934 | one image | n/a |

The MP4s are H.264/yuv420p, 960 x 176, 20 fps. Each consists of three 320 x 176
RGB views. The screenshot is the valid folded reset. No names, usernames, paths,
logos, score overlays or identifying audio are present. There is no temporal
cutting; reset plus every action accounts for the exact frame count. Exports
were reused by hash without another encode or upload during initial preparation.
The repository uploader assigned **M-JTNCGU** to this task; the review revision
retains that upload and the same three media hashes.

## Submission-only changes

The original development folder was named `folding-multitool-nut-retrieval`;
the submitted folder/branch uses `folding-multitool`. The scheduler ID was already
`folding-multitool` and is unchanged. Documentation now uses portable installation
roots and describes actual completed outcomes. Caches, recordings, inference
inputs and development logs are excluded from Git.

Exactly two runner lines changed: the scheduler import path now accepts
`ROBOCURVE_SCHEDULER_ROOT` (defaulting to the same home-relative installation),
and the default output directory is home-relative. `--out` still takes an
explicit directory. Native initialization, GPU verification, reset, stepping,
termination, actions and recording code are unchanged. `preflight.py` checks
legacy frozen-source hashes and permits only those two exact text substitutions;
any other runner difference or evaluation-boundary change requires new native
interface evidence. CPU tests reject altered device, timing and controller code.
The preflight CLI default evidence directory is also home-relative.

At initial submission all 16 shared episode files and the controllers, scorer,
geometry and native adapter were byte-identical to the completed runs. The
23-test CPU suite and independent rescores passed during that packaging step.
The scoring revision below changes only `scorer.py` among those episode files.
`validate_submission.py` prints field lengths, verifies the explicitly revised
contract, unchanged runtime sources and optional media hashes, then can refresh
`package-validation.json`.

## Review revision: retrieval without an empty-jaw prerequisite

The original scorer required a prior empty-jaw cycle and tip-down dwell before
recognizing retrieval. Revision `retrieval-without-empty-cycle-v2` removes those
prerequisites from extraction/carry/full credit. The cycle remains optional 0.25
credit and the separate load diagnostic is unchanged. Functional handle loading
does not prescribe which arm holds which handle; original fingers must load both
handles while the opposing inner jaws alone retain the nut. Folded reset,
deployed fold stops, extraction/carry history, release, clearance and stability
checks remain. Physics, reset, robot, controllers, instruction, cameras, action
budget and recordings are unchanged.

`rescored-demo.json` and `rescored-model.json` are separate revised assessments of
the immutable original measured traces. Every frozen file and the complete trace
were verified against the recorded hashes before assessment. Results remain
**demo 1.0 at 3,018 actions; Dream-1 0.0 at 3,600 actions**, with no per-step score
or snapshot differences. `evidence.json` retains the original episode contract
and run records alongside the revised contract and report hashes. No new
simulation, model attempt or media upload was performed for this revision.

The 30-test CPU suite includes full outer-scorer retrieval without an empty cycle,
alternative handle/finger assignments, rejected head/nut support and direct
placement, drop-then-push, insufficient final clearance/stability, and rejection
of non-scoring source changes or altered evidence. Synthetic measurements test
score logic; they do not prove an additional feasible grasp. Preflight permits
the old native interface evidence only through the verified scoring-only
compatibility path. The original runner and model I/O checks still apply.

The review's hardware note remains unresolved by simulation: verify actual jaw
fit, hinge behavior and robot grasps in the specified recess before hardware
trials. Neither the original evaluation nor this scoring revision establishes
those specimen measurements.

The earlier short-nut reset failures and separate load diagnostic remain original
historical investigations outside Git. Load-scene modules are retained because
current scorer/controller imports depend on them; their diagnostic status does
not change the submission score.
