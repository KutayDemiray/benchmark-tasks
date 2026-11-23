# Scoring review revision

The reviewer requested a practical real-trial scoring method and an explicit
boundary between observable task outcomes and simulation-only strain/contact
checks. [REAL_TRIAL_SCORING.md](REAL_TRIAL_SCORING.md) now provides that procedure,
with a reference diagram and [worksheet](REAL_TRIAL_WORKSHEET.md).

## Response to the review

The primary metadata rubric now describes observable pocket seating, intact
cloth/seams, handle-only support, whole-arm clearance and the timed retention
motion. A real judge uses continuous video, calibrated ruler/angle references
and before/after close-up inspection. Temporary textile stretching is not a
real-trial penalty; new visible tears, holes, separated seams, detached pockets
or frame/latch damage are disqualifying. No real-trial force sensor, mesh strain,
penetration depth, microscopic contact gap or physics-substep data is required.

The procedure explicitly classifies the 20% native strain limit, 1 mm numerical
penetration/seam-gap limits, 2 mm contact/separation proxies and substep auditing
as **simulation-only**. The measurable outcome thresholds and motion remain
40 mm insertion over the central 80 mm, 3-degree flatness, 100 mm lift/translation,
10 mm return, 20 mm other-arm clearance and two-second holds. Unknown evidence
is unscorable instead of being guessed to pass or recorded as a model failure.

The simulation audit/scorer implementation and return values are unchanged.
Only four Python module docstrings were clarified. Simulation-only tests still
reject invalid numerical evidence; no validity bypass was added. The previously
recorded demonstration and replay remain 1.0, and the genuine Dream-1 run remains
0.0. Original archives and uploaded media M-G7NS83 are preserved. The real-trial
protocol has **not** been tested on hardware; the assessment below is a review
of existing simulation evidence, not a claim of a physical trial.

## Walkthrough of the existing evidence

Media times use the unchanged `demo.mp4`/`model_run.mp4` exports. In a native
20 Hz trace, step N corresponds to video frame time N/20; the associated native
control interval completes at (N+1)/20 seconds. This matters for the two-second
hold: frames at 90.90 and 92.85 alone span only 1.95 seconds, while the recorded
control intervals complete the hold at 92.90 seconds.

| Evidence | Timed walkthrough | What establishes the numeric result |
|---|---|---|
| R168 scripted demo | Proof begins at 86.95 s; maximum displacement at 89.25 s; return hold starts 90.90 s and completes 92.90 s; recording ends 93.25 s. | Archived outcomes, native support/material/contact audits and the original 1.0 assessment. |
| R170 fresh-process replay | Proof begins at 87.00 s; maximum displacement at 89.25 s; return hold 90.90–92.90 s; recording ends 93.25 s. | Its own archived outcomes and complete native audits, independently assessed 1.0. |
| R171 genuine Dream-1 episode | Final montage frame at 119.95 s shows the frame and cover still separate. The episode ends at 120 s. | Native per-step diagnostics never seat a pocket; the original rigid-contact audit invalidates the run, preserving 0.0. |

The following are **native simulation measurements**, not ruler measurements
read from the exported RGB video. They summarize the successful proof through
the end of each recording:

| Native quantity | R168 demo | R170 replay |
|---|---|---|
| Minimum cloth height | 272.613 mm | 262.681 mm |
| Maximum horizontal translation | 124.751 mm | 124.815 mm |
| Final return error | 0.534 mm | 0.476 mm |
| Return hold | 2.000 s | 2.000 s |
| Maximum wing angle | 1.882 degrees | 1.895 degrees |
| Minimum other-arm clearance | 77.945 mm | 77.791 mm |

Both native traces certify pocket enclosure/orientation throughout the proof
and right-hand-only support. Their observation fields encode passed 40 mm/80%
containment thresholds; those two fields are not continuous measured insertion
depths or width estimates. The extra native non-handle margins remain in the
machine-readable report and are not presented as real-trial measurements.

Sampled RGB frames at demo times 86.95 and 92.85 show the raised assembly and the
holding hand at the handle; the model's final frame shows the separate objects.
The supplied montage is not a calibrated physical-trial record. It does not
independently resolve all depth references, layer relationships, angle/clearance
bounds or the pre/post inspection of both cloth faces and every seam. Filling a
physical worksheet solely from those images would leave required items unknown.
The native simulation evidence supports the original scores, but does not supply
a completed hardware worksheet or validate camera placement for a future trial.

[scoring-revision-evidence.json](scoring-revision-evidence.json) records the source
artifact hashes, exact native quantities, timestamps, visual observations and
limits. No original assessment, native state, action, inference trace or media
file was rewritten. The worksheet's separate worked cases are explicitly
hypothetical examples, including damage, support interruption and missing data.

## Verification and reproduction scope

[scoring-revision-validation.json](scoring-revision-validation.json) is the current
source/metadata validation manifest for this review revision. It records the CPU
suite, parsed sources, documentation links, field lengths, source hashes and
module-docstring-only AST comparison. The original cleanup validation manifest
remains a historical checkpoint; do not mistake its old task/source hashes for
this revision's hashes.

Reproduction continues to use the existing scheduler, stock YAM, installed
runtime and archived-run sources. This revision makes no simulator or hosted
model call. A future newly executed episode must obey the existing source and
configuration qualification checks; old manifests must never be rewritten to
match a new package. The review revision goes onto the same task PR.
