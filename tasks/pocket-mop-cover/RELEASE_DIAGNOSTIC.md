# Passive release diagnostic — unresolved implementation blocker

> Historical investigation: results and pending-work statements below refer to
> earlier prototypes, released-v1, or the deferred SDK path. The active submitted
> endpoint and completed evidence are in [README.md](README.md). These diagnostics
> do not change the submitted physics or scoring.

The current full task remains at audited score **0.5**. This bounded batch localizes the release failure; it does not supply a fix or a successful demonstration.

## Comparison

All four runs initialize the same recorded held assembly at R123 control step 2092 and the same clear robot poses from step 2215. Initial velocities are zero. Every run applies the same 143 constant ordinary 14D commands over 7.15 seconds. Frame geometry and material constants are unchanged; the bare control omits the cloth.

| Run | Cover | Physics / cloth iterations | Final head inclination | Material / rigid audit |
|---|---|---|---|---|
| `release-r124j-hands-clear` | Yes | 480 Hz / 64 | 143.03545° | Pass / pass |
| `release-r124k-hands-clear-960hz` | Yes | 960 Hz / 64 | 143.18747° | Pass / pass |
| `release-r124l-hands-clear-128iters` | Yes | 480 Hz / 128 | 143.05682° | Pass / pass |
| `release-r124m-hands-clear-without-cover` | No | 480 Hz / 64 | 0.00008° | Not applicable / pass |

The complete covered-run measurements certify that the hands remain clear throughout all recorded native intervals. In the baseline their minimum clearance is 91.65 mm. Both pockets stay seated and the wings remain flat relative to the head, but the assembly overturns, fails table placement, and fails the final two-second settling condition. All three camera streams from all four comparisons decode completely.

## What the evidence establishes

Withdrawal motion is not necessary to trigger this failure. Doubling the timestep resolution or the cloth solver iterations does not resolve it. Omitting the cover prevents overturning in this comparison. The earlier explanation that a fingertip hooks under a rail is withdrawn: the covered rail rises from below when the hand stops restraining it.

This does **not** identify a unique engine, material or geometry bug. The tests start with a cold contact solver, and removing the cloth changes support height. Passing the strain and penetration bounds does not establish energy-consistent passive motion. No material parameter, robot gain or scoring threshold has been adjusted to make these recordings pass.

## Evidence and reproduction

The evidence-root file `release-passive-batch-2026-11-22.json` records exact values, assertions, limitations and SHA-256 digests. Each run contains frozen source, requests, native state chunks, video checks, rigid audit and (when covered) material audit plus `release-assessment.json`. The side-by-side clip is `progress-videos/2026-11-22-hands-clear-comparison.mp4`; it is explicitly marked as a cold-start diagnostic.

Starting from the documented full-task command in README, use a fresh name, `--seconds 10`, `--release-probe-run base-r123-fixed-arm-opening`, and `--release-probe-hands-clear`. The baseline is 480 Hz / 64 iterations. For the two numerical comparisons add `--release-probe-refine-contact` and respectively select 960 Hz or 128 iterations. For the matched bare control add `--release-probe-without-cover` to the baseline. The source recording and matching isolated qualification evidence must be present.

Run each frozen `source/audit_frame_robot.py` and `source/verify_videos.py`. For covered runs, also run frozen `source/audit_cover_parallel.py --workers 8`, then the task-local `assess_release_probe.py RUN --workers 8`. That assessor reuses all-native clearance and settling measurements, reports diagnostic conditions only, and never issues a benchmark score.

## Subsequent correction tests

See [CONTACT_SOLVER.md](CONTACT_SOLVER.md). A constant contact-position correction
limit prevents overturn in two diagnostic variants. The frame-only variant fails
fresh isolated latch retention and rigid penetration and is rejected for full use.
The cloth-only variant R125b passes complete material and rigid contact audits,
seated pockets, flat wings, level table placement and all-step hand clearance,
but fails final native/finite-difference settling (0.13998/0.08855 m/s). The
candidate is not qualified by that diagnostic. Fresh isolated insertion passes, but stock-robot grasp fails, including a fully
closed grip. No revised gate was issued. Only passing affected gates can permit
a new full-task run from reset. See [PASSIVE_CONTACT.md](PASSIVE_CONTACT.md) for
the subsequent reduced reproducer and native error checks. No scoring threshold is relaxed.

Energy diagnosis is implemented in diagnose_release_energy.py. It combines
native rigid masses, centers of mass, inertia, motion and clip spring properties
with explicitly approximate cloth membrane/bending energies. It is not an exact
native energy balance or direct contact-work measurement. The current candidate
remains unready for model evaluation or submission.
