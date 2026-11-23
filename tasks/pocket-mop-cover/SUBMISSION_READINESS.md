# Scoring review revision readiness

The existing task PR received scoring feedback about real-trial measurability.
This revision adds [REAL_TRIAL_SCORING.md](REAL_TRIAL_SCORING.md), its worksheet
and measurement diagram, and separates observable outcome conditions from
simulation-only strain/contact validity checks. See
[SCORING_REVISION.md](SCORING_REVISION.md) for the response, timed evidence and
verification scope. Physical hardware validation has not been performed.

The completed handle-held-v2 simulation results remain demo R168 **1.0**, fresh
replay R170 **1.0**, native offline interface R169 pass and hosted Dream-1 R171
**0.0**, with the model contact invalidity preserved. These are simulation
results; no real-trial score has been inferred from an incomplete video-only
worksheet. The approved behavior ends holding only the handle.

The anonymous demo/model videos and setup image retain media ID **M-G7NS83**.
No new upload, simulator episode or hosted attempt is needed for this
scoring-procedure revision. Original sources, recordings and results remain
immutable. Current source and metadata checks are recorded in
[scoring-revision-validation.json](scoring-revision-validation.json).

Submit the revision as a new signed-off commit on the existing PR branch and
await reviewer reassessment. If pushing requires terminal authentication, use
the exact task worktree and branch. Do not create a duplicate PR or force-push.
