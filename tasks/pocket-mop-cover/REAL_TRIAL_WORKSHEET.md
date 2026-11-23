# Real-trial scoring worksheet

Use with [REAL_TRIAL_SCORING.md](REAL_TRIAL_SCORING.md). This is a human judging
record, not a request for privileged simulator data. Never leave an unmeasured
item implicitly marked as passing.

## Trial and evidence

- Trial ID, evaluator, date:
- Domain: **physical trial** / **simulation illustration**:
- Continuous video file(s), frame rate and synchronization reference:
- Pre-trial and post-trial close-up files:
- Calibration images, ruler divisions and measurement uncertainty:
- First-command timestamp / declared finish timestamp / elapsed time:
- Executed target intervals, if available (maximum 2,400 at 20 Hz):
- Inspector handling begins at:

For every criterion use **PASS**, **FAIL** or **UNKNOWN**. Record measured lower
and upper bounds rather than spurious precision. Time references are relative
to the continuous recording. Labels such as "native trace" must identify any
simulation-only evidence; they cannot be used as physical video measurements.

## Trial-wide checks

| Check | PASS / FAIL / UNKNOWN | Evidence and observation |
|---|---|---|
| Declared starting layout; no added attachments/restraints | | |
| All necessary views, timing and calibration available | | |
| Execution within 120 s / 2,400 intervals | | |
| No unauthorized human intervention during execution | | |
| No new fabric tear, through-hole or detached pocket flap | | |
| No newly separated sewn seam | | |
| Frame and latch intact | | |

A proved trial-wide violation gives 0.0. If required evidence is missing and no
such disqualification is established, report unscorable with no numeric score.

## Final pocket holds

| Measurement | Left pocket | Right pocket |
|---|---|---|
| Tip between base and flap; mouth faces centre | | |
| Insertion-depth bounds (mm), minimum 40 | | |
| Central continuous width coverage (mm), minimum 80 | | |
| No textile grasp during the hold | | |
| Final hold start/end timestamps; duration at least 2 s | | |
| Close-up/frame evidence, including layer relationship | | |
| Pocket qualifies: PASS / FAIL / UNKNOWN | | |

The final pocket holds must reach the declared finish. A previously seated
pocket that slips before finishing does not earn retained partial credit.

## Retention proof

| Measurement or condition | Result / bounds | Timestamp or evidence |
|---|---|---|
| Both pockets qualified before motion proof and retained throughout | | |
| Same holding hand throughout: left / right | | |
| Visible handle grip; no cloth pinch or other assembly support | | |
| Entire other arm minimum clearance (mm), at least 20 | | |
| Maximum wing angle relative to head plane (degrees), at most 3 | | |
| Frame remains locked | | |
| Minimum height of any cloth over tabletop (mm), at least 100 | | |
| Initial lifted head-centre reference position | | |
| Maximum horizontal displacement (mm), at least 100 | | |
| Maximum return error during hold (mm), at most 10 | | |
| Return-hold start/end timestamps, at least 2 s | | |
| No interruption of seating, lift, grip, clearance or support | | |
| Finishes holding only the handle | | |

For a known interruption, record the timestamp and evaluate only a subsequent
complete sequence; do not splice together successful fragments.

## Decision

- Status: **scored** / **unscorable**:
- Trial-wide disqualification and evidence, if any:
- Qualifying pockets at finish: 0 / 1 / 2:
- Full retention proof: PASS / FAIL / UNKNOWN:
- Numeric score: 0.0 / 0.25 / 0.5 / 1.0 / **null if unscorable**:
- Unknown items or reasons for failure:
- Reviewer and optional independent review:

Decision order: a proved trial-wide violation gives 0.0. Otherwise all evidence
needed to distinguish the applicable outcome must be known; use null if it is
not. With sufficient evidence, award 1.0 for the full proof, else 0.25 times the
number of qualifying final pockets. If a known prerequisite fails (for example,
only one pocket qualifies), retention cannot pass and its downstream movement
measurements may be marked not applicable, rather than unknown.

## Worked examples (hypothetical, not physical trials)

All examples assume adequate evidence and an intact, unmodified setup unless
explicitly stated otherwise. Distances are ruler/video measurements, not native
state values.

| Example | Observation at finish | Decision |
|---|---|---|
| Neither installed | Both tips lie outside their pockets; no qualifying final hold. | 0.0 |
| One installed | Left depth 44–46 mm across 82–84 mm for 2.2 s; right tip remains outside. | 0.25 |
| Installation only | Both pockets qualify for 2.2 s; no lift/translation sequence occurs. | 0.5 |
| Complete | Both pockets qualify; locked wings 1–2 degrees; cloth height 115–120 mm; translation 110–115 mm; return error 3–6 mm held 2.2 s; other arm 30–35 mm clear; only the same hand on the handle. | 1.0 |
| Damage | Otherwise complete, but comparison photos establish a new through-tear or separated sewn seam. | 0.0 |
| Interrupted support | Both final pockets qualify, but a forearm braces the head midway through the proof and no fresh complete sequence follows. | 0.5 |
| Recovery | After the forearm contact, both pockets remain seated and a fresh complete handle-only sequence finishes within budget. | 1.0 |
| Pocket slips | Left remains qualified; right slips out before the declared finish. | 0.25 |
| Uncertain depth | A required pocket-depth measurement spans 39–42 mm and no other view resolves it. | Unscorable; null |
| Occluded support | Potential support is hidden in every view, so full retention cannot be distinguished from incomplete retention. | Unscorable; null |
| Wrong layer | A loose cloth fold hides the 40 mm mark but side views show the tip is outside the sewn pocket. | That pocket fails; score remaining known conditions. |
| Ordinary deformation | Cloth stretches or wrinkles visibly, but inspection shows no new damage and all task conditions pass. | 1.0; no inferred 20% real-cloth strain penalty. |
