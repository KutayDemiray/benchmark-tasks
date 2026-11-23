# Development evidence and experiment history

> Historical investigation: results and pending-work statements below refer to
> earlier prototypes, released-v1, or the deferred SDK path. The active submitted
> endpoint and completed evidence are in [README.md](README.md). These diagnostics
> do not change the submitted physics or scoring.

The [README](README.md) contains the current status and reproduction commands.
This file preserves detailed diagnostic methods and the sequence of controller
revisions. Statements about a candidate apply to the named revision; later
results supersede them. Failed configurations and their original recordings
remain in the output directory.


The sewn pocket is one connected deformable surface with an actual open mouth.
Self-collision and tool contact are enabled. No fabric-to-robot, support or frame
attachments exist. The original reinforced experiment retains separate native
material/seam recordings; whole-pair contact filtering and detached sewing are
explicit negative controls, never product-valid configurations.

The insertion stage is a visible, force-limited laboratory instrument.
It starts outside the coupon and physically pushes a tapered end through the
collapsed mouth. It is not a final-task fixture. The grasp diagnostic uses
the stock robot and cameras with a disclosed 250 × 100 × 120 mm support block
for transfer to the table (400 mm long in the alternative bench-return test).
The frame test uses a disclosed force-limited vertical press and a constant
gear relation; the rocker is passive contact geometry. The press cannot appear
in the final robot task.

Recorded native states support independent measurement replay. This is not
independent physics replay. Checks include all material edges, triangle/tool
and support penetration, seam consistency, real pocket enclosure, orientation,
and both native and finite-difference speed in the last two released seconds.
The fixed limits are 20% edge strain, 1 mm penetration/seam gap, and 0.03 m/s
released speed. Passing only a subset does not qualify a run.

Stock finger meshes are convex-hull approximations that PhysX simplifies during
cooking. The maintained grasp recorder exports the actual native hulls before
actions; audits verify their source scene, versions and geometry hashes. Earlier
authored-hull reports are retained as conservative outer-bound diagnostics.
Crossed cloth triangles are checked using an explicit separating translation
along either triangle normal: its length bounds the minimum separating distance
from above. The old, larger normal-span diagnostic remains in reports. Both
checks retain the original 1 mm penetration limit and leave simulated geometry
and state unchanged. Deep-crossing and rigid-transform regression tests are
included. `refine_grasp_audit.py` preserves and refines an older full audit with
explicit per-pair separation witnesses and hashes of all recorded chunks.

`audit_cover_parallel.py` runs the same material checker on disjoint consecutive
recording chunks with a bounded CPU process pool. Every partition includes the
preceding native state for velocity differences. Merging requires complete
coverage, matching source hashes and the full final two-second window. The
serial `audit_full_cover.py` remains available. A 1,440-step failing native
interval produced identical strain, crossing, collider and final-speed results
with the frozen reference checker and the partition implementation.

Robot runs save every physics step in compressed chunks, 20 Hz actions and
measured joints, and overhead/left/right camera MP4s. The auditor decodes every
video and checks frame count. Diagnostic footage is not a full-task success
demonstration. The full-cover native scene, offline outcome adapter, complete
state-reset check and fresh-process action replay are implemented. Complete
robot/frame/cloth resets have passed twice with zero observed state differences;
fresh-process action replay still awaits validation. Run r107 reaches both live
pocket-enclosure checks, but stops at a 3.87-degree fold before locking or lifting.
Its peak recorded strain is 17.2%; both complete recorded contact audits pass. The stopped controller remains diagnostic and has no benchmark score. A
successful from-reset controller remains unresolved. The full scene starts with a flat head and a separate cover
on the table; the robot physically operates the pedal and grasps an existing
rail. There is no laboratory support, guide, driven frame joint or cloth attachment.

Native recordings use a bounded, single-worker compression queue. Each chunk
owns a copy of every recorded array and becomes visible through an atomic
rename only after writing finishes. Episode completion waits for all writes;
a write failure propagates instead of producing a complete recording certificate.
`recording-io.json` records the completed chunk sequence and compression times.
The writer passes reused-buffer and write-failure checks and an exact round trip
of 309 MB of recorded native arrays. Its first live run, r105, closes all 52 chunks
with 24,960 physics steps and matching field dimensions; all three camera videos
decode 1,040 frames. The controller stops on alignment, so these recording checks
do not establish a task success. Earlier experiments keep their synchronous recorder.

The current controller first tilts the held handle inward to a planned 70-degree
gimbal angle, leaving room below its physical 85-degree stop. This is an ordinary
wrist motion; the passive joint is never commanded. The previous alignment
attempt reached that stop and stalled despite sixteen corrections. The revised
wrist path passes a native-chain geometry check and command reach/speed checks.
In r105 the handle stays away from its stop and the tips finish within 2.7 mm of
level, but the wrist correction reaches its old 40 mm cap before convergence.

The controller tracks markers fixed to the held wing and handle, initialized
at their contact grasp poses, then issues ordinary joint-position actions.
It reads native body poses for this scripted feedback and logs robot forward
kinematics separately. This accounts for rotation inside the contact grasps:
r106 recorded about 1.5 degrees at the rail and 3 degrees at the handle while
its wrist-only correction continued to struggle. No object or passive joint
receives a position command.
Actuator gains and force limits are unchanged. It checks that both noses are
level before descending, keeps the planned lower noses 1 mm above the table
during insertion, and supplies closing pressure through ordinary stock-arm
commands. Handle-only pressure stalled near 3.8 degrees in r107/r108. Moving
that handle farther upright in r109 cleared the wrist stop but still failed to
latch: the handle rotated under pressure and the wings stopped at 4.17 degrees.
The r110 candidate kept the working insertion path and used the free left
hand to press the existing Bridge/RailR top surface, clear of the release pedal.
Closed fingertips approach 1 mm above that surface, advance in 5 mm increments
up to 50 mm of compliant command overtravel, and stop when measured wing fold
reaches 1 degree. The hand then withdraws upward and moves 200 mm sideways clear of the lift path. The 11-second bounded maneuver uses
native bridge position only for horizontal tracking; it never moves the frame
or fabric through state writes. All 70 commanded approach poses pass a CPU
frame/right-arm collision screen from recorded native geometry. Run r110 physically closes the latch after 45 mm of command overtravel, with
both pockets retained and wings within .002 degrees. Its vertical-only hand
withdrawal then obstructs the lift: recorded tongue/clip contacts occur against
that hand, and the frame later reopens. The current candidate adds the 200 mm
sideways retreat. All 30 retreat poses pass the static contact screen, and the
parked arm is clear of all 5,280 recorded physics poses from the obstructed lift.
r111 validates that free lift: both pockets stay installed through about 275 mm
of all-cloth clearance, 124.5 mm of head translation and a valid return/hold.
Both complete material and rigid audits pass, with 18.58% peak strain and no
penetration/seam violations. Final release still fails: the falling handle tips
the head onto one edge, leaving its bridge 26.68 mm above the table, and the
cloth exceeds the settling-speed limit. Its independently assessed benchmark
score is 0.5, with no numerical-invalidity reasons. The original swept-box
measurement reported only 8.365 mm final hand clearance. Refined conservative
projections of the actual convex shapes certify 20.035 mm at the final interval
and remove several false support warnings. Retention still resets during
withdrawal, and the head placement and cloth settling still fail. The original
and refined outcome traces are both retained; both score 0.5.

The r113 upright-placement feedback failed to level the head: its release guard
stopped at 101.85 seconds with 58.09 degrees of head tilt. Its evidence is retained
as an incomplete attempt, without a full-episode validity certificate.

The r114 support press also failed: its shifted set-down partly unseated the
right pocket, and pressure on one side did not stop the head rolling as the
handle moved. The guard stopped before release at 103.25 seconds, with 61.77
degrees of head tilt. That stopped episode has no complete validity certificate.

The r115 controller returned to a vertical set-down at the retained head location.
The left hand pinches the exposed rigid Bridge/Crossbar and UprightR between
ordinary parallel jaws. Both wrists then level the head while preserving their
measured relative contact poses. With the crossbar still held, the right hand
lowers the shaft along the head's long axis toward passive gimbal angles [0, 82]
degrees. Up to three bounded wrist corrections preserve the measured contact
grasp. A guard requires head tilt at most 5 degrees, bridge height between -1 and
10 mm, and suitable measured handle angles before opening. The right gripper
opens to 50%, withdraws horizontally 50 mm, then raises and retreats. Native r115
successfully held and leveled the bridge and released the handle, but the left
hand then caught Bridge/RailR during its upward withdrawal and flipped the
assembly. Its rigid-body audit passes. Its material audit fails at nine native
substeps during latch closure: cloth enters Left/InnerRailR by up to 1.452 mm,
above the unchanged 1 mm limit. Its independently assessed score is 0.0, invalid.

The r116 attempt uses smaller final latch presses and a partial-opening sideways
hand exit. It completes 111.8 seconds / 2,236 actions, but the head still tips onto
its edge during withdrawal. The rigid audit passes. The complete material audit
finds eight substeps with cloth/Left/InnerRailR penetration during closure, up to
1.383 mm; its independently assessed score is 0.0, invalid. Its two-second final
native/finite-difference cloth speed maxima are 0.355 / 0.198 m/s.

The r117 candidate presses the existing crossbar nearer the head's center, at
Bridge-local [30, 25, 35] mm, with the closed gripper turned 90 degrees to clear
the handle connector. Approach and initial withdrawal use 60 mm vertical
clearance to stay within reach. Near the latch's observed snap threshold, the
press still advances in 2.5 mm / 0.3 s increments instead of 5 mm / 0.4 s.
For final release, the right hand keeps holding the handle while the left clears
the bridge. The left wrist targets its measured pose, opens to 50% while moving
6 mm sideways and 8 mm up, then widens to 90% while shifting 20 mm along its
closing axis to center over the wider rail. It raises and retreats before a
second measured-posture guard allows the right hand to release. A five-second
settling hold follows. These are ordinary stock robot motions; no object
state, constraint or passive joint is commanded.
That trial stops at the closure guard after 83.15 seconds / 1,663 actions: both
pockets are enclosed, but the wings remain 3.48 degrees folded. The selected
cloth/Left/InnerRailR check passes all 2,400 native substeps from 74–79 seconds;
this is a focused diagnostic, not a complete audit or a successful closure.
Both subsequent full-recording audits also pass all 39,912 saved physics steps,
with peak cloth strain of 16.47% and no material/contact violations. Because the
controller stopped, those reports certify only the recorded diagnostic interval;
they do not qualify a benchmark score.
That stopped trial does not reach the new final hand withdrawal.

The r118 candidate retains the central contact and extends its bounded wrist
overtravel from 50 to 65 mm, using the same 2.5 mm / 0.3 s increments near closure.
It still stops when measured fold reaches 1 degree. The additional 1.8 seconds
come from shortening the post-withdrawal dwell from 2 to 0.2 seconds; the press
already holds for 0.5 seconds and withdraws over 3 seconds. The worst-case
2,399-action bound, scoring thresholds and stock actuator limits are unchanged.
R118 closes the latch at 55 mm of overtravel. Both pockets survive the lift and
return, and the wider left-hand exit leaves the head at 0.93 degrees of tilt and
4.79 mm height. The right-hand release then flips the assembly. Selected native
contacts show the right fingers near the shaft during opening and a finger
contacting the right frame end during horizontal withdrawal. The recorded right
wrist starts 7.54 mm below its loaded command and rises as the grasp opens.
Both complete physics audits pass all 53,184 native steps, with peak strain of
17.79% and no material/contact violations. The final head is tipped and the
final two-second native/finite-difference cloth speed maxima are 0.476 / 0.242 m/s,
above the settling limit. The independently assessed score is 0.5 with no
numerical-invalidity reasons. Retention qualifies at 92.6 seconds and persists
through the left-hand exit; right-finger/frame contact resets it at 104.5 seconds.
Immediately before the right release, native/finite-difference cloth speeds are
0.0184 / 0.0162 m/s, already below the settling limit. A focused
check of Left/InnerRailR also passes all 3,840 substeps from 74–82 seconds,
including the latch closure; that focused check alone is not a full certificate.

R119 retains that assembly and left-hand exit. The right hand now opens to 60%
while targeting its measured wrist pose, removing the holding-load command bias.
It then backs 70 mm along the full negative approach axis before its final
retreat. This path includes upward clearance instead of the previous horizontal
escape. The durations and 2,399-action ceiling are unchanged. The focused geometry
screen starts from r118's actual pre-release state and includes zero, half and
full recorded link deflection. An additional model places the released handle at
gimbal angles [-32.3,85] degrees while keeping the head fixed. These are planning
approximations, not force or dynamic feasibility certificates.
R119 completes 111.4 seconds / 2,228 actions and passes both complete physics
audits, with peak strain of 17.75%. Its left hand again clears with the head flat
(0.25 degrees, 4.20 mm height), but right-hand opening and withdrawal still tip
the assembly. Final native/finite-difference cloth speed maxima are 0.183 / 0.104
m/s. Its independently assessed final score is 0.25, with a maximum of 0.5 and
no numerical-invalidity reasons. Selected snapshots do not reproduce r118's
right-finger/frame-end contact, so the changed path alone does not explain or
resolve the unsupported release dynamics.

R120 keeps the left crossbar grasp closed while releasing and withdrawing the
right hand. The right opens to 50% while backing 10 mm along its measured negative
approach axis over 0.3 seconds, then continues to 70 mm total retreat while opening
to 70% over 0.65 seconds. It completes the usual one-second retreat. A measured
guard checks that the supported head remains within 5 degrees and between -1 and
10 mm above the table; the already released handle may rotate freely. The left
then performs the same wider, centered exit, followed by five seconds of settling.
The full path remains bounded by 2,399 actions. A stationary 60% right opening was
rejected by gripper-to-gripper geometry checks before simulation. The accepted
path passes whole-sequence and recorded-deflection screens, including the left
exit with the right hand already withdrawn. Native feasibility remains pending.
`preflight_supported_release.py` screens reach and rigid geometry, including an
optional handle-at-stop approximation. It cannot certify native contact or
release stability. Full independent physics and outcome audits remain required.
`preflight_bridge_exit.py` additionally starts at a recorded pre-withdrawal
native state and screens zero, half and all of its recorded per-link arm
deflection. Frame and cloth remain fixed in this planning approximation; it is
not a dynamic certificate. Select the right hand with `--side right
--source-phase withdraw_release_support`; `--released-handle` enables the
explicit handle-rotation approximation above.
A failed closure stops the retention attempt. The complete nominal command path
is bounded by 2,399 actions (119.95 seconds), including up to ten measured
alignment corrections before and after carry. Descent still requires marker
errors below 1 mm and .01 radians, plus a separate level-head check.
Position corrections are bounded by 45 mm; the previous 40 mm bound prevented
r105 from removing its final 1.25 mm loaded right-wrist error. These are ordinary
position targets, with unchanged stock actuator gains, forces and speed limits.
`preflight_loaded_cover.py` additionally screens ordinary command reach using
recorded arm deflection as a constant-load approximation. It flags a possible
late-opening reach limit in the original layout; moving the initial cover 20 mm
along table X passes that check. The optional `--cover-offset-x .02` records this
layout variant explicitly and is checked during action replay. The current
candidate uses this variant with body-marker feedback. More conservative
constant body-deflection screens flag a late reach limit here, but r107 completes
the real opening motion. Moving the rail grasp farther outward did
not improve the corresponding reach checks.
The earlier 86.5-second combined run took
about 27.5 wall-clock minutes at the qualified 480 Hz settings.

Read `$OUT/STATUS.md`, `HANDOFF.md`,
`product-gate.json` and the per-run audits for current results. Failed
configurations are retained. A recorded seed alone is not a meaningful
robustness variation. Runtime is recorded in each run's `process.json`.

The `--finite-seam` experiment uses an estimated 2 mm sewn-edge construction
and flat panel rest geometry. It is a new geometric hypothesis and requires
fresh cycle and robot-grasp qualification. Material moduli, friction, damping
and gate limits are unchanged. Historical zero-height seam evidence is not
transferred to this variant. `qualify.py` requires matching material, geometry
and solver settings before it can combine evidence.

`score_logic.py` supplies the deterministic outcome state machine; its unit
tests cover partial credit, interrupted retention, numerical invalidity and
reset. `outcome_native.py` converts native recordings into observations without
reading controller phases or intended actions. Both material and rigid contact
audits must cover the complete run and match hashes before `score_full_run.py`
can award any nonzero score. A separately recomputed native edge-strain
counterexample can reject a run with score zero immediately; logged maxima
alone cannot establish that counterexample, and its absence never certifies a
pass. The failing r101 run has such a hash-bound witness. Pocket enclosure is sampled at 20 Hz; numerical limits and
conservative swept clearance bounds cover every recorded physics substep.
Rigid clearance uses actual convex vertices at every native substep, with
separating projections when axis-aligned bounds are inconclusive. The projection
extrema include every pose, including intermediate contact; the unit-axis gap is
a conservative lower bound on Euclidean separation. Capsule proxies remain outer
hulls. This avoids false support caused by rotated local bounding boxes without
changing the 2 mm support or 20 mm final-clearance thresholds. Cloth clearance
still uses every swept triangle box. Counterexamples cover overlapping rotated
solids and contact that occurs only during an intermediate rotation. See
`$OUT/clearance-measurement-revision.json`. The refined r111 measurements preserve
all non-clearance observations exactly and match independent serial checks.
Possible robot support prevents credit for free retention. Final release requires
the head over the tabletop and its bridge between -1 and 10 mm above it, both
cloth speed measurements at most .03 m/s, and hands at least 20 mm clear for 2 s.
`outcomes.jsonl` contains assessed observations and scores; the original action
record remains unchanged. `benchmark-result.json` is the separate outcome.
`--diagnostic-error-run` lets the material and rigid auditors inspect all saved
steps from a stopped controller. Such reports distinguish recorded contact
checks from qualification: `complete` / `numerical_checks_passed` or
`contact_checks_passed` remain false, even when their recorded geometric checks
pass. They cannot supply a benchmark success certificate.

Material auditing supports up to 32 independent CPU workers when capacity is
available; the example uses four to limit resource use. Outcome scoring supports
eight workers for geometry followed by one ordered scoring reduction. Its
single-worker mode remains the reference. On the complete r111 recording, the
32-worker audit has identical numerical findings and input hashes across all
50,568 physics steps. The eight-worker scorer produces byte-identical outcome
and benchmark-result files across all 2,107 control observations. Chunk-boundary
velocity differences include the preceding native state. See
`$OUT/parallel-backend-validation.json`; the original reference artifacts remain
unchanged. These are processing-speed changes, not different numerical checks
or scoring thresholds.

`mechanism-gate.json` qualifies only the rigid laboratory experiment. Both
isolated gates pass. The stopped r107–r109 two-pocket insertion recordings also
pass complete recorded material and rigid contact checks. r110 additionally
closes the passive latch, but its pressing hand obstructs the later lift; the
frame reopens and loses the installation. Its complete audits find subsequent
cloth/table and rigid contact violations, giving a benchmark score of zero.
r111 passes both complete physics audits and the geometric retention sequence;
its final placement and settling fail. r112 reaches a nearly vertical shaft
but releases with the head still tipped 40 degrees; withdrawal flips the
assembly and a native material-strain witness invalidates the episode. Its
rigid audit also finds one 1.206 mm finger/shaft penetration. r113 rejects release
with the head still tipped. r114's support press also fails. r115's crossbar
pinch successfully supports handle repositioning, but its left withdrawal flips
the assembly and its latch closure has a verified cloth penetration. r116 tests
a slower final latch press and a measured-pose sideways hand exit, but still
fails both closure penetration and final placement. r117's central crossbar press
reduces head roll but stops before latch closure, leaving its new final hand exit
untested. r118's extended press closes the latch and its left hand clears without
tipping, but the right hand release flips the head. R119 tests removal of the
right wrist's holding bias and withdrawal along its full approach axis.
That attempt still tips the head. R120 instead retains the left crossbar support
until the right hand is clear, then uses the wider left-hand exit.
A valid final release and full-score execution remain unverified.


## R120 result and R121 correction

R120 closes the latch at 57.5 mm commanded overtravel, retains both pockets
through about 273 mm of all-cloth lift, translation and return, and clears the
right hand while the left crossbar pinch holds. The head rolls 7.18 degrees and
its bridge reaches 10.19 mm above the table. The guard stops before left release
at 103.95 seconds / 2,079 actions. Both complete recorded material and rigid
checks pass all 49,896 native substeps; peak strain is 17.11%. Because the
controller stopped, these are diagnostic reports, not benchmark certificates.
The focused closure check also passes all 3,840 substeps from 74 to 82 seconds.

R121 preserves that sequence and adds a 0.75-second left-wrist correction plus
a 0.25-second hold after the right hand clears. The target preserves the measured
left/bridge contact pose, accounts for the current command-to-measured wrist
bias, retains head XY/yaw, and aims for a level head at 5 mm table clearance.
Correction is limited to 20 mm and 15 degrees; a tilted state outside 15 degrees
or -1 to 20 mm height stops before correction. The original final guard of at
most 5 degrees and -1 to 10 mm remains after correction. The left then performs
the same opening/centering exit. The final dwell changes from five to four
seconds, preserving the 2,399-action bound and unchanged two-second scoring hold.

All 83 CPU unit checks and the complete reach/speed preflight pass. The whole
release geometry screen passes. The recorded-state screen starts at R120's
actual 103.95-second tilted head. It assumes the measured pinch holds during
leveling, keeps the cloth/opposite arm fixed, and models the arm's recorded
deflection relaxing during opening to full, half or zero residual. It also
preserves the measured closed-jaw aperture until the opening command exceeds it.
All three continuous models pass, both with the recorded handle pose and an
additional released-handle rotation hypothesis. Minimum wing and cloth gaps
are 6.20 and 24.53 mm. Earlier constant-unloaded models flag Connector/Cross
overlap even at the initial loaded grasp; those failed planning reports remain
archived. These alternatives describe uncertainty in a planning approximation.
They do not certify friction, changing load, contact forces or native success.
No physical parameters, contact tolerances, robot gains or scoring rules change.


## R121 result and R122 hand exit

R121 completes 111.1 seconds / 2,222 actions. It closes the latch at 57.5 mm
commanded overtravel, retains both pockets through about 264 mm of all-cloth
lift and the translation/return check, and clears the right hand with the head
at 0.66 degrees / 4.54 mm height. The left-hand correction reaches 0.021 degrees
and 4.01 mm. Final left opening and withdrawal then flip the head. Peak live
strain is 19.52%. Both complete physics audits pass all 53,328 native steps,
with no material/contact violations. Final native/finite-difference speed maxima
are 0.1785/0.1182 m/s. The independent final score is 0.5, with no
numerical-invalidity reasons.

Measured 20 Hz contact snapshots show the left negative-X fingertip near
Bridge/RailR from 105.15 seconds and overlapping it by conservative bounds of
up to 0.83 mm. Later samples show Left/InnerRailR and Right/End contacts before
the head overturns. The head starts rising during the initial opening while
the wrist is commanded 8 mm upward. These are geometric contact candidates,
not contact-force measurements. `diagnose_release_contacts.py` records the
selected native poses, conservative bounds and input hashes; complete native
audits remain separate.

R122 keeps the assembly, supported right release and head-leveling correction.
The left hand now targets its measured pose and opens to 25% for 0.4 seconds
without any commanded lift or sideways movement. That aperture clears the
14 mm crossbar. It then slides 40 mm along the horizontal outward gripper axis
for 0.4 seconds, holding that aperture and height. Once outside the rail, it
opens to 90% while rising 80 mm over 0.75 seconds, then retreats for 0.6 seconds.
This preserves the 2.15-second exit and 2,399-action full-path bound. Physics,
actuator gains and scoring are unchanged.

All 83 CPU tests and the full reach/speed and whole-release geometry checks
pass. Additional screens start at R121's actual leveled 104.95-second pose,
using continuous opening/unloading models and the measured held aperture.
All three residual models pass with both recorded and alternative released
handle poses. Minimum predicted wing and cloth gaps are 11.19 and 25.69 mm.
These are static/kinematic planning approximations; they do not establish
native release stability.


## R122 outcome and R123 fixed-arm opening

R122 completed 111.1 seconds, 2,222 actions and 53,328 native steps. Both complete physics audits pass (peak material strain 17.368%; maximum rigid penetration bound 0.999989 mm). The head levels to 0.138 degrees at 4.105 mm but flips after the final left exit. Its nominally stationary Cartesian opening removed an approximately 4.5 mm loaded command bias and raised the hand before releasing. R123 changes only the ordinary release controller: preserve all arm joint targets bitwise during the eight opening actions, then use the existing 40 mm horizontal slide and lift.

Recorded R122-state screens pass with continuously unloaded zero/half/full link residuals, minimum wing clearance 7.609 mm, cloth clearance 21.609 mm and final hand clearance 146.507 mm. The alternative HandleX=-32.3 degree posture fails (up to 3.952 mm predicted finger/shaft overlap); all tested jaw apertures 0.18–0.35 also fail that alternative. The actual released handle is +32.3 degrees. A new measured-posture guard rejects a handle leaning toward the positive-Y exit (HandleX below zero, with floating-point tolerance) or HandleY outside 78–85.1 degrees. The failed geometry reports are retained. These screens are planning approximations, not native collision or stability certificates. Command preflight remains 2,399 worst-case actions.


## Pocket orientation comparison after R122

The new read-only diagnose_pocket_orientation.py reproduces the existing orientation Boolean at every selected control boundary. It changes neither scoring nor simulation. Over 82–95 seconds, R121 has minimum right-pocket projection ratio0.804411 (0.004411 above the0.8 threshold); R122 reaches0.765696 (0.034304 below). Both retain the containment samples. R122 has94 orientation-failing control boundaries during82–95s and never completes the functional retention history; R121 completes retention at94.0s. R122 official final/maxscore is0.5 with no numerical invalidity reasons. Reports and input hashes are saved in each run/pocket-orientation-diagnostic.json. The measured ratio describes the whole upper panel, including its deformed side regions; it is not a direct measurement of pocket loss. No scoring change is made.
