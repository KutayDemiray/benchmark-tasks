# Parameter and implementation revisions

All revisions are uncalibrated research prototypes. No hardware force curve has
been measured. Exact sources for every attempt are frozen in that run's bundle.
The entries below preserve the assessment made at each revision. The criteria
review at the end supersedes earlier claims that strict trajectory agreement
or measured hardware calibration are universal acceptance prerequisites.

- **p01 smoke / smoke-debug:** native PhysX initialized, but the tensor frontend
  was on CPU despite GPU dynamics. Startup failed the device assertion. Enabled
  GPU tensor readback suppression; added explicit exception artifacts because
  SimulationApp shutdown masked an exception and returned exit code zero.
- **p01 rigid:** the early scalar seating/retention checks returned true, but
  inspection found a fixture servo instability: up to 18.4 m/s during pressing
  and over 100 N during settling. **Invalid run, not evidence of working snap
  physics.** At 960 Hz, explicit gains of 10000 N/m and 5 Ns/m were too stiff
  for a 1.5 g test body. These were fixture gains, never robot gains.
- **p02:** reduced only the diagnostic servo to 1000 N/m and 1 Ns/m and bounded
  all its loads to 25 N. Raised the socket ring top from 3 to 6 mm and detents
  from 3 to 5 mm so the male head can seat without going through the lower fabric.
  Widened radial clearance: ring center radius 5.1 mm and radial width 1.8 mm.
  The p01 ring interfered with outward spring travel. These geometry repairs
  change the estimate and require fresh validation; they are not calibration.
  Diagnostic seating thresholds moved with the revised physical seating plane.
- **p02 gate suite:** press, no-press, 4 mm offset, 35° tilt and torque-only
  unguided peel were run through five complete scene reconstructions in one
  process. Axial release and retention worked; negatives did not seat. Torque
  alone did not reopen the stud up to 0.05 Nm. Inspection disproved the suspected
  housing-stop cause: measured socket radial positions remained below 3.083 mm,
  while the housing permits 3.5 mm. No spring or housing change was made in
  response to this failure.
- **p03 pinch:** GPU rigid-body kinematic target setter is unimplemented in this
  installed backend. Replaced the diagnostic pads with force-limited, passive
  prismatic drives in an articulation (5 N per closing joint, 10 N lift).
  This is a bench fixture, not a YAM robot alteration. Both normal-friction and
  zero-friction coupons failed to lift with a 0.2-second closing stroke.
- **p04 diagnostics:** moved the pad lower edge above the snap cap to exclude a
  hardware grasp, shortened closing to 0.03 seconds to catch the freely sagging
  cloth, and added measured fixture joint/link states. Added combined 1 N pull
  plus torque-ramp peel and reverse-side diagnostics. Snap spring, material and
  scoring thresholds are unchanged. Minimal native Kit experience avoids
  unrelated policy-example startup and remote asset queries; installed packages
  are unchanged. The first minimal experience failed for a missing layers
  extension, then was corrected locally.
- **p05:** the p04 jaw trace showed the commanded 0.3 mm gap produced almost
  no compression (drive error about a micrometre), so friction could not carry
  the snap mass. Commanded ordinary full closing (zero target gap) with the same
  finite drive gains and force limits. p04's earlier-closing hypothesis was not
  supported by the trace: the sheet remained in reach and the jaws moved, but
  lacked normal preload. Added full physics-rate cloth observations because
  80 Hz samples could miss strain peaks. Tightened diagnostic release scoring:
  separation without prior captured retention is not a successful release.
  p04 already exceeded the original 15% strain threshold (17.06% observed),
  and both torque-only and combined 1 N pull/torque peel remained unvalidated.
- **p06 fixture correction:** full-rate replay localized the 17.10% strain spike
  to the 44 mm upper guide hard stop, after the fastener had already separated.
  Maximum upper strain through the first 12 mm separation was 6.38%. The old
  force ramp kept accelerating a 1.5 g released part and drove it into the stop
  at about 7.6 m/s. That is an invalid withdrawal fixture, not evidence that the
  snap itself damages cloth. Changed only the diagnostic withdrawal to a
  5 mm/s displacement ramp using the existing 1000 N/m, 1 Ns/m actuator.
  Retained all old failures. No material, snap spring or scoring limit changed.
  The separate peel failure persists even with the cloth removed. Original
  open-loop replay also failed its 1 mm trajectory tolerance (3.50 mm maximum),
  despite byte-identical initial USD; final diagnostic validity was zero in both.
- **Numerical torque investigation:** the 1920 Hz refinement also remained
  captured under combined peel. A free 1.5 g cap responds to 1e-7 Nm torque,
  confirming the actuator path. Its native inertia is about 1.245e-8 kg m².
  At 960 Hz, a 0.05 Nm load would change free angular speed by about 4184 rad/s
  in one step, well above PhysX's default 100 rad/s pre-solver clamp. This can
  discard load before contact resolution. Added explicit snap-only numerical
  limit and free-torque diagnostic options to test this hypothesis. Defaults,
  robot parameters, material values, spring constants and scoring are unchanged.
  The installed USD schema stores the limit in degrees/s (default 5729.58).
  See the [official PhysX rigid-body API](https://nvidia-omniverse.github.io/PhysX/physx/5.7.0/_api_build/classPxRigidBody.html#_CPPv4N5physx11PxRigidBody21setMaxAngularVelocityE6PxReal)
  for the pre-solver clamp behavior. A one-step 0.05 Nm / 0.1 ms experiment
  confirms it: expected 401.686 rad/s, default measured exactly 100, extended
  snap-only limit measured 401.684 rad/s. Earlier peel tests did not faithfully
  apply the requested load. With a 10000 rad/s numerical ceiling, the rigid
  combined-pull test reopens, but the continuing force/torque accelerates the
  free cap to 384 m/s. This is another invalid post-release fixture, not an
  acceptable cloth release demonstration. Added a bounded, damped angular
  displacement actuator with the same axial withdrawal protocol for a fresh
  test. The 10000 rad/s option remains diagnostic, not a validated default.
  The scorer now also explicitly rejects exceeding the unchanged torque limit.
- **Replay distinction:** even the corrected axial fixture's raw-wrench replay
  diverged by 36.95 mm and exceeded 15% cloth strain. The original run computes
  a bounded feedback wrench from a displacement command; replaying that output
  without feedback is a different actuator. Added a per-step actuator-command
  stream and separate `commands` / `forces` replay modes. Raw-force failures
  remain failures and are not overwritten. Command replay must independently
  meet the original 1 mm position tolerance before being called reproducible.
- **p07:** controlled combined withdrawal/rotation opened the cloth-mounted
  snap with 2.20% peak strain, 0.28 mm maximum rivet error and 0.00206 Nm peak
  requested torque. This is positive fixture evidence, not robot success.
  Adopted the tested 10000 rad/s numerical ceiling for the snap body only:
  it exceeds the analytically required one-step torque impulse; it is not a
  material, retention-force, scoring or robot-limit change. All cases need fresh
  validation with this numerical setting and recorded actuator commands.
  Moved the pinch pads 1 mm upward for the new clearance control: the earlier
  closest cap-pad gap was only 0.211 mm against 0.200 mm summed rigid contact
  offsets. The earlier lift/slip evidence remains provisional until a fresh
  test confirms cloth-only support with a larger clearance margin. Parameter
  records now directly drive all documented fixture dimensions and gains.
  Independent guided command replay passed (0.0251 mm cap error, 0.4074 mm
  maximum upper-cloth coordinate error). Unguided combined-peel command replay
  did not: 2.109 mm cap error, 38.80 degrees orientation error and 15.081 mm
  upper-cloth error, despite identical initial USD. Both runs seated, retained
  and reopened, but that agreement does not pass the frozen replay tolerances.
  First cap error above 1 mm occurred at 3.7635 seconds during peeling.
  No state tolerance or material parameter was relaxed after the failure.
  User confirmed no hardware measurements are available; calibration remains
  unresolved. Full shirt, hanger and robot stages have not been implemented.
- **p07 refinement:** controlled peel and independent command replay at 1920 Hz
  both opened with intact cloth, but replay still failed: 2.034 mm cap error,
  47.90 degrees orientation error, 23.876 mm upper-cloth error and 1.563 mm lower
  cloth error. Initial USD matched. Halving the timestep did not establish
  reproducibility. Three separate sensitivity probes (500–502) closed/held/opened
  with 2.33–3.14% peak strain, but they do not satisfy full-task robustness
  acceptance. The task remains at the isolated gate with all failures retained.

Early native startup also used Isaac Sim's default portable cache and telemetry
configuration. Later launch revisions explicitly place the portable root,
logs and caches under the owned output folder and disable telemetry consent and
deferred startup. No task media, model call, upload script or submission ran.

## Documentation revision: criteria review, 2026-11-20 UTC

Rechecked the public RoboCurve AGENTS.md and README.md. Neither specifies 5/5
nominal, 9/10 robustness or 1 mm / 5 degree trajectory agreement. The trial
counts came from the original project brief; trajectory limits were local
diagnostics. Their results and numerical values are unchanged. The full-score
demonstration, physical credibility, reproducible setup and valid scorer remain
necessary; a demonstration alone does not establish all acceptance criteria.

Both p07 peel command replays closed, held and released, despite strict
trajectory failures. This supports further isolated testing, not a passed
functional gate or a completed garment task. Missing hardware measurements
limit calibration claims without automatically stopping provisional engineering.
The next priority is ordinary YAM single-snap grasp/alignment/pressing and
assessment of whether replay divergence changes relevant mechanics or scores.
Full shirt integration still follows the isolated functional gate.

Updated README, task metadata, output status/handoff/validation interpretation
and packaging descriptions. No physics, scoring, actions, tolerances or archived
results changed, and no new simulation was run. See the output folder's
`criteria-and-assessment.md` and `criteria-review-2026-11-20/manifest.json`.

## Fidelity-scope clarification, 2026-11-20 UTC

The user clarified that a sufficiently close garment model is appropriate.
The target is a representative adult snap-front shirt, with Wrangler retained
as a real-product reference for scale and fabric. Exact pattern/stitching,
proprietary snap geometry and brand-specific calibration are not task goals.
Missing exact-SKU measurements alone are not a validation-gate failure.
Alignment, insertion, pressing, retention, release, cloth deformation and slip
still need credible evidence; wrong compatible pairings must remain possible.
No physical parameters, scorer, tolerances or archived outcomes changed.
Current provenance: output `fidelity-review-2026-11-20/manifest.json`.

## p08: approved implementation plan

One complete 300-second demonstration plus one successful fresh-process rerun of the frozen feedback controller replaces the historical 5/5 and 9/10 targets. Native control is 10 Hz with unchanged 14D actions, gains and limits. The hanger starts freely on a rail and the open shirt on the table. Added a task-local Arena/YAM runner and movable-frame hardware builder. Initial work tests a free cloth coupon over a passive raised-table edge; it is not a snap-cycle or full-task success. Existing mechanical thresholds are unchanged.

- p08 robot grasp 01: measured >=28.9 mm lift and <=2.96% strain before opening. Opening above the table edge dropped the coupon 0.95 m onto the floor; 25.57% peak strain invalidates the entire diagnostic. Added carry/lower onto the passive table before opening; material, gains and strain threshold unchanged. Moved the task overhead camera to see the raised coupon; wrist mounts remain stock.

- p08 controlled release: ordinary robot grasp/lift/carry/place/open passes its isolated diagnostic at seed 100 (6.117% strain, 0.113 mm rivet error). Zero-friction control does not lift (2.754% strain, 0.628 mm rivet error). Both independent archived-source rescores match; six camera videos fully decode. These do not close a snap or pass the full gate.
- Added freely moving socket/upper/lower coupon press probes, explicit initial alignment disclosure, native robot contact-force logging and measured-state press scoring. First press attempt failed before motion because a tensor contact-filter wildcard matched multiple rigid bodies; changed to exact installed YAM link paths. No physical or scoring tolerance was loosened.

- Corrected press scoring to use measured contact history and continuous released seating independently of controller phase names. Physical thresholds unchanged. p08-free-pair-press-02 retains its original false-negative result and a separate correction review.
- Fresh p08 press confirmation: episode 00 passes; episode 01 retracts without full seating and the process receives SIGTERM before its final result (exit 143). Its interrupted logs remain a failed/incomplete confirmation, not 2/2 success. Initial snap poses match exactly across reset. Added bounded feedback pressing: continue 0.25 mm/command until four consecutive seated observations, or the press deadline, while retaining native gains and the 25 N measured rigid-contact cutoff.
- Added an unvalidated press_lift diagnostic with offset flexible tabs, native socket-to-robot contact reporting, conservative lower-cloth/socket clearance from both robot hulls, per-physics-step retention checks, initial measured reset arrays, and four direct-support shortcut tests. No garment stage or full-task credit added.

- **p08 combined lift failure:** p08-pair-retention-lift presses but does not
  grasp its upper tab; neither coupon rises. At 22.0 s the candidate cloth
  nodes lie at the end of the fingers, unlike the earlier successful cap-only
  cloth pinch. The robot then carries empty jaws and disturbs the supported
  pair. This run fails retention validation. Added an early missed-grasp abort
  before carrying. A separate `--grasp-mode bunch` rotates the jaws 90 degrees
  about their approach axis and closes gradually from full opening to gather
  the vertical tab from its sides. Physics and scoring limits are unchanged.
  GPU state reads are batched with explicit path/order/content assertions;
  observations are still recorded at every 960 Hz physics step.

- **p08 offline clearance precision:** the first combined-lift audit rebuilt
  native float32 robot poses as float64 JSON lists, producing at most
  1.9031e-8 m clearance differences. Reconstructing the source float32 pose
  arrays reduces maximum difference to 1.3878e-17 m, within the unchanged
  1e-10 m audit tolerance. Future clearance calculations explicitly use float64
  on both online and offline paths, with a JSON-roundtrip regression check.
  The physical 1 mm support-exclusion threshold and failed-run score are unchanged.

- **p08 bunch failure / side-pinch prototype:** p08-pair-retention-bunch also
  fails to capture and lift, terminating early after 245 controls; peak strain
  3.589%, rivet error 0.422 mm, rigid contact 9.785 N. The tab deflects out of the
  closing jaws. New experimental side pinch uses a reachable wrist roll and a
  local free-edge observation, then retracts from the edge before lifting.
  Its physical support is a 4 mm rigid plate on rear posts, explicitly selected
  by --support-thickness .004 to provide finger access below the tab. Original
  40 mm support tests remain reproducible. Snap geometry, material parameters,
  robot gains and scoring limits are unchanged. Release maintains housing
  clearance above the plate; any resulting impact still faces the 15% strain
  and 1 mm rivet limits. CPU mesh-vertex screens are preliminary, not a full
  collision certificate or successful grasp evidence.

- **p08 upper-hardware shortcut check:** coupled lift scoring now also requires
  the male hardware's enclosing 9 mm sphere to stay more than 1 mm clear of
  both robots during the hold. Direct upper-metal grasp can prove retention
  but cannot prove a fabric grasp. A dedicated negative test covers this.
  For older frozen runs, aggregate review recomputes this supplemental exclusion
  from archived link/hardware poses without modifying their original results.
  The active side-pinch run predates this additional recorded field; its result
  must receive the supplemental check before any gate promotion.

- **p08 short side-pinch failure / longer-swatch access test:** the side run
  completed with upper_tab_grasp_failed, 256 controls, 1.677% peak strain,
  0.445 mm rivet error and 8.371 N rigid contact. At 20.5 s the selected patch
  lay 19–23 mm from the jaw centre along the closing axis, outside the open
  gap; it was pushed away. This is a missed grasp, not demonstrated slip.
  The next explicit fixture uses a 120 x 60 mm upper swatch, preserving its
  right edge and local snap rivet; lower swatch remains 40 x 60 mm. The larger
  flexible overhang makes a free end accessible below the 4 mm plate. The
  side_long controller targets the observed patch in XYZ, then moves out and
  up before lifting to avoid dropping the pair from the table edge. Area/mass
  change with physical swatch dimensions; thickness, areal density, elastic
  laws, snap properties and scoring limits stay fixed. This tests accessible
  garment-scale fabric; it is not shirt integration or a material calibration.

  CPU screening rejects a horizontal approach to the lower free end because it
  hits a stock wrist limit. A 20-degree downward pitch about the closing axis
  makes the complete nominal path reachable (under 0.08 mm IK error). Closing
  direction and all joint limits/gains remain unchanged. This kinematic check
  is covered by a regression test and is not physical grasp evidence.

- **p08 longer-swatch press failure / explicit hover start:** the long-tab run
  fails before grasping: gravity creates approximately 4 mm male/socket lateral
  error before contact and the subsequent press separates them further. It
  stops with single_snap_did_not_seat after 181 controls; peak strain 13.216%,
  rivet error 0.502 mm, rigid normal contact 15.940 N. Limits are unchanged.
  New --press-start hover is an explicit isolated reset pose: the empty, closed
  left gripper starts at TCP (0.30, 0.12, 0.935) m above the open pair; right arm
  remains stock READY. Contact approach takes 0.6 s and its XY target follows
  the midpoint of the measured halves to apply a physical alignment correction.
  Actual force/contact and uncommanded seating checks still apply. This changes
  only task-local initial joints and ordinary actions; gains, limits, hardware,
  material laws and scoring are untouched. It is not a stock-READY reach demo
  or a full shirt result. Effective initial joints are archived in setup.json.

- **p08 hover start closes, wide plate obstructs wrist:** the hover-start run
  seats and remains closed after withdrawal, but the cloth grasp fails. It ends
  at 195 controls with 4.470% peak strain, 0.429 mm rivet error and 7.401 N rigid
  normal contact. At 14.5/15.3 s the actual TCP is ~10 mm off target; native
  collision-mesh vertices of the commanded gripper and link5 penetrate the
  350 mm-wide plate by up to 1.97 mm. A 60 mm-wide flat support beam removes
  those planned overlaps in the vertex screen. --support-width .06 records this
  access revision; a nearby post at x=0.385 supports the beam under pressing.
  No guide, object attraction, material, snap, robot gain or score relaxation.
  This is a physical bench-access change, not full garment construction.
  Released-pair scoring now also checks both fabrics and hardware clear of both
  robots, plus cap centres inside the actual support dimensions with cap-radius
  margins. Open fingers alone cannot prove release. New negative tests cover
  retained fabric on the robot and release beyond the narrow support edge.

### p08 straight pad-axis grasp approach

The narrow-support coupled run (source 91bc2bedceded48385ee8ff4ec9a01253cefa77e6c7da5015e3198d0618613ca) closed the snap but failed fabric capture at 193 controls. Its measured patch moved from x=0.280 m at 12.5 s to x=0.338 m at closing. An all-facet convex/box diagnostic finds commanded gripper/plate overlap at closing (maximum intersection-ball radius 1.60 mm) and measured contact consistent with the 5.24 mm TCP error. The initial diagonal approach placed the patch against one pad, then followed its displacement.

The long-tab controller now approaches along the pad axis, with normalized opening 0.65 (61 mm total gap) and smooth one-second closure. It still follows the observed free-edge patch and uses only stock 14D actions. The scene, material laws, snap springs, robot gains/limits and scoring limits are unchanged. This is an unvalidated controller revision until the next native run. diagnose_grasp.py records actual patch coordinates and convex hull/support intersections on CPU; it does not run physics or promote a gate.

### Preflight termination records

New launchers write exit.json with returncode 75 and simulation_started=false if the exclusive GPU lock or host GPU occupancy prevents startup. Preflight now also archives the full graphics/compute process table. This does not change GPU selection or bypass occupancy checks; active frozen jobs are unaffected.

### Recovered isolated gate decision

The detached fresh-process reproduction completed normally at 15:37 UTC after the chat interruption. Recovery completed its independent rescore, action/state audit, three complete H.264 decodes and the aggregate p08-isolated-gate-review.json. All required local gate checks pass. The successful source hash remains 9dc69b5e9ff8fbdecea958077e06e9146ff4564bf0fd26071706e03a9779f559. This revision updates documentation and status metadata only; no physical parameter, scorer threshold, archived result or full-task score is changed. Shirt/hanger integration remains unimplemented.

## p09 — 2026-11-21: one scored snap and sewn garment integration

User reduced the required closures to one designated upper-front press-stud.
The rest of the table-to-hanger task remains. First shared the existing p08
press/lift/release video before further implementation. Added a procedural
connected three-panel long-sleeve FEM shirt, local hardware rivets, seven
mechanically identical snap pairs, free wire hanger and static rail. Only the
uppermost pair will count. No mechanical parameter or validation limit relaxed.
Initial native work is passive already-hanging settling with all snaps open;
this cannot receive full-task credit. Records include every physics-step
strain/rivet measurement, 10 Hz standard actions/states and all three cameras.
CPU topology: 24,793 vertices, 48,976 triangles, 3 open boundary loops, one
orientable connected component; estimated fabric mass 177.9 g. No native garment
success claimed by this revision entry.

### p09 garment support diagnostics and full-rate contact recording

Initial arm/sleeve and rail-post/sleeve intersections were corrected. Mesh,
neck, hook balance, bar primitive and seam variants still fail passive shirt
support. A small closed fabric sleeve contact control passes; a preclosed upper
snap shirt control does not. 1920 Hz plus 128 cloth iterations was unstable and
is rejected. Current material laws and integrity limits remain unchanged.
Shared-vertex seams use continuous flexible FEM elements, not constraints to
external objects. Optional initial full-rate cloth geometry records resolve
contact events missed at 10 Hz. No full-task credit is awarded by any diagnostic.

The temporary garment seating predicate used an inconsistent axial bound that
missed retained cap gaps near 6.8 mm. It now reuses the already validated
robot_scoring.snap_seated predicate. A regression checks that case. Archived
scorers/results remain unchanged; all shirt support trials remain failures.
Current CPU suite: 41 passing checks. See owned p09-development.md for experiment
hashes, numerical results, failures and current diagnosis.

### p09 confirmed contact failure

Full cloth geometry recorded at 960 Hz confirms hanger centrelines crossing the
shoulder/neck fabric between normal 10 Hz frames. Earlier contact generation,
a closer initially clear placement and 64 contact-pair refreshes per step do
not fix it. This is a rejected garment-support prototype, not normal slipping
or full-task success. Complete-triangle capsule clearance checks were added
for reset geometry; scoped packaging can audit p09 without repeating all prior
isolated evidence. The parameter description now calls the lightweight hanger
body hollow polymer with metal hook: the earlier solid-wire wording was
inconsistent with its estimated 60 g mass and 10 mm body diameter. This is a
provisional material-description correction, with no physical values changed.

### p09 load-dependent contact failure

Added identical elongated-loop controls, with and without a free steel rod
(20 mm diameter, 80 mm length, assumed density 7850 kg/m3, computed mass 197.3 g).
The unloaded loop passes 5 s: 2.288% strain and 0.614 mm final-two-second drift.
Adding the rod fails at 0.5 s with 19.60% strain and 4.921 mm body-bar penetration.
A separate 1920 Hz run keeps cloth iterations at 64 and also fails: 27.80% strain,
0.4 s. These findings establish failure under the tested load, not a particular
engine defect. No model, material, mass, score or robot gain was tuned for credit.
Full task remains incomplete. Added a negative scorer test for a dropped load;
44 CPU checks pass. Native controls use no new attachments or external forces.

## p10 — bounded hanger decision, 2026-11-21

User approved fixed-versus-free loaded support, gentle distributed garment load,
and a representative wooden hanger, followed by a go/no-go decision. All three
fail support and have measured cloth/bar penetration. The hanger branch is
shelved; the original full task is incomplete. Isolated p08 remains valid.

- Fixed loaded loop: 13.888% strain, 4.349 mm penetration, loop/rod on floor.
  One declared world joint only; fixture cannot validate free support or task.
- Gentle shirt: 29.724% strain, 0.529 mm rivet error, 3.717 mm penetration.
  Same-radius capsule rail fixes gross hook overlap but not cloth contact.
- Wooden hanger: 29.044% strain, 0.691 mm rivet error, 6.318 mm penetration.
  430 mm width / 14 mm thickness reference; 20 mm shoulder height, 8 mm bottom
  dowel, 3.5 mm hook, 100 g and friction .4 estimated. Complete-triangle reset
  clearance; visible/collision geometry agrees. Fabric material law unchanged;
  initial front/back profile depths widen for physical wood clearance.

All seed 100, 50 stock 14D actions / 5 s, 4,801 native states, full score zero.
All three evidence/scorer audits and nine video decodes pass independently of
physical failure. New hook/rail witnesses invalidate the stronger interpretation
of the archived unloaded loop's limited passive pass. Historical files unchanged.

Added fixture allowlist/credit guards, hook/rail overlap witnesses, wooden box
triangle checks, independent exported-USD box geometry validation and bounded
result aggregation. 15% strain / 1 mm rivet limits retained. A tabletop full-shirt
variant is a different, unvalidated scope, not a renamed coupon success. Details:
the archived `p10-decision-round.md`.


## p11: user-authorized tabletop adaptation (2026-11-21)

The user requested an alternative to hanger contact, including fastening a
laid-out shirt. Selected a broad work table at stock YAM base height rather
than the floor, to keep the manipulation within ordinary robot reach. The goal
is now three upper press-studs on a complete front-up shirt, followed by both
robots withdrawing at least 30 mm and 3 s retention. The original hanger task
remains incomplete. This is a separate task scope, not a relabelled coupon.

The sewn adult shirt keeps its approximately 0.76 m body length, 0.56 m flat
chest, 1.56 m sleeve span and estimated broadcloth material. Rotation into the
horizontal layout preserves intrinsic rest lengths. Removed only hanger-specific
extra mesh lines; body resolution is 10 mm with existing 2 mm local placket
refinement (24,517 vertices / 48,440 triangles). Nominal placket overlap/alignment
is supplied at reset. A rigid wood-like table is 1.0 x 1.8 m, 40 mm thick, top
0.75 m, centre (0.605, 0.12) m. No external cloth attachment or grasp lock.

After passive whole-shirt support passed, added estimated physical exterior
rivet plates: 12 mm diameter, 0.4 mm thickness, local centre +/-1.15 mm and
+0.4 g per half. These supply visible exterior contact surfaces; the established
head, socket ring and spring-loaded balls are unchanged. All seven pairs are
mechanically compatible. One ordinary-action press/withdrawal diagnostic on the
whole shirt passes, independently audited, score zero for the three-snap task.

The controller uses stock 14D joint/gripper actions at 10 Hz. It lowers until
contact, aligns by frictional feedback, presses in 0.15 mm control increments,
requires four seated observations, withdraws, and repeats. The right arm moves
to a useful wrist-camera observation pose. A final withdrawal/viewing motion
exposes the finished garment. That Cartesian viewing target leaves about 35 mm
IK residual, while measured whole-robot clearance exceeds 75 mm; scoring uses
actual geometry, not intended poses. No robot gains or limits were changed.

A separate full-task scorer checks all 49 possible pairings, 6.0–7.2 mm axial
seating, <=0.75 mm lateral error, <12 degrees normal error, 3 s continuous native
retention and conservative full-mesh/full-robot clearance. Correct required
pairs earn 1/3 each; any wrong final pairing, prior damage, hidden attachment,
invalid state or cap forced through the rim invalidates the result. At least
98% of cloth nodes stay over the table, mean nodal drift <=20 mm. The 15% edge
strain / 1 mm rivet error / 1 mm table penetration / 25 N rigid normal contact
guards are unchanged. Controller labels/order earn no credit.

The first three-snap attempt failed before pressing at 3.9 s on 1.0715 mm table
penetration. Passive 960 Hz comparison passed, but two 1920 Hz episodes failed
(one reached 97.6% transient cuff strain). Smaller timestep was rejected. Added
full-state witnesses at peak table penetration. A separate 960 Hz comparison
increased only the cloth contact-generation margin from 0.3 to 1.0 mm, leaving
the physical rest offset at 0.15 mm. Both passive episodes passed, peak table
penetration 0.0628 / 0.0256 mm. Independent resting-gap analysis found identical
median 0.150144 mm and p99 0.150442 mm back-panel gaps, ruling out inflation of
the resting surface as the explanation. Material laws, rest geometry, snap
springs and scoring limits were not changed. All failed trials are retained.

`p11-tabletop-three-snaps-r2/episode-00` then achieved full score 1.0: seed 100,
489 actions / 48.9 s, all pairs 0,1,2 retained after withdrawal. Peak strain
5.1874%, rivet error 0.0193 mm, table penetration 0.1654 mm, rigid normal contact
10.7301 N. Independent source/GPU/exported-geometry/full-state/scorer audit and
all three videos pass. Frozen runtime source bundle:
`f8a80bb157b70cc1cfa432389efc0bece667d00decada78692f362d694d6691d`.
Fresh-process controller reproduction from that exact bundle also passes:
`p11-tabletop-three-snaps-reproduction`, seed 100, 465 actions / 46.5 s,
44,641 native states, full score 1.0. Peak strain 5.4804%, rivet 0.0150 mm,
table penetration 0.1661 mm, rigid normal contact 11.9177 N. All native metrics,
source/GPU/scorer/video audits pass. Initial measured positions match exactly
and maximum velocity difference is 3.47e-17. Feedback actions differ; no exact
open-loop replay or statistical reliability is asserted.

The same process rebuilt the full scene after this success. Initial positions
matched exactly and native gravity-warmup velocities differed by at most
3.47e-17; all snaps were open again. However, the following passive episode
failed at 4.95 s on 15.684% upper-placket edge strain. Recorded reset restoration
is verified, but this remains evidence of intermittent passive numerical
instability. It is not counted as success or hidden by the complete first run.

After freezing the native run, canonical post-episode scoring was strengthened
so one cap-through-rim state globally rejects the episode instead of allowing
other closures partial credit. Native controller/physics are unchanged. The
unaltered demonstration still yields 1.0 with identical per-control scores;
13 offline corrupted-copy shortcut checks pass. Source hashes bind that stricter
scorer to its review. Frozen source/results remain unchanged, so final delivery
and runtime source hashes differ explicitly. No hardware calibration, exact
Wrangler replica, statistical success rate, model run or submission is claimed.

Final current-scorer review also passes all 13 record-copy cases on the
reproduction. Both native demonstrations still score 1.0 with their original
per-control scores. Overhead and both wrists from both complete runs are fully
decoded and visually inspected at initial, pressing and final frames. The
successful tabletop adaptation supports the requested local feasibility claim;
the separate post-reset passive instability remains unresolved.

Packaging found an audit compatibility bug: a dispatch dictionary eagerly
accessed press_score in the earliest support-only frozen bundle, before that
function existed. Changed audit dispatch to select the function name before
attribute lookup. No physics or scoring rule changed. The incomplete first
package and its failure record remain; a fresh delivery-p11-tabletop-r2 packages
all evidence with the corrected auditor.


## p12 — stability and bimanual pilot (implementation in progress, 2026-11-22)

The accepted plan preserves p11 and first instruments the native strain maximum.
`--strain-witness` records full cloth positions/velocities, hardware and robot
poses/velocities, and eight native states on either side of the largest event.
`verify_strain_witness.py` recomputes signed and absolute strain on CPU; the
15% absolute-strain validity rule is unchanged.

The `stability` launcher stage runs fresh native processes under one inherited
exclusive GPU lock. Candidates are bounded to (64 iterations, 10 mm body),
(128 iterations, 10 mm body), and (64 iterations, 5 mm body). Each receives
10 s screens with seeds 100/101 and, after passing, a 300 s passive horizon.
The 5 mm refinement preserves the existing 2 mm snap patch sampling and local
rivet count. No material, fastener, physical rest offset, robot or score tuning
is part of this round. Validity failures remain recorded and stop that candidate.

Bimanual control and 20 Hz native validation have not yet been implemented or
validated. Native diagnostics currently use the preserved 10 Hz baseline.


p12 implementation update: the 20 Hz IPC backend, fresh-process reset adapter,
offline client contract, matched-action support ablation and benefit analyzer
are implemented. The complete pre-pilot CPU suite passed 95 tests. An input
regression over 483 recorded steps found exactly identical baseline actions
after adding the optional right-arm path. The first bimanual path has an 8.75 mm
conservative inter-arm clearance and at most 4.74 mm IK residual in the CPU
screen. These are nominal commanded kinematics, not native contact evidence.
Two fresh 10 s stability screens pass independent audits; the 300 s horizon
remains in progress. No model call or submission has occurred.

## p12 follow-up — 22 November 2026

- 109 CPU checks pass, including pause-aware native IPC deadlines. All physical
  source hashes remain identical to the running stability gate. Installed YAM
  source hashes are preserved.
- Independent audits confirm failed outcomes for the 64/10 mm long horizon
  (97.9117% table coverage at 136.6 s) and 128/10 mm first screen (22.3898%
  strain by 0.4 s). Final 64/5 mm candidate is pending.
- The long-run table-plane guard flagged cloth beneath the underside. The exact
  snapshot has no node inside the finite table; this does not rule out every
  triangle/history intersection. The coverage failure is independent, and the
  canonical result/limits remain unchanged.
- Added independent evidence/dependency hash checks, a finite-table diagnostic,
  and revision packaging with explicit incomplete-checkpoint handling.
- No native 20 Hz or bimanual success is claimed. No model call or submission.


## p13 cooperative scheduling — 22 November 2026

Scheduler integration is an orchestration revision. The task uses the stable
`snap-front-shirt` ID, validates canonical scheduler ancestry before skipping
its old exclusive gate, and passes every reservation descriptor to fresh
workers and the IPC backend. The unscheduled exclusive GPU-0 fallback remains.
Both native startup routes now select the assigned physical renderer and
masked physics `cuda:0`, disable multi-GPU rendering, and verify a real native
rigid-body tensor plus CUDA/RTX/process evidence in every simulator descendant.
The scheduler helper stays external and read-only; its source hashes are
recorded. Task caches, snapshots and evidence remain under the shirt output.

No parameter, geometry, contact, solver, scoring, controller, gain, limit,
action-dimension or installed dependency change is part of this revision.
The strict full-source evaluation gate remains; changed runner hashes require
new matching stability evidence despite unchanged environment-builder code.
The old p12 running/paused tree and its coordination guard are not migrated or
modified. Future scheduled runs must not attach that legacy occupancy guard.

Validation: 122 CPU tests pass. Scheduler job
`5c8475b27ea040c1b121d5d3fb578d12` ran two fresh one-second passive camera
workers (PIDs 1697075 and 1703477); both independent evidence audits passed.
All six H.264 camera clips decoded (10 frames each, 640×360 at 10 Hz); the three
first-worker views were visually inspected. Job `aa4da25c4b66429ab38701c940a59a30`
then verified the isolated-bench startup (PID 1709771, one no-press step).
All three verified native CUDA tensors and simulation device cuda:0, physical
renderer 1, CUDA UUID `GPU-cc9ca88f-3544-a5fd-4676-362e0d647c69`, and disabled
multi-GPU. Each recorded 8 MiB graphics-only enumeration on inactive GPU 0,
within the canonical helper's allowance; no compute was observed there from
these processes. Both scheduler reservations completed naturally with ECHILD
and zero return codes. These checks are not fastening, bimanual contact or
long-horizon stability evidence; passive task scores remain zero.

See `p13-scheduler-integration`, `p13-scheduler-smoke`, and
`p13-scheduler-bench-smoke` in the owned output root for exact commands, device
reports, frozen sources, seeds, versions, scores and complete results. The
source-preservation record verifies unchanged physical/scoring files and YAM
sources. The old legacy run remained under its original launcher and guard;
this integration did not signal, migrate, adopt or clear existing jobs.


## p14 submission preparation — 22 November 2026

- User-authorized guard-only handover replaced legacy guard 1434212 with 1896711.
  The native worker, coordinator, launcher and original exclusive gate remained
  alive. Only runtime-verified inactive graphics enumeration <=16 MiB is exempt;
  unknown or active compute/graphics workloads still pause the old tree.
- Added an exact-hash source provenance bridge for the reviewed scheduler-only
  runner change. Physical definitions and all other physical file hashes must
  match. The complete passive gate, physical validity limits and task scorer
  are unchanged. Unknown runner/helper changes remain rejected.
- Added a bounded local continuation queue and blind media preparation command.
  Media preparation independently re-audited the historical p12 score-1.0 run.
  An initial export failed on tuple/list dimension comparison; corrected export
  preserved all 483 frames at 10 Hz, 48.3 s, H.264, 2,053,701 bytes. Original
  footage and native results were not edited.
- Public criteria rechecked: a model attempt remains required for submission.
  No model calls, credentials, uploads, commits or submissions were performed.
  Native bimanual usefulness and the long-horizon gate remain pending.


## p15 — Submission evidence audit and persistent recovery record (22 November 2026)

Added a CPU-only completion preflight that separates pending/failed evidence,
independently audits completed native records, binds demonstration media to the
selected worker and source, checks replay action identity and configuration, and
requires published 20 Hz metadata only after native validation. Nine focused
negative checks cover partial scores, invalid states, old physical/control
profiles, substituted or edited media, missing camera frames and false progress
flags. The historical three-camera export passes actual decoding/provenance;
substitution as the future selected run and as a 20 Hz result is rejected.

A bounded, CPU-only watcher records live queue/coordination state and audits the
queue once it exits. It does not modify the frozen p14 continuation, its native
workers, reservations, physics or scoring. Source additions make the live source
bundle newer than p14; the physical-source provenance bridge remains unchanged.

Public novelty screening was refreshed using read-only unauthenticated GitHub
queries and the existing 3,747-PR snapshot. Five targeted queries found no match;
a positive-control shirt query returned ten PRs. This is not a signed-in board
or exhaustive benchmark novelty check. The title-only apparent overlap #668 is
a machine reset control, not garment fastening. Nearby accepted flexible-cloth
fastening includes safety-pin hem repair #2651. All raw search records and their
limitations are saved in the owned p15 output directory.

At the development checkpoint the current passive horizon was paused at 88.9 s
because unrelated uncoordinated GPU-0 compute jobs started. The new guard correctly
allowed the separately verified tiny graphics context and blocked those compute
workloads. No other task was signalled. The 300 s gate and native follow-up remain
incomplete; no new task success, bimanual benefit, model attempt or upload is claimed.


## p16 — Genuine Dream evaluation path (22 November 2026)

The user authorized one hosted Dream-1 episode after local validation, using
configured credentials; uploads and PR submission remain separate. Added a live
coordinator using the installed inspect-robots-dream integration and unchanged
native worker/physics. The worker stays network-isolated, external execution
requires controller=none, and strict public inputs/20 Hz timing are checked.
One attempt record, no episode/model fallback and no application inference retry
preserve the first actual outcome. SDK optional recording is disabled. CPU tests
exercise public observations, ordering, clock/preroll behavior, zero and error
preservation, credential-safe error logs, attempt reuse refusal and offline
socket isolation.

The separate native local check uses the same action loop with an injected local
session and small ordinary joint targets; it cannot call a hosted service. The
bounded preparation queue waits for the existing queue and stability job, reuses
valid demonstrations, and preserves failures. Independent model audit and the
existing media compositor now support genuine zero-score episodes. Media ID
remains unset. A successful CPU test is not native interface validation.

Initial inspection found stability still incomplete (93.7/300 s at the first
p16 check) and paused by uncoordinated GPU use. No new native stage or hosted
attempt has run at that point. The SDK found neither DREAM_API_KEY nor the
default credential file. Runtime packages remain installed unchanged; no keys
were printed, written or copied. Latest evidence lives in the p16 output records.

Final p16 CPU validation: 158 tests passed, including fifteen new Dream-path
checks. They cover camera identity and state ordering, ordinary action
provenance, rejected scripted assistance and offline socket isolation, zero
scores, refusals and no-retry behavior. Existing native physics/scoring sources
and installed YAM sources match the previous checkpoint. Final live readiness
also requires the same 300-second action budget as its scripted demonstration;
the older queue's 90-second cap cannot qualify for that profile. Native
preparation remains blocked by the unfinished stability job and GPU contention.

At the 19:00 UTC recheck, a configured Dream credential file had become available.
The initial absent-credential record is preserved; it is no longer the current
blocker. No authentication/inference request was made, and credential contents
were neither printed nor copied. The already-frozen native continuation keeps
its earlier metadata snapshot; current canonical metadata records this change.

## p17 — Required preparation prioritized (22 November 2026)

The user approved stopping only the idle older queue to skip optional bimanual
studies and duplicate 90-second demonstrations. Added a narrowly scoped pidfd
helper that refuses non-idle queues, children or changed process identities.
The original stability simulation, guard and exclusive lock remain untouched.
Its KeyboardInterrupt exit is a deliberate queue cancellation, not a physics
failure. Required p16 preparation stays frozen and will generate its own
interface checks. Credentials are present; no hosted model attempt yet.

Added exact local public-image/state records to the live session and independent
checks for JPEG size/identity, measured-state provenance, private-field rejection
and refusal timing. All 161 CPU tests pass. This changes input logging/auditing,
not native simulation or actions. The final p17 source requires its own short
native adapter check after the frozen p16 physical/demo preparation completes.

## p17 completed 20 Hz feasibility and packaging evidence

The full passive gate passed on the selected 64-iteration / 5 mm configuration. Two native 20 Hz, seed-100 scripted episodes reached 1.0 in 905 and 895 actions, with independent rescoring. The first three-camera review video is inspected. The final evidence packager now derives hosted request activity from retained live inference events and preserves zero/error outcomes; five focused CPU tests pass. No physics, scorer, native action loop, robot or live-runner source changed in this packaging correction. Hosted access remains unattempted while native adapter/replay checks are pending.

## p18 reset-camera diagnostic preparation

Actual visual review of the p17 adapter found a stale-looking first wrist view despite unchanged measured joints. A separate, offline `camera_reset_check` stage captures original and refreshed frames, actual camera/link poses, and all physics positions/velocities around zero-time sensor refreshes. It must finish before changing shared runtime or calling Dream. The old 20 Hz action replay remains untouched. The complete 166-test CPU suite passes; no new native or hosted result is claimed.

Review-media cards now state a model termination reason or explicitly say no model targets were received if an actual hosted request fails before producing targets. Its manifest preserves the actual model error and completed-request count. This is reporting only; no policy, physics or score change.

## p18 single startup camera sample filter

The two render-only probes showed that the wrist render transform is stale only before the first physical action. The first ordinary hold matches the fixed stock gripper-camera transform within 1e-6 m and 1e-5 rad; all measured robot joints remain unchanged. Neither sensor refresh nor render-Fabric reattachment is integrated into production.

A session wrapper drops only input index 0 during the installed, unchanged 32-action preroll. The first forwarded RGB/state sample is at 0.05 s. No replacement image, additional action, object state input or scripted controller is used. All native shared-runtime and physical-source hashes still equal the independently verified 20 Hz demonstration, so no physical demo rerun or source-matching exception is needed. Readiness requires the updated offline native adapter evidence. The complete CPU suite passes 170 tests.


## p18 completed local validation and genuine interrupted model episode (23 November 2026)

The final five-second offline installed-adapter check passed on physical GPU 1:
100 actions at 20 Hz, two injected chunks, measured joint ordering, script
rejection, no physics advance during inference, and exactly one excluded startup
image sample. Native source hashes remain identical to the two full-score 20 Hz
feedback demonstrations. All 905 recorded replay actions also executed exactly;
that valid open-loop replay scored 0.0. Both feedback demonstrations scored 1.0.
The scorer shortcut suite passed 13 cases. The CPU suite passed 170 tests.

One genuine `dreamlabs/dream-1-beta` episode then ran with verified exclusive
scheduler assignment to physical GPU 0, physics `cuda:0`, renderer 0 and
multi-GPU rendering disabled. The first request returned 48 targets in 2.1413 s.
After the standard 32 preroll holds and those 48 model actions, the next request
raised `DreamSessionExpired`: the service replaced the episode server-side.
The preserved result is score 0.0, 80 actions / 4.0 simulated seconds, no invalid
physics, termination `model_error`. Two requests started; one completed.
No scripted assistance, fallback, extra episode or application retry was used.
This does not measure model performance over the full 6000-action budget.

Independent raw-state and model-provenance audits passed. The uncut anonymous
three-camera H.264 model video has 80 frames at 20 fps and is 221336 bytes;
actual start, intermediate and final composite frames were inspected. The
full-score 905-frame demo remains unchanged and inspected. The live source is
`f44ad36cd58f4dd7dceb0f5486f84c4f58369d409ad1ad4faffeec4f0055a233`.
Only documentation and review records change after this frozen run. The durable
attempt claim is retained; no further hosted attempt is authorized. Media ID
stays null; board review, upload and PR submission remain separate.


## p19 submission packaging (23 November 2026)

Submission occurs in an isolated task-only worktree. Public fidelity and score
summaries are shortened to their current field limits; full details move to
README/EVIDENCE and compact evidence JSON. The actual model score remains zero
with the exact DreamSessionExpired error; both feedback-scripted scores remain
one. No new simulation, paid attempt or physical adjustment is performed.

Host-specific output/scheduler roots now resolve through task_paths.py with
explicit environment overrides. On the original host, the resolved directories
and canonical scheduler are unchanged. The GPU helper's function/class ASTs
remain identical; its exact portable hash is recorded as a second reviewed
helper identity, without permitting unknown edits. All physical/shared native,
controller, policy and scoring sources match the evaluated source. Historical
investigations remain at their module names because imports and frozen evidence
reference them. Generated caches and raw recordings are excluded from Git.
