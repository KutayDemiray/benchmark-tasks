# Installed standalone PhysX comparison

> Historical investigation: results and pending-work statements below refer to
> earlier prototypes, released-v1, or the deferred SDK path. The active submitted
> endpoint and completed evidence are in [README.md](README.md). These diagnostics
> do not change the submitted physics or scoring.

**Passive settling and isolated latch pass; full task remains unqualified.** The ordinary task remains
on Kit PhysX and its best complete audited score is 0.5. These runs are cold,
initialized diagnostics without robots; they cannot earn benchmark credit.
No dependency, installed package, shared environment or task score was changed.

## Recorded evidence

| Run | Interface and step rate | Final head inclination | Maximum cloth speed over the entire final two seconds, native / differences |
|---|---|---:|---:|
| R153a | Kit PhysX, 480 Hz, 4 s | 143.1963 degrees | 0.20973 / 0.13315 m/s |
| R154c | Installed standalone ovphysx 0.5.11, 480 Hz, 4 s | 0.00109 degrees | 0.10445 / 0.07891 m/s |
| R154d | Same SDK, 480 Hz, 8 s | 0.00122 degrees | 0.10815 / 0.06218 m/s |
| R154e | Same SDK, 960 Hz, 8 s | 0.00030 degrees | 0.06065 / 0.05205 m/s |
| R154f | Same SDK, 1920 Hz, 8 s | 0.00027 degrees | 0.34239 / 0.29936 m/s |
| R156a | Same SDK, 480 Hz, 16 s | 0.00103 degrees | 0.07563 / 0.04601 m/s |
| R157a | Kit, SDK-matching native link order, 480 Hz, 4 s | 143.09983 degrees | 0.09415 / 0.08590 m/s |

All three eight-second SDK runs remain below 0.48 degrees of inclination throughout.
Native checks establish a floating base, gravity enabled on every link, positive
inverse masses and all six original joint limits. The wing gear residual stays
below 0.003 degrees. Masses, COM, inertia, clip stiffness/damping/force limit,
cloth material and damping values, rest nodes, triangles and initial poses are
checked against the source. Initialization restores positions and zero velocities;
there are no physical state writes, clamps or attachments during recording.

The complete R154c cloth audit covers all 1,920 native states: maximum edge strain
3.8734%, no cloth/frame or cloth/table penetration above the existing 1 mm tolerance,
and no nonadjacent self-crossing. It does not audit rigid/rigid contact. No complete
cloth audit for the longer/refined runs is claimed. All settling windows above fail
the unchanged 0.03000 m/s threshold. R154f completed all 15,360 native steps and
failed the refinement check: cloth motion increased at 1920 Hz. A smaller timestep
is therefore not an established correction. No ordinary timestep was changed.

## Force-transfer control

R155a reads the exact existing R137a zero-gravity scene through the installed SDK.
Both assemblies start 0.2 m above the table, with zero velocity and no subsequent
state writes or forces. The original held-state chunk hash verifies reconstruction
of the older initial frame pose; cloth rest geometry and scene files are unchanged.
Every frame collider and cloth node remains more than 0.19 m above the table.

At the same early peak (22.917 ms), native frame linear momentum is
`[-0.001170, -0.001773, -0.000839]` kg m/s in Kit and
`[-0.001138, -0.001775, -0.000841]` kg m/s in the SDK, with opposite cloth-momentum
estimates. This supports two-way force transfer; the SDK frame is not simply held
still or disconnected from cloth. Cloth mass distribution is analytical, so this
is not a complete momentum-conservation certificate. Later momentum differs.

All 25 native frame shapes report friction 0.55/0.55, restitution zero, contact
offset 0.5 mm and rest offset zero, matching authored values. No additional
material or contact changes were made. Both processes completed cleanly and the
scheduler retained the reservation through descendant completion.

## Equivalence limits

The SDK consumes the exact accepted scene. Its refinement changes only the
`physxScene:timeStepsPerSecond` declaration and explicit timestep; each change has
hash evidence. Identical USD does not establish identical engine behavior or every
native property. The SDK's newer friction-properties tensor reports zeros for its separate
friction model; R158a/b OmniPVD captures now independently confirm the original
0.03 legacy gimbal friction in both engines.
Native mechanism release now passes the isolated SDK positive/negative gate.
Cloth insertion and stock robot grasp still need SDK qualification. No specific vendor defect is established.

`sdk-property-order.json` corrects a metadata-label error in early SDK reports:
property arrays were correctly reordered to the original source link/joint order,
while the labels initially showed SDK discovery order. The physical comparisons
and recordings are unaffected; originals and their hashes are retained.

R154a/b failed before dynamics because the SDK requires explicit DirectGPU opt-in
for surface TensorAPI readback. R154c onward enables it before instance creation,
matching the source mode, and completes a timestep-sized warmup before restoration.
The physics-only SDK process does not start a renderer and records this explicitly.
Ordinary Kit task runs retain their actual RTX device verification.

## Current implementation boundary

The ordinary Kit task backend stays intact. Task-local SDK workers use only the
already installed packages. They are explicit diagnostic paths, separate from
the scheduler integration, with their own gates. No installed code is patched.
The remaining order is:

1. Complete the unchanged contact insertion/withdrawal coupon and stock YAM
   grasp/lift/place/release in this SDK. Preserve native gains, joint limits,
   friction, per-physics-step gravity compensation and14D actions. Check every
   recorded state and all three actual RGB camera outputs.
2. Only after matching gates pass, run a complete from-reset task candidate,
   audit it, and execute a fresh independent action replay. One full score is
   required; there is no90% success threshold.
3. Finish reproducible reset and the portable task/model adapter and package.

A successful process exit alone does not qualify physical behavior. The SDK
camera worker is implemented but has no native validation at this checkpoint.
GPU assignment must be verified from actual native physics/render outputs and
allocations after warmup. Initialized passive scenes never receive task credit.

## Native-property investigation

The longer R156a observation still fails the full two-second velocity window.
Each node stays inside a box with diagonal at most 0.3112 mm in that window;
this additional displacement diagnostic does not change the velocity score.
R157a changes only USD child ordering, with exact attribute comparison and native
mass/COM/inertia verification. Native body/joint discovery order matches the SDK,
but the frame still flips. Ordering is therefore not a demonstrated correction.

`--passive-pvd-record` enables a diagnostic-only native OmniPVD capture, limited
to 0.25 seconds. `read_native_properties.py` reads complete 0.3/0.4 streams offline,
preserves first and last published values, hashes large arrays, and rejects unknown
versions/commands or truncated input. It does not infer missing properties.
NVIDIA's bundled recording and independent USD conversion agree on all 24 checked
mass, damping and depenetration values. Its message fixture also decodes correctly.
Reference sources and checks are in `native-property-reference` under task outputs.
These captures are intended to compare actual solver defaults before more physics
changes; their status and results are recorded in the output handoff.

R158a/b completed native property recordings on the assigned physical GPU1, with
physics cuda:0 and actual RTX1 in the Kit process (the SDK process is physics-only).
The records identify PhysX5.10.0 in installed ovphysx0.5.11 and PhysX5.9.1 in Kit.
Cloth native defaults differ: maxDepenetrationVelocity infinity versus3m/s,
linearDamping0 versus.005, and sleepThreshold.05 versus.005 (SDK thenKit).
Articulation sleep/stabilization defaults also differ. Active material constants,
scene flags, contact parameters, joint limits/drives and legacy friction match.
The large native joint velocity limits differ by the degree/radian factor; both
are far above physical trajectories. COM rounding is compared with explicit
floating-point tolerances, not interpreted as a mechanical change.

The reciprocal diagnostic `--passive-cloth-correction bounded|unbounded` changes
only that one authored cloth attribute before cooking. `passive_defaults.py`
checks layer equality after removing the new property. The bounded sequential
`run_passive_default_comparison.py --prefix passive-FRESH` can run both short
property captures and four-second recordings under one exclusive scheduler
reservation. It verifies the expected native setting and rejects extra recorded
physical changes before running the longer pair. No ordinary setting is changed.

## Reciprocal contact-bound result (R159)

The native-property comparison covers258 selected published static properties.
Each reciprocal capture differs from its own baseline in exactly one property:
cloth maxDepenetrationVelocity. SDK with3m/s staysflat at0.0012degrees after0.25s;
Kit with an authored infinity tips71.31degrees in the same observation interval.
Kit represents infinity by the largest finitefloat32,3.4028234663852886e38.
The initial batch assertion expected literal infinity and stopped after the two
valid native captures. Its verifier was corrected to accept the observed maximum
float representation; no physics was changed to correct that assertion.

The SDK4s continuation staysflat at0.001087degrees, but the complete final2s
velocity maxima are0.128183/0.079016m/s (native/finite differences), still failing.
Kit's longer continuation was skipped because its short control already tips.
Thus the cloth correction bound does not explain the frame response difference
and has not fixed settling. The paired native captures verify the old0.03
legacy gimbal friction in both engines; it is no longer an open equivalence issue.

## Matched-default settling milestone (R161)

The installed SDK is PhysX5.10.0; Kit embeds5.9.1. R161 explicitly matches the
captured Kit cloth correction3m/s, linear damping.005 and sleep.005, articulation
sleep.005/stabilization1, and effectively unlimited angular velocity bounds.
The latter use the documented native setter before reset: the legacy USD
max-float sentinel did not reproduce the degree/radian conversion. The complete
short OmniPVD comparison matches all258 selected published static fields, with
zero differing or unrecorded fields. This does not prove all solver/cache/cooked
internals identical or establish a specific vendor defect.

A fresh maximum16s passive test stops ONLINE at its first qualifying20Hz
boundary,10.2s/4896 native steps. Every native step checks all-node native AND
finite-difference speed, head/wing angles and support height; any violation
restarts the timer. The independent final-window analyzer confirms the entire
last2s: maximum speeds0.01862191 and0.01373264m/s against the unchanged0.03 limit.
Final head inclination0.00102185deg; maximum0.46850490deg. The observer does not
write physical state or change forces, material, geometry or score.

Complete cloth and rigid audits of all4896 states PASS. Maximum strain0.03874706;
no excessive cloth penetration or self-crossing; maximum nonadjacent rigid
penetration bound0.00097210m, using the unchanged hinge-bearing exclusions.
The video `progress-videos/2026-11-22-passive-release-validated.mp4` shows the
entire10.2s at actual speed (204 decoded frames), full velocity trace and hold.

Re-examining the FULL olderR156 history also finds a real2s control-aligned hold
at6.65–8.65s. Its fixed final2s still fail because of five brief speed spikes.
The new result is not a permanent-quiescence or trajectory-convergence claim.
`find_passive_holds.py` records the historical search; `passive_settling.py`
implements the predeclared online observer used in the freshR161 test.

## SDK latch qualification (R162b)

The original force-limited laboratory trajectories match all28800 source steps,
with maximum command difference5.73e-17m. The positive36s run retains a146.88mm
lift with maximum retention wing angle1.61684deg, then physically unlocks to
51.179deg. The disabled-catch24s control folds51.217deg during a148.55mm lift.
Both complete rigid audits pass; maximum positive penetration bound0.39182mm.
`mechanism-sdk-gate.json` passes independently of the ordinary mechanism gate.
A mixed Kit/SDK positive/negative pair is explicitly rejected by the qualifier.

The firstR162 attempt stopped at a USD metadata restoration assertion before
engine initialization. Preserving the authored raw list-op instead of its
composed equivalent corrected that provenance check. No physics was changed.
All successful SDK children verified maskedcuda:0 on assigned physicalGPU1;
these tests had rendering disabled and do not claim RTX camera verification.
