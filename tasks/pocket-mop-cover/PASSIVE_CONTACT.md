# Passive contact investigation

> Historical investigation: results and pending-work statements below refer to
> earlier prototypes, released-v1, or the deferred SDK path. The active submitted
> endpoint and completed evidence are in [README.md](README.md). These diagnostics
> do not change the submitted physics or scoring.

**Unresolved; no qualified physics correction or full-score demonstration.** The ordinary task still uses its previously qualified model. These diagnostics initialize an installed assembly and cannot earn benchmark credit.

`passive_contact.py` reproduces the covered-frame overturn without robots, controllers or state writes after initialization. It restores the actual held geometry from R123 via `release-r124j-hands-clear`, copies the original table and physics scene, checks native initial cloth/link poses, and records every native step. The original material, frame authoring and scoring files remain unchanged.

## What the comparisons show

| Comparison | Observation | Interpretation / limit |
|---|---|---|
| R129b original articulated frame | About 143 degrees of overturn | Robot contact is unnecessary |
| R129d bare frame | Stays flat | Cloth coupling is necessary in this comparison |
| R129c rigid compound; R132c one-link articulation | Head stays flat; complete final two-second cloth speed checks still fail | Freezing internal joints is an invalid task model; flatness alone is insufficient |
| R131a latch in separate articulation; R131c handle as articulation root | Both overturn | These equivalent native representations do not fix it |
| R132a cloth self-contact disabled | Overturns | Self-contact is not necessary; disabling it is not an acceptable task revision |
| R133a one cloth contact subiteration; R134a zero handle-joint friction | Both overturn | Neither isolated numerical setting resolves it |
| R135c redundant wing stop made inactive, with finite limits and native readback | Overturns; gear remains coupled | The theoretically redundant limit is not the fix |
| R136a passive gear compliance, 480 rad/s | Overturns | Not a qualified material/constraint revision |
| R137a assembly raised 0.2 m, gravity zero | About 1.95 degrees of head inclination over two seconds | Internal contact alone does not reproduce the large overturn; this is a counterfactual, not settling success |
| R138a zero cloth bending stiffness | Overturns | Removing stored bending forces does not fix loaded contact |
| R139a body origins moved to their COM | Overturns; native masses and inertia preserved | A coordinate convention change does not fix it |
| R140a four rigid position iterations instead of 64 | Overturns | Repeated articulation position solves are not a standalone fix |
| R141a release 10 mm above the table under normal gravity | Overturns after landing | A small clearance alone does not resolve the problem |
| R142a all contact friction zero | Overturns | Tangential contact friction is not necessary |

The frozen-joint controls overturn even when **all six joints are fixed** (R143c/e). Native masses and inertias match the source exactly. Disabling internal frame collisions as well (R146a) gives the same result. Joint motion and internal frame contact are therefore not necessary for this failure. The single-body and one-link-articulation controls stay flat, but still fail complete cloth settling. The multibody representation matters; this does not by itself identify a particular engine defect.

Further comparisons:

| Comparison | Observation | Interpretation / limit |
|---|---|---|
| R144c ordinary rigid bodies, native hinges and gear, live USD cloth readback | 14.47 degrees tilt; cloth settling fails | The installed host tensor API cannot read surface nodes, but native USD points and velocities work; ordinary joints omit the articulation-only gimbal friction |
| R145c decimetre units with recorded native mass properties | 143.10 degrees overturn | Equivalent SI masses/inertias verified; unit conditioning is not a fix. Cooked collision geometry equivalence is not independently established |
| R147a supported PGS at 480 Hz | 1.85 degrees; final two-second speeds 0.24283/0.24284 m/s | Avoids overturn but fails settling |
| R147b supported PGS at 960 Hz | 142.88 degrees overturn | The apparent PGS improvement does not survive refinement |
| R148a direct native attach/start/step with public tensor views | 143.12 degrees overturn; native material, mass and spring checks pass | Framework asset callbacks and its startup sequence are unnecessary for the failure |
| R149d hybrid: rigid wings, articulated gimbal and flexure, native hinge/gear | 14.49 degrees tilt; final two-second speeds 0.35077/0.13031 m/s | Restores original native gimbal friction and clip spring, but still fails |
| R149e same hybrid with PGS | 14.33 degrees tilt | Solver change does not restore level placement |
| R150a centered initial handle roll | 143.11 degrees overturn | Centering the released handle does not fix contact |
| R151a one native GPU constraint partition | 143.04 degrees overturn | Does not support attributing this to the published partition-gap defect |
| R152a head-only 0.1 m/s contact correction, 480 Hz | 2.54 degrees; final two-second speeds 0.03172/0.01986 m/s | Native settling exceeds the unchanged 0.03000 m/s limit |
| R152b same head-only correction, 960 Hz | 16.04 degrees; final two-second speeds 0.28248/0.26946 m/s | Refinement fails; candidate rejected |

The centered-handle comparison changes only the initialized diagnostic pose; it is not a robot execution. Every new recording includes exact initial frame state for the energy proxy. The selective head correction leaves all original gimbal and elastic-latch correction values unchanged. Its apparent 480 Hz improvement does not qualify: the entire two-second window fails, and the 960 Hz comparison is worse.

R153a repeats the baseline through the cooperative GPU scheduler: 143.20 degrees, final two-second speeds 0.20973/0.13315 m/s. The scene, initial geometry/frame state, material parameters and installed versions are byte-identical to the preceding archived native control (143.16 degrees). R153b verifies the hybrid host-readback route while preserving CPU tensors and GPU dynamics. See `GPU_SCHEDULING.md` for exact commands and native assignment evidence.

A separate standalone SDK diagnostic uses only the already installed `ovphysx` package. `standalone_contact.py` imports the byte-identical accepted scene and requires matching native masses, COM, inertia, joint drives, cloth properties, element order and restored state. It cannot enter ordinary episodes. This is another diagnostic interface, not a package upgrade or qualified task model.

## Native configurations rejected

R129f cannot establish ordinary rigid-body gear stability: DirectGPU rejects the gear constraint. The installed CPU tensor backend does not implement deformable nodal-position readback. R144c subsequently establishes a working host-readback route using native USD surface exports, with live positions and velocities explicitly verified. This supersedes the earlier conclusion that the entire host route was unavailable.

R135a/b set the follower limits to infinity. PhysX then rejected the mimic constraint; the wings moved independently. Their smaller head motion is **not** evidence of equivalent-constraint improvement. The theoretical interval proof was insufficient without native constraint validation. Both recordings have explicit rejection sidecars. R135c uses finite inactive bounds, checks native limits, and records the wing gear residual throughout.

R130a/b combined PGS with the TGS-only every-iteration external-force option. Their native logs reject that combination; earlier conclusions about PGS convergence are withdrawn. The diagnostic now disables that option when selecting PGS. Corrected PGS runs R147a/b are accepted by native physics, but fail settling and refinement respectively; neither is qualified.

R143d faults when the native generalized-mass-matrix API is called on a zero-DOF floating GPU articulation. That query was removed and must not be repeated; per-link mass-property readback remains verified. R149a/b aborted during hybrid initialization/reporting and provide no dynamics evidence. R149c was blocked by another GPU reservation and never started.

The diagnostic and analyzer now reject both `PhysX error:` and native physics plugin error messages, including USD parser errors. `passive-native-error-audit-2026-11-22.json` inventories the historical rejections. Original recordings are retained. Earlier analyses of rejected runs are superseded by their rejection records.

## Measurement and qualification boundary

`analyze_passive_contact.py` measures all cloth nodes over the entire final two seconds, using both native velocity and position differences, including chunk boundaries. An instantaneous low speed does not establish settling. It can also estimate first-second mechanical energy from native rigid properties and analytical cloth proxies; this is not native energy or contact-work measurement. It does not provide a full strain/penetration audit or task score.

Every counterfactual flag is rejected in ordinary episodes. Frozen compound bodies, fixed joint groups, disabled friction/self-contact/bending and zero gravity are never candidate task settings. Physically equivalent representations and numerical solver changes would still need a stable diagnostic, timestep checks, fresh affected qualification gates, complete task execution and full audits before adoption. No scoring threshold has been relaxed.

## Reproduce the baseline

Use the existing native environment and a fresh run name:

```bash
source $HOME/IsaacLab-Arena/yam.env
$HOME/robocurve-gpu-scheduler/gpu-scheduler launch \
  --task pocket-mop-cover --mode exclusive --gpu auto --wait 300 -- \
  python -B $HOME/benchmark-tasks/tasks/pocket-mop-cover/launch.py \
  --name passive-new --experiment passive_contact \
  --passive-source release-r124j-hands-clear --passive-mode articulated \
  --product-profile base_micro --physics-hz 480 --iterations 64 \
  --spacing .0025 --seconds 4 --contact-updates 8 --contact-subiterations 2 --ccd
```

The launcher validates scheduler membership and holds the complete descendant tree under the assigned reservation. Unscheduled launches retain exclusive admission and actual occupancy checks. Source recordings are required. Each run preserves frozen source, request, native logs, initial readbacks, geometry and every native state. The source version that produced a recording is the copy inside its `source/` directory.

The native constraint and solver references are [NVIDIA's mimic-joint documentation](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/rigid_bodies_articulations/articulations.html#articulation-mimic-joints) and [articulation stability guide](https://docs.omniverse.nvidia.com/kit/docs/omni_physics/107.3/dev_guide/guides/articulation_stability_guide.html). The actual parser rejections and tensor limitations above were observed locally; they are not inferred solely from documentation.

The first standalone attempts R154a/b passed native rigid and material property
checks but stopped before dynamics recording: the surface tensor reported zero
nodes. The installed SDK requires DirectGPU readback to be enabled explicitly;
GPU dynamics alone does not provide its cloth TensorAPI. R154c onward sets
`/physics/suppressReadback=true` before creating the SDK, matching the source
recording, and restores the exact state after the same-duration warmup. R154a/b
are failed initialization attempts, not stability evidence.


The accepted installed-SDK comparisons R154c-f keep the frame level, including
at 960 and 1920 Hz, but fail all complete final-two-second cloth settling windows.
The 1920 Hz result is worse, so timestep refinement is not a correction. R154c's
complete 1,920-state cloth audit passes strain, frame/table penetration and
self-crossing checks; it is not a rigid/rigid audit or benchmark qualification.
R155a reproduces the original free-space control with nearly matching early
cloth/frame reaction, which supports active two-way coupling. Native frame shape
friction/contact offsets also match. See `SDK_COMPARISON.md` for exact results,
legacy-friction uncertainty and the remaining implementation boundary.
