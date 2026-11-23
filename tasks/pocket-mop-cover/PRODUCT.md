# Current product model and assumptions

The active candidate is an **estimated Vileda CombiSpeed Pro 40 cm frame with a BaseMicro pocket cover and the compatible short telescopic handle**. It is not a calibrated digital replica. No hardware measurements have been made. The earlier VERMOP rubber-pocket and finite-seam experiments are preserved in [PRODUCT_HISTORY.md](PRODUCT_HISTORY.md); they are not the active benchmark model.

## Published reference and chosen variant

The frame reference is CombiSpeed Pro 143572 / 143573. The BaseMicro cover reference is 173357 / 173358, tied specifically to the manufacturer technical sheet dated 2023-07-11: 68 g dry mass, polyester, no separate backing. The manufacturer's export webpage describes a different composition, so that webpage must not be treated as the specification for this modeled variant. The [2023 technical sheet](https://www.franz-mensch.de/downloadfile?file=%2Ffiles%2Fdocuments%2FCUSTOM_DOWNLOADS%2FTVI173358%2FTDB%2Fde%2FTDS_Vileda+Professional_CombiSpeed+Pro+Base+Micro.pdf) and its source record are preserved externally under `$OUT/research/vileda/`; manufacturer PDFs are not bundled in the source package.

The [manufacturer catalogue](https://ce.vileda-professional.com/medias/2020-The-Clean-Facility-Catalog-Low-Res.pdf?context=bWFzdGVyfHJvb3R8MTQ1Njg2MzZ8YXBwbGljYXRpb24vcGRmfGFEVXhMMmcxTUM4NU1qRTJNVFkzTnpBMk5qVTBMbkJrWmd8ODgyZjU4MjZkZjA0ZDk0YzUwM2I1YjNmY2M2NmQ4YWEzODM5MGYwMzMxMTE4MzlmYTIzMjJhNTRmNzRjZmJhOQ) describes angled frame ends and a double-axis connector. Its printed pages 9 and 25 list the compatible short telescopic handle, 111384 / 111389. The [manufacturer handle listing](https://www.vileda-professional.si/en/Floorcleaning/Mopping-Systems/CombiSpeed-Pro-System/Telescopic-Handle/p/Telescopic_Handle) gives a 50–90 cm range; this scene uses 50 cm. The older CombiSpeed 101200 leaflet is historical context and does not establish current frame dimensions.

## Active estimated geometry and physics

| Quantity | Active model | Evidence |
|---|---|---|
| Head length | 400 mm nominal | Product size; detailed molding unmeasured |
| Head width / end thickness | 100 mm / 6 mm | Estimates |
| End bevel | 12 mm long, 1 mm nose | Estimated geometry motivated by angled product ends |
| Cover pattern | 440 × 140 mm, two 75 mm pockets | Estimates |
| Shell thickness | 1 mm | Effective model estimate |
| Membrane / bending coefficients | 2 MPa / 20 kPa | Uncalibrated effective surface parameters |
| Poisson ratio / dynamic friction | 0.3 / 0.55 | Estimates |
| Elasticity / bending damping | 0.005 / 0.005 | Native coefficients; not a dimensionless damping ratio |
| Rest shape | Curved pocket panels, initial mouth 16 mm | Estimated stress-free construction; free to close under gravity |
| Sewing | Shared vertices on three closed edges of each pocket | Fixed material topology; no frame attachment |
| Contact / rest offset | 1 mm / 0.5 mm | Numerical shell-contact representation |
| Qualified discretization | 2.5 mm, 480 Hz, 64 iterations, 8 contact updates, 2 contact subiterations, speculative CCD | Matching isolated gate evidence |

`robot_gate.fabric()` derives the effective density from the 68 g reference mass and estimated flat pattern area. `parameters.json` retains legacy research fields for provenance; its top-level VERMOP label does not select the active full-scene material. `--product-profile base_micro` and `FullCover` select the active model. Each run saves the actual values in `effective-parameters.json` and its native mesh in `geometry.npz`.

The frame uses a passive geared wing pair, contact rocker and elastic clip lip. [Freudenberg EP2838411B1](https://patents.google.com/patent/EP2838411B1/en), especially figures 3–5, is a related mechanism reference, not verified CombiSpeed internal metrology. The clip approximation uses an estimated 24 × 36 × 1.5 mm polypropylene cantilever, 1.3 GPa modulus and 900 kg/m³ density, giving 1.6453125 N m/rad rotational stiffness. The handle joints are passive. No seating trigger, proximity lock, cover capture or runtime object-pose controller is used.

## Task procedure and validation boundary

The active endpoint is **handle-held-v2**. The robot opens the frame, seats both
ends in the sewn pockets, and opens and locks the frame flat. Holding only the
handle, it lifts all cloth at least 100 mm above the table, translates the head
at least 100 mm, returns within 10 mm and holds for two seconds. It finishes
holding the handle. The other entire arm stays at least 20 mm clear; the cloth
and non-handle frame parts receive no extra support. In simulation, conservative
2 mm contact/separation tests enforce that support condition. Physical judges
use visible support and close-up inspection as described in
[REAL_TRIAL_SCORING.md](REAL_TRIAL_SCORING.md). Exact outcome conditions are in
[HANDLE_HELD_V2.md](HANDLE_HELD_V2.md).
Cleaning, water, dirt and wringing are outside this task.

The matching isolated insertion/grasp and passive mechanism gates pass. Fresh
continuous R168 and fresh-process R170 replay each independently score **1.0**
within 1,865 actions, with complete material, rigid-contact and exclusive-handle
support audits. The genuine R171 Dream-1 attempt scores **0.0**; no pocket is
installed and the rigid-contact audit rejects right-arm self-intersections.
See [evaluation-evidence.json](evaluation-evidence.json) for recorded evidence.

The original **released-v1** endpoint additionally required table placement,
withdrawing both hands and two seconds of settling. R123 remains **0.5** under
that rubric. Its covered-head overturning problem is unresolved; a matching
bare-frame control stays flat. This limitation is preserved, and is not a claim
about real mop behavior. The held endpoint does not qualify unsupported final
placement. [RELEASE_DIAGNOSTIC.md](RELEASE_DIAGNOSTIC.md) records that investigation.

The held revision preserves the material, geometry, solver and simulation-only
invalidation thresholds. Passing penetration and strain bounds alone does not
establish calibrated passive dynamics. Material changes require matching fresh
qualification. [MATERIAL_MODEL.md](MATERIAL_MODEL.md) records an earlier damping
hypothesis, not an adopted fix.
