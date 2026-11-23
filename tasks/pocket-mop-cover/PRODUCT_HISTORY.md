# Historical product investigations — superseded

> Historical investigation: results and pending-work statements below refer to
> earlier prototypes, released-v1, or the deferred SDK path. The active submitted
> endpoint and completed evidence are in [README.md](README.md). These diagnostics
> do not change the submitted physics or scoring.

This preserves earlier hypotheses and validation states. The active model is described in PRODUCT.md; claims below are historical, not current qualification.

# Product and procedure record

Reference selected on 2026-11-20: **VERMOP Sprint V holder, 40 cm, article
10049**, with **Sprint Tronic cover, 40 cm, article 14084**. This is a research
candidate, not a calibrated digital replica.

The [manufacturer's product page](https://www.vermop.com/en/floor-cleaning-en/sprint-system/)
describes rubber reinforcement at the pocket mouths and handle-operated
attachment using the folding V-shaped holder. These useful features are part
of the candidate; neither the floor/table support nor the handle will be
removed to require two hands on the fabric.

The [manufacturer catalogue](https://www.vermop.com/wp-content/uploads/VERMOP_Catalog_ENG.pdf)
lists the 40 cm holder and cover articles and describes the cover as tufted
polyester/cotton/microfibre. The nominal 40 cm size is published, not a precise
measurement of either the fabric pattern or the frame moulding.

The [Sprint V brochure](https://www.vermop.com/wp-content/uploads/2406_Broschuere_SprintV_RZ_EN.pdf),
page 4, shows the supported cover, both frame halves folded downward, and the
operator holding the handle above the central hinge. The practical sequence
is to present the folded ends to the pockets and press the handle downward
until the frame lies flat. Opening for removal uses the hand switch. A copy
and a page image are in the output research folder. Exact latch geometry,
actuation force, and joint travel are not established by these photographs;
published mechanical drawings and an explicitly estimated model are required
before claiming a physically justified passive lock. Hardware measurement is
not part of the currently authorized software-only implementation. No artificial seating trigger is acceptable.

## Parameters and uncertainty

`parameters.json` distinguishes published facts from unmeasured estimates.
There are **no measured hardware parameters**. The 200 × 140 mm coupon is a
partial-cover laboratory specimen with a 75 mm deep pocket; its dimensions,
1 mm shell thickness, 250 g/m² effective mass, 2 MPa membrane modulus, 20 kPa
bending modulus, and friction coefficient 0.55 are engineering estimates.
They are not manufacturer specifications. The initial 16 mm opening is a
stress-free curved mesh assumption, not a hidden constraint holding it open.
FEM gravity and contact can close it.

The initial uniform-shell diagnostic is deliberately earlier than the product
gate. Later probes add a finite 10 mm wide, 1.5 mm thick rubber lip with 2 MPa
modulus and 1100 kg/m³ density, all estimates, using explicit sewn correspondences.
Its contact dynamics have not passed numerical validation. Actual rubber strip
dimensions and modulus, frame end cross-section, hinge, lock, handle length,
and full cover dimensions remain unmeasured. Geometric or material changes
are revisioned, never selected silently from trial score.

## Candidate outcome rubric (CPU logic tested; native adapter pending)

Each pocket should receive 0.25 only when the correctly oriented frame end is
inside the pocket, inserted at least 40 mm across at least 80% of its width.
Both together earn 0.50. Full credit additionally requires a passive flat
lock (both half angles within 3 degrees of flat), released fabric, and a
functional check: lift by the handle until all fabric is at least 100 mm above
the support, translate the head 100 mm laterally and back, hold for two seconds,
then set down and release for two seconds. Engagement must persist throughout.
Gripping the handle for this check is allowed; gripping/supporting the textile
is not. Final grippers must be at least 20 mm from the textile.

Fabric penetration above 1 mm, broken seams, nonfinite state, out-of-workspace
objects, or more than 20% mesh-edge strain invalidate credit. These conservative
simulation diagnostics are not claims of textile failure strength. Photos,
a ruler, and the same lift/check would support real-trial judging. A score
requires testing on-top placement, single-pocket seating, wrong orientation,
penetration, and gripper-supported retention. Thresholds are provisional and
must be frozen after physical validation, before any evaluation.

Cleaning performance, water, dirt, absorption, and wringing are out of scope.
Importance and live board reservations remain unverified; this local research
does not assert benchmark novelty or submission eligibility.

## Evidence added during implementation

NVIDIA's installed schema documents collision-filter groups as sets of incident
elements: vertices/edges shared with an unfiltered triangle can remain active.
Current tests therefore compare individual triangle pairs with complete seam
vertex stars. The old assertion that authored surface/surface filters were
physically effective is withdrawn. No successful product gate is implied.

The VERMOP patent US20150208894A1 describes a spring-loaded sliding latch and
wing stops; correspondence to the Sprint V exterior must be checked, and
internal dimensions remain estimates.

An approved fallback is CombiSpeed Pro 40 cm with BaseMicro pocket cover
173357 (bulk article; individual pack 173358). A manufacturer-authored technical
sheet dated 2023-07-11 specifies 100% polyester, 68 g dry mass, and no backing
material. It is archived in the task output research/vileda directory. The
manufacturer's export webpage describes a different mixed-fibre composition;
the variant must be tied explicitly to the archived technical sheet. The
older frame leaflet is archived for context only: it names CombiSpeed 101200,
so it cannot silently establish exact CombiSpeed Pro 143572/143573 geometry.


The manufacturer Clean Facility catalogue describes CombiSpeed Pro's slim,
angled frame ends and double axis as aids to pocket entry (page 25). This
supports using an estimated tapered end rather than the original square bench
blade. The r22/r23 coupon tool retains its 100 x 100 mm planform and 6 mm main
thickness, with an estimated 12 mm top bevel ending in a 1 mm nose. These
dimensions are not hardware measurements. The support-facing surface is flat;
there is no fabric guide, capture constraint or disabled frame contact.

Source: https://ce.vileda-professional.com/medias/2020-The-Clean-Facility-Catalog-Low-Res.pdf?context=bWFzdGVyfHJvb3R8MTQ1Njg2MzZ8YXBwbGljYXRpb24vcGRmfGFEVXhMMmcxTUM4NU1qRTJNVFkzTnpBMk5qVTBMbkJrWmd8ODgyZjU4MjZkZjA0ZDk0YzUwM2I1YjNmY2M2NmQ4YWEzODM5MGYwMzMxMTE4MzlmYTIzMjJhNTRmNzRjZmJhOQ

A stock-YAM grasp diagnostic now uses a visible 250 x 100 x 120 mm bench
block to expose the coupon's free side margins. It is a laboratory support,
not a proposed full-task fixture or evidence of a complete installation.


The full-frame design may use the real Vileda short telescopic handle,
111384 (case) / 111389 (pack), at its published minimum length of 50 cm.
The 50–90 cm range and universal fitting are listed by the manufacturer:
https://www.vileda-professional.si/en/Floorcleaning/Mopping-Systems/CombiSpeed-Pro-System/Telescopic-Handle/p/Telescopic_Handle
The archived Clean Facility catalogue also explicitly lists 111384 as
compatible with CombiSpeed Pro (printed pages 9 and 25). This retains a real
handle and its connector; it is not an invented cut-down handle.

Mechanism research: Freudenberg WO2013156100A1 / EP2838411 describes two
pivoting wings, intermeshing teeth, and a rocker with a retaining groove.
The rocker is fixed in its working position and free in its release position.
The proposed embodiment does not use a return spring. This is a mechanism
family reference, not proof of the exact CombiSpeed Pro internal dimensions.
Product matching and physical locking still need validation.
https://patents.google.com/patent/WO2013156100A1/en
The previously considered EP1764024 family belongs to Leifheit, so it must
not be described as a Vileda/CombiSpeed product drawing.

The EP2838411B1 drawings are now archived from the EPO publication server (2016-12-21), especially page 8, figures 3–5. They show the short groove and triangular ground-contacting pedal feet omitted by the first frame hypothesis. The revised independent mechanism includes those features. A constant elastic hinge approximates the plastic groove lip using an estimated 24 x 36 x 1.5 mm polypropylene cantilever (E=1.3 GPa, density 900 kg/m^3, equivalent rotational stiffness 1.6453125 N m/rad). These dimensions/material values and the 30 mm handle-connector offset are estimates, not verified product specifications. The r46 geometry and matched r49/r53 tests now pass the isolated mechanism gate: 147 mm lift, 2.35-degree retained angle, disabled-catch folding, and physical pedal unlock assisted by 25 mm handle-side lift. Audited unintended penetration stays below 0.391 mm. Laboratory press and pusher are excluded from any final robot scene.


The r56 textile variant replaces coincident sewn-edge midplanes with an
estimated 2 mm flexible edge strip. All three closed boundaries remain sewn
through shared material edges; the mouth remains open. The panels have a flat
rest metric and begin arched only as a placement deformation. It introduces no
attachment, guide, or collision exclusion. The same 68 g published cover mass
is distributed over the extra seam area, and material constants are unchanged.
This hypothesis needs fresh insertion and grasp/release validation; it cannot
inherit the earlier curved-rest coupon pass.
