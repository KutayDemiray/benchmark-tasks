# Handle-held finish, revision 2

Authorized for testing on 2026-11-22. This is a new task endpoint, not a correction
to earlier scores. Historical version 1 scores and the known hands-clear release
instability remain recorded. No physical, material, robot or numerical parameter
is changed by this revision. The ordinary Kit backend is retained.

Fit both sewn pockets, physically open and lock the frame, then demonstrate
retention while one robot holds only its handle. Lift all cloth at least 100 mm
above the table, move the head at least 100 mm horizontally, return within 10 mm,
and hold for 2 s. Finish holding the handle, with the other arm clear.

The original 40 mm pocket depth, 80% width, correct orientation and 2 s pocket hold
remain required. The cloth must be ungripped and the wings flat within 3 degrees.
All robot geometry must stay at least 2 mm from every frame part other than the
handle; the non-holding robot must stay 20 mm clear of the entire assembly.
Both fingers of the holding hand must remain geometrically adjacent to the
handle (conservative swept clearance below 2 mm at every native state). This is a geometric support
inference, not measured contact-force instrumentation. Combined with the free
assembly's sustained lifted motion and absence of other load paths, it tests
handle-only retention. Support interruption or changing hands resets the
functional motion proof. The test is independent of controller phase names.

Partial credit remains 0.25 per seated pocket. Full reward 1.0 requires the full
retention sequence above. Penetration/seam gaps above 1 mm, strain above 20%,
nonfinite states, hidden constraints, diagnostic fixtures, or exceeding 2400
20Hz actions still invalidate the episode. All numerical/contact audits cover
every recorded physics step. These thresholds are unchanged.

Version 2 does not require final table placement, release of the handle or the
version 1 final resting-cloth velocity window. No dirt-removal task is added.
Existing disabled-catch physical controls demonstrate that handle-side support
alone does not keep the folding head locked. Unit negative controls cover
incorrect seating/orientation, folded wings, direct head support, textile
support, interrupted holds and invalid physics. Fresh native demonstration R168 and independent replay R170 now each score 1.0
with complete physical audits. The historical re-assessment was a preliminary
screen; it is not the submitted demonstration.
