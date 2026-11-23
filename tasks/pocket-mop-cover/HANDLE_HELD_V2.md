# Handle-held endpoint: outcomes and simulation assurance

The handle-held-v2 endpoint was authorized on 2026-11-22. This document now
separates its observable task conditions from its **simulation-only** validity
checks in response to scoring review. The physical model, controller, native
scorers and thresholds are unchanged. Historical released-v1 scores remain
preserved; unsupported final placement is outside this endpoint.

## Observable task conditions

Fit both sewn pockets, open and physically lock the frame, then demonstrate
retention while one hand holds only its handle. Each correctly oriented pocket
must enclose 40 mm of its frame end across the central 80 mm of its 100 mm width,
with the textile ungripped for two seconds. Either qualifying final pocket earns
0.25 partial credit; both earn 0.5 without a complete functional proof.

For 1.0, maintain both pockets and a locked frame flat within 3 degrees while
lifting all cloth at least 100 mm above the table, moving the head at least
100 mm horizontally, returning within 10 mm and holding for two seconds.
Only the same hand on the handle supports the assembly; the other entire arm
stays at least 20 mm clear. Finish holding the handle. Loss of lift, seating,
lock/flatness or exclusive support, a hand change or inadequate other-arm
clearance resets the functional motion proof. Interrupted holds cannot be added
together. The execution budget remains 2,400 ordinary targets at 20 Hz.

Physical trials use [REAL_TRIAL_SCORING.md](REAL_TRIAL_SCORING.md) and its linked
worksheet. Before/after inspection replaces unavailable real-cloth strain and
seam-gap measurements: new visible tears, holes, separated seams, detached
pockets or broken frame/latch invalidate a real trial. Unauthorized restraints
or intervention and exceeding the budget also give zero. Inadequate evidence is
unscorable; it is not guessed to be a pass or recorded as policy failure.

## Additional simulation-only validity and measurement methods

The unchanged simulation implementation requires:

- Native edge strain no greater than 20%, penetration/seam gaps no greater than
  1 mm, finite states and no hidden constraints or diagnostic initialization.
- At least 2 mm geometric separation between all non-handle frame parts and
  every robot link; both holding finger pads geometrically adjacent to the
  handle within 2 mm at every native state. Separate cloth-proximity checks
  conservatively reject textile grasp/support.
- Complete native-substep material/contact/support auditing. The 20 mm other-arm
  clearance and the task's centimetre-scale motion conditions are measured from
  recorded geometry; pocket enclosure/orientation is sampled at 20 Hz.

The 2 mm values infer handle contact and exclude other load paths in simulation;
they are not real-trial instructions to measure microscopic gaps. Real judges
use visible grip, absence of non-handle support and the freely lifted motion.
The same distinction applies to numerical strain and penetration. Simulation
validity can still veto a numerical success, and no bypass is introduced.

The Python result fields `score`, `success` and `invalid_reasons` retain their
simulation meaning and exact behavior. Real-trial worksheets have their own
explicit domain/status and contain no mesh or substep requirements.

## Preserved evidence and limits

Fresh demo R168 and fresh-process replay R170 each independently score 1.0 in
1,865 actions with complete simulation audits. Hosted Dream-1 R171 remains 0.0,
including its recorded contact invalidity. The original released-v1 R123 remains
0.5. Neither the earlier retrospective screen nor this documentation revision
creates a new successful episode. [SCORING_REVISION.md](SCORING_REVISION.md)
provides timed examples and the limits of visual measurement. No hardware trial
has validated the real-trial protocol.

Existing disabled-catch controls establish that handle-side support alone does
not keep the folding head locked. Negative controls cover incorrect seating,
wrong orientation, folded wings, direct head/textile support, interrupted holds
and invalid simulated physics. Final unsupported placement and cloth settling
remain unqualified under the preserved released-v1 endpoint.
