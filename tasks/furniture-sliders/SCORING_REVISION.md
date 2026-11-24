# Scoring revision: observable transport retention

Review of TK-6RE8Q6 found that continuous +/-0.5 mm foot-to-pad and pad-to-floor gaps cannot be established from the recorded cameras. Scoring version 2 removes both gap tests from the transport condition. It separates the moving retention assessment from the final stationary support assessment.

| Assessment | During the transport segment | Final released state |
| --- | --- | --- |
| Foot/pad association | Each foot-base centre stays within 20 mm horizontally of the same distinct pad centre | All four foot corners at least 2 mm inside the usable foam edge |
| Height | Each foot-base centre within 10 mm of panel height plus loaded slider thickness; each pad underside centre within 10 mm of panel height | Lowest foot-to-pad and pad-to-floor gaps within +/-0.5 mm, checked at rest with a feeler gauge |
| Upright | Table and each pad within 15 degrees | Table within 3 degrees; pads within 5 degrees |
| Continuity | Same four pairings at every 20 Hz sample; footprints stay on the panel; any failure resets the segment | Same final pairings; 2 seconds settled, hands 50 mm clear, no robot support |
| Distance/goal | At least 150 mm net table displacement in a retained segment continuing through release | Entire tabletop inside the marked destination |

The limits refer to observable body positions and orientations, not contact sensors. The 20 mm alignment radius is 40% of the 50 mm foot width; the 10 mm height tolerance is 20% of that width. They can be assessed against known object dimensions in the low views, with overhead video for travel and the goal. They are deliberately much coarser than a bearing gap. The height bands are absolute relative to the floor, so repeated small upward motions cannot accumulate into a carry. Separation or unloading inside those bands is not distinguished during transport: this is a visual retention proxy, not continuous force/contact metrology. The final precise support check remains mandatory.

Use synchronized 20 fps overhead and opposing low views, including the rear evidence view, with a ruler/reference image for scale. Track the same pads, compare base centres with the 20 mm alignment region, and check gross lift/tipping against the height/upright limits. Follow the full real-trial procedure in [README.md](README.md), including how to handle obscured views. The rear camera is not supplied to Dream. Whole-object lifting or upending before the qualifying segment remains legal; a visibly carried interval outside the height bands cannot establish transport credit.

## Implementation and validation

`scoring.py` now produces separate `matches` (stationary contact assessment) and `transport_matches` / `transport_retained` (observable motion assessment). The public transport accumulator depends only on the latter; the final score still requires the former. Pairing changes and observable retention loss clear the accumulated segment. `spec.json` adds only three transport-score thresholds. Physical construction, materials, masses, robot, actions, instruction, controller, run limit, cameras and scheduler binding are unchanged. The stricter one-pad load/contact gate remains a separately labeled diagnostic.

The 45-test CPU suite passes. Regression cases cover small moving height excursions with final contact restored, failure to restore final contact, alignment and height boundaries, pad expulsion, tipping, identity changes, off-panel/duplicate supports, slow accumulated lifting, missing samples, and independence from force/attachment diagnostics. Existing stationary support and native-interface tests still pass.

The existing recordings were rescored from every raw observation with the revised scorer; no simulation or hosted inference was repeated:

| Original recording | Observations / actions | Original score | Revised score |
| --- | --- | --- | --- |
| `full-002`, bare feet and four loose pads | 5839 / 5838 | 1.0 | 1.0 |
| `dream-001`, genuine unassisted hosted episode | 6001 / 6000 | 0.0 | 0.0 |

Stationary foot/pad matches are identical at every recorded sample in both runs. The revised demo accumulator reaches 204.390 mm (the original gap-based accumulator reported 203.966 mm). An independent choice of the actual movement-and-release interval, actions **5468–5838 / 273.40–291.90 seconds**, establishes **200.403 mm** net displacement. All 371 samples in that interval retain the same four pairs; maximum foot/pad centre offset is 2.608 mm, comfortably within 20 mm. Sampled reinspection of the unchanged uploaded video covers movement and released settling; it does not assert submillimetre camera resolution. Dream has no qualifying transport segment and remains at zero.

The original snapshots, recorded scores, measured states, inference logs and media are preserved. [provenance.json](provenance.json) retains original evaluated hashes separately from the current source and revised rescoring reports. Media ID **M-7HHEF3** remains valid for the same recordings and final scores.

From this task folder, in the existing native environment:

```bash
python -B -m unittest discover -s . -p 'test_*.py' -v
python -B tools/rescore_revision.py /path/to/full-002 --out /path/to/new-demo-rescore.json
python -B tools/rescore_revision.py /path/to/dream-001 --out /path/to/new-model-rescore.json
```

The rescorer verifies that the saved physical specification and scene/controller helpers match, loads the current scoring specification, and never overwrites the recording or its original result. `replay_score.py` can also replay legacy saved specifications: missing version-2 threshold fields take the explicitly reported version-2 defaults, not the old continuous-gap rule. Replaying the demo through both paths gives identical revised scoring details. This revision does not relax the runner's source checks for any future live model episode.
