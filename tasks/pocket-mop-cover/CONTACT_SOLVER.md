# Contact-correction investigation

> Historical investigation: results and pending-work statements below refer to
> earlier prototypes, released-v1, or the deferred SDK path. The active submitted
> endpoint and completed evidence are in [README.md](README.md). These diagnostics
> do not change the submitted physics or scoring.

The baseline native rigid-body schema allows contact position corrections up to 3 m/s. In the covered, hands-clear cold-start diagnostic R124j the head overturns by approximately 143 degrees. Its native masses, centers of mass, inertia, gravity and passive joint drives were checked. Omitting the cloth prevents the overturn. Refining the timestep, cloth solver iterations, contact updates, or rigid velocity iterations alone has not removed it. Removing proven-contained duplicate end colliders also does not fix it.

A constant 0.1 m/s frame contact-correction limit, authored before native cooking, prevents the overturn in R125a. The matched explicit-default 3 m/s control R125c overturns again. This affects contact correction, not the permitted physical velocity of the frame or cloth. It does not add damping, a support, a capture condition, a pose controller or a state-dependent parameter. Geometry, masses, materials, robot actions and outcome thresholds remain unchanged.

NVIDIA documents reducing maximum depenetration velocity as a way to avoid contact overshoot: https://github.com/NVIDIA-Omniverse/PhysX/blob/main/ovphysx/docs/guides/collision_tuning.md . The installed rigid schema default is 3 m/s. At 480 Hz, a 0.1 m/s correction can resolve 1 mm over roughly five native steps. This is a numerical stability setting, not a measured product parameter; penetration must still pass the unchanged 1 mm audit.

The R125a run is not a passing physical demonstration. Seven initial native intervals exceed the penetration threshold, with a worst Right/End depth bracket of 1.09374–1.09457 mm. The final two-second maximum native and finite-difference cloth speeds are 0.09810 and 0.05888 m/s, exceeding the unchanged 0.03 m/s requirement. Its rigid audit passes. The residual fast nodes are concentrated near the right upper pocket and frame end. The estimated first-second mechanical-energy rise falls from 0.2655 J in R124j to 0.00653 J in R125a, but the cloth elastic energy terms are analytical proxies, not exact native solver energy or direct contact work.

Cold diagnostics restore held cloth geometry generated under the old solver and do not restore contact warm-start impulses. They cannot demonstrate installation, replay or a full benchmark score. A proposed ordinary configuration must first pass fresh affected isolated mechanism tests, including the disabled-contact negative control and physical pedal unlock. Then a fresh complete installation can test whether the initial penetration and residual jitter also arise when the new setting is used from reset. The ordinary benchmark remains on its previously qualified configuration until those checks are complete.

## Both correction variants rejected for ordinary task use

R126a applies the constant0.1m/s limit to the seven rigid frame links from reset. The positive bare mechanism fails retention (51.213degree wing angle) and the unchanged rigid penetration audit (1.44988mm). The frame-only setting is rejected for full-task use; no revised mechanism gate is issued. The baseline gate remains unchanged.

The cloth-only 0.1 m/s variant also prevents overturn in R125b, but its final native/finite-difference speeds of 0.13998/0.08855 m/s fail settling. Fresh insertion/withdrawal/collapse R127a passes. Stock-robot grasp R127b fails: the measured cloth patch rises only 2.896 mm. Fully closing the gripper in R127c briefly lifts about 60 mm, then the cloth slips back onto its support. Neither variant has a revised qualification gate, and neither is enabled by default.

The subsequent reduced diagnostics and rejected native configurations are documented in [PASSIVE_CONTACT.md](PASSIVE_CONTACT.md). No coupled-physics fix has been qualified.
