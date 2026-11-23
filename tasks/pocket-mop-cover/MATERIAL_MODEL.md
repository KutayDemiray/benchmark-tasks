# Surface damping hypothesis — not a qualified material revision

> Historical investigation: results and pending-work statements below refer to
> earlier prototypes, released-v1, or the deferred SDK path. The active submitted
> endpoint and completed evidence are in [README.md](README.md). These diagnostics
> do not change the submitted physics or scoring.

The original uniform effective surface uses bending damping 0.005. NVIDIA's
surface material API labels bending damping in 1/s; it is not a dimensionless
5 per mille damping ratio. Its surface and volume elasticity-damping conventions
also differ. No value is inferred for volume rubber from this surface work.

Primary API references:
- https://nvidia-omniverse.github.io/PhysX/ovphysx/0.8.0/read_write/attributes/deformable_material.html
- https://isaac-sim.github.io/IsaacLab/develop/source/api/lab_physx/isaaclab_physx.sim.spawners.html

Matsudaira, Zhang and Pan, Bending Vibrational Properties of Polyester Shingosen
Fabrics, Journal of the Textile Machinery Society of Japan 45(4), 101–107,
https://doi.org/10.4188/jte1955.45.101, tests 10 x 30 mm samples in a pendulum
apparatus with 1 mm initial amplitude. Figure 6 reports total decay durations
mostly around 2–8 seconds; the authors emphasize two approximately linear
friction-dominated regimes. These are clothing fabrics, not BaseMicro mop
material, and the apparatus inertia matters. This source does not measure or
calibrate the mop's PhysX coefficient.

An explicitly estimated constant bending rate of 1/s is therefore tested as an
order-of-magnitude approximation to internal textile dissipation, selected
before any task result. For a simple viscously damped modal oscillator that
coefficient would reduce amplitude by a factor of five in about 3.2 seconds.
That calculation motivates the trial, but is not assumed to describe the
native solver exactly. `material_decay.py` compares the two native rates on
free sheets without contact, gravity or scoring. No material value changes at
runtime or in response to contact, seating, phase or score. No air drag is added.

The baseline stays available. `--bend-damping-rate 1` is a diagnostic hypothesis,
not a pass. A useful revision must retain flexible motion and must pass fresh
insertion and stock-robot release checks with matched material parameters.
Stiffness, friction, mass, numerical acceptance limits and task scoring stay
unchanged. Hardware calibration and a full hysteresis model remain unavailable.
