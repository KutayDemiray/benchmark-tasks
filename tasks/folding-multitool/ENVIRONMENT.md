# Evaluated environment

Use the already installed native `yam.env`; do not run a new dependency install
or change shared configuration to reproduce this package. The task relies on
`yam_arena` from the native IsaacLab-Arena checkout, stock YAM assets and the
cooperative GPU scheduler with `isaac_binding.assigned_gpu`, `kit_args`, `verify`.
These external runtime components are not vendored here. Their original source
hashes and stock URDF hash are preserved in run evidence. Equivalent installation
is required on a different workstation; this is not a standalone pip package.

| Component | Installed version |
| --- | --- |
| Python | 3.12.16 |
| Isaac Sim | 6.1.0.0 |
| Isaac Lab | 17.0.2 |
| IsaacLab Arena | 0.3.0 |
| inspect-robots | 1.0.0 |
| inspect-robots-dream | 0.1.0 |
| dreamlabs-client | 1.1.7 |
| torch | 2.11.0+cu128 |
| numpy | 2.3.1 |
| scipy | 1.17.0 |
| trimesh | 4.11.1 |
| imageio | 2.37.2 |
| imageio-ffmpeg | 0.6.0 |
| Pillow | 12.3.0 |

CPU geometry/tests use numpy/scipy/trimesh and native YAM kinematics; media
checks use the already installed imageio ffmpeg binary (system ffprobe is not
required). Hosted evaluation uses the installed Dream integration and local
credentials. No credentials, headers or tokens belong in this repository.

The runtime is native GPU PhysX at 240 Hz with 20 Hz actions. Preserve stock
geometry, limits, gains and the 14D action order. Scheduler ID is always
`folding-multitool`; GPU reservation and verification are operational coordination,
not task physics or scoring. The scheduler root may be supplied through
`ROBOCURVE_SCHEDULER_ROOT`. Output directories are explicit `--out` arguments or
the existing home-relative `yam-eval/folding-multitool/runs` default.
