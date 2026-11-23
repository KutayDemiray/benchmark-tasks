# Recorded native dependencies

These are the versions recorded by the genuine model worker. This is an
existing-runtime record, not an instruction to upgrade or reinstall packages.
The task was validated natively; Docker migration was not performed.

| Component | Recorded version |
| --- | --- |
| isaacsim | 6.1.0.0 |
| isaaclab | 17.0.2 |
| isaaclab-arena | 0.3.0 |
| yam-arena | 0.1.0 |
| numpy | 2.3.1 |
| torch | 2.11.0+cu128 |
| warp-lang | 1.16.0 |
| inspect-robots | 1.0.0 |
| inspect-robots-yam | 0.38.0 |
| inspect-robots-dream | 0.1.0 |
| dreamlabs-client | 1.1.7 |
| imageio | 2.37.2 |
| imageio-ffmpeg | 0.6.0 |
| scipy | 1.17.0 |
| python | 3.12.16 (main, Nov 11 2026, 00:15:59) [Clang 23.1.3 ] |

The `inspect-robots-dream` checkout commit was
`b5e936a201cec66c54c297e522c920d98445e339`. Arena checkout revision was
`c7edf3a9d5d8276565208f72422f8f8a0806d6cf`; YAM checkout revision was
`bd62771e60320433893bd1d2df0fc79b5ddcd6d1`. The recorded environment included an
existing Arena asset-resolution customization; no installed package or shared
configuration was changed by this task. Stock YAM robot/action source hashes
are preserved in `submission-evidence.json`.

GPU execution needs the existing NVIDIA driver and compatible CUDA/RTX runtime.
The cooperative scheduler is a shared host dependency, selected with
`ROBOCURVE_SCHEDULER_ROOT`; it is not copied or replaced by this task. Its source
hashes and actual GPU binding are recorded in each scheduled run. The stable ID
is `snap-front-shirt` for every stage. Standalone non-Dream stages retain the
exclusive GPU-0 fallback with explicit occupancy and device checks.

CPU audits use NumPy, Pillow, imageio/FFmpeg, and the installed Inspect integration
for exact image/state encoding checks. Historical plot helpers additionally use
Matplotlib; geometric investigations use SciPy. The current scorer and metadata
validator do not make hosted calls. `validate_submission.py` uses only the
Python standard library and can run without the native runtime.
