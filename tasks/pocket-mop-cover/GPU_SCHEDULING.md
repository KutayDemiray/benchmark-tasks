# Cooperative GPU scheduling

The active R168–R171 evaluations retain verified scheduler membership, masked
`cuda:0` physics, explicit physical renderer selection and multi-GPU rendering
disabled. R171 records physical GPU 1. Runtime paths are now configured through
`task_paths.py`; see [REPRODUCTION.md](REPRODUCTION.md). Changing the task output
root never changes the shared gate directory. The measurements below record the
earlier scheduler integration; they are historical, not new cleanup GPU runs.

The launcher validates `isaac_binding.assigned_gpu()` and the stable task ID
`pocket-mop-cover` before bypassing its exclusive gate. Invalid membership fails
closed. Scheduler descriptors are inherited without locking or unlocking them;
the scheduler supervises the complete descendant tree. Unscheduled launches retain
an exclusive gate, real compute/graphics occupancy checks, and inherited inode
verification. Their default preference remains GPU 1, resolved to its current UUID.

All seven simulator entry points use `ROBOCURVE_GPU_INDEX` and `ROBOCURVE_GPU_UUID`.
Physics stays on masked `cuda:0`; RTX selects the physical index, and multi-GPU
rendering is disabled. Runtime verification reads a native physics tensor rather
than an asset configuration string, checks the native scene GPU setting, CUDA UUID,
Kit's active renderer row and this process's NVIDIA allocations. Host-readback
comparisons retain CPU tensors with GPU dynamics. Scheduler-provided cache paths
are preserved. No backend, timestep, contact, gain, material or score was changed.

## First native verification

Run `passive-r153a-scheduled-baseline` completed under scheduler job
`5c608596cd884a418888ef88e852693c` in exclusive mode. The assigned and observed
physical GPU was 1, UUID `GPU-cc9ca88f-3544-a5fd-4676-362e0d647c69`. Both simulation
and actual native link-transform tensors were `cuda:0`; the active renderer row
was `[1]`; multi-GPU was disabled. NVIDIA reported 6204 MiB C+G on the assignment
and only 8 MiB graphics enumeration on GPU 0. Every device check passed. The
scheduler confirmed descendant completion with `waitpid(-1)` returning `ECHILD`.

`gpu-reservation.json` proves verified membership and no legacy gate acquisition.
`gpu-verification.json` contains runtime device evidence and process identity.
`scheduler-physics-preservation.json` proves byte-identical scene, geometry,
initial frame state, material parameters, preserved shared source hashes and
installed package versions relative to the immediately preceding unscheduled
baseline. The frame still overturns about 143 degrees: scheduling integration
is verified, but it is not a physics correction or task success.

Exact executed command (choose a fresh run name when repeating):

```bash
source $HOME/IsaacLab-Arena/yam.env
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
$HOME/robocurve-gpu-scheduler/gpu-scheduler launch \
  --task pocket-mop-cover --mode exclusive --gpu auto --wait 300 -- \
  python -B $HOME/benchmark-tasks/tasks/pocket-mop-cover/launch.py \
  --name passive-r153a-scheduled-baseline --experiment passive_contact \
  --passive-source release-r124j-hands-clear --passive-mode articulated \
  --product-profile base_micro --physics-hz 480 --iterations 64 \
  --spacing .0025 --seconds 4 --contact-updates 8 --contact-subiterations 2 --ccd
```

Use `--mode dev` for ordinary development previews and `--mode exclusive` for
numerical/performance comparisons and qualification. Reuse the same task ID.
All artifacts above are under
`$HOME/yam-eval/pocket-mop-cover/passive-r153a-scheduled-baseline/`.

## Host readback verification

`passive-r153b-scheduled-host` completed under job
`1279e6b57c084b3f926de8786acac893`, also exclusive on physical GPU 1.
Native link-transform tensors remained `cpu`, while the simulation device was
`cuda:0` and native GPU dynamics remained enabled. RTX was physically on GPU 1;
NVIDIA reported 5468 MiB C+G there and only 8 MiB graphics enumeration on GPU 0.
All checks passed; the complete descendant tree exited. This verifies admission
and device selection without changing the existing host-readback mode.

The eight task-local scheduler tests cover invalid membership, stable task ID,
no scheduler-descriptor flock, exclusive fallback lifetime, replaced gate inode,
uncoordinated compute, actual native tensor readback and cross-device rejection.
The complete current suite passed 120 tests, including the standalone scene-gravity guard.
The log is `2026-11-22-scheduler-sdk-final-tests.log` in the task output directory.
