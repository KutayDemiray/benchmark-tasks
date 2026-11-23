# Reproduction

## Environment and paths

Use the installed native YAM/Arena environment used for evaluation; no dependency
installation or environment change is part of this cleanup. The recorded
versions and critical installed-source hashes are in
[runtime-versions.json](runtime-versions.json). Principal versions are Python
3.12.16, Isaac Sim 6.1.0.0, isaaclab 17.0.2, isaaclab_arena 0.3.0, yam-arena 0.1.0,
inspect-robots 1.0.0, inspect-robots-dream 0.1.0 and dreamlabs-client 1.1.7.
The Dream integration source commit is
`b5e936a201cec66c54c297e522c920d98445e339`.
These are recorded installed versions, not a portable pip lockfile.

The runtime also requires the stock YAM assets/calibration, working native Kit
PhysX GPU support, `ffmpeg`, and the existing cooperative GPU scheduler with its
`isaac_binding.py` helper. These external installations are not vendored in this
task. Preserve their versions and source hashes when reproducing the evidence;
a checkout of this folder alone does not install a simulator or the scheduler.

From the benchmark repository/worktree root, set paths once in the current shell:

```bash
TASK="$(pwd)/tasks/pocket-mop-cover"
export POCKET_MOP_OUTPUT_ROOT="${POCKET_MOP_OUTPUT_ROOT:-$HOME/yam-eval/pocket-mop-cover}"
export POCKET_MOP_ENV_FILE="${POCKET_MOP_ENV_FILE:-$HOME/IsaacLab-Arena/yam.env}"
export ROBOCURVE_SCHEDULER_ROOT="${ROBOCURVE_SCHEDULER_ROOT:-$HOME/robocurve-gpu-scheduler}"
export ROBOCURVE_GATE_DIRECTORY="${ROBOCURVE_GATE_DIRECTORY:-$HOME/yam-eval}"
source "$POCKET_MOP_ENV_FILE"
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
OUT="$POCKET_MOP_OUTPUT_ROOT"
SCHEDULER="$ROBOCURVE_SCHEDULER_ROOT/gpu-scheduler"
mkdir -p "$OUT"
```

`task_paths.py` resolves these variables before each process starts. All task
output writers retain their ownership checks against `POCKET_MOP_OUTPUT_ROOT`.
The output default still selects the existing evidence on the original machine.
The GPU gate directory is independent of the task output directory and **must
match the existing scheduler's shared gate directory**. Never create a second
scheduler runtime or alternate locks to bypass reservations. On the original
machine, keep the gate and scheduler defaults. No existing jobs or reservations
need to be changed. GPU assignment remains scheduler-controlled, including
verified membership, UUID masking, `cuda:0` physics, explicit physical rendering,
disabled multi-GPU rendering and verification after warmup.

## CPU and command checks

These commands do not launch a simulator or contact Dream:

```bash
python -B "$TASK/launch.py" --help
python -B -m unittest discover -s "$TASK" -p 'test_*.py' -v
```

For an isolated CPU check, point `POCKET_MOP_OUTPUT_ROOT` at a new temporary
directory and create it before running the suite. Restore the evaluation output
root afterward. The tests generate their own fixtures and do not need old runs.

## Regenerate isolated qualifications

For a new evidence directory, run these four isolated simulations and their
assessors before the full task. Choose fresh run names if any already exist.
Commands below run in the foreground; finish each simulation before assessing
it. Exclusive scheduling preserves meaningful physics comparisons.

```bash
"$SCHEDULER" launch --task pocket-mop-cover --mode exclusive --gpu auto --wait 300 -- \
  python -B "$TASK/launch.py" --name cycle-new --product-profile base_micro \
  --blade-profile beveled --mode cycle --physics-hz 480 --iterations 64 --seconds 28 \
  --spacing .0025 --contact-updates 8 --contact-subiterations 2 --ccd
python -B "$TASK/analyze.py" "$OUT/cycle-new"
python -B "$TASK/diagnostics.py" "$OUT/cycle-new"

"$SCHEDULER" launch --task pocket-mop-cover --mode exclusive --gpu auto --wait 300 -- \
  python -B "$TASK/launch.py" --name grasp-new --experiment robot_grasp \
  --product-profile base_micro --placement table --table-origin --spacing .0025 \
  --grasp-margin-clearance .006 --grasp-opening .008 --close-seconds 3 \
  --release-opening .25 --physics-hz 480 --iterations 64 --seconds 40 \
  --contact-updates 8 --contact-subiterations 2 --ccd
python -B "$TASK/audit_grasp.py" "$OUT/grasp-new"
python -B "$TASK/qualify.py" --cycle cycle-new --grasp grasp-new

"$SCHEDULER" launch --task pocket-mop-cover --mode exclusive --gpu auto --wait 300 -- \
  python -B "$TASK/launch.py" --name frame-new --experiment mechanism \
  --product-profile base_micro --physics-hz 480 --iterations 64 --seconds 36 \
  --press-overtravel .03 --snap-throat --retention-extension 12 \
  --mechanism-unlock --unlock-lift .04 --pedal-overtravel .01
python -B "$TASK/audit_mechanism.py" "$OUT/frame-new"

"$SCHEDULER" launch --task pocket-mop-cover --mode exclusive --gpu auto --wait 300 -- \
  python -B "$TASK/launch.py" --name frame-control-new --experiment mechanism \
  --product-profile base_micro --physics-hz 480 --iterations 64 --seconds 24 \
  --press-overtravel .03 --snap-throat --retention-extension 12 --disable-latch-contact
python -B "$TASK/audit_mechanism.py" "$OUT/frame-control-new"
python -B "$TASK/qualify_mechanism.py" --positive frame-new --negative frame-control-new
```

Both `$OUT/product-gate.json` and `$OUT/mechanism-gate.json` must report `passed`.
They reference their supporting recordings, assessments and source hashes in the
same output tree. Copying a gate JSON alone does not reconstruct qualification.
The submitted [qualification-evidence.json](qualification-evidence.json) preserves
the original passing gate reports and artifact hashes for review; the commands
above regenerate gates rather than silently assuming access to old run folders.

## Full-task execution and assessment

Follow [DREAM_EVALUATION.md](DREAM_EVALUATION.md) for the exact R168–R171 command
arguments, independent audits, replay and media preparation. Seed 0 is fixed;
reset reconstructs the declared geometry and all robot/cloth/frame state. The
budget is 2,400 ordinary 14D targets at 20 Hz with 480 Hz physics. The successful
script and replay stop after 1,865 actions; the model used all 2,400.

For a new reproduction, choose new names consistently for the demo, interface,
replay and their evidence arguments. Use the source archived by that new launch
for independent scoring. The historical recordings keep their original source
and manifests; they must not be overwritten with cleaned source. Their archived
scripts retain the original host paths, so exact archival re-assessment uses the
original runtime/output location. The cleaned code supports relocated *new*
runs. Cleanup changes source hashes: hosted preflight correctly refuses to mix
fresh source with old demonstration/interface bundles. A new hosted run would
require matching fresh evidence and explicit authorization; the completed model
attempt does not need repeating for submission.

Raw evidence stays outside Git. Each run contains `request.json`, source and
manifest, native state chunks, actions/measured observations, reset and GPU
verification, camera recordings and independent scores. The active result is
`handle-held-benchmark-result.json`; `benchmark-result.json` preserves the
released-v1 assessment. Model logs and observations preserve actual inference
and termination. Media can be regenerated from the archived matching sources;
existing final media and their hashes are retained.
