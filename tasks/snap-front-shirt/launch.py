"""Launch a complete native stage tree under a verified GPU reservation."""

import argparse
import ctypes
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import subprocess
import shutil
import sys

HERE = Path(__file__).resolve().parent
from task_paths import OUTPUT_ROOT as OUTPUT, GPU_LOCK as LOCK
from gpu_runtime import scheduled_assignment, inherited_fds, scheduler_record


def query(*args):
    return subprocess.check_output(['nvidia-smi', *args], text=True).strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--stage', choices=('bench', 'single_snap', 'garment','tabletop','stability','evaluation','scheduler_smoke','dream_local_check','dream_live','camera_reset_check'), default='bench')
    parser.add_argument('--detach', action='store_true', help='Start a bounded local job in its own session; inspect exit.json')
    parser.add_argument('args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    out = args.out.resolve()
    assert out.is_relative_to(OUTPUT), 'Outputs must stay in the owned output folder'
    scheduled = scheduled_assignment()  # Validate ancestry before bypassing the legacy gate.
    if scheduled is not None:
        assert not args.detach, 'Use gpu-scheduler launch --detach around the complete launcher'
        if args.stage in ('stability','dream_local_check','dream_live','camera_reset_check'):
            assert os.environ['ROBOCURVE_GPU_MODE'] == 'exclusive', 'Stability comparisons require exclusive mode'
    if args.stage in ('dream_local_check','dream_live','camera_reset_check'):
        assert scheduled is not None, 'Dream evaluation requires the canonical scheduler'
    if args.detach:
        assert not out.exists(), 'Use a fresh output path'
        control_dir = OUTPUT/'job-control'
        control_dir.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, '-B', str(Path(__file__).resolve()),
                   '--out', str(out), '--stage', args.stage, *args.args]
        log_path = control_dir/(out.name+'-launcher.log')
        with log_path.open('x') as log:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        record = {'launcher_pid': process.pid, 'command': command, 'launcher_log': str(log_path),
                  'output': str(out), 'status': 'started; not a simulation success result'}
        (control_dir/(out.name+'.json')).write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record), flush=True)
        return 0
    out.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        if scheduled is not None:
            descriptors = inherited_fds()
            physical = scheduled['index']
        else:
            lock = stack.enter_context(LOCK.open('a+'))
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                (out/'exit.json').write_text(json.dumps({'returncode': 75, 'reason': 'gpu_lock_occupied',
                    'simulation_started': False})+'\n')
                print('GPU 0 lock occupied; no simulation started.', flush=True)
                return 75
            descriptors, physical = (lock.fileno(),), 0
        gpu = query(f'--id={physical}', '--query-gpu=uuid,pci.bus_id,name,memory.used,utilization.gpu', '--format=csv,noheader')
        uuid = gpu.split(',')[0].strip()
        if scheduled is not None:
            assert uuid == scheduled['uuid'], 'NVIDIA inventory differs from scheduler assignment'
        apps = query('--query-compute-apps=pid,gpu_uuid,process_name,used_memory', '--format=csv,noheader')
        processes = subprocess.check_output(['ps', '-eo', 'user,pid,ppid,comm'], text=True)
        audit = {'gpu': gpu, 'physical_gpu_index': physical, 'compute_processes': apps, 'processes': processes,
                 'graphics_and_compute_table': query(), 'scheduler': scheduler_record(),
                 'lock': str(LOCK) if scheduled is None else None}
        (out / 'preflight.json').write_text(json.dumps(audit, indent=2) + '\n')
        # Scheduler admission checks unknown jobs and cooperative peers. Only the
        # unscheduled fallback needs the legacy all-occupants exclusion test here.
        if scheduled is None and (uuid in apps or int(gpu.split(',')[-2].strip().split()[0]) > 500
                                  or int(gpu.split(',')[-1].strip().split()[0]) > 5):
            (out/'exit.json').write_text(json.dumps({'returncode': 75, 'reason': 'gpu_occupied',
                'simulation_started': False})+'\n')
            print('GPU 0 is occupied despite the free lock; no simulation started.', flush=True)
            return 75
        runtime = out / 'runtime'
        runtime.mkdir()
        env = dict(os.environ)
        env.update(CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES=uuid,
                   PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
                   XDG_CACHE_HOME=str(OUTPUT / 'runtime-cache'), XDG_CONFIG_HOME=str(runtime / 'config'),
                   XDG_DATA_HOME=str(runtime / 'data'), TMPDIR=str(runtime / 'tmp'),
                   WARP_CACHE_PATH=str(runtime / 'warp'), HF_HUB_OFFLINE='1',
                   TRANSFORMERS_OFFLINE='1', SNAP_GPU_UUID=uuid, SNAP_GPU_INDEX=str(physical))
        if scheduled is None:
            env['SNAP_LOCK_FD'] = str(descriptors[0])
        else:
            env.pop('SNAP_LOCK_FD', None)
        for key, name in {'XDG_STATE_HOME': 'state', 'CUDA_CACHE_PATH': 'cuda-cache',
                          'NUMBA_CACHE_DIR': 'numba', 'TRITON_CACHE_DIR': 'triton',
                          'MPLCONFIGDIR': 'matplotlib'}.items():
            env[key] = str(runtime / name)
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        for key in ('XDG_CACHE_HOME', 'XDG_CONFIG_HOME', 'XDG_DATA_HOME', 'TMPDIR', 'WARP_CACHE_PATH'):
            Path(env[key]).mkdir(parents=True, exist_ok=True)
        frozen = out / 'task-bundle'
        frozen.mkdir()
        for source in sorted(HERE.iterdir()):
            if source.is_file():
                shutil.copy2(source, frozen / source.name)
        entry = {'bench': 'bench.py', 'single_snap': 'robot_runner.py', 'garment': 'garment_runner.py','tabletop':'tabletop_runner.py','stability':'stability_round.py','evaluation':'evaluation_runner.py','scheduler_smoke':'scheduler_smoke.py','dream_local_check':'dream_local_check.py','dream_live':'dream_live_runner.py','camera_reset_check':'camera_reset_probe.py'}[args.stage]
        command = [sys.executable, '-B', str(frozen / entry), '--out', str(out)]
        command.extend(args.args[1:] if args.args[:1] == ['--'] else args.args)
        with (out / 'console.log').open('w') as log:
            # Adopt descendants so the lock survives their full lifetime as well.
            assert ctypes.CDLL(None).prctl(36, 1, 0, 0, 0) == 0
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       pass_fds=descriptors, cwd=frozen)
            print(json.dumps({'pid': process.pid, 'log': str(out / 'console.log'), 'gpu_uuid': uuid, 'physical_gpu_index': physical,
                              'scheduler_job_id': env.get('ROBOCURVE_JOB_ID')}), flush=True)
            code = process.wait()
            while True:
                try:
                    os.waitpid(-1, 0)
                except ChildProcessError:
                    break
        if ((out / 'failure.json').exists() or not (out / 'results.json').exists()) and code == 0:
            code = 1  # SimulationApp shutdown can mask an active Python exception.
        (out / 'exit.json').write_text(json.dumps({'returncode': code, 'command': command}) + '\n')
        return code


if __name__ == '__main__':
    sys.exit(main())
