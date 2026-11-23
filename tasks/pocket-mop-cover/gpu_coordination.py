"""Share verified scheduler reservations, with an exclusive unscheduled fallback."""

from task_paths import GPU_GATE_DIRECTORY, SCHEDULER_ROOT

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

TASK_ID = 'pocket-mop-cover'
SCHEDULER = SCHEDULER_ROOT
GATE_DIRECTORY = GPU_GATE_DIRECTORY


def scheduler_binding():
    """Load the user-authorized scheduler helper from its canonical directory."""
    if str(SCHEDULER) not in sys.path:
        sys.path.insert(0, str(SCHEDULER))
    import isaac_binding
    assert Path(isaac_binding.__file__).resolve() == SCHEDULER/'isaac_binding.py'
    return isaac_binding


def scheduled_assignment():
    """Reject invalid membership instead of falling back to an unlocked job."""
    assert os.environ['ROBOCURVE_TASK_ID'] == TASK_ID, 'Use the stable pocket-mop-cover task ID'
    gpu = scheduler_binding().assigned_gpu()
    assert os.environ['ROBOCURVE_CUDA_DEVICE'] == 'cuda:0'
    assert os.environ['ROBOCURVE_GPU_MODE'] in ('dev', 'exclusive')
    return gpu


def command(args):
    return subprocess.check_output(args, text=True, timeout=5).strip()


def snapshot():
    return dict(
        gpus=command(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,memory.used,utilization.gpu', '--format=csv,noheader,nounits']),
        compute=command(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory', '--format=csv,noheader,nounits']),
        pmon=command(['nvidia-smi', 'pmon', '-c', '1', '-s', 'um']),
        processes=command(['ps', '-eo', 'user,pid,ppid,etimes,comm']))


def fallback_assignment(state):
    """Resolve an unscheduled physical device; GPU 1 remains the fallback preference."""
    index = int(os.environ.get('ROBOCURVE_GPU_INDEX', '1'))
    rows = [[part.strip() for part in row.split(',')] for row in state['gpus'].splitlines()]
    matches = [row for row in rows if int(row[0]) == index]
    assert len(matches) == 1, 'Requested fallback GPU does not exist'
    row = matches[0]
    assert os.environ.get('ROBOCURVE_GPU_UUID', row[1]) == row[1], 'Fallback GPU UUID mismatch'
    return dict(index=index, uuid=row[1], pci_bus_id=row[2])


def ensure_idle(state, gpu):
    """Require exclusive fallback access and no uncoordinated GPU workload."""
    rows = [[part.strip() for part in row.split(',')] for row in state['gpus'].splitlines()]
    row = next(row for row in rows if int(row[0]) == gpu['index'])
    assert row[1] == gpu['uuid'], 'Physical GPU mapping changed'
    assert gpu['uuid'] not in state['compute'], 'Assigned GPU has an uncoordinated compute job'
    for line in state['pmon'].splitlines():
        fields = line.split()
        if fields and fields[0] == str(gpu['index']) and fields[1] != '-':
            executable = command(['ps', '-p', fields[1], '-o', 'comm='])
            if executable == 'Xorg' and fields[2] == 'G':
                continue
            raise RuntimeError('Assigned GPU has an uncoordinated graphics/compute process: '+line)
    assert int(row[3]) < 200 and int(row[4]) < 5, 'Assigned GPU is not idle'


@contextmanager
def reservation(out):
    """Yield assignment and inherited descriptors without reacquiring scheduled gates."""
    lock = None
    try:
        if 'ROBOCURVE_JOB_ID' in os.environ:
            gpu = scheduled_assignment()
            descriptors = tuple(int(fd) for fd in os.environ['ROBOCURVE_LOCK_FDS'].split(','))
            assert descriptors
            for fd in descriptors:
                os.fstat(fd)
            state = snapshot()
            mode = os.environ['ROBOCURVE_GPU_MODE']
        else:
            state = snapshot()
            gpu = fallback_assignment(state)
            gate = GATE_DIRECTORY/f'.robocurve-gpu-{gpu["index"]}.lock'
            lock = gate.open('a+')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            descriptors = (lock.fileno(),)
            mode = 'legacy-exclusive'
            observations = []
            for attempt in range(5):
                state = snapshot()
                try:
                    ensure_idle(state, gpu)
                    break
                except AssertionError as error:
                    if str(error) != 'Assigned GPU is not idle' or attempt == 4:
                        raise
                    observations.append(state)
                    time.sleep(2)
            if observations:
                (out/'idle-observations.json').write_text(json.dumps(observations, indent=2))
        (out/'before-gpu.json').write_text(json.dumps(state, indent=2))
        record = dict(task_id=TASK_ID, job_id=os.environ.get('ROBOCURVE_JOB_ID'), mode=mode, gpu=gpu,
            scheduler_membership_verified=lock is None, acquired_legacy_gate=lock is not None)
        if lock is None:
            record['binding_helper_sha256'] = hashlib.sha256((SCHEDULER/'isaac_binding.py').read_bytes()).hexdigest()
        (out/'gpu-reservation.json').write_text(json.dumps(record, indent=2))
        yield gpu, descriptors, lock is not None
    finally:
        if lock is not None:
            lock.close()


def require_reservation():
    """Verify each simulator descendant before creating any GPU runtime."""
    if 'ROBOCURVE_JOB_ID' in os.environ:
        return scheduled_assignment()
    index = int(os.environ['ROBOCURVE_GPU_INDEX'])
    fd = int(os.environ['POCKET_GPU_LOCK_FD'])
    gate = GATE_DIRECTORY/f'.robocurve-gpu-{index}.lock'
    stat, expected = os.fstat(fd), gate.stat()
    assert (stat.st_dev, stat.st_ino) == (expected.st_dev, expected.st_ino), 'Inherited gate inode mismatch'
    # Same inherited open-file description: retains, never unlocks, the exclusive gate.
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == os.environ['ROBOCURVE_GPU_UUID']
    return dict(index=index, uuid=os.environ['ROBOCURVE_GPU_UUID'])


def kit_arguments(out):
    """Bind masked physics and the physical renderer without changing task physics."""
    gpu = require_reservation()
    if 'ROBOCURVE_JOB_ID' in os.environ:
        return scheduler_binding().kit_args(out/'kit.log')
    return (f'--/renderer/activeGpu={gpu["index"]} --/renderer/multiGpu/activeCudaGpus=0, '
            '--/renderer/multiGpu/enabled=false --/physics/cudaDevice=0 '
            f'--/log/file={out}/kit.log --/app/settings/persistent=false --/app/settings/loadUserConfig=false')
