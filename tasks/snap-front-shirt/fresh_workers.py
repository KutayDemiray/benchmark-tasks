"""Run native episodes in fresh workers under the launcher's inherited GPU reservation."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
from task_paths import OUTPUT_ROOT as OWNED_OUTPUT
from gpu_runtime import inherited_fds


def inherited_lock():
    """Validate and return all scheduler or legacy reservation descriptors."""
    return inherited_fds()


def prepare_worker(out, parent):
    """Freeze source and allocate all worker writes inside an owned run."""
    out, parent = Path(out).resolve(), Path(parent).resolve()
    assert out.is_relative_to(OWNED_OUTPUT) and out.is_relative_to(parent)
    descriptors = inherited_lock()
    out.mkdir(parents=True, exist_ok=False)
    runtime = out / 'runtime'
    runtime.mkdir()
    bundle = out / 'task-bundle'
    bundle.mkdir()
    for source in HERE.iterdir():
        if source.is_file():
            shutil.copy2(source, bundle / source.name)
    shutil.copy2(parent / 'preflight.json', out / 'preflight.json')
    env = dict(os.environ)
    env.update(XDG_CONFIG_HOME=str(runtime/'config'), XDG_DATA_HOME=str(runtime/'data'),
               TMPDIR=str(runtime/'tmp'), WARP_CACHE_PATH=str(runtime/'warp'),
               PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    for key in ('XDG_CONFIG_HOME','XDG_DATA_HOME','TMPDIR','WARP_CACHE_PATH'):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    return bundle, env, descriptors


def run_worker(out, parent, arguments):
    """Wait for one fresh process and preserve failures without retrying."""
    bundle, env, descriptors = prepare_worker(out, parent)
    command = [sys.executable, '-B', str(bundle/'tabletop_runner.py'), '--out', str(out), *arguments]
    with (out/'console.log').open('x') as log:
        process = subprocess.Popen(command, env=env, cwd=bundle, stdout=log,
                                   stderr=subprocess.STDOUT, pass_fds=descriptors)
        (out/'worker.json').write_text(json.dumps(dict(coordinator_pid=os.getpid(),
            worker_pid=process.pid,command=command,reset_mode='fresh_native_process'),indent=2)+'\n')
        code = process.wait()
    if code == 0 and (not (out/'results.json').exists() or (out/'failure.json').exists()):
        code = 1
    result = dict(returncode=code,worker_pid=process.pid,coordinator_pid=os.getpid())
    (out/'exit.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
