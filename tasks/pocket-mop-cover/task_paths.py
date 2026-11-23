"""Resolve task-owned output and shared runtime paths without changing physics."""

import os
from pathlib import Path


def configured_path(variable, default):
    """Resolve an optional environment override before a process starts."""
    value = os.environ.get(variable)
    assert value is None or value.strip(), f"{variable} must not be empty"
    return Path(value if value is not None else default).expanduser().resolve()


OUTPUT_ROOT = configured_path('POCKET_MOP_OUTPUT_ROOT', Path.home() / 'yam-eval/pocket-mop-cover')
SCHEDULER_ROOT = configured_path('ROBOCURVE_SCHEDULER_ROOT', Path.home() / 'robocurve-gpu-scheduler')
# Shared with other tasks: never derive the gate directory from OUTPUT_ROOT.
GPU_GATE_DIRECTORY = configured_path('ROBOCURVE_GATE_DIRECTORY', Path.home() / 'yam-eval')
ENV_FILE = configured_path('POCKET_MOP_ENV_FILE', Path.home() / 'IsaacLab-Arena/yam.env')
