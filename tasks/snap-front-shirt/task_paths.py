"""Configure private outputs and the existing shared scheduler without host-specific paths."""

import os
from pathlib import Path


def configured_path(variable, default):
    """Resolve an explicit nonempty path without creating or changing it."""
    value = os.environ.get(variable, str(default))
    assert value.strip(), f'{variable} must not be empty'
    return Path(value).expanduser().resolve()


OUTPUT_ROOT = configured_path('SNAP_OUTPUT_ROOT', Path.home() / 'yam-eval' / 'snap-front-shirt')
GPU_LOCK = configured_path('SNAP_GPU_LOCK', OUTPUT_ROOT.parent / '.robocurve-gpu-0.lock')
SCHEDULER_ROOT = configured_path('ROBOCURVE_SCHEDULER_ROOT', Path.home() / 'robocurve-gpu-scheduler')
SCHEDULER_COMMAND = str(SCHEDULER_ROOT / 'gpu-scheduler')
SOURCE_CHECKOUT_ROOT = configured_path('SNAP_SOURCE_CHECKOUT_ROOT', Path.home())
