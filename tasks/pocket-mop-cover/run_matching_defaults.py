"""Verify published native default equivalence, then record the stable SDK control."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from gpu_coordination import require_reservation
from run_passive_default_comparison import COMMON, HERE, OUT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--settling-hold', action='store_true', help='Stop a fresh continuation at its first two-second native hold, bounded by16s')
    args = parser.parse_args()
    assert Path(args.prefix).name == args.prefix and args.prefix.startswith('passive-')
    require_reservation()
    assert os.environ.get('ROBOCURVE_GPU_MODE') == 'exclusive'
    descriptors = tuple(int(fd) for fd in os.environ['ROBOCURVE_LOCK_FDS'].split(','))
    capture, continuation = [OUT/(args.prefix+suffix) for suffix in ('-capture','-hold' if args.settling_hold else '-4s')]
    assert not capture.exists() and not continuation.exists()
    options = ['--passive-sdk-scene', 'passive-r153a-scheduled-baseline', '--passive-match-kit-defaults']
    def launch(run, more):
        subprocess.run([sys.executable, '-B', str(HERE/'launch.py'), '--name', run.name,
                        *COMMON, *options, *more], check=True, pass_fds=descriptors)
    launch(capture, ['--passive-pvd-record', '--seconds', '.25'])
    from read_native_properties import read_capture
    from compare_native_properties import compare
    paths = list((capture/'omnipvd').glob('*_rec.ovd'))
    assert len(paths) == 1
    data = read_capture(paths[0])
    (capture/'native-properties.json').write_text(json.dumps(data, indent=2))
    report = compare(OUT/'passive-r158b-kit-properties', capture)
    assert report['different'] == 0 and report['unrecorded'] == 0, 'Published native defaults still differ'
    result = json.loads((capture/'result.json').read_text())
    assert result['maximum_head_inclination_deg'] <= 3., 'Matched control already tips; no settling continuation justified'
    launch(continuation, ['--seconds', '16', '--passive-stop-when-settled'] if args.settling_hold else ['--seconds', '4'])
    print('Matched-default control complete; no task score claimed.', flush=True)


if __name__ == '__main__':
    main()
