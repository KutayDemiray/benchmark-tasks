"""Run isolated SDK cloth and stock-robot checks under one scheduler reservation."""

from task_paths import OUTPUT_ROOT

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from gpu_coordination import require_reservation

HERE = Path(__file__).resolve().parent
OUT = OUTPUT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cycle', required=True)
    parser.add_argument('--reference', required=True)
    parser.add_argument('--grasp', required=True)
    args = parser.parse_args()
    require_reservation()
    assert os.environ.get('ROBOCURVE_GPU_MODE') == 'exclusive'
    descriptors = tuple(int(fd) for fd in os.environ['ROBOCURVE_LOCK_FDS'].split(','))
    for name in (args.cycle, args.reference, args.grasp):
        assert Path(name).name == name and name not in ('.', '..')
        assert not (OUT/name).exists(), 'Refuse to overwrite a prior trial'
    assert json.loads((OUT/'mechanism-sdk-gate.json').read_text())['passed']
    common = ['--product-profile', 'base_micro', '--physics-hz', '480', '--iterations', '64',
              '--spacing', '.0025', '--contact-updates', '8', '--contact-subiterations', '2', '--ccd']
    grasp = ['--experiment', 'robot_grasp', '--seconds', '40', '--table-origin', '--placement', 'table',
             '--grasp-margin-clearance', '.006', '--grasp-opening', '.008', '--close-seconds', '3']
    cases = [
        (args.cycle, ['--experiment', 'probe', '--probe-sdk-source', 'base-r85-fine-contact-cycle',
                      '--mode', 'cycle', '--blade-profile', 'beveled', '--seconds', '28']),
        (args.reference, [*grasp, '--cooked-collider-probe', '--state-run', 'base-r84-clear-margin-table']),
        (args.grasp, [*grasp, '--grasp-sdk-source', 'base-r84-clear-margin-table',
                      '--grasp-sdk-reference', args.reference]),
    ]
    for name, options in cases:
        subprocess.run([sys.executable, '-B', str(HERE/'launch.py'), '--name', name, *common, *options],
                       check=True, pass_fds=descriptors)
        result = json.loads((OUT/name/'result.json').read_text())
        assert result['status'] == 'completed'
        print(json.dumps(dict(run=name, result=result)), flush=True)
    print('Isolated recordings completed; complete independent audits and camera checks remain required.', flush=True)


if __name__ == '__main__':
    main()
