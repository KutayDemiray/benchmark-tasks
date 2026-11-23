"""Run two reciprocal native-default comparisons sequentially in one GPU reservation."""

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
COMMON = ['--experiment', 'passive_contact', '--passive-source', 'release-r124j-hands-clear',
          '--product-profile', 'base_micro', '--physics-hz', '480', '--iterations', '64',
          '--spacing', '.0025', '--contact-updates', '8', '--contact-subiterations', '2', '--ccd']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prefix', required=True)
    parser.add_argument('--continue-from-captures', action='store_true', help='Verify existing completed short captures and run only stable continuations')
    args = parser.parse_args()
    assert Path(args.prefix).name == args.prefix and args.prefix.startswith('passive-')
    require_reservation()
    assert os.environ.get('ROBOCURVE_GPU_MODE') == 'exclusive', 'Numerical comparisons require scheduled exclusive admission'
    descriptors = tuple(int(fd) for fd in os.environ['ROBOCURVE_LOCK_FDS'].split(','))
    configurations = [('sdk-bounded', ['--passive-sdk-scene', 'passive-r153a-scheduled-baseline',
                                     '--passive-cloth-correction', 'bounded'], 'passive-r158a-sdk-properties'),
                      ('kit-unbounded', ['--passive-cloth-correction', 'unbounded'], 'passive-r158b-kit-properties')]
    names = [args.prefix+'-'+kind+'-'+duration for duration in ('capture','4s') for kind,_,_ in configurations]
    assert not any((OUT/name).exists() for name in names if name.endswith('-4s')), 'Longer recordings need fresh output names'
    if not args.continue_from_captures:
        assert not any((OUT/name).exists() for name in names), 'Each comparison needs fresh output names'
    from read_native_properties import read_capture
    from compare_native_properties import compare
    for kind, options, baseline in configurations:
        run = OUT/(args.prefix+'-'+kind+'-capture')
        if not args.continue_from_captures:
            subprocess.run([sys.executable, '-B', str(HERE/'launch.py'), '--name', run.name,
                            *COMMON, *options, '--passive-pvd-record', '--seconds', '.25'], check=True, pass_fds=descriptors)
        request = json.loads((run/'request.json').read_text())
        expected = dict(experiment='passive_contact', passive_source='release-r124j-hands-clear',
                        passive_sdk_scene='passive-r153a-scheduled-baseline' if kind=='sdk-bounded' else None,
                        passive_cloth_correction='bounded' if kind=='sdk-bounded' else 'unbounded',
                        physics_hz=480, iterations=64, spacing=.0025, contact_updates=8,
                        contact_subiterations=2, ccd=True, seconds=.25, passive_pvd_record=True)
        assert all(request.get(key)==value for key,value in expected.items())
        assert json.loads((run/'result.json').read_text())['status']=='completed'
        assert json.loads((run/'process.json').read_text())['returncode']==0
        assert json.loads((run/'gpu-verification.json').read_text())['verified']
        captures = list((run/'omnipvd').glob('*_rec.ovd'))
        assert len(captures) == 1, 'A finalized native property capture is required'
        data = read_capture(captures[0])
        (run/'native-properties.json').write_text(json.dumps(data, indent=2))
        report = compare(OUT/baseline, run)
        differences = [row for row in report['properties'] if row['same'] is False]
        assert len(differences) == 1, 'Reciprocal diagnostic changed additional published properties'
        row = differences[0]
        assert row['object'] == '/World/Pocket' and row['property'] == 'maxDepenetrationVelocity'
        expected_values = (3.,) if kind=='sdk-bounded' else ('inf', 3.4028234663852886e38)
        assert row['second'] in expected_values, 'Unexpected native correction limit'
        # Kit represents an authored infinity by the largest finite float32.
        # Both documented values are effectively unbounded for this experiment.
    for kind, options, _ in configurations:
        capture = OUT/(args.prefix+'-'+kind+'-capture')
        result = json.loads((capture/'result.json').read_text())
        if result['maximum_head_inclination_deg'] > 3.:
            (capture/'continuation-skipped.json').write_text(json.dumps(dict(
                diagnostic_only=True, reason='The short control already tips beyond 3 degrees; no additional settling test justified.',
                maximum_head_inclination_deg=result['maximum_head_inclination_deg']), indent=2))
            continue
        run = OUT/(args.prefix+'-'+kind+'-4s')
        subprocess.run([sys.executable, '-B', str(HERE/'launch.py'), '--name', run.name,
                        *COMMON, *options, '--seconds', '4'], check=True, pass_fds=descriptors)
    print('Native-default controls checked and stable continuations recorded; no ordinary task score claimed.', flush=True)


if __name__ == '__main__':
    main()
