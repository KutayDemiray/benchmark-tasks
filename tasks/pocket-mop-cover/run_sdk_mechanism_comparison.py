"""Run the original latch positive and negative controls sequentially in the SDK."""

from task_paths import OUTPUT_ROOT

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from gpu_coordination import require_reservation

HERE=Path(__file__).resolve().parent
OUT=OUTPUT_ROOT


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--prefix',required=True)
    args=parser.parse_args();assert Path(args.prefix).name==args.prefix and args.prefix.startswith('frame-')
    require_reservation();assert os.environ.get('ROBOCURVE_GPU_MODE')=='exclusive'
    descriptors=tuple(int(fd) for fd in os.environ['ROBOCURVE_LOCK_FDS'].split(','))
    assert json.loads((OUT/'passive-r161-matched-hold'/'passive-analysis.json').read_text())['final_two_seconds']['speed_threshold_passed']
    comparison=json.loads((OUT/'passive-r161-matched-capture'/'native-property-comparison.json').read_text())
    assert comparison['matched']==258 and comparison['different']==comparison['unrecorded']==0
    for suffix in ('positive','negative'):assert not (OUT/(args.prefix+'-'+suffix)).exists()
    common=['--experiment','mechanism','--product-profile','base_micro','--physics-hz','480',
            '--iterations','64','--snap-throat','--press-overtravel','.03','--retention-extension','12']
    cases=[('positive','frame-r77b-restarted-unlock',['--seconds','36','--mechanism-unlock','--pedal-overtravel','.01','--unlock-lift','.04']),
           ('negative','frame-r74-snap-throat-control',['--seconds','24','--disable-latch-contact'])]
    for suffix,source,more in cases:
        name=args.prefix+'-'+suffix
        subprocess.run([sys.executable,'-B',str(HERE/'launch.py'),'--name',name,
                        '--mechanism-sdk-source',source,*common,*more],check=True,pass_fds=descriptors)
        result=json.loads((OUT/name/'result.json').read_text())
        print(json.dumps(dict(run=name,result=result)),flush=True)
        if suffix=='positive' and not (result['passive_retention_observed'] and result['physical_unlock_observed']):
            print('Positive physical latch control failed; no negative continuation justified.',flush=True)
            return
    print('Both native latch controls recorded; independent contact audits still required.',flush=True)


if __name__=='__main__':main()
