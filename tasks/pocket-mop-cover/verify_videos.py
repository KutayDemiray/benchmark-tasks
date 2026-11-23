"""Decode every frame of all three diagnostic camera recordings."""

from task_paths import OUTPUT_ROOT
import argparse
import json
from pathlib import Path
import subprocess


def verify(run):
    rows=[json.loads(line) for line in (run/'steps.jsonl').read_text().splitlines()]
    holds=[r['time_s'] for r in rows if r['phase'] in ('hold_grasp','hold_flat_frame')]
    instant=holds[len(holds)//2] if holds else rows[-1]['time_s']/2
    videos={}
    for camera in ('top_cam','left_cam','right_cam'):
        path=run/(camera+'.mp4')
        check=subprocess.run(['ffmpeg','-v','error','-i',str(path),'-map','0:v:0',
            '-f','null','-','-progress','pipe:1','-nostats'],capture_output=True,text=True)
        progress={}
        for line in check.stdout.splitlines():
            if '=' in line:
                key,value=line.split('=',1);progress[key]=value
        count=int(progress.get('frame',0))
        videos[camera]=dict(decoded_frames=count,expected_frames=len(rows),returncode=check.returncode,
            passed=check.returncode==0 and count==len(rows),decoder_error=check.stderr)
        subprocess.run(['ffmpeg','-y','-v','error','-ss',str(instant),'-i',str(path),
            '-frames:v','1',str(run/(camera+'-hold.png'))],check=True)
    report=dict(passed=all(v['passed'] for v in videos.values()),cameras=videos,
        scope='Decoded diagnostic footage; not a successful benchmark demonstration.')
    (run/'video-verification.json').write_text(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path)
    run=p.parse_args().run
    assert run.resolve().parent==OUTPUT_ROOT
    report=verify(run);print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['passed'] else 1)
