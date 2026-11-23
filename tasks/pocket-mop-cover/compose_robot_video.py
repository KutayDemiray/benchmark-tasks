"""Compose the three verified stock camera recordings at their original speed."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from PIL import Image, ImageDraw, ImageFont


def compose(run, output, caption,start=0.,duration=None,allow_incomplete=False):
    """Save a labelled overhead/wrist montage with source hashes and decode proof."""
    result=json.loads((run/'result.json').read_text())
    verification=json.loads((run/'video-verification.json').read_text())
    assert verification['passed']
    completed=result['status']=='completed'
    assert completed or (allow_incomplete and result['status']=='error'), 'Incomplete runs require an explicit diagnostic request'
    assert completed or 'diagnostic' in caption.lower(), 'Label incomplete footage as diagnostic'
    request=json.loads((run/'request.json').read_text())
    revision='handle-held-v2' if request.get('completion_mode')=='handle-held' else 'released-v1'
    assessment=run/('handle-held-benchmark-result.json' if revision=='handle-held-v2' else 'benchmark-result.json')
    assessed=json.loads(assessment.read_text()) if assessment.exists() else {}
    success=completed and assessed.get('score')==1. and assessed.get('success',False)
    cameras=['top_cam','left_cam','right_cam']
    source_frames=result['control_steps']
    assert all(verification['cameras'][name]['decoded_frames']==source_frames for name in cameras)
    duration=source_frames/20-start if duration is None else duration
    assert start>=0 and duration>0 and start+duration<=source_frames/20
    assert abs(start*20-round(start*20))<1e-9 and abs(duration*20-round(duration*20))<1e-9
    expected=round(duration*20)
    output.parent.mkdir(parents=True,exist_ok=True)
    labels=Image.new('RGBA',(1440,576),(0,0,0,0))
    draw=ImageDraw.Draw(labels)
    font=ImageFont.truetype('DejaVuSans.ttf',21)
    small=ImageFont.truetype('DejaVuSans.ttf',17)
    for label,x,y in [('Overhead',16,14),('Left wrist',976,14),('Right wrist',976,284)]:
        bounds=draw.textbbox((x,y),label,font=font)
        draw.rectangle((bounds[0]-7,bounds[1]-5,bounds[2]+7,bounds[3]+5),fill=(12,20,30,210))
        draw.text((x,y),label,font=font,fill='white')
    draw.rectangle((0,540,1440,576),fill=(12,20,30,225))
    draw.text((16,548),caption,font=small,fill='white')
    overlay=output.with_suffix('.labels.png')
    labels.save(overlay)
    command=['ffmpeg','-y','-v','error']
    for name in cameras:
        command+=['-ss',str(start),'-i',str(run/(name+'.mp4'))]
    command+=['-i',str(overlay),'-filter_complex',
        '[0:v]scale=960:540:force_original_aspect_ratio=decrease,pad=960:540:(ow-iw)/2:(oh-ih)/2[top];'
        '[1:v]scale=480:270:force_original_aspect_ratio=decrease,pad=480:270:(ow-iw)/2:(oh-ih)/2[left];'
        '[2:v]scale=480:270:force_original_aspect_ratio=decrease,pad=480:270:(ow-iw)/2:(oh-ih)/2[right];'
        '[left][right]vstack[wrists];[top][wrists]hstack,pad=1440:576:0:0[combined];'
        '[combined][3:v]overlay=0:0[out]',
        '-map','[out]','-t',str(duration),'-an','-c:v','libx264','-preset','fast','-crf','22','-threads','2',
        '-maxrate','2800k','-bufsize','5600k','-map_metadata','-1',
        '-pix_fmt','yuv420p','-movflags','+faststart',str(output)]
    subprocess.run(command,check=True)
    assert output.stat().st_size < 50_000_000, 'Video exceeds submission size limit'
    decoded=subprocess.run(['ffmpeg','-v','error','-i',str(output),'-f','null','-',
        '-progress','pipe:1','-nostats'],capture_output=True,text=True)
    counts=[int(line.split('=',1)[1]) for line in decoded.stdout.splitlines() if line.startswith('frame=')]
    assert decoded.returncode==0 and counts[-1]==expected,decoded.stderr
    subprocess.run(['ffmpeg','-y','-v','error','-ss',str(min(17.,duration/2)),'-i',str(output),
        '-frames:v','1',str(output.with_suffix('.preview.png'))],check=True)
    report=dict(source_run=run.name,layout='Overhead left; left/right wrists upper/lower right',
        source_aspect_ratios_preserved=True,output_dimensions=[1440,576],
        composition_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        caption=caption,decoded_frames=counts[-1],duration_s=expected/20,playback_speed=1,
        source_start_s=start,source_end_s=start+duration,
        size_bytes=output.stat().st_size,diagnostic_only=result.get('diagnostic_only',False) or not request.get('robot_cover',False),full_task_success=success,
        measured_score=assessed.get('score'),policy=request.get('policy','scripted'),
        rubric_revision=revision,source_controller_status=result['status'],termination_reason=result.get('termination_reason'),
        benchmark_result_sha256=hashlib.sha256(assessment.read_bytes()).hexdigest() if assessment.exists() else None,
        source_sha256={name:hashlib.sha256((run/(name+'.mp4')).read_bytes()).hexdigest() for name in cameras})
    output.with_suffix('.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--caption',required=True)
    parser.add_argument('--start',type=float,default=0.)
    parser.add_argument('--duration',type=float)
    parser.add_argument('--allow-incomplete',action='store_true',help='Allow explicitly labelled diagnostic footage from a stopped controller')
    args=parser.parse_args()
    assert args.output.resolve().is_relative_to(OUTPUT_ROOT)
    compose(args.run,args.output,args.caption,args.start,args.duration,args.allow_incomplete)
