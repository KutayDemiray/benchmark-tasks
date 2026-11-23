"""Build blind review media from an audited scripted demonstration or actual model attempt."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from task_paths import OUTPUT_ROOT


def digest(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def inspect_video(path):
    import imageio.v2 as imageio
    reader=imageio.get_reader(path);info=reader.get_meta_data()
    count=sum(1 for frame in reader);reader.close()
    assert 'h264' in info['codec'] and count>0
    assert path.stat().st_size<=50_000_000, 'Video exceeds public upload size limit'
    return dict(frames=count,fps=info['fps'],size=info['size'],codec=info['codec'],
                bytes=path.stat().st_size,sha256=digest(path),decoded=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--episode',default='episode-00')
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--kind',choices=('demo','model'),default='demo')
    args=parser.parse_args();run=args.run.resolve();out=args.out.resolve()
    assert all(p.is_relative_to(OUTPUT_ROOT) for p in (run,out))
    assert args.episode.startswith('episode-') and '/' not in args.episode
    out.mkdir(parents=True,exist_ok=False)
    from verify_tabletop_run import verify
    verification=verify(run)
    (out/'independent-audit.json').write_text(json.dumps(verification,indent=2)+'\n')
    entry=next(e for e in verification['episodes'] if e['episode']==args.episode)
    assert verification['verification_pass']
    model_evidence=None
    if args.kind=='demo':
        assert entry['full_task_success'] and entry['task_score']==1.
    else:
        model_evidence=json.loads((run.parent/'model-evaluation.json').read_text())
        assert model_evidence['mode']=='dream_live' and model_evidence['model_attempt'] is True
        assert model_evidence['hosted_requests_started']>0 and model_evidence['episode_retries']==0
        assert model_evidence['result']==json.loads((run/args.episode/'result.json').read_text())
        assert model_evidence['source_bundle_sha256']==verification['source_bundle_sha256']
        from dream_readiness import audit_model_run
        demo=Path(json.loads((run.parent/'readiness.json').read_text())['demonstration_path'])
        model_audit=audit_model_run(run.parent,demo)
        (out/'model-audit.json').write_text(json.dumps(model_audit,indent=2)+'\n')
    video_name='demo.mp4' if args.kind=='demo' else 'model_run.mp4'
    ep=run/args.episode
    setup=json.loads((ep/'setup.json').read_text());result=json.loads((ep/'result.json').read_text())
    if args.kind=='demo':assert not result['invalid_reasons']
    assert entry['controls']>0, 'No executed action frames; preserve the actual error instead of fabricating a model video'
    import imageio.v2 as imageio
    import imageio_ffmpeg
    from PIL import Image,ImageDraw,ImageFont
    cameras=('top_cam','left_cam','right_cam')
    frame_count=entry['controls'];fps=entry['videos']['top_cam']['fps'];duration=frame_count/fps
    assert all(entry['videos'][name]['decoded_frames']==frame_count and entry['videos'][name]['fps']==fps
               and tuple(entry['videos'][name]['size'])==(640,360) for name in cameras)
    card=Image.new('RGB',(640,360),(24,28,34));draw=ImageDraw.Draw(card)
    font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
    title=ImageFont.truetype(font,30);body=ImageFont.truetype(font,23)
    lines=[('Scripted demonstration' if args.kind=='demo' else 'Dream-1 evaluation',title),('Fasten three press-studs',body),
           (f'Recorded score: {entry["task_score"]:.3g}',body),(f'{duration:.1f} simulated seconds',body),
           (f'{frame_count} actions at {fps:g} Hz',body),('Front-up, aligned starting layout',body)]
    if model_evidence is not None:
        outcome = ('No model targets received' if model_evidence['hosted_requests_completed'] == 0
                   else 'Stop: ' + result['termination_reason'])
        lines.append((outcome, body))
    for index,(text,font_object) in enumerate(lines):draw.text((24,28+index*49),text,font=font_object,fill=(240,244,249))
    card.save(out/'review-card.png')
    exe=imageio_ffmpeg.get_ffmpeg_exe();command=[exe,'-nostdin','-hide_banner','-loglevel','warning']
    for name in cameras:command.extend(['-i',str(ep/(name+'.mp4'))])
    command.extend(['-loop','1','-framerate',str(fps),'-i',str(out/'review-card.png'),
        '-filter_complex','[0:v]setpts=PTS-STARTPTS[a];[1:v]setpts=PTS-STARTPTS[b];'
        '[2:v]setpts=PTS-STARTPTS[c];[3:v]setpts=PTS-STARTPTS[d];'
        '[a][b][c][d]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0:shortest=1[v]',
        '-map','[v]','-an','-t',str(duration),'-r',str(fps),'-c:v','libx264','-preset','medium',
        '-crf','20','-pix_fmt','yuv420p','-movflags','+faststart','-map_metadata','-1','-threads','2',str(out/video_name)])
    completed=subprocess.run(command,text=True,capture_output=True)
    (out/'ffmpeg.log').write_text(completed.stdout+completed.stderr)
    assert completed.returncode==0, 'Review composition failed; inspect ffmpeg.log'
    video=inspect_video(out/video_name);assert video['frames']==frame_count and abs(video['fps']-fps)<1e-6
    reader=imageio.get_reader(ep/'top_cam.mp4');imageio.imwrite(out/'setup.png',reader.get_data(0));reader.close()
    reader=imageio.get_reader(out/video_name)
    for index in (0,frame_count//2,frame_count-1):imageio.imwrite(out/f'review-frame-{index:04d}.png',reader.get_data(index))
    reader.close()
    report=dict(scope=('Local blind review assets from a verified scripted demonstration.' if args.kind=='demo' else 'Uncut genuine Dream-1 episode; actual outcome, including zero, preserved.'),
        kind=args.kind,video_file=video_name,
        source_run=str(run),source_episode=args.episode,source_bundle_sha256=verification['source_bundle_sha256'],
        source_seed=setup['seed'],source_score=entry['task_score'],source_controls=frame_count,
        source_cameras=entry['videos'],demo=video,setup_image=dict(sha256=digest(out/'setup.png'),
        source='First recorded overhead frame after one control interval',time_s=1/fps),
        layout=['top-left: overhead','top-right: left wrist','bottom-left: right wrist','bottom-right: factual review card'],
        temporal_edits=False,playback_speed=1.,images_synthesized=False,model_attempt=args.kind=='model',
        model=None if model_evidence is None else model_evidence['model'],
        termination_reason=result['termination_reason'],
        model_requests_completed=None if model_evidence is None else model_evidence['hosted_requests_completed'],
        model_error=None if model_evidence is None else model_evidence['error'],
        media_uploaded=False,visual_review_pending=True,ffmpeg_command=command)
    (out/'media-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),score=entry['task_score'],frames=frame_count,duration_s=duration,video_bytes=video['bytes'])))


if __name__=='__main__':main()
