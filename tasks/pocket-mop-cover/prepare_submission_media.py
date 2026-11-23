"""Prepare anonymous submission media only from audited genuine episode recordings."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
from fractions import Fraction
import re
from pathlib import Path
import shutil
import subprocess

from compose_robot_video import compose
from evaluation_contract import assessment_paths, configuration_identity, task_specification, validate_task_revision

OUT = OUTPUT_ROOT


def read(path):
    return json.loads(path.read_text())


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()



def probe_video(path):
    """Read codec, pixel format, dimensions and rate using the installed decoder."""
    if shutil.which('ffprobe'):
        probe = json.loads(subprocess.check_output(['ffprobe', '-v', 'error',
            '-select_streams', 'v:0', '-show_entries',
            'stream=codec_name,pix_fmt,width,height,r_frame_rate', '-of', 'json', str(path)], text=True))
        return probe['streams'][0]
    # The installed imageio FFmpeg distribution contains ffmpeg without ffprobe.
    # Decode a real frame and inspect its input stream; do not install a dependency
    # or infer the codec from the requested encoder or filename.
    decoded = subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-i', str(path),
        '-map', '0:v:0', '-frames:v', '1', '-f', 'null', '-'],
        capture_output=True, text=True, check=True)
    input_text = decoded.stderr.split('Stream mapping:', 1)[0]
    lines = [line for line in input_text.splitlines() if 'Stream #0:' in line and 'Video:' in line]
    assert len(lines) == 1, 'Expected one unambiguous input video stream'
    line = lines[0]
    codec = re.search(r'Video:\s*([A-Za-z0-9_]+)', line)
    pixels = re.search(r', ([A-Za-z0-9_]+)(?:\([^)]*\))?, (\d+)x(\d+)', line)
    rate = re.search(r', ([0-9.]+) fps', line)
    assert codec and pixels and rate, 'Cannot verify encoded video properties'
    fps = Fraction(rate.group(1))
    return dict(codec_name=codec.group(1), pix_fmt=pixels.group(1),
        width=int(pixels.group(2)), height=int(pixels.group(3)),
        r_frame_rate=f'{fps.numerator}/{fps.denominator}', verification_backend='ffmpeg_decode')


def verify_recording(run, *, model):
    """Require an audited declared task episode and a fully decoded recording."""
    request, result = read(run/'request.json'), read(run/'result.json')
    assert request['robot_cover']
    revision = validate_task_revision(request, run / 'source')
    _, result_file, observations_file = assessment_paths(request)
    assert not request.get('release_probe_run') and not result.get('diagnostic_only', False)
    assert result['status'] == 'completed' and read(run/'process.json')['returncode'] == 0
    assert 0 < result['control_steps'] <= 2400
    assert read(run/'video-verification.json')['passed'] and read(run/'gpu-verification.json')['verified']
    assert read(run/'reset-verification.json')['passed']
    score = read(run/result_file)
    if revision == 'handle-held-v2':
        assert score['rubric_revision'] == revision and not score['historical_screen_only']
        assert digest(run/'benchmark-result.json') == score['original_assessment_sha256']
        for file, expected in score['source_sha256'].items():
            assert digest(run/'source'/file) == expected
        if score.get('original_observations_sha256'):
            assert digest(run/'outcomes.jsonl') == score['original_observations_sha256']
    if score.get('observations_sha256'):
        assert digest(run/observations_file) == score['observations_sha256']
    for name, value in score.get('audit_sha256', {}).items():
        assert digest(run/name) == value
        for source, expected in read(run/name)['source_sha256'].items():
            assert digest(run/source) == expected, 'Stale audit: '+source
    if score.get('native_counterexample'):
        from invalid_witness import strain_witness
        assert score['score'] == 0. and strain_witness(run) == score['native_counterexample']
    else:
        assert score.get('audit_sha256'), 'Missing native assessment proof'
    if model:
        assert request['policy'] == 'dream' and result['hosted_model_attempt']
        transcript = read(run/'policy-transcript.json')
        assert transcript['hosted'] and transcript['received_chunks'] > 0
        assert not transcript['scripted_assistance'] and not transcript['privileged_data_in_policy']
        claim = read(OUT/'hosted-dream-attempt.json')
        assert claim['run'] == run.name and claim['authorized_episodes'] == 1
        assert (run/'model-observations.jsonl.gz').is_file() and (run/'inference.jsonl').is_file()
    else:
        assert request.get('policy', 'scripted') == 'scripted' and not request.get('replay_run')
        assert score['score'] == 1. and score['success'] and not score['invalid_reasons']
    return request, score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo', required=True, help='Audited full-score demonstration under the declared rubric')
    parser.add_argument('--model', required=True, help='Single genuine hosted Dream run name; zero score is accepted')
    parser.add_argument('--output', type=Path, default=OUT/'submission-media')
    args = parser.parse_args()
    for name in (args.demo, args.model):
        assert Path(name).name == name and name not in ('.', '..')
    assert args.output.resolve().is_relative_to(OUT) and not args.output.exists()
    demo, model = OUT/args.demo, OUT/args.model
    demo_request, demo_score = verify_recording(demo, model=False)
    model_request, model_score = verify_recording(model, model=True)
    assert configuration_identity(demo_request) == configuration_identity(model_request)
    assert task_specification(demo/'source') == task_specification(model/'source')
    assert model_request['demo_run'] == demo.name
    args.output.mkdir(parents=True)
    compose(demo, args.output/'demo.mp4', 'Pocket mop cover | Scripted demonstration | Score 1.0')
    compose(model, args.output/'model_run.mp4', f"Pocket mop cover | Dream-1 | Measured score {model_score['score']:g}")
    shutil.copyfile(demo/'setup.png', args.output/'setup.png')
    media = {}
    for name, run in (('demo.mp4', demo), ('model_run.mp4', model)):
        path = args.output/name
        stream = probe_video(path)
        assert stream['codec_name'] == 'h264' and stream['pix_fmt'] == 'yuv420p'
        assert stream['r_frame_rate'] == '20/1' and path.stat().st_size < 50_000_000
        media[name] = dict(sha256=digest(path), size_bytes=path.stat().st_size, stream=stream,
            source_run=run.name, rubric_revision=assessment_paths(read(run/'request.json'))[0],
            benchmark_result_file=assessment_paths(read(run/'request.json'))[1],
            benchmark_result_sha256=digest(run/assessment_paths(read(run/'request.json'))[1]))
    media['setup.png'] = dict(sha256=digest(args.output/'setup.png'), source_run=demo.name,
        source='Initial reset camera before any action; no identifiers added')
    report = dict(media=media, demo_score=demo_score['score'], model_score=model_score['score'],
        model_termination_reason=read(model/'result.json')['termination_reason'],
        media_id=None, uploaded=False, pr_submitted=False, visual_review_required=True)
    (args.output/'media-manifest.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
