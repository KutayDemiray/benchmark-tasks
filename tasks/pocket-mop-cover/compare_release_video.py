"""Compare matched release diagnostics with and without their textile cover."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def compose(covered, bare, output):
    """Label matched ordinary commands and verify the resulting comparison clip."""
    runs = (covered, bare)
    rows = []
    for run in runs:
        request = json.loads((run / 'request.json').read_text())
        assert request.get('release_probe_run'), 'Only diagnostic runs may enter this comparison'
        result = json.loads((run / 'result.json').read_text())
        assert result['status'] == 'completed'
        verification = json.loads((run / 'video-verification.json').read_text())
        assert verification['passed'] and len(verification['cameras']) == 3
        rows.append([json.loads(line) for line in (run / 'steps.jsonl').read_text().splitlines()])
    assert not json.loads((covered / 'request.json').read_text()).get('release_probe_without_cover')
    assert json.loads((bare / 'request.json').read_text())['release_probe_without_cover']
    commands = [np.asarray([row['robot_action'] for row in values]) for values in rows]
    assert np.array_equal(*commands), 'Comparison requires identical ordinary robot commands'
    hands_clear = all(json.loads((run / 'request.json').read_text()).get('release_probe_hands_clear')
                      for run in runs)
    if hands_clear:
        assert np.all(commands[0] == commands[0][0]), 'Hands-clear comparison must hold constant commands'
    assert (covered / 'frame-geometry.json').read_bytes() == (bare / 'frame-geometry.json').read_bytes()
    expected = len(rows[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    labels = Image.new('RGBA', (1280, 400), (0, 0, 0, 0))
    draw = ImageDraw.Draw(labels)
    font = ImageFont.truetype('DejaVuSans.ttf', 22)
    for text, x in [('With cover - head overturns', 14), ('Without cover - head stays flat', 654)]:
        bounds = draw.textbbox((x, 12), text, font=font)
        draw.rectangle((bounds[0]-6, bounds[1]-5, bounds[2]+6, bounds[3]+5), fill=(12, 20, 30, 210))
        draw.text((x, 12), text, font=font, fill='white')
    draw.rectangle((0, 360, 1280, 400), fill=(12, 20, 30, 255))
    caption = ('Hands held clear and still - cold-start diagnostic, not a complete task demonstration'
               if hands_clear else
               'Release-only diagnostic - identical commands - neither run demonstrates the full task')
    draw.text((16, 374), caption,
              font=ImageFont.truetype('DejaVuSans.ttf', 19), fill='white')
    overlay = output.with_suffix('.labels.png')
    labels.save(overlay)
    filters = '[0:v][1:v]hstack,pad=1280:400:0:0[combined];[combined][2:v]overlay=0:0[out]'
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-i', str(covered / 'top_cam.mp4'),
                    '-i', str(bare / 'top_cam.mp4'), '-i', str(overlay), '-filter_complex', filters, '-map', '[out]',
                    '-frames:v', str(expected), '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '21',
                    '-threads', '2', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)], check=True)
    decoded = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(output), '-f', 'null', '-',
                              '-progress', 'pipe:1', '-nostats'], capture_output=True, text=True)
    counts = [int(line.split('=', 1)[1]) for line in decoded.stdout.splitlines() if line.startswith('frame=')]
    assert decoded.returncode == 0 and counts[-1] == expected, decoded.stderr
    subprocess.run(['ffmpeg', '-y', '-v', 'error', '-ss', '3', '-i', str(output),
                    '-frames:v', '1', str(output.with_suffix('.preview.png'))], check=True)
    report = dict(diagnostic_only=True, benchmark_eligible=False, full_task_success=False,
                  covered_run=covered.name, omitted_cover_control=bare.name,
                  identical_ordinary_actions=True, decoded_frames=expected, duration_s=expected / 20,
                  hands_initialized_clear=hands_clear, constant_commands=bool(np.all(commands[0] == commands[0][0])),
                  playback_speed=1, size_bytes=output.stat().st_size,
                  composition_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for run in runs for path in
                                 (run / 'top_cam.mp4', run / 'request.json', run / 'steps.jsonl', run / 'video-verification.json')})
    output.with_suffix('.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('covered', type=Path)
    parser.add_argument('bare', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    root = OUTPUT_ROOT
    assert args.covered.resolve().parent == args.bare.resolve().parent == root
    assert args.output.resolve().is_relative_to(root)
    compose(args.covered, args.bare, args.output)
