"""Verify frozen robot diagnostics, measured resets and full camera-file decoding on CPU."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from audit import audit

OUTPUT=OUTPUT_ROOT


def verify(run):
    """Return source, score, reset and video evidence without granting full-task credit."""
    result={'run':str(run),'independent_audit':audit(run),'episodes':[],
            'reset_comparisons':[],'full_task_validated':False}
    initial=[]
    for episode in sorted(run.glob('episode-*')):
        entry={'episode':episode.name,'complete':(episode/'result.json').exists(),'videos':{}}
        setup=json.loads((episode/'setup.json').read_text()) if (episode/'setup.json').exists() else {}
        recorded=setup.get('recorded_cameras',any(episode.glob('*.mp4')))
        if recorded:
            import imageio.v2 as imageio
            count=len((episode/'trace.jsonl').read_text().splitlines())
            for camera in ('top_cam','left_cam','right_cam'):
                path=episode/(camera+'.mp4')
                try:
                    reader=imageio.get_reader(path)
                    meta=reader.get_meta_data()
                    decoded=sum(1 for _ in reader)
                    reader.close()
                    assert decoded==count,'camera_frame_count_differs_from_actions'
                    assert abs(meta['fps']-10)<1e-6,'wrong_video_frame_rate'
                    assert 'h264' in meta['codec'],'wrong_video_codec'
                    entry['videos'][camera]={'decoded_frames':decoded,'fps':meta['fps'],
                        'codec':meta['codec'],'size':meta['size'],
                        'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'passed':True}
                except Exception as exc:
                    entry['videos'][camera]={'passed':False,'error':repr(exc)}
        else:
            entry['video_note']='Cameras were not recorded for this diagnostic.'
        path=episode/'initial-measured-state.npz'
        if path.exists():
            with np.load(path) as data:
                arrays={k:data[k].copy() for k in data.files}
            assert all(np.isfinite(v).all() for v in arrays.values()),'nonfinite_initial_reset_state'
            initial.append((episode.name,arrays))
        result['episodes'].append(entry)
    for (first,a),(second,b) in zip(initial,initial[1:]):
        assert set(a)==set(b),'reset_state_fields_changed'
        assert all(a[k].shape==b[k].shape for k in a),'reset_state_shapes_changed'
        deltas={k:float(np.max(np.abs(a[k]-b[k]))) for k in a}
        result['reset_comparisons'].append({'first':first,'second':second,
            'maximum_absolute_deltas':deltas,'exactly_equal':all(v==0 for v in deltas.values()),
            'note':'Report numerical differences explicitly; bit-identical GPU trajectories are not required.'})
    exit_path=run/'exit.json'
    result['launcher_exit']=json.loads(exit_path.read_text()) if exit_path.exists() else None
    result['complete_records']=bool(result['launcher_exit'] and result['launcher_exit']['returncode']==0
        and result['episodes'] and all(e['complete'] for e in result['episodes']))
    result['recorded_video_decodes_pass']=all(v['passed'] for e in result['episodes'] for v in e['videos'].values())
    result['scope']='Diagnostic verification only. Camera decoding does not establish that the task is visible or completed.'
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert args.run.resolve().is_relative_to(OUTPUT) and args.out.resolve().is_relative_to(OUTPUT)
    result=verify(args.run)
    args.out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result[k] for k in ('complete_records','recorded_video_decodes_pass','full_task_validated')}))
    if not (result['complete_records'] and result['recorded_video_decodes_pass']):
        raise SystemExit(1)


if __name__=='__main__':
    main()
