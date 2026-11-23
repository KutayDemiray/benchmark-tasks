"""Replay recorded native measurements through the incremental scorer on CPU."""

import argparse
import json
from pathlib import Path
from task_paths import OUTPUT_ROOT
from streaming_score import StreamingScore
from episode_timing import physics_stride


def verify(episode):
    setup=json.loads((episode/'setup.json').read_text());cfg=setup['parameters']
    meta=json.loads((episode/'topology.json').read_text())
    scorer=StreamingScore(cfg,meta);stride=physics_stride(cfg)
    count=0
    with (episode/'physics.jsonl').open() as physics,(episode/'trace.jsonl').open() as controls:
        scorer.observe_physics(json.loads(next(physics)))
        for line in controls:
            row=json.loads(line)
            for _ in range(stride):scorer.observe_physics(json.loads(next(physics)))
            result=scorer.observe_control(row)
            assert result['task_score']==row['task_score'], (row['step'],result,row['task_score'])
            count+=1
        assert next(physics,None) is None, 'Extra native state after controls'
    stored=json.loads((episode/'result.json').read_text())
    assert scorer.result()['task_score']==stored['task_score']
    return dict(episode=str(episode),control_steps=count,physics_states=scorer.index+1,
                identical_per_control_scores=True,result=scorer.result(),
                scope='CPU replay of recorded states; not native action-trajectory replay')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert args.out.resolve().is_relative_to(OUTPUT_ROOT)
    result=verify(args.episode)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':main()
