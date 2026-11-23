"""Compute unchanged native geometry in parallel, then reduce outcomes in order."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np

from outcome_native import NativeOutcome
from score_logic import Observation


def observer_for(run, numerical):
    """Construct the same observer used by the sequential complete-run scorer."""
    geometry = np.load(run / 'geometry.npz')
    scene = json.loads((run / 'scene-audit.json').read_text())
    request = json.loads((run / 'request.json').read_text())
    assert not scene['laboratory_fixture'] and not scene['cloth_attachments']
    center = np.array(scene['table_center_m'][:2])
    half = np.array(scene['table_size_m'][:2]) / 2
    return NativeOutcome(geometry['initial'], geometry['faces'],
        json.loads((run / 'cover-mesh.json').read_text()),
        json.loads((run / 'frame-geometry.json').read_text()),
        json.loads((run / 'frame-names.json').read_text()),
        json.loads((run / 'robot-cooked-colliders.json').read_text()),
        float(scene['table_center_m'][2] + scene['table_size_m'][2] / 2),
        request['physics_hz'], numerical, (center - half, center + half))


def geometry_part(run, directory, start, stop, numerical):
    """Write observations only; partition-local scores are never published."""
    observer = observer_for(run, numerical)
    request = json.loads((run / 'request.json').read_text())
    decimation = request['physics_hz'] // 20
    first = start * 20
    observer.scorer.last_step = first - 1
    if start:
        with np.load(run / 'physics-chunks' / f'{start - 1:05d}.npz') as previous:
            observer.previous = previous['positions'][-1].copy()
    fields = ('positions', 'velocities', 'frame_body_poses', 'frame_joint_positions',
              'left_body_poses', 'right_body_poses')
    measures = ('pockets', 'cloth_robot_clearance_lower_bound_m',
                'wings_robot_clearance_lower_bound_m', 'handle_robot_clearance_lower_bound_m')
    count = first
    path = directory / f'{start:05d}-{stop:05d}.jsonl'
    temporary = path.with_suffix('.tmp')
    with temporary.open('w') as output:
        for chunk_index in range(start, stop):
            with np.load(run / 'physics-chunks' / f'{chunk_index:05d}.npz') as chunk:
                length = len(chunk['positions'])
                assert length % decimation == 0
                recorded = {name: chunk[name] for name in fields}
            for offset in range(0, length, decimation):
                state = {name: values[offset:offset + decimation] for name, values in recorded.items()}
                observation, derived = observer.update(count, state)
                output.write(json.dumps(dict(observation=asdict(observation),
                    measurements={name: derived[name] for name in measures})) + '\n')
                count += 1
    temporary.replace(path)
    return dict(path=str(path), first=first, stop=count,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def observations(run, numerical, workers, prefix=''):
    """Yield every observation exactly once in its original temporal order."""
    result = json.loads((run / 'result.json').read_text())
    chunks = sorted((run / 'physics-chunks').glob('[0-9][0-9][0-9][0-9][0-9].npz'))
    assert [int(path.stem) for path in chunks] == list(range(len(chunks)))
    assert 1 <= workers <= 8
    workers = min(workers, len(chunks))
    directory = run / (prefix + 'score-observations')
    directory.mkdir(exist_ok=False)
    ranges = []
    for indices in np.array_split(np.arange(len(chunks)), workers):
        ranges.append((int(indices[0]), int(indices[-1]) + 1))
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        futures = [pool.submit(geometry_part, run, directory, start, stop, numerical) for start, stop in ranges]
        parts = [future.result() for future in futures]
    count = 0
    for part in parts:
        assert part['first'] == count
        path = Path(part['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == part['sha256']
        with path.open() as stream:
            for line in stream:
                row = json.loads(line)
                observation = Observation(**row['observation'])
                assert observation.step == count
                yield observation, row['measurements']
                count += 1
        assert count == part['stop']
    assert count == result['control_steps']
    proof = dict(scope='Unchanged NativeOutcome geometry per interval, including previous-chunk velocity differences; one ordered OutcomeScorer reduction.',
        control_steps=count, partitions=parts,
        source_sha256={name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('parallel_outcomes.py', 'outcome_native.py', 'score_logic.py', 'measurements.py')})
    (run / (prefix + 'observation-parallel-provenance.json')).write_text(json.dumps(proof, indent=2))
