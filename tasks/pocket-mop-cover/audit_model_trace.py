"""Independently verify a completed model episode's public inputs and action trace."""

from task_paths import OUTPUT_ROOT

import argparse
import base64
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil

import numpy as np
from PIL import Image

OUT = OUTPUT_ROOT
CAMERAS = ('top_cam', 'left_cam', 'right_cam')


def read(path):
    return json.loads(path.read_text())


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def equal_vector(actual, expected, description):
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape == (14,), description
    assert np.isfinite(actual).all() and np.array_equal(actual, expected), description


def verify(run):
    """Check every recorded public observation and executed ordinary action."""
    assert run.resolve().parent == OUT
    request, result = read(run / 'request.json'), read(run / 'result.json')
    transcript = read(run / 'policy-transcript.json')
    assert request['policy'] in ('dream', 'interface-check')
    assert result['status'] == 'completed' and read(run / 'process.json')['returncode'] == 0
    assert transcript['completed'] and not transcript['dry_run']
    assert not transcript['scripted_assistance'] and not transcript['privileged_data_in_policy']
    assert not (run / 'controller-plan.json').exists(), 'Scripted controller activity recorded'
    assert transcript['control_hz'] == 20 and transcript['preroll_steps'] == 32
    assert transcript['rows_per_request'] == 48
    hosted = request['policy'] == 'dream'
    assert transcript['hosted'] == hosted and result['hosted_model_attempt'] == hosted
    assert result['termination_reason'] == ('action_budget' if hosted else 'interface_check_complete')

    # Bounds are read through the launch-archived adapter only after verifying
    # the installed stock controller source hashes on which that adapter depends.
    preserved = read(run / 'preserved-sources.json')
    for name, expected in preserved.items():
        assert digest(Path(name)) == expected, 'Installed runtime changed: ' + name
    source = run / 'source' / 'evaluation_contract.py'
    assert digest(source) == read(run / 'manifest.json')['files'][source.name]
    spec = importlib.util.spec_from_file_location('recorded_evaluation_contract', source)
    contract = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(contract)

    actions = [json.loads(line) for line in (run / 'policy-actions.jsonl').read_text().splitlines()]
    steps = [json.loads(line) for line in (run / 'steps.jsonl').read_text().splitlines()]
    replies = [json.loads(line) for line in (run / 'inference.jsonl').read_text().splitlines()]
    count = len(actions)
    assert count == len(steps) == result['control_steps']
    assert count == (2400 if hosted else request['policy_check_steps'])
    assert result['physics_steps'] == count * 24 and request['physics_hz'] == 480
    assert len(replies) == len(transcript['requests']) == transcript['received_chunks'] > 0
    assert len(replies) == (count - 32 + 47) // 48
    initial = read(run / 'initial-public-state.json')['measured_joint_pos']
    expected_actions = [initial] * 32
    for index, reply in enumerate(replies):
        assert reply['request'] == index and reply['hosted'] == hosted
        assert reply['after_observation_step'] == 32 + 48 * index
        values = np.asarray(reply['rows'])
        assert values.shape == (48, 14) and np.isfinite(values).all()
        expected_actions.extend(reply['rows'])
    clipped = 0
    for index, (action, step) in enumerate(zip(actions, steps)):
        assert action['step'] == step['step'] == index
        assert action['simulation_time_s'] == step['time_s'] == (index + 1) / 20
        assert action['native_steps'] == 24 * (index + 1)
        assert step['phase'] == ('dream' if hosted else 'offline_interface_check')
        raw = expected_actions[index]
        equal_vector(action['requested_action'], raw, 'Non-model action in policy trace')
        equal_vector(step['requested_robot_action'], raw, 'Native requested action differs')
        executed = contract.process_action(raw)
        equal_vector(action['executed_stock_target'], executed, 'Unexpected action processing')
        equal_vector(step['robot_action'], executed, 'Native execution differs from stock clamping')
        equal_vector(action['measured_joint_pos'], step['measured_joint_pos'], 'Measured state differs')
        clipped += int(not np.array_equal(raw, executed))

    instruction = read(run / 'source' / 'task.json')['instruction']
    observation_count = 0
    image_count = 0
    with gzip.open(run / 'model-observations.jsonl.gz', 'rt') as stream:
        for index, line in enumerate(stream):
            observation = json.loads(line)
            assert set(observation) == {
                'step', 'simulation_time_s', 'images_jpeg_base64', 'joint_pos', 'instruction'}
            assert observation['step'] == index and observation['simulation_time_s'] == index / 20
            assert observation['instruction'] == instruction
            expected = initial if index == 0 else actions[index - 1]['measured_joint_pos']
            equal_vector(observation['joint_pos'], expected, 'Observation is not the measured state')
            assert set(observation['images_jpeg_base64']) == set(CAMERAS)
            for name in CAMERAS:
                data = base64.b64decode(observation['images_jpeg_base64'][name], validate=True)
                with Image.open(io.BytesIO(data)) as image:
                    assert image.format == 'JPEG' and image.size == (320, 176) and image.mode == 'RGB'
                    image.load()
                image_count += 1
            observation_count += 1
    assert observation_count == count and image_count == 3 * count

    files = ('request.json', 'result.json', 'process.json', 'policy-transcript.json',
             'policy-actions.jsonl', 'steps.jsonl', 'inference.jsonl', 'model-observations.jsonl.gz',
             'initial-public-state.json', 'preserved-sources.json', 'manifest.json',
             'source/task.json', 'source/evaluation_contract.py')
    saved = run / 'assessment-source' / Path(__file__).name
    saved.parent.mkdir(exist_ok=True)
    if saved.exists():
        assert digest(saved) == digest(Path(__file__)), 'Assessment source already exists with different content'
    else:
        shutil.copyfile(__file__, saved)
    configuration = Path(__file__).with_name('task_paths.py')
    saved_configuration = saved.with_name('task_paths.py')
    if saved_configuration.exists():
        assert digest(saved_configuration) == digest(configuration), 'Archived path configuration changed'
    else:
        shutil.copyfile(configuration, saved_configuration)
    files = (*files, 'assessment-source/task_paths.py')
    report = dict(
        passed=True, run=run.name, hosted=hosted, control_steps=count,
        physics_steps=result['physics_steps'], control_hz=20, observation_count=observation_count,
        decoded_rgb_images=image_count, camera_resolution=[320, 176],
        joint_labels=list(contract.LABELS), preroll_actions=32,
        received_chunks=len(replies), model_action_steps=count - 32,
        stock_clamped_actions=clipped, client_request_attempts=sum(row['attempts'] for row in replies),
        inference_elapsed_wall_seconds=sum(row['elapsed_wall_s'] for row in replies),
        termination_reason=result['termination_reason'],
        source_sha256={name: digest(run / name) for name in files},
        audit_source_sha256=digest(saved), installed_runtime_sha256=preserved,
        scope='Every input handed to the installed Dream integration and every executed target was checked. '
              'No network call, simulation, privileged input, scripted assistance, or task reward is introduced '
              'by this offline audit. Hosted service internals are outside its scope.')
    (run / 'model-trace-verification.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    report = verify(args.run.resolve())
    print(json.dumps({key: report[key] for key in (
        'passed', 'hosted', 'control_steps', 'decoded_rgb_images', 'received_chunks',
        'stock_clamped_actions', 'client_request_attempts', 'termination_reason')}, indent=2))
