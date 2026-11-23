"""Assess the handle-held simulation endpoint after complete native audits.

Native validity failures still reject simulated evidence. They are not
real-trial measurement requirements; see REAL_TRIAL_SCORING.md.
"""

from task_paths import OUTPUT_ROOT

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from handle_held_finish import HandleHeldScorer, HandleSupport, HandleSupportObserver, REVISION
from score_logic import Observation

OUTPUT = OUTPUT_ROOT


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def audited_observations(run):
    """Reuse hash-bound native observations without changing the original assessment."""
    baseline = json.loads((run / 'benchmark-result.json').read_text())
    assert not baseline['invalid_reasons'], 'Invalid physics cannot qualify for the revised endpoint'
    assert sha256(run / 'outcomes.jsonl') == baseline['observations_sha256']
    for name, expected in baseline['audit_sha256'].items():
        assert sha256(run / name) == expected, f'Audit changed: {name}'
        audit = json.loads((run / name).read_text())
        for source, digest in audit['source_sha256'].items():
            assert sha256(run / source) == digest, f'Stale audit input: {source}'
    cloth = json.loads((run / 'full-cloth-audit.json').read_text())
    rigid = json.loads((run / 'frame-robot-audit.json').read_text())
    assert cloth['complete'] and cloth['numerical_checks_passed']
    assert rigid['robot_table_and_self_contacts_audited'] and rigid['contact_checks_passed']
    scene = json.loads((run / 'scene-audit.json').read_text())
    assert not scene['laboratory_fixture'] and not scene['cloth_attachments']
    assert not scene.get('diagnostic_initialization')
    rows = [json.loads(line) for line in (run / 'outcomes.jsonl').read_text().splitlines()]
    assert len(rows) == baseline['control_steps']
    for index, row in enumerate(rows):
        assert row['observation']['step'] == index
    return rows, baseline


def can_begin_retention(observation):
    """Skip extra support geometry only when the unchanged functional checks fail."""
    orientations = observation.pocket_orientation_correct
    correct = observation.orientation_correct if orientations is None else all(orientations)
    return (correct and min(observation.pocket_depth_m) >= .04
            and min(observation.pocket_width_fraction) >= .8
            and abs(observation.wing_angle_deg) <= 3.
            and observation.all_cloth_clearance_m >= .1
            and not observation.textile_grasped and not observation.wings_externally_supported)


def score(run, historical=False):
    source_hashes = {name: sha256(Path(__file__).with_name(name)) for name in
                     ('score_handle_held.py', 'handle_held_finish.py', 'score_logic.py', 'outcome_native.py')}
    request = json.loads((run / 'request.json').read_text())
    result = json.loads((run / 'result.json').read_text())
    assert request['robot_cover'] and not request.get('release_probe_run')
    assert request.get('policy', 'scripted') != 'interface-check', 'Offline interface checks are not benchmark episodes'
    assert result['status'] == 'completed'
    revised = request.get('completion_mode', 'released') == 'handle-held'
    assert historical != revised, 'Fresh v2 assessments require a predeclared v2 episode; old runs are screens only'
    target = OUTPUT / ('handle-held-v2-screen-' + run.name) if historical else run
    target.mkdir(exist_ok=True)
    baseline = json.loads((run / 'benchmark-result.json').read_text())
    if baseline['invalid_reasons']:
        assert baseline['score'] == 0. and not baseline['success']
        # The unchanged physics rules invalidate both rubrics. Preserve a genuine
        # model zero without pretending that invalid native states earned support.
        if baseline.get('native_counterexample'):
            from invalid_witness import strain_witness
            assert strain_witness(run) == baseline['native_counterexample']
        else:
            assert baseline.get('audit_sha256'), 'Missing independent invalidity evidence'
            for name, digest in baseline['audit_sha256'].items():
                assert sha256(run / name) == digest
                for path, expected in json.loads((run / name).read_text())['source_sha256'].items():
                    assert sha256(run / path) == expected
        report = dict(rubric_revision=REVISION, historical_screen_only=historical,
            source_run=run.name, score=0., maximum_score=0., success=False,
            observed_endpoint_success=False, invalid_reasons=baseline['invalid_reasons'],
            termination_reason='invalid', control_steps=result['control_steps'],
            native_steps=result['control_steps'] * (request['physics_hz'] // 20),
            original_released_v1_score=baseline['score'],
            original_assessment_sha256=sha256(run / 'benchmark-result.json'),
            audit_sha256=baseline.get('audit_sha256', {}), source_sha256=source_hashes,
            assessment='Verified unchanged physics invalidity applies to the declared handle-held endpoint.')
        if baseline.get('native_counterexample'):
            report['native_counterexample'] = baseline['native_counterexample']
        (target / 'handle-held-benchmark-result.json').write_text(json.dumps(report, indent=2))
        return report
    rows, baseline = audited_observations(run)
    original_hash = sha256(run / 'benchmark-result.json')
    with np.load(run / 'geometry.npz') as geometry:
        faces = geometry['faces']
    observer = HandleSupportObserver(json.loads((run / 'frame-geometry.json').read_text()),
        json.loads((run / 'frame-names.json').read_text()),
        json.loads((run / 'robot-cooked-colliders.json').read_text()), faces)
    scorer = HandleHeldScorer()
    decimation = request['physics_hz'] // 20
    fields = ('positions', 'frame_body_poses', 'left_body_poses', 'right_body_poses')
    count, measured, maximum = 0, 0, 0.
    first_success = first_proof = proof_start = None
    windows = []
    output_path = target / 'handle-held-outcomes.jsonl'
    with output_path.open('w') as stream:
        for path in sorted((run / 'physics-chunks').glob('*.npz')):
            with np.load(path) as chunk:
                length = len(chunk['frame_body_poses'])
                assert length % decimation == 0
                candidate_rows = rows[count:count + length // decimation]
                needed = any(can_begin_retention(Observation(**row['observation'])) for row in candidate_rows)
                recorded = {name: chunk[name] for name in fields} if needed else None
            for offset in range(0, length, decimation):
                row = rows[count]
                observation = Observation(**row['observation'])
                support, details = HandleSupport(), {}
                if can_begin_retention(observation):
                    state = {name: value[offset:offset + decimation] for name, value in recorded.items()}
                    support, details = observer.measure(state)
                    measured += 1
                final = scorer.update(observation, support)
                if scorer.base.lift_xy is None:
                    proof_start = None
                elif proof_start is None:
                    proof_start = count
                maximum = max(maximum, final['score'])
                if final['success']:
                    if first_success is None:
                        first_success, first_proof = count, proof_start
                    if windows and windows[-1]['last_step'] == count - 1:
                        windows[-1]['last_step'] = count
                    else:
                        windows.append(dict(first_step=count, last_step=count))
                stream.write(json.dumps(dict(**final, phase=row.get('phase'),
                    observation=asdict(observation), handle_support=asdict(support),
                    candidate_support=details, support_geometry_measured=bool(details))) + '\n')
                count += 1
            if measured:
                print(json.dumps(dict(control_steps=count, support_intervals=measured,
                                      maximum_score=maximum)), flush=True)
    assert count == result['control_steps'] == len(rows)
    assert sha256(run / 'benchmark-result.json') == original_hash
    assert all(sha256(Path(__file__).with_name(name)) == digest for name, digest in source_hashes.items()), 'Assessment source changed while running'
    report = dict(rubric_revision=REVISION, historical_screen_only=historical,
        source_run=run.name, score=final['score'], maximum_score=maximum,
        success=final['success'] and not historical,
        observed_endpoint_success=final['success'], invalid_reasons=final['invalid_reasons'],
        termination_reason='historical_screen' if historical else final['termination_reason'],
        control_steps=count, native_steps=count * decimation, support_intervals=measured,
        first_success_step=first_success, first_successful_proof_start_step=first_proof,
        successful_windows=windows, original_released_v1_score=baseline['score'],
        original_assessment_sha256=original_hash, audit_sha256=baseline['audit_sha256'],
        original_observations_sha256=baseline['observations_sha256'],
        observations_sha256=sha256(output_path),
        source_sha256=source_hashes,
        assessment='One unchanged native-audited recording, in order; functional proof resets on any extra support, hand change or loss of lift. Original score is preserved.',
        support_method='Swept actual collision geometry covers every native substep; both finger pads must remain possible contacts of the handle at every native state. All non-handle frame parts are separated from all robot links by at least 2 mm, and the other entire arm from the assembly by at least 20 mm. This is geometric support inference, not a contact-force measurement.')
    (target / 'handle-held-benchmark-result.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--historical-screen', action='store_true')
    args = parser.parse_args()
    assert args.run.resolve().parent == OUTPUT
    print(json.dumps(score(args.run, args.historical_screen), indent=2))
