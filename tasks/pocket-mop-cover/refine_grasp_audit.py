"""Refine conservative crossing flags while retaining the original full audit."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from self_contact import crossings, separating_translations


def refine(run, source_name, report_name):
    """Check separation witnesses for every flagged step of an archived audit."""
    assert Path(source_name).name == source_name and Path(report_name).name == report_name
    assert source_name != report_name
    source_path = run / source_name
    report = json.loads(source_path.read_text())
    assert 'self_crossing_penetration_upper_bound_m' not in report, 'Already refined'
    assert report['gripper_geometry_kind'] == 'native_cooked'
    evidence = report['cooked_geometry_evidence']
    assert hashlib.sha256(Path(evidence['path']).read_bytes()).hexdigest() == evidence['sha256']
    geometry = np.load(run / 'geometry.npz')
    flagged = set(report['self_crossing_violation_steps'])
    remaining = []
    proofs = []
    chunk_hashes = {}
    count = 0
    finite = True
    refined_max = 0.
    for path in sorted((run / 'physics-chunks').glob('*.npz')):
        chunk_hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        with np.load(path) as chunk:
            positions = chunk['positions']
            finite = finite and bool(np.isfinite(positions).all() and np.isfinite(chunk['velocities']).all())
            selected = sorted(flagged.intersection(range(count, count + len(positions))))
            for step in selected:
                points = positions[step - count]
                pairs, spans = crossings(points, geometry['faces'])
                assert len(pairs) and spans.max() > .001
                shifts = separating_translations(points, geometry['faces'], pairs)
                bounds = np.linalg.norm(shifts, axis=1)
                refined_max = max(refined_max, float(bounds.max()))
                if bounds.max() > .001:
                    remaining.append(step)
                witnesses = []
                for pair, span, shift, bound in zip(pairs, spans, shifts, bounds, strict=True):
                    if span <= .001:
                        continue
                    pair_points = np.asarray(points[geometry['faces'][pair]], dtype=float).reshape(6, 3)
                    pair_points[3:] += shift * (1 + 1e-4)
                    separated = len(crossings(pair_points, np.array([[0, 1, 2], [3, 4, 5]]))[0]) == 0
                    assert separated, 'Proposed geometric separation witness does not separate'
                    witnesses.append(dict(pair=pair.tolist(), old_span_m=float(span),
                        separation_upper_bound_m=float(bound), translation_m=shift.tolist(),
                        independently_separated=separated))
                proofs.append(dict(step=step, witnesses=witnesses))
            count += len(positions)
    assert count == report['physics_steps']
    assert {proof['step'] for proof in proofs} == flagged
    # The previous audit already certified all unflagged steps using a larger
    # upper bound. Their exact new maximum is not claimed without rescanning.
    unrefined_bound = .001 if len(report['self_crossing_steps']) > len(flagged) else 0.
    report.update(self_crossing_normal_span_violation_steps=sorted(flagged),
        self_crossing_violation_steps=remaining,
        self_crossing_penetration_upper_bound_m=max(unrefined_bound, refined_max),
        self_crossing_metric='Shortest separating translation along either triangle normal; conservative upper bound, unchanged 1 mm limit.',
        crossing_refinement=dict(source_report=source_name,
            source_report_sha256=hashlib.sha256(source_path.read_bytes()).hexdigest(),
            geometry_sha256=hashlib.sha256((run / 'geometry.npz').read_bytes()).hexdigest(),
            physics_chunk_sha256=chunk_hashes, refined_step_max_upper_bound_m=refined_max,
            unflagged_steps_bound_from_original_audit_m=unrefined_bound, proofs=proofs),
        recorded_states_finite=finite)
    report['numerical_checks_passed'] = bool(finite and report['max_abs_edge_strain'] <= .2
        and abs(report['max_abs_edge_strain'] - report['reported_max_abs_edge_strain']) < 1e-6
        and not report['bench_penetrating_steps'] and not report['table_penetrating_steps']
        and not report['gripper_penetrating_steps'] and not remaining)
    report['grasp_gate_passed'] = bool(report['numerical_checks_passed']
        and report['contact_grasp_observed'] and report['released_settling_passed']
        and report['no_attachment'] and report['gripper_contact_audit_complete'])
    report['refinement_source_sha256'] = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in ('refine_grasp_audit.py', 'self_contact.py', 'test_validation.py')}
    (run / report_name).write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--source', required=True)
    parser.add_argument('--report-name', default='grasp-audit-refined.json')
    args = parser.parse_args()
    assert args.run.resolve().parent == OUTPUT_ROOT
    report = refine(args.run, args.source, args.report_name)
    print(json.dumps({key: report[key] for key in ('grasp_gate_passed',
        'numerical_checks_passed', 'self_crossing_penetration_upper_bound_m')}, indent=2))
