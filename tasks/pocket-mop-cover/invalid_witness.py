"""Reject a recording using a hash-bound native material counterexample."""

import hashlib
import json

import numpy as np

from mesh import surface_edges


def strain_witness(run):
    """Find a native edge exceeding 20%; absence never certifies validity.

    Logged maxima only select a candidate interval. Rejection requires an
    independent calculation from that interval's recorded native positions.
    Successful or partially credited runs still require complete audits.
    """
    rows = [json.loads(line) for line in (run / 'steps.jsonl').read_text().splitlines()]
    candidates = [i for i, row in enumerate(rows) if row.get('max_abs_edge_strain', 0.) > .2]
    if not candidates:
        return None
    request = json.loads((run / 'request.json').read_text())
    geometry = np.load(run / 'geometry.npz')
    edges = surface_edges(geometry['faces'])
    lengths = np.linalg.norm(geometry['rest'][edges[:, 1]] - geometry['rest'][edges[:, 0]], axis=1)
    hz = request['physics_hz']
    decimation = hz // 20
    for control_step in candidates:
        start = control_step * decimation
        path = run / 'physics-chunks' / f'{start // hz:05d}.npz'
        with np.load(path) as chunk:
            points = chunk['positions'][start % hz:start % hz + decimation]
        measured = np.linalg.norm(points[:, edges[:, 1]] - points[:, edges[:, 0]], axis=2)
        strain = abs(measured / lengths - 1.)
        bad = np.argwhere(strain > .2)
        if not len(bad):
            continue
        frame, edge = (int(value) for value in bad[0])
        value = float(strain[frame, edge])
        assert np.isfinite(value) and value > .2
        sources = [path, *[run / name for name in ('geometry.npz', 'steps.jsonl', 'request.json', 'manifest.json')]]
        witness = dict(reason='native_material_edge_strain', threshold=.2,
            physics_step=start + frame, material_edge=edges[edge].tolist(),
            rest_length_m=float(lengths[edge]), measured_length_m=float(measured[frame, edge]),
            absolute_strain=value, source_sha256={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest()
                                                for p in sources},
            scope='One independently measured counterexample certifies rejection, never validity.')
        (run / 'invalid-witness.json').write_text(json.dumps(witness, indent=2))
        return witness
    return None
