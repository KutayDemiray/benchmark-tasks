"""Bounded full-state history and independent CPU verification of strain peaks."""

from collections import deque
import json

import numpy as np


class StrainWitness:
    """Retain the largest finite native strain and surrounding full states."""

    def __init__(self, edges, rest_lengths, faces, history_frames=8):
        self.edges = np.asarray(edges, dtype=np.int64)
        self.rest_lengths = np.asarray(rest_lengths, dtype=np.float64)
        self.faces = np.asarray(faces, dtype=np.int64)
        self.history_frames = history_frames
        self.before = deque(maxlen=history_frames)
        self.frames = []
        self.peak = None
        self.remaining = 0

    def record(self, state):
        value = float(state['reported_strain'])
        if np.isfinite(value) and (self.peak is None or value > self.peak['reported_strain']):
            self.peak = state
            self.frames = [*self.before, state]
            self.remaining = self.history_frames
        elif self.remaining:
            self.frames.append(state)
            self.remaining -= 1
        self.before.append(state)

    def audit(self):
        assert self.peak is not None, 'No finite strain witness'
        state = self.peak
        nodes = np.asarray(state['cloth_positions_m'], dtype=np.float64)
        lengths = np.linalg.norm(nodes[self.edges[:, 0]] - nodes[self.edges[:, 1]], axis=1)
        signed = lengths / self.rest_lengths - 1.
        index = int(np.argmax(np.abs(signed)))
        recorded = int(state['reported_edge_index'])
        error = abs(float(np.abs(signed).max()) - float(state['reported_strain']))
        endpoints = self.edges[recorded]
        nearby = np.flatnonzero(np.isin(self.faces, endpoints).any(axis=1))
        return dict(physics_step=int(state['physics_step']), edge_index=recorded,
                    edge_nodes=endpoints.tolist(), rest_length_m=float(self.rest_lengths[recorded]),
                    measured_length_m=float(lengths[recorded]), signed_strain=float(signed[recorded]),
                    classification='extension' if signed[recorded] >= 0 else 'compression',
                    reported_absolute_strain=float(state['reported_strain']),
                    cpu_absolute_strain=float(np.abs(signed).max()), cpu_peak_edge_index=index,
                    cpu_agreement=bool(error < 1e-9 and abs(abs(signed[recorded])-abs(signed[index])) < 1e-9),
                    cpu_absolute_error=error, incident_face_indices=nearby.tolist(),
                    history_physics_steps=[int(frame['physics_step']) for frame in self.frames],
                    contact_geometry_note='Full cloth, hardware and native robot link poses; '
                    'table geometry and collision shapes are in setup.json and initial.usda. '
                    'Distributed cloth contact impulses are not available in this record.')

    def save(self, destination):
        if self.peak is None:
            return
        np.savez_compressed(destination / 'worst-strain-state.npz', **self.peak,
                            edge_nodes=self.edges[int(self.peak['reported_edge_index'])],
                            rest_length_m=self.rest_lengths[int(self.peak['reported_edge_index'])])
        history = {key: np.stack([frame[key] for frame in self.frames]) for key in self.peak}
        np.savez_compressed(destination / 'strain-event-history.npz', **history)
        (destination / 'strain-witness.json').write_text(json.dumps(self.audit(), indent=2) + '\n')
