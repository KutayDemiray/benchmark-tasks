"""Account for recorded rigid energy and explicit continuum cloth-energy estimates.

The cloth terms are analytical proxies, not native solver energy readback.
This diagnostic does not award a score or certify energy conservation.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


class SurfaceEnergy:
    """Evaluate fixed-rest membrane and hinge-bending proxies without simulation writes."""

    def __init__(self, rest, faces, material):
        self.faces = faces
        self.material = material
        tri = rest[faces]
        a, b = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
        normal = np.cross(a, b)
        self.area = np.linalg.norm(normal, axis=1) / 2
        assert np.all(self.area > 1e-12)
        x = a / np.linalg.norm(a, axis=1)[:, None]
        y = np.cross(normal / (2 * self.area[:, None]), x)
        metric = np.stack((np.stack(((a*x).sum(1), (a*y).sum(1)), -1),
                           np.stack(((b*x).sum(1), (b*y).sum(1)), -1)), -1)
        self.inverse_metric = np.linalg.inv(metric)
        self.node_mass = np.zeros(len(rest))
        triangle_mass = self.area * material['density_kg_m3'] * material['shell_thickness_m']
        for column in range(3):
            np.add.at(self.node_mass, faces[:, column], triangle_mass / 3)
        incident = {}
        for index, face in enumerate(faces):
            for i, j in ((0, 1), (1, 2), (2, 0)):
                edge = tuple(sorted((int(face[i]), int(face[j]))))
                incident.setdefault(edge, []).append(index)
        assert all(len(value) <= 2 for value in incident.values()), 'Nonmanifold material edge'
        pairs = [(edge, value) for edge, value in incident.items() if len(value) == 2]
        self.edges = np.array([edge for edge, _ in pairs])
        self.pairs = np.array([value for _, value in pairs])
        length_squared = np.sum((rest[self.edges[:, 1]] - rest[self.edges[:, 0]])**2, axis=1)
        self.bend_weights = 3 * length_squared / self.area[self.pairs].sum(1)
        self.rest_angles = self.angles(rest)

    def angles(self, points):
        tri = points[self.faces]
        normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        normal /= np.maximum(np.linalg.norm(normal, axis=1)[:, None], 1e-15)
        a, b = normal[self.pairs[:, 0]], normal[self.pairs[:, 1]]
        edge = points[self.edges[:, 1]] - points[self.edges[:, 0]]
        edge /= np.maximum(np.linalg.norm(edge, axis=1)[:, None], 1e-15)
        return np.arctan2(np.sum(np.cross(a, b)*edge, axis=1), np.sum(a*b, axis=1))

    def measure(self, points, velocities):
        p = self.material
        tri = points[self.faces]
        ds = np.stack((tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), -1)
        gradient = ds @ self.inverse_metric
        strain = (np.swapaxes(gradient, 1, 2) @ gradient - np.eye(2)) / 2
        young, poisson = p['young_modulus_pa'], p['poisson_ratio']
        mu, lam = young / (2*(1+poisson)), young*poisson / (1-poisson**2)
        density = mu*np.sum(strain**2, axis=(1, 2)) + .5*lam*np.trace(strain, axis1=1, axis2=2)**2
        delta = self.angles(points) - self.rest_angles
        delta = np.arctan2(np.sin(delta), np.cos(delta))
        bend = p['bend_modulus_pa'] * p['shell_thickness_m']**3
        return dict(cloth_gravity_j=float(np.sum(self.node_mass*9.81*points[:, 2])),
                    cloth_kinetic_j=float(.5*np.sum(self.node_mass*np.sum(velocities**2, axis=1))),
                    membrane_proxy_j=float(np.sum(density*self.area*p['shell_thickness_m'])),
                    bending_proxy_j=float(.5*bend*np.sum(self.bend_weights*delta**2)))


def rigid_energy(poses, velocities, joints, dynamics):
    """Use native masses, body-local inertias, COM offsets and world velocities."""
    masses = np.array(dynamics['masses_kg'][0])
    rotations = Rotation.from_quat(poses[:, 3:])
    center = poses[:, :3] + rotations.apply(np.array(dynamics['local_com_poses_xyzw'][0])[:, :3])
    omega = rotations.inv().apply(velocities[:, 3:])
    inertia = np.array(dynamics['inertias_kg_m2'][0]).reshape(-1, 3, 3)
    clip = dynamics['joint_names'].index('ClipFlexure')
    stiffness = dynamics['joint_stiffnesses'][0][clip]
    return dict(rigid_gravity_j=float(np.sum(masses*9.81*center[:, 2])),
                rigid_kinetic_j=float(.5*np.sum(masses*np.sum(velocities[:, :3]**2, axis=1))
                    + .5*np.einsum('bi,bij,bj->', omega, inertia, omega)),
                clip_spring_j=float(.5*stiffness*joints[clip]**2))


def diagnose(run, seconds=1., stride=4):
    def read(name): return json.loads((run/name).read_text())
    request, init, dynamics = read('request.json'), read('release-probe-initialization.json'), read('frame-dynamics.json')
    assert request.get('release_probe_hands_clear') and not request.get('release_probe_without_cover')
    g = np.load(run/'geometry.npz')
    surface = SurfaceEnergy(g['rest'], g['faces'], read('effective-parameters.json'))
    source = run.parent/init['source_run']
    source_hz = json.loads((source/'request.json').read_text())['physics_hz']
    index = (init['source_control_step']+1)*(source_hz//20)-1
    with np.load(source/'physics-chunks'/f'{index//source_hz:05d}.npz') as original:
        poses = original['frame_body_poses'][index%source_hz]
        joints = original['frame_joint_positions'][index%source_hz]
    first = dict(time_s=0., **surface.measure(g['initial'], np.zeros_like(g['initial'])),
                 **rigid_energy(poses, np.zeros((len(poses), 6)), joints, dynamics))
    rows = [first]
    files = ['request.json', 'release-probe-initialization.json', 'frame-dynamics.json', 'geometry.npz', 'effective-parameters.json']
    hz = request['physics_hz']
    for chunk_path in sorted((run/'physics-chunks').glob('*.npz')):
        start = int(chunk_path.stem)*hz
        if start >= seconds*hz:
            break
        files.append(str(chunk_path.relative_to(run)))
        with np.load(chunk_path) as chunk:
            for offset in range(0, min(len(chunk['positions']), round(seconds*hz)-start), stride):
                rows.append(dict(time_s=(start+offset+1)/hz,
                    **surface.measure(chunk['positions'][offset], chunk['velocities'][offset]),
                    **rigid_energy(chunk['frame_body_poses'][offset], chunk['frame_body_velocities'][offset],
                                   chunk['frame_joint_positions'][offset], dynamics)))
    for row in rows:
        row['total_proxy_j'] = sum(value for name, value in row.items() if name.endswith('_j'))
    report = dict(diagnostic_only=True, benchmark_eligible=False, rows=rows,
                  initial_energy_terms=first, maximum_total_proxy_increase_j=max(r['total_proxy_j'] for r in rows)-first['total_proxy_j'],
                  maximum_rigid_gravity_increase_j=max(r['rigid_gravity_j'] for r in rows)-first['rigid_gravity_j'],
                  cloth_lumped_mass_kg=float(surface.node_mass.sum()),
                  limitations=['Membrane is a plane-stress St Venant-Kirchhoff estimate; bending is a discrete hinge estimate using stiffness times thickness cubed. Neither is native solver energy readback.',
                               'Contact correction work and numerical dissipation are not directly measured. A positive proxy balance is a diagnostic lead, not a complete conservation proof.'],
                  source_sha256={name: hashlib.sha256((run/name).read_bytes()).hexdigest() for name in files},
                  assessor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (run/'release-energy.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({key: value for key, value in report.items() if key not in ('rows','source_sha256')}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--seconds', type=float, default=1.)
    args = parser.parse_args()
    diagnose(args.run, args.seconds)
