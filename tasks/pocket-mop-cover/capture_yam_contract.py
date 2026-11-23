"""Read the stock Kit YAM native properties without changing its physical state."""

import hashlib
import json
from pathlib import Path

import numpy as np


PROPERTIES = {
    'stiffness': 'get_dof_stiffnesses', 'damping': 'get_dof_dampings',
    'effort': 'get_dof_max_forces', 'velocity': 'get_dof_max_velocities',
    'friction_static_dynamic_viscous': 'get_dof_friction_properties',
    'limits_rad_or_m': 'get_dof_limits', 'ARTICULATION_DOF_ARMATURE': 'get_dof_armatures',
    'ARTICULATION_DOF_DRIVE_TYPE': 'get_drive_types',
    'ARTICULATION_BODY_MASS': 'get_masses', 'ARTICULATION_BODY_COM_POSE': 'get_coms',
    'ARTICULATION_BODY_INERTIA': 'get_inertias',
}


def capture(env, out):
    """Save native-order body/joint properties after the stock reset."""
    from robot_gate import array
    from yam_arena import constants
    report = {}
    for side in ('left', 'right'):
        robot = env.scene[side+'_robot']
        view = robot.root_view
        meta = view.shared_metatype
        values = {key: array(getattr(view, method)()).tolist() for key, method in PROPERTIES.items()}
        report[side] = dict(joint_names=list(meta.dof_names), body_names=list(meta.link_names),
            fixed_base=meta.fixed_base, native_properties=values,
            initial_native_q=array(view.get_dof_positions()).tolist(),
            initial_link_poses=array(view.get_link_transforms()).tolist(),
            initial_gravity_compensation=array(view.get_gravity_compensation_forces()).tolist())
        assert meta.fixed_base and set(meta.dof_names) == set(constants.ALL_JOINTS)
        assert np.isfinite(np.asarray(report[side]['initial_gravity_compensation'])).all()
    root = Path(constants.__file__).parent
    report['provenance'] = dict(backend='kit_physx', diagnostic_only=True, physical_state_modified=False,
        native_api='omni.physics.tensors.ArticulationView',
        installed_source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                                for name in ('constants.py', 'robot.py', 'actions.py')})
    (out/'native-yam-contract.json').write_text(json.dumps(report, indent=2))
    return report


def compare(reference, sdk, out):
    """Compare independent native readbacks in canonical joint/body order."""
    original = json.loads((reference/'native-yam-contract.json').read_text())
    assert original['provenance']['backend'] == 'kit_physx'
    assert original['provenance']['installed_source_sha256'] == sdk['contract']['installed_source_sha256']
    records = {}
    for side in ('left', 'right'):
        ref = original[side]
        observed = sdk[side]
        for key in PROPERTIES:
            names_key = 'body_names' if key.startswith('ARTICULATION_BODY_') else 'joint_names'
            ids = [observed[names_key].index(name) for name in ref[names_key]]
            value = np.asarray(observed['native_properties'][key])[:, ids]
            expected = np.asarray(ref['native_properties'][key])
            passed = value.shape == expected.shape and np.allclose(value, expected, atol=1e-7, rtol=1e-5)
            records[side+'/'+key] = dict(passed=bool(passed), maximum_absolute_difference=float(abs(value-expected).max()))
        ids = [observed['joint_names'].index(name) for name in ref['joint_names']]
        for key, tolerance in (('initial_native_q', 1e-6), ('initial_gravity_compensation', 1e-4)):
            value = np.asarray(observed[key])[:, ids]
            expected = np.asarray(ref[key])
            records[side+'/'+key] = dict(passed=bool(np.allclose(value, expected, atol=tolerance, rtol=1e-5)),
                maximum_absolute_difference=float(abs(value-expected).max()), absolute_tolerance=tolerance)
    result = dict(passed=all(value['passed'] for value in records.values()), properties=records,
        reference_run=reference.name, reference_sha256=hashlib.sha256((reference/'native-yam-contract.json').read_bytes()).hexdigest(),
        tolerance=dict(absolute=1e-7, relative=1e-5), independent_native_readbacks=True)
    (out/'sdk-yam-reference-comparison.json').write_text(json.dumps(result, indent=2))
    assert result['passed'], 'SDK robot differs from independent stock Kit native contract'
    return result
