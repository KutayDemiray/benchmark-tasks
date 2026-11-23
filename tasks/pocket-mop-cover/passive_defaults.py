"""Author one explicit native-default comparison before passive diagnostics."""

import hashlib
import json


def cloth_correction(stage, out, mode):
    """Set only the cloth contact correction bound and verify the layer change."""
    from pxr import Sdf
    assert mode in ('bounded', 'unbounded')
    prim = stage.GetPrimAtPath('/World/Pocket')
    name = 'physxDeformableBody:maxDepenetrationVelocity'
    attr = prim.GetAttribute(name)
    assert not attr or not attr.HasAuthoredValueOpinion(), 'Expected an implicit source default'
    before = stage.GetRootLayer().ExportToString()
    speed = 3. if mode == 'bounded' else float('inf')
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Float, custom=False).Set(speed)
    after = stage.GetRootLayer().ExportToString()
    assert prim.RemoveProperty(name)
    assert stage.GetRootLayer().ExportToString() == before, 'Unexpected physical scene edits'
    prim.CreateAttribute(name, Sdf.ValueTypeNames.Float, custom=False).Set(speed)
    assert stage.GetRootLayer().ExportToString() == after
    report = dict(diagnostic_only=True, mode=mode, attribute=name,
                  path=str(prim.GetPath()), value_m_s=3. if mode == 'bounded' else 'inf',
                  only_one_authored_attribute_changed=True,
                  before_layer_sha256=hashlib.sha256(before.encode()).hexdigest(),
                  after_layer_sha256=hashlib.sha256(after.encode()).hexdigest(),
                  physical_velocity_limit_changed=False, material_parameters_changed=False,
                  rationale='Reciprocal test of native Kit=3 m/s versus installed SDK=unbounded contact correction, observed in OmniPVD.')
    (out/'native-default-override.json').write_text(json.dumps(report, indent=2))
    return report


def match_kit_defaults(stage, out):
    """Explicitly author the Kit defaults independently observed in R158b."""
    from pxr import Sdf
    overrides = {
        '/World/Pocket': {
            'physxDeformableBody:maxDepenetrationVelocity': 3.,
            'physxDeformableBody:linearDamping': .005,
            'physxDeformableBody:sleepThreshold': .005,
        },
        '/World/Frame': {
            'physxArticulation:sleepThreshold': .005,
            'physxArticulation:stabilizationThreshold': 1.,
        },
    }
    before = stage.GetRootLayer().ExportToString()
    for path, properties in overrides.items():
        prim = stage.GetPrimAtPath(path)
        assert prim
        for name, value in properties.items():
            attr = prim.GetAttribute(name)
            assert not attr or not attr.HasAuthoredValueOpinion(), 'Expected an implicit source default'
            prim.CreateAttribute(name, Sdf.ValueTypeNames.Float, custom=False).Set(value)
    after = stage.GetRootLayer().ExportToString()
    for path, properties in overrides.items():
        for name in properties:
            assert stage.GetPrimAtPath(path).RemoveProperty(name)
    assert stage.GetRootLayer().ExportToString() == before, 'Unexpected scene changes'
    for path, properties in overrides.items():
        for name, value in properties.items():
            stage.GetPrimAtPath(path).CreateAttribute(name, Sdf.ValueTypeNames.Float, custom=False).Set(value)
    assert stage.GetRootLayer().ExportToString() == after
    report = dict(diagnostic_only=True, mode='match_recorded_kit_defaults', overrides=overrides,
                  source_native_capture='passive-r158b-kit-properties',
                  only_listed_attributes_changed=True,
                  before_layer_sha256=hashlib.sha256(before.encode()).hexdigest(),
                  after_layer_sha256=hashlib.sha256(after.encode()).hexdigest(),
                  native_readback_required=True, ordinary_task_changed=False)
    (out/'native-default-override.json').write_text(json.dumps(report, indent=2))
    return report


def match_kit_joint_velocity_defaults(out, binding, read):
    """Set the recorded, effectively unlimited Kit bounds before state restoration."""
    import numpy as np
    from compare_native_properties import select

    capture = out.parent/'passive-r158b-kit-properties'/'native-properties.json'
    native = select(json.loads(capture.read_text()))
    requested = []
    for name in binding.dof_names:
        kind, properties = native['/World/Frame/'+name]
        assert kind == 'PxArticulationJointReducedCoordinate'
        limit = properties['maxJointVelocity']
        assert limit > 1e30 and properties['maxJointDofVelocity'][0] == limit
        requested.append(limit)
    target = np.asarray([requested], dtype=np.float32)
    assert target.shape == binding.shape
    before = read(binding)
    binding.write(target)
    after = read(binding)
    assert np.array_equal(after, target), 'Native joint velocity defaults did not update'
    path = out/'native-default-override.json'
    report = json.loads(path.read_text())
    report['native_joint_velocity_defaults'] = dict(
        api='ARTICULATION_DOF_MAX_VELOCITY', units='rad/s',
        native_joint_order=binding.dof_names, before=before.tolist(),
        requested=target.tolist(), readback=after.tolist(),
        source_capture_sha256=hashlib.sha256(capture.read_bytes()).hexdigest(),
        applied_before_initial_state_restoration=True,
        rationale='The legacy USD attribute retained the SDK float32-max sentinel. The documented native property setter matches the original Kit bound; both are effectively unlimited.')
    path.write_text(json.dumps(report, indent=2))
