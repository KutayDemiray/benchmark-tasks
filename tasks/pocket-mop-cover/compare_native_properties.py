"""Compare explicitly selected native physical properties from two OmniPVD captures."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


FIELDS = {
    'PxScene': 'gravity flags frictionType broadPhaseType solverType bounceThresholdVelocity frictionOffsetThreshold frictionCorrelationDistance solverBatchSize solverArticulationBatchSize ccdMaxPasses ccdThreshold ccdMaxSeparation gpuMaxNumPartitions gpuMaxNumStaticPartitions tolerancesScale',
    'PxArticulationReducedCoordinate': 'positionIterations velocityIterations sleepThreshold stabilizationThreshold articulationFlags dofs',
    'PxArticulationLink': 'flags cMassLocalPose mass massSpaceInertiaTensor linearDamping angularDamping rigidBodyFlags maxLinearVelocity maxAngularVelocity maxDepenetrationVelocity maxContactImpulse contactSlopCoefficient',
    'PxArticulationJointReducedCoordinate': 'type motion parentTranslation parentRotation childTranslation childRotation armature frictionCoefficient staticFrictionEffort dynamicFrictionEffort viscousFrictionCoefficient limitLow limitHigh driveStiffness driveDamping driveMaxForce driveType driveTarget driveVelocity maxJointVelocity maxJointDofVelocity',
    'PxDeformableSurface': 'flags deformableBodyFlags linearDamping maxLinearVelocity maxDepenetrationVelocity selfCollisionFilterDistance solverIterationCount sleepThreshold settlingThreshold settlingDamping nbCollisionPairUpdatesPerTimestep nbCollisionSubsteps nbVertices',
    'PxDeformableSurfaceMaterial': 'youngsModulus poissons dynamicFriction elasticityDamping thickness bendingStiffness bendingDamping',
    'PxArticulationMimicJoint': 'axisA axisB gearRatio offset naturalFrequency dampingRatio',
}


def select(data):
    """Identify recorded objects by USD name or their single relevant native class."""
    selected = {}
    for obj in data['objects']:
        kind = obj['class_name']
        if kind not in FIELDS or not obj.get('created', True):
            continue
        properties = {key: entry['last'] for key, entry in obj['properties'].items()}
        # Default fallback materials are not assigned to this task cloth.
        if kind == 'PxDeformableSurfaceMaterial' and properties['youngsModulus'] != 2e6:
            continue
        key = properties.get('name', kind)
        assert key not in selected, f'Ambiguous native object {key}'
        selected[key] = (kind, properties)
    return selected


def same(a, b):
    """Compare enum storage exactly and native float properties with roundoff tolerance."""
    if isinstance(a, dict) or isinstance(b, dict):
        return a == b
    if isinstance(a, (int, float, list)) and isinstance(b, (int, float, list)):
        return np.shape(a) == np.shape(b) and bool(np.allclose(a, b, rtol=1e-6, atol=1e-10, equal_nan=False))
    return a == b


def safe(value):
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    if isinstance(value, list):
        return [safe(item) for item in value]
    if isinstance(value, dict):
        return {key: safe(item) for key, item in value.items()}
    return value


def compare(first, second):
    """Write a source-hashed comparison without treating it as a qualification gate."""
    paths = [run/'native-properties.json' for run in (first, second)]
    data = [json.loads(path.read_text()) for path in paths]
    a, b = [select(record) for record in data]
    rows = []
    assert a.keys() == b.keys(), 'Different native objects in comparison'
    for name, (kind, properties) in a.items():
        assert b[name][0] == kind
        for key in FIELDS[kind].split():
            left, right = properties.get(key), b[name][1].get(key)
            matched = same(left, right) if key in properties and key in b[name][1] else None
            if key == 'cMassLocalPose' and matched is not None:
                matched = bool(np.allclose(left[:4], right[:4], rtol=0., atol=1e-6) and
                               np.allclose(left[4:], right[4:], rtol=0., atol=1e-8))
            rows.append(dict(object=name, native_class=kind, property=key,
                             first=safe(left), second=safe(right),
                             recorded_in_both=key in properties and key in b[name][1],
                             same=matched))
    versions = []
    for record in data:
        metadata = [obj for obj in record['objects'] if obj['class_name'] == 'PxOmniPvdMetaData']
        assert len(metadata) == 1
        versions.append({key: value['last'] for key, value in metadata[0]['properties'].items()})
    report = dict(diagnostic_only=True, benchmark_eligible=False,
                  first_run=first.name, second_run=second.name, native_versions=versions,
                  comparison_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                  matched=sum(row['same'] is True for row in rows),
                  different=sum(row['same'] is False for row in rows),
                  unrecorded=sum(not row['recorded_in_both'] for row in rows), properties=rows,
                  limitations=['Only published properties listed explicitly in FIELDS are compared.',
                               'Native engine versions differ; matching these properties does not prove solver equivalence.',
                               'Native state trajectories, unused fallback materials and pointer-valued object handles are excluded.'])
    (second/'native-property-comparison.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key not in ('properties','source_sha256')}, indent=2))
    print(json.dumps([row for row in rows if row['same'] is False], indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    args = parser.parse_args()
    compare(args.first, args.second)
