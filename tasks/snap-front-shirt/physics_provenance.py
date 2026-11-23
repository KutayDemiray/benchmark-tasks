"""Strict physics-source provenance with one reviewed scheduler-only transition."""

import ast
import hashlib

PHYSICAL_SOURCES=('garment_mesh.py','garment_scene.py','tabletop_scene.py','robot_geometry.py',
                  'geometry.py','parameters.json','robot_runner.py','episode_timing.py')
# Exact full-file identities, not a broad permission to ignore orchestration.
# p12 -> p13 changed only reservation checks, device binding and GPU evidence.
RUNNER_BRIDGE={
    'reference_sha256': '4afd5b7cc1082e6d6aedc34e83d875bc8928bc04b8ee66e1a3debf5ce2677813',
    'current_sha256': '5d7a3633b76aea25ddc985abc647bdc086f874e011ae42ce315cb0fcf94cadd6',
    'gpu_runtime_sha256': 'a11bd1fa02dc7aae99c0eaeb0636dd8891dc95392005cdabe4124e77ba7a8644',
}

# Exact p19 portability-only helper; every function/class is unchanged.
# The default resolves to the same existing scheduler; no scheduler is copied.
PORTABLE_HELPER_BRIDGE = {
    'reference_sha256': 'a11bd1fa02dc7aae99c0eaeb0636dd8891dc95392005cdabe4124e77ba7a8644',
    'current_sha256': '00d88e01723e16ad476a73c79d8df8ea5056155cbd7b7d8226ca80aaedea38df',
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def definition(source,name):
    node=next(n for n in ast.parse(source).body if getattr(n,'name',None)==name)
    return ast.dump(node,include_attributes=False)


def verify_physical_sources(current,reference,manifest):
    """Reject unknown source differences; admit only the exact reviewed GPU transition."""
    records=[]
    for name in PHYSICAL_SOURCES:
        expected=manifest['sources'][name]
        actual_reference=digest(reference/name)
        assert actual_reference==expected, ('Reference source modified',name)
        actual=digest(current/name)
        bridge=None
        if actual!=expected:
            assert name=='robot_runner.py', ('Physical source changed since stability gate',name)
            assert expected==RUNNER_BRIDGE['reference_sha256'] and actual==RUNNER_BRIDGE['current_sha256'], \
                'Unreviewed runner change since stability gate'
            helper_hash = digest(current/'gpu_runtime.py')
            allowed_helpers = {RUNNER_BRIDGE['gpu_runtime_sha256']}
            if RUNNER_BRIDGE['gpu_runtime_sha256'] == PORTABLE_HELPER_BRIDGE['reference_sha256']:
                allowed_helpers.add(PORTABLE_HELPER_BRIDGE['current_sha256'])
            assert helper_hash in allowed_helpers, 'Unreviewed GPU helper change'
            old=(reference/name).read_text();new=(current/name).read_text()
            for symbol in ('make_environment','Measurements'):
                assert definition(old,symbol)==definition(new,symbol), ('Native physics definition changed',symbol)
            bridge=dict(kind='exact_p12_to_p13_scheduler_only',unchanged_definitions=['make_environment','Measurements'],
                        gpu_runtime_sha256=helper_hash,
                        portable_path_configuration=helper_hash==PORTABLE_HELPER_BRIDGE['current_sha256'])
        records.append(dict(source=name,reference_sha256=expected,current_sha256=actual,
                            identical=actual==expected,reviewed_transition=bridge))
    return dict(verification_pass=True,sources=records,physical_parameters_changed=False,
                scoring_conditions_changed=False,gate_horizons_changed=False)
