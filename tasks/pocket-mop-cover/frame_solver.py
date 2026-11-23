"""Explicit constant frame contact settings, authored only before native cooking."""

import functools
import json


def configuration(request):
    """Return the declared frame correction setting; None preserves the native default."""
    speed = request.get('frame_contact_correction_speed')
    assert speed in (None, .1)
    if speed is not None:
        assert request['experiment'] in ('mechanism', 'frame_robot')
        assert not request.get('release_probe_run'), 'Cold diagnostic overrides have their own provenance'
    return dict(maximum_depenetration_speed_m_s=speed)


def install(request, out):
    """Apply a declared constant correction limit to all seven physical frame links."""
    settings = configuration(request)
    speed = settings['maximum_depenetration_speed_m_s']
    if speed is None:
        return
    from pxr import PhysxSchema, Usd, UsdPhysics
    import frame_native

    original = frame_native.author_frame
    assert not getattr(original, '_constant_frame_solver', False)

    @functools.wraps(original)
    def author_frame(stage, *args, **kwargs):
        metadata = original(stage, *args, **kwargs)
        changes = []
        for prim in Usd.PrimRange(stage.GetPrimAtPath(metadata['path'])):
            if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
                continue
            api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
            attr = api.GetMaxDepenetrationVelocityAttr()
            before = attr.Get()
            assert attr.Set(speed)
            assert abs(attr.Get()-speed) < 1e-7
            changes.append(dict(body=prim.GetName(), attribute=attr.GetName(),
                                before=before, after=attr.Get()))
        assert {item['body'] for item in changes} == set(metadata['body_masses_kg'])
        report = dict(configuration=settings, authored_before_simulation=True,
                      constant_throughout_episode=True, changes=changes,
                      physical_velocity_limit_changed=False, material_damping_changed=False,
                      rationale='Limit contact-position correction overshoot; preserve physical velocities, material constants, geometry and robot controls. Qualification and all penetration audits remain required.')
        (out/'frame-numerics.json').write_text(json.dumps(report, indent=2))
        return metadata

    author_frame._constant_frame_solver = True
    frame_native.author_frame = author_frame
