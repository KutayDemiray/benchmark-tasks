"""Constant cloth contact correction, requiring fresh isolated material qualification."""

import functools
import json


def configuration(request):
    """Return the declared cloth setting; None preserves the native default."""
    speed = request.get('cloth_contact_correction_speed')
    assert speed in (None, .1)
    if speed is not None:
        assert request['experiment'] in ('probe', 'robot_grasp', 'mechanism', 'frame_robot')
        assert request['product_profile'] == 'base_micro' and not request.get('reinforced')
        assert not request.get('release_probe_run'), 'Cold diagnostic overrides have their own provenance'
        assert request.get('frame_contact_correction_speed') is None, 'Qualify each numerical change independently'
    return dict(maximum_depenetration_speed_m_s=speed)


def install(request, out):
    """Author contact correction before cooking; never alter physical node velocities."""
    settings = configuration(request)
    speed = settings['maximum_depenetration_speed_m_s']
    if speed is None:
        return
    import native
    original = native.cloth
    assert not getattr(original, '_constant_cloth_solver', False)

    @functools.wraps(original)
    def cloth(*args, **kwargs):
        mesh = original(*args, **kwargs)
        attr = mesh.GetPrim().GetAttribute('physxDeformableBody:maxDepenetrationVelocity')
        assert attr
        before = attr.Get()
        assert attr.Set(speed)
        assert abs(attr.Get()-speed) < 1e-7
        report = dict(configuration=settings, authored_before_simulation=True,
                      constant_throughout_episode=True, attribute=attr.GetName(),
                      before=before if before < float('inf') else 'inf', after=attr.Get(),
                      physical_velocity_limit_changed=False, material_damping_changed=False,
                      rigid_contact_settings_changed=False,
                      rationale='Limit deformable contact-position correction overshoot; preserve material constants and rigid latch contact resolution. Fresh isolated insertion/grasp gates and all penetration audits remain required.')
        (out/'cloth-numerics.json').write_text(json.dumps(report,indent=2))
        return mesh

    cloth._constant_cloth_solver = True
    native.cloth = cloth
