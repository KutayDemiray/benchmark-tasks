"""Explicit pre-simulation numerical experiments; never a benchmark configuration."""

import functools
import json

import numpy as np
from scipy.spatial import ConvexHull


def redundant_clamps(metadata):
    """Prove each removed convex solid is contained in its retained same-body end."""
    proof = []
    for body in ('Left', 'Right'):
        shapes = {shape['name']: shape for shape in metadata['colliders'] if shape['body']==body}
        end, clamp = shapes['End'], shapes['ClosedStripClamp']
        assert end['enabled'] and clamp['enabled']
        planes = ConvexHull(np.asarray(end['points'])).equations
        distances = np.asarray(clamp['points']) @ planes[:,:3].T + planes[:,3]
        maximum = float(distances.max())
        assert maximum <= 1e-10, 'Removing this collider would change the solid boundary'
        proof.append(dict(body=body,removed='ClosedStripClamp',retained='End',
                          maximum_vertex_outside_retained_m=maximum,
                          convex_union_unchanged=True))
    return proof


def install(request, out):
    """Author explicit diagnostic comparisons before native cooking."""
    mode = request.get('release_probe_depenetration')
    deduplicate = request.get('release_probe_deduplicate_colliders', False)
    velocity_iterations = request.get('release_probe_velocity_iterations')
    no_ccd = request.get('release_probe_no_speculative_ccd',False)
    if not any((mode,deduplicate,velocity_iterations is not None,no_ccd)):
        return
    assert request.get('release_probe_run') and request.get('release_probe_hands_clear')
    assert mode in (None, 'frame', 'cloth', 'both')
    assert not request.get('release_probe_without_cover')
    speed = request.get('release_probe_depenetration_speed', .1)
    assert speed in (.1, .2, 3.)
    assert velocity_iterations in (None,0,1,32)
    from pxr import PhysxSchema, Usd, UsdPhysics
    import frame_native
    import native

    report = dict(diagnostic_only=True, benchmark_eligible=False, subsystem=mode,
                  maximum_depenetration_speed_m_s=speed if mode else None, authored_before_simulation=True,
                  frame_velocity_iterations=velocity_iterations, speculative_ccd_disabled=no_ccd,
                  changes=[], rationale='Constant contact-position correction limit; this is not a velocity clamp or material damping. At 480 Hz, 0.1 m/s can correct a 1 mm overlap over about five timesteps. The 0.2 m/s test checks sensitivity; 3 m/s reproduces the rigid schema default. No benchmark use without fresh qualification.')

    def save(prim, attribute, value):
        attr = prim.GetAttribute(attribute)
        assert attr, attribute
        before = attr.Get()
        assert attr.Set(value)
        report['changes'].append(dict(path=str(prim.GetPath()), attribute=attribute,
                                      before=before if before < float('inf') else 'inf', after=attr.Get()))
        (out / 'release-physics-overrides.json').write_text(json.dumps(report, indent=2))

    if mode in ('frame', 'both') or deduplicate or velocity_iterations is not None:
        original_frame = frame_native.author_frame

        @functools.wraps(original_frame)
        def author_frame(stage, *args, **kwargs):
            metadata = original_frame(stage, *args, **kwargs)
            if mode in ('frame', 'both'):
                for prim in Usd.PrimRange(stage.GetPrimAtPath(metadata['path'])):
                    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
                        PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                        save(prim, 'physxRigidBody:maxDepenetrationVelocity',speed)
            if velocity_iterations is not None:
                save(stage.GetPrimAtPath(metadata['path']),
                     'physxArticulation:solverVelocityIterationCount',velocity_iterations)
            if deduplicate:
                report['redundant_collider_proof'] = redundant_clamps(metadata)
                for item in report['redundant_collider_proof']:
                    path = metadata['path']+'/'+item['body']+'/'+item['removed']
                    prim = stage.GetPrimAtPath(path)
                    assert UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
                # Audit geometry keeps the same solid union, including the
                # redundant subset, so its conservative checks are unchanged.
                (out/'release-physics-overrides.json').write_text(json.dumps(report,indent=2))
            return metadata

        frame_native.author_frame = author_frame
    if mode in ('cloth', 'both') or no_ccd:
        original_cloth = native.cloth

        @functools.wraps(original_cloth)
        def cloth(*args, **kwargs):
            mesh = original_cloth(*args, **kwargs)
            if mode in ('cloth', 'both'):
                save(mesh.GetPrim(), 'physxDeformableBody:maxDepenetrationVelocity',speed)
            if no_ccd:
                save(mesh.GetPrim(), 'physxDeformableBody:enableSpeculativeCCD',False)
            return mesh

        native.cloth = cloth


def configure_scene(request, cfg):
    """Change solver ordering only for an explicitly initialized diagnostic."""
    if not request.get('release_probe_contact_last'):
        return
    assert request.get('release_probe_run') and request.get('release_probe_hands_clear')
    assert not request.get('release_probe_without_cover')
    cfg.sim.physics.solve_articulation_contact_last = True


def verify_scene(request, stage, path, out):
    """Record the authored solver order without modifying the running scene."""
    if not request.get('release_probe_contact_last'):
        return
    value = stage.GetPrimAtPath(path).GetAttribute('physxScene:solveArticulationContactLast').Get()
    assert value is True
    (out/'release-scene-numerics.json').write_text(json.dumps(dict(
        diagnostic_only=True, benchmark_eligible=False, configured_before_simulation=True,
        attribute='physxScene:solveArticulationContactLast', authored_value=value,
        contact_correction_limits_changed=bool(request.get('release_probe_depenetration')),
        rationale='NVIDIA documents later dynamic contact resolution for stiff articulated contact systems; no material, geometry, robot gain or scoring change.'),indent=2))
