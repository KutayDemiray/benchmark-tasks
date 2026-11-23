"""Read-only substep tracing of native reset settling, for scheduled diagnostics."""
import json
from pathlib import Path
import numpy as np


def host(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return value.numpy()


def capture_reset(emb, scene, seed, destination):
    """Run the ordinary reset unchanged, observing actual PhysX substeps."""
    emb._ensure_sim(None)
    base, sim = emb._base, emb._base.sim
    paths = {n: "/World/envs/env_0/"+n for n in ("nut", "bolted_coupon", "tray")}
    views = {n: sim.physics_sim_view.create_rigid_body_view(p) for n, p in paths.items()}
    contact = sim.physics_sim_view.create_rigid_contact_view(
        paths["nut"], filter_patterns=[paths["bolted_coupon"], paths["tray"]],
        max_contact_data_count=256)
    destination = Path(destination)
    properties = {}
    for name, view in views.items():
        entry = {"body_count": int(view.count), "shape_count": int(view.max_shapes)}
        for method in ("get_coms", "get_masses", "get_inertias", "get_contact_offsets",
                       "get_rest_offsets", "get_material_properties"):
            try:
                entry[method] = host(getattr(view, method)()).tolist()
            except Exception as exc:
                entry[method] = {"unavailable": str(exc)}
        properties[name] = entry
    (destination/"reset-physics-properties.json").write_text(json.dumps(properties, indent=2)+"\n")
    original_step = sim.step
    count = 0
    with (destination/"reset-substeps.jsonl").open("w") as log:
        def record(label):
            row = {"substep": count, "label": label, "physics_dt": base.physics_dt,
                   "bodies": {n: {"pose": host(v.get_transforms()).tolist(),
                                   "velocity": host(v.get_velocities()).tolist()} for n, v in views.items()}}
            try:
                row["fixture_forces_n"] = host(contact.get_contact_force_matrix(base.physics_dt)).tolist()
                force, point, normal, separation, number, start = [host(a) for a in contact.get_contact_data(base.physics_dt)]
                row["contacts"] = {}
                for i, name in enumerate(("bolted_coupon", "tray")):
                    begin, size = int(start[0, i]), int(number[0, i])
                    selected = slice(begin, begin+size)
                    row["contacts"][name] = {"count": size, "force": force[selected].tolist(),
                        "point": point[selected].tolist(), "normal": normal[selected].tolist(),
                        "separation": separation[selected].tolist()}
            except Exception as exc:
                row["contact_read_error"] = str(exc)
            log.write(json.dumps(row)+"\n")
            log.flush()

        def observed_step(*args, **kwargs):
            nonlocal count
            result = original_step(*args, **kwargs)
            count += 1
            record("after_native_physics_step")
            return result

        record("before_native_reset")
        sim.step = observed_step
        try:
            observation = emb.reset(scene, seed=seed)
            record("after_native_reset")
            return observation
        finally:
            sim.step = original_step
