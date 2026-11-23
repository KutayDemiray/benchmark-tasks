"""Offline geometry validation, not a physics experiment or feasibility proof."""
import itertools
import json
import argparse
from pathlib import Path
import numpy as np
from mechanism import load_spec, reset_joints, link_transforms, fingerprint
from geometry import tool_links, bounds, mass_properties, separating_axis_gap


def inspect(spec):
    links = tool_links(spec)
    poses = [("folded", reset_joints(spec)), ("deployed", dict(jaw=0., fold_a=0., fold_b=0.))]
    for a in np.linspace(0, np.pi, 19):
        poses.append((f"fan_{np.degrees(a):.0f}", dict(jaw=0., fold_a=float(a), fold_b=-float(a))))
        poses.append((f"open_a_{np.degrees(a):.0f}", dict(jaw=0., fold_a=float(a), fold_b=-np.pi)))
        poses.append((f"open_b_{np.degrees(a):.0f}", dict(jaw=0., fold_a=0., fold_b=-float(a))))
    for j in (0., -.1, -.2):
        poses.append((f"one_handle_head_motion_{j:.1f}", dict(jaw=j, fold_a=0., fold_b=-np.pi)))
    # A deliberately large jaw angle with B still folded is an access probe,
    # not a commanded path: the folded spine should physically block it.
    poses.append(("blocked_access_probe", dict(jaw=-.3, fold_a=0., fold_b=-np.pi)))
    for j in np.linspace(-.7, 0, 8):
        poses.append((f"jaw_{np.degrees(j):.0f}", dict(jaw=float(j), fold_a=0., fold_b=0.)))
    collisions = []
    for label, q in poses:
        transforms = link_transforms(spec, q)
        vertices = {body: [(p.name, p.vertices@transforms[body][:3, :3].T+transforms[body][:3, 3])
                          for p in pieces if p.collision] for body, pieces in links.items()}
        for a, b in itertools.combinations(vertices, 2):
            for an, av in vertices[a]:
                for bn, bv in vertices[b]:
                    if np.any(av.max(0) < bv.min(0)) or np.any(bv.max(0) < av.min(0)):
                        continue
                    gap = separating_axis_gap(av, bv)
                    if gap < -.00005:
                        collisions.append(dict(pose=label, a=a+"/"+an, b=b+"/"+bn, penetration_mm=-gap*1000))
    result = {"mechanism_hash": fingerprint(spec), "offline_only": True,
              "sampled_poses": len(poses),
              "bounds_m": {label: (bounds(spec, q)[1]-bounds(spec, q)[0]).tolist() for label, q in poses[:2]},
              "masses_and_inertias": {},
              "self_intersections": [c for c in collisions if c["pose"] != "blocked_access_probe"],
              "access_probe_intersections": [c for c in collisions if c["pose"] == "blocked_access_probe"],
              "access_note": "At jaw=-0.3 rad with B folded, jaw A strikes B's spine. This is a physical access limit; the controller must not demand this pose."}
    for body, pieces in links.items():
        m = spec["body_masses_kg"][body]
        com, inertia = mass_properties(pieces, m)
        result["masses_and_inertias"][body] = {"mass_kg": m, "com_m": com.tolist(), "inertia_kgm2": inertia.tolist()}
    result["pass"] = not result["self_intersections"]
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out")
    args = parser.parse_args()
    result = inspect(load_spec())
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "masses_and_inertias"}, indent=2))
    raise SystemExit(0 if result["pass"] else 1)
