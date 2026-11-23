"""Offline stock-gripper clearance samples; not a dynamic grasp proof."""
import argparse
import json
from pathlib import Path
import numpy as np
from controller import working_grasp
from mechanism import load_spec, link_transforms, transform, jaw_angle_for_gap
from geometry import tool_links
from motion import orientation
from stock_geometry import gripper_meshes, placed, overlaps


def inspect():
    spec, records = load_spec(), []
    pieces = tool_links(spec)

    def sample(label, q, grasps):
        links = link_transforms(spec, q)
        tool = {b: placed([(b+"/"+p.name, p.vertices) for p in ps if p.collision], links[b])
                for b, ps in pieces.items()}
        hands = []
        record = {"label": label, "q": q, "unintended_overlaps": []}
        for side, body, local, gap in grasps:
            meshes = placed(gripper_meshes(gap), links[body]@local)
            hands.append(meshes)
            collisions = overlaps(meshes, sum([v for b, v in tool.items() if b != body], []))
            record["unintended_overlaps"] += [{"side": side, **c} for c in collisions]
        record["inter_gripper_overlaps"] = overlaps(*hands)
        records.append(record)

    for i, angle in enumerate(np.linspace(-np.pi, -np.pi/2, 10)):
        q = dict(jaw=-.04, fold_a=0., fold_b=float(angle))
        sample(f"head_support_{i}", q, [
            ("left", "handle_b", transform([-.030, -.008, 0], orientation([0, 0, 1], [0, 1, 0])), .01511),
            ("right", "jaw_b", transform([.020, -.016, 0], orientation([0, 0, -1], [0, 1, 0])), .0065)])
    for gap in (.0007, .010, .016):
        q = dict(jaw=jaw_angle_for_gap(spec, gap), fold_a=0., fold_b=0.)
        sample(f"working_gap_{gap}", q, [("left", "handle_b", working_grasp("handle_b"), .01511),
                                         ("right", "handle_a", working_grasp("handle_a"), .01511)])
    for angle in np.linspace(-np.pi/2, -.45, 6):
        q = dict(jaw=0., fold_a=0., fold_b=float(angle))
        sample(f"width_to_working_{angle:.3f}", q, [
            ("left", "handle_b", transform([-.030, 0., .003], orientation([0, -1, 0], [.6, 0, -.8])), .014972),
            ("right", "jaw_b", transform([.020, -.016, 0], orientation([0, 0, -1], [0, 1, 0])), .0065)])
    for angle in np.linspace(-.45, 0., 6):
        q = dict(jaw=0., fold_a=0., fold_b=float(angle))
        sample(f"finish_b_working_{angle:.3f}", q, [
            ("left", "handle_b", working_grasp("handle_b"), .01511),
            ("right", "jaw_b", transform([.020, -.016, 0], orientation([0, 0, -1], [0, 1, 0])), .0065)])
    return {"offline_only": True, "records": records,
            "pass": not any(r["unintended_overlaps"] or r["inter_gripper_overlaps"] for r in records),
            "scope": "Actual stock gripper/finger convex collision meshes. Nominal static poses, 0.05 mm penetration tolerance. Does not prove full-arm swept clearance, friction, or grasp stability."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    result = inspect()
    Path(args.out).write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({"pass": result["pass"], "samples": len(result["records"]), "out": args.out}))
    raise SystemExit(0 if result["pass"] else 1)
