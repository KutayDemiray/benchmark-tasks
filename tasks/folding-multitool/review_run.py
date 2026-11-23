"""CPU-only review of recorded physics; does not launch or alter a simulation.

Relative grasp motion is reported per stage and contact interval. It is a
diagnostic, not a replacement for reviewing the video and complete contact trace.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from yam_arena import ik
from yam_arena.urdf import quat_xyzw_to_matrix
from mechanism import BODIES
from scorer import DeploymentScorer


def review(directory):
    directory = Path(directory)
    rows = [json.loads(line) for line in (directory/"trace.jsonl").read_text().splitlines()]
    scorer = DeploymentScorer()
    stages = {}
    for m in rows:
        scorer.update(m)
        s = stages.setdefault(m["stage"], {"first_step": m["step"], "last_step": m["step"],
                    "nonfinger_contact_peak_n": 0., "grasp_intervals": [], "_active": {}})
        s["last_step"] = m["step"]
        for side, offset in (("left", 0), ("right", 7)):
            for name, force in zip(m[side+"_contact_names"], m[side+"_contact_forces_n"]):
                if name not in ("tip_left", "tip_right"):
                    s["nonfinger_contact_peak_n"] = max(s["nonfinger_contact_peak_n"], float(np.linalg.norm(force)))
            if "robot_state" not in m:
                continue
            q = np.array(m["robot_state"][offset:offset+6])
            pos, rot = ik.tcp_pose(q, [0., .305 if side == "left" else -.305, .75])
            p = np.array(m["pads_n"])[(0 if side == "left" else 2):(2 if side == "left" else 4)]
            for bi, body in enumerate(BODIES):
                key = side+":"+body
                if min(p[:, bi]) <= .1:
                    s["_active"].pop(key, None)
                    continue
                bp = m["link_poses"][body]
                br = quat_xyzw_to_matrix(bp[3:])
                point, relative_rot = br.T@(pos-np.array(bp[:3])), br.T@rot
                if key not in s["_active"]:
                    interval = {"side": side, "body": body, "first_step": m["step"],
                                "last_step": m["step"], "max_translation_mm": 0., "max_rotation_deg": 0.}
                    s["grasp_intervals"].append(interval)
                    s["_active"][key] = (point, relative_rot, interval)
                p0, r0, interval = s["_active"][key]
                interval["last_step"] = m["step"]
                interval["max_translation_mm"] = max(interval["max_translation_mm"], float(1000*np.linalg.norm(point-p0)))
                interval["max_rotation_deg"] = max(interval["max_rotation_deg"], float(np.degrees(np.linalg.norm(ik._rot_error(relative_rot, r0)))))
    for s in stages.values():
        del s["_active"]
    return {"source_run": str(directory), "rescored_with_current_code": scorer.snapshot(), "stages": stages,
            "note": "Per-contact-interval motion can omit slip across contact loss. Acquisition and release involve intentional motion. Inspect traces/video."}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--out")
    args = parser.parse_args()
    result = review(args.run)
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2)+"\n")
    else:
        print(json.dumps(result, indent=2))
