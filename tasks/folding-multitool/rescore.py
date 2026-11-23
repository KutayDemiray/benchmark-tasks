"""Independently rescore a trace in a new CPU process with its frozen source.

Reconstructs scorer state from measured geometry/contacts only. Controller
stage labels, saved scores and requested actions are not inputs to the scorer.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def rescore(directory):
    directory = Path(directory).resolve()
    meta = json.loads((directory/"metadata.json").read_text())
    result = json.loads((directory/"result.json").read_text())
    for name, digest in meta["hashes"].items():
        assert hashlib.sha256((directory/"source"/name).read_bytes()).hexdigest() == digest, name
    if "scorer" in sys.modules:
        raise RuntimeError("Rescoring needs a fresh process to import this run's frozen scorer")
    sys.path.insert(0, str(directory/"source"))
    from scorer import DeploymentScorer
    scorer = DeploymentScorer()
    milestones, differences = {}, []
    count = 0
    for i, line in enumerate((directory/"trace.jsonl").open()):
        m = json.loads(line)
        assert m["step"] == i, ("non-contiguous trace", i, m["step"])
        recorded = m.pop("scoring", None)
        for ignored in ("stage", "action", "applied_action"):
            m.pop(ignored, None)
        scorer.update(m)
        now = scorer.snapshot()
        if recorded is not None and recorded != now:
            differences.append(i)
        for name in ("folded_reset_seen", "deployment_cycle_pass", "jaw_extraction_seen", "jaw_carry_pass", "retrieval_pass"):
            if now.get(name):
                milestones.setdefault(name, i)
        count = i
    assert count == result["steps"], (count, result["steps"])
    assert count <= meta.get("max_steps", 3600) <= 3600
    passed = not differences and scorer.snapshot() == result["scoring"]
    return {"passed": passed, "source_run": str(directory), "source_hashes_verified": True,
            "scorer_imported_from": str(Path(sys.modules["scorer"].__file__).resolve()),
            "trace_sha256": hashlib.sha256((directory/"trace.jsonl").read_bytes()).hexdigest(),
            "continuous_actions": count, "scoring": scorer.snapshot(), "first_step": milestones,
            "recorded_score_mismatch_steps": differences,
            "note": "Independent CPU rescore of frozen measured trace; does not replace visual/physical review."}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run")
    p.add_argument("--out")
    args = p.parse_args()
    result = rescore(args.run)
    destination = Path(args.out) if args.out else Path(args.run)/"independent-rescore.json"
    destination.write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
