"""Rescore immutable measured traces with the revised scorer, without simulation.

Writes a separate report; never replaces recorded scores, source or old rescores.
Run each assessment in a new process. Synthetic tests are not demo evidence.
"""
import argparse
import json
from pathlib import Path
import sys
from scoring_revision import REVISION, digest, verify_revision

ROOT = Path(__file__).resolve().parent


def rescore(directory, role):
    directory = Path(directory).resolve()
    evidence = verify_revision(ROOT, require_reports=False)
    recorded = evidence["runs"][role]
    assert directory.name == recorded["run_id"], "Wrong original run"
    assert digest(directory/"metadata.json") == recorded["metadata_sha256"]
    assert digest(directory/"trace.jsonl") == recorded["trace_sha256"]
    meta = json.loads((directory/"metadata.json").read_text())
    result = json.loads((directory/"result.json").read_text())
    assert meta["contract"] == evidence["episode_contract"]
    # The compact submission manifest omits the original README/task text;
    # metadata still binds these documents, and every frozen file is checked.
    assert {k: v for k, v in meta["hashes"].items()
            if k not in {"README.md", "task.json"}} == recorded["source_hashes"]
    assert set(meta["hashes"]) == set(recorded["source_hashes"]) | {"README.md", "task.json"}
    for name, expected in meta["hashes"].items():
        assert digest(directory/"source"/name) == expected, name
    assert "scorer" not in sys.modules, "Use a fresh process for each assessment"
    from scorer import DeploymentScorer
    assert Path(sys.modules["scorer"].__file__).resolve() == ROOT/"scorer.py"
    scorer = DeploymentScorer()
    milestones, score_changes, snapshot_changes = {}, [], []
    count = -1
    with (directory/"trace.jsonl").open() as lines:
        for i, line in enumerate(lines):
            m = json.loads(line)
            assert m["step"] == i, ("non-contiguous trace", i, m["step"])
            old = m.pop("scoring")
            for ignored in ("stage", "action", "applied_action"):
                m.pop(ignored, None)
            scorer.update(m)
            now = scorer.snapshot()
            if now != old:
                snapshot_changes.append(i)
            if now["score"] != old["score"]:
                score_changes.append(i)
            for name in ("folded_reset_seen", "deployment_cycle_pass", "jaw_extraction_seen",
                         "jaw_carry_pass", "retrieval_pass"):
                if now.get(name):
                    milestones.setdefault(name, i)
            count = i
    assert count == result["steps"] == recorded["actions"] <= meta["max_steps"] == 3600
    assert result["scoring"]["score"] == recorded["score"]
    assert meta["seed"] == recorded["seed"]
    assert result["reason"] == recorded["termination"] and result["error"] is None
    return dict(passed=True, revision=REVISION, role=role, run_id=directory.name,
                metadata_sha256=digest(directory/"metadata.json"),
                result_sha256=digest(directory/"result.json"),
                trace_sha256=digest(directory/"trace.jsonl"),
                source_hashes_verified=True, trace_hash_verified=True,
                original_source_hashes=meta["hashes"],
                revised_episode_contract=evidence["scoring_revision"]["revised_episode_contract"],
                assessor_sha256=digest(ROOT/"rescore_revision.py"),
                continuous_actions=count, seed=meta["seed"], termination=recorded["termination"],
                original_score=result["scoring"]["score"], scoring=scorer.snapshot(),
                recorded_final_score_preserved=scorer.score == result["scoring"]["score"],
                changed_score_steps=score_changes, changed_snapshot_count=len(snapshot_changes),
                first_changed_snapshot_steps=snapshot_changes[:20], first_step=milestones,
                note="CPU assessment of original measured trace with revised scorer; no new rollout, model call or hardware validation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run")
    parser.add_argument("--role", choices=("demo", "model"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    destination = Path(args.out).resolve()
    original = Path(args.run).resolve()
    assert destination != original and original not in destination.parents, "Keep original run artifacts immutable"
    report = rescore(original, args.role)
    destination.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps({k: report[k] for k in ("passed", "run_id", "continuous_actions", "original_score",
          "scoring", "recorded_final_score_preserved", "changed_score_steps", "changed_snapshot_count")}, indent=2))
