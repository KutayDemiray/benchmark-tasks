"""Strict CPU provenance checks for the reviewed scoring-only revision."""
import copy
import hashlib
import json
from pathlib import Path
from episode_config import contract

REVISION = "retrieval-without-empty-cycle-v2"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def only_scorer_changed(original, revised):
    """Do not admit instruction, physics, geometry, budget or other source edits."""
    a, b = copy.deepcopy(original), copy.deepcopy(revised)
    try:
        old = a["source_hashes"].pop("scorer.py")
        new = b["source_hashes"].pop("scorer.py")
    except KeyError:
        return False
    return a == b and old != new


def verify_revision(root, require_reports=True):
    root = Path(root)
    evidence = json.loads((root/"evidence.json").read_text())
    revision = evidence["scoring_revision"]
    original, current = evidence["episode_contract"], contract(root)
    assert revision["id"] == REVISION
    assert only_scorer_changed(original, current), "Revision changed more than scorer.py"
    assert revision["original_scorer_sha256"] == original["source_hashes"]["scorer.py"]
    assert revision["revised_episode_contract"] == current, "Revised source hash mismatch"
    if require_reports:
        for role, expected in (("demo", 1.), ("model", 0.)):
            run, item = evidence["runs"][role], revision["reports"][role]
            path = root/item["file"]
            assert digest(path) == item["sha256"], f"Revised {role} report hash mismatch"
            report = json.loads(path.read_text())
            assert report["passed"] and report["revision"] == REVISION
            assert report["run_id"] == run["run_id"]
            assert report["metadata_sha256"] == run["metadata_sha256"]
            assert report["trace_sha256"] == run["trace_sha256"]
            assert {k: v for k, v in report["original_source_hashes"].items()
                    if k not in {"README.md", "task.json"}} == run["source_hashes"]
            assert set(report["original_source_hashes"]) == set(run["source_hashes"]) | {"README.md", "task.json"}
            assert report["revised_episode_contract"] == current
            assert report["source_hashes_verified"] and report["trace_hash_verified"]
            assert report["assessor_sha256"] == digest(root/"rescore_revision.py")
            assert report["continuous_actions"] == run["actions"] <= current["max_actions"]
            assert report["seed"] == run["seed"]
            assert report["original_score"] == run["score"] == expected
            assert report["scoring"]["score"] == expected
            assert report["termination"] == run["termination"]
            assert report["recorded_final_score_preserved"]
    return evidence


def compatible_contract(root, recorded):
    """Accept current evidence or the hash-checked, explicitly rescored revision."""
    if recorded == contract(root):
        return True
    return recorded == verify_revision(root)["episode_contract"]
