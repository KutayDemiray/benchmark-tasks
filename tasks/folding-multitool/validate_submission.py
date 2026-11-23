"""CPU-only package checks; never launches a simulator, uploads or calls a model."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
from episode_config import contract, INSTRUCTION

ROOT = Path(__file__).resolve().parent
REQUIRED = ("title", "category", "instruction", "setup", "scoring", "run_limit",
            "fidelity", "scores", "media")
CATEGORIES = {"Impact on jobs", "Impact on AI timelines", "Safety and alignment",
              "Societal relatability", "Other"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate(allow_unuploaded=False, media_dir=None):
    spec = json.loads((ROOT/"task.json").read_text())
    evidence = json.loads((ROOT/"evidence.json").read_text())
    lengths = {k: len(v) for k, v in spec.items() if isinstance(v, str)}
    print("Actual text field lengths:", json.dumps(lengths, sort_keys=True))
    for key in REQUIRED:
        if key == "media" and allow_unuploaded and spec.get(key) is None:
            continue
        assert isinstance(spec.get(key), str) and spec[key].strip(), f"Missing/empty {key}"
    assert len(spec["fidelity"]) <= 500, "fidelity exceeds 500 characters"
    assert len(spec["scores"]) <= 300, "scores exceeds 300 characters"
    assert spec["category"] in CATEGORIES
    assert spec["instruction"] == INSTRUCTION
    assert "Dream-1: 0.0" in spec["scores"] and "demo: 1.0" in spec["scores"]
    if spec["media"] is not None:
        assert re.fullmatch(r"M-[A-Za-z0-9]+", spec["media"]), "Unexpected media ID format"
    for value in (v for v in spec.values() if isinstance(v, str)):
        assert "/home/" not in value and not re.search(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", value), "Identifying path/email in task text"
    assert contract(ROOT) == evidence["episode_contract"], "Validated task definition changed"
    for role, score, count in (("demo", 1., 3018), ("model", 0., 3600)):
        run = evidence["runs"][role]
        assert run["score"] == score and run["actions"] == count and run["seed"] == 0
        assert run["independent_rescore"]["passed"] and not run["independent_rescore"]["mismatch_steps"]
        assert run["recording_review_accepted"] and run["native_audit_passed"] and run["device"]["verified"]
    assert evidence["runs"]["model"]["hosted"]
    assert evidence["runs"]["model"]["inference"]["episode_count"] == 1
    assert evidence["runs"]["model"]["inference"]["scripted_action_rows"] == 0
    assert evidence["runs"]["interface"]["interface_passed"]
    # Every original implementation file except the disclosed path/preflight
    # changes must be identical to the evaluated snapshot.
    for name, expected in evidence["runs"]["model"]["source_hashes"].items():
        if name not in {"run.py", "preflight.py"}:
            assert digest(ROOT/name) == expected, f"Evaluated source changed: {name}"
    assert digest(ROOT/"run.py") == evidence["submission_portability"]["submitted_runner_sha256"]
    for path in ROOT.glob("*.py"):
        ast.parse(path.read_text(), filename=path.name)
    if media_dir:
        for item in evidence["media"].values():
            path = Path(media_dir)/item["name"]
            assert path.is_file() and digest(path) == item["sha256"], f"Wrong media: {path.name}"
            assert path.stat().st_size == item["bytes"]
            limit = 10_000_000 if path.suffix == ".png" else 50_000_000
            assert path.stat().st_size < limit
    source_hashes = {p.relative_to(ROOT).as_posix(): digest(p)
                     for p in sorted(ROOT.rglob("*")) if p.is_file()
                     and "__pycache__" not in p.parts and p.name != "package-validation.json"}
    assert not any(Path(n).suffix in {".mp4", ".jsonl", ".usd", ".usda", ".usdc", ".eval"}
                   for n in source_hashes), "Raw media/runtime data in submission"
    return {"passed": True, "ready_to_commit": spec["media"] is not None,
            "field_lengths": lengths, "media_id": spec["media"],
            "media_hashes_checked": bool(media_dir), "cpu_tests_passed": evidence["cpu_tests"]["passed"],
            "task_contract_verified": True, "evaluated_runtime_hashes_verified": True,
            "source_hashes": source_hashes,
            "pending": [] if spec["media"] else ["Repository uploader must populate task-specific media ID"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-unuploaded", action="store_true", help="Preparation only; never a ready-to-commit pass")
    parser.add_argument("--media-dir")
    parser.add_argument("--write-manifest", action="store_true")
    args = parser.parse_args()
    result = validate(args.allow_unuploaded, args.media_dir)
    if args.write_manifest:
        (ROOT/"package-validation.json").write_text(json.dumps(result, indent=2)+"\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source_hashes"}, indent=2))
