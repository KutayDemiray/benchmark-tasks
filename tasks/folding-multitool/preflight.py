"""CPU-only readiness check. Resolves credentials without contacting a service."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from episode_config import contract


def same_runner_except_paths(tested, current):
    """Permit only the two submission portability edits to the tested runner.

    Full source equality is required after these exact path-only substitutions.
    Any execution, timing, input, physics or control edit still needs a new
    native interface check. The caller verifies the frozen source hash first.
    """
    if tested == current:
        return True
    substitutions = (
        (r'^    sys.path.insert\(0, "[^"\n]+/robocurve-gpu-scheduler"\)$',
         '    sys.path.insert(0, os.environ.get("ROBOCURVE_SCHEDULER_ROOT", str(Path.home()/"robocurve-gpu-scheduler")))'),
        (r'^    parser.add_argument\("--out", default="[^"\n]+/yam-eval/folding-multitool/runs"\)$',
         '    parser.add_argument("--out", default=str(Path.home()/"yam-eval/folding-multitool/runs"))'),
    )
    for pattern, replacement in substitutions:
        tested, count = re.subn(pattern, replacement, tested, count=1, flags=re.MULTILINE)
        if count != 1:
            return False
    return tested == current


def credentials_status():
    from dream.credentials import load
    c = load()
    return {"api_key_present": bool(c.api_key), "credentials_path": str(c.path),
            "credentials_file_exists": c.path.is_file(),
            "gateway_url": c.gateway_url, "platform_url": c.platform_url,
            "network_called": False}


def ready(root, demo_run=None, interface_run=None, runs=None):
    report = {"credentials": credentials_status(), "blockers": [], "contract": contract(root)}
    # Optional discovery only selects completed local evidence. The detailed
    # checks below still reject unreviewed demos, failed interface tests and
    # changed source. Nothing here launches a simulator or a provider call.
    if runs is not None:
        for path in sorted(Path(runs).glob("*/metadata.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                meta = json.loads(path.read_text())
                result = json.loads((path.parent/"result.json").read_text())
                if meta.get("mode") != "retrieval" or meta.get("contract") != report["contract"]:
                    continue
                if demo_run is None and meta.get("controller") == "scripted" and result["scoring"]["score"] == 1.:
                    demo_run = path.parent
                if interface_run is None and meta.get("controller") == "interface-check":
                    check = json.loads((path.parent/"interface-check.json").read_text())
                    if check["passed"]:
                        interface_run = path.parent
            except (OSError, ValueError, KeyError):
                continue
    if not report["credentials"]["api_key_present"]:
        report["blockers"].append("DREAM_API_KEY is absent from the environment and the resolved credentials file")
    for label, directory in (("demo", demo_run), ("interface", interface_run)):
        if directory is None:
            report["blockers"].append(f"No {label} evidence directory supplied")
            continue
        path = Path(directory)
        try:
            metadata = json.loads((path/"metadata.json").read_text())
            result = json.loads((path/"result.json").read_text())
            assert metadata["mode"] == "retrieval", "diagnostic scene"
            assert metadata["contract"] == report["contract"], "episode source/configuration changed"
            assert json.loads((path/"device-verification.json").read_text())["verified"], "GPU not verified"
            assert result["device_verified"], "GPU not verified"
            if label == "demo":
                assert metadata["controller"] == "scripted", "not a scripted demonstration"
                assert metadata["max_steps"] == 3600 and result["steps"] <= 3600, "wrong budget"
                assert result["scoring"]["score"] == 1., "demo did not score 1.0"
                rescore = json.loads((path/"independent-rescore.json").read_text())
                assert rescore["passed"] and rescore["scoring"]["score"] == 1., "independent rescore failed"
                review = json.loads((path/"recording-review.json").read_text())
                assert review["continuous_physical_demonstration_accepted"], "recording not accepted"
            else:
                assert metadata["controller"] == "interface-check", "not an offline interface check"
                assert json.loads((path/"interface-check.json").read_text())["passed"], "interface check failed"
                # The I/O boundary and runner must match the tested versions.
                for name in ("run.py", "evaluation.py"):
                    current = (Path(root)/name).read_bytes()
                    if metadata["hashes"][name] != hashlib.sha256(current).hexdigest():
                        tested = (path/"source"/name).read_bytes()
                        assert metadata["hashes"][name] == hashlib.sha256(tested).hexdigest(), f"{name} evidence hash mismatch"
                        assert name == "run.py" and same_runner_except_paths(tested.decode(), current.decode()), f"{name} changed"
                        report["runner_path_only_portability_verified"] = True
            report[label+"_evidence"] = str(path.resolve())
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            report["blockers"].append(f"{label}: {exc}")
    report["ready"] = not report["blockers"]
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--demo-run")
    p.add_argument("--interface-run")
    p.add_argument("--runs", default=str(Path.home()/"yam-eval/folding-multitool/runs"))
    p.add_argument("--out")
    args = p.parse_args()
    result = ready(Path(__file__).resolve().parent, args.demo_run, args.interface_run, args.runs)
    text = json.dumps(result, indent=2)+"\n"
    if args.out:
        Path(args.out).write_text(text)
    print(text)
