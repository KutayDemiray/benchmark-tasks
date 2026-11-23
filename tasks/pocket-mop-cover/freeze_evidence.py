"""Inventory recorded evidence; do not infer task success from process exit codes."""

from task_paths import OUTPUT_ROOT

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = OUTPUT_ROOT


def read(path):
    """Return a JSON artifact when it exists."""
    return json.loads(path.read_text()) if path.exists() else None


def main():
    hashes = {file.name: hashlib.sha256(file.read_bytes()).hexdigest()
              for file in sorted(HERE.iterdir()) if file.is_file()}
    bundle_hash = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    runs = []
    for folder in sorted(OUT.iterdir()):
        request = read(folder / "request.json")
        if request is None:
            continue
        row = {"name": folder.name, "request": request}
        for name in ("result", "process", "blocked", "manifest", "analysis", "physics-audit",
                     "grasp-audit", "mechanism-audit", "folded-pocket-audit", "frame-robot-audit", "release-screen", "manual-stop", "reset-comparison", "score-replay", "physics-replay", "video-verification",
                     "interface-verification", "policy-transcript", "model-trace-verification", "evaluation-parity", "initial-public-state", "full-cloth-audit", "benchmark-result", "handle-held-benchmark-result", "reset-verification", "replay-provenance",
                     "invalid-witness", "material-audit-stop", "release-probe",
                     "release-probe-initialization", "release-assessment",
                     "native-initialization-check", "native-constraint-rejection", "passive-initialization",
                     "passive-analysis", "redundant-limit-proof", "body-frame-equivalence", "mimic-compliance",
                     "native-drive-equivalence", "native-mass-response", "partition-readback",
                     "usd-surface-readback", "unit-conversion", "initial-handle-pose-change", "settling-localization",
                     "head-contact-correction", "gpu-reservation", "gpu-verification", "scheduler-physics-preservation",
                     "sdk-native-properties", "sdk-cloth-layout", "sdk-timestep-refinement", "sdk-property-order",
                     "initial-frame-reconstruction", "free-space-momentum", "passive-geometry-stride-1", "launcher-error",
                     "passive-rigid-audit", "passive-hold-search", "sdk-source-preservation", "sdk-cloth-properties", "native-yam-contract", "sdk-yam-contract", "sdk-yam-reference-comparison", "sdk-reset-check", "sdk-camera-provenance",
                     "link-order-comparison", "native-property-comparison", "native-default-override", "native-default-verification", "continuation-skipped"):
            value = read(folder / (name + ".json"))
            if value is not None:
                if name in ("analysis","folded-pocket-audit"):
                    value.pop("samples", None)
                if name == "passive-analysis" and value.get('energy'):
                    value['energy'].pop('sampled_first_second', None)
                if name == "manifest":
                    value = {"bundle_sha256": value["bundle_sha256"]}
                row[name] = value
        row["has_shutdown_gpu_audit"] = (folder / "after-gpu.json").exists()
        row["complete_native_recording"] = (((folder / "states.npz").exists() or (folder / "physics-chunks").is_dir())
                                            and row.get("result", {}).get("status") == "completed")
        if request.get('experiment') == 'passive_contact' or any(request.get(key) for key in ('probe_sdk_source','grasp_sdk_source','mechanism_sdk_source')):
            from passive_contact import native_physics_errors
            errors = native_physics_errors((folder/'kit.log').read_text()) if (folder/'kit.log').exists() else ['Missing native log']
            row['native_physics_accepted'] = (not errors and 'native-constraint-rejection' not in row
                and row.get('native-initialization-check', {}).get('passed', False))
            row['native_errors_at_freeze'] = errors
        runs.append(row)
    demos = []
    stable = []
    replays = []
    for row in runs:
        result = row.get("result", {})
        if row.get("physics-audit", {}).get("diagnostic_stability_passed"):
            stable.append(row["name"])
        held = row['request'].get('completion_mode') == 'handle-held'
        revision = 'handle-held-v2' if held else 'released-v1'
        score=row.get('handle-held-benchmark-result' if held else 'benchmark-result',{})
        row['assessed_rubric_revision'] = revision
        if held and score:
            assert score.get('rubric_revision') == revision and not score.get('historical_screen_only'), 'Historical screens are not demonstrations'
        if row['request'].get('replay_run'):
            replays.append(dict(name=row['name'],source_run=row['request']['replay_run'],
                completed=result.get('status')=='completed',score=score.get('score'),
                success=score.get('success',False),rubric_revision=revision,provenance_recorded='replay-provenance' in row))
        if (not row['request'].get('release_probe_run')
                and not result.get('diagnostic_only',False)
                and score.get('success') and score.get('score')==1.
                and result.get("status") == "completed"
                and row.get('full-cloth-audit',{}).get('complete')
                and row.get('full-cloth-audit',{}).get('numerical_checks_passed')
                and row.get('frame-robot-audit',{}).get('contact_checks_passed')
                and row.get('frame-robot-audit',{}).get('robot_table_and_self_contacts_audited')
                and row.get("video-verification", {}).get("passed")
                and row.get("process", {}).get("returncode") == 0):
            demos.append(row["name"])
    released_demos = [name for name in demos if next(row for row in runs if row['name'] == name)['assessed_rubric_revision'] == 'released-v1']
    model_runs = [dict(name=row['name'], result=row.get('result'), score=row.get('handle-held-benchmark-result' if row['assessed_rubric_revision']=='handle-held-v2' else 'benchmark-result'),
                      transcript=row.get('policy-transcript'), public_trace=row.get('model-trace-verification')) for row in runs if row['request'].get('policy') == 'dream']
    interface_runs = [dict(name=row['name'], check=row.get('interface-verification')) for row in runs
                      if row['request'].get('policy') == 'interface-check']
    claim = read(OUT/'hosted-dream-attempt.json')
    received = sum(row.get('policy-transcript',{}).get('received_chunks',0) for row in runs
                   if row['request'].get('policy') == 'dream')
    gate = read(OUT / "product-gate.json")
    metadata = read(HERE / "task.json")
    active_revision = metadata.get("rubric_revision", "released-v1")
    active_demos = [row["name"] for row in runs if row["name"] in demos and row["assessed_rubric_revision"] == active_revision]
    status = ("local_demonstration_verified" if active_demos else
              "two_pocket_integration_in_progress" if any(row['request'].get('robot_cover') for row in runs) else
              "isolated_product_gate_passed" if gate and gate.get("passed") else
              "isolated_diagnostics_in_progress")
    report = {
        "task": "pocket-mop-cover", "status": status, "active_rubric_revision": active_revision, "full_task_success": bool(active_demos),
        "submission_ready": bool(metadata.get('submission_ready',False)),
        "passive_release_comparison": read(OUT / 'release-passive-batch-2026-11-22.json'),
        "passive_native_error_audit": read(OUT / 'passive-native-error-audit-2026-11-22.json'),
        "current_source_bundle_sha256": bundle_hash, "current_source_files": hashes,
        "cpu_tests": read(OUT / "cpu-tests-current.json"),
        "completion_criteria": {"full_score_demonstrations_required": 1,
                                "minimum_success_rate": None,
                                "scope": "Predeclared task revision demonstration, fresh audited replay and one genuine hosted Dream-1 episode; uploads and PR separate"},
        "verified_demonstration_runs": demos,
        "verified_demonstrations_by_rubric": {revision: [row["name"] for row in runs if row["name"] in demos and row["assessed_rubric_revision"] == revision] for revision in ("released-v1", "handle-held-v2")},
        "stable_diagnostic_runs": stable,
        "independent_action_replays":replays,
        "product_gate": gate, "product_sdk_gate": read(OUT / "product-sdk-gate.json"), "mechanism_sdk_gate": read(OUT / "mechanism-sdk-gate.json"),
        "mechanism_gate": read(OUT / "mechanism-gate.json"), "task_metadata": metadata,
        "model_runs": model_runs, "native_interface_runs": interface_runs,
        "hosted_attempt_claim": claim,
        "external_actions": {"hosted_episode_claims": int(claim is not None),
            "recorded_hosted_chunks": received, "uploads": 0, "submissions": 0, "commits": 0},
        "runs": runs,
    }
    (OUT / "VALIDATION.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (OUT / "source-manifest.json").write_text(
        json.dumps({"bundle_sha256": bundle_hash, "files": hashes}, indent=2) + "\n")
    print(json.dumps({"bundle_sha256": bundle_hash, "runs": len(runs), "status": status}, indent=2))


if __name__ == "__main__":
    main()
