"""Combine independently audited single-pocket evidence without granting task score."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

OUT=OUTPUT_ROOT


def qualify(cycle_name,grasp_name,output_name="product-gate.json"):
    assert Path(output_name).name==output_name and output_name.endswith(".json")
    def read(run,name):
        return json.loads((run/name).read_text())
    cycle,grasp=OUT/cycle_name,OUT/grasp_name
    for path in (cycle,grasp):
        assert path.resolve().parent==OUT
    c=read(cycle,"result.json")
    a=read(cycle,"analysis.json")
    physics=read(cycle,"physics-audit.json")
    g=read(grasp,"grasp-audit.json")
    video=read(grasp,"video-verification.json")
    cr,gr=read(cycle,"request.json"),read(grasp,"request.json")
    cp=read(cycle,"effective-parameters.json")["effective_probe"]
    gp=read(grasp,"effective-parameters.json")
    physical_keys=("density_kg_m3","young_modulus_pa","poisson_ratio","bend_modulus_pa",
        "dynamic_friction","elasticity_damping","bend_damping","shell_thickness_m",
        "contact_offset_m","rest_offset_m","coupon_length_m","coupon_width_m","pocket_depth_m")
    same_material=all(cp[key]==gp[key] for key in physical_keys)
    model_defaults={"seam_height_m":0.,"flat_rest_panels":False,"flat_rest_bending":False}
    same_model=all(cp.get(key,default)==gp.get(key,default) for key,default in model_defaults.items())
    numeric_defaults={"physics_hz":240,"iterations":32,"contact_updates":1,"contact_subiterations":1,"ccd":False,"spacing":.005}
    same_numerics=all(cr.get(key,default)==gr.get(key,default) for key,default in numeric_defaults.items())
    rows=[json.loads(line) for line in (cycle/"steps.jsonl").read_text().splitlines()]
    hold=[r for r in a["samples"] if r["phase"]=="tool_hold"]
    seated_hold=sum(r["seated_samples"] for r in hold)/20
    early=[r["mouth_median_height_m"] for r in rows if .8<r["time_s"]<1.]
    during=[r["mouth_median_height_m"] for r in rows if 12.<r["time_s"]<13.]
    after=[r["mouth_median_height_m"] for r in rows if r["time_s"]>26.]
    from cloth_solver import configuration
    same_contact_solver=configuration(cr)==configuration(gr)
    if cr.get('cloth_contact_correction_speed') is not None:
        hashes=[hashlib.sha256((path/'source/cloth_solver.py').read_bytes()).hexdigest() for path in (cycle,grasp)]
        same_contact_solver=same_contact_solver and hashes[0]==hashes[1]
        for path,request in ((cycle,cr),(grasp,gr)):
            authored=read(path,'cloth-numerics.json')
            same_contact_solver=same_contact_solver and authored['configuration']==configuration(request)
            same_contact_solver=same_contact_solver and authored['authored_before_simulation'] and authored['constant_throughout_episode']
            same_contact_solver=same_contact_solver and abs(authored['after']-request['cloth_contact_correction_speed'])<1e-7
    backends = ['installed_ovphysx' if cr.get('probe_sdk_source') else 'kit_physx',
                'installed_ovphysx' if gr.get('grasp_sdk_source') else 'kit_physx']
    from passive_contact import native_physics_errors
    native_ok = all(not native_physics_errors((path/'kit.log').read_text()) for path in (cycle, grasp))
    sdk_matches = True
    if 'installed_ovphysx' in backends:
        sdk_matches = backends[0] == backends[1]
        if sdk_matches:
            cloth = [read(path, 'sdk-cloth-properties.json') for path in (cycle, grasp)]
            sdk_matches = cloth[0]['material'] == cloth[1]['material']
            sdk_matches = sdk_matches and all(read(path, 'native-initialization-check.json')['passed'] for path in (cycle, grasp))
            versions = [read(path, 'versions.json') for path in (cycle, grasp)]
            sdk_matches = sdk_matches and versions[0].get('ovphysx') == versions[1].get('ovphysx')
            defaults = [read(path, 'native-default-override.json') for path in (cycle, grasp)]
            cloth_defaults = [record['overrides']['/World/Pocket' if i == 0 else '/World/envs/env_0/Pocket']
                              for i, record in enumerate(defaults)]
            sdk_matches = sdk_matches and cloth_defaults[0] == cloth_defaults[1]
            sdk_matches = sdk_matches and all(record['only_listed_attributes_and_required_api_metadata_changed'] for record in defaults)
            sdk_matches = sdk_matches and read(grasp, 'sdk-yam-contract.json')['contract']['stock_constants_verified']
            sdk_matches = sdk_matches and read(grasp, 'sdk-yam-reference-comparison.json')['passed']
    checks={
        'matched_physics_backend': backends[0] == backends[1],
        'native_error_free': native_ok,
        'matched_sdk_native_properties': sdk_matches,
        "same_cloth_contact_solver":same_contact_solver,
        "cycle_completed":c["status"]=="completed" and read(cycle,"process.json")["returncode"]==0,
        "native_cycle_physics":physics["diagnostic_stability_passed"],
        "geometric_replay":a["metric_replay_passed"],
        "seated_hold_two_seconds":seated_hold>=2.-1e-8,
        "complete_withdrawal":a["final_seating"]["inside_fraction"]==0.,
        "contact_opening":bool(np.median(during)>np.median(early)+.002),
        "passive_collapse":bool(np.median(after)<np.median(during)-.002),
        "robot_grasp_release":g["grasp_gate_passed"],
        "robot_process_completed":read(grasp,"result.json")["status"]=="completed" and read(grasp,"process.json")["returncode"]==0,
        "three_decodable_camera_videos":video["passed"] and len(video["cameras"])==3,
        "same_material_and_sewn_model":same_material and same_model,
        "same_initial_pocket_geometry":cr.get("mouth_height",.016)==gr.get("mouth_height",.016),
        "same_solver_configuration":same_numerics,
        "same_product_and_physics_step":cr["product_profile"]==gr["product_profile"]=="base_micro" and cr["physics_hz"]==gr["physics_hz"],
    }
    artifacts={}
    for folder,names in [(cycle,["manifest.json","analysis.json","physics-audit.json","gpu-verification.json"]),
                         (grasp,["manifest.json","grasp-audit.json","video-verification.json","gpu-verification.json"])]:
        for name in names:
            artifacts[str((folder/name).relative_to(OUT))]=hashlib.sha256((folder/name).read_bytes()).hexdigest()
    if cr.get('cloth_contact_correction_speed') is not None:
        for folder in (cycle,grasp):
            for name in ('cloth-numerics.json','request.json'):
                artifacts[str((folder/name).relative_to(OUT))]=hashlib.sha256((folder/name).read_bytes()).hexdigest()
    if backends[0] == backends[1] == 'installed_ovphysx':
        for folder in (cycle, grasp):
            for name in ('sdk-cloth-properties.json', 'native-default-override.json', 'native-initialization-check.json',
                         'sdk-source-preservation.json', 'request.json', 'versions.json'):
                artifacts[str((folder/name).relative_to(OUT))] = hashlib.sha256((folder/name).read_bytes()).hexdigest()
        for name in ('sdk-yam-contract.json', 'sdk-yam-reference-comparison.json', 'sdk-reset-check.json', 'sdk-camera-provenance.json'):
            artifacts[str((grasp/name).relative_to(OUT))] = hashlib.sha256((grasp/name).read_bytes()).hexdigest()
    cooked=g.get('cooked_geometry_evidence')
    if cooked:
        path=Path(cooked['path']).resolve()
        assert path.is_relative_to(OUT)
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==cooked['sha256']
        artifacts[str(path.relative_to(OUT))]=digest
        for name,digest in cooked['provenance']['source_sha256'].items():
            assert hashlib.sha256((grasp/name).read_bytes()).hexdigest()==digest
            artifacts[str((grasp/name).relative_to(OUT))]=digest
    refinement=g.get('crossing_refinement')
    if refinement:
        path=grasp/refinement['source_report']
        digest=hashlib.sha256(path.read_bytes()).hexdigest()
        assert digest==refinement['source_report_sha256']
        artifacts[str(path.relative_to(OUT))]=digest
    report=dict(passed=all(checks.values()),physics_backend=backends[0],scope="Isolated single-pocket material, contact insertion, withdrawal and robot grasp/release.",
        full_task_success=False,product="Vileda BaseMicro polyester pocket cover / estimated CombiSpeed Pro shaped end",
        cycle_run=cycle_name,grasp_run=grasp_name,checks=checks,seated_hold_s=seated_hold,
        mouth_height_m=dict(settled=float(np.median(early)),inserted=float(np.median(during)),released=float(np.median(after))),
        evidence_sha256=artifacts,
        limitations=["One pocket and laboratory supports only; no two-pocket frame or passive-lock validation.",
            "Material and end-profile dimensions remain estimates; hardware measurements unavailable.",
            "Earlier failed and unstable configurations are retained. This gate is not a reliability-rate claim."])
    (OUT/output_name).write_text(json.dumps(report,indent=2)+"\n")
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle",required=True)
    parser.add_argument("--grasp",required=True)
    parser.add_argument("--output",default="product-gate.json")
    args=parser.parse_args()
    result=qualify(args.cycle,args.grasp,args.output)
    print(json.dumps(result,indent=2))
    raise SystemExit(0 if result["passed"] else 1)
