"""Independent replay of the single-pocket robot grasp diagnostic."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

from measurements import triangles_in_box
from self_contact import crossings, separating_translations
from robot_contacts import penetrating_gripper_shapes
from mesh import surface_edges


def audit_grasp(run,cooked_colliders=None,report_name="grasp-audit.json"):
    assert Path(report_name).name==report_name and report_name.endswith('.json')
    if cooked_colliders is None and (run/'gripper-cooked-colliders.json').exists():
        cooked_colliders=run/'gripper-cooked-colliders.json'
    assert not (run/"invalid-setup.json").exists(), "Known invalid setup cannot qualify"
    from yam_arena import ik, station
    request=json.loads((run/"request.json").read_text())
    if request.get("table_origin",False):
        checked=json.loads((run/"station-pose-check.json").read_text())
        assert {"floor","table","LeftArm","RightArm","TopCam"}.issubset(checked)
        assert all(v["position_error_m"]<2e-6 and v["orientation_error_rad"]<2e-5 for v in checked.values())
    result=json.loads((run/"result.json").read_text())
    assert result["status"]=="completed"
    rows=[json.loads(line) for line in (run/"steps.jsonl").read_text().splitlines()]
    scene=json.loads((run/"scene-audit.json").read_text())
    shift=np.asarray(scene.get("translation_S_W_m",[0.,0.,0.]))
    geometry=np.load(run/"geometry.npz")
    patch=json.loads((run/"grasp-observation.json").read_text())["patch_ids"]
    chunks=[]
    velocities=[]
    gripper_poses={"left":[],"right":[]}
    measured_strain=0.
    edges=surface_edges(geometry["faces"])
    rest_length=np.linalg.norm(geometry["rest"][edges[:,1]]-geometry["rest"][edges[:,0]],axis=1)
    for path in sorted((run/"physics-chunks").glob("*.npz")):
        with np.load(path) as chunk:
            chunks.append(chunk["positions"])
            velocities.append(chunk["velocities"])
            x=chunk["positions"]
            lengths=np.linalg.norm(x[:,edges[:,1]]-x[:,edges[:,0]],axis=2)
            measured_strain=max(measured_strain,float(abs(lengths/rest_length-1).max()))
            for side in gripper_poses:
                if side+"_grasp_poses" in chunk:
                    gripper_poses[side].append(chunk[side+"_grasp_poses"])
    positions=np.concatenate(chunks)
    velocities=np.concatenate(velocities)
    gripper_complete=all(len(values)==len(chunks) for values in gripper_poses.values()) and (run/"gripper-colliders.json").exists()
    gripper_geometry={}
    geometry_evidence=None
    if gripper_complete:
        gripper_geometry=json.loads((run/"gripper-colliders.json").read_text())
        if cooked_colliders is not None:
            cooked=json.loads(Path(cooked_colliders).read_text())
            proof=cooked['_provenance']
            assert proof['source_run']==run.name and not proof['scene_modified']
            for name,digest in proof['source_sha256'].items():
                assert hashlib.sha256((run/name).read_bytes()).hexdigest()==digest, f'Cooked geometry belongs to a different {name}'
            for side in gripper_poses:
                assert cooked[side]['body_names']==gripper_geometry[side]['body_names']
                assert cooked[side]['indices']==gripper_geometry[side]['indices']
                assert [v['name'] for v in cooked[side]['colliders']]==[v['name'] for v in gripper_geometry[side]['colliders']]
                if cooked[side].get('geometry_source')=='native_sdk_omnipvd':
                    capture=json.loads((run/'native-geometry-capture.json').read_text())
                    assert capture['native_geometry_used'] and capture['parsed_all_bytes']
                    assert not capture['scene_modified'] and not capture['physics_state_reused']
                    assert capture['scene_sha256']==proof['source_sha256']['scene.usda']
                    assert capture['native_capture_sha256']==proof['source_sha256'][capture['native_capture']]
                else:
                    assert all(v['archived_geometry_input_error_m']<2e-7 for v in cooked[side]['colliders'])
            gripper_geometry=cooked
            geometry_evidence=dict(path=str(Path(cooked_colliders).resolve()),
                sha256=hashlib.sha256(Path(cooked_colliders).read_bytes()).hexdigest(),provenance=proof)
        gripper_poses={side:np.concatenate(values) for side,values in gripper_poses.items()}
    hz=request["physics_hz"]
    decimation=hz//20
    assert len(positions)==len(rows)*decimation==result["physics_steps"]
    selected=np.array([row["phase"]=="hold_grasp" for row in rows])
    indices=np.flatnonzero(selected)
    centers=positions[indices*decimation+decimation-1][:,patch].mean(1)
    tcp=[]
    for index in indices:
        q=np.asarray(rows[index]["measured_joint_pos"][:6])
        tcp.append(ik.tcp_pose(q,station.BASES["left"])[0]+shift)
    distance=np.linalg.norm(centers-np.asarray(tcp),axis=1)
    scene=json.loads((run/"scene-audit.json").read_text())
    support_center=np.array(scene.get("support_center_m",[.3,.1,station.TABLE_TOP+.06]))
    support_size=np.array(scene.get("support_size_m",[.25,.10,.12]))
    gripper_steps=[]
    gripper_hits=set()
    penetration_steps=[]
    table_steps=[]
    self_crossing_steps=[]
    self_crossing_violation_steps=[]
    self_crossing_span_m=0.
    self_crossing_depth_m=0.
    for index,points in enumerate(positions):
        if gripper_complete:
            hit=[]
            for side in gripper_poses:
                names=penetrating_gripper_shapes(points,geometry["faces"],gripper_poses[side][index],gripper_geometry[side])
                hit.extend(side+name for name in names)
            if hit:
                gripper_steps.append(index)
                gripper_hits.update(hit)
        if index%1000==0:
            progress=dict(physics_step=index,total_steps=len(positions),gripper_penetrating_steps=len(gripper_steps))
            heartbeat="audit-heartbeat.json" if report_name=="grasp-audit.json" else Path(report_name).stem+"-heartbeat.json"
            (run/heartbeat).write_text(json.dumps(progress))
            print(json.dumps(progress),flush=True)
        pairs,spans=crossings(points,geometry["faces"])
        if len(pairs):
            self_crossing_steps.append(index)
            self_crossing_span_m=max(self_crossing_span_m,float(spans.max()))
            depth=float(np.linalg.norm(separating_translations(points,geometry['faces'],pairs),axis=1).max())
            self_crossing_depth_m=max(self_crossing_depth_m,depth)
            if depth>.001:
                self_crossing_violation_steps.append(index)
        if triangles_in_box(points,geometry["faces"],support_center,support_size):
            penetration_steps.append(index)
        if triangles_in_box(points,geometry["faces"],np.array(scene.get("table_center_m",station.TABLE_CENTER)),
                            np.array(scene.get("table_size_m",station.TABLE_SIZE))):
            table_steps.append(index)
    final=positions[-2*hz:]
    speed=float(np.linalg.norm(velocities[-2*hz:],axis=2).max())
    difference_speed=float(np.linalg.norm(np.diff(final,axis=0),axis=2).max()*hz)
    scene=json.loads((run/"scene-audit.json").read_text())
    finite=bool(np.isfinite(positions).all() and np.isfinite(velocities).all())
    strain=measured_strain
    reported_strain=float(result["max_abs_edge_strain"])
    report=dict(physics_steps=len(positions),control_steps=len(rows),
        hold_duration_s=len(indices)/20,hold_patch_tcp_distance_range_m=[float(distance.min()),float(distance.max())],
        hold_patch_height_range_m=[float(centers[:,2].min()),float(centers[:,2].max())],
        bench_penetrating_steps=penetration_steps,table_penetrating_steps=table_steps,
        gripper_contact_audit_complete=gripper_complete,gripper_penetrating_steps=gripper_steps,
        gripper_geometry_kind="native_cooked" if geometry_evidence else "authored_convex_hull_outer_bound",
        cooked_geometry_evidence=geometry_evidence,
        gripper_penetrated_colliders=sorted(gripper_hits),reported_max_abs_edge_strain=reported_strain,
        self_crossing_steps=self_crossing_steps,self_crossing_violation_steps=self_crossing_violation_steps,self_crossing_normal_span_estimate_m=self_crossing_span_m,
        self_crossing_penetration_upper_bound_m=self_crossing_depth_m,
        self_crossing_metric='Shortest separating translation along either triangle normal; conservative upper bound, 1 mm limit.',
        final_two_seconds_native_speed_m_s=speed,final_two_seconds_difference_speed_m_s=difference_speed,
        max_abs_edge_strain=strain,no_attachment=not scene["cloth_attachments"],
        recorded_states_finite=finite,
        patch_lift_m=result["patch_lift_m"],
        contact_grasp_observed=bool(len(indices)==40 and distance.max()<.025 and result["patch_lift_m"]>.08),
        released_settling_passed=speed<=.03 and difference_speed<=.03,
        numerical_checks_passed=finite and strain<=.2 and abs(strain-reported_strain)<1e-6 and not penetration_steps and not table_steps and not self_crossing_violation_steps and not gripper_steps,
        full_task_success=False,full_product_gate_passed=False,
        scope="Contact lift, support penetration, nonadjacent triangle crossings and released settling. No pocket insertion or full-task score.")
    report["grasp_gate_passed"]=bool(report["contact_grasp_observed"] and report["released_settling_passed"]
        and report["numerical_checks_passed"] and report["no_attachment"] and gripper_complete)
    (run/report_name).write_text(json.dumps(report,indent=2))
    videos={}
    for camera in ("top_cam","left_cam","right_cam"):
        path=run/(camera+".mp4")
        check=subprocess.run(["ffmpeg","-v","error","-i",str(path),"-map","0:v:0",
            "-f","null","-","-progress","pipe:1","-nostats"],capture_output=True,text=True)
        progress={}
        for line in check.stdout.splitlines():
            if "=" in line:
                key,value=line.split("=",1)
                progress[key]=value
        count=int(progress.get("frame",0))
        videos[camera]=dict(decoded_frames=count,expected_frames=len(rows),returncode=check.returncode,
            passed=check.returncode==0 and count==len(rows),decoder_error=check.stderr)
        subprocess.run(["ffmpeg","-y","-v","error","-ss",str(rows[indices[len(indices)//2]]["time_s"]),
            "-i",str(path),"-frames:v","1",str(run/(camera+"-hold.png"))],check=True)
    (run/"video-verification.json").write_text(json.dumps(dict(passed=all(v["passed"] for v in videos.values()),
        cameras=videos,scope="Decoded diagnostic videos; not a successful full-task demonstration."),indent=2))
    return report


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run",type=Path)
    parser.add_argument("--cooked-colliders",type=Path)
    parser.add_argument("--report-name",default="grasp-audit.json")
    args=parser.parse_args()
    assert args.run.resolve().is_relative_to(OUTPUT_ROOT)
    if args.cooked_colliders is not None:
        assert args.cooked_colliders.resolve().is_relative_to(OUTPUT_ROOT)
    report=audit_grasp(args.run,args.cooked_colliders,args.report_name)
    for key in ("self_crossing_steps","self_crossing_violation_steps","bench_penetrating_steps","table_penetrating_steps","gripper_penetrating_steps"):
        report[key]=len(report[key])
    print(json.dumps(report,indent=2))
