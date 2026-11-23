"""Independent CPU evidence verification for the tabletop shirt adaptation."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from robot_geometry import rotation
from tabletop_clearance import robot_clearance
from episode_timing import control_hz,physics_stride,retention_samples

from task_paths import OUTPUT_ROOT as OUTPUT


def independent_final_pairs(rows,physics,cfg):
    """Recompute terminal closures with explicit rotation matrices on the CPU."""
    p=cfg['tabletop'];samples=retention_samples(cfg)
    if len(rows)<samples:return []
    tail=rows[-samples:]
    if any(min(r['robot_clearance_m'].values())<p['released_robot_clearance_m'] for r in tail):return []
    center=np.array([r['physics']['mean_cloth_position_m'] for r in tail])
    if np.linalg.norm(center-center[-1],axis=1).max()>p['max_released_drift_m']:return []
    first=round(tail[0]['time_s']*cfg['simulation']['physics_hz'])
    count=len(cfg['garment']['snap_heights_from_hem_m']);retained=set(p['required_snap_indices'])
    snap=cfg['snap']
    axial_limit=min(snap['ring_height_m']+snap['cap_thickness_m']/2+.0007,
        snap['socket_ball_height_m']+snap['head_from_cap_m']-.0006)
    axial_minimum=(snap['ring_height_m']+snap['cap_thickness_m']/2-p['seat_rim_penetration_allowance_m']
                   if 'seat_rim_penetration_allowance_m' in p else .001)
    for state in physics[first:]:
        if state['table_coverage_fraction']<.98 or max(state['snap_robot_contact_norm_N'].values())>.05:return []
        males=np.array([state['hardware'][f'male{i}'] for i in range(count)])
        sockets=np.array([state['hardware'][f'socket{i}'] for i in range(count)])
        mr=[rotation(v[3:]) for v in males];sr=[rotation(v[3:]) for v in sockets]
        correct=set()
        for i in range(count):
            for j in range(count):
                offset=sr[j].T@(males[i,:3]-sockets[j,:3])
                closed=(np.linalg.norm(offset[:2])<=.00075 and axial_minimum<offset[2]<axial_limit
                    and mr[i][:,2]@sr[j][:,2]>np.cos(np.deg2rad(12)))
                if closed and i!=j:return []
                if closed:correct.add(i)
        retained &= correct
    return sorted(retained)


def verify(run):
    """Check frozen sources, native observations, independent score and videos."""
    manifest=json.loads((run/'manifest.json').read_text())
    hashes={name:hashlib.sha256((run/'task-bundle'/name).read_bytes()).hexdigest() for name in manifest['sources']}
    assert hashes==manifest['sources'],'source_modified'
    assert hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()==manifest['bundle_sha256']
    result=dict(run=str(run),source_bundle_sha256=manifest['bundle_sha256'],source_hashes_match=True,
                episodes=[],original_hanger_task_success=False,versions=manifest['versions'])
    dependencies=('verify_tabletop_run.py','robot_geometry.py','tabletop_clearance.py','episode_timing.py','gpu_runtime.py','task_paths.py')
    result['verifier_dependencies_sha256']={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in dependencies}
    initial_reference=None;reset_checks=[]
    for episode in sorted(run.glob('episode-*')):
        setup=json.loads((episode/'setup.json').read_text());cfg=setup['parameters'];p=cfg['tabletop']
        stored=json.loads((episode/'result.json').read_text())
        rows=[json.loads(s) for s in (episode/'trace.jsonl').read_text().splitlines()]
        physics=[json.loads(s) for s in (episode/'physics.jsonl').read_text().splitlines()]
        meta=json.loads((episode/'topology.json').read_text());cloth=meta['cloth']
        from pxr import Gf,Usd,UsdGeom
        stage=Usd.Stage.Open(str(episode/'initial.usda'))
        assert abs(UsdGeom.GetStageMetersPerUnit(stage)-1.)<1e-10 and UsdGeom.GetStageUpAxis(stage)=='Z'
        table=UsdGeom.Cube(stage.GetPrimAtPath(meta['root']+'/table'))
        transform=UsdGeom.XformCache().GetLocalToWorldTransform(table.GetPrim())
        half_size=float(table.GetSizeAttr().Get())/2
        corners=[]
        for x in (-half_size,half_size):
            for y in (-half_size,half_size):
                for z in (-half_size,half_size):corners.append(transform.Transform(Gf.Vec3d(x,y,z)))
        corners=np.asarray(corners)
        assert abs(corners[:,2].max()-p['table_top_m'])<1e-8,'Table auditor plane differs from exported collider'
        assert np.max(np.abs(np.ptp(corners[:,:2],axis=0)-p['table_size_xy_m']))<1e-7
        assert np.max(np.abs(corners[:,:2].mean(axis=0)-p['table_center_xy_m']))<1e-7
        gpu=json.loads((episode/'gpu.json').read_text())
        preflight=json.loads((run/'preflight.json').read_text())
        from gpu_runtime import audit_record
        audit_record(gpu,preflight,bool(setup['recorded_cameras']))
        attachments=json.loads((episode/'attachment-audit.json').read_text())
        assert not attachments['world_mounts'] and not attachments['robot_grasp_attachments']
        assert not attachments['shirt_to_hanger_attachments'] and len(attachments['local_rivets'])==14
        initial_data=np.load(episode/'initial-measured-state.npz')
        assert all(np.isfinite(initial_data[k]).all() for k in initial_data.files)
        # Arena performs native warmup steps before the first exposed state.
        # Cloth/hardware already have a small gravitational velocity; rebuilding
        # must reproduce it, rather than silently zeroing live object states.
        assert all(np.max(np.abs(initial_data[k]))<1e-7 for k in ('left_joint_velocities','right_joint_velocities'))
        if initial_reference is None:initial_reference={k:initial_data[k].copy() for k in initial_data.files}
        else:
            differences={k:float(np.max(np.abs(initial_data[k]-initial_reference[k]))) for k in initial_data.files}
            assert max(differences.values())<1e-7,'Incomplete scene reset'
            reset_checks.append(dict(episode=episode.name,maximum_absolute_difference=differences,pass_=True))
        data=np.load(episode/'control-steps.npz');stride=physics_stride(cfg)
        assert len(physics)==len(rows)*stride+1
        assert [r['physics_step'] for r in physics]==list(range(len(physics)))
        assert np.array_equal(data['actions'],np.asarray([r['robot_action'] for r in rows]).reshape(-1,14))
        assert data['actions'].shape==(len(rows),14) and np.isfinite(data['actions']).all()
        assert np.all((data['actions'][:,[6,13]]>=0)&(data['actions'][:,[6,13]]<=1))
        edges=np.asarray(cloth['edges']);rest=np.asarray(cloth['rest_lengths'])
        error=np.zeros(4);clearance_error=0.
        for row,nodes in zip(rows,data['cloth_nodes'],strict=True):
            assert row['physics']==physics[(row['step']+1)*stride]
            assert not row['original_hanger_task_success']
            nodes=nodes.astype(float)
            strain=float(np.abs(np.linalg.norm(nodes[edges[:,0]]-nodes[edges[:,1]],axis=1)/rest-1).max())
            rivet=0.
            for att in cloth['attachments']:
                pose=np.asarray(row['physics']['hardware'][att['hardware']])
                expected=np.asarray(att['local'])@rotation(pose[3:]).T+pose[:3]
                rivet=max(rivet,float(np.linalg.norm(nodes[att['ids']]-expected,axis=1).max()))
            inside=(np.abs(nodes[:,:2]-p['table_center_xy_m'])<np.asarray(p['table_size_xy_m'])/2).all(axis=1)
            penetration=float(max(0.,p['table_top_m']-nodes[inside,2].min())) if inside.any() else 0.
            differences=np.abs(np.array([strain,rivet,penetration,inside.mean()])-np.array([
                row['physics'][key] for key in ('strain','rivet_error_m','table_penetration_m','table_coverage_fraction')]))
            error=np.maximum(error,differences)
            if 'robot_clearance_m' in row:
                clearance=robot_clearance(row,nodes,meta,meta['robot_collision_bounds'])
                clearance_error=max(clearance_error,max(abs(clearance[k]-row['robot_clearance_m'][k]) for k in clearance))
        assert error.max()<1e-10,'Independent CPU measurements disagree'
        assert clearance_error<1e-10,'Robot withdrawal measurements disagree'
        if (episode/'worst-table-contact.npz').exists():
            witness=np.load(episode/'worst-table-contact.npz')
            nodes=witness['nodes'];step=int(witness['physics_step'])
            inside=(np.abs(nodes[:,:2]-p['table_center_xy_m'])<np.asarray(p['table_size_xy_m'])/2).all(axis=1)
            penetration=float(max(0.,p['table_top_m']-nodes[inside,2].min()))
            assert abs(penetration-float(witness['penetration_m']))<1e-10
            assert abs(penetration-physics[step]['table_penetration_m'])<1e-10
            assert abs(penetration-max(r['table_penetration_m'] for r in physics))<1e-10
            assert int(np.flatnonzero(inside)[np.argmin(nodes[inside,2])])==physics[step]['lowest_table_node_index']
            assert np.array_equal(nodes[physics[step]['lowest_table_node_index']],physics[step]['lowest_table_node_position_m'])
        if (episode/'contact-substeps.npy').exists():
            frames=np.load(episode/'contact-substeps.npy',mmap_mode='r')
            count=min(len(frames),len(physics))
            assert len(frames)==round(cfg['tabletop_test']['contact_substeps_seconds']*cfg['simulation']['physics_hz'])+1
            assert np.isfinite(frames[:count]).all()
            initial=np.load(episode/'initial-measured-state.npz')['cloth']
            assert np.array_equal(frames[0],initial)
            for i,nodes in enumerate(data['cloth_nodes']):
                step=(i+1)*stride
                if step<count:assert np.array_equal(frames[step],nodes)
        code="""import json,sys
from pathlib import Path
import tabletop_scoring
p=Path(sys.argv[1]);setup=json.loads((p/'setup.json').read_text());cfg=setup['parameters']
rows=[json.loads(s) for s in (p/'trace.jsonl').read_text().splitlines()]
physics=[json.loads(s) for s in (p/'physics.jsonl').read_text().splitlines()]
meta=json.loads((p/'topology.json').read_text());r=json.loads((p/'result.json').read_text())
name={'settle':'support_score','press_one':'press_score'}.get(setup['case'],'task_score')
scorer=getattr(tabletop_scoring,name)
print(json.dumps(scorer(rows,physics,cfg,meta,r['termination_reason'])))
"""
        score=json.loads(subprocess.check_output([sys.executable,'-B','-c',code,str(episode)],cwd=run/'task-bundle',text=True))
        if 'per_control_scores' in score:
            scores=score.pop('per_control_scores')
            score_rows=[json.loads(s) for s in (episode/'scores.jsonl').read_text().splitlines()]
            assert scores==[r['task_score'] for r in rows]==[r['task_score'] for r in score_rows]
            assert [s==1. for s in scores]==[r['full_task_success'] for r in rows]
            pairs=independent_final_pairs(rows,physics,cfg)
            if not score['invalid_reasons']:
                assert pairs==score['retained_pairs'] and len(pairs)/3==score['task_score'],'Independent final geometry disagrees'
        assert all(score[k]==stored[k] for k in score),'Archived rescore differs'
        entry=dict(episode=episode.name,seed=setup['seed'],independent_rescore_matches=True,
            maximum_CPU_metric_difference=error.tolist(),physics_states=len(physics),controls=len(rows),
            diagnostic_pass=stored.get('diagnostic_pass'),task_score=stored['task_score'],
            full_task_success=stored['full_task_success'],physical_gpu_verified=True,
            maximum_clearance_difference=clearance_error,videos={})
        if setup['recorded_cameras']:
            import imageio.v2 as imageio
            for camera in ('top_cam','left_cam','right_cam'):
                path=episode/(camera+'.mp4')
                if not rows:
                    assert not stored['full_task_success'] and stored['task_score']==0.
                    entry['videos'][camera]=dict(decoded_frames=0,not_created=True)
                    continue
                reader=imageio.get_reader(path);info=reader.get_meta_data()
                count=sum(1 for _ in reader);reader.close()
                assert count==len(rows) and abs(info['fps']-control_hz(cfg))<1e-6 and 'h264' in info['codec']
                entry['videos'][camera]=dict(decoded_frames=count,fps=info['fps'],size=info['size'],
                    codec=info['codec'],sha256=hashlib.sha256(path.read_bytes()).hexdigest(),decode_pass=True)
        result['episodes'].append(entry)
    evidence={'task-bundle/'+name:value for name,value in hashes.items()}
    for name in ('manifest.json','preflight.json','exit.json'):
        with (run/name).open('rb') as stream:evidence[name]=hashlib.file_digest(stream,'sha256').hexdigest()
    for episode in sorted(run.glob('episode-*')):
        for name in ('setup.json','topology.json','initial.usda','initial-measured-state.npz',
                     'control-steps.npz','physics.jsonl','trace.jsonl','scores.jsonl','result.json','gpu.json','device-verification.json',
                     'attachment-audit.json','worst-table-contact.npz','contact-substeps.npy',
                     'top_cam.mp4','left_cam.mp4','right_cam.mp4'):
            path=episode/name
            if path.exists():
                with path.open('rb') as stream:evidence[str(path.relative_to(run))]=hashlib.file_digest(stream,'sha256').hexdigest()
    result['evidence_file_sha256']=evidence
    result['verifier_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    result['launcher_exit']=json.loads((run/'exit.json').read_text())
    result['same_process_reset_checks']=reset_checks
    result['verification_pass']=result['launcher_exit']['returncode']==0 and bool(result['episodes'])
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert args.run.resolve().is_relative_to(OUTPUT) and args.out.resolve().is_relative_to(OUTPUT)
    report=verify(args.run);args.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(verification_pass=report['verification_pass'],episodes=len(report['episodes']))))
