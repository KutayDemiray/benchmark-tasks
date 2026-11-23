"""CPU audit of frozen garment diagnostics, including failed experiments."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import numpy as np

from robot_geometry import rotation

OUTPUT=OUTPUT_ROOT


def verify_authored_wooden_geometry(episode,nodes,faces,pose,hanger):
    """Check penetration against actual exported box transforms independently."""
    from scipy.spatial.transform import Rotation
    from garment_contact_audit import bar_contacts
    text=(episode/'initial.usda').read_text()
    local=(nodes-np.asarray(pose[:3]))@Rotation.from_quat(pose[3:]).as_matrix()
    audited=bar_contacts(nodes,faces,pose,hanger)
    checks=[]
    for i in range(2):
        chunk=text.split(f'def Cube "bar{i}"',1)[1].split('def Material',1)[0]
        assert '"PhysicsCollisionAPI"' in chunk
        assert '["xformOp:translate", "xformOp:orient", "xformOp:scale"]' in chunk
        def vector(name):
            value=re.search(r'xformOp:'+name+r' = \(([^)]+)\)',chunk)[1]
            return np.fromstring(value,sep=',')
        q_wxyz=vector('orient')
        r=Rotation.from_quat(np.r_[q_wxyz[1:],q_wxyz[0]]).as_matrix()
        half=vector('scale')/2
        points=(local-vector('translate'))@r
        depth=float(max(0.,(half-np.abs(points)).min(axis=1).max()))
        difference=abs(depth-audited[i]['maximum_vertex_penetration_m'])
        assert difference<1e-6,'Authored box differs from contact auditor'
        checks.append(dict(bar=i,authored_size_m=(2*half).tolist(),
            actual_USD_box_penetration_m=depth,auditor_difference_m=difference,
            visual_and_collision_use_same_prim=True))
    return checks


def verify(run):
    """Verify evidence consistency without promoting an intermediate result."""
    manifest=json.loads((run/'manifest.json').read_text())
    hashes={name:hashlib.sha256((run/'task-bundle'/name).read_bytes()).hexdigest() for name in manifest['sources']}
    assert hashes==manifest['sources'],'source_modified'
    assert hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()==manifest['bundle_sha256']
    result=dict(run=str(run),source_bundle_sha256=manifest['bundle_sha256'],source_hashes_match=True,
                episodes=[],full_task_validated=False)
    for episode in sorted(run.glob('episode-*')):
        setup=json.loads((episode/'setup.json').read_text())
        cfg=setup['parameters']
        stored=json.loads((episode/'result.json').read_text())
        rows=[json.loads(line) for line in (episode/'trace.jsonl').read_text().splitlines()]
        physical=[json.loads(line) for line in (episode/'physics.jsonl').read_text().splitlines()]
        topology=json.loads((episode/'topology.json').read_text())
        data=np.load(episode/'control-steps.npz')
        stride=round(cfg['simulation']['physics_hz']/10)
        assert len(physical)==len(rows)*stride+1,'missing_physics_steps'
        assert [r['physics_step'] for r in physical]==list(range(len(physical))),'physics_step_sequence'
        assert np.array_equal(data['actions'],np.array([r['robot_action'] for r in rows])),'actions_differ'
        assert data['actions'].shape==(len(rows),14),'wrong_action_dimension'
        assert np.isfinite(data['actions']).all(),'nonfinite_actions'
        assert np.all((data['actions'][:,[6,13]]>=0)&(data['actions'][:,[6,13]]<=1)),'invalid_gripper_action'
        assert len(data['cloth_nodes'])==len(rows),'missing_cloth_frames'
        max_strain=max(r['strain'] for r in physical)
        max_rivet=max(r['rivet_error_m'] for r in physical)
        assert abs(max_strain-stored['max_cloth_strain'])<1e-12,'max_strain_differs'
        assert abs(max_rivet-stored['max_rivet_error_m'])<1e-12,'max_rivet_differs'
        edges=np.asarray(topology['cloth']['edges'])
        rest=np.asarray(topology['cloth']['rest_lengths'])
        max_strain_delta=max_rivet_delta=0.
        for row,nodes in zip(rows,data['cloth_nodes'],strict=True):
            assert row['physics']==physical[(row['step']+1)*stride],'control_state_not_native_substep'
            assert row['full_task_score']==0 and row['full_task_success'] is False,'intermediate_task_credit'
            nodes=nodes.astype(np.float64)
            strain=float(np.abs(np.linalg.norm(nodes[edges[:,0]]-nodes[edges[:,1]],axis=1)/rest-1).max())
            error=0.
            for att in topology['cloth']['attachments']:
                pose=np.asarray(row['physics']['hardware'][att['hardware']])
                expected=np.asarray(att['local'])@rotation(pose[3:]).T+pose[:3]
                error=max(error,float(np.linalg.norm(nodes[att['ids']]-expected,axis=1).max()))
            max_strain_delta=max(max_strain_delta,abs(strain-row['physics']['strain']))
            max_rivet_delta=max(max_rivet_delta,abs(error-row['physics']['rivet_error_m']))
        assert max_strain_delta<1e-10 and max_rivet_delta<1e-10,'GPU_measurement_differs_from_CPU'
        if (episode/'contact-substeps.npy').exists():
            substeps=np.load(episode/'contact-substeps.npy',mmap_mode='r')
            expected=round(cfg['garment_test']['contact_substeps_seconds']*cfg['simulation']['physics_hz'])+1
            assert len(substeps)==expected,'contact_substep_capacity'
            count=min(expected,len(physical))
            assert np.isfinite(substeps[:count]).all(),'nonfinite_contact_record'
            initial=np.load(episode/'initial-measured-state.npz')['cloth']
            assert np.array_equal(substeps[0],initial),'contact_initial_state_differs'
            for i,nodes in enumerate(data['cloth_nodes']):
                step=(i+1)*stride
                if step<count:
                    assert np.array_equal(nodes,substeps[step]),'contact_control_state_differs'
            result.setdefault('contact_substep_recordings',[]).append(dict(
                episode=episode.name,valid_recorded_frames=count,physics_hz=cfg['simulation']['physics_hz'],
                sha256=hashlib.sha256((episode/'contact-substeps.npy').read_bytes()).hexdigest()))
        # Execute the archived scorer in a separate interpreter with its own
        # module search path. No live task implementation enters this rescore.
        code="""import json,sys
from pathlib import Path
from garment_runner import passive_score
p=Path(sys.argv[1])
rows=[json.loads(x) for x in (p/'trace.jsonl').read_text().splitlines()]
physics=[json.loads(x) for x in (p/'physics.jsonl').read_text().splitlines()]
cfg=json.loads((p/'setup.json').read_text())['parameters']
r=json.loads((p/'result.json').read_text())
print(json.dumps(passive_score(rows,cfg,r['termination_reason'],max(x['strain'] for x in physics),max(x['rivet_error_m'] for x in physics),all(x['finite'] for x in physics))))
"""
        rescore=json.loads(subprocess.check_output([sys.executable,'-B','-c',code,str(episode)],cwd=run/'task-bundle',text=True))
        assert all(rescore[k]==stored[k] for k in rescore),'frozen_rescore_differs'
        entry=dict(episode=episode.name,seed=setup['seed'],rescore_matches=True,
                   physics_steps=len(physical),control_steps=len(rows),max_CPU_strain_difference=max_strain_delta,
                   max_CPU_rivet_difference_m=max_rivet_delta,diagnostic_pass=stored['diagnostic_pass'],
                   termination_reason=stored['termination_reason'],videos={})
        if cfg['hanger'].get('body_style')=='wooden':
            entry['authored_wooden_geometry']=verify_authored_wooden_geometry(
                episode,data['cloth_nodes'][0].astype(float),np.asarray(topology['cloth']['faces']),
                rows[0]['physics']['hardware']['hanger'],cfg['hanger'])
        if setup['recorded_cameras']:
            import imageio.v2 as imageio
            for camera in ('top_cam','left_cam','right_cam'):
                path=episode/(camera+'.mp4')
                reader=imageio.get_reader(path)
                info=reader.get_meta_data()
                count=sum(1 for _ in reader)
                reader.close()
                assert count==len(rows) and abs(info['fps']-10)<1e-6 and 'h264' in info['codec'],'video_encoding_or_length'
                entry['videos'][camera]=dict(decoded_frames=count,fps=info['fps'],codec=info['codec'],size=info['size'],
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),decode_pass=True)
        result['episodes'].append(entry)
    result['launcher_exit']=json.loads((run/'exit.json').read_text())
    result['verification_pass']=result['launcher_exit']['returncode']==0 and bool(result['episodes'])
    result['note']='Evidence verification can pass for a failed experiment. Decoding does not prove camera visibility or task success.'
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert args.run.resolve().is_relative_to(OUTPUT) and args.out.resolve().is_relative_to(OUTPUT)
    result=verify(args.run)
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('verification_pass','source_bundle_sha256','full_task_validated')}))
