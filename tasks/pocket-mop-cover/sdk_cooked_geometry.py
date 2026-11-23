"""Read actual SDK cooked robot hulls from a byte-identical pre-episode scene."""

from task_paths import OUTPUT_ROOT

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial import ConvexHull,cKDTree
from scipy.spatial.transform import Rotation


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def properties(obj):
    return {name:value['last'] for name,value in obj['properties'].items()}


def hulls(record,authored,reference):
    """Extract native shape vertices in their owning robot-link coordinates."""
    objects={}
    for obj in record['objects']:
        if obj['class_name'] not in ('PxArticulationLink','PxShape','PxConvexMeshGeometry','PxConvexMesh'):
            continue
        key=(obj['context'],obj['handle'])
        assert key not in objects, 'Native handles were reused; use a shorter capture'
        objects[key]=obj
    actors={properties(obj).get('name'):obj for obj in record['objects'] if obj['class_name']=='PxArticulationLink'}
    result={};comparisons=[]
    for side,root in (('left','LeftArm'),('right','RightArm')):
        output=[]
        for body in authored[side]['body_names']:
            matches=[value for path,value in actors.items() if path and path.startswith('/World/envs/env_0/'+root+'/') and path.rsplit('/',1)[-1]==body]
            assert len(matches)==1, (side,body)
            actor=matches[0]
            context=actor['context']
            shape_ids=[event['value'] for event in actor['list_events'] if event['command']==7 and event['property']=='shapes']
            assert len(shape_ids)==len(set(shape_ids))
            native=[]
            for shape_id in shape_ids:
                shape=objects[context,shape_id];sp=properties(shape)
                flag=sp['shapeFlags']['value']
                if not flag&1:continue
                geometry=objects[context,sp['geom']];gp=properties(geometry)
                assert geometry['class_name']=='PxConvexMeshGeometry'
                mesh=objects[context,gp['convexMesh']]
                assert mesh['class_name']=='PxConvexMesh'
                points=np.asarray(properties(mesh)['verts'],dtype=float).reshape(-1,3)
                assert np.isfinite(points).all() and len(points)>=4
                scale=np.asarray(gp['scale'])
                assert scale.shape==(3,) and np.all(scale>0)
                pose=np.asarray(sp['localPose']);assert pose.shape==(7,)
                local=Rotation.from_quat(pose[:4]).apply(points*scale)+pose[4:]
                native.append(dict(points=local,shape_handle=shape_id,mesh_handle=gp['convexMesh'],
                                   contact_offset_m=sp['contactOffset'],rest_offset_m=sp['restOffset']))
            expected=[item for item in reference[side]['colliders'] if item['body']==body]
            assert len(native)==len(expected)>0, f'Native robot shape count differs: {side}/{body}'
            # Names are audit labels only. The native geometry itself is always retained.
            def descriptor(points):
                values=np.asarray(points)
                return np.r_[values.mean(0),values.min(0),values.max(0)]
            first=np.array([descriptor(item['points']) for item in expected])
            second=np.array([descriptor(item['points']) for item in native])
            rows,columns=linear_sum_assignment(np.linalg.norm(first[:,None]-second[None],axis=-1))
            assert np.array_equal(rows,np.arange(len(expected)))
            for item,index in zip(expected,columns):
                actual=native[index];points=actual['points'];original=np.asarray(item['points'])
                error=max(cKDTree(points).query(original)[0].max(),cKDTree(original).query(points)[0].max())
                convex=ConvexHull(points)
                output.append(dict(body=body,name=item['name'],points=points[convex.vertices].tolist(),
                    planes=convex.equations.tolist(),native_vertex_count=len(points),
                    native_shape_handle=actual['shape_handle'],native_mesh_handle=actual['mesh_handle'],
                    reference_kit_hull_difference_m=float(error)))
                comparisons.append(dict(side=side,body=body,name=item['name'],hausdorff_distance_m=float(error)))
        by_name={item['name']:item for item in output}
        assert len(by_name)==len(output)
        ordered=[by_name[item['name']] for item in authored[side]['colliders']]
        result[side]={**authored[side],'colliders':ordered,'geometry_source':'native_sdk_omnipvd',
                      'scene_modified':False}
    return result,comparisons


def capture(run,source,geometry_kind="robot"):
    """Launch a short native inspection under the parent's verified reservation."""
    from gpu_coordination import require_reservation
    require_reservation()
    assert geometry_kind in ('robot','gripper')
    authored_file=geometry_kind+'-colliders.json'
    cooked_file=geometry_kind+'-cooked-colliders.json'
    out=run/'sdk-geometry'
    out.mkdir()
    shutil.copyfile(run/'scene.usda',out/'scene.usda')
    scene_hash=digest(run/'scene.usda')
    inherited=os.environ.get('ROBOCURVE_LOCK_FDS',os.environ.get('POCKET_GPU_LOCK_FD'))
    assert inherited, 'The complete inspection tree must retain the verified reservation'
    descriptors=tuple(int(value) for value in inherited.split(','))
    with (out/'console.log').open('w') as log:
        subprocess.run([sys.executable,'-B',str(Path(__file__).resolve()),'--capture',str(out)],
            check=True,pass_fds=descriptors,stdout=log,stderr=subprocess.STDOUT)
    assert digest(out/'scene.usda')==scene_hash==digest(run/'scene.usda')
    from read_native_properties import read_capture
    paths=list((out/'omnipvd').rglob('*.ovd'));assert len(paths)==1
    record=read_capture(paths[0],array_properties=('PxConvexMesh.verts',))
    assert not record['messages'], 'Native diagnostic capture reported errors/messages'
    authored=json.loads((run/authored_file).read_text())
    reference=json.loads((source/cooked_file).read_text())
    cooked,comparison=hulls(record,authored,reference)
    report=dict(scene_sha256=scene_hash,native_capture_sha256=record['sha256'],
        native_capture=str(paths[0].relative_to(run)),parsed_all_bytes=record['parsed_all_bytes'],
        scene_modified=False,physics_state_reused=False,native_geometry_used=True,
        property_source='PhysX PxArticulationLink/PxShape/PxConvexMeshGeometry/PxConvexMesh OmniPVD objects',
        stock_kit_reference=source.name,comparisons=comparison,
        maximum_reference_hull_difference_m=max(item['hausdorff_distance_m'] for item in comparison))
    (run/'native-geometry-capture.json').write_text(json.dumps(report,indent=2))
    files=['scene.usda',authored_file,'versions.json','manifest.json','native-geometry-capture.json',str(paths[0].relative_to(run))]
    cooked['_provenance']=dict(source_run=run.name,scene_modified=False,
        native_api='Installed ovphysx OmniPVD cooked geometry from a byte-identical scene before the episode',
        source_sha256={name:digest(run/name) for name in files})
    (run/cooked_file).write_text(json.dumps(cooked,indent=2))


def worker(out):
    """Capture two initialization substeps only; never produce a benchmark recording."""
    from gpu_coordination import require_reservation
    from sdk_session import SDKSession
    require_reservation()
    sim=None
    try:
        sim=SDKSession(out,record_geometry=True,scene_path=out.parent/'scene.usda')
        request=json.loads((out.parent/'request.json').read_text())
        for _ in range(2):sim.step(1/request['physics_hz'])
        binding=sim.bind('ARTICULATION_LINK_POSE','/World/envs/env_0/LeftArm')
        sim.verify(binding)
        assert not sim.errors
    finally:
        if sim is not None:sim.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--capture',type=Path,required=True)
    args=parser.parse_args()
    assert args.capture.resolve().is_relative_to(OUTPUT_ROOT)
    worker(args.capture)
