"""Qualified two-pocket material in the stock robot scene, with native measurements."""

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from cover_mesh import cover_mesh
from measurements import pocket_seating
from mesh import audit,material_rest,surface_edges
from robot_gate import array,fabric,surface_asset


class FullCover:
    """Build and observe an unattached sewn cover; never write state during control."""

    def __init__(self,request,out,center):
        config=json.loads(Path(__file__).with_name('parameters.json').read_text())
        self.material=fabric(config)
        self.material.update(collision_pair_updates=request['contact_updates'],
            collision_subiterations=request['contact_subiterations'],speculative_ccd=request['ccd'],
            mesh_spacing_m=request['spacing'],initial_mouth_height_m=request['mouth_height'])
        local,self.faces,self.geometry=cover_mesh(self.material,spacing=request['spacing'],
            mouth_height=request['mouth_height'])
        self.center=np.asarray(center)
        self.initial=local+self.center- np.array([self.geometry['length_m']/2,0.,0.])
        self.rest=material_rest(self.initial,self.material['flat_rest_panels'],self.material['seam_height_m'])
        self.edges=surface_edges(self.faces)
        self.rest_lengths=np.linalg.norm(self.rest[self.edges[:,1]]-self.rest[self.edges[:,0]],axis=1)
        self.max_strain=0.
        self.asset=surface_asset(self.initial,self.faces,self.material,request['iterations'])
        (out/'effective-parameters.json').write_text(json.dumps(self.material,indent=2))
        (out/'cover-mesh.json').write_text(json.dumps(self.geometry,indent=2))
        (out/'mesh.json').write_text(json.dumps({**audit(self.rest,self.faces),
            'markers':self.geometry['pockets'][0]['markers']},indent=2))
        np.savez_compressed(out/'geometry.npz',rest=self.rest,initial=self.initial,faces=self.faces)

    def bind(self,base):
        """Check the native material rest shape after the environment reset."""
        self.obj=base.scene['pocket']
        native=array(self.obj.root_view.get_rest_nodal_positions())[0].reshape(-1,3)[:len(self.rest)]
        assert np.allclose(native,self.rest,atol=1e-7,rtol=0.), 'Native cloth rest shape changed'

    def nodes(self):
        """Return the native cloth positions and velocities."""
        count=len(self.rest)
        points=array(self.obj.root_view.get_simulation_nodal_positions())[0].reshape(-1,3)[:count]
        velocity=array(self.obj.root_view.get_simulation_nodal_velocities())[0].reshape(-1,3)[:count]
        assert np.isfinite(points).all() and np.isfinite(velocity).all(), 'Nonfinite native cloth state'
        return points,velocity

    def observe_native(self):
        """Record every material node and the cumulative all-edge strain bound."""
        points,velocity=self.nodes()
        lengths=np.linalg.norm(points[self.edges[:,1]]-points[self.edges[:,0]],axis=1)
        self.max_strain=max(self.max_strain,float(abs(lengths/self.rest_lengths-1).max()))
        return dict(positions=points,velocities=velocity)

    def observe_control(self,poses,body_names):
        """Measure both pocket volumes against the actual frame end poses."""
        points,velocity=self.nodes()
        seats=[]
        for side,body in enumerate(('Left','Right')):
            pose=poses[body_names.index(body)]
            rotation=Rotation.from_quat(pose[3:]).as_matrix()
            sign=-1. if side==0 else 1.
            center=pose[:3]+rotation@np.array([sign*.142,0.,0.])
            end_rotation=rotation if side==0 else rotation@np.diag([-1.,-1.,1.])
            pocket=self.geometry['pockets'][side]
            seats.append(pocket_seating(points,np.asarray(pocket['closed_faces']),center,[.1,.1,.006],
                markers=pocket['markers'],faces=np.asarray(pocket['faces']),profile='beveled',rotation=end_rotation))
        return dict(pockets=seats,both_pockets_seated=all(seat['seated_samples'] for seat in seats),
            max_abs_edge_strain=self.max_strain,min_cloth_z_m=float(points[:,2].min()),
            max_cloth_speed_m_s=float(np.linalg.norm(velocity,axis=1).max()),
            score_status='Pending independent all-substep contact and outcome audit')
