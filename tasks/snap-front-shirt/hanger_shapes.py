"""Estimated physical wooden-hanger geometry shared by authoring and audits."""

import numpy as np


def wooden_beams(hanger):
    """Return the two rectangular shoulder beams in the hanger frame."""
    thickness=hanger['wood_thickness_m']
    height=hanger['wood_shoulder_height_m']
    drop=hanger['arm_drop_m']
    # Include the beam's projected cross-section in the published overall width.
    lo,hi=0.,hanger['width_m']/2
    for _ in range(50):
        y=(lo+hi)/2
        outer=y+height/2*drop/np.hypot(y,drop)
        if outer>hanger['width_m']/2:
            hi=y
        else:
            lo=y
    result=[]
    for sign in (1,-1):
        a=np.zeros(3)
        b=np.array([0.,sign*lo,-drop])
        length=np.linalg.norm(b-a)
        z=(b-a)/length
        x=np.array([1.,0.,0.])
        y=np.cross(z,x)
        result.append(dict(a=a,b=b,center=(a+b)/2,rotation=np.column_stack((x,y,z)),
                           size=np.array([thickness,height,length])))
    return result


def triangle_box_intersections(tri,half):
    """Test complete triangles against a centred axis-aligned box using SAT."""
    tri=np.asarray(tri,dtype=float)
    half=np.asarray(half,dtype=float)
    hit=np.all((tri.min(axis=1)<=half)&(tri.max(axis=1)>=-half),axis=1)
    edges=[tri[:,1]-tri[:,0],tri[:,2]-tri[:,1],tri[:,0]-tri[:,2]]
    axes=[np.cross(edges[0],edges[1])]
    for edge in edges:
        for axis in np.eye(3):
            axes.append(np.cross(edge,axis))
    for axis in axes:
        projections=np.einsum('nij,nj->ni',tri,axis)
        radius=np.abs(axis)@half
        hit&=(projections.min(axis=1)<=radius+1e-14)&(projections.max(axis=1)>=-radius-1e-14)
    return hit


def wooden_initial_clearance(nodes,faces,pose,hanger,margin):
    """Prove triangle separation from shoulder boxes inflated by a margin."""
    from robot_geometry import rotation
    local=(np.asarray(nodes)-np.asarray(pose[:3]))@rotation(pose[3:])
    counts=[]
    for beam in wooden_beams(hanger):
        tri=((local-beam['center'])@beam['rotation'])[faces]
        counts.append(int(triangle_box_intersections(tri,beam['size']/2+margin).sum()))
    return dict(inflation_margin_m=margin,intersecting_triangle_counts=counts,
                separated=not any(counts),method='complete_triangle_vs_inflated_shoulder_box_SAT')


def configure_wooden_hanger(cfg):
    """Select a representative 430 mm wooden hanger, with explicit estimates."""
    h=cfg['hanger']
    h.update(body_style='wooden',wood_thickness_m=.014,wood_shoulder_height_m=.020,
             crossbar_radius_m=.004,hook_wire_radius_m=.00175,mass_kg=.100,
             bar_geometry='rectangular_shoulders_and_capsule_crossbar',
             bar_geometry_note='Two visible/colliding 14 x 20 mm shoulder beams, 8 mm capsule crossbar; legacy body_radius_m is unused for this variant.',
             material_assumption_note='This explicit wooden alternative supersedes the baseline polymer-body assumption; no measured SKU calibration.',
             description='Representative wooden hanger; published width/thickness, estimated profile, mass and hook. No SKU replica claimed.',
             reference='https://www.ikea.com/my/en/p/bumerang-hanger-natural-30238538/',
             published=dict(width_m=.43,thickness_m=.014,body='solid wood with clear acrylic lacquer',hook='galvanized steel'),
             estimates=dict(shoulder_height_m=.020,crossbar_diameter_m=.008,hook_wire_diameter_m=.0035,
                            effective_mass_kg=.100,friction=.4,hardware_measurements_available=False),
             mass_basis='Approx. 76 g shoulder wood + 13 g crossbar at assumed 600 kg/m^3, plus roughly 10 g steel hook/stem; rounded to 100 g. Package weight is not a hanger measurement.')
    cfg['garment'].update(back_depth_m=.017,inner_front_depth_m=-.011,outer_front_depth_m=-.025)
    cfg['garment_test']['wooden_reset_note']='Wider initial front/back clearance for the visible 14 mm wood thickness; same 14 mm open placket separation and fabric material. No runtime pose writes.'
