"""Measured-state checks for the separate tabletop shirt candidate."""

import numpy as np

from robot_scoring import snap_seated
from episode_timing import control_hz,physics_stride,max_actions,retention_samples


def seating_matrix(physics,count,cfg,with_invalid=False):
    """Evaluate all mechanically compatible pairings at every recorded state."""
    male=np.array([[r['hardware'][f'male{i}'] for i in range(count)] for r in physics],dtype=float)
    socket=np.array([[r['hardware'][f'socket{i}'] for i in range(count)] for r in physics],dtype=float)
    def normal(q):
        x,y,z,w=np.moveaxis(q,-1,0)
        return np.stack((2*(x*z+y*w),2*(y*z-x*w),1-2*(x*x+y*y)),axis=-1)
    m=male[:,:,None,:];s=socket[:,None,:,:]
    delta=m[...,:3]-s[...,:3];q=s[...,3:6];w=s[...,6:7]
    relative=delta-2*w*np.cross(q,delta)+2*np.cross(q,np.cross(q,delta))
    aligned=(normal(m[...,3:])*normal(s[...,3:])).sum(axis=-1)>np.cos(np.deg2rad(12))
    p=cfg['snap'];limit=min(p['ring_height_m']+p['cap_thickness_m']/2+.0007,
                          p['socket_ball_height_m']+p['head_from_cap_m']-.0006)
    minimum=p['ring_height_m']+p['cap_thickness_m']/2-cfg['tabletop']['seat_rim_penetration_allowance_m']
    lateral=np.linalg.norm(relative[...,:2],axis=-1)<=.00075
    seated=(lateral&(relative[...,2]>minimum)
        &(relative[...,2]<limit)&aligned&np.isfinite(m).all(axis=-1)&np.isfinite(s).all(axis=-1)
        &(np.abs((m[...,3:]**2).sum(axis=-1)-1)<.01)&(np.abs((s[...,3:]**2).sum(axis=-1)-1)<.01))
    if with_invalid:
        rim_intersection=lateral&aligned&(relative[...,2]>-.001)&(relative[...,2]<minimum)
        return seated,rim_intersection
    return seated


def task_score(rows,physics,cfg,meta,reason):
    """Score retained required closures after complete measured withdrawal.

    No controller labels, intended contact partner, or approach strategy enters
    the mechanics or earns credit. Damage and invalid records earn zero.
    """
    result=dict(stage='tabletop_task',task_score=0.,full_task_success=False,
        original_hanger_task_success=False,invalid_reasons=[],retained_pairs=[],
        per_control_scores=[0.]*len(rows),termination_reason=reason)
    try:
        p=cfg['tabletop'];gate=cfg['gate'];hz=control_hz(cfg);stride=physics_stride(cfg)
        required=p['required_snap_indices'];count=len(cfg['garment']['snap_heights_from_hem_m'])
        assert required==[0,1,2],'unexpected_task_definition'
        assert not meta.get('native_execution_error',False),'native_execution_error'
        assert rows and len(rows)<=max_actions(cfg) and len(physics)==len(rows)*stride+1,'incomplete_episode'
        assert [r['physics_step'] for r in physics]==list(range(len(physics))),'missing_physics_state'
        assert not meta['diagnostic_world_mounts'] and not meta['grasp_attachments'],'hidden_attachment'
        assert not meta.get('hanger_attachments',False),'hidden_attachment'
        assert len(meta['cloth']['attachments'])==2*count,'missing_local_rivets'
        for i,row in enumerate(rows):
            assert row['step']==i and abs(row['time_s']-(i+1)/hz)<1e-7,'invalid_control_time'
            assert row['physics']==physics[(i+1)*stride],'control_physics_mismatch'
            action=np.asarray(row['robot_action'])
            assert action.shape==(14,) and np.isfinite(action).all(),'invalid_action'
            assert np.all((action[[6,13]]>=0)&(action[[6,13]]<=1)),'invalid_gripper_action'
            clearance=np.array([row['robot_clearance_m'][k] for k in ('fabric_m','hardware_m')])
            assert np.isfinite(clearance).all(),'invalid_clearance'
        poses=np.array([list(r['hardware'].values()) for r in physics],dtype=float)
        assert poses.shape[-1]==7 and np.isfinite(poses).all(),'nonfinite_hardware'
        assert (np.abs((poses[...,3:]**2).sum(axis=-1)-1)<.01).all(),'invalid_hardware_quaternion'
        centers=np.array([r['physics']['mean_cloth_position_m'] for r in rows],dtype=float)
        assert centers.shape==(len(rows),3) and np.isfinite(centers).all(),'invalid_cloth_position'
        values=np.array([[r['strain'],r['rivet_error_m'],r['table_penetration_m'],r['table_coverage_fraction']] for r in physics])
        assert np.isfinite(values).all() and (values>=0).all(),'nonfinite_measurement'
        assert (values[:,3]<=1.).all(),'invalid_table_coverage'
        assert all(r['finite'] for r in physics),'nonfinite_state'
        assert values[:,0].max()<=gate['max_edge_strain'],'cloth_strain_limit'
        assert values[:,1].max()<=gate['max_attachment_error_m'],'rivet_error_limit'
        assert values[:,2].max()<=p['max_table_penetration_m'],'table_penetration'
        sensor_names=[]
        for kind in ('male','socket'):
            sensor_names.extend(f'{kind}{i}' for i in range(count))
        contact_rows=[]
        for r in physics:
            contact_rows.append([r['snap_robot_contact_norm_N'][n] for n in sensor_names])
        contact_values=np.asarray(contact_rows)
        assert np.isfinite(contact_values).all() and (contact_values>=0).all(),'invalid_contact_measurement'
        forces=contact_values.max(axis=1)
        assert forces.max()<=25.,'robot_normal_force_limit'
        states,rim_intersection=seating_matrix(physics,count,cfg,with_invalid=True)
        assert not rim_intersection.any(),'cap_through_socket_rim'
        assert not states[0].any(),'initially_closed_snap'
        for i in required:
            assert not snap_seated(meta['initial_hardware'][f'male{i}'],meta['initial_hardware'][f'socket{i}'],cfg),'initially_closed_snap'
        bad_prefix=np.concatenate((np.zeros((1,count,count),dtype=int),np.cumsum(~states,axis=0)))
        wrong=~np.eye(count,dtype=bool)
        wrong_prefix=np.r_[0,np.cumsum(states[:,wrong].any(axis=1))]
        samples=retention_samples(cfg)
        final_retained=[]
        for end in range(samples-1,len(rows)):
            window=rows[end-samples+1:end+1]
            clear=all(min(r.get('robot_clearance_m',{'missing':-1.}).values())>=p['released_robot_clearance_m'] for r in window)
            if not clear:continue
            first=(end-samples+2)*stride;last=(end+1)*stride
            if (values[first:last+1,3]<.98).any() or forces[first:last+1].max()>.05:continue
            centers=np.array([r['physics']['mean_cloth_position_m'] for r in window])
            if np.linalg.norm(centers-centers[-1],axis=1).max()>p['max_released_drift_m']:continue
            if wrong_prefix[last+1]-wrong_prefix[first]:continue
            retained=[i for i in required if bad_prefix[last+1,i,i]-bad_prefix[first,i,i]==0]
            result['per_control_scores'][end]=len(retained)/len(required)
            if end==len(rows)-1:final_retained=retained
        result.update(task_score=result['per_control_scores'][-1],retained_pairs=final_retained,
            full_task_success=len(final_retained)==len(required),max_cloth_strain=float(values[:,0].max()),
            max_rivet_error_m=float(values[:,1].max()),max_table_penetration_m=float(values[:,2].max()),
            peak_robot_normal_contact_N=float(forces.max()))
    except (AssertionError,KeyError,ValueError,TypeError,IndexError) as exc:
        result['invalid_reasons'].append(str(exc))
    return result


def support_score(rows,physics,cfg,meta,reason):
    """Validate passive whole-shirt support without awarding task credit."""
    p=cfg['tabletop'];gate=cfg['gate']
    invalid=[]
    max_strain=max((r['strain'] for r in physics),default=float('inf'))
    max_rivet=max((r['rivet_error_m'] for r in physics),default=float('inf'))
    max_pen=max((r['table_penetration_m'] for r in physics),default=float('inf'))
    for condition,label in (
        (bool(rows) and bool(physics),'empty_record'),
        (all(r['finite'] for r in physics),'nonfinite_state'),
        (max_strain<=gate['max_edge_strain'],'cloth_strain_limit'),
        (max_rivet<=gate['max_attachment_error_m'],'rivet_error_limit'),
        (max_pen<=p['max_table_penetration_m'],'table_penetration'),
        (not meta['diagnostic_world_mounts'] and not meta['grasp_attachments'],'hidden_attachment'),
    ):
        if not condition:invalid.append(label)
    count=len(cfg['garment']['snap_heights_from_hem_m'])
    first_seated={}
    for row in physics:
        for i in range(count):
            if snap_seated(row['hardware'][f'male{i}'],row['hardware'][f'socket{i}'],cfg):
                first_seated.setdefault(i,row['physics_step'])
    tail=rows[-round(2*control_hz(cfg)):]
    centers=np.array([r['physics']['mean_cloth_position_m'] for r in tail])
    drift=float(np.linalg.norm(centers-centers[-1],axis=1).max()) if len(tail) else float('inf')
    checks=dict(intact=not invalid,no_uncommanded_seating=not first_seated,
        shirt_on_table=bool(tail and min(r['physics']['table_coverage_fraction'] for r in tail)>.98),
        stable=drift<p['max_released_drift_m'],
        completed=reason=='diagnostic_completed' and len(rows)>=round(5*control_hz(cfg)))
    return dict(stage='tabletop_support_diagnostic',diagnostic_pass=all(checks.values()),
        checks=checks,invalid_reasons=invalid,first_seated_physics_steps=first_seated,
        max_cloth_strain=max_strain,max_rivet_error_m=max_rivet,max_table_penetration_m=max_pen,
        final_two_seconds_drift_m=drift,task_score=0.,full_task_success=False,
        original_hanger_task_success=False,note='Whole-shirt table support only; no robot fastening demonstrated.')


def press_score(rows,physics,cfg,meta,reason):
    """Validate one robot closure as development; never claim the three-snap task."""
    result=support_score(rows,physics,cfg,meta,reason)
    result['stage']='tabletop_one_snap_diagnostic'
    forces=[r.get('snap_robot_contact_norm_N',{}).get('male0',0.) for r in physics]
    all_forces=[v for r in physics for v in r.get('snap_robot_contact_norm_N',{}).values()]
    peak=max(all_forces,default=0.)
    contact=next((i for i,f in enumerate(forces) if f>.05),None)
    seated=next((i for i,r in enumerate(physics) if snap_seated(r['hardware']['male0'],r['hardware']['socket0'],cfg)),None)
    samples=retention_samples(cfg)
    tail=rows[-samples:]
    released=bool(len(tail)==samples and tail[-1]['time_s']-tail[0]['time_s']>=3.-1e-8)
    if released:
        first=round(tail[0]['time_s']*cfg['simulation']['physics_hz'])
        released=all(snap_seated(r['hardware']['male0'],r['hardware']['socket0'],cfg)
            and r['snap_robot_contact_norm_N']['male0']<.05 for r in physics[first:])
        released &= all(np.linalg.norm(np.asarray(r['measured_left_tcp_m'])-r['physics']['hardware']['male0'][:3])>.06
                        for r in tail)
    checks=dict(intact=not result['invalid_reasons'],press_after_contact=contact is not None and seated is not None and contact<seated,
                retained_after_withdrawal=released,rigid_contact_under_25N=peak<=25.,
                controller_completed=reason=='controller_completed')
    if peak>25.:result['invalid_reasons'].append('robot_normal_force_limit')
    result.update(checks=checks,diagnostic_pass=all(checks.values()),peak_robot_normal_contact_N=peak,
        first_robot_contact_step=contact,first_target_seated_step=seated,
        note='One aligned-start full-shirt snap diagnostic. Three-snap task/scorer and independent placket alignment remain unvalidated.',
        task_score=0.,full_task_success=False)
    return result
