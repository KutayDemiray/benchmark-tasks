"""Read-only rigid-contact measurements for the bounded bimanual diagnostic."""

from bench import write


def add_pilot_measurements(measure,meta,destination):
    """Observe robot/table and inter-arm contacts without changing mechanics."""
    from yam_arena.robot import asset_link_paths
    paths=asset_link_paths();root=meta['root'].rsplit('/',1)[0]
    arms={side:[root+'/'+name+'/'+suffix.lstrip('/') for suffix in paths.values()]
          for side,name in (('left','LeftArm'),('right','RightArm'))}
    robot_paths=arms['left']+arms['right']
    table=measure.sim.create_rigid_contact_view(robot_paths,
        filter_patterns=[[meta['root']+'/table']]*len(robot_paths))
    interarm=measure.sim.create_rigid_contact_view(arms['left'],
        filter_patterns=[arms['right']]*len(arms['left']))
    assert table.sensor_count==len(robot_paths) and table.filter_count==1
    assert interarm.sensor_count==len(arms['left']) and interarm.filter_count==len(arms['right'])
    table_indices={side:[list(table.sensor_paths).index(path) for path in arms[side]] for side in arms}
    original=measure.extra_observer
    measure.pilot_peak_table_force=0.
    measure.pilot_peak_interarm_force=0.
    def observe(current,nodes,poses):
        result=original(current,nodes,poses)
        force=table.get_contact_force_matrix(current.base.physics_dt)
        norms=current.torch.linalg.vector_norm(force,dim=-1).sum(dim=1).cpu().numpy()
        contacts={side:float(norms[indices].sum()) for side,indices in table_indices.items()}
        cross=interarm.get_contact_force_matrix(current.base.physics_dt)
        cross_norm=float(current.torch.linalg.vector_norm(cross,dim=-1).sum().item())
        current.pilot_peak_table_force=max(current.pilot_peak_table_force,max(contacts.values()))
        current.pilot_peak_interarm_force=max(current.pilot_peak_interarm_force,cross_norm)
        result.update(robot_table_contact_norm_N=contacts,interarm_contact_norm_N=cross_norm)
        return result
    measure.extra_observer=observe
    write(destination/'pilot-contact-sensors.json',dict(robot_table_sensors=table.sensor_paths,
        table_indices=table_indices,interarm_sensors=interarm.sensor_paths,
        rigid_only=True,distributed_robot_cloth_force_measured=False))
