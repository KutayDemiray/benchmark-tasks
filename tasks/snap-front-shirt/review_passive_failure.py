"""Plot recorded passive diagnostics and distinguish a plane guard from solid geometry."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from bench import write
from task_paths import OUTPUT_ROOT


def review(episode,out):
    """Keep the canonical failed outcome unchanged while explaining its measurements."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    setup=json.loads((episode/'setup.json').read_text())
    result=json.loads((episode/'result.json').read_text())
    table=setup['parameters']['tabletop']
    rows=[json.loads(line) for line in (episode/'trace.jsonl').open()]
    witness=np.load(episode/'worst-table-contact.npz')
    nodes=witness['nodes'].astype(float)
    low=np.r_[np.array(table['table_center_xy_m'])-np.array(table['table_size_xy_m'])/2,
              table['table_top_m']-table['table_thickness_m']]
    high=np.r_[np.array(table['table_center_xy_m'])+np.array(table['table_size_xy_m'])/2,table['table_top_m']]
    inside_xy=((nodes[:,:2]>low[:2])&(nodes[:,:2]<high[:2])).all(axis=1)
    inside_solid=((nodes>low)&(nodes<high)).all(axis=1)
    box_depth=np.minimum(nodes-low,high-nodes).min(axis=1)
    index=int(np.where(inside_xy,high[2]-nodes[:,2],-np.inf).argmax())
    report=dict(episode=str(episode),seed=setup['seed'],canonical_result_unchanged=True,
        termination_reason=result['termination_reason'],diagnostic_pass=result['diagnostic_pass'],
        simulated_seconds=result['control_steps']/setup['control_hz'],
        maximum_native_strain=result['max_cloth_strain'],
        reported_maximum_tabletop_plane_depth_m=result['max_table_penetration_m'],
        witness_physics_step=int(witness['physics_step']),flagged_node=index,
        flagged_position_m=nodes[index].tolist(),table_low_xyz_m=low.tolist(),table_high_xyz_m=high.tolist(),
        nodes_strictly_inside_solid_at_witness=int(inside_solid.sum()),
        maximum_nodal_depth_inside_solid_box_m=float(max(0.,box_depth.max())),
        nodes_under_underside_within_footprint=int((inside_xy&(nodes[:,2]<low[2])).sum()),
        final_table_coverage=rows[-1]['physics']['table_coverage_fraction'],
        first_control_below_98_percent=next((r['time_s'] for r in rows if r['physics']['table_coverage_fraction']<.98),None),
        interpretation='The existing guard measures depth below the tabletop plane within its XY footprint. '
            'That measurement also includes cloth below the underside and does not alone prove penetration '
            'of the finite solid table. This snapshot check examines nodes, not every triangle or earlier state. '
            'The canonical outcome and all thresholds remain unchanged.',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    report['raw_evidence_sha256']={}
    for name in ('setup.json','result.json','trace.jsonl','worst-table-contact.npz'):
        with (episode/name).open('rb') as stream:
            report['raw_evidence_sha256'][name]=hashlib.file_digest(stream,'sha256').hexdigest()
    times=np.array([r['time_s'] for r in rows])
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    axes[0,0].plot(times,[100*r['physics']['table_coverage_fraction'] for r in rows])
    axes[0,0].axhline(98,color='firebrick',ls='--',label='unchanged 98% criterion')
    axes[0,0].set(xlabel='Simulation time (s)',ylabel='Nodes over tabletop (%)');axes[0,0].legend()
    axes[0,1].plot(times,[100*r['physics']['strain'] for r in rows],label='Control-step samples')
    axes[0,1].axhline(15,color='firebrick',ls='--',label='unchanged strain guard')
    axes[0,1].set(xlabel='Simulation time (s)',ylabel='Maximum absolute edge strain (%)');axes[0,1].legend()
    axes[1,0].scatter(nodes[:,0],nodes[:,1],s=.2,color='#41698a',rasterized=True)
    axes[1,0].add_patch(plt.Rectangle(low[:2],*(high-low)[:2],fill=False,ec='black'))
    axes[1,0].scatter(*nodes[index,:2],color='firebrick',s=28,label='Flagged node')
    axes[1,0].set(xlabel='World x (m)',ylabel='World y (m)',title='Exact worst-plane-depth snapshot',aspect='equal')
    axes[1,0].legend()
    near=np.abs(nodes[:,1]-nodes[index,1])<.025
    axes[1,1].add_patch(plt.Rectangle((low[0],low[2]),high[0]-low[0],high[2]-low[2],color='#d0c0a5'))
    axes[1,1].scatter(nodes[near,0],nodes[near,2],s=9,color='#41698a',label='Nodes in 50 mm y slice')
    axes[1,1].scatter(nodes[index,0],nodes[index,2],s=36,color='firebrick',label='Flagged node')
    axes[1,1].set(xlim=(nodes[index,0]-.08,nodes[index,0]+.08),
        ylim=(min(nodes[index,2]-.025,low[2]-.025),high[2]+.035),xlabel='World x (m)',ylabel='World z (m)',
        title='Finite table section and nearby cloth')
    axes[1,1].legend(loc='lower right')
    for axis in axes.flat:axis.grid(alpha=.2)
    fig.suptitle('Passive shirt diagnostic — measured states, not a camera demonstration')
    fig.tight_layout();fig.savefig(out.with_suffix('.png'),dpi=150);plt.close(fig)
    write(out,report)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('episode',type=Path);parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    assert all(p.resolve().is_relative_to(OUTPUT_ROOT) for p in (args.episode,args.out))
    report=review(args.episode,args.out)
    print(json.dumps({k:report[k] for k in ('diagnostic_pass','maximum_nodal_depth_inside_solid_box_m','final_table_coverage')}))


if __name__=='__main__':main()
