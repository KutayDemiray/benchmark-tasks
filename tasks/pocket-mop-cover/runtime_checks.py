"""Read native tensors and verify the reserved physics and render device."""

import json
import os
import re
import subprocess
import xml.etree.ElementTree as ET

from gpu_coordination import require_reservation, scheduler_binding


def native_tensor(deform):
    """Read the actual backend rather than trusting an asset device property."""
    view = deform.root_view
    for method in ('get_simulation_nodal_positions', 'get_link_transforms', 'get_transforms'):
        if hasattr(view, method):
            value = getattr(view, method)()
            return getattr(value, 'torch', value), method
    raise AssertionError('No supported native readback on this physics object')


def verification_checks(report, expected_tensor_device):
    """Require actual allocation, renderer and native readback to match admission."""
    index, uuid = report['expected_index'], report['expected_uuid']
    settings = report['settings']
    return dict(
        physics_setting=settings['/physics/cudaDevice'] == 0,
        render_setting=settings['/renderer/activeGpu'] == index,
        single_renderer=settings['/renderer/multiGpu/enabled'] is False,
        relative_renderer_cuda=settings['/renderer/multiGpu/activeCudaGpus'] == '0,',
        physics_device=report['simulation_device'] == 'cuda:0' and report['physics_tensor_device'] == expected_tensor_device,
        gpu_scene=bool(report['scene_gpu_dynamics']) and all(report['scene_gpu_dynamics'].values()),
        cuda_uuid=report['cuda_uuid'].removeprefix('GPU-').lower() == uuid.removeprefix('GPU-').lower(),
        actual_renderer=report['active_renderer_physical_indices'] == [index],
        compute_observed=any(p['uuid'] == uuid and 'C' in p['type'] for p in report['allocations']),
        no_other_work=all(p['uuid'] == uuid or (p['type'] == 'G' and p['memory_mib'] <= 16)
            for p in report['allocations']))


def verify_gpu(out, sim, deform, *, expected_tensor_device='cuda:0'):
    """Verify scheduled or exclusive CUDA/host readback after native warmup."""
    import carb
    import torch
    import omni.usd
    from pxr import UsdPhysics
    assignment = require_reservation()
    assert expected_tensor_device in ('cuda:0', 'cpu')
    # Rendering alone does not advance the recorded physics trajectory.
    sim.render()
    tensor, method = native_tensor(deform)
    settings = carb.settings.get_settings()
    keys = ('/physics/cudaDevice', '/renderer/activeGpu', '/renderer/multiGpu/enabled',
        '/renderer/multiGpu/activeCudaGpus')
    xml = subprocess.check_output(['nvidia-smi', '-q', '-x'], text=True, timeout=5)
    allocations = []
    for gpu in ET.fromstring(xml).findall('gpu'):
        for process in gpu.findall('processes/process_info'):
            if process.findtext('pid') == str(os.getpid()):
                allocations.append(dict(uuid=gpu.findtext('uuid'), type=process.findtext('type'),
                    memory_mib=float(process.findtext('used_memory').split()[0])))
    active = sorted(set(int(value) for value in re.findall(r'^\|\s*(\d+)\s*\|[^\n]*Yes:',
        (out/'kit.log').read_text(errors='replace'), flags=re.MULTILINE)))
    stage = omni.usd.get_context().get_stage()
    scenes = {str(prim.GetPath()): prim.GetAttribute('physxScene:enableGPUDynamics').Get()
        for prim in stage.Traverse() if prim.IsA(UsdPhysics.Scene)}
    report = dict(pid=os.getpid(), job_id=os.environ.get('ROBOCURVE_JOB_ID'),
        expected_index=assignment['index'], expected_uuid=assignment['uuid'],
        settings={key: settings.get(key) for key in keys},
        simulation_device=str(sim.device), physics_tensor_device=str(tensor.device),
        native_tensor_method=method, expected_tensor_device=expected_tensor_device,
        scene_gpu_dynamics=scenes,
        host_readback_preserved=expected_tensor_device == 'cpu',
        cuda_uuid=str(torch.cuda.get_device_properties(0).uuid),
        active_renderer_physical_indices=active, allocations=allocations, verified=False)
    checks = verification_checks(report, expected_tensor_device)
    report.update(checks=checks, verified=all(checks.values()))
    if 'ROBOCURVE_JOB_ID' in os.environ:
        scheduler_binding()  # imports only the canonical scheduler path
        from scheduler import identity, write_json
        from pathlib import Path
        report['process_identity'] = identity(os.getpid())
        write_json(Path(os.environ['ROBOCURVE_JOB_DIR'])/'device-verification.json', report)
    (out/'gpu-verification.json').write_text(json.dumps(report, indent=2))
    assert report['verified'], 'GPU binding verification failed: '+', '.join(key for key, passed in checks.items() if not passed)
    return report
