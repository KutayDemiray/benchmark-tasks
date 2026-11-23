"""Task-owned integration with verified cooperative scheduling and legacy locks."""

import hashlib
import importlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

from task_paths import GPU_LOCK, SCHEDULER_ROOT

TASK_ID = 'snap-front-shirt'


def scheduler_binding():
    """Load the canonical scheduler helper without copying or modifying it."""
    location = str(SCHEDULER_ROOT)
    if location not in sys.path:
        sys.path.insert(0, location)
    module = importlib.import_module('isaac_binding')
    assert Path(module.__file__).resolve() == SCHEDULER_ROOT / 'isaac_binding.py'
    return module


def scheduled_assignment():
    """Reject invalid membership instead of falling back to an unlocked launch."""
    if 'ROBOCURVE_JOB_ID' not in os.environ:
        return None
    assignment = scheduler_binding().assigned_gpu()
    assert os.environ['ROBOCURVE_TASK_ID'] == TASK_ID, 'Use the stable snap-front-shirt task ID'
    assert os.environ['ROBOCURVE_CUDA_DEVICE'] == 'cuda:0'
    assert os.environ['CUDA_DEVICE_ORDER'] == 'PCI_BUS_ID'
    return assignment


def inherited_fds():
    """Validate the reservation and preserve its descriptors in every child."""
    if scheduled_assignment() is not None:
        descriptors = tuple(int(value) for value in os.environ['ROBOCURVE_LOCK_FDS'].split(','))
        assert descriptors and len(set(descriptors)) == len(descriptors)
        for descriptor in descriptors:
            assert descriptor >= 3
            os.fstat(descriptor)
        # The scheduler owns these open-file descriptions; never flock/unlock them.
        return descriptors
    descriptor = int(os.environ['SNAP_LOCK_FD'])
    actual, expected = os.fstat(descriptor), GPU_LOCK.stat()
    assert (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino), 'Wrong legacy GPU lock'
    information = Path(f'/proc/self/fdinfo/{descriptor}').read_text()
    assert 'FLOCK' in information and 'WRITE' in information, 'Legacy launch requires an exclusive lock'
    assert os.environ['SNAP_GPU_UUID'], 'Missing legacy physical GPU identity'
    return (descriptor,)


def assignment():
    """Return the verified scheduler assignment or the explicit legacy GPU 0."""
    scheduled = scheduled_assignment()
    if scheduled is not None:
        return scheduled
    assert int(os.environ.get('SNAP_GPU_INDEX', '0')) == 0, 'Unscheduled fallback reserves GPU 0 only'
    uuid = os.environ['SNAP_GPU_UUID']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == uuid
    return dict(index=0, uuid=uuid)


def scheduler_record():
    """Record the externally owned protocol sources and validated reservation."""
    gpu = scheduled_assignment()
    if gpu is None:
        return None
    sources = {}
    for name in ('isaac_binding.py', 'scheduler.py'):
        sources[name] = hashlib.sha256((SCHEDULER_ROOT / name).read_bytes()).hexdigest()
    return dict(job_id=os.environ['ROBOCURVE_JOB_ID'], task_id=TASK_ID,
                mode=os.environ['ROBOCURVE_GPU_MODE'], gpu=gpu,
                job_directory=os.environ['ROBOCURVE_JOB_DIR'], helper_source_sha256=sources)


def kit_arguments(log_path):
    """Bind masked physics and the assigned physical renderer without dynamics changes."""
    inherited_fds()
    if scheduled_assignment() is not None:
        return scheduler_binding().kit_args(log_path=log_path)
    physical = assignment()['index']
    return ' '.join((f'--/renderer/activeGpu={physical}',
                     '--/renderer/multiGpu/activeCudaGpus=0,',
                     '--/renderer/multiGpu/enabled=false', '--/physics/cudaDevice=0'))


def _legacy_verification(sim, tensor_device, log_path):
    import carb
    import torch
    expected = assignment()
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
    active = sorted(set(int(value) for value in re.findall(
        r'^\|\s*(\d+)\s*\|[^\n]*Yes:', Path(log_path).read_text(errors='replace'), re.M)))
    observed_uuid = str(torch.cuda.get_device_properties(0).uuid)
    report = dict(pid=os.getpid(), job_id=None, expected_index=expected['index'], expected_uuid=expected['uuid'],
                  settings={key: settings.get(key) for key in keys}, simulation_device=str(sim.device),
                  physics_tensor_device=str(tensor_device), cuda_uuid=observed_uuid,
                  active_renderer_physical_indices=active, allocations=allocations)
    report['checks'] = dict(
        physics_setting=report['settings']['/physics/cudaDevice'] == 0,
        render_setting=report['settings']['/renderer/activeGpu'] == expected['index'],
        single_renderer=report['settings']['/renderer/multiGpu/enabled'] is False,
        physics_device=str(sim.device) == str(tensor_device) == 'cuda:0',
        cuda_uuid=observed_uuid.removeprefix('GPU-').lower() == expected['uuid'].removeprefix('GPU-').lower(),
        actual_renderer=active == [expected['index']],
        compute_observed=any(p['uuid'] == expected['uuid'] and 'C' in p['type'] for p in allocations),
        no_other_work=all(p['uuid'] == expected['uuid'] for p in allocations))
    report['verified'] = all(report['checks'].values())
    return report


def verify_native(sim, tensor_device, destination, log_path, recorded=False):
    """Verify actual backend tensors, CUDA UUID, renderer log and process allocations."""
    inherited_fds()
    scheduled = scheduled_assignment()
    if scheduled is not None:
        binding = scheduler_binding()
        try:
            evidence = binding.verify(sim, tensor_device, log_path=log_path)
        except Exception:
            source = Path(os.environ['ROBOCURVE_JOB_DIR']) / 'device-verification.json'
            if source.exists():
                failed = json.loads(source.read_text())
                if failed.get('pid') == os.getpid():
                    (destination / 'device-verification.json').write_text(json.dumps(failed, indent=2) + '\n')
            raise
    else:
        evidence = _legacy_verification(sim, tensor_device, log_path)
    (destination / 'device-verification.json').write_text(json.dumps(evidence, indent=2) + '\n')
    assert evidence['verified'], 'Native physics/render GPU verification failed'
    active = [p for p in evidence['allocations'] if p['uuid'] == evidence['expected_uuid']]
    assert not recorded or any('G' in p['type'] for p in active), 'Assigned rendering allocation missing'
    other = [p for p in evidence['allocations'] if p['uuid'] != evidence['expected_uuid']]
    result = dict(physical_gpu_uuid=evidence['expected_uuid'], renderer_index=evidence['expected_index'],
                  physics_device=evidence['physics_tensor_device'], binding_verification=evidence,
                  graphics_and_compute_process_entries=[dict(physical_gpu=evidence['expected_index'], **p) for p in active],
                  inactive_device_enumeration=other, scheduler=scheduler_record(),
                  allocation_scope='Assigned active-device entries; inactive enumeration is recorded separately.')
    (destination / 'gpu.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def verify_environment(base, metadata, destination, recorded):
    """Read a real native rigid transform tensor after the environment reset."""
    import omni.physics.tensors as tensors
    view = tensors.create_simulation_view('torch')
    view.set_subspace_roots('/')
    bodies = view.create_rigid_body_view(next(iter(metadata['hardware'].values())))
    assert bodies.count == 1
    transforms = bodies.get_transforms()
    return verify_native(base.sim, transforms.device, destination, destination.parent / 'kit.log', recorded)


def audit_record(gpu, preflight, recorded=False):
    """Audit both historical GPU-0 evidence and new assigned-device records on CPU."""
    expected_index = preflight.get('physical_gpu_index', 0)
    expected_uuid = preflight.get('gpu', preflight.get('gpu0', '')).split(',')[0].strip()
    assert gpu['physics_device'] == 'cuda:0' and gpu['renderer_index'] == expected_index
    assert gpu['physical_gpu_uuid'] == expected_uuid
    entries = gpu['graphics_and_compute_process_entries']
    assert entries and all(e['physical_gpu'] == expected_index for e in entries)
    assert not recorded or any('G' in e['type'] for e in entries)
    evidence = gpu.get('binding_verification')
    if preflight.get('scheduler') is not None:
        assert evidence is not None and evidence['job_id'] == preflight['scheduler']['job_id']
        assert gpu['scheduler']['task_id'] == preflight['scheduler']['task_id'] == TASK_ID
    if evidence is not None:
        required = {'physics_setting', 'render_setting', 'single_renderer', 'physics_device',
                    'cuda_uuid', 'actual_renderer', 'compute_observed', 'no_other_work'}
        assert required.issubset(evidence['checks'])
        assert evidence['verified'] and all(evidence['checks'].values())
        assert evidence['settings']['/physics/cudaDevice'] == 0
        assert evidence['settings']['/renderer/activeGpu'] == expected_index
        assert evidence['settings']['/renderer/multiGpu/enabled'] is False
        assert evidence['settings']['/renderer/multiGpu/activeCudaGpus'] == '0,'
        assert evidence['expected_uuid'] == expected_uuid and evidence['expected_index'] == expected_index
        assert evidence['active_renderer_physical_indices'] == [expected_index]
        assert evidence['simulation_device'] == evidence['physics_tensor_device'] == 'cuda:0'
        assert evidence['cuda_uuid'].removeprefix('GPU-').lower() == expected_uuid.removeprefix('GPU-').lower()
        assert any(p['uuid'] == expected_uuid and 'C' in p['type'] for p in evidence['allocations'])
        assert all(p['uuid'] == expected_uuid or (p['type'] == 'G' and p['memory_mib'] <= 16)
                   for p in evidence['allocations'])
