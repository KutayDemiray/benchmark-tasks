"""Check that scheduled admission and actual device evidence fail closed."""

from task_paths import OUTPUT_ROOT

import os
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import Mock, patch

import gpu_coordination as gpu
from runtime_checks import native_tensor, verification_checks


class GpuCoordinationTests(unittest.TestCase):
    def test_spoofed_scheduler_membership_never_uses_fallback_gate(self):
        helper = SimpleNamespace(assigned_gpu=Mock(side_effect=RuntimeError('not a member')))
        with patch.dict(os.environ, {'ROBOCURVE_JOB_ID':'fake', 'ROBOCURVE_TASK_ID':gpu.TASK_ID}, clear=True), \
                patch.object(gpu, 'scheduler_binding', return_value=helper), patch.object(gpu, 'snapshot') as snapshot, \
                patch.object(gpu.fcntl, 'flock') as lock:
            with self.assertRaisesRegex(RuntimeError, 'not a member'):
                with gpu.reservation(Path('/unused')):
                    self.fail('Invalid reservation admitted')
            snapshot.assert_not_called()
            lock.assert_not_called()

    def test_different_task_id_cannot_bypass_gate(self):
        with patch.dict(os.environ, {'ROBOCURVE_JOB_ID':'fake', 'ROBOCURVE_TASK_ID':'different-task'}, clear=True), \
                patch.object(gpu, 'scheduler_binding') as helper:
            with self.assertRaises(AssertionError):
                gpu.require_reservation()
            helper.assert_not_called()

    def test_verified_membership_does_not_flock_scheduler_descriptors(self):
        assignment = dict(index=0, uuid='GPU-a', pci_bus_id='test')
        env = dict(ROBOCURVE_JOB_ID='valid', ROBOCURVE_TASK_ID=gpu.TASK_ID,
            ROBOCURVE_CUDA_DEVICE='cuda:0', ROBOCURVE_GPU_MODE='dev')
        with patch.dict(os.environ, env, clear=True), \
                patch.object(gpu, 'scheduler_binding', return_value=SimpleNamespace(assigned_gpu=lambda: assignment)), \
                patch.object(gpu.fcntl, 'flock') as lock:
            self.assertEqual(gpu.require_reservation(), assignment)
            lock.assert_not_called()

    def test_fallback_rejects_an_uncoordinated_compute_job(self):
        state = dict(gpus='0, GPU-a, bus, 100, 0', compute='GPU-a, 42, python, 90', pmon='')
        with self.assertRaisesRegex(AssertionError, 'uncoordinated compute'):
            gpu.ensure_idle(state, dict(index=0, uuid='GPU-a'))

    def test_fallback_does_not_accept_a_replaced_gate(self):
        env = dict(ROBOCURVE_GPU_INDEX='1', POCKET_GPU_LOCK_FD='8')
        with patch.dict(os.environ, env, clear=True), \
                patch.object(os, 'fstat', return_value=SimpleNamespace(st_dev=1, st_ino=2)), \
                patch.object(Path, 'stat', return_value=SimpleNamespace(st_dev=1, st_ino=3)), \
                patch.object(gpu.fcntl, 'flock') as lock:
            with self.assertRaisesRegex(AssertionError, 'inode'):
                gpu.require_reservation()
            lock.assert_not_called()

    def test_unscheduled_reservation_holds_exclusive_gate_through_body(self):
        state = dict(gpus='1, GPU-a, bus, 24, 0', compute='', pmon='')
        with tempfile.TemporaryDirectory(dir=OUTPUT_ROOT) as directory:
            root = Path(directory)
            with patch.dict(os.environ, {}, clear=True), patch.object(gpu, 'GATE_DIRECTORY', root), \
                    patch.object(gpu, 'snapshot', return_value=state):
                with gpu.reservation(root) as (assignment, descriptors, legacy):
                    self.assertTrue(legacy)
                    self.assertEqual(assignment['uuid'], 'GPU-a')
                    self.assertEqual(len(descriptors), 1)
                    with (root/'.robocurve-gpu-1.lock').open('a+') as contender:
                        with self.assertRaises(BlockingIOError):
                            gpu.fcntl.flock(contender, gpu.fcntl.LOCK_EX | gpu.fcntl.LOCK_NB)
                with (root/'.robocurve-gpu-1.lock').open('a+') as contender:
                    gpu.fcntl.flock(contender, gpu.fcntl.LOCK_EX | gpu.fcntl.LOCK_NB)

    def test_readback_device_is_observed_not_the_asset_configuration(self):
        value = SimpleNamespace(device='cpu')
        asset = SimpleNamespace(device='cuda:0', root_view=SimpleNamespace(get_link_transforms=lambda: value))
        observed, method = native_tensor(asset)
        self.assertEqual(observed.device, 'cpu')
        self.assertEqual(method, 'get_link_transforms')

    def test_cross_device_compute_fails_even_with_correct_render_settings(self):
        report = dict(expected_index=1, expected_uuid='GPU-a', simulation_device='cuda:0',
            physics_tensor_device='cuda:0', cuda_uuid='a', active_renderer_physical_indices=[1],
            scene_gpu_dynamics={'/physicsScene': True},
            settings={'/physics/cudaDevice':0, '/renderer/activeGpu':1,
                '/renderer/multiGpu/enabled':False, '/renderer/multiGpu/activeCudaGpus':'0,'},
            allocations=[dict(uuid='GPU-a', type='C+G', memory_mib=1000), dict(uuid='GPU-b', type='G', memory_mib=8)])
        self.assertTrue(all(verification_checks(report, 'cuda:0').values()))
        report['allocations'][1]['type'] = 'C'
        self.assertFalse(verification_checks(report, 'cuda:0')['no_other_work'])
        report['physics_tensor_device'] = 'cpu'
        self.assertFalse(verification_checks(report, 'cuda:0')['physics_device'])
        self.assertTrue(verification_checks(report, 'cpu')['physics_device'])
        report['active_renderer_physical_indices'] = [0]
        self.assertFalse(verification_checks(report, 'cpu')['actual_renderer'])


if __name__ == '__main__':
    unittest.main()
