"""CPU checks for reservation validation, full-tree inheritance and device audits."""

import copy
import fcntl
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import gpu_runtime as runtime
import launch


SCHEDULED = dict(ROBOCURVE_JOB_ID='job', ROBOCURVE_TASK_ID='snap-front-shirt',
                 ROBOCURVE_CUDA_DEVICE='cuda:0', CUDA_DEVICE_ORDER='PCI_BUS_ID',
                 ROBOCURVE_LOCK_FDS='41,42,43', ROBOCURVE_GPU_MODE='dev')


def device_record():
    checks = {key: True for key in ('physics_setting', 'render_setting', 'single_renderer',
              'physics_device', 'cuda_uuid', 'actual_renderer', 'compute_observed', 'no_other_work')}
    evidence = dict(verified=True, job_id='job', expected_index=1, expected_uuid='GPU-b',
                    simulation_device='cuda:0', physics_tensor_device='cuda:0', cuda_uuid='b',
                    active_renderer_physical_indices=[1], checks=checks,
                    settings={'/physics/cudaDevice': 0, '/renderer/activeGpu': 1,
                              '/renderer/multiGpu/enabled': False, '/renderer/multiGpu/activeCudaGpus': '0,'},
                    allocations=[dict(uuid='GPU-b', type='C+G', memory_mib=5000),
                                 dict(uuid='GPU-a', type='G', memory_mib=8)])
    job = dict(job_id='job', task_id='snap-front-shirt')
    gpu = dict(physics_device='cuda:0', renderer_index=1, physical_gpu_uuid='GPU-b', scheduler=job,
               graphics_and_compute_process_entries=[dict(physical_gpu=1, type='C+G')],
               binding_verification=evidence)
    return gpu, dict(physical_gpu_index=1, gpu='GPU-b, PCI, GPU', scheduler=job)


class BindingTests(unittest.TestCase):
    def test_legacy_does_not_need_scheduler_module(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(runtime, 'scheduler_binding') as helper:
            self.assertIsNone(runtime.scheduled_assignment())
            helper.assert_not_called()

    def test_membership_must_validate_before_bypass(self):
        helper = Mock()
        helper.assigned_gpu.side_effect = RuntimeError('not a member')
        with patch.dict(os.environ, SCHEDULED, clear=True), patch.object(runtime, 'scheduler_binding', return_value=helper):
            with self.assertRaisesRegex(RuntimeError, 'not a member'):
                runtime.inherited_fds()
            helper.assigned_gpu.assert_called_once()

    def test_wrong_stable_task_id_rejected(self):
        with patch.dict(os.environ, dict(SCHEDULED, ROBOCURVE_TASK_ID='snap-front-shirt-run-2'), clear=True), \
             patch.object(runtime, 'scheduler_binding', return_value=Mock()):
            with self.assertRaisesRegex(AssertionError, 'stable'):
                runtime.scheduled_assignment()

    def test_all_reservation_descriptors_validated_and_inherited(self):
        with patch.dict(os.environ, SCHEDULED, clear=True), \
             patch.object(runtime, 'scheduler_binding', return_value=Mock()), patch.object(os, 'fstat') as stat:
            self.assertEqual(runtime.inherited_fds(), (41, 42, 43))
            self.assertEqual([call.args[0] for call in stat.call_args_list], [41, 42, 43])

    def test_invalid_descriptor_fails_closed(self):
        for values in ('41,41', '0,41', ''):
            with self.subTest(values=values), patch.dict(os.environ, dict(SCHEDULED, ROBOCURVE_LOCK_FDS=values), clear=True), \
                 patch.object(runtime, 'scheduler_binding', return_value=Mock()), patch.object(os, 'fstat'):
                with self.assertRaises((AssertionError, ValueError)):
                    runtime.inherited_fds()

    def test_legacy_requires_actual_exclusive_lock_and_matching_inode(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = Path(directory)/'gate'
            wrong = Path(directory)/'wrong'; wrong.touch()
            with lock.open('a+') as stream, patch.object(runtime, 'GPU_LOCK', lock), \
                 patch.dict(os.environ, dict(SNAP_LOCK_FD=str(stream.fileno()), SNAP_GPU_UUID='GPU-a'), clear=True):
                with self.assertRaisesRegex(AssertionError, 'exclusive'):
                    runtime.inherited_fds()
                fcntl.flock(stream, fcntl.LOCK_SH)
                with self.assertRaisesRegex(AssertionError, 'exclusive'):
                    runtime.inherited_fds()
                fcntl.flock(stream, fcntl.LOCK_EX)
                self.assertEqual(runtime.inherited_fds(), (stream.fileno(),))
                with patch.object(runtime, 'GPU_LOCK', wrong):
                    with self.assertRaisesRegex(AssertionError, 'Wrong legacy'):
                        runtime.inherited_fds()

    def test_kit_uses_canonical_scheduler_arguments(self):
        helper = Mock()
        helper.kit_args.return_value = '--/renderer/activeGpu=1 --/physics/cudaDevice=0'
        with patch.object(runtime, 'inherited_fds'), patch.object(runtime, 'scheduled_assignment', return_value={'index': 1}), \
             patch.object(runtime, 'scheduler_binding', return_value=helper):
            self.assertIn('activeGpu=1', runtime.kit_arguments(Path('/owned/kit.log')))
            helper.kit_args.assert_called_once_with(log_path=Path('/owned/kit.log'))

    def test_scheduled_launcher_skips_legacy_gate_and_passes_all_descriptors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); sources = root/'sources'; sources.mkdir()
            (sources/'bench.py').write_text('# test launcher child\n')
            out = root/'run'
            def child(*args, **kwargs):
                (out/'results.json').write_text('{}')
                return Mock(pid=123, wait=Mock(return_value=0))
            with patch.object(launch, 'OUTPUT', root), patch.object(launch, 'HERE', sources), \
                 patch.object(launch, 'LOCK') as gate, patch.object(launch.fcntl, 'flock') as flock, \
                 patch.object(launch, 'scheduled_assignment', return_value={'index': 1, 'uuid': 'GPU-b'}), \
                 patch.object(launch, 'inherited_fds', return_value=(41, 42, 43)), \
                 patch.object(launch, 'scheduler_record', return_value={'job_id': 'job'}), \
                 patch.object(launch, 'query', side_effect=['GPU-b, PCI, GPU, 6000 MiB, 70 %', 'peer on GPU-b', 'table']), \
                 patch.object(launch.subprocess, 'check_output', return_value='processes'), \
                 patch.object(launch.subprocess, 'Popen', side_effect=child) as popen, \
                 patch.object(launch.ctypes, 'CDLL', return_value=Mock(prctl=Mock(return_value=0))), \
                 patch.object(launch.os, 'waitpid', side_effect=ChildProcessError), \
                 patch('sys.argv', ['launch.py', '--out', str(out)]):
                self.assertEqual(launch.main(), 0)
                gate.open.assert_not_called(); flock.assert_not_called()
                self.assertEqual(popen.call_args.kwargs['pass_fds'], (41, 42, 43))
                self.assertEqual(popen.call_args.kwargs['env']['CUDA_VISIBLE_DEVICES'], 'GPU-b')
                self.assertNotIn('SNAP_LOCK_FD', popen.call_args.kwargs['env'])

    def test_invalid_membership_fails_before_lock_or_process(self):
        with patch('sys.argv', ['launch.py', '--out', str(launch.OUTPUT/'invalid-test')]), \
             patch.object(launch, 'scheduled_assignment', side_effect=RuntimeError('invalid reservation')), \
             patch.object(launch, 'LOCK') as gate, patch.object(launch.subprocess, 'Popen') as popen:
            with self.assertRaisesRegex(RuntimeError, 'invalid reservation'):
                launch.main()
            gate.open.assert_not_called(); popen.assert_not_called()

    def test_busy_legacy_gate_starts_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); lock = root/'gate'
            with lock.open('a+') as holder:
                fcntl.flock(holder, fcntl.LOCK_EX)
                with patch.object(launch, 'OUTPUT', root), patch.object(launch, 'LOCK', lock), \
                     patch.object(launch, 'scheduled_assignment', return_value=None), \
                     patch('sys.argv', ['launch.py', '--out', str(root/'run')]), \
                     patch.object(launch.subprocess, 'Popen') as popen:
                    self.assertEqual(launch.main(), 75)
                    popen.assert_not_called()
                    self.assertFalse(json.loads((root/'run/exit.json').read_text())['simulation_started'])


class EvidenceTests(unittest.TestCase):
    def test_gpu1_with_small_inactive_enumeration_is_valid(self):
        gpu, preflight = device_record()
        runtime.audit_record(gpu, preflight, recorded=True)

    def test_historical_gpu0_record_is_still_auditable(self):
        gpu = dict(physics_device='cuda:0', renderer_index=0, physical_gpu_uuid='GPU-a',
                   graphics_and_compute_process_entries=[dict(physical_gpu=0, type='C+G')])
        runtime.audit_record(gpu, {'gpu0': 'GPU-a, PCI, GPU'}, recorded=True)

    def test_wrong_gpu_compute_or_missing_checks_are_rejected(self):
        gpu, preflight = device_record()
        bad_records = []
        def changed():
            result = copy.deepcopy(gpu); bad_records.append(result); return result
        changed()['renderer_index'] = 0
        changed()['binding_verification']['physics_tensor_device'] = 'cpu'
        changed()['binding_verification']['checks'] = {}
        changed()['binding_verification']['job_id'] = 'different-job'
        changed()['binding_verification']['allocations'][1]['type'] = 'C'
        changed()['binding_verification']['allocations'][1]['memory_mib'] = 17
        changed()['binding_verification']['active_renderer_physical_indices'] = [0, 1]
        changed()['binding_verification']['settings']['/renderer/multiGpu/enabled'] = True
        for record in bad_records:
            with self.subTest(record=record), self.assertRaises(AssertionError):
                runtime.audit_record(record, preflight, recorded=True)


if __name__ == '__main__':
    unittest.main()
