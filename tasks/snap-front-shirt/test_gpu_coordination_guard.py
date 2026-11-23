"""Keep pause/resume signalling confined to unchanged owned process identities."""

import os
import unittest
from unittest.mock import patch
from gpu_coordination_guard import descendants,graphics_entries,same_process,signal_owned,scheduler_enumeration,foreign_processes


class CoordinationChecks(unittest.TestCase):
    def test_unrelated_process_trees_are_excluded(self):
        self.assertEqual(descendants({2:1,3:2,4:90,5:3,6:4},1),[2,3,5])

    def test_changed_pid_identity_or_owner_cannot_be_signalled(self):
        original=dict(pid=123,uid=os.getuid(),start_ticks=20,command=[b'owned'])
        for actual in (None,dict(original,start_ticks=21),dict(original,uid=os.getuid()+1),dict(original,command=[b'other'])):
            with patch('gpu_coordination_guard.process_record',return_value=actual),patch('os.kill') as kill:
                self.assertFalse(signal_owned(original,19));kill.assert_not_called()

    def test_both_compute_and_graphics_rows_are_recognized(self):
        rows='| 0 N/A N/A 123 C+G /owned 4000MiB |\n| 1 N/A N/A 456 C python 5000MiB |\n| 0 N/A N/A 12 G Xorg 50MiB |'
        self.assertEqual(graphics_entries(rows),[(0,123,'C+G'),(1,456,'C'),(0,12,'G')])

    def test_valid_signal_uses_a_kernel_process_handle(self):
        original=dict(pid=123,uid=os.getuid(),start_ticks=20,command=[b'owned'])
        with patch('gpu_coordination_guard.process_record',return_value=original), \
             patch('os.pidfd_open',return_value=42),patch('os.close') as close, \
             patch('signal.pidfd_send_signal') as send,patch('os.kill') as kill:
            self.assertTrue(signal_owned(original,19))
            send.assert_called_once_with(42,19,None,0)
            close.assert_called_once_with(42);kill.assert_not_called()


class EnumerationChecks(unittest.TestCase):
    def entry(self, **changes):
        return dict(dict(pid=9,type='G',memory_mib=8,classification='registered_graphics_enumeration',job_id='verified'),**changes)

    def accepted(self, entry):
        return scheduler_enumeration('GPU-a',{'gpus':[{'uuid':'GPU-a','processes':[entry]}]})

    def test_verified_small_inactive_graphics_does_not_pause_legacy_compute(self):
        accepted=self.accepted(self.entry())
        self.assertEqual(foreign_processes({1},[(1,'C'),(9,'G')],set(),{1},accepted),[])

    def test_unidentified_unverified_large_or_compute_context_is_blocked(self):
        for entry in (self.entry(classification='unidentified'),self.entry(job_id=None),
                      self.entry(memory_mib=17),self.entry(type='C+G'),self.entry(memory_mib=-1)):
            with self.subTest(entry=entry):
                self.assertEqual(self.accepted(entry),[])
                self.assertEqual(foreign_processes({1},[(1,'C'),(9,'G')],set(),{1},[]),[9])

    def test_new_compute_seen_between_snapshots_is_never_exempted(self):
        accepted=self.accepted(self.entry())
        self.assertEqual(foreign_processes({1,9},[(1,'C'),(9,'G')],set(),{1},accepted),[9])
        self.assertEqual(foreign_processes({1},[(1,'C'),(9,'C+G')],set(),{1},accepted),[9])

    def test_missing_inventory_fails_closed(self):
        with self.assertRaises(AssertionError):scheduler_enumeration('GPU-a',{'gpus':[]})
