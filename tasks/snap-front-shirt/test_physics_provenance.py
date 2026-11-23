"""Unknown dynamics or runner edits must not reuse a prior stability gate."""

import copy
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from physics_provenance import PHYSICAL_SOURCES,verify_physical_sources


class ProvenanceChecks(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.root=Path(self.temporary.name);self.old=self.root/'old';self.new=self.root/'new'
        self.old.mkdir();self.new.mkdir()
        for name in PHYSICAL_SOURCES:
            content='{}' if name.endswith('.json') else '# unchanged physical source\n'
            if name=='robot_runner.py':
                content='def make_environment(cfg): return cfg\nclass Measurements: pass\ndef start_app(): return 0\n'
            (self.old/name).write_text(content);(self.new/name).write_text(content)
        (self.new/'gpu_runtime.py').write_text('# reviewed GPU helper\n')
        self.manifest={'sources':{n:self.hash(self.old/n) for n in PHYSICAL_SOURCES}}
    def tearDown(self):self.temporary.cleanup()
    def hash(self,p):return hashlib.sha256(p.read_bytes()).hexdigest()
    def bridge(self):
        return dict(reference_sha256=self.hash(self.old/'robot_runner.py'),
                    current_sha256=self.hash(self.new/'robot_runner.py'),
                    gpu_runtime_sha256=self.hash(self.new/'gpu_runtime.py'))
    def changed_runner(self):
        p=self.new/'robot_runner.py';p.write_text(p.read_text().replace('start_app(): return 0','start_app(): return 1'))

    def test_identical_physical_sources_pass(self):
        result=verify_physical_sources(self.new,self.old,self.manifest)
        self.assertTrue(all(r['identical'] for r in result['sources']))

    def test_exact_reviewed_startup_transition_passes(self):
        self.changed_runner()
        with patch('physics_provenance.RUNNER_BRIDGE',self.bridge()):
            result=verify_physical_sources(self.new,self.old,self.manifest)
        self.assertEqual(sum(bool(r['reviewed_transition']) for r in result['sources']),1)

    def test_only_exact_portable_helper_transition_is_accepted(self):
        self.changed_runner()
        bridge = self.bridge()
        helper = self.new / 'gpu_runtime.py'
        helper.write_text('# exact portable path helper\n')
        portable = dict(reference_sha256=bridge['gpu_runtime_sha256'], current_sha256=self.hash(helper))
        with patch('physics_provenance.RUNNER_BRIDGE', bridge), patch('physics_provenance.PORTABLE_HELPER_BRIDGE', portable):
            result = verify_physical_sources(self.new, self.old, self.manifest)
            record = next(r for r in result['sources'] if r['source'] == 'robot_runner.py')
            self.assertTrue(record['reviewed_transition']['portable_path_configuration'])
            helper.write_text('# unreviewed second change\n')
            with self.assertRaisesRegex(AssertionError, 'Unreviewed GPU'):
                verify_physical_sources(self.new, self.old, self.manifest)

    def test_unknown_runner_or_helper_is_rejected(self):
        self.changed_runner()
        with self.assertRaisesRegex(AssertionError,'Unreviewed runner'):
            verify_physical_sources(self.new,self.old,self.manifest)
        bridge=self.bridge();(self.new/'gpu_runtime.py').write_text('# changed helper')
        with patch('physics_provenance.RUNNER_BRIDGE',bridge),self.assertRaisesRegex(AssertionError,'Unreviewed GPU'):
            verify_physical_sources(self.new,self.old,self.manifest)

    def test_physics_builder_change_rejected_even_with_claimed_hash_bridge(self):
        p=self.new/'robot_runner.py';p.write_text(p.read_text().replace('return cfg','return 3'))
        with patch('physics_provenance.RUNNER_BRIDGE',self.bridge()),self.assertRaisesRegex(AssertionError,'Native physics'):
            verify_physical_sources(self.new,self.old,self.manifest)

    def test_parameter_change_or_modified_reference_is_rejected(self):
        p=self.new/'parameters.json';p.write_text('{"friction":0}')
        with self.assertRaisesRegex(AssertionError,'Physical source'):
            verify_physical_sources(self.new,self.old,self.manifest)
        p.write_text((self.old/'parameters.json').read_text());(self.old/'geometry.py').write_text('# tampered reference')
        with self.assertRaisesRegex(AssertionError,'Reference source'):
            verify_physical_sources(self.new,self.old,self.manifest)
