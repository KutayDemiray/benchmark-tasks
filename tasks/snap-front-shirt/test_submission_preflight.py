"""Reject misleading completion, configuration and media claims offline."""

import copy
from pathlib import Path
import unittest

import numpy as np
from unittest.mock import patch

from submission_preflight import (PendingEvidence, Review, audit, full_demonstration,
                                  media_binding, public_metadata, replay_action_match)


class SubmissionPreflightChecks(unittest.TestCase):
    def setUp(self):
        self.configuration = dict(iterations=64,pitch=.005)
        self.result = dict(stage='tabletop_task',full_task_success=True,task_score=1.,
            retained_pairs=[0,1,2],invalid_reasons=[],original_hanger_task_success=False,
            control_steps=1000,physics_steps_observed=48001)
        self.setup = dict(case='fasten_three',action_dimension=14,control_hz=20,
            recorded_cameras=True,seed=100,parameters=dict(
                completion=dict(control_hz=20,run_limit_seconds=300),
                tabletop=dict(required_snap_indices=[0,1,2]),
                fabric=dict(solver_iterations=64,contact_offset_m=.001),
                garment=dict(body_mesh_pitch_m=.005),simulation=dict(physics_hz=960)))
        self.worker = Path('/selected/baseline/worker-00')
        self.native = dict(bundle_sha256='selected-bundle')
        self.media = dict(source_run=str(self.worker),source_episode='episode-00',
            source_bundle_sha256='selected-bundle',source_seed=100,source_score=1.,
            source_controls=1000,model_attempt=False,temporal_edits=False,
            playback_speed=1.,images_synthesized=False,
            source_cameras={name:dict(decoded_frames=1000,fps=20,decode_pass=True)
                            for name in ('top_cam','left_cam','right_cam')},
            demo=dict(frames=1000,fps=20,decoded=True,codec='h264',bytes=2000000))

    def test_valid_selected_profile_can_establish_full_local_demo(self):
        evidence = full_demonstration(self.result,self.setup,self.configuration)
        self.assertEqual(evidence['score'],1.)
        self.assertEqual(evidence['seconds'],50.)

    def test_one_snap_or_invalid_physics_never_becomes_full_task(self):
        for change in (dict(task_score=1/3,retained_pairs=[0]),
                       dict(invalid_reasons=['cloth_strain_limit']),
                       dict(full_task_success=False),dict(original_hanger_task_success=True),
                       dict(physics_steps_observed=48000),dict(control_steps=6001)):
            with self.subTest(change=change), self.assertRaises(AssertionError):
                full_demonstration(dict(self.result,**change),self.setup,self.configuration)

    def test_a_success_in_an_old_mesh_or_control_profile_does_not_qualify(self):
        for key,value in (('pitch',.01),('iterations',128)):
            with self.assertRaises(AssertionError):
                full_demonstration(self.result,self.setup,dict(self.configuration,**{key:value}))
        old = copy.deepcopy(self.setup)
        old['control_hz'] = old['parameters']['completion']['control_hz'] = 10
        with self.assertRaises(AssertionError):
            full_demonstration(self.result,old,self.configuration)

    def test_media_requires_the_selected_run_source_and_all_original_cameras(self):
        media_binding(self.media,self.worker,self.native,self.result,self.setup)
        for change in (dict(source_run='/historical/success'),dict(source_bundle_sha256='older-code'),
                       dict(source_seed=101),dict(source_controls=999),dict(temporal_edits=True),
                       dict(playback_speed=2.),dict(model_attempt=True)):
            with self.subTest(change=change), self.assertRaises(AssertionError):
                media_binding(dict(self.media,**change),self.worker,self.native,self.result,self.setup)
        edited = copy.deepcopy(self.media)
        edited['source_cameras'].pop('left_cam')
        with self.assertRaises(AssertionError):
            media_binding(edited,self.worker,self.native,self.result,self.setup)

    def test_missing_wrist_frame_and_wrong_playback_rate_fail(self):
        for camera_change in (dict(decoded_frames=999),dict(fps=10),dict(decode_pass=False)):
            media = copy.deepcopy(self.media)
            media['source_cameras']['right_cam'].update(camera_change)
            with self.assertRaises(AssertionError):
                media_binding(media,self.worker,self.native,self.result,self.setup)

    def test_pending_and_failed_evidence_are_distinct(self):
        review = Review()
        def pending():
            raise PendingEvidence('still running')
        def failure():
            assert False, 'invalid native outcome'
        review.check('incomplete',pending)
        review.check('invalid',failure)
        self.assertEqual([item['status'] for item in review.checks],['pending','failed'])

    def test_public_text_excludes_obvious_local_identifiers_but_no_media_is_a_draft(self):
        task = {name:'Specified text' for name in ('title','instruction','setup','scoring','run_limit','fidelity','scores')}
        task.update(category='Societal relatability',media=None)
        self.assertFalse(public_metadata(task)['media_present'])
        with self.assertRaises(AssertionError):
            public_metadata(dict(task,setup='Read /home/example/scene.usd'))

    def test_unfinished_queue_is_not_rescored_or_promoted_by_a_progress_flag(self):
        queue = Path('/queue')
        task = {name:'Specified text' for name in ('title','instruction','setup','scoring','run_limit','fidelity','scores')}
        task.update(category='Societal relatability',media=None)
        def fake_read(path):
            if path.name == 'task.json':
                return task
            if path == queue/'progress.json':
                return dict(phase='waiting_for_stability',local_validation_complete=True)
            raise PendingEvidence(str(path))
        with patch('submission_preflight.read',side_effect=fake_read), \
             patch('review_stability_round.review') as stability_review, \
             patch('review_stability_round.verified_worker') as native_review:
            report = audit(queue,Path('/stability'))
        stability_review.assert_not_called()
        native_review.assert_not_called()
        self.assertFalse(report['local_validation_ready'])
        self.assertFalse(report['submission_ready'])
        self.assertEqual(len(report['external_requirements']),3)
        self.assertFalse(report['bimanual_status']['required_by_reviewed_public_criteria'])
        self.assertIsNone(report['statistics']['required_success_rate'])

    def test_native_replay_may_stop_early_but_cannot_repair_or_append_actions(self):
        source = np.arange(140,dtype=float).reshape(10,14)
        self.assertTrue(replay_action_match(source,source[:7])['exact_executed_prefix'])
        altered = source.copy()
        altered[4,3] += .001
        for actions in (altered,np.concatenate([source,source[-1:]]),source[:0]):
            with self.assertRaises(AssertionError):
                replay_action_match(source,actions)
