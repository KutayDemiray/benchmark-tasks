"""Check recorded model provenance without assuming that the model succeeds."""

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from inspect_robots_dream import _station
from policy_contract import CAMERAS, INSTRUCTION

from dream_readiness import PHYSICAL_SOURCES, SHARED_RUNTIME, audit_model_run


class DreamEvidenceChecks(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run=Path(self.directory.name)/'run'
        self.worker=self.run/'worker-00'
        self.demo=Path(self.directory.name)/'demo'
        (self.worker/'episode-00').mkdir(parents=True)
        (self.demo/'episode-00').mkdir(parents=True)
        self.result=dict(task_score=0.,control_steps=36,termination_reason='time_limit',invalid_reasons=[])
        self.evidence=dict(mode='dream_live',model='dreamlabs/dream-1-beta',model_attempt=True,
            hosted_requests_started=1,hosted_requests_completed=1,episode_retries=0,model_fallback=False,
            controller='none',scripted_assistance=False,control_hz=20,declared_action_budget=6000,
            result=self.result,error=None,source_bundle_sha256='recorded-bundle')
        self.write(self.run/'exit.json',dict(returncode=0))
        self.write(self.run/'model-evaluation.json',self.evidence)
        self.write(self.worker/'episode-00/result.json',self.result)
        setup=dict(seed=100,control_hz=20,controller='none',parameters={
            key:{} for key in ('fabric','snap','simulation','tabletop','garment','completion')})
        self.write(self.worker/'episode-00/setup.json',setup)
        setup=dict(setup,controller='baseline')
        self.write(self.demo/'episode-00/setup.json',setup)
        sources={name:'preserved-'+name for name in (*SHARED_RUNTIME,*PHYSICAL_SOURCES)}
        self.write(self.worker/'manifest.json',dict(sources=sources))
        self.write(self.demo/'manifest.json',dict(sources=sources))
        self.write(self.run/'policy-transcript.json',dict(preroll_steps=32,rows_per_request=48,
            dry_run=False,model=self.evidence['model']))
        self.write(self.run/'initial-policy-state.json',dict(joint_pos=[0.]*14,instruction=INSTRUCTION,policy_context_forwarded=False))
        self.write(self.run/'first-forwarded-policy-state.json',dict(joint_pos=[0.]*14,instruction=INSTRUCTION,step=1,time_s=.05))
        self.frames={}
        for index,name in enumerate(CAMERAS):
            pixels=np.zeros((176,320,3),dtype=np.uint8);pixels[:,:,index]=180
            Image.fromarray(pixels).save(self.run/('initial-'+name+'.png'))
            Image.fromarray(pixels).save(self.run/('first-forwarded-'+name+'.png'))
            self.frames[name]=_station.jpeg(pixels)
        self.events=[dict(request=0,event='started'),dict(request=0,event='completed',
            chunk=dict(attempts=1,rows=[[.02]*7+[-.02]*6+[.2] for _ in range(48)]))]
        self.native=[];self.policy=[]
        for step in range(36):
            action=[0.]*14 if step<32 else self.events[1]['chunk']['rows'][step-32]
            measured=[step/1000]*14
            self.native.append(dict(joint_positions=measured,requested_action=action))
            self.policy.append(dict(step=step,input_time_s=step/20,output_time_s=(step+1)/20,
                requested_action=action,input_joint_pos=[0.]*14 if step==0 else self.native[step-1]['joint_positions'],
                native_physics_steps_during_inference=0))
        self.records()
        # Raw physics/video validation has its own independent verifier tests.
        # This fixture isolates policy provenance after that verifier succeeds.
        mock=patch('dream_readiness.verified_worker',return_value=dict(verification_pass=True,
            episodes=[dict(task_score=0.)]))
        mock.start();self.addCleanup(mock.stop)

    def write(self,path,value):
        path.write_text(json.dumps(value)+'\n')

    def records(self):
        for path,rows in ((self.run/'inference.jsonl',self.events),
                          (self.run/'policy-steps.jsonl',self.policy),
                          (self.worker/'episode-00/trace.jsonl',self.native)):
            path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        inputs=[]
        for index in range(1,len(self.policy)+(1 if self.evidence['error'] else 0)):
            measured=[0.]*14 if index==0 else self.native[index-1]['joint_positions']
            inputs.append(dict(index=index,time_s=index/20,instruction=INSTRUCTION,
                joint_pos=measured,camera_jpeg_base64=self.frames))
        (self.run/'inference-observations.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in inputs))
        startup=dict(discarded_observations=1,discarded_time_s=0.,first_forwarded_time_s=.05,
            forwarded_observation_calls=len(inputs),additional_actions=0,reason='Confirmed stale reset camera frame')
        self.write(self.run/'startup-camera-filter.json',startup)
        self.evidence['startup_camera_filter']=startup
        self.write(self.run/'model-evaluation.json',self.evidence)

    def test_a_genuine_recorded_zero_is_a_valid_model_outcome(self):
        result=audit_model_run(self.run,self.demo)
        self.assertEqual(result['score'],0.)
        self.assertFalse(result['model_success_required'])
        self.assertEqual(result['actions'],36)

    def test_fake_mode_script_assistance_and_second_attempt_are_rejected(self):
        for change in (dict(mode='fake'),dict(controller='baseline'),dict(scripted_assistance=True),
                       dict(model_attempt=False),dict(episode_retries=1),dict(model_fallback=True)):
            with self.subTest(change=change):
                self.write(self.run/'model-evaluation.json',dict(self.evidence,**change))
                with self.assertRaises(AssertionError):audit_model_run(self.run,self.demo)

    def test_edited_targets_fail_even_when_both_policy_and_native_log_are_changed(self):
        self.policy[34]['requested_action']=[.8]*14
        self.native[34]['requested_action']=[.8]*14
        self.records()
        with self.assertRaisesRegex(AssertionError,'model row'):audit_model_run(self.run,self.demo)

    def test_fabric_config_and_measured_state_cannot_be_substituted(self):
        path=self.worker/'episode-00/setup.json'
        original=json.loads(path.read_text())
        changed=copy.deepcopy(original);changed['parameters']['fabric']['stiffness']=999
        self.write(path,changed)
        with self.assertRaisesRegex(AssertionError,'configuration differs'):audit_model_run(self.run,self.demo)
        self.write(path,original)
        self.policy[35]['input_joint_pos']=[.7]*14;self.records()
        with self.assertRaisesRegex(AssertionError,'measured native state'):audit_model_run(self.run,self.demo)

    def test_refusal_after_preroll_preserves_error_and_zero_without_extra_steps(self):
        self.result.update(control_steps=32,termination_reason='model_error')
        self.evidence.update(hosted_requests_completed=0,error=dict(type='RuntimeError',message='service refusal'))
        self.events=[dict(request=0,event='started'),dict(request=0,event='error',error='service refusal')]
        self.native=self.native[:32];self.policy=self.policy[:32];self.records()
        self.write(self.run/'model-evaluation.json',self.evidence)
        self.write(self.worker/'episode-00/result.json',self.result)
        result=audit_model_run(self.run,self.demo)
        self.assertEqual(result['completed_requests'],0)
        self.assertEqual(result['actions'],32)
        self.assertEqual(result['score'],0.)
        self.assertEqual(result['termination_reason'],'model_error')

    def test_swapped_initial_camera_and_private_input_fields_are_rejected(self):
        path=self.run/'inference-observations.jsonl'
        original=path.read_text();rows=[json.loads(line) for line in original.splitlines()]
        rows[0]['camera_jpeg_base64']['top_cam']=self.frames['right_cam']
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        with self.assertRaisesRegex(AssertionError,'First forwarded camera'):audit_model_run(self.run,self.demo)
        rows=[json.loads(line) for line in original.splitlines()]
        rows[4]['contact_force']=1.2
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        with self.assertRaises(AssertionError):audit_model_run(self.run,self.demo)

    def test_logged_proprioception_cannot_be_replaced_with_commanded_targets(self):
        path=self.run/'inference-observations.jsonl'
        rows=[json.loads(line) for line in path.read_text().splitlines()]
        rows[34]['joint_pos']=self.policy[34]['requested_action']
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
        with self.assertRaisesRegex(AssertionError,'not measured state'):audit_model_run(self.run,self.demo)

    def test_a_stale_reset_frame_cannot_be_claimed_as_forwarded_context(self):
        path=self.run/'inference-observations.jsonl'
        original=path.read_text()
        stale=dict(index=0,time_s=0.,instruction=INSTRUCTION,joint_pos=[0.]*14,camera_jpeg_base64=self.frames)
        path.write_text(json.dumps(stale)+'\n'+original)
        with self.assertRaises(AssertionError):
            audit_model_run(self.run,self.demo)

    def test_filter_cannot_hide_extra_warmup_actions(self):
        path=self.run/'startup-camera-filter.json'
        record=json.loads(path.read_text())
        record['additional_actions']=1
        self.write(path,record)
        with self.assertRaises(AssertionError):
            audit_model_run(self.run,self.demo)
