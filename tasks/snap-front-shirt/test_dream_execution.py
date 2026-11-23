"""Exercise live integration locally without credentials, hosted calls or a simulator."""

import base64
import dataclasses
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from inspect_robots_dream.session import Episode, DEFAULT_MODEL

from dream_execution import LoggedSession, StartupCameraFilter, execute, public_observation
from dream_live_runner import claim_attempt
from local_episode import LocalEpisodeCoordinator
from offline_policy import OfflineSession
from policy_contract import CAMERAS, observation
from test_local_episode import FakeWorker


class ClockWorker(FakeWorker):
    def observe(self):
        self.last_native_index=self.steps*48+1
        return self.observation

    def finish(self,reason):
        return dict(task_score=0.,control_steps=self.steps,termination_reason=reason,
                    invalid_reasons=[],full_task_success=False)


class Client:
    def __init__(self,error=None):
        self.calls=0;self.error=error;self.session_id='local-test';self.last_timing={}
    def get_models(self):return [SimpleNamespace(id=DEFAULT_MODEL)]
    def policy(self,config):
        assert config['_record'] is False and config['checkpoint']==DEFAULT_MODEL
    def get_action(self,*,commanded):
        self.calls+=1
        self.commanded=commanded
        if self.error:raise self.error
        return {'q_l_chunk':np.full((48,7),.2).tolist(),'q_r_chunk':np.full((48,7),.3).tolist()}
    def set_observation(self,frames,state,*,timestamp):
        self.observed=(frames,state,timestamp)
    def close(self):pass


class DreamExecutionChecks(unittest.TestCase):
    def sample(self):
        return observation({name:np.zeros((176,320,3),dtype=np.uint8) for name in CAMERAS},np.zeros(14),0.)

    def test_full_observation_whitelist_rejects_privileged_data_and_bad_clocks(self):
        sample=self.sample();public_observation(sample,0)
        for changed in (dataclasses.replace(sample,extra={'snap_pose':[0,0,0]}),
                        dataclasses.replace(sample,state={'joint_pos':np.zeros(14),'contact':1.}),
                        dataclasses.replace(sample,instruction='Different task')):
            with self.assertRaises(AssertionError):public_observation(changed,0)
        with self.assertRaises(AssertionError):public_observation(sample,1)

    def test_session_preserves_left_right_actions_and_records_one_request(self):
        with tempfile.TemporaryDirectory() as directory:
            client=Client();session=LoggedSession(Path(directory),make_client=lambda:client)
            session.check(Episode());session.start(Episode())
            chunk=session.act(np.arange(14,dtype=float))
            np.testing.assert_array_equal(client.commanded['q_l_chunk'],np.arange(7))
            np.testing.assert_array_equal(client.commanded['q_r_chunk'],np.arange(7,14))
            self.assertEqual(chunk.rows.shape,(48,14))
            self.assertTrue(np.all(chunk.rows[:,:7]==.2) and np.all(chunk.rows[:,7:]==.3))
            self.assertEqual(client.calls,1);self.assertEqual(chunk.attempts,1)
            session.close()

    def test_error_is_recorded_without_retry_and_secrets_are_redacted(self):
        with tempfile.TemporaryDirectory() as directory:
            client=Client(RuntimeError('refused secret-test'))
            session=LoggedSession(Path(directory),make_client=lambda:client,
                                  redact=lambda value:value.replace('secret-test','[REDACTED]'))
            session.start(Episode())
            with self.assertRaises(RuntimeError):session.act(None)
            self.assertEqual(client.calls,1);self.assertEqual(session.requests_completed,0)
            session.close()
            text=(Path(directory)/'inference.jsonl').read_text()
            self.assertNotIn('secret-test',text);self.assertIn('[REDACTED]',text)

    def test_episode_cannot_restart_or_use_a_different_model(self):
        with tempfile.TemporaryDirectory() as directory:
            session=LoggedSession(Path(directory),make_client=Client)
            with self.assertRaises(AssertionError):session.start(Episode(model='another-model'))
            session.start(Episode())
            with self.assertRaises(AssertionError):session.start(Episode())
            session.close()

    def test_shared_loop_counts_preroll_and_preserves_zero_without_assistance(self):
        with tempfile.TemporaryDirectory() as directory, patch('socket.socket',side_effect=AssertionError('No network')):
            coordinator=LocalEpisodeCoordinator(directory,seconds=5.,controller='none',worker_factory=ClockWorker)
            session=OfflineSession()
            report=execute(coordinator,Path(directory),session,model='offline/injected-session',seconds=5.)
            self.assertEqual(report['result']['task_score'],0.)
            self.assertEqual(report['executed_steps'],100)
            self.assertEqual(session.requests,2)
            self.assertFalse(report['scripted_assistance'])
            self.assertEqual(session.observations,99)
            self.assertEqual(report['startup_camera_filter']['discarded_observations'],1)
            self.assertEqual(report['startup_camera_filter']['additional_actions'],0)
            rows=[json.loads(line) for line in (Path(directory)/'policy-steps.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),100)
            self.assertTrue(all(row['native_physics_steps_during_inference']==0 for row in rows))

    def test_policy_error_keeps_actual_executed_steps_without_a_fallback(self):
        class Broken(OfflineSession):
            def act(self,commanded):
                self.requests+=1
                raise RuntimeError('injected refusal')
        with tempfile.TemporaryDirectory() as directory, patch('socket.socket',side_effect=AssertionError('No network')):
            coordinator=LocalEpisodeCoordinator(directory,seconds=5.,controller='none',worker_factory=ClockWorker)
            session=Broken()
            report=execute(coordinator,Path(directory),session,model='offline/injected-session',seconds=5.)
            self.assertEqual(session.requests,1)
            self.assertEqual(report['executed_steps'],32)
            self.assertEqual(report['error']['message'],'injected refusal')
            self.assertEqual(report['result']['task_score'],0.)

    def test_authorization_record_cannot_be_reused_even_from_another_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'attempt.json'
            claim_attempt(path,Path('/first'))
            original=path.read_bytes()
            with self.assertRaises(FileExistsError):claim_attempt(path,Path('/second'))
            self.assertEqual(path.read_bytes(),original)

    def test_offline_native_filter_blocks_internet_but_retains_local_ipc(self):
        code = """from network_guard import deny_internet_sockets
import socket
deny_internet_sockets()
for family in (socket.AF_INET,socket.AF_INET6):
 try:socket.socket(family)
 except PermissionError:pass
 else:raise AssertionError('Internet socket allowed')
a,b=socket.socketpair();a.send(b'ok');assert b.recv(2)==b'ok';a.close();b.close()
"""
        result=subprocess.run([sys.executable,'-B','-c',code],cwd=Path(__file__).resolve().parent,
                              capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_preparation_queue_cannot_launch_hosted_or_fake_policy_modes(self):
        from dream_preparation_queue import command
        for stage,args in (('evaluation',['--mode','scripted']),('dream_local_check',[])):
            cmd=command(stage,Path('/owned'),args)
            self.assertEqual(cmd[cmd.index('--task')+1],'snap-front-shirt')
            self.assertEqual(cmd[cmd.index('--mode')+1],'exclusive')
        for stage,args in (('dream_live',[]),('evaluation',['--mode','fake'])):
            with self.assertRaises(AssertionError):command(stage,Path('/owned'),args)

    def test_installed_station_preserves_camera_identity_and_measured_joint_order(self):
        from PIL import Image
        from inspect_robots_dream import _station
        colors={'top_cam':(220,10,10),'left_cam':(10,220,10),'right_cam':(10,10,220)}
        frames={name:_station.jpeg(np.full((176,320,3),color,dtype=np.uint8))
                for name,color in colors.items()}
        measured=np.arange(14,dtype=float)/20
        with tempfile.TemporaryDirectory() as directory:
            client=Client();session=LoggedSession(Path(directory),make_client=lambda:client)
            session.start(Episode());session.observe(frames,measured,'Visible task instruction',1.6)
            images,state,timestamp=client.observed
            self.assertEqual(set(images),{'image_window.top','image_window.left','image_window.right'})
            for name,channel in (('top',0),('left',1),('right',2)):
                pixels=np.asarray(Image.open(io.BytesIO(base64.b64decode(images['image_window.'+name]))))
                self.assertEqual(pixels.shape,(176,320,3))
                self.assertEqual(np.argmax(pixels.mean(axis=(0,1))),channel)
            self.assertEqual(state['state.arm_l.joint_pos'],measured[:6].tolist())
            self.assertEqual(state['state.arm_l.gripper_pos'],[measured[6]])
            self.assertEqual(state['state.arm_r.joint_pos'],measured[7:13].tolist())
            self.assertEqual(state['state.arm_r.gripper_pos'],[measured[13]])
            self.assertEqual(state['prompt'],'Visible task instruction')
            self.assertEqual(timestamp,1.6)
            self.assertEqual(set(state),{'prompt','robot.arm','robot.exec_head',
                'robot.max_joint_vel_rad_s','robot.gripper_open','state.arm_l.joint_pos',
                'state.arm_l.gripper_pos','state.arm_r.joint_pos','state.arm_r.gripper_pos'})
            session.close()

    def test_exact_public_inputs_are_recorded_without_private_fields_or_buffer_aliasing(self):
        with tempfile.TemporaryDirectory() as directory:
            client=Client();session=LoggedSession(Path(directory),make_client=lambda:client)
            session.start(Episode())
            frames={name:'encoded-'+name for name in CAMERAS}
            state=np.arange(14,dtype=float)/20
            expected=state.tolist()
            session.observe(frames,state,'Task instruction',0.)
            state[:]=0.;frames['top_cam']='changed-later'
            session.close()
            rows=[json.loads(line) for line in (Path(directory)/'inference-observations.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),1)
            self.assertEqual(set(rows[0]),{'index','time_s','instruction','joint_pos','camera_jpeg_base64'})
            self.assertEqual(rows[0]['joint_pos'],expected)
            self.assertEqual(rows[0]['camera_jpeg_base64']['top_cam'],'encoded-top_cam')
            self.assertEqual(rows[0]['index'],0)
            self.assertEqual(rows[0]['time_s'],0.)

    def test_startup_filter_discards_only_stale_sample_and_does_not_add_actions(self):
        class RecordingSession(OfflineSession):
            def __init__(self):
                super().__init__()
                self.seen = []
            def observe(self, frames, state, prompt, timestamp):
                self.seen.append((frames, state.copy(), prompt, timestamp))
                return super().observe(frames, state, prompt, timestamp)
        session = RecordingSession()
        filtered = StartupCameraFilter(session)
        filtered.start(Episode(model='offline/injected-session'))
        frames = {name:'actual-' + name for name in CAMERAS}
        initial = np.zeros(14)
        filtered.observe(frames,initial,'Instruction',0.)
        self.assertEqual(session.seen,[])
        with self.assertRaises(AssertionError):
            filtered.act(None)
        for step in range(1,33):
            measured = np.full(14,step/1000)
            filtered.observe(frames,measured,'Instruction',step/20)
            self.assertIs(session.seen[-1][0],frames)
            np.testing.assert_array_equal(session.seen[-1][1],measured)
        self.assertEqual([row[3] for row in session.seen],[step/20 for step in range(1,33)])
        self.assertEqual(filtered.record()['discarded_observations'],1)
        self.assertEqual(filtered.record()['additional_actions'],0)
        filtered.act(None)
        self.assertEqual(session.requests,1)
        with self.assertRaises(AssertionError):
            filtered.observe(frames,initial,'Instruction',0.)

    def test_live_encoding_records_only_post_hold_inputs_in_the_unchanged_preroll(self):
        with tempfile.TemporaryDirectory() as directory, patch('socket.socket',side_effect=AssertionError('No network')):
            out=Path(directory)
            coordinator=LocalEpisodeCoordinator(directory,seconds=5.,controller='none',worker_factory=ClockWorker)
            client=Client()
            session=LoggedSession(out,make_client=lambda:client)
            report=execute(coordinator,out,session,seconds=5.)
            self.assertIsNone(report['error'])
            records=[json.loads(line) for line in (out/'inference-observations.jsonl').read_text().splitlines()]
            self.assertEqual([row['index'] for row in records],list(range(1,100)))
            self.assertEqual(records[0]['time_s'],.05)
            self.assertEqual(report['executed_steps'],100)
            self.assertEqual(client.calls,2)
            transcript=json.loads((out/'policy-transcript.json').read_text())
            self.assertEqual(transcript['preroll_steps'],32)
            self.assertEqual(transcript['requests'][0]['step'],32)
