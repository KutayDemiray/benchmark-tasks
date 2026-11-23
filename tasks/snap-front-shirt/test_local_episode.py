"""Check fresh resets, ordinary actions and termination without native simulation."""

import unittest
import numpy as np
from inspect_robots import Action,Scene
from inspect_robots.types import StepResult,Termination
from local_episode import LocalEpisodeCoordinator,SnapShirtEmbodiment
from policy_contract import observation,CAMERAS


class FakeWorker:
    created=[]
    def __init__(self,out,parent,**settings):
        self.settings=settings;self.steps=0;self.finished=[];self.pid=1000+len(self.created)
        self.created.append(self)
        self.observation=self.make_observation()
    def make_observation(self):
        return observation({k:np.zeros((176,320,3),dtype=np.uint8) for k in CAMERAS},
                           np.zeros(14),self.steps/self.settings['control_hz'])
    def observe(self):return self.observation
    def step(self,action,terminal=None):
        self.steps+=1;self.observation=self.make_observation()
        return StepResult(observation=self.observation,reward=0.,terminated=terminal is not None,
                          termination_reason=None if terminal is None else terminal['kind'])
    def finish(self,reason):
        self.finished.append(reason)
        return dict(task_score=0.,steps=self.steps,termination_reason=reason,pid=self.pid)


class CoordinatorChecks(unittest.TestCase):
    def setUp(self):
        FakeWorker.created=[]
        self.coordinator=LocalEpisodeCoordinator('/unused-fake-output',worker_factory=FakeWorker)
        self.adapter=SnapShirtEmbodiment(self.coordinator)
        self.scene=Scene(id='snap-shirt-offline',instruction='Fasten the upper snaps.')
    def tearDown(self):self.adapter.close()

    def test_reset_uses_new_worker_and_restores_time(self):
        self.adapter.reset(self.scene,seed=100)
        first=FakeWorker.created[0]
        self.assertEqual(self.adapter.step(Action(np.zeros(14))).observation.state_time,.05)
        reset=self.adapter.reset(self.scene,seed=101)
        self.assertEqual(reset.state_time,0.)
        self.assertEqual(first.finished,['reset'])
        self.assertNotEqual(FakeWorker.created[0].pid,FakeWorker.created[1].pid)
        self.assertEqual(FakeWorker.created[1].settings['seed'],101)

    def test_observation_and_terminal_only_finish_do_not_advance(self):
        self.adapter.reset(self.scene)
        self.adapter.observe();self.adapter.observe()
        result=self.coordinator.finish('policy_done')
        self.assertEqual(result['steps'],0)

    def test_malformed_action_does_not_reach_native_worker(self):
        self.adapter.reset(self.scene)
        with self.assertRaises(ValueError):self.adapter.step(Action(np.full(14,np.nan)))
        self.assertEqual(FakeWorker.created[0].steps,0)

    def test_terminal_action_executes_exactly_one_step(self):
        self.adapter.reset(self.scene)
        result=self.adapter.step(Action(np.zeros(14),terminal=Termination('done')))
        self.assertTrue(result.terminated)
        self.assertEqual(result.observation.state_time,.05)
        self.assertEqual(FakeWorker.created[0].steps,1)

    def test_policy_adapter_cannot_enable_script_assistance(self):
        with self.assertRaises(AssertionError):
            SnapShirtEmbodiment(LocalEpisodeCoordinator('/unused-fake-output',controller='baseline',worker_factory=FakeWorker))
        with self.assertRaises(AssertionError):self.coordinator.script_step()
