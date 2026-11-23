"""Persistent local coordinator with a new native worker for every reset."""

import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

from inspect_robots.embodiment import EmbodimentBase
from policy_contract import INSTRUCTION,embodiment_info,validate_action
from fresh_workers import prepare_worker
from coordinated_timeout import ActiveDeadline


class NativeWorker:
    """Own one native child and its private inherited observation/action pipe."""

    def __init__(self, out, parent, *, seed, control_hz, seconds, iterations, pitch, controller, instruction, pilot_diagnostics=False):
        self.out=Path(out);self.coordination_root=Path(parent)
        bundle,env,descriptors=prepare_worker(self.out,parent)
        self.conn,child=multiprocessing.Pipe(duplex=True)
        self.log=(self.out/'console.log').open('x')
        command=[sys.executable,'-B',str(bundle/'policy_worker.py'),'--out',str(self.out),
            '--ipc-fd',str(child.fileno()),'--seed',str(seed),'--control-hz',str(control_hz),
            '--seconds',str(seconds),'--cloth-solver-iterations',str(iterations),
            '--body-mesh-pitch',str(pitch),'--controller',controller,'--instruction',instruction]
        if pilot_diagnostics:command.append('--pilot-diagnostics')
        self.process=subprocess.Popen(command,env=env,cwd=bundle,stdout=self.log,
            stderr=subprocess.STDOUT,pass_fds=descriptors+(child.fileno(),),start_new_session=True)
        child.close()
        self.finished=False
        (self.out/'worker.json').write_text(json.dumps(dict(worker_pid=self.process.pid,
            coordinator_pid=os.getpid(),command=command,reset_mode='fresh_native_process'),indent=2)+'\n')
        try:
            ready=self.receive(timeout=240)
            self.observation=ready['observation']
            self.pid=ready['pid']
            self.last_native_index=ready['native_physics_index']
        except BaseException:
            self.stop_owned_process()
            raise

    def receive(self, timeout=120):
        deadline=ActiveDeadline(self.coordination_root,timeout)
        while not self.conn.poll(min(.5,deadline.remaining())):
            if deadline.remaining()<=0:
                raise TimeoutError('Native worker response deadline exceeded')
        response=self.conn.recv()
        if not response.get('ok'):
            raise RuntimeError(response.get('error','Native worker failed'))
        return response

    def request(self, **message):
        assert not self.finished, 'Native worker is closed'
        self.conn.send(message)
        return self.receive()

    def step(self, action, terminal=None):
        result=self.request(op='step',action=validate_action(action),terminal=terminal)['result']
        self.observation=result.observation
        return result

    def script_step(self):
        result=self.request(op='script_step')['result']
        self.observation=result.observation
        return result

    def observe(self):
        response=self.request(op='observe')
        self.observation=response['observation']
        self.last_native_index=response['native_physics_index']
        return self.observation

    def stop_owned_process(self):
        """Clean up only this exact child after its normal shutdown deadline."""
        deadline=ActiveDeadline(self.coordination_root,60)
        while self.process.poll() is None and deadline.remaining()>0:
            try:self.process.wait(timeout=min(.5,deadline.remaining()))
            except subprocess.TimeoutExpired:pass
        if self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill();self.process.wait(timeout=10)
        self.conn.close();self.log.close()
        code=self.process.returncode
        if code==0 and (not (self.out/'results.json').exists() or (self.out/'failure.json').exists()):code=1
        (self.out/'exit.json').write_text(json.dumps(dict(returncode=code,worker_pid=self.process.pid,
            coordinator_pid=os.getpid()),indent=2)+'\n')
        return code

    def finish(self, reason='closed'):
        if self.finished:return self.result
        try:self.result=self.request(op='finish',reason=reason)['result']
        finally:
            self.finished=True
            code=self.stop_owned_process()
        if code!=0:raise RuntimeError(f'Native worker exited with status {code}')
        return self.result


class LocalEpisodeCoordinator:
    """Keep control/scoring time inside fresh native episodes, never wall time."""

    def __init__(self, out, *, control_hz=20, seconds=300., iterations=64, pitch=.01,
                 controller='none', pilot_diagnostics=False, worker_factory=NativeWorker):
        assert control_hz in (10,20) and 0<seconds<=300
        self.out=Path(out);self.worker_factory=worker_factory
        self.settings=dict(control_hz=control_hz,seconds=seconds,iterations=iterations,pitch=pitch,controller=controller,pilot_diagnostics=pilot_diagnostics)
        self.index=0;self.worker=None;self.results=[]

    def reset(self, *, seed=100, instruction=INSTRUCTION):
        self.finish('reset')
        path=self.out/f'worker-{self.index:02d}'
        self.worker=self.worker_factory(path,self.out,seed=seed,instruction=instruction,**self.settings)
        self.index+=1
        return self.worker.observation

    def observe(self):
        assert self.worker is not None, 'Reset is required'
        return self.worker.observe()

    def step(self, action, terminal=None):
        assert self.worker is not None, 'Reset is required'
        return self.worker.step(validate_action(action),terminal)

    def script_step(self):
        assert self.worker is not None and self.settings['controller'] in ('baseline','bimanual_1','bimanual_2','bimanual_3')
        return self.worker.script_step()

    def finish(self, reason='closed'):
        if self.worker is None:return None
        worker=self.worker
        self.worker=None
        result=worker.finish(reason)
        self.results.append(result)
        return result

    def close(self):
        return self.finish('closed')


class SnapShirtEmbodiment(EmbodimentBase):
    """Inspect Robots adapter; scripted native assistance is unavailable here."""

    def __init__(self, coordinator):
        assert coordinator.settings['controller']=='none'
        self.coordinator=coordinator
        self.info=embodiment_info(coordinator.settings['control_hz'])

    def reset(self, scene, *, seed=None):
        return self.coordinator.reset(seed=100 if seed is None else seed,
                                      instruction=scene.instruction or INSTRUCTION)

    def observe(self):
        return self.coordinator.observe()

    def step(self, action):
        terminal=None if action.terminal is None else dict(kind=action.terminal.kind,reason=action.terminal.reason)
        return self.coordinator.step(action.data,terminal)

    def close(self):
        self.coordinator.close()
