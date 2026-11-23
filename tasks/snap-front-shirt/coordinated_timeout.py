"""Exclude recorded GPU coordination pauses from local worker deadlines."""

import json
from pathlib import Path
import time


def paused_seconds(run, now):
    """Read completed pauses and the most recent guarded pause interval."""
    try:
        state=json.loads((Path(run)/'coordination-state.json').read_text())
    except (OSError,ValueError):
        return 0.
    total=float(state.get('paused_total_seconds',0.))
    started=state.get('pause_started_monotonic')
    heartbeat=state.get('monotonic')
    if started is not None and heartbeat is not None:
        # A dead guard must not disable deadlines indefinitely.
        total+=max(0.,min(now,float(heartbeat)+10.)-float(started))
    return total


class ActiveDeadline:
    """Measure wall time while excluding this job's documented GPU waits."""

    def __init__(self,run,timeout):
        self.run=Path(run);self.timeout=timeout;self.started=time.monotonic()
        self.initial_pause=paused_seconds(run,self.started)

    def remaining(self):
        now=time.monotonic()
        pause=max(0.,paused_seconds(self.run,now)-self.initial_pause)
        return max(0.,self.timeout-(now-self.started-pause))
