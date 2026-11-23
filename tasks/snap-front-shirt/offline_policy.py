"""Injected local session for exercising Dream's installed adapter without a service."""

import numpy as np


class OfflineSession:
    """Return recorded targets or a measured-position hold; never open a client."""

    def __init__(self, rows=None):
        self.rows=None if rows is None else np.asarray(rows,dtype=float)
        if self.rows is not None:
            assert self.rows.ndim==2 and self.rows.shape[1]==14 and np.isfinite(self.rows).all()
        self.index=0;self.state=None;self.observations=0;self.requests=0

    def check(self, episode):
        assert episode.model=='offline/injected-session', 'This session is exclusively an offline test'

    def start(self, episode):
        self.check(episode)
        self.index=0;self.state=None;self.observations=0;self.requests=0

    def observe(self, frames, state, prompt, timestamp):
        assert len(frames)==3 and np.isfinite(timestamp)
        self.state=np.asarray(state,dtype=float).copy()
        assert self.state.shape==(14,) and np.isfinite(self.state).all()
        self.observations+=1

    def act(self, commanded):
        from inspect_robots_dream.session import Chunk
        assert self.state is not None
        self.requests+=1
        if self.rows is None:
            rows=np.repeat(self.state[None],48,axis=0)
        else:
            if self.index>=len(self.rows):raise EOFError('Offline action recording exhausted')
            rows=self.rows[self.index:self.index+48]
            self.index+=len(rows)
        return Chunk(rows=rows,rtt_s=0.,attempts=1)

    def close(self):
        pass
