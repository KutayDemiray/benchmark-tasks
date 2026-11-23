"""Bounded, ordered background compression of owned native recording arrays."""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time

import numpy as np


class ChunkWriter:
    """Write every submitted state in order and publish each complete NPZ atomically."""

    def __init__(self, directory, max_pending=2):
        assert max_pending>=1
        self.directory=Path(directory)
        self.max_pending=max_pending
        self.pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='pocket-native-recording')
        self.pending=deque()
        self.records=[]
        self.next_index=0
        self.closed=False

    def _write(self,index,values,end_step):
        started=time.perf_counter()
        temporary=self.directory/f'{index:05d}.tmp.npz'
        np.savez_compressed(temporary,**values,physics_end_step=end_step)
        output=self.directory/f'{index:05d}.npz'
        temporary.replace(output)
        return dict(chunk=index,physics_end_step=end_step,bytes=output.stat().st_size,
                    compression_seconds=time.perf_counter()-started)

    def submit(self,values,end_step):
        """Copy the caller's arrays before the next physics step can reuse them."""
        assert not self.closed
        if len(self.pending)>=self.max_pending:
            self.records.append(self.pending.popleft().result())
        owned={name:np.array(array,copy=True) for name,array in values.items()}
        self.pending.append(self.pool.submit(self._write,self.next_index,owned,end_step))
        self.next_index+=1

    def close(self):
        """Wait for every chunk and propagate any write failure to the recorder."""
        assert not self.closed
        try:
            while self.pending:
                self.records.append(self.pending.popleft().result())
        finally:
            self.pool.shutdown(wait=True)
            self.closed=True
        assert [record['chunk'] for record in self.records]==list(range(self.next_index))
        (self.directory.parent/'recording-io.json').write_text(json.dumps(dict(
            ordered_single_worker=True,max_pending_chunks=self.max_pending,
            native_values_copied_before_submission=True,all_writes_completed=True,
            chunks=self.records),indent=2))
