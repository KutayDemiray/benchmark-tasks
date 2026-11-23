"""Recording integrity, ordering and error propagation across the writer thread."""

from task_paths import OUTPUT_ROOT

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from chunk_writer import ChunkWriter


class ChunkWriterTests(unittest.TestCase):
    def test_reused_buffers_preserve_every_native_value_and_chunk_order(self):
        with tempfile.TemporaryDirectory(dir=OUTPUT_ROOT) as tmp:
            directory=Path(tmp)/'physics-chunks';directory.mkdir()
            writer=ChunkWriter(directory,max_pending=2)
            positions=np.arange(24,dtype=np.float32).reshape(2,4,3)
            velocity=np.zeros_like(positions)
            expected=[]
            for index in range(5):
                expected.append(positions.copy())
                writer.submit(dict(positions=positions,velocities=velocity),2*(index+1))
                positions+=.25
            writer.close()
            self.assertEqual([p.name for p in sorted(directory.glob('*.npz'))],
                             [f'{i:05d}.npz' for i in range(5)])
            for index,array in enumerate(expected):
                with np.load(directory/f'{index:05d}.npz') as chunk:
                    np.testing.assert_array_equal(chunk['positions'],array)
                    np.testing.assert_array_equal(chunk['velocities'],velocity)
                    self.assertEqual(chunk['positions'].dtype,np.float32)
                    self.assertEqual(int(chunk['physics_end_step']),2*(index+1))

    def test_write_failure_cannot_silently_complete_a_recording(self):
        with tempfile.TemporaryDirectory(dir=OUTPUT_ROOT) as tmp:
            directory=Path(tmp)/'physics-chunks';directory.mkdir()
            writer=ChunkWriter(directory)
            with patch('chunk_writer.np.savez_compressed',side_effect=OSError('test write failure')):
                writer.submit(dict(positions=np.zeros((2,3))),2)
                with self.assertRaises(OSError):writer.close()
            self.assertFalse((directory/'00000.npz').exists())
            self.assertFalse((directory.parent/'recording-io.json').exists())


if __name__=='__main__':
    unittest.main()
