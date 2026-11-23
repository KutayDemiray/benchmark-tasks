"""Refitted triangle bounding boxes for conservative CPU contact candidates.

The hierarchy only orders triangles. Every box is recomputed from the current
native positions; arbitrarily large deformation does not invalidate it. The
existing crossing narrow phase and its tolerances remain unchanged.
"""

import numpy as np

from sat_accelerated import njit


@njit(cache=False)
def _overlap(low_a, high_a, low_b, high_b):
    for axis in range(3):
        if high_a[axis] < low_b[axis] or high_b[axis] < low_a[axis]:
            return False
    return True


@njit(cache=False)
def _pairs(triangle_low, triangle_high, faces, order, children, ranges):
    count = len(children)
    low = np.empty((count, 3))
    high = np.empty((count, 3))
    # Children precede their parent, so this single pass refits the whole tree.
    for node in range(count):
        left, right = children[node]
        if left < 0:
            for axis in range(3):
                low[node, axis] = np.inf
                high[node, axis] = -np.inf
            start, stop = ranges[node]
            for offset in range(start, stop):
                face = order[offset]
                for axis in range(3):
                    low[node, axis] = min(low[node, axis], triangle_low[face, axis])
                    high[node, axis] = max(high[node, axis], triangle_high[face, axis])
        else:
            for axis in range(3):
                low[node, axis] = min(low[left, axis], low[right, axis])
                high[node, axis] = max(high[left, axis], high[right, axis])

    pending = [(count - 1, count - 1)]
    found = []
    while pending:
        a, b = pending.pop()
        if not _overlap(low[a], high[a], low[b], high[b]):
            continue
        al, ar = children[a]
        bl, br = children[b]
        if al < 0 and bl < 0:
            for ai in range(ranges[a, 0], ranges[a, 1]):
                i = order[ai]
                for bi in range(ranges[b, 0], ranges[b, 1]):
                    j = order[bi]
                    if a == b and ai >= bi:
                        continue
                    if not _overlap(triangle_low[i], triangle_high[i], triangle_low[j], triangle_high[j]):
                        continue
                    shared = False
                    for v in range(3):
                        for w in range(3):
                            if faces[i, v] == faces[j, w]:
                                shared = True
                    if not shared:
                        found.append((min(i, j), max(i, j)))
        elif a == b:
            pending.append((al, al))
            pending.append((al, ar))
            pending.append((ar, ar))
        elif al >= 0 and (bl < 0 or ranges[a, 1] - ranges[a, 0] >= ranges[b, 1] - ranges[b, 0]):
            pending.append((al, b))
            pending.append((ar, b))
        else:
            pending.append((a, bl))
            pending.append((a, br))
    return found


class TriangleBroadphase:
    """Find all nonadjacent overlapping triangle boxes after each deformation."""

    def __init__(self, points, faces, leaf_size=8):
        self.faces = np.asarray(faces, dtype=np.int64).copy()
        assert len(self.faces) and leaf_size >= 1
        centers = np.asarray(points, dtype=np.float64)[self.faces].mean(1)
        children, ranges, ordered = [], [], []

        def build(indices):
            start = len(ordered)
            if len(indices) <= leaf_size:
                ordered.extend(indices)
                left = right = -1
            else:
                local = centers[indices]
                axis = np.ptp(local, axis=0).argmax()
                indices = indices[np.argsort(local[:, axis], kind='stable')]
                middle = len(indices) // 2
                left, right = build(indices[:middle]), build(indices[middle:])
            node = len(children)
            children.append((left, right))
            ranges.append((start, len(ordered)))
            return node

        build(np.arange(len(self.faces)))
        self.order = np.asarray(ordered, dtype=np.int64)
        self.children = np.asarray(children, dtype=np.int64)
        self.ranges = np.asarray(ranges, dtype=np.int64)

    def candidates(self, triangles):
        """Return candidate face indices using only current triangle bounds."""
        triangles = np.asarray(triangles, dtype=np.float64)
        assert triangles.shape == (len(self.faces), 3, 3) and np.isfinite(triangles).all()
        pairs = _pairs(triangles.min(1), triangles.max(1), self.faces,
                       self.order, self.children, self.ranges)
        return np.asarray(pairs, dtype=np.int64).reshape((-1, 2))
