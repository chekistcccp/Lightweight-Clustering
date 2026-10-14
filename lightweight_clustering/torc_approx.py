"""Small-data mass–distance agglomeration inspired by TORC (Yang & Lin, TPAMI 2025).

IMPORTANT: independent *approximation*. The official MATLAB core is distributed
as P-code/MEX, so equivalence to TorqueClustering.p has NOT been established.
No original protected MATLAB code is copied or decoded here.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.metrics.pairwise import cosine_distances


@dataclass
class TorqueResult:
    labels: np.ndarray
    requested_clusters: int
    merge_torques: np.ndarray
    cluster_sizes: np.ndarray


def torque_approx(features: np.ndarray, max_clusters: int = 12, fixed_k: int = 0) -> TorqueResult:
    """Mass–distance hierarchical merging of a bounded normalized coreset.

    At each level a current group proposes its closest group having at least
    equal mass. From proposals, choose the smallest mass-distance energy.
    Distance between new groups follows single linkage. Cut at an intrinsic
    energy jump, or use fixed_k. NOT an official implementation.
    Complexity: O(n^3) worst-case (vectorized); require n <= 1024.
    """
    x = np.asarray(features, dtype=np.float32)
    if x.ndim != 2 or len(x) < 3 or not np.all(np.isfinite(x)):
        raise ValueError("Expected finite N x D features with N >= 3")
    n = len(x)
    if n > 1024:
        raise ValueError("TORC-inspired exact distance graph capped at 1024 points")
    if fixed_k < 0 or fixed_k >= n:
        raise ValueError("fixed_k must be 0 (automatic) or an integer 1..N-1")
    dm = np.full((2 * n, 2 * n), np.inf, dtype=np.float32)
    d0 = cosine_distances(x).astype(np.float32)
    np.fill_diagonal(d0, np.inf)
    dm[:n, :n] = np.maximum(d0, 0.0)
    sizes = np.zeros(2 * n, dtype=np.int32)
    sizes[:n] = 1
    alive = list(range(n))
    children = {}
    torques = []
    states = []
    for step in range(n - 1):
        ids = np.asarray(alive, dtype=np.int64)
        ms = sizes[ids]
        d = dm[np.ix_(ids, ids)]
        # nearest admissible group of at least equal mass; deterministic ties
        admissible = ms[None, :] >= ms[:, None]
        np.fill_diagonal(admissible, False)
        candidates = np.where(admissible, d, np.inf)
        nearest = np.argmin(candidates, axis=1)
        nd = candidates[np.arange(len(ids)), nearest]
        energies = ms * ms[nearest] * nd**2
        if not np.isfinite(energies).any():
            raise RuntimeError("Cannot locate a valid merge partner")
        a_loc = int(np.argmin(energies))
        a, b = int(ids[a_loc]), int(ids[nearest[a_loc]])
        t = float(energies[a_loc])
        node = n + step
        children[node] = (a, b)
        sizes[node] = sizes[a] + sizes[b]
        remainder = [int(z) for z in ids if z != a and z != b]
        if remainder:
            other = np.asarray(remainder, dtype=np.int64)
            newdist = np.minimum(dm[a, other], dm[b, other])
            dm[node, other] = newdist
            dm[other, node] = newdist
        alive = remainder + [node]
        torques.append(t)
        states.append(list(alive))
    energy = np.asarray(torques, dtype=np.float64)
    if fixed_k:
        k = fixed_k
    else:
        lo = 2
        hi = min(int(max_clusters), n - 1)
        if hi < lo:
            k = 1
        else:
            choices = np.arange(lo, hi + 1)
            # Compare successive merge energies near the end of the hierarchy.
            gains = np.log1p(energy[n - choices]) - np.log1p(energy[n - choices - 1])
            k = int(choices[int(np.argmax(gains))])
    active = states[n - k - 1] if k < n else list(range(n))
    labels = np.full(n, -1, dtype=np.int32)
    for label, root in enumerate(sorted(active)):
        stack = [root]
        while stack:
            v = stack.pop()
            if v < n:
                labels[v] = label
            else:
                stack.extend(children[v])
    if (labels < 0).any():
        raise RuntimeError("Unassigned reference samples in clustering")
    unique, counts = np.unique(labels, return_counts=True)
    return TorqueResult(labels=labels, requested_clusters=len(unique),
                        merge_torques=energy, cluster_sizes=counts)


def cluster_patient_support(labels: np.ndarray, patients: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-Patch relative cluster size and distinct-patient support."""
    y = np.asarray(labels)
    p = np.asarray(patients)
    if len(y) != len(p):
        raise ValueError("labels/patients mismatch")
    counts = np.bincount(y)
    sizes = counts[y] / max(len(y), 1)
    n_people = max(len(np.unique(p)), 1)
    coverage = {int(k): len(np.unique(p[y == k])) / n_people for k in np.unique(y)}
    return sizes.astype(np.float32), np.asarray([coverage[int(k)] for k in y], np.float32)
