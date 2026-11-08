"""Plan a bounded LIP reference from a known contact schedule (no motion edits)."""

import numpy as np


def smooth_step(value):
    value = np.clip(value, 0.0, 1.0)
    return value**3 * (10.0 - 15.0 * value + 6.0 * value**2)


class BalanceReference:
    """Preview known support changes using a constant-height, zero-Hdot LIP.

    Sole positions are frozen for each planned stance. This is a reference
    generator, not a proof of full-body feasibility. The WBC still enforces
    its dynamics, contact, friction, CoP and torque constraints independently.
    """

    def __init__(self, reference, speed, transfer_time, height, gravity=9.81):
        self.time = reference["time"] / speed
        self.omega = np.sqrt(gravity / height)
        support = reference["support"]
        count = len(support)
        self.feet = np.empty((count, 2, 2))
        positions = np.stack(
            [reference["pose"][name][:, :2] for name in ("left_foot", "right_foot")],
            axis=1,
        )
        held = positions[0].copy()
        for i in range(count):
            touchdown = support[i] if i == 0 else support[i] & ~support[i - 1]
            held[touchdown] = positions[i, touchdown]
            self.feet[i] = held

        boundaries = np.r_[
            0, np.flatnonzero(np.any(support[1:] != support[:-1], axis=1)) + 1, count
        ]
        self.load = np.zeros((count, 2))
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            active = support[start]
            if active.sum() == 1:
                self.load[start:end] = active.astype(float)
            elif active.sum() == 2:
                initial = self.load[start - 1] if start else np.full(2, 0.5)
                final = support[end].astype(float) if end < count else np.full(2, 0.5)
                final /= final.sum()
                end_time = self.time[end] if end < count else self.time[-1]
                begin_time = max(self.time[start], end_time - transfer_time)
                span = max(end_time - begin_time, 1e-9)
                blend = smooth_step((self.time[start:end] - begin_time) / span)
                self.load[start:end] = initial + blend[:, None] * (final - initial)
            else:
                raise ValueError("LIP preview requires at least one scheduled support foot")
        self.zmp = np.einsum("nf,nfd->nd", self.load, self.feet)

        # Bounded DCM from future piecewise-constant ZMP, terminally at rest.
        self.dcm = np.empty((count, 2))
        self.dcm[-1] = self.zmp[-1]
        for i in range(count - 2, -1, -1):
            decay = np.exp(-self.omega * (self.time[i + 1] - self.time[i]))
            self.dcm[i] = self.zmp[i] + decay * (self.dcm[i + 1] - self.zmp[i])
        self.com = np.empty((count, 2))
        self.com[0] = reference["com"][0, :2]
        for i in range(count - 1):
            x = self.omega * (self.time[i + 1] - self.time[i])
            self.com[i + 1] = (
                np.exp(-x) * self.com[i]
                + np.sinh(x) * self.dcm[i]
                + (1.0 - np.cosh(x)) * self.zmp[i]
            )

    def sample(self, time_s):
        """Exactly evolve the piecewise-constant-ZMP LIP between sample times."""
        index = min(
            max(np.searchsorted(self.time, time_s, side="right") - 1, 0), len(self.time) - 1
        )
        elapsed = max(time_s - self.time[index], 0.0)
        if index == len(self.time) - 1:
            elapsed = 0.0
        x = self.omega * elapsed
        p = self.zmp[index]
        dcm = p + np.exp(x) * (self.dcm[index] - p)
        com = np.exp(-x) * self.com[index] + np.sinh(x) * self.dcm[index] + (1.0 - np.cosh(x)) * p
        velocity = self.omega * (dcm - com)
        return com, velocity, dcm, p.copy(), self.load[index].copy(), self.feet[index].copy()


def project_to_polygon(point, equations, margin):
    """Euclidean projection onto a shrunken convex polygon in two dimensions."""
    normals = equations[:, :2]
    bounds = -equations[:, 2] - margin
    if np.all(normals @ point <= bounds + 1e-10):
        return point.copy()
    candidates = []
    for normal, bound in zip(normals, bounds):
        candidate = point - normal * ((normal @ point - bound) / (normal @ normal))
        if np.all(normals @ candidate <= bounds + 1e-9):
            candidates.append(candidate)
    for i in range(len(bounds)):
        for j in range(i):
            matrix = normals[[i, j]]
            if abs(np.linalg.det(matrix)) < 1e-10:
                continue
            candidate = np.linalg.solve(matrix, bounds[[i, j]])
            if np.all(normals @ candidate <= bounds + 1e-9):
                candidates.append(candidate)
    if not candidates:
        raise ValueError("Support polygon is degenerate or ZMP margin is too large")
    return min(candidates, key=lambda candidate: np.sum((candidate - point) ** 2)).copy()
