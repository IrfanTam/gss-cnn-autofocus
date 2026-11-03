"""
Autofocus strategy simulators, replayed against pre-acquired focus stacks
(full Brenner curves per FOV). See PLAN.md and PROJECT.md ("PENTING -- sifat
tahap ini SIMULASI") for the offline-replay rationale and its limitations.

Each simulate_* function takes a FOVCurve (all scanned positions + Brenner
scores for one field of view) and returns a SimResult with step count, final
position, and success/error relative to the FOV's ground-truth peak.
"""

from dataclasses import dataclass, field

import numpy as np

PHI = (1 + np.sqrt(5)) / 2


@dataclass
class FOVCurve:
    """All scanned positions for one FOV, sorted by angular position."""
    dataset: str
    fov_id: str
    positions: np.ndarray   # sorted ascending
    brenner: np.ndarray     # aligned with positions
    image_paths: np.ndarray = None  # aligned with positions, only needed for GSS+CNN

    @property
    def ground_truth_pos(self):
        return float(self.positions[np.argmax(self.brenner)])

    @property
    def pos_min(self):
        return float(self.positions[0])

    @property
    def pos_max(self):
        return float(self.positions[-1])

    def nearest_index(self, target_pos):
        """Snap a requested (possibly non-scanned) position to the nearest
        actually-scanned point -- GSS's golden-ratio points rarely land
        exactly on a scanned position."""
        return int(np.argmin(np.abs(self.positions - target_pos)))

    def brenner_at(self, target_pos):
        return float(self.brenner[self.nearest_index(target_pos)])

    def image_at(self, target_pos):
        idx = self.nearest_index(target_pos)
        return self.image_paths[idx] if self.image_paths is not None else None


@dataclass
class SimResult:
    strategy: str
    dataset: str
    fov_id: str
    final_pos: float
    ground_truth_pos: float
    n_steps: int
    trigger_reason: str  # "full", "angular", "cnn", "hill_climb_stuck"
    success: bool = field(init=False)
    error: float = field(init=False)

    def __post_init__(self):
        self.error = abs(self.final_pos - self.ground_truth_pos)


def evaluate_success(result, delta=1.0):
    result.success = result.error <= delta
    return result


# =============================================================================
# STRATEGY 1: Full Sweep -- exhaustively visit every scanned position
# =============================================================================

def simulate_full_sweep(curve: FOVCurve, delta=1.0) -> SimResult:
    peak_idx = int(np.argmax(curve.brenner))
    result = SimResult(
        strategy="full_sweep", dataset=curve.dataset, fov_id=curve.fov_id,
        final_pos=float(curve.positions[peak_idx]),
        ground_truth_pos=curve.ground_truth_pos,
        n_steps=len(curve.positions),
        trigger_reason="full",
    )
    return evaluate_success(result, delta)


# =============================================================================
# STRATEGY 2: Hill Climbing -- greedy ascent from the smallest scanned angle
# =============================================================================

def simulate_hill_climbing(curve: FOVCurve, delta=1.0) -> SimResult:
    idx = 0  # start from the smallest available angle, same convention as access_final.tex
    steps = 1
    while True:
        current_b = curve.brenner[idx]
        next_idx = idx + 1 if idx + 1 < len(curve.positions) else None
        next_b = curve.brenner[next_idx] if next_idx is not None else -np.inf

        if next_b > current_b:
            idx = next_idx
            steps += 1
        else:
            break

    result = SimResult(
        strategy="hill_climbing", dataset=curve.dataset, fov_id=curve.fov_id,
        final_pos=float(curve.positions[idx]),
        ground_truth_pos=curve.ground_truth_pos,
        n_steps=steps,
        trigger_reason="hill_climb_stuck",
    )
    return evaluate_success(result, delta)


# =============================================================================
# STRATEGY 3: Golden Section Search only (no CNN)
# =============================================================================

def simulate_gss_only(curve: FOVCurve, theta_min=1.0, delta=1.0) -> SimResult:
    a, b = curve.pos_min, curve.pos_max
    steps = 0

    c = b - (b - a) / PHI
    d = a + (b - a) / PHI
    fc = curve.brenner_at(c)
    fd = curve.brenner_at(d)
    steps += 2

    while (b - a) > theta_min:
        if fc > fd:
            b = d
            d, fd = c, fc
            c = b - (b - a) / PHI
            fc = curve.brenner_at(c)
        else:
            a = c
            c, fc = d, fd
            d = a + (b - a) / PHI
            fd = curve.brenner_at(d)
        steps += 1

    final_pos = curve.positions[curve.nearest_index((a + b) / 2)]
    result = SimResult(
        strategy="gss_only", dataset=curve.dataset, fov_id=curve.fov_id,
        final_pos=float(final_pos),
        ground_truth_pos=curve.ground_truth_pos,
        n_steps=steps,
        trigger_reason="angular",
    )
    return evaluate_success(result, delta)


# =============================================================================
# STRATEGY 4: GSS + CNN dual-criterion termination
# =============================================================================

def simulate_gss_cnn(curve: FOVCurve, cnn_predict_fn, theta_min=1.0, p_min=0.86,
                      trigger_class="Focused", delta=1.0) -> SimResult:
    """
    cnn_predict_fn: callable(image_path) -> (predicted_class: str, confidence: float).
    Typically a cached lookup (see cnn_inference_cache.py), not live inference,
    to keep sweeping many (FOV x backbone x p_min) combinations fast.
    """
    a, b = curve.pos_min, curve.pos_max
    steps = 0

    def check_cnn(pos):
        img = curve.image_at(pos)
        if img is None:
            return False
        cls, conf = cnn_predict_fn(img)
        return cls == trigger_class and conf >= p_min

    c = b - (b - a) / PHI
    d = a + (b - a) / PHI
    fc = curve.brenner_at(c)
    fd = curve.brenner_at(d)
    steps += 2

    if check_cnn(c) or check_cnn(d):
        final_pos = c if check_cnn(c) else d
        result = SimResult(
            strategy="gss_cnn", dataset=curve.dataset, fov_id=curve.fov_id,
            final_pos=float(curve.positions[curve.nearest_index(final_pos)]),
            ground_truth_pos=curve.ground_truth_pos, n_steps=steps, trigger_reason="cnn",
        )
        return evaluate_success(result, delta)

    while (b - a) > theta_min:
        if fc > fd:
            b = d
            d, fd = c, fc
            c = b - (b - a) / PHI
            fc = curve.brenner_at(c)
            new_pos = c
        else:
            a = c
            c, fc = d, fd
            d = a + (b - a) / PHI
            fd = curve.brenner_at(d)
            new_pos = d
        steps += 1

        if check_cnn(new_pos):
            result = SimResult(
                strategy="gss_cnn", dataset=curve.dataset, fov_id=curve.fov_id,
                final_pos=float(curve.positions[curve.nearest_index(new_pos)]),
                ground_truth_pos=curve.ground_truth_pos, n_steps=steps, trigger_reason="cnn",
            )
            return evaluate_success(result, delta)

    final_pos = curve.positions[curve.nearest_index((a + b) / 2)]
    result = SimResult(
        strategy="gss_cnn", dataset=curve.dataset, fov_id=curve.fov_id,
        final_pos=float(final_pos),
        ground_truth_pos=curve.ground_truth_pos,
        n_steps=steps,
        trigger_reason="angular",
    )
    return evaluate_success(result, delta)
