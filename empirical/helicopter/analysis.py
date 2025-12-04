# -*- coding: utf-8 -*-
"""
Inertia of Belief - Prediction Testing
=======================================

Test specific predictions of the Hamiltonian / "Inertia of Belief" theory
against human behavioral data from the helicopter task.

KEY PREDICTIONS TO TEST:
========================

1. MOMENTUM SIGNATURE
   - After large prediction errors, subsequent updates should be
     correlated (momentum carries over)
   - Gradient descent predicts independent updates

2. OVERSHOOTING
   - After changepoints, beliefs may "overshoot" the new mean
   - Gradient descent predicts monotonic approach

3. SMOOTH TRAJECTORIES
   - Belief trajectories should be smoother than gradient predicts
   - Measure: autocorrelation of updates, trajectory curvature

4. LEARNING RATE DYNAMICS
   - Effective learning rate should depend on recent history (momentum)
   - Not just on current prediction error

5. SETTLING TIME
   - Time to converge after changepoint depends on mass/friction
   - Heavier beliefs take longer to settle

Author: Hamiltonian-VFE Team
Date: December 2025
"""

import sys
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from scipy import stats

# Handle both module execution and direct script execution
try:
    from .data_loader import SubjectData
    from .fitting import ModelFit, ModelComparison
except ImportError:
    _this_dir = Path(__file__).parent
    _project_root = _this_dir.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))

    from empirical.helicopter.data_loader import SubjectData
    from empirical.helicopter.fitting import ModelFit, ModelComparison


# =============================================================================
# Prediction 1: Momentum Signature
# =============================================================================

@dataclass
class MomentumAnalysis:
    """Results of momentum signature analysis."""
    update_autocorr_lag1: float     # Autocorrelation of updates at lag 1
    update_autocorr_lag2: float     # Autocorrelation at lag 2
    consecutive_same_sign: float    # Fraction of consecutive same-sign updates
    mean_run_length: float          # Mean length of same-direction runs

    # Comparison to shuffled baseline
    autocorr_z_score: float         # How many SDs above shuffled?
    p_value: float                  # Statistical significance


def analyze_momentum_signature(subject: SubjectData,
                               n_shuffles: int = 1000) -> MomentumAnalysis:
    """
    Test for momentum in human belief updates.

    Momentum predicts that updates should be autocorrelated:
    - If you updated upward last trial, you're more likely to update upward again
    - Gradient descent predicts no autocorrelation (independent updates)
    """
    arrays = subject.get_arrays()
    updates = arrays['update']

    # Remove NaN/Inf
    valid = np.isfinite(updates)
    updates = updates[valid]

    if len(updates) < 10:
        return MomentumAnalysis(0, 0, 0, 0, 0, 1.0)

    # Compute autocorrelation
    def autocorr(x, lag):
        n = len(x)
        if lag >= n:
            return 0.0
        return np.corrcoef(x[:-lag], x[lag:])[0, 1]

    ac_lag1 = autocorr(updates, 1)
    ac_lag2 = autocorr(updates, 2)

    # Consecutive same-sign updates
    signs = np.sign(updates)
    same_sign = np.mean(signs[1:] == signs[:-1])

    # Run lengths (sequences of same-direction updates)
    run_lengths = []
    current_run = 1
    for i in range(1, len(signs)):
        if signs[i] == signs[i-1]:
            current_run += 1
        else:
            run_lengths.append(current_run)
            current_run = 1
    run_lengths.append(current_run)
    mean_run = np.mean(run_lengths)

    # Compare to shuffled baseline
    shuffled_ac = []
    for _ in range(n_shuffles):
        shuffled = np.random.permutation(updates)
        shuffled_ac.append(autocorr(shuffled, 1))

    z_score = (ac_lag1 - np.mean(shuffled_ac)) / (np.std(shuffled_ac) + 1e-6)
    p_value = np.mean(np.array(shuffled_ac) >= ac_lag1)

    return MomentumAnalysis(
        update_autocorr_lag1=ac_lag1,
        update_autocorr_lag2=ac_lag2,
        consecutive_same_sign=same_sign,
        mean_run_length=mean_run,
        autocorr_z_score=z_score,
        p_value=p_value,
    )


# =============================================================================
# Prediction 2: Overshooting After Changepoints
# =============================================================================

@dataclass
class OvershootAnalysis:
    """Results of overshoot analysis."""
    n_changepoints: int
    n_overshoots: int               # Times belief crossed true mean
    overshoot_rate: float           # Fraction of CPs with overshoot
    mean_overshoot_magnitude: float # How far past the true mean
    mean_trials_to_peak: int        # Trials until max deviation


def analyze_overshooting(subject: SubjectData,
                         window_after: int = 15) -> OvershootAnalysis:
    """
    Test for overshooting after changepoints.

    Hamiltonian dynamics predict that after a changepoint, beliefs may
    overshoot the new true mean before settling.

    Definition of overshoot:
    - Belief crosses the new true mean
    - Then moves back toward it
    """
    arrays = subject.get_arrays()
    predictions = arrays['prediction']
    true_means = arrays['true_mean']
    changepoints = arrays['is_changepoint']

    cp_indices = np.where(changepoints)[0]
    n_trials = len(predictions)

    overshoots = 0
    overshoot_magnitudes = []
    trials_to_peak = []

    for cp_idx in cp_indices:
        if cp_idx + window_after >= n_trials:
            continue

        # Get segment after changepoint
        new_mean = true_means[cp_idx]
        old_mean = true_means[cp_idx - 1] if cp_idx > 0 else new_mean

        # Direction of change
        direction = np.sign(new_mean - old_mean)
        if direction == 0:
            continue

        # Track beliefs after changepoint
        segment_beliefs = predictions[cp_idx:cp_idx + window_after]

        # Check for overshoot: belief goes past new_mean
        if direction > 0:  # Mean increased
            # Overshoot if belief exceeds new_mean
            past_mean = segment_beliefs > new_mean
        else:  # Mean decreased
            # Overshoot if belief falls below new_mean
            past_mean = segment_beliefs < new_mean

        if np.any(past_mean):
            overshoots += 1
            # Magnitude: max distance past the new mean
            if direction > 0:
                magnitude = np.max(segment_beliefs - new_mean)
            else:
                magnitude = np.max(new_mean - segment_beliefs)
            overshoot_magnitudes.append(magnitude)

            # Trials to peak
            peak_idx = np.argmax(np.abs(segment_beliefs - new_mean))
            trials_to_peak.append(peak_idx)

    n_cps = len(cp_indices)
    return OvershootAnalysis(
        n_changepoints=n_cps,
        n_overshoots=overshoots,
        overshoot_rate=overshoots / n_cps if n_cps > 0 else 0.0,
        mean_overshoot_magnitude=np.mean(overshoot_magnitudes) if overshoot_magnitudes else 0.0,
        mean_trials_to_peak=int(np.mean(trials_to_peak)) if trials_to_peak else 0,
    )


# =============================================================================
# Prediction 3: Trajectory Smoothness
# =============================================================================

@dataclass
class SmoothnessAnalysis:
    """Results of trajectory smoothness analysis."""
    update_variance: float          # Variance of updates
    second_diff_variance: float     # Variance of acceleration (d²μ/dt²)
    smoothness_ratio: float         # Ratio indicating smoothness
    direction_changes: int          # Number of sign changes in updates
    direction_change_rate: float    # Rate of direction changes


def analyze_smoothness(subject: SubjectData) -> SmoothnessAnalysis:
    """
    Test for smooth belief trajectories.

    Hamiltonian dynamics produce smoother trajectories than gradient descent
    because momentum resists sudden direction changes.

    Metrics:
    - Lower second-derivative variance = smoother
    - Fewer direction changes in updates = smoother
    """
    arrays = subject.get_arrays()
    updates = arrays['update']

    valid = np.isfinite(updates)
    updates = updates[valid]

    if len(updates) < 5:
        return SmoothnessAnalysis(0, 0, 0, 0, 0)

    # First difference of updates (acceleration)
    acceleration = np.diff(updates)

    update_var = np.var(updates)
    accel_var = np.var(acceleration)

    # Smoothness ratio: lower = smoother
    # (variance of acceleration relative to variance of velocity)
    smoothness_ratio = accel_var / (update_var + 1e-6)

    # Direction changes
    signs = np.sign(updates)
    direction_changes = np.sum(signs[1:] != signs[:-1])
    direction_change_rate = direction_changes / (len(updates) - 1)

    return SmoothnessAnalysis(
        update_variance=update_var,
        second_diff_variance=accel_var,
        smoothness_ratio=smoothness_ratio,
        direction_changes=direction_changes,
        direction_change_rate=direction_change_rate,
    )


# =============================================================================
# Prediction 4: Learning Rate Dynamics
# =============================================================================

@dataclass
class LearningRateDynamics:
    """Results of learning rate dynamics analysis."""
    mean_lr: float
    std_lr: float
    lr_after_large_error: float     # LR after large prediction errors
    lr_after_small_error: float     # LR after small prediction errors
    lr_autocorrelation: float       # Autocorrelation of learning rates
    lr_depends_on_history: bool     # Statistical test result


def analyze_learning_rate_dynamics(subject: SubjectData,
                                   large_error_threshold: float = 50.0) -> LearningRateDynamics:
    """
    Test whether learning rate depends on history.

    Gradient descent: LR is fixed or depends only on current precision
    Hamiltonian: LR depends on momentum state (history of updates)
    """
    arrays = subject.get_arrays()
    updates = arrays['update']
    errors = arrays['prediction_error']

    # Compute learning rates
    with np.errstate(divide='ignore', invalid='ignore'):
        lrs = updates / errors
        valid = np.isfinite(lrs) & (np.abs(lrs) < 5)  # Filter extreme values

    lrs_valid = lrs[valid]

    if len(lrs_valid) < 10:
        return LearningRateDynamics(0, 0, 0, 0, 0, False)

    # Basic stats
    mean_lr = np.mean(lrs_valid)
    std_lr = np.std(lrs_valid)

    # LR conditioned on previous error magnitude
    large_error = np.abs(errors) > large_error_threshold
    small_error = np.abs(errors) <= large_error_threshold

    # LR on trials AFTER large vs small errors
    lr_after_large = []
    lr_after_small = []
    for i in range(1, len(lrs)):
        if valid[i]:
            if large_error[i-1]:
                lr_after_large.append(lrs[i])
            elif small_error[i-1]:
                lr_after_small.append(lrs[i])

    mean_lr_after_large = np.mean(lr_after_large) if lr_after_large else mean_lr
    mean_lr_after_small = np.mean(lr_after_small) if lr_after_small else mean_lr

    # Autocorrelation of learning rates
    lr_autocorr = np.corrcoef(lrs_valid[:-1], lrs_valid[1:])[0, 1]

    # Statistical test: does LR differ based on history?
    if len(lr_after_large) > 5 and len(lr_after_small) > 5:
        _, p_val = stats.ttest_ind(lr_after_large, lr_after_small)
        lr_depends = p_val < 0.05
    else:
        lr_depends = False

    return LearningRateDynamics(
        mean_lr=mean_lr,
        std_lr=std_lr,
        lr_after_large_error=mean_lr_after_large,
        lr_after_small_error=mean_lr_after_small,
        lr_autocorrelation=lr_autocorr,
        lr_depends_on_history=lr_depends,
    )


# =============================================================================
# Prediction 5: Settling Time After Changepoints
# =============================================================================

@dataclass
class SettlingAnalysis:
    """Results of settling time analysis."""
    mean_settling_time: float       # Trials to reach 90% of new mean
    std_settling_time: float
    mean_error_at_5_trials: float   # Error 5 trials after CP
    mean_error_at_10_trials: float  # Error 10 trials after CP


def analyze_settling_time(subject: SubjectData,
                          threshold: float = 0.1,
                          max_window: int = 30) -> SettlingAnalysis:
    """
    Analyze settling time after changepoints.

    Heavier beliefs (larger mass) take longer to settle.
    """
    arrays = subject.get_arrays()
    predictions = arrays['prediction']
    true_means = arrays['true_mean']
    changepoints = arrays['is_changepoint']

    cp_indices = np.where(changepoints)[0]
    n_trials = len(predictions)

    settling_times = []
    errors_at_5 = []
    errors_at_10 = []

    for cp_idx in cp_indices:
        if cp_idx + max_window >= n_trials or cp_idx < 1:
            continue

        new_mean = true_means[cp_idx]
        old_mean = true_means[cp_idx - 1]
        change_magnitude = abs(new_mean - old_mean)

        if change_magnitude < 10:  # Skip small changes
            continue

        # Find settling time
        for t in range(1, max_window):
            if cp_idx + t >= n_trials:
                break
            error = abs(predictions[cp_idx + t] - new_mean)
            if error < threshold * change_magnitude:
                settling_times.append(t)
                break

        # Errors at fixed times
        if cp_idx + 5 < n_trials:
            errors_at_5.append(abs(predictions[cp_idx + 5] - new_mean))
        if cp_idx + 10 < n_trials:
            errors_at_10.append(abs(predictions[cp_idx + 10] - new_mean))

    return SettlingAnalysis(
        mean_settling_time=np.mean(settling_times) if settling_times else max_window,
        std_settling_time=np.std(settling_times) if settling_times else 0,
        mean_error_at_5_trials=np.mean(errors_at_5) if errors_at_5 else 0,
        mean_error_at_10_trials=np.mean(errors_at_10) if errors_at_10 else 0,
    )


# =============================================================================
# Comprehensive Analysis
# =============================================================================

@dataclass
class InertiaAnalysis:
    """Complete analysis of inertia-of-belief predictions."""
    subject_id: int
    momentum: MomentumAnalysis
    overshoot: OvershootAnalysis
    smoothness: SmoothnessAnalysis
    learning_rate: LearningRateDynamics
    settling: SettlingAnalysis

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"=== Subject {self.subject_id} - Inertia Analysis ===",
            f"",
            f"MOMENTUM SIGNATURE:",
            f"  Update autocorrelation: {self.momentum.update_autocorr_lag1:.3f} (z={self.momentum.autocorr_z_score:.2f}, p={self.momentum.p_value:.4f})",
            f"  Consecutive same-sign: {100*self.momentum.consecutive_same_sign:.1f}%",
            f"",
            f"OVERSHOOTING:",
            f"  Overshoot rate: {100*self.overshoot.overshoot_rate:.1f}%",
            f"  Mean magnitude: {self.overshoot.mean_overshoot_magnitude:.1f}",
            f"",
            f"SMOOTHNESS:",
            f"  Direction change rate: {100*self.smoothness.direction_change_rate:.1f}%",
            f"  Smoothness ratio: {self.smoothness.smoothness_ratio:.3f}",
            f"",
            f"LEARNING RATE:",
            f"  Mean LR: {self.learning_rate.mean_lr:.3f} ± {self.learning_rate.std_lr:.3f}",
            f"  LR autocorrelation: {self.learning_rate.lr_autocorrelation:.3f}",
            f"  History-dependent: {self.learning_rate.lr_depends_on_history}",
            f"",
            f"SETTLING TIME:",
            f"  Mean settling: {self.settling.mean_settling_time:.1f} trials",
        ]
        return "\n".join(lines)


def analyze_inertia_predictions(subject: SubjectData) -> InertiaAnalysis:
    """Run complete inertia-of-belief analysis on a subject."""
    return InertiaAnalysis(
        subject_id=subject.subject_id,
        momentum=analyze_momentum_signature(subject),
        overshoot=analyze_overshooting(subject),
        smoothness=analyze_smoothness(subject),
        learning_rate=analyze_learning_rate_dynamics(subject),
        settling=analyze_settling_time(subject),
    )


def analyze_all_subjects(subjects: Dict[int, SubjectData],
                         verbose: bool = True) -> List[InertiaAnalysis]:
    """Run inertia analysis on all subjects."""
    analyses = []
    for subj_id, subject in subjects.items():
        if verbose:
            print(f"Analyzing subject {subj_id}...")
        analysis = analyze_inertia_predictions(subject)
        analyses.append(analysis)
    return analyses


def summarize_inertia_evidence(analyses: List[InertiaAnalysis]) -> Dict:
    """
    Summarize evidence for inertia-of-belief across subjects.

    Returns aggregate statistics testing each prediction.
    """
    n = len(analyses)

    # Momentum: significant positive autocorrelation?
    momentum_significant = sum(1 for a in analyses if a.momentum.p_value < 0.05)
    mean_autocorr = np.mean([a.momentum.update_autocorr_lag1 for a in analyses])

    # Overshooting
    mean_overshoot_rate = np.mean([a.overshoot.overshoot_rate for a in analyses])

    # Smoothness
    mean_direction_change = np.mean([a.smoothness.direction_change_rate for a in analyses])

    # Learning rate
    lr_history_dependent = sum(1 for a in analyses if a.learning_rate.lr_depends_on_history)
    mean_lr_autocorr = np.mean([a.learning_rate.lr_autocorrelation for a in analyses])

    # Settling
    mean_settling = np.mean([a.settling.mean_settling_time for a in analyses])

    return {
        'n_subjects': n,

        # Prediction 1: Momentum
        'momentum_significant_count': momentum_significant,
        'momentum_significant_rate': momentum_significant / n,
        'mean_update_autocorrelation': mean_autocorr,

        # Prediction 2: Overshooting
        'mean_overshoot_rate': mean_overshoot_rate,

        # Prediction 3: Smoothness
        'mean_direction_change_rate': mean_direction_change,

        # Prediction 4: Learning rate dynamics
        'lr_history_dependent_count': lr_history_dependent,
        'mean_lr_autocorrelation': mean_lr_autocorr,

        # Prediction 5: Settling
        'mean_settling_time': mean_settling,
    }


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    from .data_loader import load_mcguire_nassar_2014

    print("Loading data...")
    subjects = load_mcguire_nassar_2014()

    print("\nAnalyzing inertia predictions...")
    analyses = analyze_all_subjects(subjects, verbose=False)

    print("\n" + "="*60)
    print("INERTIA OF BELIEF - EVIDENCE SUMMARY")
    print("="*60)

    summary = summarize_inertia_evidence(analyses)
    for key, val in summary.items():
        if isinstance(val, float):
            print(f"  {key}: {val:.4f}")
        else:
            print(f"  {key}: {val}")

    # Show one example
    print("\n" + analyses[0].summary())
