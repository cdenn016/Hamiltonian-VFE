# -*- coding: utf-8 -*-
"""
Visualization for Inertia of Belief Analysis
=============================================

Plotting tools for comparing model predictions to human behavior.

Author: Hamiltonian-VFE Team
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict
from pathlib import Path

from .data_loader import SubjectData
from .fitting import ModelFit, ModelComparison
from .analysis import InertiaAnalysis


# =============================================================================
# Style Configuration
# =============================================================================

COLORS = {
    'human': '#2E86AB',       # Blue
    'gradient': '#E94F37',     # Red
    'hamiltonian': '#44AF69',  # Green
    'changepoint': '#F18F01',  # Orange
    'outcome': '#A23B72',      # Purple
}


def setup_style():
    """Configure matplotlib style."""
    plt.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'legend.fontsize': 10,
        'figure.figsize': (10, 6),
        'figure.dpi': 100,
    })


# =============================================================================
# Single Subject Plots
# =============================================================================

def plot_belief_trajectory(subject: SubjectData,
                           gradient_fit: Optional[ModelFit] = None,
                           hamiltonian_fit: Optional[ModelFit] = None,
                           trial_range: Optional[tuple] = None,
                           save_path: Optional[Path] = None):
    """
    Plot belief trajectories: human vs model predictions.

    Shows:
    - Observations (dots)
    - True mean (dashed line)
    - Human predictions (solid line)
    - Model predictions (if provided)
    - Changepoints (vertical lines)
    """
    setup_style()

    arrays = subject.get_arrays()
    trials = np.arange(1, len(arrays['outcome']) + 1)

    if trial_range:
        start, end = trial_range
        mask = (trials >= start) & (trials <= end)
    else:
        mask = np.ones(len(trials), dtype=bool)

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})

    # Top panel: Beliefs
    ax = axes[0]
    ax.scatter(trials[mask], arrays['outcome'][mask], alpha=0.3, s=20,
               color=COLORS['outcome'], label='Observations')
    ax.plot(trials[mask], arrays['true_mean'][mask], '--', color='black',
            alpha=0.7, linewidth=1.5, label='True Mean')
    ax.plot(trials[mask], arrays['prediction'][mask], '-', color=COLORS['human'],
            linewidth=2, label='Human')

    if gradient_fit is not None:
        ax.plot(trials[mask], gradient_fit.predicted_beliefs[mask], '-',
                color=COLORS['gradient'], linewidth=1.5, alpha=0.8, label='Gradient')
    if hamiltonian_fit is not None:
        ax.plot(trials[mask], hamiltonian_fit.predicted_beliefs[mask], '-',
                color=COLORS['hamiltonian'], linewidth=1.5, alpha=0.8, label='Hamiltonian')

    # Mark changepoints
    cp_trials = trials[arrays['is_changepoint'].astype(bool) & mask]
    for cp in cp_trials:
        ax.axvline(cp, color=COLORS['changepoint'], linestyle=':', alpha=0.5)

    ax.set_ylabel('Position')
    ax.set_title(f'Subject {subject.subject_id} - Belief Trajectory')
    ax.legend(loc='upper right')

    # Bottom panel: Updates
    ax = axes[1]
    ax.bar(trials[mask], arrays['update'][mask], alpha=0.6, color=COLORS['human'],
           label='Human', width=0.8)

    if gradient_fit is not None:
        ax.plot(trials[mask], gradient_fit.predicted_updates[mask], 'o-',
                color=COLORS['gradient'], markersize=3, alpha=0.7, label='Gradient')
    if hamiltonian_fit is not None:
        ax.plot(trials[mask], hamiltonian_fit.predicted_updates[mask], 's-',
                color=COLORS['hamiltonian'], markersize=3, alpha=0.7, label='Hamiltonian')

    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Trial')
    ax.set_ylabel('Update')
    ax.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, axes


def plot_changepoint_response(subject: SubjectData,
                              gradient_fit: Optional[ModelFit] = None,
                              hamiltonian_fit: Optional[ModelFit] = None,
                              before: int = 5,
                              after: int = 15,
                              save_path: Optional[Path] = None):
    """
    Plot average response to changepoints.

    Aligned to changepoint = trial 0.
    Shows potential overshooting and settling dynamics.
    """
    setup_style()

    segments = subject.get_trials_around_changepoints(before=before, after=after)
    if not segments:
        print("No changepoints found")
        return None, None

    # Compute average trajectory aligned to changepoint
    # Normalize by change magnitude
    rel_trials = np.arange(-before, after + 1)
    n_cps = len(segments)

    # Collect normalized belief errors
    human_errors = np.zeros((n_cps, len(rel_trials)))
    human_errors[:] = np.nan

    for i, seg in enumerate(segments):
        rel = seg['relative_trial']
        # Normalize: error from new mean / change magnitude
        new_mean = seg['true_mean'][before] if len(seg['true_mean']) > before else seg['true_mean'][-1]
        old_mean = seg['true_mean'][0]
        change_mag = abs(new_mean - old_mean)

        if change_mag < 10:
            continue

        for j, r in enumerate(rel):
            if -before <= r <= after:
                idx = r + before
                belief = seg['prediction'][j]
                error = (belief - new_mean) / change_mag
                human_errors[i, idx] = error

    mean_error = np.nanmean(human_errors, axis=0)
    std_error = np.nanstd(human_errors, axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.fill_between(rel_trials, mean_error - std_error, mean_error + std_error,
                    alpha=0.2, color=COLORS['human'])
    ax.plot(rel_trials, mean_error, '-o', color=COLORS['human'], linewidth=2,
            markersize=5, label='Human')

    ax.axvline(0, color=COLORS['changepoint'], linestyle='--', linewidth=2,
               label='Changepoint')
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)

    ax.set_xlabel('Trials Relative to Changepoint')
    ax.set_ylabel('Normalized Error (belief - new mean) / |change|')
    ax.set_title(f'Subject {subject.subject_id} - Changepoint Response (n={n_cps})')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")

    return fig, ax


def plot_update_autocorrelation(subject: SubjectData,
                                max_lag: int = 10,
                                save_path: Optional[Path] = None):
    """
    Plot autocorrelation of belief updates.

    Momentum predicts positive autocorrelation at short lags.
    """
    setup_style()

    arrays = subject.get_arrays()
    updates = arrays['update']
    valid = np.isfinite(updates)
    updates = updates[valid]

    # Compute autocorrelation
    n = len(updates)
    mean = np.mean(updates)
    var = np.var(updates)

    autocorrs = []
    for lag in range(max_lag + 1):
        if lag == 0:
            autocorrs.append(1.0)
        else:
            cov = np.mean((updates[:-lag] - mean) * (updates[lag:] - mean))
            autocorrs.append(cov / var)

    lags = np.arange(max_lag + 1)

    # Significance bounds (95% CI for white noise)
    ci = 1.96 / np.sqrt(n)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(lags, autocorrs, color=COLORS['human'], alpha=0.7)
    ax.axhline(ci, color='red', linestyle='--', alpha=0.5, label='95% CI')
    ax.axhline(-ci, color='red', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)

    ax.set_xlabel('Lag (trials)')
    ax.set_ylabel('Autocorrelation')
    ax.set_title(f'Subject {subject.subject_id} - Update Autocorrelation')
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, ax


# =============================================================================
# Group-Level Plots
# =============================================================================

def plot_model_comparison_summary(comparisons: List[ModelComparison],
                                  save_path: Optional[Path] = None):
    """
    Plot summary of model comparison across subjects.

    Shows MSE distribution for both models.
    """
    setup_style()

    grad_mses = [c.gradient_fit.mse for c in comparisons]
    ham_mses = [c.hamiltonian_fit.mse for c in comparisons]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Paired comparison
    ax = axes[0]
    n = len(comparisons)
    x = np.arange(n)
    width = 0.35

    ax.bar(x - width/2, grad_mses, width, color=COLORS['gradient'],
           label='Gradient', alpha=0.8)
    ax.bar(x + width/2, ham_mses, width, color=COLORS['hamiltonian'],
           label='Hamiltonian', alpha=0.8)

    ax.set_xlabel('Subject')
    ax.set_ylabel('MSE (updates)')
    ax.set_title('Model Fit by Subject')
    ax.legend()

    # Scatter plot
    ax = axes[1]
    ax.scatter(grad_mses, ham_mses, s=50, alpha=0.7, c=COLORS['human'])

    # Identity line
    max_mse = max(max(grad_mses), max(ham_mses))
    ax.plot([0, max_mse], [0, max_mse], 'k--', alpha=0.5)

    ax.set_xlabel('Gradient MSE')
    ax.set_ylabel('Hamiltonian MSE')
    ax.set_title('Model Comparison')

    # Count wins
    ham_wins = sum(1 for c in comparisons if c.hamiltonian_wins)
    ax.text(0.05, 0.95, f'Hamiltonian wins: {ham_wins}/{n}',
            transform=ax.transAxes, verticalalignment='top')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes


def plot_inertia_evidence_summary(analyses: List[InertiaAnalysis],
                                  save_path: Optional[Path] = None):
    """
    Plot summary of inertia evidence across subjects.

    Multi-panel figure showing key predictions.
    """
    setup_style()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel 1: Update autocorrelation distribution
    ax = axes[0, 0]
    autocorrs = [a.momentum.update_autocorr_lag1 for a in analyses]
    ax.hist(autocorrs, bins=15, color=COLORS['human'], alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(np.mean(autocorrs), color='green', linestyle='-', linewidth=2,
               label=f'Mean={np.mean(autocorrs):.3f}')
    ax.set_xlabel('Update Autocorrelation (lag 1)')
    ax.set_ylabel('Count')
    ax.set_title('Momentum Signature')
    ax.legend()

    # Panel 2: Overshoot rate
    ax = axes[0, 1]
    overshoot_rates = [a.overshoot.overshoot_rate for a in analyses]
    ax.hist(overshoot_rates, bins=15, color=COLORS['changepoint'], alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(overshoot_rates), color='green', linestyle='-', linewidth=2,
               label=f'Mean={np.mean(overshoot_rates):.2f}')
    ax.set_xlabel('Overshoot Rate')
    ax.set_ylabel('Count')
    ax.set_title('Overshooting After Changepoints')
    ax.legend()

    # Panel 3: Learning rate autocorrelation
    ax = axes[1, 0]
    lr_autocorrs = [a.learning_rate.lr_autocorrelation for a in analyses]
    ax.hist(lr_autocorrs, bins=15, color=COLORS['gradient'], alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2)
    ax.axvline(np.mean(lr_autocorrs), color='green', linestyle='-', linewidth=2,
               label=f'Mean={np.mean(lr_autocorrs):.3f}')
    ax.set_xlabel('Learning Rate Autocorrelation')
    ax.set_ylabel('Count')
    ax.set_title('Learning Rate Dynamics')
    ax.legend()

    # Panel 4: Settling time
    ax = axes[1, 1]
    settling_times = [a.settling.mean_settling_time for a in analyses]
    ax.hist(settling_times, bins=15, color=COLORS['hamiltonian'], alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(settling_times), color='green', linestyle='-', linewidth=2,
               label=f'Mean={np.mean(settling_times):.1f}')
    ax.set_xlabel('Settling Time (trials)')
    ax.set_ylabel('Count')
    ax.set_title('Settling After Changepoints')
    ax.legend()

    plt.suptitle('Evidence for Inertia of Belief', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig, axes


# =============================================================================
# Main
# =============================================================================

if __name__ == '__main__':
    from .data_loader import load_mcguire_nassar_2014, get_subject_data
    from .fitting import fit_subject
    from .analysis import analyze_inertia_predictions

    print("Loading data...")
    subjects = load_mcguire_nassar_2014()

    # Plot for subject 1
    subject = subjects[1]
    print(f"\nPlotting subject {subject.subject_id}...")

    # Fit models
    comparison = fit_subject(subject)

    # Plot trajectory
    plot_belief_trajectory(
        subject,
        gradient_fit=comparison.gradient_fit,
        hamiltonian_fit=comparison.hamiltonian_fit,
        trial_range=(1, 100),
        save_path=Path('trajectory_subject1.png')
    )

    # Plot changepoint response
    plot_changepoint_response(
        subject,
        save_path=Path('changepoint_response_subject1.png')
    )

    # Plot autocorrelation
    plot_update_autocorrelation(
        subject,
        save_path=Path('autocorrelation_subject1.png')
    )

    print("\nDone! Check generated PNG files.")
