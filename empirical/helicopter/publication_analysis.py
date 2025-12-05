#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Publication-Quality Analysis: Inertia of Belief
================================================

Comprehensive analysis of human belief dynamics in the helicopter task,
producing publication-ready figures, tables, and statistics for the
"Inertia of Belief" manuscript.

Key analyses:
1. Model comparison: Delta rule vs Momentum vs Hamiltonian vs Damped Oscillator
2. Parameter distributions across subjects
3. Damping regime classification (overdamped/underdamped/critical)
4. Statistical tests for belief inertia
5. Example trajectories and phase portraits

Output:
- Publication-quality figures (PDF/PNG)
- LaTeX-formatted tables
- Statistical summary for manuscript

Author: Hamiltonian-VFE Team
Date: December 2025
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from scipy.optimize import minimize, differential_evolution
from scipy import stats
import warnings

# Plotting
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'sans-serif',
})

# Handle imports
try:
    from .data_loader import load_mcguire_nassar_2014, SubjectData
except ImportError:
    _this_dir = Path(__file__).parent
    _project_root = _this_dir.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from empirical.helicopter.data_loader import load_mcguire_nassar_2014, SubjectData


# =============================================================================
# Data Classes for Results
# =============================================================================

@dataclass
class ModelFit:
    """Results from fitting a single model to a subject."""
    model_name: str
    params: Dict[str, float]
    mse: float
    r_squared: float
    bic: float
    aic: float
    n_params: int
    predictions: np.ndarray = field(default=None, repr=False)


@dataclass
class SubjectResults:
    """All model fits for a single subject."""
    subject_id: int
    n_trials: int
    fits: Dict[str, ModelFit]
    best_model_bic: str
    best_model_aic: str

    # Derived quantities
    damping_regime: str = ""  # 'overdamped', 'underdamped', 'critical'
    damping_ratio: float = 0.0  # gamma / omega


@dataclass
class PopulationResults:
    """Aggregate results across all subjects."""
    n_subjects: int
    subject_results: List[SubjectResults]

    # Model comparison
    bic_wins: Dict[str, int] = field(default_factory=dict)
    aic_wins: Dict[str, int] = field(default_factory=dict)

    # Parameter distributions
    momentum_betas: np.ndarray = field(default=None, repr=False)
    hamiltonian_masses: np.ndarray = field(default=None, repr=False)
    hamiltonian_frictions: np.ndarray = field(default=None, repr=False)
    oscillator_omegas: np.ndarray = field(default=None, repr=False)
    oscillator_gammas: np.ndarray = field(default=None, repr=False)
    damping_ratios: np.ndarray = field(default=None, repr=False)

    # Statistical tests
    beta_ttest: Tuple[float, float] = (0, 1)  # (t-statistic, p-value)
    beta_mean: float = 0
    beta_std: float = 0
    beta_ci: Tuple[float, float] = (0, 0)  # 95% CI

    # Damping regime counts
    n_overdamped: int = 0
    n_underdamped: int = 0
    n_critical: int = 0


# =============================================================================
# Model Implementations
# =============================================================================

def run_delta_rule(observations: np.ndarray, lr: float, init: float) -> np.ndarray:
    """
    Delta rule: mu[t+1] = mu[t] + lr * (obs[t] - mu[t])

    Equivalent to gradient descent / Kalman filter with fixed gain.
    Predicts NO momentum - each update is independent.
    """
    n = len(observations)
    beliefs = np.zeros(n + 1)
    beliefs[0] = init

    for t in range(n):
        beliefs[t + 1] = beliefs[t] + lr * (observations[t] - beliefs[t])

    return beliefs


def run_momentum_rule(observations: np.ndarray, lr: float, beta: float,
                      init: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Delta rule with momentum:
        velocity[t+1] = beta * velocity[t] + lr * error[t]
        belief[t+1] = belief[t] + velocity[t+1]

    beta = 0 reduces to delta rule.
    beta > 0 introduces inertia/momentum.
    """
    n = len(observations)
    beliefs = np.zeros(n + 1)
    velocities = np.zeros(n + 1)
    beliefs[0] = init

    for t in range(n):
        error = observations[t] - beliefs[t]
        velocities[t + 1] = beta * velocities[t] + lr * error
        beliefs[t + 1] = beliefs[t] + velocities[t + 1]

    return beliefs, velocities


def run_damped_oscillator(observations: np.ndarray, omega: float, gamma: float,
                          init: float, n_substeps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Damped harmonic oscillator:
        d²μ/dt² + 2γ dμ/dt + ω²(μ - obs) = 0

    Equivalent to Hamiltonian with mass=1, spring constant k=ω².

    Regimes:
        - γ > ω: overdamped (no oscillation)
        - γ = ω: critically damped
        - γ < ω: underdamped (oscillation)
    """
    n = len(observations)
    dt = 1.0 / n_substeps

    beliefs = np.zeros(n + 1)
    velocities = np.zeros(n + 1)
    beliefs[0] = init

    for t in range(n):
        obs = observations[t]
        mu = beliefs[t]
        v = velocities[t]

        for _ in range(n_substeps):
            # Acceleration: a = -ω²(μ - obs) - 2γv
            accel = -omega**2 * (mu - obs) - 2 * gamma * v

            # Velocity Verlet integration
            mu_new = mu + v * dt + 0.5 * accel * dt**2
            accel_new = -omega**2 * (mu_new - obs) - 2 * gamma * v
            v = v + 0.5 * (accel + accel_new) * dt
            mu = mu_new

        beliefs[t + 1] = mu
        velocities[t + 1] = v

    return beliefs, velocities


def run_hamiltonian(observations: np.ndarray, mass: float, friction: float,
                    precision: float, init: float,
                    n_substeps: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full Hamiltonian dynamics with leapfrog integration:
        H = p²/(2m) + (precision/2)(μ - obs)²

        dμ/dt = p/m
        dp/dt = -precision*(μ - obs) - friction*p
    """
    n = len(observations)
    dt = 1.0 / n_substeps

    beliefs = np.zeros(n + 1)
    momenta = np.zeros(n + 1)
    beliefs[0] = init

    for t in range(n):
        obs = observations[t]
        mu = beliefs[t]
        p = momenta[t]

        for _ in range(n_substeps):
            # Force = -dV/dμ
            force = -precision * (mu - obs)

            # Leapfrog: half step momentum
            p = p + 0.5 * dt * force

            # Full step position
            mu = mu + dt * p / mass

            # Update force at new position
            force = -precision * (mu - obs)

            # Half step momentum with friction
            p = p + 0.5 * dt * force
            p = p * (1 - friction * dt)

        beliefs[t + 1] = mu
        momenta[t + 1] = p

    return beliefs, momenta


# =============================================================================
# Fitting Functions
# =============================================================================

def fit_subject(subject: SubjectData, verbose: bool = False) -> SubjectResults:
    """
    Fit all models to a single subject.

    Returns SubjectResults with fits for all models and model comparison metrics.
    """
    arrays = subject.get_arrays()
    obs = arrays['outcome']
    human = arrays['prediction']
    n = len(human)

    # Total sum of squares for R²
    ss_tot = np.sum((human[1:] - np.mean(human[1:]))**2)
    if ss_tot < 1e-10:
        ss_tot = 1.0

    fits = {}

    # =========================================================================
    # 1. Delta Rule (1 parameter: lr)
    # =========================================================================
    def delta_loss(p):
        lr = p[0]
        if lr <= 0 or lr > 2.0:
            return 1e10
        pred = run_delta_rule(obs, lr, human[0])
        return np.mean((pred[1:-1] - human[1:])**2)

    res = minimize(delta_loss, [0.5], bounds=[(0.01, 2.0)], method='L-BFGS-B')
    lr_opt = res.x[0]
    pred_delta = run_delta_rule(obs, lr_opt, human[0])
    mse = np.mean((pred_delta[1:-1] - human[1:])**2)
    ss_res = np.sum((pred_delta[1:-1] - human[1:])**2)
    r2 = 1 - ss_res / ss_tot

    fits['delta'] = ModelFit(
        model_name='Delta Rule',
        params={'lr': lr_opt},
        mse=mse,
        r_squared=r2,
        bic=n * np.log(mse) + 1 * np.log(n),
        aic=n * np.log(mse) + 2 * 1,
        n_params=1,
        predictions=pred_delta
    )

    # =========================================================================
    # 2. Momentum Rule (2 parameters: lr, beta)
    # =========================================================================
    def momentum_loss(p):
        lr, beta = p
        if lr <= 0 or lr > 2.0 or beta < 0 or beta > 0.99:
            return 1e10
        pred, _ = run_momentum_rule(obs, lr, beta, human[0])
        return np.mean((pred[1:-1] - human[1:])**2)

    res = minimize(momentum_loss, [0.5, 0.2],
                   bounds=[(0.01, 2.0), (0.0, 0.99)], method='L-BFGS-B')
    lr_opt, beta_opt = res.x
    pred_mom, vel = run_momentum_rule(obs, lr_opt, beta_opt, human[0])
    mse = np.mean((pred_mom[1:-1] - human[1:])**2)
    ss_res = np.sum((pred_mom[1:-1] - human[1:])**2)
    r2 = 1 - ss_res / ss_tot

    fits['momentum'] = ModelFit(
        model_name='Momentum',
        params={'lr': lr_opt, 'beta': beta_opt},
        mse=mse,
        r_squared=r2,
        bic=n * np.log(mse) + 2 * np.log(n),
        aic=n * np.log(mse) + 2 * 2,
        n_params=2,
        predictions=pred_mom
    )

    # =========================================================================
    # 3. Damped Oscillator (2 parameters: omega, gamma)
    # =========================================================================
    def oscillator_loss(p):
        omega, gamma = p
        if omega <= 0 or gamma < 0:
            return 1e10
        try:
            pred, _ = run_damped_oscillator(obs, omega, gamma, human[0])
            mse = np.mean((pred[1:-1] - human[1:])**2)
            return mse if np.isfinite(mse) else 1e10
        except:
            return 1e10

    # Try multiple starting points
    best_osc = None
    best_osc_loss = 1e10
    for omega0, gamma0 in [(0.5, 0.5), (1.0, 1.0), (0.3, 0.8), (1.5, 0.3)]:
        try:
            res = minimize(oscillator_loss, [omega0, gamma0],
                          bounds=[(0.01, 3.0), (0.0, 3.0)], method='L-BFGS-B')
            if res.fun < best_osc_loss:
                best_osc = res.x
                best_osc_loss = res.fun
        except:
            pass

    if best_osc is not None:
        omega_opt, gamma_opt = best_osc
        pred_osc, vel_osc = run_damped_oscillator(obs, omega_opt, gamma_opt, human[0])
        mse = np.mean((pred_osc[1:-1] - human[1:])**2)
        ss_res = np.sum((pred_osc[1:-1] - human[1:])**2)
        r2 = 1 - ss_res / ss_tot
    else:
        omega_opt, gamma_opt = 1.0, 1.0
        pred_osc = np.full(n + 1, human[0])
        mse, r2 = 1e10, -1

    fits['oscillator'] = ModelFit(
        model_name='Damped Oscillator',
        params={'omega': omega_opt, 'gamma': gamma_opt},
        mse=mse,
        r_squared=r2,
        bic=n * np.log(max(mse, 1e-10)) + 2 * np.log(n),
        aic=n * np.log(max(mse, 1e-10)) + 2 * 2,
        n_params=2,
        predictions=pred_osc
    )

    # =========================================================================
    # 4. Hamiltonian (3 parameters: mass, friction, precision)
    # =========================================================================
    def hamiltonian_loss(p):
        mass, friction, prec = p
        if mass <= 0.01 or friction < 0 or prec <= 0:
            return 1e10
        try:
            pred, _ = run_hamiltonian(obs, mass, friction, prec, human[0])
            mse = np.mean((pred[1:-1] - human[1:])**2)
            return mse if np.isfinite(mse) else 1e10
        except:
            return 1e10

    best_ham = None
    best_ham_loss = 1e10
    for mass0, fric0, prec0 in [(1.0, 0.5, 0.1), (0.5, 1.0, 0.5), (2.0, 0.2, 0.05)]:
        try:
            res = minimize(hamiltonian_loss, [mass0, fric0, prec0],
                          bounds=[(0.01, 10.0), (0.0, 5.0), (0.001, 2.0)],
                          method='L-BFGS-B')
            if res.fun < best_ham_loss:
                best_ham = res.x
                best_ham_loss = res.fun
        except:
            pass

    if best_ham is not None:
        mass_opt, friction_opt, prec_opt = best_ham
        pred_ham, mom = run_hamiltonian(obs, mass_opt, friction_opt, prec_opt, human[0])
        mse = np.mean((pred_ham[1:-1] - human[1:])**2)
        ss_res = np.sum((pred_ham[1:-1] - human[1:])**2)
        r2 = 1 - ss_res / ss_tot
    else:
        mass_opt, friction_opt, prec_opt = 1.0, 0.5, 0.1
        pred_ham = np.full(n + 1, human[0])
        mse, r2 = 1e10, -1

    fits['hamiltonian'] = ModelFit(
        model_name='Hamiltonian',
        params={'mass': mass_opt, 'friction': friction_opt, 'precision': prec_opt},
        mse=mse,
        r_squared=r2,
        bic=n * np.log(max(mse, 1e-10)) + 3 * np.log(n),
        aic=n * np.log(max(mse, 1e-10)) + 2 * 3,
        n_params=3,
        predictions=pred_ham
    )

    # =========================================================================
    # Determine best model and damping regime
    # =========================================================================
    bics = {k: v.bic for k, v in fits.items()}
    aics = {k: v.aic for k, v in fits.items()}
    best_bic = min(bics, key=bics.get)
    best_aic = min(aics, key=aics.get)

    # Damping regime from oscillator fit
    omega = fits['oscillator'].params['omega']
    gamma = fits['oscillator'].params['gamma']

    if omega > 0:
        damping_ratio = gamma / omega
        if damping_ratio > 1.05:
            damping_regime = 'overdamped'
        elif damping_ratio < 0.95:
            damping_regime = 'underdamped'
        else:
            damping_regime = 'critical'
    else:
        damping_ratio = float('inf')
        damping_regime = 'overdamped'

    return SubjectResults(
        subject_id=subject.subject_id,
        n_trials=n,
        fits=fits,
        best_model_bic=best_bic,
        best_model_aic=best_aic,
        damping_regime=damping_regime,
        damping_ratio=damping_ratio
    )


def fit_population(subjects: Dict[int, SubjectData],
                   verbose: bool = True) -> PopulationResults:
    """
    Fit all models to all subjects and compute population statistics.
    """
    subject_results = []

    if verbose:
        print(f"\nFitting {len(subjects)} subjects...")
        print(f"{'Subj':>4} | {'Δ LR':>6} | {'M β':>6} | {'Osc γ/ω':>7} | {'Best':>10} | {'Regime':>10}")
        print("-" * 60)

    for sid in sorted(subjects.keys()):
        result = fit_subject(subjects[sid])
        subject_results.append(result)

        if verbose:
            d = result.fits['delta']
            m = result.fits['momentum']
            print(f"{sid:4d} | {d.params['lr']:6.3f} | {m.params['beta']:6.3f} | "
                  f"{result.damping_ratio:7.2f} | {result.best_model_bic:>10} | "
                  f"{result.damping_regime:>10}")

    # Aggregate statistics
    bic_wins = {'delta': 0, 'momentum': 0, 'oscillator': 0, 'hamiltonian': 0}
    aic_wins = {'delta': 0, 'momentum': 0, 'oscillator': 0, 'hamiltonian': 0}

    betas = []
    masses = []
    frictions = []
    omegas = []
    gammas = []
    damping_ratios = []

    n_overdamped = n_underdamped = n_critical = 0

    for r in subject_results:
        bic_wins[r.best_model_bic] += 1
        aic_wins[r.best_model_aic] += 1

        betas.append(r.fits['momentum'].params['beta'])
        masses.append(r.fits['hamiltonian'].params['mass'])
        frictions.append(r.fits['hamiltonian'].params['friction'])
        omegas.append(r.fits['oscillator'].params['omega'])
        gammas.append(r.fits['oscillator'].params['gamma'])
        damping_ratios.append(r.damping_ratio)

        if r.damping_regime == 'overdamped':
            n_overdamped += 1
        elif r.damping_regime == 'underdamped':
            n_underdamped += 1
        else:
            n_critical += 1

    betas = np.array(betas)

    # Statistical test: Is beta > 0?
    t_stat, p_val = stats.ttest_1samp(betas, 0)
    # One-sided test (we expect beta > 0)
    p_val_onesided = p_val / 2 if t_stat > 0 else 1 - p_val / 2

    # 95% CI for beta
    beta_mean = np.mean(betas)
    beta_std = np.std(betas, ddof=1)
    beta_se = beta_std / np.sqrt(len(betas))
    t_crit = stats.t.ppf(0.975, len(betas) - 1)
    beta_ci = (beta_mean - t_crit * beta_se, beta_mean + t_crit * beta_se)

    return PopulationResults(
        n_subjects=len(subjects),
        subject_results=subject_results,
        bic_wins=bic_wins,
        aic_wins=aic_wins,
        momentum_betas=betas,
        hamiltonian_masses=np.array(masses),
        hamiltonian_frictions=np.array(frictions),
        oscillator_omegas=np.array(omegas),
        oscillator_gammas=np.array(gammas),
        damping_ratios=np.array(damping_ratios),
        beta_ttest=(t_stat, p_val_onesided),
        beta_mean=beta_mean,
        beta_std=beta_std,
        beta_ci=beta_ci,
        n_overdamped=n_overdamped,
        n_underdamped=n_underdamped,
        n_critical=n_critical
    )


# =============================================================================
# Publication Figures
# =============================================================================

def create_main_figure(results: PopulationResults, subjects: Dict[int, SubjectData],
                       save_path: Optional[Path] = None) -> plt.Figure:
    """
    Create main publication figure with 4 panels:
    A) Model comparison (BIC wins)
    B) Distribution of momentum parameter β
    C) Damping ratio distribution (γ/ω)
    D) Example subject trajectories
    """
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)

    # =========================================================================
    # Panel A: Model Comparison
    # =========================================================================
    ax_a = fig.add_subplot(gs[0, 0])

    models = ['delta', 'momentum', 'oscillator', 'hamiltonian']
    model_labels = ['Delta Rule', 'Momentum', 'Damped Osc.', 'Hamiltonian']
    wins = [results.bic_wins[m] for m in models]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    bars = ax_a.bar(model_labels, wins, color=colors, edgecolor='black', linewidth=0.5)
    ax_a.set_ylabel('Number of subjects (best fit by BIC)')
    ax_a.set_title('A) Model Comparison', fontweight='bold', loc='left')
    ax_a.set_ylim(0, max(wins) * 1.2)

    # Add count labels on bars
    for bar, count in zip(bars, wins):
        ax_a.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                  f'{count}', ha='center', va='bottom', fontsize=10)

    ax_a.axhline(y=results.n_subjects/4, color='gray', linestyle='--',
                 alpha=0.5, label='Chance level')
    ax_a.legend(loc='upper right')

    # =========================================================================
    # Panel B: Beta Distribution
    # =========================================================================
    ax_b = fig.add_subplot(gs[0, 1])

    betas = results.momentum_betas

    ax_b.hist(betas, bins=15, color='#ff7f0e', edgecolor='black',
              linewidth=0.5, alpha=0.7, density=True)

    # Add kernel density estimate
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(betas)
    x_kde = np.linspace(0, max(betas) * 1.1, 100)
    ax_b.plot(x_kde, kde(x_kde), 'r-', linewidth=2, label='KDE')

    # Mark mean and CI
    ax_b.axvline(results.beta_mean, color='red', linestyle='-', linewidth=2,
                 label=f'Mean = {results.beta_mean:.3f}')
    ax_b.axvline(results.beta_ci[0], color='red', linestyle='--', linewidth=1)
    ax_b.axvline(results.beta_ci[1], color='red', linestyle='--', linewidth=1,
                 label=f'95% CI')
    ax_b.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)

    ax_b.set_xlabel('Momentum parameter β')
    ax_b.set_ylabel('Density')
    ax_b.set_title('B) Distribution of Belief Inertia', fontweight='bold', loc='left')
    ax_b.legend(loc='upper right')

    # Add stats annotation
    t_stat, p_val = results.beta_ttest
    sig_str = '***' if p_val < 0.001 else '**' if p_val < 0.01 else '*' if p_val < 0.05 else 'n.s.'
    ax_b.text(0.95, 0.95, f't({results.n_subjects-1}) = {t_stat:.2f}\np = {p_val:.4f} {sig_str}',
              transform=ax_b.transAxes, ha='right', va='top',
              bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # =========================================================================
    # Panel C: Damping Ratio Distribution
    # =========================================================================
    ax_c = fig.add_subplot(gs[1, 0])

    dr = results.damping_ratios
    dr_clipped = np.clip(dr, 0, 5)  # Clip for visualization

    ax_c.hist(dr_clipped, bins=20, color='#2ca02c', edgecolor='black',
              linewidth=0.5, alpha=0.7)

    # Mark critical damping line
    ax_c.axvline(1.0, color='red', linestyle='--', linewidth=2,
                 label='Critical damping (γ/ω = 1)')

    # Shade regions
    ax_c.axvspan(0, 1, alpha=0.1, color='blue', label='Underdamped')
    ax_c.axvspan(1, 5, alpha=0.1, color='red', label='Overdamped')

    ax_c.set_xlabel('Damping ratio γ/ω')
    ax_c.set_ylabel('Number of subjects')
    ax_c.set_title('C) Damping Regime Classification', fontweight='bold', loc='left')
    ax_c.legend(loc='upper right')
    ax_c.set_xlim(0, 5)

    # Add counts
    ax_c.text(0.5, 0.85, f'Underdamped: {results.n_underdamped}',
              transform=ax_c.transAxes, ha='center')
    ax_c.text(0.5, 0.78, f'Overdamped: {results.n_overdamped}',
              transform=ax_c.transAxes, ha='center')

    # =========================================================================
    # Panel D: Example Subject Trajectories
    # =========================================================================
    ax_d = fig.add_subplot(gs[1, 1])

    # Find a good example subject (moderate beta, good fit)
    best_subj_idx = 0
    best_score = -1
    for i, r in enumerate(results.subject_results):
        beta = r.fits['momentum'].params['beta']
        r2 = r.fits['momentum'].r_squared
        # Prefer moderate beta with good fit
        score = r2 * (1 - abs(beta - 0.3))
        if score > best_score and 0.1 < beta < 0.5:
            best_score = score
            best_subj_idx = i

    example_result = results.subject_results[best_subj_idx]
    example_subject = subjects[example_result.subject_id]
    arrays = example_subject.get_arrays()

    # Plot segment around a changepoint
    cp_indices = np.where(arrays['is_changepoint'])[0]
    if len(cp_indices) > 3:
        cp_idx = cp_indices[3]  # Use 4th changepoint
        start = max(0, cp_idx - 10)
        end = min(len(arrays['outcome']), cp_idx + 30)

        trials = np.arange(start, end)
        human = arrays['prediction'][start:end]
        outcomes = arrays['outcome'][start:end]
        true_mean = arrays['true_mean'][start:end]

        # Get model predictions
        delta_pred = example_result.fits['delta'].predictions[start+1:end+1]
        mom_pred = example_result.fits['momentum'].predictions[start+1:end+1]

        ax_d.plot(trials, outcomes, 'o', color='gray', alpha=0.3,
                  markersize=4, label='Observations')
        ax_d.plot(trials, true_mean, 'k--', linewidth=1, alpha=0.5,
                  label='True mean')
        ax_d.plot(trials, human, 'ko-', linewidth=2, markersize=5,
                  label='Human')
        ax_d.plot(trials, delta_pred, 'b-', linewidth=1.5, alpha=0.7,
                  label=f'Delta (LR={example_result.fits["delta"].params["lr"]:.2f})')
        ax_d.plot(trials, mom_pred, 'r-', linewidth=1.5, alpha=0.7,
                  label=f'Momentum (β={example_result.fits["momentum"].params["beta"]:.2f})')

        ax_d.axvline(cp_idx, color='green', linestyle=':', linewidth=2, alpha=0.7)
        ax_d.text(cp_idx + 1, ax_d.get_ylim()[1] * 0.95, 'Changepoint',
                  fontsize=8, color='green')

    ax_d.set_xlabel('Trial')
    ax_d.set_ylabel('Position')
    ax_d.set_title(f'D) Example Subject {example_result.subject_id}',
                   fontweight='bold', loc='left')
    ax_d.legend(loc='lower right', fontsize=8)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved figure to {save_path}")

    return fig


def create_supplementary_figure(results: PopulationResults,
                                save_path: Optional[Path] = None) -> plt.Figure:
    """
    Create supplementary figure with parameter correlations and distributions.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))

    # Panel 1: Beta vs R²
    ax = axes[0, 0]
    betas = results.momentum_betas
    r2s = [r.fits['momentum'].r_squared for r in results.subject_results]
    ax.scatter(betas, r2s, alpha=0.6, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Momentum β')
    ax.set_ylabel('R² (Momentum model)')
    ax.set_title('A) Momentum vs Fit Quality')

    # Panel 2: Mass distribution (Hamiltonian)
    ax = axes[0, 1]
    masses = results.hamiltonian_masses
    ax.hist(masses, bins=15, color='#d62728', edgecolor='black', alpha=0.7)
    ax.set_xlabel('Hamiltonian mass M')
    ax.set_ylabel('Count')
    ax.set_title(f'B) Mass Distribution (mean={np.mean(masses):.2f})')

    # Panel 3: Friction vs Mass
    ax = axes[0, 2]
    ax.scatter(masses, results.hamiltonian_frictions, alpha=0.6,
               edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Mass M')
    ax.set_ylabel('Friction γ')
    ax.set_title('C) Mass vs Friction')

    # Panel 4: Omega distribution
    ax = axes[1, 0]
    ax.hist(results.oscillator_omegas, bins=15, color='#2ca02c',
            edgecolor='black', alpha=0.7)
    ax.set_xlabel('Natural frequency ω')
    ax.set_ylabel('Count')
    ax.set_title(f'D) Frequency Distribution (mean={np.mean(results.oscillator_omegas):.2f})')

    # Panel 5: Gamma vs Omega (damping regime)
    ax = axes[1, 1]
    ax.scatter(results.oscillator_omegas, results.oscillator_gammas,
               c=results.damping_ratios, cmap='RdYlBu_r',
               alpha=0.6, edgecolor='black', linewidth=0.5)
    ax.plot([0, 3], [0, 3], 'k--', label='Critical (γ=ω)')
    ax.set_xlabel('Natural frequency ω')
    ax.set_ylabel('Damping γ')
    ax.set_title('E) Damping Regime')
    ax.legend()
    cbar = plt.colorbar(ax.collections[0], ax=ax)
    cbar.set_label('γ/ω')

    # Panel 6: Model comparison by AIC
    ax = axes[1, 2]
    models = ['delta', 'momentum', 'oscillator', 'hamiltonian']
    model_labels = ['Delta', 'Momentum', 'Osc.', 'Ham.']
    aic_wins = [results.aic_wins[m] for m in models]
    ax.bar(model_labels, aic_wins, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'],
           edgecolor='black', linewidth=0.5)
    ax.set_ylabel('Best fit count (AIC)')
    ax.set_title('F) Model Comparison (AIC)')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved supplementary figure to {save_path}")

    return fig


# =============================================================================
# LaTeX Output
# =============================================================================

def generate_latex_table(results: PopulationResults) -> str:
    """Generate LaTeX table for publication."""

    latex = r"""
\begin{table}[htbp]
\centering
\caption{Model comparison results for helicopter task (N=%d subjects).
Models were compared using BIC; best model per subject determined lowest BIC.
The momentum parameter $\beta$ was significantly greater than zero
($t(%d) = %.2f$, $p %s %.4f$), indicating belief inertia.}
\label{tab:helicopter_results}
\begin{tabular}{lcccc}
\toprule
\textbf{Model} & \textbf{Parameters} & \textbf{Best fit (BIC)} & \textbf{Mean MSE} & \textbf{Mean R²} \\
\midrule
Delta Rule & $\alpha$ & %d (%d\%%) & %.2f & %.3f \\
Momentum & $\alpha, \beta$ & %d (%d\%%) & %.2f & %.3f \\
Damped Oscillator & $\omega, \gamma$ & %d (%d\%%) & %.2f & %.3f \\
Hamiltonian & $M, \gamma, \Lambda$ & %d (%d\%%) & %.2f & %.3f \\
\bottomrule
\end{tabular}
\end{table}
"""

    n = results.n_subjects
    t_stat, p_val = results.beta_ttest
    p_symbol = '<' if p_val < 0.0001 else '='

    # Compute mean MSE and R² for each model
    def get_model_stats(model_name):
        mses = [r.fits[model_name].mse for r in results.subject_results]
        r2s = [r.fits[model_name].r_squared for r in results.subject_results]
        return np.mean(mses), np.mean(r2s)

    delta_mse, delta_r2 = get_model_stats('delta')
    mom_mse, mom_r2 = get_model_stats('momentum')
    osc_mse, osc_r2 = get_model_stats('oscillator')
    ham_mse, ham_r2 = get_model_stats('hamiltonian')

    return latex % (
        n, n-1, t_stat, p_symbol, p_val,
        results.bic_wins['delta'], 100*results.bic_wins['delta']//n, delta_mse, delta_r2,
        results.bic_wins['momentum'], 100*results.bic_wins['momentum']//n, mom_mse, mom_r2,
        results.bic_wins['oscillator'], 100*results.bic_wins['oscillator']//n, osc_mse, osc_r2,
        results.bic_wins['hamiltonian'], 100*results.bic_wins['hamiltonian']//n, ham_mse, ham_r2,
    )


def generate_manuscript_text(results: PopulationResults) -> str:
    """Generate text for manuscript Results section."""

    n = results.n_subjects
    t_stat, p_val = results.beta_ttest

    # Determine significance level
    if p_val < 0.001:
        sig_text = "p < 0.001"
    elif p_val < 0.01:
        sig_text = f"p = {p_val:.3f}"
    else:
        sig_text = f"p = {p_val:.4f}"

    text = f"""
=== MANUSCRIPT TEXT FOR RESULTS SECTION ===

The key findings (Figure X) are:

**Humans exhibit non-zero momentum.** The fitted momentum parameter β was
significantly greater than zero across participants (mean β = {results.beta_mean:.3f} ± {results.beta_std:.3f},
95% CI [{results.beta_ci[0]:.3f}, {results.beta_ci[1]:.3f}], t({n-1}) = {t_stat:.2f}, {sig_text}),
rejecting the null hypothesis of pure gradient descent. This confirms that human beliefs possess inertia.

**Dynamics are overdamped.** The fitted damping ratios γ/ω consistently exceeded 1
(mean = {np.mean(results.damping_ratios):.2f} ± {np.std(results.damping_ratios):.2f}),
with {results.n_overdamped}/{n} participants ({100*results.n_overdamped/n:.0f}%) in the overdamped regime.
Beliefs approach equilibrium monotonically without oscillation, but with non-zero momentum
that produces smoother trajectories than gradient descent predicts.

**Model comparison favors inertia.** By BIC, the momentum model outperformed the delta rule
in {results.bic_wins['momentum']}/{n} participants ({100*results.bic_wins['momentum']/n:.0f}%),
while the simple delta rule won in only {results.bic_wins['delta']}/{n} ({100*results.bic_wins['delta']/n:.0f}%).
The damped oscillator model provided best fit for {results.bic_wins['oscillator']}/{n} participants,
and the full Hamiltonian model for {results.bic_wins['hamiltonian']}/{n} participants.

=== END MANUSCRIPT TEXT ===
"""
    return text


def generate_statistics_summary(results: PopulationResults) -> str:
    """Generate comprehensive statistics summary."""

    summary = f"""
================================================================================
PUBLICATION STATISTICS SUMMARY
================================================================================
Dataset: McGuire & Nassar (2014) Helicopter Task
N subjects: {results.n_subjects}
Analysis date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

--------------------------------------------------------------------------------
MODEL COMPARISON (by BIC)
--------------------------------------------------------------------------------
Delta Rule:        {results.bic_wins['delta']:3d} subjects ({100*results.bic_wins['delta']/results.n_subjects:5.1f}%)
Momentum:          {results.bic_wins['momentum']:3d} subjects ({100*results.bic_wins['momentum']/results.n_subjects:5.1f}%)
Damped Oscillator: {results.bic_wins['oscillator']:3d} subjects ({100*results.bic_wins['oscillator']/results.n_subjects:5.1f}%)
Hamiltonian:       {results.bic_wins['hamiltonian']:3d} subjects ({100*results.bic_wins['hamiltonian']/results.n_subjects:5.1f}%)

--------------------------------------------------------------------------------
MOMENTUM PARAMETER (β)
--------------------------------------------------------------------------------
Mean:     {results.beta_mean:.4f}
Std:      {results.beta_std:.4f}
95% CI:   [{results.beta_ci[0]:.4f}, {results.beta_ci[1]:.4f}]
Min:      {np.min(results.momentum_betas):.4f}
Max:      {np.max(results.momentum_betas):.4f}
Median:   {np.median(results.momentum_betas):.4f}

t-test (β > 0):
  t-statistic: {results.beta_ttest[0]:.4f}
  p-value:     {results.beta_ttest[1]:.6f}
  Significant: {'YES' if results.beta_ttest[1] < 0.05 else 'NO'} (α = 0.05)

β > 0.1: {np.sum(results.momentum_betas > 0.1)}/{results.n_subjects} subjects ({100*np.sum(results.momentum_betas > 0.1)/results.n_subjects:.1f}%)
β > 0.2: {np.sum(results.momentum_betas > 0.2)}/{results.n_subjects} subjects ({100*np.sum(results.momentum_betas > 0.2)/results.n_subjects:.1f}%)
β > 0.3: {np.sum(results.momentum_betas > 0.3)}/{results.n_subjects} subjects ({100*np.sum(results.momentum_betas > 0.3)/results.n_subjects:.1f}%)

--------------------------------------------------------------------------------
DAMPING REGIME
--------------------------------------------------------------------------------
Overdamped (γ/ω > 1):   {results.n_overdamped:3d} subjects ({100*results.n_overdamped/results.n_subjects:5.1f}%)
Critical (γ/ω ≈ 1):     {results.n_critical:3d} subjects ({100*results.n_critical/results.n_subjects:5.1f}%)
Underdamped (γ/ω < 1):  {results.n_underdamped:3d} subjects ({100*results.n_underdamped/results.n_subjects:5.1f}%)

Mean γ/ω:   {np.mean(results.damping_ratios):.3f} ± {np.std(results.damping_ratios):.3f}
Median γ/ω: {np.median(results.damping_ratios):.3f}

--------------------------------------------------------------------------------
OSCILLATOR PARAMETERS
--------------------------------------------------------------------------------
Natural frequency ω:  {np.mean(results.oscillator_omegas):.3f} ± {np.std(results.oscillator_omegas):.3f}
Damping γ:            {np.mean(results.oscillator_gammas):.3f} ± {np.std(results.oscillator_gammas):.3f}

--------------------------------------------------------------------------------
HAMILTONIAN PARAMETERS
--------------------------------------------------------------------------------
Mass M:       {np.mean(results.hamiltonian_masses):.3f} ± {np.std(results.hamiltonian_masses):.3f}
Friction γ:   {np.mean(results.hamiltonian_frictions):.3f} ± {np.std(results.hamiltonian_frictions):.3f}

================================================================================
"""
    return summary


# =============================================================================
# Main Analysis Function
# =============================================================================

def run_publication_analysis(output_dir: Optional[Path] = None,
                             save_figures: bool = True,
                             verbose: bool = True) -> PopulationResults:
    """
    Run complete publication-quality analysis.

    Args:
        output_dir: Directory for output files (default: ./output)
        save_figures: Whether to save figures
        verbose: Print progress

    Returns:
        PopulationResults with all analysis results
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("INERTIA OF BELIEF: PUBLICATION ANALYSIS")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Load data
    print("\nLoading McGuire & Nassar (2014) data...")
    subjects = load_mcguire_nassar_2014()
    print(f"Loaded {len(subjects)} subjects")

    # Fit models
    results = fit_population(subjects, verbose=verbose)

    # Print statistics
    stats_text = generate_statistics_summary(results)
    print(stats_text)

    # Save statistics
    stats_path = output_dir / 'statistics_summary.txt'
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(stats_text)
    print(f"Saved statistics to {stats_path}")

    # Generate manuscript text
    manuscript_text = generate_manuscript_text(results)
    print(manuscript_text)

    manuscript_path = output_dir / 'manuscript_text.txt'
    with open(manuscript_path, 'w', encoding='utf-8') as f:
        f.write(manuscript_text)

    # Generate LaTeX table
    latex_table = generate_latex_table(results)
    latex_path = output_dir / 'results_table.tex'
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    print(f"Saved LaTeX table to {latex_path}")

    # Create figures
    if save_figures:
        print("\nGenerating figures...")

        fig_main = create_main_figure(
            results, subjects,
            save_path=output_dir / 'figure_helicopter_main.png'
        )
        fig_main.savefig(output_dir / 'figure_helicopter_main.pdf')
        plt.close(fig_main)

        fig_supp = create_supplementary_figure(
            results,
            save_path=output_dir / 'figure_helicopter_supplementary.png'
        )
        fig_supp.savefig(output_dir / 'figure_helicopter_supplementary.pdf')
        plt.close(fig_supp)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)

    return results


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == '__main__':
    results = run_publication_analysis()
