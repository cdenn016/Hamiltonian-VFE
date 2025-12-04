# -*- coding: utf-8 -*-
"""
Corrected Model Comparison
==========================

Properly aligns model predictions with human data.

Key insight:
  - Human prediction[t] is made AFTER seeing outcome[t-1]
  - So we compare model belief after seeing outcome[t-1] to human prediction[t]

Author: Hamiltonian-VFE Team
Date: December 2025
"""

import sys
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from scipy.optimize import minimize

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
# Simple Delta-Rule Models
# =============================================================================

def run_delta_rule(observations: np.ndarray,
                   learning_rate: float,
                   initial_belief: float = None) -> np.ndarray:
    """
    Simple delta rule: belief[t+1] = belief[t] + lr * (obs[t] - belief[t])

    Returns array of beliefs AFTER seeing each observation.
    """
    n = len(observations)
    beliefs = np.zeros(n + 1)
    beliefs[0] = initial_belief if initial_belief is not None else observations[0]

    for t in range(n):
        error = observations[t] - beliefs[t]
        beliefs[t + 1] = beliefs[t] + learning_rate * error

    return beliefs


def run_momentum_rule(observations: np.ndarray,
                      learning_rate: float,
                      momentum: float,
                      initial_belief: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Delta rule with momentum:
        velocity[t+1] = momentum * velocity[t] + lr * error[t]
        belief[t+1] = belief[t] + velocity[t+1]

    Returns beliefs and velocities.
    """
    n = len(observations)
    beliefs = np.zeros(n + 1)
    velocities = np.zeros(n + 1)

    beliefs[0] = initial_belief if initial_belief is not None else observations[0]
    velocities[0] = 0

    for t in range(n):
        error = observations[t] - beliefs[t]
        velocities[t + 1] = momentum * velocities[t] + learning_rate * error
        beliefs[t + 1] = beliefs[t] + velocities[t + 1]

    return beliefs, velocities


def run_hamiltonian(observations: np.ndarray,
                    mass: float,
                    friction: float,
                    obs_precision: float = 1.0,
                    initial_belief: float = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Hamiltonian dynamics:
        dp/dt = -dV/dmu - friction * p = obs_precision * error - friction * p
        dmu/dt = p / mass

    Using leapfrog integration with dt=1.
    """
    n = len(observations)
    beliefs = np.zeros(n + 1)
    momenta = np.zeros(n + 1)

    beliefs[0] = initial_belief if initial_belief is not None else observations[0]
    momenta[0] = 0

    dt = 1.0

    for t in range(n):
        error = observations[t] - beliefs[t]
        force = obs_precision * error

        # Leapfrog
        p_half = momenta[t] + 0.5 * dt * (force - friction * momenta[t])
        beliefs[t + 1] = beliefs[t] + dt * p_half / mass

        new_error = observations[t] - beliefs[t + 1]
        new_force = obs_precision * new_error
        momenta[t + 1] = p_half + 0.5 * dt * (new_force - friction * p_half)

    return beliefs, momenta


# =============================================================================
# Fitting Functions
# =============================================================================

@dataclass
class FitResult:
    model: str
    params: Dict[str, float]
    mse_belief: float      # MSE on beliefs
    mse_update: float      # MSE on updates
    corr_belief: float     # Correlation of beliefs
    corr_update: float     # Correlation of updates
    r_squared: float       # R² for belief prediction


def fit_delta_rule(subject: SubjectData) -> FitResult:
    """Fit simple delta rule to subject."""
    arrays = subject.get_arrays()
    observations = arrays['outcome']
    human_beliefs = arrays['prediction']

    # Human updates (properly lagged)
    human_updates = np.diff(human_beliefs)

    def loss(params):
        lr = params[0]
        if lr <= 0 or lr > 1.5:
            return 1e10

        model_beliefs = run_delta_rule(observations, lr, human_beliefs[0])

        # Compare model_beliefs[1:] to human_beliefs[1:]
        # (after seeing first observation)
        mse = np.mean((model_beliefs[1:-1] - human_beliefs[1:])**2)
        return mse

    result = minimize(loss, x0=[0.3], bounds=[(0.01, 1.5)], method='L-BFGS-B')
    best_lr = result.x[0]

    # Get predictions with best params
    model_beliefs = run_delta_rule(observations, best_lr, human_beliefs[0])
    model_updates = np.diff(model_beliefs[:-1])  # Align with human_updates

    # Metrics
    belief_residuals = model_beliefs[1:-1] - human_beliefs[1:]
    mse_belief = np.mean(belief_residuals**2)

    valid = np.isfinite(model_updates) & np.isfinite(human_updates)
    mse_update = np.mean((model_updates[valid] - human_updates[valid])**2)

    corr_belief = np.corrcoef(model_beliefs[1:-1], human_beliefs[1:])[0, 1]
    corr_update = np.corrcoef(model_updates[valid], human_updates[valid])[0, 1]

    ss_res = np.sum(belief_residuals**2)
    ss_tot = np.sum((human_beliefs[1:] - np.mean(human_beliefs[1:]))**2)
    r_squared = 1 - ss_res / ss_tot

    return FitResult(
        model='delta_rule',
        params={'learning_rate': best_lr},
        mse_belief=mse_belief,
        mse_update=mse_update,
        corr_belief=corr_belief,
        corr_update=corr_update,
        r_squared=r_squared
    )


def fit_momentum(subject: SubjectData) -> FitResult:
    """Fit delta rule with momentum."""
    arrays = subject.get_arrays()
    observations = arrays['outcome']
    human_beliefs = arrays['prediction']
    human_updates = np.diff(human_beliefs)

    def loss(params):
        lr, mom = params
        if lr <= 0 or lr > 1.5 or mom < 0 or mom > 0.99:
            return 1e10

        model_beliefs, _ = run_momentum_rule(observations, lr, mom, human_beliefs[0])
        mse = np.mean((model_beliefs[1:-1] - human_beliefs[1:])**2)
        return mse

    result = minimize(loss, x0=[0.3, 0.3],
                     bounds=[(0.01, 1.5), (0.0, 0.99)],
                     method='L-BFGS-B')
    best_lr, best_mom = result.x

    model_beliefs, velocities = run_momentum_rule(observations, best_lr, best_mom, human_beliefs[0])
    model_updates = np.diff(model_beliefs[:-1])

    belief_residuals = model_beliefs[1:-1] - human_beliefs[1:]
    mse_belief = np.mean(belief_residuals**2)

    valid = np.isfinite(model_updates) & np.isfinite(human_updates)
    mse_update = np.mean((model_updates[valid] - human_updates[valid])**2)

    corr_belief = np.corrcoef(model_beliefs[1:-1], human_beliefs[1:])[0, 1]
    corr_update = np.corrcoef(model_updates[valid], human_updates[valid])[0, 1]

    ss_res = np.sum(belief_residuals**2)
    ss_tot = np.sum((human_beliefs[1:] - np.mean(human_beliefs[1:]))**2)
    r_squared = 1 - ss_res / ss_tot

    return FitResult(
        model='momentum',
        params={'learning_rate': best_lr, 'momentum': best_mom},
        mse_belief=mse_belief,
        mse_update=mse_update,
        corr_belief=corr_belief,
        corr_update=corr_update,
        r_squared=r_squared
    )


def fit_hamiltonian(subject: SubjectData) -> FitResult:
    """Fit Hamiltonian dynamics."""
    arrays = subject.get_arrays()
    observations = arrays['outcome']
    human_beliefs = arrays['prediction']
    human_updates = np.diff(human_beliefs)

    def loss(params):
        mass, friction = params
        if mass <= 0.1 or friction < 0:
            return 1e10

        model_beliefs, _ = run_hamiltonian(observations, mass, friction,
                                           obs_precision=0.01,
                                           initial_belief=human_beliefs[0])
        mse = np.mean((model_beliefs[1:-1] - human_beliefs[1:])**2)
        return mse

    result = minimize(loss, x0=[1.0, 0.5],
                     bounds=[(0.1, 20.0), (0.0, 5.0)],
                     method='L-BFGS-B')
    best_mass, best_friction = result.x

    model_beliefs, momenta = run_hamiltonian(observations, best_mass, best_friction,
                                             obs_precision=0.01,
                                             initial_belief=human_beliefs[0])
    model_updates = np.diff(model_beliefs[:-1])

    belief_residuals = model_beliefs[1:-1] - human_beliefs[1:]
    mse_belief = np.mean(belief_residuals**2)

    valid = np.isfinite(model_updates) & np.isfinite(human_updates)
    mse_update = np.mean((model_updates[valid] - human_updates[valid])**2)

    corr_belief = np.corrcoef(model_beliefs[1:-1], human_beliefs[1:])[0, 1]
    corr_update = np.corrcoef(model_updates[valid], human_updates[valid])[0, 1]

    ss_res = np.sum(belief_residuals**2)
    ss_tot = np.sum((human_beliefs[1:] - np.mean(human_beliefs[1:]))**2)
    r_squared = 1 - ss_res / ss_tot

    return FitResult(
        model='hamiltonian',
        params={'mass': best_mass, 'friction': best_friction},
        mse_belief=mse_belief,
        mse_update=mse_update,
        corr_belief=corr_belief,
        corr_update=corr_update,
        r_squared=r_squared
    )


# =============================================================================
# Main Analysis
# =============================================================================

def run_corrected_analysis():
    """Run the corrected model comparison."""
    print("="*70)
    print("CORRECTED MODEL COMPARISON")
    print("="*70)

    subjects = load_mcguire_nassar_2014()

    results = {
        'delta_rule': [],
        'momentum': [],
        'hamiltonian': []
    }

    print(f"\nFitting {len(subjects)} subjects...\n")
    print(f"{'Subj':>4} | {'Delta LR':>8} | {'Mom LR':>7} {'Mom β':>6} | {'Ham M':>6} {'Ham γ':>6} | {'Best Model':>12}")
    print("-"*70)

    for subj_id, subject in subjects.items():
        delta_fit = fit_delta_rule(subject)
        mom_fit = fit_momentum(subject)
        ham_fit = fit_hamiltonian(subject)

        results['delta_rule'].append(delta_fit)
        results['momentum'].append(mom_fit)
        results['hamiltonian'].append(ham_fit)

        # Find best
        fits = [('Delta', delta_fit), ('Momentum', mom_fit), ('Hamiltonian', ham_fit)]
        best = min(fits, key=lambda x: x[1].mse_belief)

        print(f"{subj_id:4d} | {delta_fit.params['learning_rate']:8.3f} | "
              f"{mom_fit.params['learning_rate']:7.3f} {mom_fit.params['momentum']:6.3f} | "
              f"{ham_fit.params['mass']:6.2f} {ham_fit.params['friction']:6.2f} | "
              f"{best[0]:>12}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    for model_name, fits in results.items():
        mses = [f.mse_belief for f in fits]
        r2s = [f.r_squared for f in fits]
        corrs = [f.corr_belief for f in fits]

        print(f"\n{model_name.upper()}:")
        print(f"  Mean MSE (belief):  {np.mean(mses):.2f} ± {np.std(mses):.2f}")
        print(f"  Mean R²:            {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
        print(f"  Mean correlation:   {np.mean(corrs):.4f}")

    # Count wins
    delta_wins = 0
    mom_wins = 0
    ham_wins = 0

    for i in range(len(subjects)):
        mses = [
            results['delta_rule'][i].mse_belief,
            results['momentum'][i].mse_belief,
            results['hamiltonian'][i].mse_belief
        ]
        winner = np.argmin(mses)
        if winner == 0:
            delta_wins += 1
        elif winner == 1:
            mom_wins += 1
        else:
            ham_wins += 1

    print(f"\n{'MODEL COMPARISON':^70}")
    print("-"*70)
    print(f"Delta Rule wins:   {delta_wins}/{len(subjects)} ({100*delta_wins/len(subjects):.1f}%)")
    print(f"Momentum wins:     {mom_wins}/{len(subjects)} ({100*mom_wins/len(subjects):.1f}%)")
    print(f"Hamiltonian wins:  {ham_wins}/{len(subjects)} ({100*ham_wins/len(subjects):.1f}%)")

    # Fitted parameter distributions
    print(f"\n{'FITTED PARAMETERS':^70}")
    print("-"*70)

    delta_lrs = [f.params['learning_rate'] for f in results['delta_rule']]
    print(f"Delta LR:       {np.mean(delta_lrs):.3f} ± {np.std(delta_lrs):.3f}")

    mom_lrs = [f.params['learning_rate'] for f in results['momentum']]
    mom_betas = [f.params['momentum'] for f in results['momentum']]
    print(f"Momentum LR:    {np.mean(mom_lrs):.3f} ± {np.std(mom_lrs):.3f}")
    print(f"Momentum β:     {np.mean(mom_betas):.3f} ± {np.std(mom_betas):.3f}")

    ham_masses = [f.params['mass'] for f in results['hamiltonian']]
    ham_frictions = [f.params['friction'] for f in results['hamiltonian']]
    print(f"Hamiltonian M:  {np.mean(ham_masses):.2f} ± {np.std(ham_masses):.2f}")
    print(f"Hamiltonian γ:  {np.mean(ham_frictions):.2f} ± {np.std(ham_frictions):.2f}")

    return results


if __name__ == '__main__':
    run_corrected_analysis()
