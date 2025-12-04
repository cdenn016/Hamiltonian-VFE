#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Inertia of Belief Analysis
====================================

Fixes Hamiltonian formulation issues and analyzes all datasets:
1. McGuireNassar2014 (32 subjects, behavioral)
2. jNeuroBehav (behavioral)
3. Pupil paper (behavioral + pupil)

Key fixes to Hamiltonian model:
- Proper leapfrog integration (symplectic)
- Correct force = -dF/dμ where F = 0.5*prec*(μ-obs)^2
- Free precision as fitted parameter
- Smaller step size dt

Author: Hamiltonian-VFE Team
Date: December 2025
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.optimize import minimize
from scipy import stats

# Handle both module execution and direct script execution
try:
    from .data_loader import (
        load_mcguire_nassar_2014,
        load_jneurobehav,
        load_pupil_data
    )
except ImportError:
    _this_dir = Path(__file__).parent
    _project_root = _this_dir.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from empirical.helicopter.data_loader import (
        load_mcguire_nassar_2014,
        load_jneurobehav,
        load_pupil_data
    )


# =============================================================================
# Models - Core Update Rules
# =============================================================================

def run_delta_rule(observations, lr, init):
    """
    Simple delta rule: belief += lr * (obs - belief)

    This is equivalent to:
    - A Kalman filter with fixed gain
    - Gradient descent on prediction error with learning rate lr
    """
    n = len(observations)
    beliefs = np.zeros(n + 1)
    beliefs[0] = init
    for t in range(n):
        beliefs[t + 1] = beliefs[t] + lr * (observations[t] - beliefs[t])
    return beliefs


def run_momentum_rule(observations, lr, beta, init):
    """
    Delta rule with momentum: velocity = beta*velocity + lr*error

    This adds inertia - the update "carries over" from previous updates.
    When beta=0, reduces to delta rule.
    When beta>0, updates overshoot and oscillate.
    """
    n = len(observations)
    beliefs = np.zeros(n + 1)
    velocity = np.zeros(n + 1)
    beliefs[0] = init

    for t in range(n):
        error = observations[t] - beliefs[t]
        velocity[t + 1] = beta * velocity[t] + lr * error
        beliefs[t + 1] = beliefs[t] + velocity[t + 1]

    return beliefs, velocity


def run_hamiltonian_proper(observations, mass, friction, prec, init, n_substeps=5):
    """
    Proper Hamiltonian dynamics with leapfrog integration.

    Hamiltonian: H = p²/(2m) + (prec/2)(μ - obs)²

    Equations of motion:
        dμ/dt = p/m          (velocity = momentum/mass)
        dp/dt = -prec*(μ - obs) - γ*p   (force from potential + friction)

    Leapfrog integrator (symplectic, energy-conserving):
        p_{1/2} = p_0 + (dt/2) * force(μ_0)
        μ_1 = μ_0 + dt * p_{1/2}/m
        p_1 = p_{1/2} + (dt/2) * force(μ_1) - γ*dt*p_{1/2}

    Args:
        observations: Sequence of observations
        mass: Inertial mass (higher = slower response)
        friction: Damping coefficient (higher = faster energy dissipation)
        prec: Precision (inverse variance) of likelihood
        init: Initial belief
        n_substeps: Number of leapfrog steps per observation (for stability)

    Returns:
        beliefs: Belief trajectory
        momentum: Momentum trajectory
    """
    n = len(observations)
    dt = 1.0 / n_substeps  # Substep size

    beliefs = np.zeros(n + 1)
    momentum = np.zeros(n + 1)
    beliefs[0] = init

    for t in range(n):
        obs = observations[t]
        mu = beliefs[t]
        p = momentum[t]

        # Leapfrog integration over substeps
        for _ in range(n_substeps):
            # Force = -dV/dμ = -prec*(μ - obs)
            force = -prec * (mu - obs)

            # Half step for momentum
            p = p + 0.5 * dt * force

            # Full step for position
            mu = mu + dt * p / mass

            # Update force at new position
            force = -prec * (mu - obs)

            # Half step for momentum with friction
            p = p + 0.5 * dt * force
            p = p * (1 - friction * dt)  # Apply friction

        beliefs[t + 1] = mu
        momentum[t + 1] = p

    return beliefs, momentum


def run_damped_oscillator(observations, omega, gamma, init):
    """
    Damped harmonic oscillator belief dynamics.

    d²μ/dt² + 2γ dμ/dt + ω²(μ - obs) = 0

    This is equivalent to Hamiltonian with:
    - mass = 1
    - spring constant k = ω²
    - friction = 2γ

    Args:
        observations: Sequence of observations
        omega: Natural frequency (higher = faster oscillation)
        gamma: Damping ratio (γ < ω: underdamped, γ > ω: overdamped)
        init: Initial belief

    Returns:
        beliefs: Belief trajectory
        velocity: Velocity trajectory
    """
    n = len(observations)
    dt = 0.1  # Small step for accuracy
    n_substeps = 10

    beliefs = np.zeros(n + 1)
    velocity = np.zeros(n + 1)
    beliefs[0] = init

    for t in range(n):
        obs = observations[t]
        mu = beliefs[t]
        v = velocity[t]

        for _ in range(n_substeps):
            # Second-order ODE: a = -ω²(μ-obs) - 2γv
            accel = -omega**2 * (mu - obs) - 2*gamma*v

            # Velocity Verlet
            mu_new = mu + v*dt + 0.5*accel*dt**2
            accel_new = -omega**2 * (mu_new - obs) - 2*gamma*v  # Estimate
            v = v + 0.5*(accel + accel_new)*dt
            mu = mu_new

        beliefs[t + 1] = mu
        velocity[t + 1] = v

    return beliefs, velocity


# =============================================================================
# Fitting Functions
# =============================================================================

def fit_all_models(obs, human, verbose=False):
    """
    Fit delta, momentum, and improved Hamiltonian models.

    Args:
        obs: Observations array
        human: Human predictions array (aligned: human[t] is prediction BEFORE seeing obs[t])

    Returns:
        Dictionary with fit results for each model
    """
    # Compute total sum of squares for R²
    ss_tot = np.sum((human[1:] - np.mean(human[1:]))**2)
    if ss_tot < 1e-10:
        ss_tot = 1.0  # Avoid division by zero

    results = {}

    # 1. Delta Rule (1 parameter: lr)
    def delta_loss(p):
        lr = p[0]
        if lr <= 0 or lr > 2.0:
            return 1e10
        pred = run_delta_rule(obs, lr, human[0])
        return np.mean((pred[1:-1] - human[1:])**2)

    res = minimize(delta_loss, [0.5], bounds=[(0.01, 2.0)], method='L-BFGS-B')
    lr_delta = res.x[0]
    pred_delta = run_delta_rule(obs, lr_delta, human[0])
    mse_delta = np.mean((pred_delta[1:-1] - human[1:])**2)
    ss_res_delta = np.sum((pred_delta[1:-1] - human[1:])**2)
    r2_delta = 1 - ss_res_delta / ss_tot

    results['delta'] = {
        'lr': lr_delta,
        'mse': mse_delta,
        'r2': r2_delta,
        'n_params': 1,
        'bic': len(human) * np.log(mse_delta) + 1 * np.log(len(human))
    }

    # 2. Momentum Rule (2 parameters: lr, beta)
    def momentum_loss(p):
        lr, beta = p
        if lr <= 0 or lr > 2.0 or beta < 0 or beta > 0.99:
            return 1e10
        pred, _ = run_momentum_rule(obs, lr, beta, human[0])
        return np.mean((pred[1:-1] - human[1:])**2)

    res = minimize(momentum_loss, [0.5, 0.3],
                   bounds=[(0.01, 2.0), (0.0, 0.99)], method='L-BFGS-B')
    lr_mom, beta = res.x
    pred_mom, vel = run_momentum_rule(obs, lr_mom, beta, human[0])
    mse_mom = np.mean((pred_mom[1:-1] - human[1:])**2)
    ss_res_mom = np.sum((pred_mom[1:-1] - human[1:])**2)
    r2_mom = 1 - ss_res_mom / ss_tot

    results['momentum'] = {
        'lr': lr_mom,
        'beta': beta,
        'mse': mse_mom,
        'r2': r2_mom,
        'n_params': 2,
        'bic': len(human) * np.log(mse_mom) + 2 * np.log(len(human))
    }

    # 3. Improved Hamiltonian (3 parameters: mass, friction, precision)
    def hamiltonian_loss(p):
        mass, friction, prec = p
        if mass <= 0.01 or friction < 0 or prec <= 0:
            return 1e10
        try:
            pred, _ = run_hamiltonian_proper(obs, mass, friction, prec, human[0], n_substeps=5)
            mse = np.mean((pred[1:-1] - human[1:])**2)
            if not np.isfinite(mse):
                return 1e10
            return mse
        except:
            return 1e10

    # Try multiple starting points
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
        mass, friction, prec = best_ham
        pred_ham, mom = run_hamiltonian_proper(obs, mass, friction, prec, human[0], n_substeps=5)
        mse_ham = np.mean((pred_ham[1:-1] - human[1:])**2)
        ss_res_ham = np.sum((pred_ham[1:-1] - human[1:])**2)
        r2_ham = 1 - ss_res_ham / ss_tot
    else:
        mass, friction, prec = 1.0, 0.5, 0.1
        mse_ham = 1e10
        r2_ham = -1

    results['hamiltonian'] = {
        'mass': mass,
        'friction': friction,
        'precision': prec,
        'mse': mse_ham,
        'r2': r2_ham,
        'n_params': 3,
        'bic': len(human) * np.log(max(mse_ham, 1e-10)) + 3 * np.log(len(human))
    }

    # 4. Damped Oscillator (2 parameters: omega, gamma)
    def oscillator_loss(p):
        omega, gamma = p
        if omega <= 0 or gamma < 0:
            return 1e10
        try:
            pred, _ = run_damped_oscillator(obs, omega, gamma, human[0])
            mse = np.mean((pred[1:-1] - human[1:])**2)
            if not np.isfinite(mse):
                return 1e10
            return mse
        except:
            return 1e10

    res = minimize(oscillator_loss, [1.0, 0.5],
                   bounds=[(0.01, 5.0), (0.0, 5.0)], method='L-BFGS-B')
    omega, gamma = res.x
    pred_osc, vel_osc = run_damped_oscillator(obs, omega, gamma, human[0])
    mse_osc = np.mean((pred_osc[1:-1] - human[1:])**2)
    ss_res_osc = np.sum((pred_osc[1:-1] - human[1:])**2)
    r2_osc = 1 - ss_res_osc / ss_tot

    results['oscillator'] = {
        'omega': omega,
        'gamma': gamma,
        'mse': mse_osc,
        'r2': r2_osc,
        'n_params': 2,
        'bic': len(human) * np.log(max(mse_osc, 1e-10)) + 2 * np.log(len(human))
    }

    return results


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_mcguire_nassar():
    """Analyze McGuireNassar2014 dataset."""
    print("\n" + "="*70)
    print("DATASET: McGuireNassar2014 (32 subjects)")
    print("="*70)

    subjects = load_mcguire_nassar_2014()
    all_results = []

    print(f"\n{'Subj':>4} | {'Δ LR':>6} | {'M LR':>5} {'β':>5} | {'H mass':>6} {'γ':>5} | {'Osc ω':>5} {'γ':>5} | {'Winner':>10}")
    print("-"*80)

    for sid in sorted(subjects.keys()):
        subj = subjects[sid]
        arrays = subj.get_arrays()
        obs = arrays['outcome']
        human = arrays['prediction']

        fits = fit_all_models(obs, human)
        all_results.append({'sid': sid, 'fits': fits})

        # Determine winner by BIC
        bics = {k: v['bic'] for k, v in fits.items()}
        winner = min(bics, key=bics.get)

        d = fits['delta']
        m = fits['momentum']
        h = fits['hamiltonian']
        o = fits['oscillator']

        print(f"{sid:4d} | {d['lr']:6.3f} | {m['lr']:5.3f} {m['beta']:5.3f} | "
              f"{h['mass']:6.2f} {h['friction']:5.2f} | {o['omega']:5.2f} {o['gamma']:5.2f} | {winner:>10}")

    return all_results


def analyze_jneurobehav():
    """Analyze jNeuroBehav dataset."""
    print("\n" + "="*70)
    print("DATASET: jNeuroBehav")
    print("="*70)

    data = load_jneurobehav()

    # Get unique sessions
    sessions = np.unique(data['session'])
    print(f"\nFound {len(sessions)} sessions")

    all_results = []

    for sess in sessions[:10]:  # Analyze first 10 sessions
        mask = data['session'] == sess
        obs = data['outcome'][mask]
        human = data['prediction'][mask]

        if len(obs) < 20:  # Skip short sessions
            continue

        try:
            fits = fit_all_models(obs, human)
            all_results.append({'session': sess, 'fits': fits})

            bics = {k: v['bic'] for k, v in fits.items()}
            winner = min(bics, key=bics.get)

            print(f"Session {int(sess):3d}: β={fits['momentum']['beta']:.3f}, "
                  f"mass={fits['hamiltonian']['mass']:.2f}, winner={winner}")
        except Exception as e:
            print(f"Session {int(sess):3d}: Error - {e}")

    return all_results


def analyze_pupil_data():
    """Analyze pupil paper dataset."""
    print("\n" + "="*70)
    print("DATASET: Pupil Paper Behavioral Data")
    print("="*70)

    data = load_pupil_data()

    # Data structure is 2D: (subjects, trials)
    n_subjects = data['outcome'].shape[0]
    print(f"\nFound {n_subjects} subjects")

    all_results = []

    for subj_idx in range(min(n_subjects, 20)):  # Analyze first 20
        obs = data['outcome'][subj_idx, :].flatten()
        human = data['prediction'][subj_idx, :].flatten()

        # Remove NaN values
        valid = ~(np.isnan(obs) | np.isnan(human))
        obs = obs[valid]
        human = human[valid]

        if len(obs) < 20:
            continue

        try:
            fits = fit_all_models(obs, human)
            all_results.append({'subject': subj_idx, 'fits': fits})

            bics = {k: v['bic'] for k, v in fits.items()}
            winner = min(bics, key=bics.get)

            print(f"Subject {subj_idx:3d}: β={fits['momentum']['beta']:.3f}, "
                  f"mass={fits['hamiltonian']['mass']:.2f}, winner={winner}")
        except Exception as e:
            print(f"Subject {subj_idx:3d}: Error - {e}")

    return all_results


def summarize_results(results_list, dataset_name):
    """Summarize results for a dataset."""
    print(f"\n{'='*70}")
    print(f"SUMMARY: {dataset_name}")
    print("="*70)

    if not results_list:
        print("No results to summarize")
        return

    # Extract parameters
    betas = [r['fits']['momentum']['beta'] for r in results_list]
    delta_lrs = [r['fits']['delta']['lr'] for r in results_list]
    masses = [r['fits']['hamiltonian']['mass'] for r in results_list]
    frictions = [r['fits']['hamiltonian']['friction'] for r in results_list]

    # Model comparison by BIC
    delta_wins = momentum_wins = ham_wins = osc_wins = 0
    for r in results_list:
        bics = {k: v['bic'] for k, v in r['fits'].items()}
        winner = min(bics, key=bics.get)
        if winner == 'delta':
            delta_wins += 1
        elif winner == 'momentum':
            momentum_wins += 1
        elif winner == 'hamiltonian':
            ham_wins += 1
        else:
            osc_wins += 1

    n = len(results_list)

    print(f"\nModel Comparison (by BIC, n={n}):")
    print(f"  Delta Rule:       {delta_wins:3d} ({100*delta_wins/n:.1f}%)")
    print(f"  Momentum:         {momentum_wins:3d} ({100*momentum_wins/n:.1f}%)")
    print(f"  Hamiltonian:      {ham_wins:3d} ({100*ham_wins/n:.1f}%)")
    print(f"  Damped Oscillator:{osc_wins:3d} ({100*osc_wins/n:.1f}%)")

    print(f"\nKey Parameters:")
    print(f"  Mean β (momentum):     {np.mean(betas):.4f} ± {np.std(betas):.4f}")
    print(f"  β > 0.1:              {sum(b > 0.1 for b in betas)}/{n} subjects")
    print(f"  β > 0.3:              {sum(b > 0.3 for b in betas)}/{n} subjects")
    print(f"  Mean learning rate:    {np.mean(delta_lrs):.3f} ± {np.std(delta_lrs):.3f}")
    print(f"  Mean Hamiltonian mass: {np.mean(masses):.3f} ± {np.std(masses):.3f}")

    # Statistical test: Is β > 0?
    t_stat, p_val = stats.ttest_1samp(betas, 0)
    print(f"\nTest for inertia (β > 0):")
    print(f"  t = {t_stat:.3f}, p = {p_val:.4f}")
    if p_val < 0.05 and np.mean(betas) > 0.05:
        print(f"  → SIGNIFICANT evidence for inertia")
    else:
        print(f"  → NO meaningful evidence for inertia")

    return {
        'betas': betas,
        'delta_lrs': delta_lrs,
        'model_wins': {'delta': delta_wins, 'momentum': momentum_wins,
                       'hamiltonian': ham_wins, 'oscillator': osc_wins}
    }


# =============================================================================
# Main Analysis
# =============================================================================

def run_improved_analysis():
    """Run improved analysis on all datasets."""
    print("="*70)
    print("IMPROVED INERTIA OF BELIEF ANALYSIS")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nModels tested:")
    print(f"  1. Delta Rule (1 param): belief += lr * error")
    print(f"  2. Momentum (2 params):  velocity = β*velocity + lr*error")
    print(f"  3. Hamiltonian (3 params): proper leapfrog integration")
    print(f"  4. Damped Oscillator (2 params): d²μ/dt² + 2γ dμ/dt + ω²(μ-obs) = 0")

    # Analyze each dataset
    results = {}

    # 1. McGuireNassar2014
    try:
        mcguire_results = analyze_mcguire_nassar()
        results['mcguire'] = summarize_results(mcguire_results, 'McGuireNassar2014')
    except Exception as e:
        print(f"\nError analyzing McGuireNassar2014: {e}")

    # 2. jNeuroBehav
    try:
        jneuro_results = analyze_jneurobehav()
        results['jneuro'] = summarize_results(jneuro_results, 'jNeuroBehav')
    except Exception as e:
        print(f"\nError analyzing jNeuroBehav: {e}")

    # 3. Pupil data
    try:
        pupil_results = analyze_pupil_data()
        results['pupil'] = summarize_results(pupil_results, 'Pupil Paper')
    except Exception as e:
        print(f"\nError analyzing Pupil data: {e}")

    # Overall conclusion
    print("\n" + "="*70)
    print("OVERALL CONCLUSION")
    print("="*70)

    all_betas = []
    for key in results:
        if results[key] and 'betas' in results[key]:
            all_betas.extend(results[key]['betas'])

    if all_betas:
        print(f"\nAcross all datasets (n={len(all_betas)} subjects/sessions):")
        print(f"  Mean β = {np.mean(all_betas):.4f} ± {np.std(all_betas):.4f}")
        print(f"  Median β = {np.median(all_betas):.4f}")
        print(f"  β > 0.1: {sum(b > 0.1 for b in all_betas)}/{len(all_betas)} ({100*sum(b > 0.1 for b in all_betas)/len(all_betas):.1f}%)")

        if np.mean(all_betas) < 0.05:
            print(f"\n  → NO evidence for belief inertia/momentum")
            print(f"  → Simple delta rule explains human prediction behavior")
        elif np.mean(all_betas) < 0.15:
            print(f"\n  → WEAK evidence for belief inertia")
            print(f"  → Effect size is small (β < 0.15)")
        else:
            print(f"\n  → EVIDENCE for belief inertia")
            print(f"  → Humans show momentum in belief updates")

    return results


if __name__ == '__main__':
    results = run_improved_analysis()
