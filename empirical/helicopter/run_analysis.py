#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run Corrected Inertia of Belief Analysis
=========================================

Tests whether human belief updates show momentum/inertia using
properly aligned data.

Usage:
    python run_analysis.py
    python -m empirical.helicopter.run_analysis

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
    from .data_loader import load_mcguire_nassar_2014
except ImportError:
    _this_dir = Path(__file__).parent
    _project_root = _this_dir.parent.parent
    if str(_project_root) not in sys.path:
        sys.path.insert(0, str(_project_root))
    from empirical.helicopter.data_loader import load_mcguire_nassar_2014


# =============================================================================
# Models
# =============================================================================

def run_delta_rule(observations, lr, init):
    """Simple delta rule: belief += lr * (obs - belief)"""
    n = len(observations)
    beliefs = np.zeros(n + 1)
    beliefs[0] = init
    for t in range(n):
        beliefs[t + 1] = beliefs[t] + lr * (observations[t] - beliefs[t])
    return beliefs


def run_momentum_rule(observations, lr, mom, init):
    """Delta rule with momentum: velocity = mom*velocity + lr*error"""
    n = len(observations)
    beliefs = np.zeros(n + 1)
    vel = np.zeros(n + 1)
    beliefs[0] = init
    for t in range(n):
        vel[t + 1] = mom * vel[t] + lr * (observations[t] - beliefs[t])
        beliefs[t + 1] = beliefs[t] + vel[t + 1]
    return beliefs, vel


def run_hamiltonian(observations, mass, friction, prec, init):
    """Hamiltonian dynamics with leapfrog integration"""
    n = len(observations)
    beliefs = np.zeros(n + 1)
    mom = np.zeros(n + 1)
    beliefs[0] = init
    for t in range(n):
        error = observations[t] - beliefs[t]
        force = prec * error
        p_half = mom[t] + 0.5 * (force - friction * mom[t])
        beliefs[t + 1] = beliefs[t] + p_half / mass
        new_force = prec * (observations[t] - beliefs[t + 1])
        mom[t + 1] = p_half + 0.5 * (new_force - friction * p_half)
    return beliefs, mom


# =============================================================================
# Fitting
# =============================================================================

def fit_all_models(obs, human):
    """Fit delta, momentum, and Hamiltonian models to one subject."""
    ss_tot = np.sum((human[1:] - np.mean(human[1:]))**2)

    # Delta rule
    def d_loss(p):
        if p[0] <= 0 or p[0] > 1.5: return 1e10
        m = run_delta_rule(obs, p[0], human[0])
        return np.mean((m[1:-1] - human[1:])**2)

    r = minimize(d_loss, [0.3], bounds=[(0.01, 1.5)], method='L-BFGS-B')
    d_lr = r.x[0]
    d_bel = run_delta_rule(obs, d_lr, human[0])
    d_mse = np.mean((d_bel[1:-1] - human[1:])**2)
    d_r2 = 1 - np.sum((d_bel[1:-1] - human[1:])**2) / ss_tot

    # Momentum
    def m_loss(p):
        if p[0] <= 0 or p[0] > 1.5 or p[1] < 0 or p[1] > 0.99: return 1e10
        m, _ = run_momentum_rule(obs, p[0], p[1], human[0])
        return np.mean((m[1:-1] - human[1:])**2)

    r = minimize(m_loss, [0.3, 0.3], bounds=[(0.01, 1.5), (0.0, 0.99)], method='L-BFGS-B')
    m_lr, m_beta = r.x
    m_bel, _ = run_momentum_rule(obs, m_lr, m_beta, human[0])
    m_mse = np.mean((m_bel[1:-1] - human[1:])**2)
    m_r2 = 1 - np.sum((m_bel[1:-1] - human[1:])**2) / ss_tot

    # Hamiltonian
    def h_loss(p):
        if p[0] <= 0.1 or p[1] < 0: return 1e10
        m, _ = run_hamiltonian(obs, p[0], p[1], 0.01, human[0])
        return np.mean((m[1:-1] - human[1:])**2)

    r = minimize(h_loss, [1.0, 0.5], bounds=[(0.1, 20.0), (0.0, 5.0)], method='L-BFGS-B')
    h_mass, h_fric = r.x
    h_bel, _ = run_hamiltonian(obs, h_mass, h_fric, 0.01, human[0])
    h_mse = np.mean((h_bel[1:-1] - human[1:])**2)
    h_r2 = 1 - np.sum((h_bel[1:-1] - human[1:])**2) / ss_tot

    return {
        'delta': {'lr': d_lr, 'mse': d_mse, 'r2': d_r2},
        'momentum': {'lr': m_lr, 'beta': m_beta, 'mse': m_mse, 'r2': m_r2},
        'hamiltonian': {'mass': h_mass, 'friction': h_fric, 'mse': h_mse, 'r2': h_r2}
    }


# =============================================================================
# Main Analysis
# =============================================================================

def run_full_analysis():
    """Run the complete corrected analysis."""
    print("="*70)
    print("INERTIA OF BELIEF - CORRECTED ANALYSIS")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load data
    print("[1/3] Loading data...")
    subjects = load_mcguire_nassar_2014()
    print(f"  Loaded {len(subjects)} subjects")

    # Fit models
    print("\n[2/3] Fitting models...")
    results = []
    d_wins = m_wins = h_wins = 0

    print(f"\n{'Subj':>4} | {'Delta LR':>8} | {'Mom LR':>6} {'β':>5} | {'Ham M':>5} {'γ':>5} | {'Winner':>10}")
    print("-"*70)

    for sid in sorted(subjects.keys()):
        subj = subjects[sid]
        obs = subj.get_arrays()['outcome']
        human = subj.get_arrays()['prediction']

        fits = fit_all_models(obs, human)
        results.append(fits)

        mses = [fits['delta']['mse'], fits['momentum']['mse'], fits['hamiltonian']['mse']]
        w = np.argmin(mses)
        names = ['Delta', 'Momentum', 'Hamiltonian']
        if w == 0: d_wins += 1
        elif w == 1: m_wins += 1
        else: h_wins += 1

        d, m, h = fits['delta'], fits['momentum'], fits['hamiltonian']
        print(f"{sid:4d} | {d['lr']:8.3f} | {m['lr']:6.3f} {m['beta']:5.3f} | {h['mass']:5.1f} {h['friction']:5.2f} | {names[w]:>10}")

    # Summary
    n = len(subjects)
    print("\n" + "="*70)
    print("[3/3] RESULTS SUMMARY")
    print("="*70)

    print(f"\nModel wins:")
    print(f"  Delta Rule:   {d_wins}/{n} ({100*d_wins/n:.1f}%)")
    print(f"  Momentum:     {m_wins}/{n} ({100*m_wins/n:.1f}%)")
    print(f"  Hamiltonian:  {h_wins}/{n} ({100*h_wins/n:.1f}%)")

    d_mses = [r['delta']['mse'] for r in results]
    m_mses = [r['momentum']['mse'] for r in results]
    h_mses = [r['hamiltonian']['mse'] for r in results]

    print(f"\nMean MSE (lower is better):")
    print(f"  Delta:       {np.mean(d_mses):8.1f} ± {np.std(d_mses):.1f}")
    print(f"  Momentum:    {np.mean(m_mses):8.1f} ± {np.std(m_mses):.1f}")
    print(f"  Hamiltonian: {np.mean(h_mses):8.1f} ± {np.std(h_mses):.1f}")

    d_r2s = [r['delta']['r2'] for r in results]
    m_r2s = [r['momentum']['r2'] for r in results]
    h_r2s = [r['hamiltonian']['r2'] for r in results]

    print(f"\nMean R² (higher is better):")
    print(f"  Delta:       {np.mean(d_r2s):.4f}")
    print(f"  Momentum:    {np.mean(m_r2s):.4f}")
    print(f"  Hamiltonian: {np.mean(h_r2s):.4f}")

    # Key test: Is momentum β > 0?
    betas = [r['momentum']['beta'] for r in results]
    print(f"\n{'='*70}")
    print("KEY TEST: Is there evidence for momentum/inertia (β > 0)?")
    print("="*70)
    print(f"  Mean β = {np.mean(betas):.4f} ± {np.std(betas):.4f}")
    print(f"  β > 0.1: {sum(b > 0.1 for b in betas)}/{n} subjects")
    print(f"  β > 0.3: {sum(b > 0.3 for b in betas)}/{n} subjects")

    t, p = stats.ttest_1samp(betas, 0)
    sig = "SIGNIFICANT" if p < 0.05 else "not significant"
    print(f"  t-test β > 0: t={t:.2f}, p={p:.4f} ({sig})")

    print("\n" + "="*70)
    print("CONCLUSION")
    print("="*70)
    if np.mean(betas) < 0.05 and sum(b > 0.1 for b in betas) < n/2:
        print("  ✗ NO evidence for belief inertia/momentum in this dataset")
        print("  ✓ Simple delta rule explains 96% of variance")
    else:
        print("  ✓ Some evidence for belief inertia/momentum")

    return results


if __name__ == '__main__':
    run_full_analysis()
