#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Changepoint-Specific Inertia Analysis
=======================================

Tests the actual prediction of Hamiltonian inertia theory:

    τ ∝ Λ

Where:
    - τ = relaxation time after a changepoint (how fast beliefs adjust)
    - Λ = precision accumulated BEFORE the changepoint

This is fundamentally different from testing a constant momentum parameter.
A subject could show β ≈ 0 on average while still exhibiting τ ~ Λ scaling
at individual changepoints.

Method:
1. Identify each changepoint event
2. Compute Λ_before = pre-CP precision (function of trials since last CP)
3. Fit exponential decay to post-CP error trajectory: error(t) = A*exp(-t/τ) + C
4. Extract τ for that specific changepoint
5. Test: does τ scale with Λ_before?

Author: Hamiltonian-VFE Team
Date: December 2025
"""

import sys
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.optimize import curve_fit
from scipy import stats
import warnings

# Handle imports
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
# Precision Computation
# =============================================================================

def compute_precision_before_cp(errors_before, method='inverse_variance'):
    """
    Compute accumulated precision before a changepoint.

    Precision is the inverse of uncertainty. In the Hamiltonian framework,
    this corresponds to the "mass" of the belief.

    Args:
        errors_before: Array of prediction errors before the changepoint
        method: How to compute precision
            - 'inverse_variance': Λ = 1 / var(errors)
            - 'trial_count': Λ = n_trials (simple accumulation)
            - 'bayesian': Λ = n / σ² (Bayesian posterior precision)
            - 'recent': Use only last 10 trials

    Returns:
        precision: Scalar precision value
    """
    if len(errors_before) < 3:
        return np.nan

    if method == 'inverse_variance':
        var = np.var(errors_before)
        return 1.0 / (var + 1e-6)

    elif method == 'trial_count':
        return float(len(errors_before))

    elif method == 'bayesian':
        # Bayesian precision: accumulates like n/σ²
        n = len(errors_before)
        var = np.var(errors_before)
        return n / (var + 1e-6)

    elif method == 'recent':
        # Use only recent trials (last 10)
        recent = errors_before[-10:]
        var = np.var(recent)
        return 1.0 / (var + 1e-6)

    else:
        raise ValueError(f"Unknown method: {method}")


def compute_learning_rate_proxy(errors_before, updates_before):
    """
    Compute average learning rate before changepoint.

    In Kalman filter terms, this should decrease as precision increases.
    """
    valid = (np.abs(errors_before) > 1e-6)
    if valid.sum() < 3:
        return np.nan

    lrs = updates_before[valid] / errors_before[valid]
    return np.median(np.clip(lrs, 0, 2))


# =============================================================================
# Relaxation Time Fitting
# =============================================================================

def exponential_decay(t, A, tau, C):
    """Exponential decay function: A * exp(-t/tau) + C"""
    return A * np.exp(-t / tau) + C


def fit_relaxation_time(errors_after, max_trials=20):
    """
    Fit exponential decay to post-changepoint error trajectory.

    The model assumes errors decay exponentially as beliefs adjust:
        |error(t)| = A * exp(-t/τ) + C

    where τ is the relaxation time constant.

    Args:
        errors_after: Array of prediction errors after changepoint
        max_trials: Number of trials to fit (default: 20)

    Returns:
        tau: Relaxation time constant (None if fit failed)
        fit_quality: R² of the fit
        fit_params: (A, tau, C)
    """
    # Use absolute errors
    abs_errors = np.abs(errors_after[:max_trials])
    t = np.arange(len(abs_errors))

    if len(abs_errors) < 5:
        return None, None, None

    # Initial guess
    A0 = abs_errors[0] - abs_errors[-1]
    tau0 = len(abs_errors) / 2
    C0 = abs_errors[-1]

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            popt, pcov = curve_fit(
                exponential_decay, t, abs_errors,
                p0=[A0, tau0, C0],
                bounds=([0, 0.5, 0], [np.inf, 50, np.inf]),
                maxfev=1000
            )

        A, tau, C = popt

        # Compute R²
        predicted = exponential_decay(t, *popt)
        ss_res = np.sum((abs_errors - predicted)**2)
        ss_tot = np.sum((abs_errors - np.mean(abs_errors))**2)
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        return tau, r2, popt

    except (RuntimeError, ValueError):
        return None, None, None


def fit_relaxation_linear(errors_after, max_trials=15):
    """
    Alternative: fit linear decay to get "half-life" approximation.

    If exponential fitting is unstable, use linear regression on
    log(|error|) to estimate decay rate.

    Args:
        errors_after: Errors after changepoint
        max_trials: Trials to fit

    Returns:
        tau_approx: Approximate relaxation time
        r2: Fit quality
    """
    abs_errors = np.abs(errors_after[:max_trials])
    t = np.arange(len(abs_errors))

    # Avoid log(0)
    abs_errors = np.clip(abs_errors, 1e-6, None)

    try:
        # Linear fit to log(error)
        slope, intercept, r, p, se = stats.linregress(t, np.log(abs_errors))

        if slope >= 0:  # No decay
            return None, None

        # τ = -1/slope for exponential decay
        tau_approx = -1.0 / slope

        return tau_approx, r**2

    except:
        return None, None


# =============================================================================
# Per-Changepoint Analysis
# =============================================================================

def analyze_changepoint(arrays, cp_idx, before_window=20, after_window=20):
    """
    Analyze a single changepoint event.

    Args:
        arrays: Dict with 'prediction_error', 'update', 'outcome', 'prediction'
                Optional: 'noise_std' for sensory precision Λ_o
        cp_idx: Index of the changepoint trial
        before_window: Trials to use before CP
        after_window: Trials to use after CP

    Returns:
        dict with precision_before, tau, fit_quality, etc.
    """
    n_trials = len(arrays['prediction_error'])

    # Define windows
    before_start = max(0, cp_idx - before_window)
    before_end = cp_idx
    after_start = cp_idx
    after_end = min(n_trials, cp_idx + after_window)

    # Get data
    errors_before = arrays['prediction_error'][before_start:before_end]
    updates_before = arrays['update'][before_start:before_end]
    errors_after = arrays['prediction_error'][after_start:after_end]

    if len(errors_before) < 5 or len(errors_after) < 5:
        return None

    # Compute precision before CP
    precision_methods = ['inverse_variance', 'trial_count', 'bayesian', 'recent']
    precisions = {}
    for method in precision_methods:
        precisions[method] = compute_precision_before_cp(errors_before, method)

    # Learning rate before CP
    lr_before = compute_learning_rate_proxy(errors_before, updates_before)

    # Fit relaxation time after CP
    tau_exp, r2_exp, params_exp = fit_relaxation_time(errors_after, after_window)
    tau_lin, r2_lin = fit_relaxation_linear(errors_after, after_window)

    # Use exponential if good fit, otherwise linear
    if tau_exp is not None and r2_exp is not None and r2_exp > 0.3:
        tau = tau_exp
        fit_quality = r2_exp
        fit_method = 'exponential'
    elif tau_lin is not None and r2_lin is not None:
        tau = tau_lin
        fit_quality = r2_lin
        fit_method = 'linear'
    else:
        tau = None
        fit_quality = None
        fit_method = None

    # Initial error magnitude (right after CP)
    initial_error = np.abs(errors_after[0]) if len(errors_after) > 0 else None

    # Sensory precision: Λ_o = n/σ² (accumulated over n trials)
    # n = trials since last changepoint (observations from current regime)
    n_trials_stable = len(errors_before)  # trials in current regime before this CP

    if 'noise_std' in arrays:
        noise_std_before = arrays['noise_std'][before_start:before_end]
        mean_noise_std = np.mean(noise_std_before) if len(noise_std_before) > 0 else None
        if mean_noise_std is not None and mean_noise_std > 0:
            # Static: single observation precision
            lambda_o_static = 1.0 / (mean_noise_std ** 2)
            # Accumulated: n observations from current regime
            lambda_o = n_trials_stable / (mean_noise_std ** 2)
        else:
            lambda_o = None
            lambda_o_static = None
    else:
        lambda_o = None
        lambda_o_static = None
        mean_noise_std = None

    # Compute total mass M = Λ_p + Λ_o (theory prediction)
    total_mass = {}
    for method, lambda_p in precisions.items():
        if lambda_o is not None and np.isfinite(lambda_p):
            total_mass[method] = lambda_p + lambda_o
        else:
            total_mass[method] = None

    return {
        'cp_idx': cp_idx,
        'precision_before': precisions,  # Λ_p (prior precision estimates)
        'lambda_o': lambda_o,  # Λ_o = n/σ² (accumulated sensory precision)
        'lambda_o_static': lambda_o_static,  # 1/σ² (single observation)
        'total_mass': total_mass,  # M = Λ_p + Λ_o
        'n_stable': n_trials_stable,  # n = trials since last CP
        'noise_std': mean_noise_std,
        'lr_before': lr_before,
        'tau': tau,
        'fit_quality': fit_quality,
        'fit_method': fit_method,
        'initial_error': initial_error,
        'n_trials_before': len(errors_before),
        'n_trials_after': len(errors_after),
    }


def analyze_subject_changepoints(subject_data, min_trials_between=5):
    """
    Analyze all changepoints for a single subject.

    Args:
        subject_data: SubjectData object
        min_trials_between: Minimum trials between CPs to analyze

    Returns:
        List of changepoint analysis results
    """
    arrays = subject_data.get_arrays()
    # Include noise_std for sensory precision calculation
    arrays['noise_std'] = arrays.get('noise_std', None)
    cp_indices = subject_data.get_changepoint_indices()

    results = []
    prev_cp = -min_trials_between  # Start from beginning

    for cp_idx in cp_indices:
        # Skip if too close to previous CP
        if cp_idx - prev_cp < min_trials_between:
            continue

        result = analyze_changepoint(arrays, cp_idx)
        if result is not None and result['tau'] is not None:
            results.append(result)

        prev_cp = cp_idx

    return results


# =============================================================================
# Statistical Tests
# =============================================================================

def test_tau_lambda_correlation(cp_results, precision_method='inverse_variance'):
    """
    Test the core prediction: τ ∝ Λ

    Args:
        cp_results: List of changepoint analysis results
        precision_method: Which precision measure to use

    Returns:
        dict with correlation, p-value, slope, etc.
    """
    # Extract valid data points
    precisions = []
    taus = []

    for cp in cp_results:
        if cp['tau'] is not None and cp['fit_quality'] is not None:
            if cp['fit_quality'] > 0.2:  # Only use reasonable fits
                prec = cp['precision_before'][precision_method]
                if np.isfinite(prec) and np.isfinite(cp['tau']):
                    precisions.append(prec)
                    taus.append(cp['tau'])

    if len(precisions) < 10:
        return {
            'n_points': len(precisions),
            'correlation': None,
            'p_value': None,
            'sufficient_data': False
        }

    precisions = np.array(precisions)
    taus = np.array(taus)

    # Log-log correlation (for power law τ ∝ Λ^α)
    log_prec = np.log(precisions + 1e-10)
    log_tau = np.log(taus + 1e-10)

    # Pearson correlation
    r_pearson, p_pearson = stats.pearsonr(precisions, taus)

    # Spearman correlation (rank-based, more robust)
    r_spearman, p_spearman = stats.spearmanr(precisions, taus)

    # Linear regression (log-log for power law)
    slope, intercept, r_log, p_log, se = stats.linregress(log_prec, log_tau)

    return {
        'n_points': len(precisions),
        'r_pearson': r_pearson,
        'p_pearson': p_pearson,
        'r_spearman': r_spearman,
        'p_spearman': p_spearman,
        'slope_loglog': slope,  # Power law exponent: τ ~ Λ^slope
        'r_loglog': r_log,
        'p_loglog': p_log,
        'sufficient_data': True,
        'precisions': precisions,
        'taus': taus
    }


# =============================================================================
# Main Analysis
# =============================================================================

def run_changepoint_analysis():
    """Run the changepoint-specific inertia analysis."""
    print("="*70)
    print("CHANGEPOINT-SPECIFIC INERTIA ANALYSIS")
    print("="*70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nTesting the actual prediction: τ ∝ Λ")
    print(f"  τ = relaxation time after changepoint")
    print(f"  Λ = precision accumulated before changepoint")
    print(f"\nNote: A subject can show β ≈ 0 on average while still")
    print(f"      exhibiting τ ~ Λ scaling at individual changepoints!")

    # Load data
    print("\n[1/4] Loading McGuireNassar2014 data...")
    subjects = load_mcguire_nassar_2014()
    print(f"  Loaded {len(subjects)} subjects")

    # Analyze all changepoints
    print("\n[2/4] Analyzing changepoints...")
    all_cp_results = []
    subject_summaries = []

    for sid in sorted(subjects.keys()):
        subj = subjects[sid]
        cp_results = analyze_subject_changepoints(subj)
        all_cp_results.extend(cp_results)

        if cp_results:
            # Test τ ~ Λ for this subject
            test = test_tau_lambda_correlation(cp_results)
            subject_summaries.append({
                'sid': sid,
                'n_changepoints': len(cp_results),
                'test': test
            })

    print(f"  Analyzed {len(all_cp_results)} changepoint events")

    # Overall test
    print("\n[3/4] Testing τ ~ Λ relationship...")
    overall_test = test_tau_lambda_correlation(all_cp_results)

    print(f"\n{'='*70}")
    print("RESULTS: τ ~ Λ Correlation (pooled across all subjects)")
    print("="*70)

    if overall_test['sufficient_data']:
        print(f"\nData points: {overall_test['n_points']} changepoints")

        print(f"\nLinear correlations:")
        print(f"  Pearson r  = {overall_test['r_pearson']:.3f} (p = {overall_test['p_pearson']:.4f})")
        print(f"  Spearman ρ = {overall_test['r_spearman']:.3f} (p = {overall_test['p_spearman']:.4f})")

        print(f"\nPower law fit (log-log): τ ~ Λ^α")
        print(f"  Exponent α = {overall_test['slope_loglog']:.3f}")
        print(f"  R² = {overall_test['r_loglog']**2:.3f}")
        print(f"  p = {overall_test['p_loglog']:.4f}")

        # Interpretation
        print(f"\n{'='*70}")
        print("INTERPRETATION")
        print("="*70)

        if overall_test['p_spearman'] < 0.05 and overall_test['r_spearman'] > 0:
            print(f"\n  ✓ POSITIVE correlation between precision and relaxation time")
            print(f"  ✓ Higher pre-CP precision → slower post-CP adjustment")
            print(f"  ✓ SUPPORTS the Hamiltonian inertia prediction (τ ∝ Λ)")
        elif overall_test['p_spearman'] < 0.05 and overall_test['r_spearman'] < 0:
            print(f"\n  ✗ NEGATIVE correlation (opposite to prediction)")
            print(f"  ✗ Higher precision → FASTER adjustment")
            print(f"  ✗ CONTRADICTS the simple inertia prediction")
        else:
            print(f"\n  ○ No significant correlation found")
            print(f"  ○ Precision does not predict relaxation time")
            print(f"  ○ Neutral evidence for inertia theory")
    else:
        print(f"  Insufficient data for analysis ({overall_test['n_points']} points)")

    # Per-subject summary
    print(f"\n{'='*70}")
    print("PER-SUBJECT RESULTS")
    print("="*70)

    positive_corr = 0
    negative_corr = 0
    sig_positive = 0
    sig_negative = 0

    for summ in subject_summaries:
        test = summ['test']
        if test['sufficient_data']:
            if test['r_spearman'] > 0:
                positive_corr += 1
                if test['p_spearman'] < 0.05:
                    sig_positive += 1
            else:
                negative_corr += 1
                if test['p_spearman'] < 0.05:
                    sig_negative += 1

    n_valid = positive_corr + negative_corr
    print(f"\nSubjects with sufficient data: {n_valid}")
    print(f"  Positive τ~Λ correlation: {positive_corr} ({100*positive_corr/max(n_valid,1):.0f}%)")
    print(f"    Significant (p<0.05): {sig_positive}")
    print(f"  Negative τ~Λ correlation: {negative_corr} ({100*negative_corr/max(n_valid,1):.0f}%)")
    print(f"    Significant (p<0.05): {sig_negative}")

    # Binomial test: are more subjects positive than expected by chance?
    if n_valid > 5:
        binom_result = stats.binomtest(positive_corr, n_valid, 0.5, alternative='greater')
        binom_p = binom_result.pvalue
        print(f"\nBinomial test (more positive than chance):")
        print(f"  p = {binom_p:.4f}")
        if binom_p < 0.05:
            print(f"  → Significantly more subjects show positive τ~Λ correlation")

    # Additional analyses
    print(f"\n{'='*70}")
    print("[4/4] ADDITIONAL ANALYSES")
    print("="*70)

    # Check different precision measures
    print("\nCorrelation using different precision measures:")
    for method in ['inverse_variance', 'trial_count', 'bayesian', 'recent']:
        test = test_tau_lambda_correlation(all_cp_results, method)
        if test['sufficient_data']:
            print(f"  {method:20s}: r = {test['r_spearman']:+.3f} (p = {test['p_spearman']:.3f})")

    # Check if initial error magnitude matters
    print("\nDoes initial error magnitude affect τ?")
    initial_errors = [cp['initial_error'] for cp in all_cp_results
                      if cp['initial_error'] is not None and cp['tau'] is not None]
    taus_for_error = [cp['tau'] for cp in all_cp_results
                      if cp['initial_error'] is not None and cp['tau'] is not None]

    if len(initial_errors) > 10:
        r_err, p_err = stats.spearmanr(initial_errors, taus_for_error)
        print(f"  Initial error vs τ: r = {r_err:.3f} (p = {p_err:.3f})")

    # Test TOTAL MASS prediction: τ ~ M = Λ_0 + n/σ²
    print("\n" + "-"*40)
    print("THEORY TEST: τ ~ M = Λ_0 + n/σ²")
    print("-"*40)
    print("\nWhere n = trials since last changepoint")

    # Test n alone (trials since last CP)
    n_stables = [cp['n_stable'] for cp in all_cp_results
                 if cp.get('n_stable') is not None and cp['tau'] is not None]
    taus_for_n = [cp['tau'] for cp in all_cp_results
                  if cp.get('n_stable') is not None and cp['tau'] is not None]

    if len(n_stables) > 10:
        r_n, p_n = stats.spearmanr(n_stables, taus_for_n)
        print(f"\nTrials since last CP (n):")
        print(f"  τ ~ n: r = {r_n:.3f} (p = {p_n:.3f})")
        if p_n < 0.05 and r_n > 0:
            print(f"  → More stable trials → longer relaxation time!")

    # Test Λ_o = n/σ² (accumulated sensory precision)
    lambda_os = [cp['lambda_o'] for cp in all_cp_results
                 if cp['lambda_o'] is not None and cp['tau'] is not None]
    taus_for_lo = [cp['tau'] for cp in all_cp_results
                   if cp['lambda_o'] is not None and cp['tau'] is not None]

    if len(lambda_os) > 10:
        r_lo, p_lo = stats.spearmanr(lambda_os, taus_for_lo)
        print(f"\nAccumulated sensory precision Λ_o = n/σ²:")
        print(f"  τ ~ Λ_o: r = {r_lo:.3f} (p = {p_lo:.3f})")
        if p_lo < 0.05 and r_lo > 0:
            print(f"  ✓ Higher accumulated precision → longer relaxation!")

    # Test static 1/σ² for comparison
    lambda_o_statics = [cp.get('lambda_o_static') for cp in all_cp_results
                        if cp.get('lambda_o_static') is not None and cp['tau'] is not None]
    taus_for_static = [cp['tau'] for cp in all_cp_results
                       if cp.get('lambda_o_static') is not None and cp['tau'] is not None]

    if len(lambda_o_statics) > 10:
        r_static, p_static = stats.spearmanr(lambda_o_statics, taus_for_static)
        print(f"\nStatic sensory precision 1/σ² (single obs):")
        print(f"  τ ~ 1/σ²: r = {r_static:.3f} (p = {p_static:.3f})")

    # Test total precision Λ = Λ_p + Λ_o (not just n/σ² which ignores prior)
    print("\n" + "-"*40)
    print("Does τ collapse onto a curve against Λ?")
    print("-"*40)

    for method in ['inverse_variance', 'bayesian']:
        masses = []
        taus_for_mass = []
        lambda_os_only = []

        for cp in all_cp_results:
            if (cp['total_mass'] is not None and
                cp['total_mass'].get(method) is not None and
                cp['tau'] is not None and
                cp['lambda_o'] is not None):
                masses.append(cp['total_mass'][method])
                taus_for_mass.append(cp['tau'])
                lambda_os_only.append(cp['lambda_o'])

        if len(masses) > 10:
            r_total, p_total = stats.spearmanr(masses, taus_for_mass)
            r_obs_only, p_obs = stats.spearmanr(lambda_os_only, taus_for_mass)

            print(f"\n{method} method:")
            print(f"  τ ~ Λ (total = Λ_p + Λ_o): r = {r_total:.3f} (p = {p_total:.3f})")
            print(f"  τ ~ Λ_o only (n/σ²):       r = {r_obs_only:.3f} (p = {p_obs:.3f})")

            # Check if total Λ explains more variance than Λ_o alone
            if abs(r_total) > abs(r_obs_only):
                print(f"  → Total Λ is better predictor (Λ_p matters!)")
            else:
                print(f"  → Λ_o alone is sufficient (Λ_p ≈ negligible or constant)")

            if p_total < 0.05:
                print(f"  ✓ τ DOES depend systematically on Λ")

    print("\n" + "="*70)

    return {
        'overall_test': overall_test,
        'subject_summaries': subject_summaries,
        'all_cp_results': all_cp_results
    }


def test_tau_lambda_controlling_error(cp_results, precision_method='inverse_variance'):
    """
    Test τ ~ Λ while controlling for initial error magnitude.

    Since initial error strongly predicts τ (r=-0.686), we need to
    partial out this effect to see if precision has independent predictive value.

    Uses partial correlation: cor(τ, Λ | error)
    """
    from scipy.stats import pearsonr

    precisions = []
    taus = []
    initial_errors = []

    for cp in cp_results:
        if (cp['tau'] is not None and
            cp['fit_quality'] is not None and
            cp['fit_quality'] > 0.2 and
            cp['initial_error'] is not None):
            prec = cp['precision_before'][precision_method]
            if np.isfinite(prec) and np.isfinite(cp['tau']):
                precisions.append(prec)
                taus.append(cp['tau'])
                initial_errors.append(cp['initial_error'])

    if len(precisions) < 20:
        return {'sufficient_data': False, 'n_points': len(precisions)}

    precisions = np.array(precisions)
    taus = np.array(taus)
    initial_errors = np.array(initial_errors)

    # Partial correlation: cor(τ, Λ | error)
    # Regress τ on error, get residuals
    slope_te, intercept_te = np.polyfit(initial_errors, taus, 1)
    tau_residuals = taus - (slope_te * initial_errors + intercept_te)

    # Regress Λ on error, get residuals
    slope_pe, intercept_pe = np.polyfit(initial_errors, precisions, 1)
    prec_residuals = precisions - (slope_pe * initial_errors + intercept_pe)

    # Correlate residuals
    r_partial, p_partial = pearsonr(tau_residuals, prec_residuals)

    # Also check: does precision predict τ beyond what error predicts?
    # Multiple regression: τ ~ error + precision
    from scipy.linalg import lstsq
    X = np.column_stack([np.ones(len(taus)), initial_errors, precisions])
    coeffs, residuals, rank, s = lstsq(X, taus)

    # Get coefficient for precision and its significance
    # Simplified: just use correlation of precision with residuals after error

    return {
        'n_points': len(precisions),
        'r_partial': r_partial,
        'p_partial': p_partial,
        'r_tau_error': pearsonr(taus, initial_errors)[0],
        'r_tau_prec': pearsonr(taus, precisions)[0],
        'r_prec_error': pearsonr(precisions, initial_errors)[0],
        'sufficient_data': True
    }


def analyze_other_datasets():
    """Analyze jNeuroBehav and pupil datasets for τ ~ Λ."""
    print("\n" + "="*70)
    print("ADDITIONAL DATASETS ANALYSIS")
    print("="*70)

    # jNeuroBehav
    print("\n[1/2] Analyzing jNeuroBehav dataset...")
    try:
        data = load_jneurobehav()
        sessions = np.unique(data['session'])

        all_cp_results = []

        for sess in sessions:
            mask = data['session'] == sess
            outcomes = data['outcome'][mask].flatten()
            predictions = data['prediction'][mask].flatten()
            true_means = data['mean'][mask].flatten()  # distMean from loader

            if len(outcomes) < 30:
                continue

            # Detect changepoints from true mean changes
            mean_changes = np.abs(np.diff(true_means)) > 1
            cp_indices = np.where(mean_changes)[0] + 1

            # Create arrays dict
            errors = outcomes - predictions
            updates = np.diff(predictions, prepend=predictions[0])
            arrays = {
                'prediction_error': errors,
                'update': updates,
                'outcome': outcomes,
                'prediction': predictions
            }

            # Analyze each changepoint
            prev_cp = 0
            for cp_idx in cp_indices:
                if cp_idx - prev_cp < 10:  # Skip if too close
                    continue
                result = analyze_changepoint(arrays, cp_idx, before_window=15, after_window=15)
                if result is not None and result['tau'] is not None:
                    all_cp_results.append(result)
                prev_cp = cp_idx

        print(f"  Analyzed {len(all_cp_results)} changepoint events")

        if len(all_cp_results) > 20:
            test = test_tau_lambda_correlation(all_cp_results)
            if test['sufficient_data']:
                print(f"  τ ~ Λ correlation: r = {test['r_spearman']:.3f} (p = {test['p_spearman']:.3f})")

    except Exception as e:
        print(f"  Error: {e}")

    # Pupil data
    print("\n[2/2] Analyzing Pupil dataset...")
    try:
        data = load_pupil_data()
        n_subjects = data['outcome'].shape[0]

        all_cp_results = []

        for subj_idx in range(n_subjects):
            outcomes = data['outcome'][subj_idx, :].flatten().astype(float)
            predictions = data['prediction'][subj_idx, :].flatten().astype(float)
            true_means = data['mean'][subj_idx, :].flatten()

            # Remove NaNs
            valid = ~(np.isnan(outcomes) | np.isnan(predictions) | np.isnan(true_means))
            outcomes = outcomes[valid]
            predictions = predictions[valid]
            true_means = true_means[valid]

            if len(outcomes) < 30:
                continue

            # Detect changepoints
            mean_changes = np.abs(np.diff(true_means)) > 1
            cp_indices = np.where(mean_changes)[0] + 1

            errors = outcomes - predictions
            updates = np.diff(predictions, prepend=predictions[0])
            arrays = {
                'prediction_error': errors,
                'update': updates,
                'outcome': outcomes,
                'prediction': predictions
            }

            prev_cp = 0
            for cp_idx in cp_indices:
                if cp_idx - prev_cp < 10:
                    continue
                result = analyze_changepoint(arrays, cp_idx, before_window=15, after_window=15)
                if result is not None and result['tau'] is not None:
                    all_cp_results.append(result)
                prev_cp = cp_idx

        print(f"  Analyzed {len(all_cp_results)} changepoint events")

        if len(all_cp_results) > 20:
            test = test_tau_lambda_correlation(all_cp_results)
            if test['sufficient_data']:
                print(f"  τ ~ Λ correlation: r = {test['r_spearman']:.3f} (p = {test['p_spearman']:.3f})")

    except Exception as e:
        print(f"  Error: {e}")


if __name__ == '__main__':
    results = run_changepoint_analysis()

    # Additional: control for initial error
    print("\n" + "="*70)
    print("PARTIAL CORRELATION: τ ~ Λ | error")
    print("="*70)
    partial = test_tau_lambda_controlling_error(results['all_cp_results'])
    if partial['sufficient_data']:
        print(f"\nControlling for initial error magnitude:")
        print(f"  Partial correlation (τ, Λ | error): r = {partial['r_partial']:.3f} (p = {partial['p_partial']:.4f})")
        print(f"\nRaw correlations:")
        print(f"  τ ~ error:     r = {partial['r_tau_error']:.3f}")
        print(f"  τ ~ Λ:         r = {partial['r_tau_prec']:.3f}")
        print(f"  Λ ~ error:     r = {partial['r_prec_error']:.3f}")

        if partial['p_partial'] < 0.05 and partial['r_partial'] > 0:
            print(f"\n  ✓ Precision predicts τ BEYOND initial error!")
            print(f"  ✓ Supports inertia theory with error-modulated response")
        else:
            print(f"\n  ✗ Precision has no predictive value beyond initial error")

    # Analyze other datasets
    analyze_other_datasets()
