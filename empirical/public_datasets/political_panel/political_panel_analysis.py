# -*- coding: utf-8 -*-
"""
Created on Thu Dec  4 14:08:19 2025

@author: chris and christine
"""

# -*- coding: utf-8 -*-
"""
Hamiltonian Belief Dynamics Analysis: 26-Wave Political Panel
==============================================================

Tests the Kaplowitz-Fink second-order model on the Brandt et al. (2021)
26-wave political psychology panel dataset.

Model: m·x''(t) + γ·x'(t) + k·x(t) = F(t)

Key predictions:
1. Underdamped (ζ < 1): Oscillation around equilibrium
2. Critically damped (ζ = 1): Fastest return without overshoot
3. Overdamped (ζ > 1): Slow exponential decay

Where ζ = γ / (2√(mk)) is the damping ratio.

Data: https://osf.io/3pwvb/
Paper: https://openpsychologydata.metajnl.com/articles/10.5334/jopd.54

Author: Hamiltonian-VFE Team
Date: December 2025
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats, optimize, signal
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(SCRIPT_DIR, 'political_panel')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class HamiltonianFit:
    """Results from fitting Hamiltonian dynamics."""
    participant_id: str
    variable: str
    m: float  # Mass (inertia)
    gamma: float  # Damping
    k: float  # Restoring force (spring constant)
    zeta: float  # Damping ratio
    omega_n: float  # Natural frequency
    regime: str  # 'underdamped', 'critically_damped', 'overdamped'
    r_squared: float
    aic: float
    trajectory: np.ndarray
    fitted: np.ndarray


def load_panel_data(data_dir: str) -> pd.DataFrame:
    """
    Load the 26-wave political psychology panel data.

    Expected file structure from OSF:
    - Main data file with columns for each wave
    - participant_id, wave1_var, wave2_var, ..., wave26_var
    """
    # Try common file names
    possible_files = [
        'political_panel_data.csv',
        'data.csv',
        'panel_data.csv',
        'brandt_2021_data.csv',
        '26wave_data.csv',
    ]

    for fname in possible_files:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            print(f"Loading data from: {fpath}")
            return pd.read_csv(fpath)

    # Try to find any CSV
    csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
    if csv_files:
        fpath = os.path.join(data_dir, csv_files[0])
        print(f"Loading data from: {fpath}")
        return pd.read_csv(fpath)

    raise FileNotFoundError(
        f"No data files found in {data_dir}. "
        f"Please download from https://osf.io/3pwvb/"
    )


def extract_time_series(df: pd.DataFrame, variable_prefix: str) -> Dict[str, np.ndarray]:
    """
    Extract time series for a variable across all waves.

    Args:
        df: Panel dataframe
        variable_prefix: Prefix like 'pol_attitude' or 'threat'

    Returns:
        Dict mapping participant_id -> time series array
    """
    # Find columns matching the variable
    wave_cols = [c for c in df.columns if variable_prefix in c.lower()]
    wave_cols = sorted(wave_cols, key=lambda x: int(''.join(filter(str.isdigit, x)) or '0'))

    if not wave_cols:
        print(f"Warning: No columns found for prefix '{variable_prefix}'")
        return {}

    print(f"Found {len(wave_cols)} waves for {variable_prefix}")

    # Extract per-participant time series
    series = {}
    id_col = 'participant_id' if 'participant_id' in df.columns else df.columns[0]

    for idx, row in df.iterrows():
        pid = str(row[id_col])
        values = row[wave_cols].values.astype(float)

        # Only include if we have enough non-null values
        valid = ~np.isnan(values)
        if np.sum(valid) >= 10:  # At least 10 waves
            series[pid] = values

    print(f"Extracted {len(series)} valid time series")
    return series


def damped_oscillator_solution(t: np.ndarray, x0: float, v0: float,
                                m: float, gamma: float, k: float) -> np.ndarray:
    """
    Analytical solution to damped harmonic oscillator.

    m·x'' + γ·x' + k·x = 0

    With initial conditions x(0) = x0, x'(0) = v0
    """
    if m <= 0 or k <= 0:
        return np.full_like(t, x0, dtype=float)

    omega_n = np.sqrt(k / m)  # Natural frequency
    zeta = gamma / (2 * np.sqrt(m * k))  # Damping ratio

    if zeta < 0:
        zeta = 0.01  # Minimum damping

    if zeta < 1:  # Underdamped
        omega_d = omega_n * np.sqrt(1 - zeta**2)
        A = x0
        B = (v0 + zeta * omega_n * x0) / omega_d
        x = np.exp(-zeta * omega_n * t) * (A * np.cos(omega_d * t) + B * np.sin(omega_d * t))

    elif zeta == 1:  # Critically damped
        A = x0
        B = v0 + omega_n * x0
        x = (A + B * t) * np.exp(-omega_n * t)

    else:  # Overdamped
        alpha1 = omega_n * (zeta + np.sqrt(zeta**2 - 1))
        alpha2 = omega_n * (zeta - np.sqrt(zeta**2 - 1))

        if np.abs(alpha1 - alpha2) < 1e-10:
            A = x0
            B = v0 + alpha1 * x0
            x = (A + B * t) * np.exp(-alpha1 * t)
        else:
            A = (v0 + alpha2 * x0) / (alpha2 - alpha1)
            B = x0 - A
            x = A * np.exp(-alpha1 * t) + B * np.exp(-alpha2 * t)

    return x


def fit_hamiltonian_dynamics(
    time_series: np.ndarray,
    dt: float = 14.0,  # 2 weeks between waves
    equilibrium: Optional[float] = None
) -> HamiltonianFit:
    """
    Fit second-order Hamiltonian dynamics to a time series.

    Model: m·x'' + γ·x' + k·(x - x_eq) = 0

    Args:
        time_series: Array of belief values over time
        dt: Time between measurements (days)
        equilibrium: Equilibrium point (if None, use mean)

    Returns:
        HamiltonianFit with estimated parameters
    """
    # Handle NaNs
    valid_mask = ~np.isnan(time_series)
    if np.sum(valid_mask) < 5:
        return None

    # Use valid points only
    t_valid = np.arange(len(time_series))[valid_mask] * dt
    x_valid = time_series[valid_mask]

    # Equilibrium
    if equilibrium is None:
        equilibrium = np.mean(x_valid)

    # Deviation from equilibrium
    y = x_valid - equilibrium
    t = t_valid - t_valid[0]

    # Initial conditions
    x0 = y[0]
    v0 = (y[1] - y[0]) / (t[1] - t[0]) if len(y) > 1 else 0

    def model(params):
        m, gamma, k = params
        if m <= 0 or k <= 0 or gamma < 0:
            return np.inf
        try:
            y_pred = damped_oscillator_solution(t, x0, v0, m, gamma, k)
            return np.sum((y - y_pred)**2)
        except:
            return np.inf

    # Grid search for initial guess
    best_loss = np.inf
    best_params = [1.0, 1.0, 1.0]

    for m_init in [0.1, 1.0, 10.0]:
        for gamma_init in [0.01, 0.1, 1.0, 10.0]:
            for k_init in [0.01, 0.1, 1.0]:
                loss = model([m_init, gamma_init, k_init])
                if loss < best_loss:
                    best_loss = loss
                    best_params = [m_init, gamma_init, k_init]

    # Optimize
    try:
        result = optimize.minimize(
            model,
            best_params,
            method='Nelder-Mead',
            options={'maxiter': 1000}
        )
        m, gamma, k = result.x
        m = max(0.01, m)
        gamma = max(0.001, gamma)
        k = max(0.001, k)
    except:
        m, gamma, k = best_params

    # Compute fit quality
    y_fitted = damped_oscillator_solution(t, x0, v0, m, gamma, k)
    ss_res = np.sum((y - y_fitted)**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0

    # AIC
    n = len(y)
    k_params = 3
    aic = n * np.log(ss_res / n + 1e-10) + 2 * k_params

    # Compute derived quantities
    omega_n = np.sqrt(k / m) if m > 0 and k > 0 else 0
    zeta = gamma / (2 * np.sqrt(m * k)) if m > 0 and k > 0 else 1

    if zeta < 1:
        regime = 'underdamped'
    elif zeta == 1:
        regime = 'critically_damped'
    else:
        regime = 'overdamped'

    return HamiltonianFit(
        participant_id='',
        variable='',
        m=m,
        gamma=gamma,
        k=k,
        zeta=zeta,
        omega_n=omega_n,
        regime=regime,
        r_squared=r_squared,
        aic=aic,
        trajectory=y + equilibrium,
        fitted=y_fitted + equilibrium
    )


def test_oscillation_hypothesis(fits: List[HamiltonianFit]) -> Dict:
    """
    Test whether the population shows evidence of oscillation (underdamped dynamics).

    Kaplowitz et al. (1983) found underdamped dynamics with period ~13.5 seconds.
    We test whether political attitudes show similar oscillatory behavior.
    """
    zetas = [f.zeta for f in fits if f is not None]
    regimes = [f.regime for f in fits if f is not None]
    r_squareds = [f.r_squared for f in fits if f is not None]

    n_underdamped = sum(1 for r in regimes if r == 'underdamped')
    n_overdamped = sum(1 for r in regimes if r == 'overdamped')
    n_critical = sum(1 for r in regimes if r == 'critically_damped')
    n_total = len(regimes)

    # Test: Is proportion underdamped > chance (33%)?
    if n_total > 0:
        prop_underdamped = n_underdamped / n_total
        # Binomial test against 33% null
        from scipy.stats import binomtest
        binom_result = binomtest(n_underdamped, n_total, 0.33, alternative='greater')
        p_value = binom_result.pvalue
    else:
        prop_underdamped = 0
        p_value = 1.0

    # Oscillation period for underdamped cases
    periods = []
    for f in fits:
        if f is not None and f.regime == 'underdamped' and f.zeta < 1:
            omega_d = f.omega_n * np.sqrt(1 - f.zeta**2)
            if omega_d > 0:
                period = 2 * np.pi / omega_d
                periods.append(period)

    return {
        'n_total': n_total,
        'n_underdamped': n_underdamped,
        'n_overdamped': n_overdamped,
        'n_critical': n_critical,
        'prop_underdamped': prop_underdamped,
        'p_value': p_value,
        'mean_zeta': np.mean(zetas) if zetas else None,
        'std_zeta': np.std(zetas) if zetas else None,
        'mean_r_squared': np.mean(r_squareds) if r_squareds else None,
        'oscillation_periods': periods,
        'mean_period_days': np.mean(periods) if periods else None,
    }


def analyze_panel(data_dir: str = DATA_DIR):
    """
    Main analysis: Test Hamiltonian dynamics on political panel data.
    """
    print("=" * 70)
    print("HAMILTONIAN BELIEF DYNAMICS: 26-Wave Political Panel")
    print("=" * 70)
    print("""
Testing the Kaplowitz-Fink second-order model:
    m·x'' + γ·x' + k·x = F(t)

Predictions:
- Underdamped (ζ < 1): Beliefs oscillate around equilibrium
- Overdamped (ζ > 1): Beliefs decay monotonically

Kaplowitz et al. (1983) found ζ < 1 (oscillation) for persuasion.
Question: Do political attitudes show the same pattern?
    """)

    # Load data
    print("\n[1/4] Loading data...")
    try:
        df = load_panel_data(data_dir)
        print(f"      Loaded {len(df)} participants, {len(df.columns)} columns")
    except FileNotFoundError as e:
        print(f"\n ERROR: {e}")
        print("\n Please download the data from https://osf.io/3pwvb/")
        print(" and place CSV files in:", data_dir)
        return None

    # Identify attitude variables
    print("\n[2/4] Identifying attitude variables...")

    # Common variable prefixes in political psychology
    variable_prefixes = [
        'attitude', 'opinion', 'position', 'pol_', 'political',
        'threat', 'stress', 'distance', 'feeling', 'approval'
    ]

    # Find matching columns
    all_fits = {}

    for prefix in variable_prefixes:
        series_dict = extract_time_series(df, prefix)

        if len(series_dict) > 10:  # Enough participants
            print(f"\n   Analyzing '{prefix}' ({len(series_dict)} participants)...")

            fits = []
            for pid, ts in series_dict.items():
                fit = fit_hamiltonian_dynamics(ts, dt=14.0)
                if fit is not None:
                    fit.participant_id = pid
                    fit.variable = prefix
                    fits.append(fit)

            if fits:
                all_fits[prefix] = fits
                print(f"      Fitted {len(fits)} trajectories")

    if not all_fits:
        print("\n ERROR: Could not extract time series from data.")
        print(" Please check the data format and column names.")
        return None

    # Test oscillation hypothesis
    print("\n[3/4] Testing oscillation hypothesis...")

    results = {}
    for var, fits in all_fits.items():
        results[var] = test_oscillation_hypothesis(fits)

        r = results[var]
        print(f"\n   {var}:")
        print(f"      Underdamped: {r['n_underdamped']}/{r['n_total']} ({r['prop_underdamped']:.1%})")
        print(f"      Overdamped:  {r['n_overdamped']}/{r['n_total']}")
        print(f"      Mean ζ: {r['mean_zeta']:.3f} ± {r['std_zeta']:.3f}")
        print(f"      Mean R²: {r['mean_r_squared']:.3f}")

        if r['p_value'] < 0.05 and r['prop_underdamped'] > 0.33:
            print(f"      ✓ Significant oscillation (p = {r['p_value']:.4f})")
            if r['mean_period_days']:
                print(f"      ✓ Mean period: {r['mean_period_days']:.1f} days")
        else:
            print(f"      ○ No significant oscillation (p = {r['p_value']:.4f})")

    # Visualize
    print("\n[4/4] Creating visualizations...")
    visualize_results(all_fits, results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_zetas = []
    all_regimes = []
    for fits in all_fits.values():
        for f in fits:
            if f is not None:
                all_zetas.append(f.zeta)
                all_regimes.append(f.regime)

    n_under = sum(1 for r in all_regimes if r == 'underdamped')
    n_over = sum(1 for r in all_regimes if r == 'overdamped')

    print(f"\nAcross all variables:")
    print(f"  Total fits: {len(all_regimes)}")
    print(f"  Underdamped: {n_under} ({n_under/len(all_regimes):.1%})")
    print(f"  Overdamped: {n_over} ({n_over/len(all_regimes):.1%})")
    print(f"  Mean damping ratio ζ: {np.mean(all_zetas):.3f}")

    if np.mean(all_zetas) < 1:
        print("""
✓ Evidence for UNDERDAMPED dynamics (oscillation)!
  This matches Kaplowitz et al. (1983) finding for persuasion.
  Political attitudes appear to oscillate around equilibrium.
""")
    else:
        print("""
○ Evidence for OVERDAMPED dynamics (no oscillation)
  Unlike Kaplowitz et al. (1983), political attitudes
  appear to decay monotonically without oscillation.

  Possible explanations:
  - 2-week sampling interval too coarse for oscillations
  - Political attitudes more stable than immediate persuasion
  - Different cognitive processes involved
""")

    return all_fits, results


def visualize_results(all_fits: Dict, results: Dict):
    """Create visualizations of the Hamiltonian analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Distribution of damping ratios
    ax = axes[0, 0]
    all_zetas = []
    for fits in all_fits.values():
        for f in fits:
            if f is not None:
                all_zetas.append(f.zeta)

    ax.hist(all_zetas, bins=30, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(1.0, color='red', ls='--', lw=2, label='Critical damping (ζ=1)')
    ax.axvline(np.mean(all_zetas), color='green', ls='-', lw=2,
               label=f'Mean ζ = {np.mean(all_zetas):.2f}')
    ax.set_xlabel('Damping Ratio ζ', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('A. Distribution of Damping Ratios\n(ζ<1: oscillation, ζ>1: overdamped)', fontsize=11)
    ax.legend()

    # Panel B: Regime breakdown
    ax = axes[0, 1]
    regimes = {'Underdamped': 0, 'Critically Damped': 0, 'Overdamped': 0}
    for fits in all_fits.values():
        for f in fits:
            if f is not None:
                if f.regime == 'underdamped':
                    regimes['Underdamped'] += 1
                elif f.regime == 'critically_damped':
                    regimes['Critically Damped'] += 1
                else:
                    regimes['Overdamped'] += 1

    colors = ['coral', 'gold', 'steelblue']
    ax.bar(regimes.keys(), regimes.values(), color=colors, edgecolor='black')
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('B. Dynamic Regimes', fontsize=11)

    # Panel C: Example trajectories
    ax = axes[1, 0]

    # Find best underdamped fit
    best_under = None
    best_over = None
    for fits in all_fits.values():
        for f in fits:
            if f is not None:
                if f.regime == 'underdamped' and f.r_squared > 0.3:
                    if best_under is None or f.r_squared > best_under.r_squared:
                        best_under = f
                if f.regime == 'overdamped' and f.r_squared > 0.3:
                    if best_over is None or f.r_squared > best_over.r_squared:
                        best_over = f

    t = np.arange(26) * 14  # Days

    if best_under is not None and len(best_under.trajectory) == 26:
        ax.plot(t, best_under.trajectory, 'o-', color='coral',
                label=f'Underdamped (ζ={best_under.zeta:.2f})', alpha=0.7)
        ax.plot(t, best_under.fitted, '--', color='coral', alpha=0.5)

    if best_over is not None and len(best_over.trajectory) == 26:
        ax.plot(t, best_over.trajectory, 's-', color='steelblue',
                label=f'Overdamped (ζ={best_over.zeta:.2f})', alpha=0.7)
        ax.plot(t, best_over.fitted, '--', color='steelblue', alpha=0.5)

    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Attitude', fontsize=12)
    ax.set_title('C. Example Trajectories', fontsize=11)
    ax.legend()

    # Panel D: R² vs ζ
    ax = axes[1, 1]
    zetas = []
    r2s = []
    for fits in all_fits.values():
        for f in fits:
            if f is not None:
                zetas.append(f.zeta)
                r2s.append(f.r_squared)

    ax.scatter(zetas, r2s, alpha=0.3, c='steelblue', s=20)
    ax.axvline(1.0, color='red', ls='--', alpha=0.5)
    ax.set_xlabel('Damping Ratio ζ', fontsize=12)
    ax.set_ylabel('R²', fontsize=12)
    ax.set_title('D. Fit Quality vs Damping', fontsize=11)

    plt.suptitle('Hamiltonian Dynamics: 26-Wave Political Panel\n(Testing Kaplowitz-Fink Model)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'political_panel_hamiltonian.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"      Saved figure to: {save_path}")


if __name__ == '__main__':
    analyze_panel()