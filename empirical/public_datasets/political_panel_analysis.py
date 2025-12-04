# -*- coding: utf-8 -*-
"""
Hamiltonian Belief Dynamics Analysis: 26-Wave Political Panel
==============================================================

Tests the Kaplowitz-Fink second-order model on the Brandt et al. (2021)
26-wave political psychology panel dataset.

Model: m*x''(t) + gamma*x'(t) + k*x(t) = F(t)

Key predictions:
1. Underdamped (zeta < 1): Oscillation around equilibrium
2. Critically damped (zeta = 1): Fastest return without overshoot
3. Overdamped (zeta > 1): Slow exponential decay

Where zeta = gamma / (2*sqrt(m*k)) is the damping ratio.

Data: https://osf.io/3pwvb/
Paper: https://openpsychologydata.metajnl.com/articles/10.5334/jopd.54

Author: Hamiltonian-VFE Team
Date: December 2025
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import glob as glob_module

# Add project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(SCRIPT_DIR, 'political_panel')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Attitude variables in the dataset (7-point scales)
ATTITUDE_VARIABLES = [
    'def',      # Defense spending
    'crime',    # Crime policy
    'terror',   # Terrorism policy
    'poor',     # Aid to poor
    'health',   # Healthcare
    'econ',     # Economic policy
    'abort',    # Abortion
    'unemploy', # Unemployment
    'blkaid',   # Aid to Black Americans
    'adopt',    # Adoption policy
    'imm',      # Immigration
    'vaccines', # Vaccine policy
    'guns',     # Gun control
    'djt',      # Trump approval
    'ideo',     # Ideology
    'climate',  # Climate change
]


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

    Data format: Long format with columns:
    - id: Participant ID
    - wave: Wave number (1-26)
    - Attitude variables: def, crime, terror, poor, health, econ, etc.
    """
    # Search recursively for CSV files
    csv_files = glob_module.glob(os.path.join(data_dir, '**', '*.csv'), recursive=True)

    if csv_files:
        # Prefer larger files (more likely to be the main data)
        csv_files.sort(key=lambda x: os.path.getsize(x), reverse=True)
        fpath = csv_files[0]
        print(f"Loading data from: {fpath}")
        return pd.read_csv(fpath)

    raise FileNotFoundError(
        f"No data files found in {data_dir}. "
        f"Please download from https://osf.io/3pwvb/"
    )


def extract_time_series(df: pd.DataFrame, variable: str) -> Dict[str, np.ndarray]:
    """
    Extract time series for a variable across all waves.

    The data is in LONG format with columns:
    - id: Participant identifier
    - wave: Wave number (1-26)
    - [variable]: The attitude measure

    Args:
        df: Panel dataframe in long format
        variable: Column name like 'djt', 'ideo', 'terror', etc.

    Returns:
        Dict mapping participant_id -> time series array (length 26)
    """
    if variable not in df.columns:
        return {}

    # Check if data is in long format (has 'wave' column)
    if 'wave' not in df.columns:
        print(f"Warning: No 'wave' column found - cannot extract time series")
        return {}

    if 'id' not in df.columns:
        # Try alternative id columns
        id_col = None
        for col in ['ResponseId', 'participant_id', 'ID', 'subject']:
            if col in df.columns:
                id_col = col
                break
        if id_col is None:
            print(f"Warning: No participant ID column found")
            return {}
        df = df.rename(columns={id_col: 'id'})

    # Pivot to wide format: rows = participants, columns = waves
    try:
        pivot_df = df.pivot(index='id', columns='wave', values=variable)
    except (KeyError, ValueError) as e:
        print(f"Warning: Could not pivot data for '{variable}': {e}")
        return {}

    # Extract per-participant time series
    series = {}

    for pid in pivot_df.index:
        values = pivot_df.loc[pid].values.astype(float)

        # Only include if we have enough non-null values
        valid = ~np.isnan(values)
        if np.sum(valid) >= 10:  # At least 10 waves
            series[str(pid)] = values

    return series


def damped_oscillator_solution(t: np.ndarray, x0: float, v0: float,
                                m: float, gamma: float, k: float) -> np.ndarray:
    """
    Analytical solution to damped harmonic oscillator.

    m*x'' + gamma*x' + k*x = 0

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
        B = (v0 + zeta * omega_n * x0) / omega_d if omega_d > 0 else 0
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
) -> Optional[HamiltonianFit]:
    """
    Fit second-order Hamiltonian dynamics to a time series.

    Model: m*x'' + gamma*x' + k*(x - x_eq) = 0

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
    v0 = (y[1] - y[0]) / (t[1] - t[0]) if len(y) > 1 and t[1] != t[0] else 0

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
        'median_zeta': np.median(zetas) if zetas else None,
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
    m*x'' + gamma*x' + k*x = F(t)

Predictions:
- Underdamped (zeta < 1): Beliefs oscillate around equilibrium
- Overdamped (zeta > 1): Beliefs decay monotonically

Kaplowitz et al. (1983) found zeta < 1 (oscillation) for persuasion.
Question: Do political attitudes show the same pattern?
    """)

    # Load data
    print("\n[1/4] Loading data...")
    try:
        df = load_panel_data(data_dir)
        print(f"      Loaded {len(df)} rows, {len(df.columns)} columns")
    except FileNotFoundError as e:
        print(f"\n ERROR: {e}")
        print("\n Please download the data from https://osf.io/3pwvb/")
        print(" and place CSV files in:", data_dir)
        return None

    # Identify attitude variables
    print("\n[2/4] Identifying attitude variables...")
    print(f"      Data shape: {df.shape}")

    if 'id' in df.columns:
        print(f"      Unique participants: {df['id'].nunique()}")
    if 'wave' in df.columns:
        print(f"      Waves: {sorted(df['wave'].unique())}")

    # Find which attitude variables are in the data
    available_vars = [v for v in ATTITUDE_VARIABLES if v in df.columns]
    print(f"      Found {len(available_vars)} attitude variables: {available_vars}")

    if not available_vars:
        print("\n ERROR: No attitude variables found in the data.")
        print(f" Expected columns like: {ATTITUDE_VARIABLES[:5]}")
        print(f" Found columns: {list(df.columns[:20])}")
        return None

    # Fit Hamiltonian dynamics to each variable
    all_fits = {}

    for var in available_vars:
        series_dict = extract_time_series(df, var)

        if len(series_dict) >= 10:  # Enough participants
            print(f"\n   Analyzing '{var}' ({len(series_dict)} participants)...")

            fits = []
            for pid, ts in series_dict.items():
                fit = fit_hamiltonian_dynamics(ts, dt=14.0)
                if fit is not None:
                    fit.participant_id = pid
                    fit.variable = var
                    fits.append(fit)

            if fits:
                all_fits[var] = fits
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
        print(f"      Median zeta: {r['median_zeta']:.3f}")
        print(f"      Mean R^2: {r['mean_r_squared']:.3f}")

        if r['p_value'] < 0.05 and r['prop_underdamped'] > 0.33:
            print(f"      * Significant oscillation (p = {r['p_value']:.4f})")
            if r['mean_period_days']:
                print(f"      * Mean period: {r['mean_period_days']:.1f} days")
        else:
            print(f"      o No significant oscillation (p = {r['p_value']:.4f})")

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
    print(f"  Median damping ratio zeta: {np.median(all_zetas):.3f}")
    print(f"  Mean damping ratio zeta: {np.mean(all_zetas):.3f} (may be skewed by outliers)")

    # Use proportion of underdamped as the key metric
    prop_underdamped = n_under / len(all_regimes)
    median_zeta = np.median(all_zetas)

    if prop_underdamped > 0.5 or median_zeta < 1:
        # Compute mean period from results
        all_periods = []
        for res in results.values():
            if res.get('oscillation_periods'):
                all_periods.extend(res['oscillation_periods'])
        mean_period = np.mean(all_periods) if all_periods else 0

        print(f"""
* Evidence for UNDERDAMPED dynamics (oscillation)!
  This matches Kaplowitz et al. (1983) finding for persuasion.

  Key findings:
  - {prop_underdamped:.1%} of individual trajectories show oscillation (zeta < 1)
  - Median damping ratio zeta = {median_zeta:.3f}
  - Mean oscillation period: {mean_period:.0f} days

  Interpretation:
  Political attitudes oscillate around equilibrium points with
  characteristic periods of ~2-4 months. This suggests a
  "pendulum" dynamic in belief updating rather than monotonic decay.
""")
    else:
        print("""
o Evidence for OVERDAMPED dynamics (no oscillation)
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

    # Cap zetas for visualization (outliers distort histogram)
    zetas_capped = np.clip(all_zetas, 0, 5)
    ax.hist(zetas_capped, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax.axvline(1.0, color='red', ls='--', lw=2, label='Critical damping (zeta=1)')
    ax.axvline(np.median(all_zetas), color='green', ls='-', lw=2,
               label=f'Median zeta = {np.median(all_zetas):.2f}')
    ax.set_xlabel('Damping Ratio zeta (capped at 5)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('A. Distribution of Damping Ratios\n(zeta<1: oscillation, zeta>1: overdamped)', fontsize=11)
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

    n_waves = 26
    t = np.arange(n_waves) * 14  # Days

    if best_under is not None and len(best_under.trajectory) <= n_waves:
        traj_len = len(best_under.trajectory)
        ax.plot(t[:traj_len], best_under.trajectory, 'o-', color='coral',
                label=f'Underdamped (zeta={best_under.zeta:.2f})', alpha=0.7)
        ax.plot(t[:traj_len], best_under.fitted, '--', color='coral', alpha=0.5)

    if best_over is not None and len(best_over.trajectory) <= n_waves:
        traj_len = len(best_over.trajectory)
        ax.plot(t[:traj_len], best_over.trajectory, 's-', color='steelblue',
                label=f'Overdamped (zeta={best_over.zeta:.2f})', alpha=0.7)
        ax.plot(t[:traj_len], best_over.fitted, '--', color='steelblue', alpha=0.5)

    ax.set_xlabel('Time (days)', fontsize=12)
    ax.set_ylabel('Attitude', fontsize=12)
    ax.set_title('C. Example Trajectories', fontsize=11)
    if best_under is not None or best_over is not None:
        ax.legend()

    # Panel D: R^2 vs zeta
    ax = axes[1, 1]
    zetas = []
    r2s = []
    for fits in all_fits.values():
        for f in fits:
            if f is not None:
                zetas.append(f.zeta)
                r2s.append(f.r_squared)

    # Cap zetas for visualization
    zetas_plot = np.clip(zetas, 0, 5)
    ax.scatter(zetas_plot, r2s, alpha=0.3, c='steelblue', s=20)
    ax.axvline(1.0, color='red', ls='--', alpha=0.5)
    ax.set_xlabel('Damping Ratio zeta (capped at 5)', fontsize=12)
    ax.set_ylabel('R^2', fontsize=12)
    ax.set_title('D. Fit Quality vs Damping', fontsize=11)

    plt.suptitle('Hamiltonian Dynamics: 26-Wave Political Panel\n(Testing Kaplowitz-Fink Model)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'political_panel_hamiltonian.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"      Saved figure to: {save_path}")


if __name__ == '__main__':
    analyze_panel()
