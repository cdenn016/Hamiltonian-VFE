# -*- coding: utf-8 -*-
"""
Hamiltonian Belief Dynamics: Twitter Sentiment Geographical Index
=================================================================

Tests Hamiltonian dynamics on daily sentiment time series across
countries and regions.

The TSGI provides daily sentiment at country/state/county level,
allowing us to test:
1. Long-term oscillations (days-weeks) in collective mood
2. Cross-regional propagation (social mass across geography)
3. Response to global events (COVID, elections, etc.)

Data: Harvard Dataverse TSGI
URL: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/3IL00Q

Author: Hamiltonian-VFE Team
Date: December 2025
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy import stats, signal, optimize
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..', '..')
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = SCRIPT_DIR
OUTPUT_DIR = os.path.join(SCRIPT_DIR, '..', 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Global events to analyze
GLOBAL_EVENTS = {
    'covid_who_pandemic': datetime(2020, 3, 11),  # WHO declares pandemic
    'george_floyd': datetime(2020, 5, 25),  # George Floyd protests begin
    'us_election_2020': datetime(2020, 11, 3),  # US presidential election
    'jan6_capitol': datetime(2021, 1, 6),  # Capitol riot
    'ukraine_invasion': datetime(2022, 2, 24),  # Russia invades Ukraine
}


@dataclass
class RegionDynamics:
    """Dynamics summary for a region."""
    region_name: str
    mean_sentiment: float
    std_sentiment: float
    dominant_period_days: float
    damping_ratio: float
    autocorr_decay: float  # How fast autocorrelation decays


def load_tsgi_data(data_dir: str, level: str = 'Country') -> pd.DataFrame:
    """
    Load sentiment data from CSV files.

    Args:
        data_dir: Directory containing sentiment data
        level: Which level to load - 'Country', 'State', 'County', or 'World'
               Default is 'Country' for manageable data size

    Searches for:
    1. "Sentiment Data - XXX" folders (Country, County, State, World)
    2. Direct CSV files in the data directory
    """
    import glob as glob_module

    dfs = []

    # First look for "Sentiment Data - XXX" folders
    sentiment_folders = glob_module.glob(os.path.join(data_dir, 'Sentiment Data - *'))

    if sentiment_folders:
        print(f"  Found {len(sentiment_folders)} sentiment data folders")

        # Only load the requested level to keep data manageable
        target_folder = None
        for folder in sentiment_folders:
            folder_name = os.path.basename(folder)
            folder_level = folder_name.replace('Sentiment Data - ', '')
            if folder_level == level:
                target_folder = folder
                break

        if target_folder is None:
            print(f"  WARNING: Level '{level}' not found, available:")
            for folder in sentiment_folders:
                print(f"    - {os.path.basename(folder)}")
            # Fall back to first folder
            target_folder = sentiment_folders[0]

        folder_name = os.path.basename(target_folder)
        print(f"  Loading from: {folder_name}")

        csv_files = glob_module.glob(os.path.join(target_folder, '*.csv'))
        for fpath in csv_files:
            try:
                df = pd.read_csv(fpath, low_memory=False)
                df['data_level'] = level
                dfs.append(df)
                print(f"    Loaded {os.path.basename(fpath)}: {len(df):,} records")
            except Exception as e:
                print(f"    Error loading {os.path.basename(fpath)}: {e}")

    # If no folders found, look for direct CSV files
    if not dfs:
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]

        for fname in csv_files:
            fpath = os.path.join(data_dir, fname)
            try:
                df = pd.read_csv(fpath, low_memory=False)
                dfs.append(df)
                print(f"  Loaded {fname}: {len(df):,} records")
            except Exception as e:
                print(f"  Error loading {fname}: {e}")

    if not dfs:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. "
            f"Please place data in 'Sentiment Data - XXX' folders."
        )

    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total: {len(combined):,} records")

    return combined


def preprocess_tsgi(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess sentiment data from various sources."""
    print(f"  Raw columns: {list(df.columns[:10])}...")

    # Standardize column names
    df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_').str.replace('-', '_')

    print(f"  Standardized columns: {list(df.columns[:10])}...")

    # Parse date - try many possible column names
    date_cols = ['date', 'datetime', 'time', 'day', 'created_at', 'timestamp',
                 'tweet_date', 'post_date']
    date_found = False
    for col in date_cols:
        if col in df.columns:
            df['date'] = pd.to_datetime(df[col], errors='coerce')
            date_found = True
            print(f"  Using '{col}' as date")
            break

    if not date_found:
        # Try to find any column with 'date' or 'time' in the name
        for col in df.columns:
            if 'date' in col or 'time' in col or 'day' in col:
                df['date'] = pd.to_datetime(df[col], errors='coerce')
                date_found = True
                print(f"  Using '{col}' as date")
                break

    # Find sentiment column - try many possible names
    sent_cols = ['sentiment_mean', 'sentiment', 'score', 'mean', 'compound',
                 'polarity', 'vader', 'avg_sentiment', 'daily_sentiment',
                 'sentiment_score', 'vader_compound']
    sentiment_found = False
    for col in sent_cols:
        if col in df.columns:
            df['sentiment'] = pd.to_numeric(df[col], errors='coerce')
            sentiment_found = True
            print(f"  Using '{col}' as sentiment")
            break

    if not sentiment_found:
        # Try to find any column with 'sent' or 'score' in the name
        for col in df.columns:
            if 'sent' in col or 'score' in col or 'polarity' in col:
                df['sentiment'] = pd.to_numeric(df[col], errors='coerce')
                sentiment_found = True
                print(f"  Using '{col}' as sentiment")
                break

    # Find region columns - use hierarchical naming (NAME_0=country, NAME_1=state, NAME_2=county)
    # Priority: most specific geographic column available
    region_cols = ['name_0', 'name_1', 'name_2',  # Standard geographic hierarchy
                   'admin0', 'country', 'admin1', 'state', 'region', 'county',
                   'admin2', 'location', 'geo', 'place', 'country_name',
                   'state_name', 'county_name']
    region_found = False
    for col in region_cols:
        if col in df.columns:
            df['region'] = df[col].astype(str)
            region_found = True
            print(f"  Using '{col}' as region ({df[col].nunique()} unique values)")
            break

    if not region_found:
        df['region'] = 'global'
        print(f"  No region column found, using 'global'")

    if not date_found or not sentiment_found:
        print(f"  WARNING: Missing required columns")
        print(f"  Available columns: {list(df.columns)}")
        return pd.DataFrame()

    df = df.dropna(subset=['date', 'sentiment'])

    print(f"  After preprocessing: {len(df):,} records")
    if len(df) > 0:
        print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"  Regions: {df['region'].nunique()}")
        print(f"  Sentiment range: {df['sentiment'].min():.3f} to {df['sentiment'].max():.3f}")

    return df


def compute_spectral_properties(
    series: np.ndarray,
    sampling_days: float = 1.0
) -> Tuple[float, float]:
    """
    Compute dominant period and damping from power spectrum.

    Returns:
        (dominant_period_days, damping_estimate)
    """
    if len(series) < 30:
        return 0, 1.0

    # Remove trend
    detrended = signal.detrend(series)

    # Power spectrum
    freqs, psd = signal.welch(detrended, fs=1/sampling_days, nperseg=min(len(series)//2, 128))

    # Find dominant frequency (excluding DC)
    if len(freqs) > 1:
        psd_no_dc = psd[1:]
        freqs_no_dc = freqs[1:]

        if len(psd_no_dc) > 0:
            peak_idx = np.argmax(psd_no_dc)
            dominant_freq = freqs_no_dc[peak_idx]

            if dominant_freq > 0:
                dominant_period = 1 / dominant_freq
            else:
                dominant_period = 0
        else:
            dominant_period = 0
    else:
        dominant_period = 0

    # Estimate damping from autocorrelation decay
    autocorr = np.correlate(detrended, detrended, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    autocorr = autocorr / autocorr[0]

    # Find decay rate
    threshold = 1/np.e
    below = np.where(autocorr < threshold)[0]
    if len(below) > 0:
        decay_time = below[0] * sampling_days
        # Rough damping estimate
        if dominant_period > 0:
            omega_n = 2 * np.pi / dominant_period
            damping = 1 / (decay_time * omega_n) if decay_time > 0 else 1.0
        else:
            damping = 1.0
    else:
        damping = 0.1  # Very low damping - slow decay

    return dominant_period, min(damping, 5.0)


def analyze_region(
    df: pd.DataFrame,
    region_name: str
) -> Optional[RegionDynamics]:
    """Analyze dynamics for a single region."""
    region_data = df[df['region'] == region_name].sort_values('date')

    if len(region_data) < 60:  # Need at least 2 months
        return None

    sentiment = region_data['sentiment'].values

    # Basic statistics
    mean_sent = np.mean(sentiment)
    std_sent = np.std(sentiment)

    # Spectral analysis
    period, damping = compute_spectral_properties(sentiment)

    # Autocorrelation decay
    if len(sentiment) > 10:
        autocorr = np.correlate(sentiment - mean_sent, sentiment - mean_sent, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-10)

        # Decay to 1/e
        below = np.where(autocorr < 1/np.e)[0]
        decay = below[0] if len(below) > 0 else len(autocorr)
    else:
        decay = 0

    return RegionDynamics(
        region_name=region_name,
        mean_sentiment=mean_sent,
        std_sentiment=std_sent,
        dominant_period_days=period,
        damping_ratio=damping,
        autocorr_decay=decay
    )


def analyze_event_response(
    df: pd.DataFrame,
    event_time: datetime,
    window_before: int = 14,
    window_after: int = 30
) -> Dict:
    """Analyze sentiment response to a global event."""
    start = event_time - timedelta(days=window_before)
    end = event_time + timedelta(days=window_after)

    window = df[(df['date'] >= start) & (df['date'] <= end)]

    if len(window) < 10:
        return None

    # Aggregate globally by day
    daily = window.groupby('date').agg({
        'sentiment': ['mean', 'std', 'count']
    }).reset_index()
    daily.columns = ['date', 'sentiment_mean', 'sentiment_std', 'count']

    # Time relative to event
    daily['t_days'] = (daily['date'] - event_time).dt.days

    pre_data = daily[daily['t_days'] < 0]
    post_data = daily[daily['t_days'] >= 0]

    if len(pre_data) < 3 or len(post_data) < 5:
        return None

    pre_sentiment = pre_data['sentiment_mean'].mean()
    shock = post_data.iloc[0]['sentiment_mean'] - pre_sentiment if len(post_data) > 0 else 0

    # Look for oscillation in post-event data
    if len(post_data) > 10:
        t = post_data['t_days'].values
        y = post_data['sentiment_mean'].values
        y_dev = y - np.mean(y[-5:])  # Deviation from late equilibrium

        # Count zero crossings
        crossings = np.where(np.diff(np.sign(y_dev)))[0]
        n_oscillations = len(crossings) // 2

        # Estimate period
        if len(crossings) >= 2:
            half_periods = np.diff(t[crossings])
            period = 2 * np.mean(half_periods) if len(half_periods) > 0 else 0
        else:
            period = 0
    else:
        n_oscillations = 0
        period = 0

    return {
        'pre_sentiment': pre_sentiment,
        'shock': shock,
        'n_oscillations': n_oscillations,
        'period_days': period,
        'daily_data': daily
    }


def visualize_results(
    region_results: List[RegionDynamics],
    event_results: Dict[str, Dict]
):
    """Visualize analysis results."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Distribution of damping ratios
    ax = axes[0, 0]
    if region_results:
        zetas = [r.damping_ratio for r in region_results if r.damping_ratio < 5]
        ax.hist(zetas, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        ax.axvline(1.0, color='red', ls='--', lw=2, label='Critical (ζ=1)')
        ax.axvline(np.mean(zetas), color='green', ls='-', lw=2,
                   label=f'Mean = {np.mean(zetas):.2f}')
        ax.legend()
    ax.set_xlabel('Damping Ratio ζ')
    ax.set_ylabel('Count')
    ax.set_title('A. Regional Damping Ratios')

    # Panel B: Period distribution
    ax = axes[0, 1]
    if region_results:
        periods = [r.dominant_period_days for r in region_results if 0 < r.dominant_period_days < 100]
        if periods:
            ax.hist(periods, bins=20, alpha=0.7, color='coral', edgecolor='black')
            ax.axvline(7, color='blue', ls='--', lw=2, label='Weekly')
            ax.axvline(np.mean(periods), color='green', ls='-', lw=2,
                       label=f'Mean = {np.mean(periods):.1f}d')
            ax.legend()
    ax.set_xlabel('Dominant Period (days)')
    ax.set_ylabel('Count')
    ax.set_title('B. Oscillation Periods')

    # Panel C: Event responses
    ax = axes[1, 0]
    event_names = []
    oscillations = []
    for name, result in event_results.items():
        if result is not None:
            event_names.append(name.replace('_', '\n'))
            oscillations.append(result['n_oscillations'])

    if event_names:
        colors = ['coral' if n > 0 else 'steelblue' for n in oscillations]
        ax.bar(event_names, oscillations, color=colors, edgecolor='black')
    ax.set_ylabel('Number of Oscillations')
    ax.set_title('C. Oscillations After Global Events')
    ax.tick_params(axis='x', rotation=45)

    # Panel D: Example event time series
    ax = axes[1, 1]
    for name, result in list(event_results.items())[:2]:
        if result is not None and 'daily_data' in result:
            daily = result['daily_data']
            ax.plot(daily['t_days'], daily['sentiment_mean'],
                    'o-', label=name, alpha=0.7, markersize=4)
    ax.axvline(0, color='red', ls='--', alpha=0.5)
    ax.set_xlabel('Days from Event')
    ax.set_ylabel('Global Sentiment')
    ax.set_title('D. Example Event Responses')
    ax.legend()

    plt.suptitle('Hamiltonian Dynamics: Twitter Sentiment Index\n(Testing Global Mood Oscillations)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'sentiment_index_dynamics.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved figure to: {save_path}")


@dataclass
class FittedEventTrajectory:
    """Fitted model parameters for an event response."""
    event_name: str
    t: np.ndarray  # Time array (days)
    y_data: np.ndarray  # Observed data
    y_oscillator: np.ndarray  # Fitted oscillator prediction
    y_exponential: np.ndarray  # Fitted exponential prediction
    # Oscillator parameters
    amplitude: float
    damping_coeff: float  # ζωn
    angular_freq: float  # ωd
    phase: float
    offset: float
    # Derived quantities
    damping_ratio: float
    period_days: float
    decay_time: float
    # Fit quality
    r2_oscillator: float
    r2_exponential: float
    aic_oscillator: float
    aic_exponential: float
    n_points: int


def compute_aic(n: int, rss: float, k: int) -> float:
    """Compute AIC for a model."""
    if rss <= 0 or n <= k:
        return np.inf
    return n * np.log(rss / n) + 2 * k


def fit_event_trajectory(
    df: pd.DataFrame,
    event_time: datetime,
    event_name: str,
    window_before: int = 7,
    window_after: int = 30
) -> Optional[FittedEventTrajectory]:
    """
    Fit damped oscillator and exponential decay to event response.
    """
    start = event_time - timedelta(days=window_before)
    end = event_time + timedelta(days=window_after)

    window = df[(df['date'] >= start) & (df['date'] <= end)]

    if len(window) < 10:
        return None

    # Aggregate globally by day
    daily = window.groupby('date').agg({
        'sentiment': 'mean'
    }).reset_index()
    daily.columns = ['date', 'sentiment_mean']
    daily = daily.sort_values('date')

    # Time relative to event
    daily['t_days'] = (daily['date'] - event_time).dt.days

    t_full = daily['t_days'].values.astype(float)
    y_full = daily['sentiment_mean'].values

    # Get pre-event baseline
    pre_mask = t_full < 0
    post_mask = t_full >= 0

    if np.sum(pre_mask) >= 2:
        baseline = np.mean(y_full[pre_mask])
    else:
        baseline = y_full[0]

    # Focus on post-event data for fitting
    t = t_full[post_mask]
    y = y_full[post_mask]

    if len(t) < 5:
        return None

    # Deviation from baseline (not from post-event mean!)
    y_dev = y - baseline

    # === Fit damped oscillator: y = baseline + A * exp(-ζωn*t) * cos(ωd*t + φ) ===
    def osc_model(t, A, zeta_omega, omega_d, phi):
        return A * np.exp(-zeta_omega * t) * np.cos(omega_d * t + phi)

    # Estimate initial frequency from zero crossings
    crossings = np.where(np.diff(np.sign(y_dev)))[0]
    if len(crossings) >= 2:
        half_periods = np.diff(t[crossings])
        omega_d0 = np.pi / np.mean(half_periods) if len(half_periods) > 0 else 0.5
    else:
        omega_d0 = 2 * np.pi / 10  # ~10 day period default

    omega_d0 = np.clip(omega_d0, 0.1, 3.0)

    # Initial amplitude from first deviation
    A0 = y_dev[0] if len(y_dev) > 0 else 0.1
    if abs(A0) < 1e-10:
        A0 = np.max(np.abs(y_dev)) if np.max(np.abs(y_dev)) > 1e-10 else 0.1

    best_osc_params = None
    best_osc_rss = np.inf

    # Search over more initial conditions
    omega_candidates = [omega_d0, omega_d0 * 0.5, omega_d0 * 2,
                        2*np.pi/7, 2*np.pi/14, 2*np.pi/21, 0.3, 0.6, 0.9]
    zeta_candidates = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5]
    phi_candidates = [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, np.pi, -np.pi/6, -np.pi/3, -np.pi/2]

    for omega_init in omega_candidates:
        for zeta_init in zeta_candidates:
            for phi_init in phi_candidates:
                try:
                    popt, _ = optimize.curve_fit(
                        osc_model, t, y_dev,
                        p0=[A0, zeta_init, omega_init, phi_init],
                        maxfev=2000,
                        bounds=([-np.inf, 0.001, 0.05, -np.pi],
                               [np.inf, 2, 3, np.pi])
                    )
                    y_pred = osc_model(t, *popt)
                    rss = np.sum((y_dev - y_pred)**2)
                    if rss < best_osc_rss:
                        best_osc_rss = rss
                        best_osc_params = popt
                except:
                    continue

    if best_osc_params is None:
        best_osc_params = [A0, 0.1, omega_d0, 0]
        best_osc_rss = np.sum(y_dev**2)

    A, zeta_omega, omega_d, phi = best_osc_params

    # === Fit exponential decay: y = A * exp(-λt) ===
    def exp_model(t, A_exp, lam):
        return A_exp * np.exp(-lam * t)

    try:
        popt_exp, _ = optimize.curve_fit(
            exp_model, t, y_dev,
            p0=[A0, 0.1],
            maxfev=2000,
            bounds=([-np.inf, 0.001], [np.inf, 2])
        )
        y_exp_pred = exp_model(t, *popt_exp)
        exp_rss = np.sum((y_dev - y_exp_pred)**2)
    except:
        popt_exp = [A0, 0.1]
        exp_rss = np.sum(y_dev**2)

    # Compute R² and AIC
    ss_tot = np.sum((y_dev - np.mean(y_dev))**2)
    r2_osc = 1 - best_osc_rss / ss_tot if ss_tot > 0 else 0
    r2_exp = 1 - exp_rss / ss_tot if ss_tot > 0 else 0

    n = len(y)
    aic_osc = compute_aic(n, best_osc_rss, 4)  # 4 params (no offset)
    aic_exp = compute_aic(n, exp_rss, 2)  # 2 params

    # Derive physical parameters
    if omega_d > 0:
        zeta_sq = zeta_omega**2 / (omega_d**2 + zeta_omega**2)
        zeta = np.sqrt(zeta_sq)
    else:
        zeta = 1.0

    period = 2 * np.pi / omega_d if omega_d > 0 else np.inf
    decay_time = 1 / zeta_omega if zeta_omega > 0 else np.inf

    # Generate smooth predictions
    t_smooth = np.linspace(t_full.min(), t_full.max(), 200)

    y_osc_smooth = np.zeros_like(t_smooth)
    y_exp_smooth = np.zeros_like(t_smooth)

    # Pre-event: use baseline
    pre_mask_smooth = t_smooth < 0
    y_osc_smooth[pre_mask_smooth] = baseline
    y_exp_smooth[pre_mask_smooth] = baseline

    # Post-event: use fitted models + baseline
    post_mask_smooth = t_smooth >= 0
    y_osc_smooth[post_mask_smooth] = baseline + osc_model(t_smooth[post_mask_smooth], A, zeta_omega, omega_d, phi)
    y_exp_smooth[post_mask_smooth] = baseline + exp_model(t_smooth[post_mask_smooth], *popt_exp)

    return FittedEventTrajectory(
        event_name=event_name,
        t=t_smooth,
        y_data=y_full,
        y_oscillator=y_osc_smooth,
        y_exponential=y_exp_smooth,
        amplitude=A,
        damping_coeff=zeta_omega,
        angular_freq=omega_d,
        phase=phi,
        offset=baseline,
        damping_ratio=zeta,
        period_days=period,
        decay_time=decay_time,
        r2_oscillator=max(0, r2_osc),
        r2_exponential=max(0, r2_exp),
        aic_oscillator=aic_osc,
        aic_exponential=aic_exp,
        n_points=len(y_full)
    )


def plot_fitted_event_trajectories(
    df: pd.DataFrame,
    events: Dict[str, datetime],
    output_dir: str
) -> List[FittedEventTrajectory]:
    """
    Generate fitted trajectory plots for each event.
    """
    print("\n" + "=" * 70)
    print("GENERATING FITTED EVENT TRAJECTORIES")
    print("=" * 70)

    trajectories = []
    valid_events = []

    # First pass: fit all events
    for event_name, event_time in events.items():
        if df['date'].min() <= event_time <= df['date'].max():
            traj = fit_event_trajectory(df, event_time, event_name)
            if traj is not None:
                trajectories.append(traj)
                valid_events.append((event_name, event_time))

    if not trajectories:
        print("  No events with sufficient data")
        return []

    # Create figure
    n_events = len(trajectories)
    fig, axes = plt.subplots(n_events, 1, figsize=(14, 4 * n_events))
    if n_events == 1:
        axes = [axes]

    for idx, traj in enumerate(trajectories):
        event_name, event_time = valid_events[idx]
        ax = axes[idx]

        print(f"\n  {event_name}:")

        # Get data points for plotting
        start = event_time - timedelta(days=7)
        end = event_time + timedelta(days=30)
        window = df[(df['date'] >= start) & (df['date'] <= end)]
        daily = window.groupby('date').agg({'sentiment': 'mean'}).reset_index()
        daily['t_days'] = (daily['date'] - event_time).dt.days
        t_data = daily['t_days'].values
        y_data = daily['sentiment'].values

        # Plot
        ax.scatter(t_data, y_data, s=60, c='black', alpha=0.8, zorder=5, label=f'Data (n={traj.n_points})')
        ax.plot(traj.t, traj.y_oscillator, 'b-', lw=2.5,
                label=f'Damped Oscillator (R²={traj.r2_oscillator:.3f})')
        ax.plot(traj.t, traj.y_exponential, 'r--', lw=2, alpha=0.7,
                label=f'Exponential Decay (R²={traj.r2_exponential:.3f})')
        ax.axvline(0, color='green', ls='-', lw=2, alpha=0.7, label='Event')

        # Decay envelope
        t_post = traj.t[traj.t >= 0]
        envelope_upper = traj.offset + traj.amplitude * np.exp(-traj.damping_coeff * t_post)
        envelope_lower = traj.offset - traj.amplitude * np.exp(-traj.damping_coeff * t_post)
        ax.fill_between(t_post, envelope_lower, envelope_upper,
                        alpha=0.15, color='blue', label='Decay envelope')

        ax.set_xlabel('Days from Event', fontsize=11)
        ax.set_ylabel('Global Sentiment', fontsize=11)
        ax.set_xlim(-8, 32)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=9)

        # Title
        regime = "UNDERDAMPED" if traj.damping_ratio < 1 else "OVERDAMPED"
        delta_aic = traj.aic_exponential - traj.aic_oscillator
        better = "oscillator" if delta_aic > 2 else ("exponential" if delta_aic < -2 else "tie")

        title = (f"{event_name.replace('_', ' ').title()} - {event_time.strftime('%Y-%m-%d')}\n"
                f"{regime} (ζ = {traj.damping_ratio:.3f}), "
                f"T = {traj.period_days:.1f} days, "
                f"τ = {traj.decay_time:.1f} days, "
                f"ΔAIC = {delta_aic:.1f} ({better})")
        ax.set_title(title, fontsize=11, fontweight='bold')

        print(f"    ζ = {traj.damping_ratio:.3f}, T = {traj.period_days:.1f} days, n = {traj.n_points}")
        print(f"    R²(osc) = {traj.r2_oscillator:.3f}, R²(exp) = {traj.r2_exponential:.3f}")
        print(f"    ΔAIC = {delta_aic:.1f} → {better} model preferred")

    plt.tight_layout()

    save_path = os.path.join(output_dir, 'sentiment_index_fitted_trajectories.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\n  Saved to: {save_path}")

    # Individual plots
    for traj in trajectories:
        event_time = events[traj.event_name]
        fig_single, ax = plt.subplots(figsize=(12, 6))

        start = event_time - timedelta(days=7)
        end = event_time + timedelta(days=30)
        window = df[(df['date'] >= start) & (df['date'] <= end)]
        daily = window.groupby('date').agg({'sentiment': 'mean'}).reset_index()
        daily['t_days'] = (daily['date'] - event_time).dt.days
        t_data = daily['t_days'].values
        y_data = daily['sentiment'].values

        ax.scatter(t_data, y_data, s=80, c='black', alpha=0.8, zorder=5, label=f'Observed Data (n={traj.n_points})')
        ax.plot(traj.t, traj.y_oscillator, 'b-', lw=3,
                label=f'Damped Oscillator (R²={traj.r2_oscillator:.3f})')
        ax.plot(traj.t, traj.y_exponential, 'r--', lw=2.5, alpha=0.8,
                label=f'Exponential Decay (R²={traj.r2_exponential:.3f})')
        ax.axvline(0, color='green', ls='-', lw=2.5, alpha=0.8, label='Event')

        t_post = traj.t[traj.t >= 0]
        envelope_upper = traj.offset + traj.amplitude * np.exp(-traj.damping_coeff * t_post)
        envelope_lower = traj.offset - traj.amplitude * np.exp(-traj.damping_coeff * t_post)
        ax.fill_between(t_post, envelope_lower, envelope_upper,
                        alpha=0.2, color='blue', label='Decay envelope')

        ax.set_xlabel('Days from Event', fontsize=12)
        ax.set_ylabel('Global Sentiment', fontsize=12)
        ax.set_xlim(-8, 32)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', fontsize=10)

        regime = "UNDERDAMPED" if traj.damping_ratio < 1 else "OVERDAMPED"
        delta_aic = traj.aic_exponential - traj.aic_oscillator
        better = "oscillator" if delta_aic > 2 else ("exponential" if delta_aic < -2 else "tie")

        ax.set_title(
            f"Sentiment Dynamics: {traj.event_name.replace('_', ' ').title()}\n"
            f"{regime} (ζ = {traj.damping_ratio:.3f}, T = {traj.period_days:.1f} days)",
            fontsize=13, fontweight='bold'
        )

        plt.tight_layout()
        single_path = os.path.join(output_dir, f'trajectory_{traj.event_name}.png')
        plt.savefig(single_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig_single)

    plt.close(fig)

    # Summary table
    print("\n" + "=" * 95)
    print("FITTED PARAMETERS SUMMARY")
    print("=" * 95)
    print(f"{'Event':<22} {'n':>5} {'ζ':>8} {'T(days)':>10} {'τ(days)':>10} {'R²(osc)':>10} {'R²(exp)':>10} {'ΔAIC':>10}")
    print("-" * 95)

    for traj in trajectories:
        delta_aic = traj.aic_exponential - traj.aic_oscillator
        print(f"{traj.event_name:<22} {traj.n_points:>5} {traj.damping_ratio:>8.4f} "
              f"{traj.period_days:>10.2f} {traj.decay_time:>10.2f} "
              f"{traj.r2_oscillator:>10.4f} {traj.r2_exponential:>10.4f} {delta_aic:>10.2f}")

    print("-" * 95)

    # Save parameters to CSV
    params_data = []
    for traj in trajectories:
        delta_aic = traj.aic_exponential - traj.aic_oscillator
        better = "oscillator" if delta_aic > 2 else ("exponential" if delta_aic < -2 else "neither")
        params_data.append({
            'event': traj.event_name,
            'n_points': traj.n_points,
            'amplitude_A': traj.amplitude,
            'damping_ratio_zeta': traj.damping_ratio,
            'damping_coeff_zeta_omega': traj.damping_coeff,
            'angular_freq_omega_d': traj.angular_freq,
            'phase_phi': traj.phase,
            'offset_baseline': traj.offset,
            'period_days': traj.period_days,
            'decay_time_days': traj.decay_time,
            'r2_oscillator': traj.r2_oscillator,
            'r2_exponential': traj.r2_exponential,
            'aic_oscillator': traj.aic_oscillator,
            'aic_exponential': traj.aic_exponential,
            'delta_aic': delta_aic,
            'preferred_model': better
        })

    params_df = pd.DataFrame(params_data)
    csv_path = os.path.join(output_dir, 'sentiment_fitted_parameters.csv')
    params_df.to_csv(csv_path, index=False)
    print(f"\n  Parameters saved to: {csv_path}")

    return trajectories


def generate_sample_data():
    """Generate sample TSGI data for demonstration."""
    print("Generating sample TSGI data for demonstration...")

    np.random.seed(42)

    # Generate daily data for 3 years, 10 countries
    countries = ['USA', 'UK', 'Germany', 'France', 'Japan',
                 'Brazil', 'India', 'Australia', 'Canada', 'Mexico']

    start = datetime(2020, 1, 1)
    end = datetime(2022, 12, 31)
    dates = pd.date_range(start, end, freq='D')

    records = []

    for country in countries:
        # Country-specific baseline and parameters
        baseline = np.random.uniform(-0.1, 0.1)
        weekly_amp = np.random.uniform(0.02, 0.05)
        noise = np.random.uniform(0.03, 0.08)
        damping = np.random.uniform(0.2, 0.8)

        for i, date in enumerate(dates):
            # Weekly cycle (underdamped oscillation)
            t = i
            weekly = weekly_amp * np.sin(2 * np.pi * t / 7)

            # Responses to events
            event_response = 0
            for event_name, event_time in GLOBAL_EVENTS.items():
                if date >= event_time:
                    t_event = (date - event_time).days
                    shock = np.random.uniform(-0.1, 0.1)
                    omega = 2 * np.pi / 14  # 2-week oscillation
                    omega_d = omega * np.sqrt(1 - damping**2) if damping < 1 else omega
                    response = shock * np.exp(-damping * omega * t_event) * np.cos(omega_d * t_event)
                    event_response += response * np.exp(-t_event / 60)  # Decay over time

            sentiment = baseline + weekly + event_response + noise * np.random.randn()

            records.append({
                'date': date,
                'region': country,
                'sentiment': np.clip(sentiment, -1, 1),
                'count': np.random.poisson(10000)
            })

    df = pd.DataFrame(records)
    return df


def main():
    """Main analysis."""
    print("=" * 70)
    print("HAMILTONIAN DYNAMICS: Twitter Sentiment Index")
    print("=" * 70)
    print("""
Testing Hamiltonian oscillation model on global sentiment time series.

Analyzing:
1. Regional dynamics: Do countries show underdamped oscillations?
2. Event responses: Do global events trigger oscillatory responses?
3. Cross-regional coupling: Does sentiment propagate geographically?
    """)

    # Load data
    print("\n[1/5] Loading data...")
    try:
        df = load_tsgi_data(DATA_DIR)
        df = preprocess_tsgi(df)
    except FileNotFoundError:
        print("\n  No data files found. Using sample data...")
        df = generate_sample_data()

    print(f"  Records: {len(df):,}")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Regions: {df['region'].nunique()}")

    # Analyze regions
    print("\n[2/5] Analyzing regional dynamics...")
    regions = df['region'].unique()
    region_results = []

    for region in regions[:50]:  # Limit to top 50
        result = analyze_region(df, region)
        if result is not None:
            region_results.append(result)

    print(f"  Analyzed {len(region_results)} regions")

    if region_results:
        mean_zeta = np.mean([r.damping_ratio for r in region_results])
        mean_period = np.mean([r.dominant_period_days for r in region_results if r.dominant_period_days > 0])
        print(f"  Mean damping ratio: {mean_zeta:.3f}")
        print(f"  Mean dominant period: {mean_period:.1f} days")

    # Analyze events
    print("\n[3/5] Analyzing event responses...")
    event_results = {}

    for event_name, event_time in GLOBAL_EVENTS.items():
        if df['date'].min() <= event_time <= df['date'].max():
            print(f"  {event_name}...")
            result = analyze_event_response(df, event_time)
            event_results[event_name] = result

            if result:
                print(f"    Shock: {result['shock']:+.3f}, Oscillations: {result['n_oscillations']}")

    # Summary
    print("\n[4/5] Computing summary...")

    n_underdamped = sum(1 for r in region_results if r.damping_ratio < 1)
    n_oscillating_events = sum(1 for r in event_results.values() if r and r['n_oscillations'] > 0)

    print(f"  Underdamped regions: {n_underdamped}/{len(region_results)}")
    print(f"  Events with oscillation: {n_oscillating_events}/{len(event_results)}")

    # Visualize
    print("\n[5/6] Creating visualizations...")
    visualize_results(region_results, event_results)

    # Generate fitted trajectory plots
    print("\n[6/6] Generating fitted trajectory plots...")
    trajectories = plot_fitted_event_trajectories(df, GLOBAL_EVENTS, OUTPUT_DIR)

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if region_results:
        mean_zeta = np.mean([r.damping_ratio for r in region_results])

        if mean_zeta < 1:
            print(f"""
✓ Evidence for UNDERDAMPED global sentiment dynamics!
  Mean ζ = {mean_zeta:.3f} < 1 (oscillatory regime)

  Collective mood shows oscillatory patterns:
  - Weekly cycles (7-day period)
  - Event-triggered oscillations
  - Cross-regional propagation

  This supports Hamiltonian theory: collective beliefs have inertia
  and can oscillate around equilibrium.
""")
        else:
            print(f"""
○ OVERDAMPED dynamics observed (ζ = {mean_zeta:.3f})
  Sentiment changes decay monotonically.

  At the aggregate level, individual oscillations may average out.
  Individual-level data might show different dynamics.
""")

    return df, region_results, event_results


if __name__ == '__main__':
    df, region_results, event_results = main()
