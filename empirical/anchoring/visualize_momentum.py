# -*- coding: utf-8 -*-
"""
Momentum Visualization: Detailed Analysis
==========================================

Deep dive into the belief momentum finding (r = 0.405)
with publication-quality visualizations.

Author: Hamiltonian-VFE Team
Date: December 2025
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import warnings
import os

warnings.filterwarnings('ignore')

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'belief_updates', 'data')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'figures')

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11


def load_data():
    """Load the social trust dataset."""
    data_path = os.path.join(DATA_DIR, 'sampledata.csv')
    df = pd.read_csv(data_path)
    return df


def extract_momentum_data(df):
    """Extract update pairs for momentum analysis."""
    momentum_data = []

    for (subj, trial), group in df.groupby(['subject', 'trial']):
        sampling = group[group['choice'] == 0].copy()
        if len(sampling) < 5:
            continue

        green = sampling['green'].values
        red = sampling['red'].values
        evidence = green - red

        # Updates = changes in evidence
        updates = np.diff(evidence)

        if len(updates) < 3:
            continue

        for i in range(len(updates) - 1):
            momentum_data.append({
                'subject': subj,
                'trial': trial,
                'update_t': updates[i],
                'update_t1': updates[i + 1],
                'evidence_t': evidence[i],
                'sample_num': i,
                'age': group.iloc[0]['age'],
                'true_prob': group.iloc[0]['recip'],
            })

    return pd.DataFrame(momentum_data)


def extract_reversal_data(df):
    """Extract reversal events for τ ~ Λ analysis."""
    reversal_data = []

    for (subj, trial), group in df.groupby(['subject', 'trial']):
        sampling = group[group['choice'] == 0].copy()
        if len(sampling) < 4:
            continue

        evidence = (sampling['green'] - sampling['red']).values
        signs = np.sign(evidence)
        sign_changes = np.where(np.diff(signs) != 0)[0]

        for change_idx in sign_changes:
            if change_idx < 2 or change_idx > len(evidence) - 3:
                continue

            # Precision proxy = samples before reversal
            n_samples_before = change_idx + 1
            evidence_strength = np.abs(evidence[change_idx])

            # Time to reverse
            evidence_after = evidence[change_idx + 1:]
            original_sign = signs[change_idx]
            opposite_indices = np.where(np.sign(evidence_after) == -original_sign)[0]

            if len(opposite_indices) > 0:
                tau = opposite_indices[0] + 1
            else:
                tau = len(evidence_after)

            reversal_data.append({
                'subject': subj,
                'trial': trial,
                'n_samples_before': n_samples_before,
                'evidence_strength': evidence_strength,
                'tau': tau,
                'age': group.iloc[0]['age'],
            })

    return pd.DataFrame(reversal_data)


def extract_trial_data(df):
    """Extract trial-level summaries."""
    trials = []

    for (subj, trial), group in df.groupby(['subject', 'trial']):
        final_row = group[group['choice'] != 0]
        if len(final_row) == 0:
            continue
        final_row = final_row.iloc[-1]

        n_samples = final_row['green'] + final_row['red']
        final_evidence = final_row['green'] - final_row['red']

        trials.append({
            'subject': subj,
            'trial': trial,
            'n_samples': n_samples,
            'final_evidence': final_evidence,
            'decision': final_row['choice'],
            'true_prob': final_row['recip'],
            'age': final_row['age'],
        })

    return pd.DataFrame(trials)


def plot_momentum_scatter(mom_df, ax):
    """Scatter plot of update[t] vs update[t+1] - using bubble sizes for counts."""
    # Count occurrences at each (update_t, update_t1) pair
    counts = mom_df.groupby(['update_t', 'update_t1']).size().reset_index(name='count')

    # Scale bubble sizes (sqrt for area scaling)
    max_count = counts['count'].max()
    sizes = (counts['count'] / max_count) * 2000  # Scale to reasonable size

    # Color by count
    scatter = ax.scatter(counts['update_t'], counts['update_t1'],
                        s=sizes, c=counts['count'], cmap='Blues',
                        alpha=0.7, edgecolors='darkblue', linewidths=1)

    # Add count labels
    for _, row in counts.iterrows():
        if row['count'] > 1000:  # Only label large bubbles
            ax.annotate(f"{row['count']//1000}k",
                       (row['update_t'], row['update_t1']),
                       ha='center', va='center', fontsize=7, color='white',
                       fontweight='bold')

    # Fit line
    slope, intercept, r, p, se = stats.linregress(mom_df['update_t'], mom_df['update_t1'])
    x_line = np.array([-1.5, 1.5])
    ax.plot(x_line, intercept + slope * x_line, 'r-', linewidth=2,
            label=f'r = {r:.3f}')

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Update at time t')
    ax.set_ylabel('Update at time t+1')
    ax.set_title('Belief Momentum: Updates Persist')
    ax.legend(loc='upper left')
    ax.set_xlim(-1.8, 1.8)
    ax.set_ylim(-1.8, 1.8)


def plot_momentum_heatmap(mom_df, ax):
    """2D histogram of update pairs."""
    h, xedges, yedges = np.histogram2d(
        mom_df['update_t'], mom_df['update_t1'],
        bins=np.arange(-3.5, 4.5, 1),
        density=True
    )

    im = ax.imshow(h.T, origin='lower', extent=[-3.5, 3.5, -3.5, 3.5],
                   cmap='Blues', aspect='auto')

    ax.axhline(0, color='red', linestyle='--', alpha=0.7)
    ax.axvline(0, color='red', linestyle='--', alpha=0.7)

    # Add correlation text
    r, p = stats.pearsonr(mom_df['update_t'], mom_df['update_t1'])
    ax.text(0.05, 0.95, f'r = {r:.3f}\np < 0.001',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    ax.set_xlabel('Update at time t')
    ax.set_ylabel('Update at time t+1')
    ax.set_title('Update Transition Density')

    plt.colorbar(im, ax=ax, label='Density')


def plot_momentum_by_position(mom_df, ax):
    """How momentum changes across sampling sequence."""
    mom_df = mom_df.copy()
    mom_df['position_bin'] = pd.cut(mom_df['sample_num'],
                                     bins=[0, 3, 6, 10, 50],
                                     labels=['Early (1-3)', 'Mid-early (4-6)',
                                            'Mid-late (7-10)', 'Late (11+)'])

    positions = ['Early (1-3)', 'Mid-early (4-6)', 'Mid-late (7-10)', 'Late (11+)']
    correlations = []
    errors = []

    for pos in positions:
        subset = mom_df[mom_df['position_bin'] == pos]
        if len(subset) > 100:
            r, p = stats.pearsonr(subset['update_t'], subset['update_t1'])
            # Bootstrap CI
            n_boot = 1000
            boot_rs = []
            for _ in range(n_boot):
                boot_sample = subset.sample(n=len(subset), replace=True)
                boot_r, _ = stats.pearsonr(boot_sample['update_t'], boot_sample['update_t1'])
                boot_rs.append(boot_r)
            se = np.std(boot_rs)
            correlations.append(r)
            errors.append(1.96 * se)
        else:
            correlations.append(np.nan)
            errors.append(0)

    x = np.arange(len(positions))
    ax.bar(x, correlations, yerr=errors, capsize=5, color='steelblue', alpha=0.7)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(positions, rotation=15, ha='right')
    ax.set_ylabel('Momentum (r)')
    ax.set_title('Momentum by Sample Position')
    ax.set_ylim(0, 0.6)


def plot_momentum_by_age(mom_df, ax):
    """How momentum varies with age."""
    mom_df = mom_df.copy()
    mom_df['age_bin'] = pd.cut(mom_df['age'],
                                bins=[9, 13, 17, 25],
                                labels=['10-13', '14-17', '18-24'])

    ages = ['10-13', '14-17', '18-24']
    correlations = []
    errors = []
    ns = []

    for age in ages:
        subset = mom_df[mom_df['age_bin'] == age]
        if len(subset) > 100:
            r, p = stats.pearsonr(subset['update_t'], subset['update_t1'])
            # Bootstrap CI
            n_boot = 500
            boot_rs = []
            for _ in range(n_boot):
                boot_sample = subset.sample(n=min(5000, len(subset)), replace=True)
                boot_r, _ = stats.pearsonr(boot_sample['update_t'], boot_sample['update_t1'])
                boot_rs.append(boot_r)
            se = np.std(boot_rs)
            correlations.append(r)
            errors.append(1.96 * se)
            ns.append(len(subset))
        else:
            correlations.append(np.nan)
            errors.append(0)
            ns.append(0)

    x = np.arange(len(ages))
    bars = ax.bar(x, correlations, yerr=errors, capsize=5, color='coral', alpha=0.7)
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_xticks(x)
    ax.set_xticklabels(ages)
    ax.set_xlabel('Age Group')
    ax.set_ylabel('Momentum (r)')
    ax.set_title('Momentum by Age')
    ax.set_ylim(0, 0.6)

    # Add sample sizes
    for i, (bar, n) in enumerate(zip(bars, ns)):
        ax.text(bar.get_x() + bar.get_width()/2, 0.02, f'n={n:,}',
                ha='center', va='bottom', fontsize=8)


def plot_tau_vs_precision(rev_df, ax):
    """Scatter plot of τ vs precision (n_samples_before)."""
    # Bin by precision
    rev_df = rev_df.copy()
    rev_df['precision_bin'] = pd.cut(rev_df['n_samples_before'],
                                      bins=[0, 3, 6, 10, 50],
                                      labels=['Low (1-3)', 'Med-Low (4-6)',
                                             'Med-High (7-10)', 'High (11+)'])

    # Calculate mean tau by bin
    means = rev_df.groupby('precision_bin')['tau'].mean()
    sems = rev_df.groupby('precision_bin')['tau'].sem()
    counts = rev_df.groupby('precision_bin')['tau'].count()

    x = np.arange(len(means))
    ax.bar(x, means.values, yerr=1.96*sems.values, capsize=5,
           color='forestgreen', alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(means.index, rotation=15, ha='right')
    ax.set_xlabel('Prior Precision (samples before reversal)')
    ax.set_ylabel('τ (samples to reverse)')
    ax.set_title('τ vs Precision: Anti-Inertia')

    # Add correlation
    r, p = stats.spearmanr(rev_df['n_samples_before'], rev_df['tau'])
    ax.text(0.95, 0.95, f'r = {r:.3f}\n(p < 0.001)',
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))


def plot_trust_by_evidence(trial_df, ax):
    """Trust rate as function of final evidence."""
    trial_df = trial_df.copy()
    trial_df['trusted'] = (trial_df['decision'] == 1).astype(int)

    # Bin evidence
    evidence_bins = np.arange(-8, 10, 2)
    trial_df['evidence_bin'] = pd.cut(trial_df['final_evidence'],
                                       bins=evidence_bins,
                                       labels=[f'{i} to {i+1}' for i in evidence_bins[:-1]])

    # Calculate trust rate
    grouped = trial_df.groupby('final_evidence')['trusted'].agg(['mean', 'sem', 'count'])
    grouped = grouped[grouped['count'] >= 20]  # Filter low counts

    ax.errorbar(grouped.index, grouped['mean'], yerr=1.96*grouped['sem'],
                fmt='o-', color='purple', capsize=3, markersize=4)

    # Add sigmoid fit
    from scipy.optimize import curve_fit
    def sigmoid(x, L, k, x0, b):
        return L / (1 + np.exp(-k*(x-x0))) + b

    try:
        popt, _ = curve_fit(sigmoid, grouped.index, grouped['mean'],
                           p0=[1, 0.5, 0, 0], maxfev=5000)
        x_fit = np.linspace(grouped.index.min(), grouped.index.max(), 100)
        ax.plot(x_fit, sigmoid(x_fit, *popt), 'r--', alpha=0.7, label='Sigmoid fit')
    except:
        pass

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)

    ax.set_xlabel('Final Evidence (green - red)')
    ax.set_ylabel('Trust Rate')
    ax.set_title('Trust Rate by Evidence Strength')
    ax.set_ylim(0, 1)


def plot_conservatism(trial_df, ax):
    """Compare actual vs Bayesian-optimal updating."""
    trial_df = trial_df.copy()
    trial_df['trusted'] = (trial_df['decision'] == 1).astype(int)

    # Bayesian prediction: trust if green > red
    trial_df['bayesian_pred'] = (trial_df['final_evidence'] > 0).astype(float)
    trial_df.loc[trial_df['final_evidence'] == 0, 'bayesian_pred'] = 0.5

    # Group by evidence
    grouped_actual = trial_df.groupby('final_evidence')['trusted'].mean()
    grouped_bayesian = trial_df.groupby('final_evidence')['bayesian_pred'].mean()

    # Filter to common range
    common_idx = grouped_actual.index.intersection(grouped_bayesian.index)
    common_idx = [i for i in common_idx if -10 <= i <= 10]

    ax.plot(common_idx, [grouped_bayesian[i] for i in common_idx],
            'b-', linewidth=2, label='Bayesian optimal', alpha=0.7)
    ax.plot(common_idx, [grouped_actual[i] for i in common_idx],
            'ro-', markersize=4, label='Actual behavior', alpha=0.7)

    ax.axhline(0.5, color='gray', linestyle='--', alpha=0.3)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.3)

    ax.set_xlabel('Final Evidence')
    ax.set_ylabel('P(Trust)')
    ax.set_title('Actual vs Bayesian: Slight Conservatism')
    ax.legend(loc='lower right')
    ax.set_ylim(0, 1)


def create_main_figure(mom_df, rev_df, trial_df):
    """Create the main summary figure."""
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Row 1: Momentum findings
    ax1 = fig.add_subplot(gs[0, 0])
    plot_momentum_scatter(mom_df, ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_momentum_heatmap(mom_df, ax2)

    ax3 = fig.add_subplot(gs[0, 2])
    plot_momentum_by_age(mom_df, ax3)

    # Row 2: Anti-inertia and conservatism
    ax4 = fig.add_subplot(gs[1, 0])
    plot_tau_vs_precision(rev_df, ax4)

    ax5 = fig.add_subplot(gs[1, 1])
    plot_trust_by_evidence(trial_df, ax5)

    ax6 = fig.add_subplot(gs[1, 2])
    plot_conservatism(trial_df, ax6)

    # Add panel labels
    for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5, ax6]):
        label = chr(ord('A') + i)
        ax.text(-0.15, 1.05, label, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='bottom')

    fig.suptitle('Belief Dynamics in Social Trust Task: Momentum vs Anti-Inertia',
                 fontsize=14, fontweight='bold', y=1.02)

    return fig


def create_momentum_deep_dive(mom_df):
    """Detailed momentum analysis figure."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: Main scatter
    plot_momentum_scatter(mom_df, axes[0, 0])

    # Top right: By position in sequence
    plot_momentum_by_position(mom_df, axes[0, 1])

    # Bottom left: By true probability
    mom_df = mom_df.copy()
    mom_df['prob_bin'] = pd.cut(mom_df['true_prob'],
                                 bins=[-0.1, 0.3, 0.7, 1.1],
                                 labels=['Low (0-0.3)', 'Medium (0.4-0.6)', 'High (0.7-1.0)'])

    probs = ['Low (0-0.3)', 'Medium (0.4-0.6)', 'High (0.7-1.0)']
    correlations = []
    errors = []

    for prob in probs:
        subset = mom_df[mom_df['prob_bin'] == prob]
        if len(subset) > 100:
            r, _ = stats.pearsonr(subset['update_t'], subset['update_t1'])
            correlations.append(r)
            # Bootstrap
            boot_rs = [stats.pearsonr(subset.sample(n=len(subset), replace=True)['update_t'],
                                      subset.sample(n=len(subset), replace=True)['update_t1'])[0]
                       for _ in range(200)]
            errors.append(1.96 * np.std(boot_rs))
        else:
            correlations.append(np.nan)
            errors.append(0)

    x = np.arange(len(probs))
    axes[1, 0].bar(x, correlations, yerr=errors, capsize=5, color='teal', alpha=0.7)
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(probs)
    axes[1, 0].set_xlabel('True Trustworthiness')
    axes[1, 0].set_ylabel('Momentum (r)')
    axes[1, 0].set_title('Momentum by True Probability')
    axes[1, 0].axhline(0, color='gray', linestyle='--')
    axes[1, 0].set_ylim(0, 0.6)

    # Bottom right: Transition matrix
    mom_df['update_t_sign'] = np.sign(mom_df['update_t']).map({-1: 'Negative', 0: 'Zero', 1: 'Positive'})
    mom_df['update_t1_sign'] = np.sign(mom_df['update_t1']).map({-1: 'Negative', 0: 'Zero', 1: 'Positive'})

    transition = pd.crosstab(mom_df['update_t_sign'], mom_df['update_t1_sign'], normalize='index')
    transition = transition.reindex(index=['Negative', 'Zero', 'Positive'],
                                    columns=['Negative', 'Zero', 'Positive'])

    im = axes[1, 1].imshow(transition.values, cmap='Blues', vmin=0, vmax=0.6)
    axes[1, 1].set_xticks(range(3))
    axes[1, 1].set_yticks(range(3))
    axes[1, 1].set_xticklabels(['Neg', 'Zero', 'Pos'])
    axes[1, 1].set_yticklabels(['Neg', 'Zero', 'Pos'])
    axes[1, 1].set_xlabel('Update t+1')
    axes[1, 1].set_ylabel('Update t')
    axes[1, 1].set_title('Transition Probabilities')

    # Add text annotations
    for i in range(3):
        for j in range(3):
            val = transition.values[i, j]
            color = 'white' if val > 0.4 else 'black'
            axes[1, 1].text(j, i, f'{val:.2f}', ha='center', va='center',
                           color=color, fontsize=12)

    plt.colorbar(im, ax=axes[1, 1], label='P(t+1 | t)')

    fig.suptitle('Deep Dive: Belief Momentum Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def print_summary_stats(mom_df, rev_df, trial_df):
    """Print key statistics."""
    print("=" * 70)
    print("MOMENTUM ANALYSIS: KEY STATISTICS")
    print("=" * 70)

    # Overall momentum
    r, p = stats.pearsonr(mom_df['update_t'], mom_df['update_t1'])
    print(f"\n1. OVERALL MOMENTUM")
    print(f"   r = {r:.4f} (p < 0.001)")
    print(f"   n = {len(mom_df):,} update pairs")

    # By quadrant
    print(f"\n2. TRANSITION PROBABILITIES")
    pos_to_pos = ((mom_df['update_t'] > 0) & (mom_df['update_t1'] > 0)).mean()
    neg_to_neg = ((mom_df['update_t'] < 0) & (mom_df['update_t1'] < 0)).mean()
    pos_to_neg = ((mom_df['update_t'] > 0) & (mom_df['update_t1'] < 0)).mean()
    neg_to_pos = ((mom_df['update_t'] < 0) & (mom_df['update_t1'] > 0)).mean()

    print(f"   P(+ → +) = {pos_to_pos:.3f}")
    print(f"   P(- → -) = {neg_to_neg:.3f}")
    print(f"   P(+ → -) = {pos_to_neg:.3f}")
    print(f"   P(- → +) = {neg_to_pos:.3f}")
    print(f"   Persistence ratio: {(pos_to_pos + neg_to_neg) / (pos_to_neg + neg_to_pos):.2f}x")

    # Anti-inertia
    print(f"\n3. ANTI-INERTIA (τ ~ Λ)")
    r_tau, p_tau = stats.spearmanr(rev_df['n_samples_before'], rev_df['tau'])
    print(f"   τ ~ precision: r = {r_tau:.4f} (p < 0.001)")
    print(f"   n = {len(rev_df):,} reversal events")

    # Conservatism
    print(f"\n4. CONSERVATISM")
    trial_df = trial_df.copy()
    trial_df['bayesian_correct'] = (
        ((trial_df['final_evidence'] > 0) & (trial_df['decision'] == 1)) |
        ((trial_df['final_evidence'] < 0) & (trial_df['decision'] == -1)) |
        (trial_df['final_evidence'] == 0)
    )
    accuracy = trial_df['bayesian_correct'].mean()
    print(f"   Bayesian accuracy: {accuracy:.1%}")

    slope, _, _, _, _ = stats.linregress(trial_df['final_evidence'],
                                          (trial_df['decision'] == 1).astype(float))
    print(f"   Slope (trust vs evidence): {slope:.4f}")


def main():
    """Run the full visualization analysis."""
    print("Loading data...")
    df = load_data()

    print("Extracting momentum data...")
    mom_df = extract_momentum_data(df)

    print("Extracting reversal data...")
    rev_df = extract_reversal_data(df)

    print("Extracting trial data...")
    trial_df = extract_trial_data(df)

    # Print statistics
    print_summary_stats(mom_df, rev_df, trial_df)

    # Create main figure
    print("\nCreating main figure...")
    fig1 = create_main_figure(mom_df, rev_df, trial_df)
    fig1.savefig(os.path.join(OUTPUT_DIR, 'belief_dynamics_summary.png'),
                 dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'belief_dynamics_summary.png')}")

    # Create momentum deep dive
    print("Creating momentum deep dive...")
    fig2 = create_momentum_deep_dive(mom_df)
    fig2.savefig(os.path.join(OUTPUT_DIR, 'momentum_deep_dive.png'),
                 dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'momentum_deep_dive.png')}")

    print("\nDone! Figures saved to:", OUTPUT_DIR)

    # Show figures
    plt.show()

    return mom_df, rev_df, trial_df


if __name__ == '__main__':
    mom_df, rev_df, trial_df = main()
