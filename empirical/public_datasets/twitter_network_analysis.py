# -*- coding: utf-8 -*-
"""
Hamiltonian Social Mass Analysis: Twitter Polarization
=======================================================

Tests the social mass equation on Twitter network data.

Social Mass Equation (from Hamiltonian VFE):
    M_i = Λ̄_i + Σ_k β_ik Λ̃_k + Σ_j β_ji Λ_i

Where:
- Λ̄_i = agent i's own precision (certainty)
- β_ik = coupling strength from k to i
- Λ̃_k = neighbor k's precision

Prediction: Higher social mass → slower belief change (τ ∝ M/γ)

Data: Twitter Polarization Dataset
URL: https://gvrkiran.github.io/polarizationTwitter/
Paper: ICWSM-17

Author: Hamiltonian-VFE Team
Date: December 2025
"""

import os
import sys
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats, sparse
from collections import defaultdict
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add project root
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, '..', '..')
sys.path.insert(0, PROJECT_ROOT)

DATA_DIR = os.path.join(SCRIPT_DIR, 'twitter_polarization')
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class UserDynamics:
    """Dynamics summary for a Twitter user."""
    user_id: str
    own_precision: float  # Λ̄_i: consistency of their opinions
    social_mass: float  # M_i: total mass including network effects
    n_neighbors: int  # degree
    neighbor_precision: float  # average neighbor precision
    opinion_variance: float  # how much their opinion varies over time
    relaxation_time: float  # τ: time to respond to network changes


def load_twitter_data(data_dir: str) -> Tuple[nx.DiGraph, pd.DataFrame]:
    """
    Load Twitter polarization dataset.

    Expected files:
    - followers.csv or edges.csv: Network edges (follower -> followee)
    - user_opinions.csv or tweets.csv: User opinion scores over time
    """
    G = nx.DiGraph()
    opinions_df = None

    # Load network
    network_files = ['followers.csv', 'edges.csv', 'network.csv', 'graph.csv']
    for fname in network_files:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            print(f"Loading network from: {fname}")
            df = pd.read_csv(fpath)

            # Find edge columns
            cols = df.columns.tolist()
            if len(cols) >= 2:
                src_col, tgt_col = cols[0], cols[1]
                for _, row in df.iterrows():
                    G.add_edge(str(row[src_col]), str(row[tgt_col]))

            print(f"  Loaded {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            break

    # Load opinions
    opinion_files = ['user_opinions.csv', 'tweets.csv', 'sentiments.csv', 'opinions.csv']
    for fname in opinion_files:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            print(f"Loading opinions from: {fname}")
            opinions_df = pd.read_csv(fpath)
            print(f"  Loaded {len(opinions_df)} records")
            break

    return G, opinions_df


def compute_user_precision(
    opinions_df: pd.DataFrame,
    user_col: str = 'user_id',
    opinion_col: str = 'opinion',
    time_col: str = 'timestamp'
) -> Dict[str, float]:
    """
    Compute precision (inverse variance) for each user's opinions.

    High precision = consistent opinions = high certainty
    Low precision = varying opinions = low certainty
    """
    precisions = {}

    for user_id, group in opinions_df.groupby(user_col):
        opinions = group[opinion_col].values

        if len(opinions) >= 2:
            variance = np.var(opinions)
            # Precision = 1 / variance (with floor to avoid infinity)
            precision = 1.0 / max(variance, 0.01)
        else:
            precision = 1.0  # Default for users with few observations

        precisions[str(user_id)] = precision

    return precisions


def compute_social_mass(
    G: nx.DiGraph,
    precisions: Dict[str, float],
    coupling_strength: float = 1.0
) -> Dict[str, float]:
    """
    Compute social mass for each user using the Hamiltonian formula:

    M_i = Λ̄_i + Σ_k β_ik Λ̃_k + Σ_j β_ji Λ_i

    Simplified version (undirected coupling, uniform β):
    M_i = Λ_i + β * Σ_neighbors Λ_neighbor

    This captures how precision propagates through the network.
    """
    social_mass = {}

    for node in G.nodes():
        own_precision = precisions.get(node, 1.0)

        # Sum neighbor precisions (weighted by coupling)
        neighbor_precision_sum = 0
        neighbors = list(G.predecessors(node)) + list(G.successors(node))

        for neighbor in neighbors:
            neighbor_precision_sum += precisions.get(neighbor, 1.0)

        # Social mass formula
        M_i = own_precision + coupling_strength * neighbor_precision_sum

        social_mass[node] = M_i

    return social_mass


def compute_relaxation_time(
    opinions_df: pd.DataFrame,
    user_col: str = 'user_id',
    opinion_col: str = 'opinion',
    time_col: str = 'timestamp'
) -> Dict[str, float]:
    """
    Estimate relaxation time τ for each user.

    τ = time scale over which opinions decorrelate from their initial value.

    Method: Fit exponential decay to autocorrelation function.
    """
    relaxation_times = {}

    for user_id, group in opinions_df.groupby(user_col):
        group = group.sort_values(time_col)
        opinions = group[opinion_col].values

        if len(opinions) < 10:
            continue

        # Compute autocorrelation
        mean_op = np.mean(opinions)
        opinions_centered = opinions - mean_op

        # Autocorrelation at different lags
        max_lag = min(20, len(opinions) // 2)
        autocorr = []

        var = np.var(opinions_centered)
        if var < 1e-10:
            relaxation_times[str(user_id)] = float('inf')
            continue

        for lag in range(max_lag):
            if lag == 0:
                autocorr.append(1.0)
            else:
                corr = np.corrcoef(opinions_centered[:-lag], opinions_centered[lag:])[0, 1]
                if np.isnan(corr):
                    corr = 0
                autocorr.append(corr)

        # Fit exponential decay: autocorr(lag) = exp(-lag / τ)
        # τ = -lag / log(autocorr)
        autocorr = np.array(autocorr)

        # Find first lag where autocorr drops below 1/e
        threshold = 1.0 / np.e
        below_threshold = np.where(autocorr < threshold)[0]

        if len(below_threshold) > 0:
            tau = below_threshold[0]
        else:
            tau = max_lag  # Hasn't decorrelated yet

        relaxation_times[str(user_id)] = float(tau)

    return relaxation_times


def analyze_mass_relaxation_correlation(
    social_mass: Dict[str, float],
    relaxation_times: Dict[str, float]
) -> Dict:
    """
    Test the Hamiltonian prediction: τ ∝ M/γ

    If the network is underdamped, higher mass → slower relaxation.
    If overdamped, higher mass → faster relaxation (τ ∝ γ/M).
    """
    # Get users present in both
    common_users = set(social_mass.keys()) & set(relaxation_times.keys())
    common_users = [u for u in common_users
                    if relaxation_times[u] < float('inf') and relaxation_times[u] > 0]

    if len(common_users) < 10:
        return {
            'n_users': len(common_users),
            'error': 'Not enough users with valid data'
        }

    masses = [social_mass[u] for u in common_users]
    taus = [relaxation_times[u] for u in common_users]

    # Spearman correlation (robust to outliers)
    r_spearman, p_spearman = stats.spearmanr(masses, taus)

    # Pearson correlation
    r_pearson, p_pearson = stats.pearsonr(masses, taus)

    return {
        'n_users': len(common_users),
        'r_spearman': r_spearman,
        'p_spearman': p_spearman,
        'r_pearson': r_pearson,
        'p_pearson': p_pearson,
        'mean_mass': np.mean(masses),
        'std_mass': np.std(masses),
        'mean_tau': np.mean(taus),
        'std_tau': np.std(taus),
        'masses': masses,
        'taus': taus,
    }


def generate_sample_network():
    """
    Generate sample Twitter-like network for demonstration.
    """
    print("Generating sample Twitter network for demonstration...")

    np.random.seed(42)

    # Create scale-free network (like Twitter)
    n_users = 500
    G = nx.barabasi_albert_graph(n_users, 3)
    G = G.to_directed()

    # Generate opinions over time (20 time points)
    n_times = 20
    records = []

    # User precisions (power-law distributed - some very certain, most uncertain)
    user_precisions = {}
    for node in G.nodes():
        # High-degree nodes tend to have higher precision (influencers)
        degree = G.degree(node)
        base_precision = np.random.exponential(1.0)
        degree_bonus = 0.1 * degree
        user_precisions[node] = base_precision + degree_bonus

    # Generate opinion trajectories
    for node in G.nodes():
        precision = user_precisions[node]
        noise_scale = 1.0 / np.sqrt(precision)

        # Initial opinion
        opinion = np.random.randn()

        for t in range(n_times):
            # Influence from neighbors
            neighbors = list(G.predecessors(node))
            if neighbors:
                neighbor_opinions = [records[-1]['opinion']
                                     for r in records
                                     if r['user_id'] in neighbors and r['timestamp'] == t-1]
                if neighbor_opinions:
                    social_pull = 0.1 * (np.mean(neighbor_opinions) - opinion)
                else:
                    social_pull = 0
            else:
                social_pull = 0

            # Update opinion
            opinion = opinion + social_pull + noise_scale * np.random.randn() * 0.1

            records.append({
                'user_id': node,
                'timestamp': t,
                'opinion': opinion
            })

    opinions_df = pd.DataFrame(records)

    # Save
    os.makedirs(DATA_DIR, exist_ok=True)

    # Save network
    edges = [(str(u), str(v)) for u, v in G.edges()]
    edges_df = pd.DataFrame(edges, columns=['source', 'target'])
    edges_df.to_csv(os.path.join(DATA_DIR, 'edges.csv'), index=False)

    # Save opinions
    opinions_df.to_csv(os.path.join(DATA_DIR, 'user_opinions.csv'), index=False)

    print(f"  Generated {n_users} users, {G.number_of_edges()} edges, {len(opinions_df)} opinion records")

    return G, opinions_df


def visualize_results(
    G: nx.DiGraph,
    precisions: Dict[str, float],
    social_mass: Dict[str, float],
    correlation_results: Dict
):
    """Visualize the social mass analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Degree vs Precision
    ax = axes[0, 0]
    degrees = [G.degree(n) for n in G.nodes() if n in precisions]
    precs = [precisions[n] for n in G.nodes() if n in precisions]

    ax.scatter(degrees, precs, alpha=0.3, c='steelblue', s=20)
    ax.set_xlabel('Node Degree', fontsize=12)
    ax.set_ylabel('Precision Λ', fontsize=12)
    ax.set_title('A. Network Position vs Certainty', fontsize=11)

    # Add correlation
    if len(degrees) > 10:
        r, p = stats.spearmanr(degrees, precs)
        ax.text(0.05, 0.95, f'r = {r:.3f}', transform=ax.transAxes, fontsize=10)

    # Panel B: Social Mass distribution
    ax = axes[0, 1]
    masses = list(social_mass.values())
    ax.hist(masses, bins=30, alpha=0.7, color='coral', edgecolor='black')
    ax.set_xlabel('Social Mass M', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title('B. Distribution of Social Mass', fontsize=11)
    ax.axvline(np.mean(masses), color='green', ls='--', lw=2,
               label=f'Mean = {np.mean(masses):.1f}')
    ax.legend()

    # Panel C: Mass vs Relaxation Time
    ax = axes[1, 0]
    if 'masses' in correlation_results and 'taus' in correlation_results:
        ax.scatter(correlation_results['masses'], correlation_results['taus'],
                   alpha=0.3, c='steelblue', s=20)

        # Fit line
        if len(correlation_results['masses']) > 10:
            slope, intercept, r, p, _ = stats.linregress(
                correlation_results['masses'], correlation_results['taus']
            )
            x_line = np.array([min(correlation_results['masses']),
                              max(correlation_results['masses'])])
            ax.plot(x_line, intercept + slope * x_line, 'r-', lw=2)

    ax.set_xlabel('Social Mass M', fontsize=12)
    ax.set_ylabel('Relaxation Time τ', fontsize=12)
    ax.set_title(f"C. Mass vs Relaxation\n(r = {correlation_results.get('r_spearman', 0):.3f})",
                 fontsize=11)

    # Panel D: Network visualization (small sample)
    ax = axes[1, 1]

    # Get subgraph of high-mass nodes
    top_mass_nodes = sorted(social_mass.keys(), key=lambda x: social_mass[x], reverse=True)[:50]
    subG = G.subgraph(top_mass_nodes)

    if len(subG) > 0:
        pos = nx.spring_layout(subG, seed=42)
        node_sizes = [social_mass.get(n, 1) * 5 for n in subG.nodes()]
        node_colors = [precisions.get(n, 1) for n in subG.nodes()]

        nx.draw(subG, pos, ax=ax,
                node_size=node_sizes,
                node_color=node_colors,
                cmap='coolwarm',
                with_labels=False,
                alpha=0.7,
                edge_color='gray',
                arrows=False)

    ax.set_title('D. High-Mass Subnetwork\n(size=mass, color=precision)', fontsize=11)

    plt.suptitle('Social Hamiltonian: Twitter Network Analysis\n(Testing M_i = Λ_i + Σ β_ij Λ_j)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()

    save_path = os.path.join(OUTPUT_DIR, 'twitter_social_mass.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved figure to: {save_path}")


def main():
    """Main analysis."""
    print("=" * 70)
    print("SOCIAL MASS ANALYSIS: Twitter Polarization Network")
    print("=" * 70)
    print("""
Testing the social mass equation from Hamiltonian VFE:

    M_i = Λ̄_i + Σ_k β_ik Λ̃_k + Σ_j β_ji Λ_i

Prediction: τ ∝ M/γ → Higher social mass → slower opinion change

In overdamped regime: τ ∝ γ/Λ → Higher precision → FASTER change
(This is what we found in individual-level data)

Question: Does network structure create effective inertia?
    """)

    # Load or generate data
    print("\n[1/5] Loading data...")
    G, opinions_df = load_twitter_data(DATA_DIR)

    if G.number_of_nodes() == 0 or opinions_df is None:
        print("\n No data found. Generating sample network...")
        G, opinions_df = generate_sample_network()

    print(f"      Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"      Opinions: {len(opinions_df)} records")

    # Compute precisions
    print("\n[2/5] Computing user precisions...")
    precisions = compute_user_precision(opinions_df)
    print(f"      Computed precision for {len(precisions)} users")
    print(f"      Mean precision: {np.mean(list(precisions.values())):.3f}")

    # Compute social mass
    print("\n[3/5] Computing social mass...")
    social_mass = compute_social_mass(G, precisions, coupling_strength=0.1)
    print(f"      Mean social mass: {np.mean(list(social_mass.values())):.3f}")

    # Compute relaxation times
    print("\n[4/5] Computing relaxation times...")
    relaxation_times = compute_relaxation_time(opinions_df)
    print(f"      Computed τ for {len(relaxation_times)} users")

    # Test correlation
    print("\n[5/5] Testing mass-relaxation correlation...")
    results = analyze_mass_relaxation_correlation(social_mass, relaxation_times)

    print(f"\n   Results (n = {results['n_users']} users):")
    print(f"      τ ~ M (Spearman): r = {results['r_spearman']:+.3f} (p = {results['p_spearman']:.4f})")
    print(f"      τ ~ M (Pearson):  r = {results['r_pearson']:+.3f} (p = {results['p_pearson']:.4f})")

    # Visualize
    visualize_results(G, precisions, social_mass, results)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results['r_spearman'] > 0 and results['p_spearman'] < 0.05:
        print("""
✓ CONFIRMED: Higher social mass → slower opinion change!
  This validates τ = M/γ from Hamiltonian theory.
  Network structure creates effective inertia.
""")
    elif results['r_spearman'] < 0 and results['p_spearman'] < 0.05:
        print("""
✗ OPPOSITE: Higher social mass → FASTER opinion change
  This suggests overdamped regime even in networks.
  Precision dominates over network inertia.
""")
    else:
        print("""
○ No significant relationship between mass and relaxation.
  Network effects may cancel out, or data is too noisy.
""")

    return G, precisions, social_mass, relaxation_times, results


if __name__ == '__main__':
    G, precisions, social_mass, relaxation_times, results = main()
