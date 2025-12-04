# -*- coding: utf-8 -*-
"""
Social Network Hamiltonian Simulation
=====================================

Test the social Hamiltonian mass equation:

τ_i = M_i / γ_i

where M_i = Λ̄_i + Σ_k β_ik Λ̃_k + Σ_j β_ji Λ_i

This simulates opinion dynamics on a network and tests whether
central/high-precision nodes have more inertia.

Author: Hamiltonian-VFE Team
Date: December 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import networkx as nx
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)


@dataclass
class SocialAgent:
    """An agent in the social network."""
    id: int
    mu: float  # Current belief
    Lambda: float  # Precision (confidence)
    p: float = 0.0  # Momentum
    gamma: float = 1.0  # Damping
    history: List[Tuple[float, float, float]] = field(default_factory=list)

    def __post_init__(self):
        self.history = [(self.mu, self.Lambda, self.p)]


class SocialNetwork:
    """
    A network of agents with Hamiltonian belief dynamics.
    """

    def __init__(self, n_agents: int, network_type: str = 'barabasi_albert',
                 m: int = 3, base_precision: float = 1.0):
        """
        Initialize social network.

        Args:
            n_agents: Number of agents
            network_type: 'barabasi_albert', 'erdos_renyi', 'watts_strogatz'
            m: Parameter for network generation
            base_precision: Base precision for agents
        """
        self.n_agents = n_agents

        # Create network structure
        if network_type == 'barabasi_albert':
            self.G = nx.barabasi_albert_graph(n_agents, m)
        elif network_type == 'erdos_renyi':
            self.G = nx.erdos_renyi_graph(n_agents, m / n_agents)
        elif network_type == 'watts_strogatz':
            self.G = nx.watts_strogatz_graph(n_agents, m * 2, 0.3)
        else:
            raise ValueError(f"Unknown network type: {network_type}")

        # Convert to directed graph for asymmetric influence
        self.G = self.G.to_directed()

        # Initialize agents with beliefs around 0, precision varies with degree
        self.agents = {}
        for i in range(n_agents):
            degree = self.G.degree(i)
            # Higher degree nodes have slightly higher precision
            precision = base_precision * (1 + 0.1 * np.log(1 + degree))
            self.agents[i] = SocialAgent(
                id=i,
                mu=np.random.normal(0, 0.5),  # Initial belief
                Lambda=precision,
                gamma=1.0
            )

        # Set influence weights (β_ij)
        self._set_influence_weights()

    def _set_influence_weights(self):
        """Set influence weights β_ij on edges."""
        for u, v in self.G.edges():
            # Weight proportional to 1/out_degree of source
            out_degree = self.G.out_degree(u)
            self.G[u][v]['beta'] = 1.0 / out_degree if out_degree > 0 else 0

    def compute_social_mass(self, agent_id: int) -> float:
        """
        Compute social mass for agent i:

        M_i = Λ̄_i + Σ_k β_ik Λ̃_k + Σ_j β_ji Λ_i

        Terms:
        - Λ̄_i: own precision
        - Σ_k β_ik Λ̃_k: incoming influence (others influencing me)
        - Σ_j β_ji Λ_i: outgoing influence (me influencing others)
        """
        agent = self.agents[agent_id]

        # Own precision
        own_term = agent.Lambda

        # Incoming influence: others → me
        incoming_term = 0
        for neighbor in self.G.predecessors(agent_id):
            beta = self.G[neighbor][agent_id]['beta']
            neighbor_precision = self.agents[neighbor].Lambda
            incoming_term += beta * neighbor_precision

        # Outgoing influence: me → others
        outgoing_term = 0
        for neighbor in self.G.successors(agent_id):
            beta = self.G[agent_id][neighbor]['beta']
            outgoing_term += beta * agent.Lambda

        return own_term + incoming_term + outgoing_term

    def compute_social_force(self, agent_id: int) -> float:
        """
        Compute force on agent from neighbors' beliefs.

        F_i = -Λ_i(μ_i - μ̄_i) where μ̄_i is weighted neighbor mean
        """
        agent = self.agents[agent_id]

        # Compute weighted neighbor mean
        neighbor_sum = 0
        weight_sum = 0

        for neighbor in self.G.predecessors(agent_id):
            beta = self.G[neighbor][agent_id]['beta']
            neighbor_belief = self.agents[neighbor].mu
            neighbor_sum += beta * neighbor_belief
            weight_sum += beta

        if weight_sum > 0:
            target = neighbor_sum / weight_sum
        else:
            target = agent.mu  # No neighbors, no force

        # Force toward neighbor consensus
        force = -agent.Lambda * (agent.mu - target)
        return force

    def step(self, dt: float = 0.01):
        """
        Advance all agents by one timestep using Hamiltonian dynamics.
        """
        # Compute forces for all agents
        forces = {}
        for i in self.agents:
            forces[i] = self.compute_social_force(i)

        # Update all agents
        for i, agent in self.agents.items():
            M = self.compute_social_mass(i)

            # Leapfrog with damping
            # Half step momentum
            damping = -agent.gamma * agent.p
            agent.p = agent.p + 0.5 * dt * (forces[i] + damping)

            # Full step position
            agent.mu = agent.mu + dt * agent.p / M

            # Half step momentum
            force = self.compute_social_force(i)
            damping = -agent.gamma * agent.p
            agent.p = agent.p + 0.5 * dt * (force + damping)

            agent.history.append((agent.mu, agent.Lambda, agent.p))

    def apply_shock(self, agent_ids: List[int], new_belief: float):
        """Apply an external shock to specific agents."""
        for i in agent_ids:
            self.agents[i].mu = new_belief

    def measure_relaxation_times(self, shock_agents: List[int],
                                  shock_value: float,
                                  threshold: float = 0.5,
                                  max_steps: int = 1000,
                                  dt: float = 0.01) -> Dict[int, float]:
        """
        Apply shock and measure relaxation rate for each agent.

        Instead of time to threshold, measure the RATE of belief change,
        which directly reflects τ = M/γ (higher τ = slower rate).
        """
        # Record initial beliefs
        initial_beliefs = {i: self.agents[i].mu for i in self.agents}

        # Apply shock
        self.apply_shock(shock_agents, shock_value)

        # Simulate
        for step in range(max_steps):
            self.step(dt)

        # Measure relaxation: use inverse of initial response rate
        # τ ∝ 1 / (dμ/dt at early times)
        tau = {}
        for i, agent in self.agents.items():
            if i in shock_agents:
                continue

            # Look at early response (first 10% of steps)
            early_window = min(100, max_steps // 10)
            history = agent.history[:early_window]

            if len(history) < 10:
                tau[i] = float('inf')
                continue

            # Compute average absolute rate of change
            beliefs = np.array([h[0] for h in history])
            rates = np.abs(np.diff(beliefs)) / dt

            if np.mean(rates) > 1e-6:
                # τ is inversely proportional to response rate
                tau[i] = 1.0 / np.mean(rates[:10])  # Use first 10 timesteps
            else:
                tau[i] = float('inf')

        # Normalize to finite values
        finite_tau = [t for t in tau.values() if t < float('inf')]
        if finite_tau:
            max_tau = max(finite_tau)
            tau = {k: (v if v < float('inf') else max_tau) for k, v in tau.items()}

        return tau


def test_mass_vs_tau():
    """Test if social mass predicts relaxation time."""
    print("=" * 70)
    print("TESTING: Does Social Mass M predict Relaxation Time τ?")
    print("=" * 70)

    # Create scale-free network (has hub structure)
    net = SocialNetwork(n_agents=100, network_type='barabasi_albert', m=3)

    # Compute social mass for each agent
    masses = {i: net.compute_social_mass(i) for i in net.agents}

    # Apply shock to a few peripheral nodes
    degrees = dict(net.G.degree())
    peripheral = [i for i, d in sorted(degrees.items(), key=lambda x: x[1])[:5]]

    print(f"\nApplying shock to {len(peripheral)} peripheral nodes...")
    print(f"Shock value: +5.0 (strong positive belief)")

    tau = net.measure_relaxation_times(
        shock_agents=peripheral,
        shock_value=5.0,
        threshold=0.3,
        max_steps=2000,
        dt=0.01
    )

    # Analyze: does mass predict tau?
    non_shocked = [i for i in net.agents if i not in peripheral]
    mass_values = [masses[i] for i in non_shocked]
    tau_values = [tau[i] for i in non_shocked]
    degree_values = [degrees[i] for i in non_shocked]

    # Correlations
    r_mass, p_mass = stats.spearmanr(mass_values, tau_values)
    r_degree, p_degree = stats.spearmanr(degree_values, tau_values)

    print(f"\nResults (n = {len(non_shocked)} non-shocked agents):")
    print(f"  τ ~ M (social mass): r = {r_mass:.3f} (p = {p_mass:.4f})")
    print(f"  τ ~ degree:          r = {r_degree:.3f} (p = {p_degree:.4f})")

    if r_mass > 0 and p_mass < 0.05:
        print("\n  ✓ CONFIRMED: Higher social mass → longer relaxation time")
    else:
        print("\n  ✗ NOT CONFIRMED: Mass doesn't predict τ as expected")

    return net, masses, tau, (r_mass, p_mass)


def visualize_network_dynamics(net: SocialNetwork, masses: dict, tau: dict):
    """Visualize the network and dynamics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: Network structure colored by mass
    ax = axes[0, 0]
    pos = nx.spring_layout(net.G, seed=42)

    mass_values = [masses[i] for i in net.G.nodes()]
    nx.draw_networkx_nodes(net.G, pos, ax=ax, node_size=50,
                           node_color=mass_values, cmap='Reds', alpha=0.8)
    nx.draw_networkx_edges(net.G, pos, ax=ax, alpha=0.1, arrows=False)

    ax.set_title('A. Network Structure (color = social mass M)')
    ax.axis('off')

    # Panel B: Mass vs τ scatter
    ax = axes[0, 1]
    non_shocked = [i for i in tau.keys()]
    m_vals = [masses[i] for i in non_shocked]
    t_vals = [tau[i] for i in non_shocked]

    ax.scatter(m_vals, t_vals, alpha=0.6, c='steelblue')

    # Fit line
    slope, intercept, r, p, _ = stats.linregress(m_vals, t_vals)
    x_line = np.array([min(m_vals), max(m_vals)])
    ax.plot(x_line, intercept + slope * x_line, 'r-', linewidth=2,
            label=f'r = {r:.3f}')

    ax.set_xlabel('Social Mass M')
    ax.set_ylabel('Relaxation Time τ')
    ax.set_title('B. Mass vs Relaxation Time')
    ax.legend()

    # Panel C: Belief trajectories for select agents
    ax = axes[1, 0]

    # Pick agents with different masses
    sorted_by_mass = sorted(masses.items(), key=lambda x: x[1])
    low_mass_agent = sorted_by_mass[10][0]  # Low mass
    high_mass_agent = sorted_by_mass[-10][0]  # High mass

    for agent_id, color, label in [(low_mass_agent, 'blue', 'Low M'),
                                    (high_mass_agent, 'red', 'High M')]:
        history = net.agents[agent_id].history
        beliefs = [h[0] for h in history]
        times = np.arange(len(beliefs)) * 0.01
        ax.plot(times, beliefs, color=color, alpha=0.7, label=label)

    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Time')
    ax.set_ylabel('Belief μ')
    ax.set_title('C. Belief Trajectories')
    ax.legend()

    # Panel D: Degree distribution and mass distribution
    ax = axes[1, 1]

    degrees = [net.G.degree(i) for i in net.G.nodes()]
    ax.hist(degrees, bins=20, alpha=0.5, label='Degree', color='blue')
    ax.set_xlabel('Degree / Mass')
    ax.set_ylabel('Count')

    ax2 = ax.twinx()
    ax2.hist(list(masses.values()), bins=20, alpha=0.5, label='Mass', color='red')
    ax2.set_ylabel('Count (Mass)', color='red')

    ax.set_title('D. Degree and Mass Distributions')
    ax.legend(loc='upper right')

    plt.suptitle('Social Network Hamiltonian Dynamics', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


def run_parameter_sweep():
    """Sweep network parameters to test robustness."""
    print("\n" + "=" * 70)
    print("PARAMETER SWEEP: Testing robustness across network types")
    print("=" * 70)

    results = []

    for network_type in ['barabasi_albert', 'erdos_renyi', 'watts_strogatz']:
        for n_agents in [50, 100, 200]:
            net = SocialNetwork(n_agents=n_agents, network_type=network_type, m=3)

            masses = {i: net.compute_social_mass(i) for i in net.agents}

            # Shock peripheral nodes
            degrees = dict(net.G.degree())
            peripheral = [i for i, d in sorted(degrees.items(), key=lambda x: x[1])[:3]]

            tau = net.measure_relaxation_times(
                shock_agents=peripheral,
                shock_value=5.0,
                threshold=0.3,
                max_steps=1000,
                dt=0.01
            )

            non_shocked = [i for i in net.agents if i not in peripheral]
            mass_values = [masses[i] for i in non_shocked]
            tau_values = [tau[i] for i in non_shocked]

            r, p = stats.spearmanr(mass_values, tau_values)

            results.append({
                'network': network_type,
                'n_agents': n_agents,
                'r': r,
                'p': p,
                'significant': p < 0.05 and r > 0
            })

            print(f"  {network_type:20s} n={n_agents:3d}: r = {r:+.3f} (p = {p:.4f})")

    return results


def main():
    print("=" * 70)
    print("SOCIAL NETWORK HAMILTONIAN SIMULATION")
    print("=" * 70)
    print("""
Testing the social mass equation:

    M_i = Λ̄_i + Σ_k β_ik Λ̃_k + Σ_j β_ji Λ_i

Prediction: Higher social mass → longer relaxation time (τ = M/γ)
    """)

    # Main test
    net, masses, tau, (r, p) = test_mass_vs_tau()

    # Visualize
    print("\nGenerating visualizations...")
    fig = visualize_network_dynamics(net, masses, tau)
    fig.savefig(os.path.join(OUTPUT_DIR, 'social_hamiltonian.png'),
                dpi=150, bbox_inches='tight', facecolor='white')
    print(f"  Saved: {os.path.join(OUTPUT_DIR, 'social_hamiltonian.png')}")

    # Parameter sweep
    sweep_results = run_parameter_sweep()

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    n_significant = sum(1 for r in sweep_results if r['significant'])
    print(f"\nSignificant positive τ ~ M relationship: {n_significant}/{len(sweep_results)} conditions")

    if n_significant > len(sweep_results) // 2:
        print("""
✓ The social Hamiltonian model is SUPPORTED:
  - Agents with higher social mass take longer to change beliefs
  - Central/influential nodes show more inertia
  - This matches the theoretical prediction τ = M/γ
""")
    else:
        print("""
✗ Results are mixed - the model needs refinement.
""")

    plt.show()

    return net, masses, tau, sweep_results


if __name__ == '__main__':
    results = main()
