"""
Utilities: Helper functions, validation, and analysis tools.
"""

from __future__ import annotations
import numpy as np
from typing import List, Dict, Tuple, Set, Optional, Callable, Any
from collections import Counter
from dataclasses import dataclass

from ..core.colors import Color, ColorPalette
from ..core.particle import Particle, ParticleBlueprint
from ..topology.graph import BondGraph, Bond, ComplexAnalyzer
from ..geometry.manifold import InformationManifold, FisherMetric


def validate_system(system) -> List[str]:
    """
    Validate internal consistency of a ParticleSystem.
    
    Returns list of error messages (empty if valid).
    """
    errors = []
    
    # Check particle availability vs bonds
    for particle in system.particles:
        for color in particle.capacities:
            bound_count = system.bond_graph.get_color_degree(particle.particle_id, color)
            available = particle.available_sites.get(color, 0)
            capacity = particle.capacities[color]
            
            if bound_count + available != capacity:
                errors.append(
                    f"Particle {particle.label} color {color.name}: "
                    f"bound({bound_count}) + available({available}) != capacity({capacity})"
                )
            
            if available < 0 or available > capacity:
                errors.append(
                    f"Particle {particle.label} color {color.name}: "
                    f"available({available}) out of bounds [0, {capacity}]"
                )
    
    # Check bond graph consistency
    for bond in system.bond_graph.get_bonds():
        # Verify particles have the bond
        p_i = system.particles[bond.particle_i]
        p_j = system.particles[bond.particle_j]
        
        # Check that particles actually have these bonds
        # (This is implicitly checked by degree counts above)
        pass
    
    # Check for duplicate bonds (shouldn't happen in MultiGraph with unique keys)
    seen_keys = set()
    for bond in system.bond_graph.get_bonds():
        if bond.edge_key in seen_keys:
            errors.append(f"Duplicate edge key: {bond.edge_key}")
        seen_keys.add(bond.edge_key)
    
    return errors


def compute_detailed_balance(system, n_samples: int = 1000) -> Dict:
    """
    Estimate detailed balance by sampling forward/backward transitions.
    
    For a system in equilibrium, the probability flux between any two states
    should balance: π(x) * P(x→y) = π(y) * P(y→x)
    
    This is a diagnostic tool to verify the SSA implementation correctness.
    """
    # This would require running the system to equilibrium and measuring
    # transition frequencies. Simplified version:
    return {
        'note': 'Detailed balance verification requires equilibrium sampling',
        'suggestion': 'Run multiple trajectories and compare state distributions'
    }


@dataclass
class SteadyStateAnalyzer:
    """
    Analyze steady-state properties of the assembly system.
    """
    system: 'ParticleSystem'
    
    def estimate_equilibrium_constants(self) -> Dict[Tuple[Color, Color], float]:
        """
        Estimate effective equilibrium constants K_eq = k_on / k_off for each channel
        from the observed binding statistics.
        
        At equilibrium: <n_bound> / <n_free> ≈ K_eq * (concentration terms)
        """
        # Count total bonds per channel
        bond_counts = Counter()
        for bond in self.system.bond_graph.get_bonds():
            channel = tuple(sorted([bond.color_i, bond.color_j], key=lambda c: c.name))
            bond_counts[channel] += 1
        
        # Compute available sites per color
        available = {c: 0 for c in self.system.palette.colors}
        for p in self.system.particles:
            for c, count in p.available_sites.items():
                available[c] += count
        
        # Estimate K_eq from ratio
        k_on = self.system.config.k_on
        k_off = self.system.config.k_off
        
        results = {}
        for channel, n_bonds in bond_counts.items():
            c1, c2 = channel
            # Simplified: K_eq ≈ n_bonds / (available_c1 * available_c2)
            # In true equilibrium: n_bonds = K_eq * available_c1 * available_c2
            if available[c1] > 0 and available[c2] > 0:
                k_eq_est = n_bonds / (available[c1] * available[c2])
                results[channel] = k_eq_est
            else:
                results[channel] = float('inf') if n_bonds > 0 else 0.0
        
        return results
    
    def compute_gel_fraction(self) -> float:
        """
        Compute gel fraction: fraction of particles in the largest complex.
        
        In polymerization theory, gelation occurs when a giant component
        emerges containing a finite fraction of all particles.
        """
        stats = self.system.complex_analyzer.get_complex_statistics()
        return stats.get('gel_fraction', 0.0)
    
    def compute_size_distribution(self) -> Counter:
        """Get complex size distribution."""
        return self.system.complex_analyzer.get_complex_statistics()['size_distribution']
    
    def compute_mean_cluster_size(self) -> float:
        """Weight-average mean cluster size (excludes gel if present)."""
        stats = self.system.complex_analyzer.get_complex_statistics()
        size_dist = stats['size_distribution']
        
        if not size_dist:
            return 0.0
        
        # Weight-average: Σ s² * n_s / Σ s * n_s
        numerator = sum(s * s * n for s, n in size_dist.items())
        denominator = sum(s * n for s, n in size_dist.items())
        
        return numerator / denominator if denominator > 0 else 0.0
    
    def is_gelated(self, threshold: float = 0.1) -> bool:
        """Check if system has gelled (largest complex > threshold fraction)."""
        return self.compute_gel_fraction() > threshold


def analyze_manifold_geometry(system) -> Dict:
    """
    Analyze the information geometric properties of the current state.
    
    Computes:
    - Distances to saturation for each particle
    - Metric tensor condition numbers
    - Geodesic distances between particles
    """
    manifold = system.manifold
    
    results = {
        'distances_to_saturation': {},
        'pairwise_distances': {},
        'metric_condition_numbers': {},
    }
    
    # Distance to saturation (how "exhausted" each particle is)
    for i, particle in enumerate(system.particles):
        dist = manifold.distance_to_saturation(i)
        results['distances_to_saturation'][particle.label] = dist
    
    # Pairwise geodesic distances on the manifold
    for i, p1 in enumerate(system.particles):
        for j, p2 in enumerate(system.particles):
            if i < j:
                d = manifold.metric.geodesic_distance(
                    manifold.get_theta(i), 
                    manifold.get_theta(j)
                )
                results['pairwise_distances'][(p1.label, p2.label)] = d
    
    # Condition numbers of metric tensors
    for i, particle in enumerate(system.particles):
        theta = manifold.get_theta(i)
        G = manifold.metric.metric_tensor(theta)
        # For diagonal matrix, condition number = max/ min eigenvalue
        evals = np.diag(G)
        finite_evals = evals[np.isfinite(evals)]
        if len(finite_evals) > 0:
            cond = np.max(finite_evals) / np.min(finite_evals)
        else:
            cond = np.inf
        results['metric_condition_numbers'][particle.label] = cond
    
    return results


def compute_thermodynamic_quantities(system) -> Dict:
    """
    Compute thermodynamic quantities from the current state.
    
    Using the information geometry framework:
    - Free energy: F = -log Z (partition function)
    - Entropy: S = -Σ p log p (from availability distribution)
    - Chemical potentials: μ_c = ∂F/∂N_c
    """
    # Availability fractions
    all_fractions = []
    for particle in system.particles:
        for color in system.palette.colors:
            cap = particle.capacities.get(color, 0)
            if cap > 0:
                frac = particle.available_sites.get(color, 0) / cap
                all_fractions.append(frac)
    
    if not all_fractions:
        return {'entropy': 0.0, 'free_energy': 0.0}
    
    # Entropy from availability distribution (Shannon entropy)
    # Treat as probability distribution (normalize)
    fractions = np.array(all_fractions)
    probs = fractions / np.sum(fractions) if np.sum(fractions) > 0 else np.ones_like(fractions) / len(fractions)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    # Free energy estimate (relative)
    # For ideal binding: F ≈ -k_on/k_off * total_bonds + entropic terms
    free_energy = - (system.config.k_on / system.config.k_off) * system.n_bonds + entropy
    
    return {
        'entropy': float(entropy),
        'free_energy': float(free_energy),
        'total_bonds': system.n_bonds,
        'mean_availability': float(np.mean(fractions)),
        'availability_std': float(np.std(fractions)),
    }


def run_ensemble(system_factory: Callable, n_runs: int = 10, 
                 max_steps: int = 1000, **run_kwargs) -> List[Dict]:
    """
    Run multiple independent simulation trajectories for ensemble statistics.
    
    Args:
        system_factory: Callable returning a new ParticleSystem instance
        n_runs: Number of independent runs
        max_steps: Steps per run
        **run_kwargs: Additional arguments passed to system.run()
    
    Returns:
        List of final statistics dictionaries
    """
    results = []
    
    for run_idx in range(n_runs):
        system = system_factory()
        # Remove verbose from run_kwargs to avoid duplication
        run_kwargs_no_verbose = {k: v for k, v in run_kwargs.items() if k != 'verbose'}
        stats = system.run(max_steps=max_steps, verbose=False, **run_kwargs_no_verbose)
        stats['run_id'] = run_idx
        results.append(stats)
    
    return results


def compute_ensemble_statistics(ensemble_results: List[Dict]) -> Dict:
    """
    Compute mean and std across ensemble runs.
    """
    if not ensemble_results:
        return {}
    
    keys = ensemble_results[0].keys()
    stats = {}
    
    for key in keys:
        values = [r[key] for r in ensemble_results if key in r and isinstance(r[key], (int, float))]
        if values:
            stats[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'n': len(values),
            }
    
    return stats


def create_standard_test_system(k_on: float = 1.5, k_off: float = 0.3) -> 'ParticleSystem':
    """Create the standard test system from the original example."""
    from ..core.system import create_system_from_strings
    return create_system_from_strings([
        ['white', 'red'],
        ['green', 'blue'],
        ['black', 'green', 'yellow'],
        ['red', 'blue'],
    ], k_on=k_on, k_off=k_off)


def create_linear_chain_system(n_particles: int, k_on: float = 1.0, k_off: float = 0.1) -> 'ParticleSystem':
    """Create a system of particles that can form linear chains (two complementary colors each)."""
    from ..core.system import create_system_from_strings
    particles = []
    for i in range(n_particles):
        if i % 2 == 0:
            particles.append(['white', 'red'])
        else:
            particles.append(['black', 'green'])
    return create_system_from_strings(particles, k_on=k_on, k_off=k_off)


def create_branched_system(n_particles: int, k_on: float = 1.0, k_off: float = 0.1) -> 'ParticleSystem':
    """Create a system with branching potential (three colors per particle)."""
    from ..core.system import create_system_from_strings
    particles = []
    for i in range(n_particles):
        if i % 3 == 0:
            particles.append(['white', 'red', 'blue'])
        elif i % 3 == 1:
            particles.append(['black', 'green', 'yellow'])
        else:
            particles.append(['white', 'green', 'blue'])
    return create_system_from_strings(particles, k_on=k_on, k_off=k_off)