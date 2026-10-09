"""
Information Geometry: Fisher Information Metric and Shahshahani Manifold.

This module implements the continuous "Implicit Information Manifold" where
particle availability parameters θ_i^c live. The manifold is equipped with
the Fisher Information Metric (Shahshahani metric), which governs distances
between availability states.

Key mathematical insight: As θ_i^c → 0 (sites exhausted), the informational
distance stretches to infinity, mirroring the thermodynamic resistance of
rare-state updates. This naturally encodes the difficulty of finding the last
few binding sites.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Mapping, Sequence
from numpy.typing import NDArray

from ..core.colors import Color, ColorPalette
from ..core.particle import Particle, ParticleBlueprint


@dataclass(frozen=True)
class FisherMetric:
    """
    Fisher Information Metric (Shahshahani Metric) on the probability simplex.
    
    For a particle with availability fractions θ = (θ^1, ..., θ^m), the metric
    tensor is diagonal: g_ij = δ_ij / θ_i
    
    The squared distance element: ds² = Σ_c (dθ^c)² / θ^c
    
    This metric has the property that as θ^c → 0, distances diverge, making
    rare states "informationally far" - a geometric encoding of scarcity.
    """
    palette: ColorPalette
    
    def metric_tensor(self, theta: Mapping[Color, float]) -> NDArray[np.float64]:
        """
        Compute the Fisher metric tensor (diagonal matrix) at point theta.
        
        Args:
            theta: Availability fractions θ^c for each color
            
        Returns:
            Diagonal matrix with entries 1/θ^c
        """
        colors = sorted(self.palette.colors, key=lambda c: c.name)
        dim = len(colors)
        g = np.zeros((dim, dim), dtype=np.float64)
        
        for i, color in enumerate(colors):
            theta_c = theta.get(color, 0.0)
            if theta_c > 0:
                g[i, i] = 1.0 / theta_c
            else:
                # At boundary: infinite distance (use large value)
                g[i, i] = np.inf
        return g
    
    def distance_squared(self, theta1: Mapping[Color, float], 
                         theta2: Mapping[Color, float]) -> float:
        """
        Compute squared Fisher-Rao distance between two availability states.
        
        For diagonal metric on simplex, this is: Σ (Δθ^c)² / θ^c (using midpoint)
        Or exact geodesic distance on probability simplex.
        """
        colors = sorted(self.palette.colors, key=lambda c: c.name)
        dist_sq = 0.0
        
        for color in colors:
            t1 = theta1.get(color, 0.0)
            t2 = theta2.get(color, 0.0)
            if t1 > 0 and t2 > 0:
                # Approximate with midpoint
                t_mid = (t1 + t2) / 2
                dist_sq += (t2 - t1) ** 2 / t_mid
            elif t1 == 0 and t2 == 0:
                continue
            else:
                # One is zero, other positive: infinite distance
                return np.inf
        return dist_sq
    
    def geodesic_distance(self, theta1: Mapping[Color, float], 
                              theta2: Mapping[Color, float]) -> float:
            """
            Exact geodesic distance on the probability simplex under Fisher metric.

            For categorical distributions, the Fisher-Rao distance is:
            d(p, q) = arccos(Σ √(p_i q_i))

            Here applied per-color to availability fractions.
            """
            colors = sorted(self.palette.colors, key=lambda c: c.name)
            total_cos = 0.0
            total_weight = 0.0

            for color in colors:
                t1 = theta1.get(color, 0.0)
                t2 = theta2.get(color, 0.0)
                if t1 > 0:
                    # Include all colors where t1 > 0, even if t2 == 0
                    total_cos += np.sqrt(t1 * t2)
                    total_weight += 1.0

            if total_weight == 0:
                return 0.0

            # Normalize and compute angle
            cos_sim = total_cos / total_weight
            cos_sim = np.clip(cos_sim, -1.0, 1.0)
            return np.arccos(cos_sim)
    
    def gradient_norm(self, theta: Mapping[Color, float], 
                      gradient: Mapping[Color, float]) -> float:
        """
        Compute norm of a gradient vector under Fisher metric.
        
        ||∇f||_g² = Σ (∂f/∂θ^c)² * θ^c
        
        This is the dual norm to the metric (using g^ij = θ^i δ^ij).
        """
        norm_sq = 0.0
        for color in self.palette.colors:
            t = theta.get(color, 0.0)
            grad = gradient.get(color, 0.0)
            if t > 0:
                norm_sq += grad ** 2 * t
        return np.sqrt(norm_sq)


@dataclass
class InformationManifold:
    """
    The Implicit Information Manifold M = Π_i M_i.
    
    Each particle i has its own manifold M_i parameterized by θ_i ∈ [0,1]^m.
    The global manifold is the product of individual particle manifolds.
    
    This manifold tracks the continuous "availability coordinates" that
    determine binding propensities. The geometry (Fisher metric) encodes
    the thermodynamic cost of state changes.
    """
    particles: List[Particle]
    palette: ColorPalette
    metric: FisherMetric = None
    
    def __post_init__(self):
        if self.metric is None:
            self.metric = FisherMetric(self.palette)
    
    @property
    def n_particles(self) -> int:
        return len(self.particles)
    
    @property
    def n_colors(self) -> int:
        return len(self.palette)
    
    def get_theta(self, particle_idx: int) -> Dict[Color, float]:
        """Get availability fractions θ_i for a particle."""
        return self.particles[particle_idx].fractions
    
    def get_global_theta(self) -> NDArray[np.float64]:
        """
        Get concatenated availability vector for all particles.
        
        Shape: (n_particles * n_colors,)
        """
        colors = sorted(self.palette.colors, key=lambda c: c.name)
        theta_vec = np.zeros(self.n_particles * self.n_colors)
        
        for i, particle in enumerate(self.particles):
            fractions = particle.fractions
            for j, color in enumerate(colors):
                theta_vec[i * self.n_colors + j] = fractions.get(color, 0.0)
        return theta_vec
    
    def get_global_metric_tensor(self) -> NDArray[np.float64]:
        """
        Block-diagonal metric tensor for the global manifold.
        
        Since particles are independent on the manifold (interaction is
        through the graph, not the manifold), the metric is block-diagonal.
        """
        colors = sorted(self.palette.colors, key=lambda c: c.name)
        total_dim = self.n_particles * self.n_colors
        G = np.zeros((total_dim, total_dim))
        
        for i, particle in enumerate(self.particles):
            theta_i = particle.fractions
            g_i = self.metric.metric_tensor(theta_i)
            start = i * self.n_colors
            G[start:start+self.n_colors, start:start+self.n_colors] = g_i
        
        return G
    
    def compute_channel_aggregates(self) -> Dict[Color, float]:
        """
        Compute global aggregate available sites per color.
        
        Σ_i K_i^c θ_i^c = Σ_i available_sites_i^c
        
        This is the key quantity for global bucket propensity calculation.
        """
        aggregates = {color: 0.0 for color in self.palette.colors}
        for particle in self.particles:
            for color, count in particle.available_sites.items():
                aggregates[color] += count
        return aggregates
    
    def compute_binding_propensities(self, k_on: float) -> Dict[tuple[Color, Color], float]:
        """
        Compute global binding propensities for all color channels.
        
        α_bind^{c,-c} = k_on * (Σ_{i∈B_c} K_i^c θ_i^c) * (Σ_{j∈B_{-c}} K_j^{-c} θ_j^{-c})
        
        This implements the global bucket optimization: O(|C|) instead of O(N²).
        """
        aggregates = self.compute_channel_aggregates()
        channels = self.palette.get_channels()
        propensities = {}
        
        for color, complement in channels:
            total_c = aggregates.get(color, 0)
            total_comp = aggregates.get(complement, 0)
            if total_c > 0 and total_comp > 0:
                propensities[(color, complement)] = k_on * total_c * total_comp
            else:
                propensities[(color, complement)] = 0.0
        
        return propensities
    
    def compute_dissociation_propensity(self, k_off: float, n_bonds: int) -> float:
        """
        Compute total dissociation propensity.
        
        α_break = k_off * |E| = k_off * n_bonds
        """
        return k_off * n_bonds
    
    def distance_to_saturation(self, particle_idx: int) -> float:
        """
        Compute Fisher-Rao distance from current state to saturation (θ=0).
        
        This quantifies how "far" a particle is from having no available sites.
        As θ→0, distance → ∞.
        """
        theta = self.get_theta(particle_idx)
        # Reference: all zeros (saturation)
        theta_sat = {c: 0.0 for c in self.palette.colors}
        return self.metric.geodesic_distance(theta, theta_sat)
    
    def __str__(self) -> str:
        lines = [f"InformationManifold(n_particles={self.n_particles}, n_colors={self.n_colors})"]
        for i, particle in enumerate(self.particles):
            theta = self.get_theta(i)
            line = f"  {particle.label}: " + ", ".join(
                f"{c.name.lower()}={t:.3f}" for c, t in theta.items()
            )
            lines.append(line)
        return "\n".join(lines)


def create_manifold_from_blueprints(
    blueprints: List[ParticleBlueprint],
    palette: ColorPalette = None
) -> InformationManifold:
    """Factory to create manifold from blueprints."""
    if palette is None:
        palette = ColorPalette.standard()
    
    particles = []
    for idx, bp in enumerate(blueprints):
        particles.append(Particle(blueprint=bp, particle_id=idx))
    
    return InformationManifold(particles=particles, palette=palette)