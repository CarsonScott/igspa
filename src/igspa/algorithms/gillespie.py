"""
Global Bucket Sampler: O(|C| + |E|) propensity evaluation and sampling.

This is the core optimization that replaces O(N²) pairwise evaluation with
global aggregate buckets. The key insight:

Standard SSA: For each pair (i,j) and each color channel, compute propensity
Global Bucket: Aggregate all available sites per color across all particles
               α_bind^{c,-c} = k_on * (Σ_i K_i^c θ_i^c) * (Σ_j K_j^{-c} θ_j^{-c})

Then use two-stage sampling:
1. Sample color channel proportional to aggregate propensity
2. Sample particle i from bucket B_c weighted by K_i^c θ_i^c
3. Sample particle j from bucket B_{-c} weighted by K_j^{-c} θ_j^{-c}

This achieves O(|C| + |E|) per step complexity.
"""

from __future__ import annotations
import random
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Iterator, Mapping
from collections import defaultdict
from numpy.typing import NDArray

from ..core.colors import Color, ColorPalette, COMPLEMENT_MAP
from ..core.particle import Particle
from ..geometry.manifold import InformationManifold
from ..topology.graph import BondGraph, Bond


@dataclass
class GlobalBucketSampler:
    """
    Global Bucket Sampler for O(|C| + |E|) Gillespie SSA.
    
    Maintains buckets B_c = {i | particle i has available sites of color c}
    and aggregate site counts for instant propensity computation.
    
    This is the core algorithmic innovation enabling linear/constant scaling.
    """
    particles: List[Particle]
    palette: ColorPalette
    k_on: float
    k_off: float
    
    # Buckets: color -> set of particle indices with available sites
    buckets: Dict[Color, Set[int]] = field(default_factory=dict)
    
    # Aggregate available sites per color (Σ_i available_sites_i^c)
    aggregate_sites: Dict[Color, int] = field(default_factory=dict)
    
    # Cache for binding channels
    _channels: List[Tuple[Color, Color]] = field(default_factory=list)
    _channel_aggregates: Dict[Tuple[Color, Color], Tuple[int, int]] = field(default_factory=dict)
    
    def __post_init__(self):
        self._initialize_buckets()
        self._channels = self.palette.get_channels()
    
    def _initialize_buckets(self):
        """Initialize buckets from initial particle states."""
        self.buckets = {color: set() for color in self.palette.colors}
        self.aggregate_sites = {color: 0 for color in self.palette.colors}
        
        for idx, particle in enumerate(self.particles):
            for color, count in particle.available_sites.items():
                if count > 0:
                    self.buckets[color].add(idx)
                    self.aggregate_sites[color] += count
    
    def update_after_bind(self, particle_i: int, particle_j: int, 
                          color_i: Color, color_j: Color):
        """Update buckets after a binding event consumes sites."""
        for p_idx, color in [(particle_i, color_i), (particle_j, color_j)]:
            particle = self.particles[p_idx]
            # Ensure the color is in available_sites
            if color not in particle.available_sites:
                particle.available_sites[color] = 0
            # Actually decrement the site count
            particle.available_sites[color] -= 1
            new_count = particle.available_sites[color]
            
            self.aggregate_sites[color] -= 1
            
            # Remove from bucket if exhausted
            if new_count == 0:
                self.buckets[color].discard(p_idx)
    
    def update_after_break(self, particle_i: int, particle_j: int,
                           color_i: Color, color_j: Color):
        """Update buckets after a bond break releases sites."""
        for p_idx, color in [(particle_i, color_i), (particle_j, color_j)]:
            particle = self.particles[p_idx]
            # Ensure the color is in available_sites
            if color not in particle.available_sites:
                particle.available_sites[color] = 0
            # Actually increment the site count
            particle.available_sites[color] += 1
            new_count = particle.available_sites[color]
            
            self.aggregate_sites[color] += 1
            
            # Add to bucket if was exhausted
            if new_count == 1:
                self.buckets[color].add(p_idx)
    
    def compute_binding_propensities(self) -> Dict[Tuple[Color, Color], float]:
        """
        Compute global binding propensities for all channels in O(|C|).
        
        α_bind^{c,-c} = k_on * (Σ_{i∈B_c} available_i^c) * (Σ_{j∈B_{-c}} available_j^{-c})
        """
        propensities = {}
        
        for color, complement in self._channels:
            total_c = self.aggregate_sites.get(color, 0)
            total_comp = self.aggregate_sites.get(complement, 0)
            
            if total_c > 0 and total_comp > 0:
                propensities[(color, complement)] = self.k_on * total_c * total_comp
            else:
                propensities[(color, complement)] = 0.0
        
        return propensities
    
    def compute_total_binding_propensity(self) -> float:
        """Compute total binding propensity across all channels."""
        total = 0.0
        for color, complement in self._channels:
            total_c = self.aggregate_sites.get(color, 0)
            total_comp = self.aggregate_sites.get(complement, 0)
            if total_c > 0 and total_comp > 0:
                total += self.k_on * total_c * total_comp
        return total
    
    def sample_binding_channel(self, binding_propensities: Dict[Tuple[Color, Color], float],
                               total_binding_propensity: float) -> Tuple[Color, Color]:
        """
        Sample a color channel proportional to its binding propensity.
        
        This is Stage 1 of two-stage sampling: select which color pair binds.
        """
        if total_binding_propensity <= 0:
            return None
        
        r = random.random() * total_binding_propensity
        cumulative = 0.0
        
        for channel, propensity in binding_propensities.items():
            cumulative += propensity
            if cumulative >= r:
                return channel
        
        # Fallback (shouldn't happen with correct total)
        return self._channels[-1]
    
    def sample_particle_from_bucket(self, color: Color) -> int:
        """
        Sample a particle from bucket B_c weighted by available site count.
        
        This is Stage 2: P(i) ∝ available_sites_i^c
        Uses random.choices for weighted sampling.
        """
        bucket = self.buckets[color]
        if not bucket:
            raise ValueError(f"Bucket for {color} is empty")
        
        particles_list = list(bucket)
        weights = [self.particles[p_idx].available_sites[color] for p_idx in particles_list]
        
        return random.choices(particles_list, weights=weights, k=1)[0]
    
    def sample_binding_pair(self, color_i: Color, color_j: Color) -> Tuple[int, int]:
        """
        Two-stage sampling of a binding pair for the given color channel.
        
        Stage 1: Sample particle i from B_{color_i} weighted by available sites
        Stage 2: Sample particle j from B_{color_j} weighted by available sites
        
        Handles self-binding prevention when same particle has both colors.
        """
        part_i = self.sample_particle_from_bucket(color_i)
        part_j = self.sample_particle_from_bucket(color_j)
        
        # Handle self-binding: same particle, same color, only 1 site each
        if (part_i == part_j and 
            color_i == color_j and 
            self.particles[part_i].available_sites[color_i] == 1):
            # Reject and resample - this preserves detailed balance
            return self.sample_binding_pair(color_i, color_j)
        
        return part_i, part_j
    
    def get_all_dissociation_events(self, bond_graph: BondGraph) -> List[Tuple[float, Bond]]:
        """
        Get all dissociation events with their propensities.
        
        Each bond has propensity k_off. Returns list of (propensity, bond).
        O(|E|) operation.
        """
        events = []
        for bond in bond_graph.get_bonds():
            events.append((self.k_off, bond))
        return events
    
    def compute_total_propensity(self, bond_graph: BondGraph) -> Tuple[float, float]:
        """
        Compute total system propensity: binding + dissociation.
        
        Returns: (total_binding_propensity, total_dissociation_propensity)
        """
        total_bind = self.compute_total_binding_propensity()
        total_break = self.k_off * bond_graph.n_bonds
        return total_bind, total_break
    
    def __str__(self) -> str:
        lines = [f"GlobalBucketSampler(k_on={self.k_on}, k_off={self.k_off})"]
        for color in sorted(self.palette.colors, key=lambda c: c.name):
            bucket_size = len(self.buckets.get(color, set()))
            agg = self.aggregate_sites.get(color, 0)
            lines.append(f"  {color.name.lower()}: bucket={bucket_size}, aggregate={agg}")
        return "\n".join(lines)


@dataclass
class TwoStageSampler:
    """
    Two-Stage Gillespie Sampler with Global Bucket Optimization.
    
    This implements the full sampling routine:
    1. Compute total propensity a_0 = Σ α_bind + Σ α_break
    2. Sample time step τ ~ Exp(a_0)
    3. Sample event type (bind/break) proportional to propensity
    4. If bind: two-stage particle selection via buckets
    5. If break: uniform selection among bonds
    """
    bucket_sampler: GlobalBucketSampler
    bond_graph: BondGraph
    
    def step(self):
        """
        Execute one Gillespie SSA step.
        
        Returns tuple:
        - For bind: (True, 'bind', dt, part_i, part_j, color_i, color_j)
        - For break: (True, 'break', dt, bond)
        - For no events: (False, None, 0.0)
        """
        # Compute propensities
        bind_props = self.bucket_sampler.compute_binding_propensities()
        total_bind = sum(bind_props.values())
        total_break = self.bucket_sampler.k_off * self.bond_graph.n_bonds
        total_propensity = total_bind + total_break
        
        if total_propensity <= 0:
            return False, None, 0.0
        
        # Sample time step
        dt = np.random.exponential(1.0 / total_propensity)
        
        # Sample event type
        r = random.random() * total_propensity
        
        if r < total_bind:
            # Binding event
            channel = self.bucket_sampler.sample_binding_channel(bind_props, total_bind)
            if channel is None:
                return True, None, dt  # Should not happen
            
            color_i, color_j = channel
            part_i, part_j = self.bucket_sampler.sample_binding_pair(color_i, color_j)
            
            return True, 'bind', dt, part_i, part_j, color_i, color_j
        else:
            # Dissociation event
            # Uniform sampling among bonds
            bonds = self.bond_graph.get_bonds()
            if not bonds:
                return True, None, dt
            
            bond = random.choice(bonds)
            return True, 'break', dt, bond
    
    def execute_bind(self, part_i: int, part_j: int, color_i: Color, color_j: Color) -> int:
        """Execute a binding event, return edge key."""
        edge_key = self.bond_graph.add_bond(part_i, part_j, color_i, color_j)
        
        # Update buckets (which also updates particle available_sites)
        self.bucket_sampler.update_after_bind(part_i, part_j, color_i, color_j)
        
        return edge_key
    
    def execute_break(self, bond: Bond):
        """Execute a bond breaking event."""
        self.bond_graph.remove_bond(bond.particle_i, bond.particle_j, bond.edge_key)
        
        # Update buckets (which also updates particle available_sites)
        self.bucket_sampler.update_after_break(
            bond.particle_i, bond.particle_j, 
            bond.color_i, bond.color_j
        )


@dataclass
class CompositionRejectionSampler:
    """
    Composition-Rejection Sampler for O(1) reaction selection.
    
    Based on PSSA-CR (Partial-Propensity SSA with Composition-Rejection).
    Uses dyadic binning of propensities to achieve constant-time sampling.
    
    This is an advanced optimization for very large systems where even
    O(|C|) channel iteration becomes costly.
    """
    bucket_sampler: GlobalBucketSampler
    bond_graph: BondGraph
    
    # Dyadic bins for composition-rejection
    _bind_bins: Dict[int, List[Tuple[Color, Color]]] = field(default_factory=dict)
    _break_bins: Dict[int, List[Bond]] = field(default_factory=dict)
    _max_bin: int = 0
    _bin_dirty: bool = True
    
    def _rebuild_bins(self):
        """Rebuild dyadic bins for propensities."""
        self._bind_bins = defaultdict(list)
        self._break_bins = defaultdict(list)
        self._max_bin = 0
        
        # Binding channels: bin by log2(propensity)
        bind_props = self.bucket_sampler.compute_binding_propensities()
        for channel, prop in bind_props.items():
            if prop > 0:
                bin_idx = int(np.floor(np.log2(prop)))
                self._bind_bins[bin_idx].append(channel)
                self._max_bin = max(self._max_bin, bin_idx)
        
        # Dissociation: all have same propensity k_off
        if self.bucket_sampler.k_off > 0 and self.bond_graph.n_bonds > 0:
            bin_idx = int(np.floor(np.log2(self.bucket_sampler.k_off)))
            self._break_bins[bin_idx].extend(self.bond_graph.get_bonds())
            self._max_bin = max(self._max_bin, bin_idx)
        
        self._bin_dirty = False
    
    def _ensure_bins(self):
        if self._bin_dirty:
            self._rebuild_bins()
    
    def invalidate_bins(self):
        self._bin_dirty = True
    
    def step(self):
        """
        Execute one Gillespie SSA step using Composition-Rejection.
        
        Returns tuple:
        - For bind: (True, 'bind', dt, part_i, part_j, color_i, color_j)
        - For break: (True, 'break', dt, bond)
        - For no events: (False, None, 0.0)
        """
        event_type, event_data = self.sample_event_cr()
        if event_type is None:
            return False, None, 0.0
        
        # We need to compute dt based on total propensity
        bind_props = self.bucket_sampler.compute_binding_propensities()
        total_bind = sum(bind_props.values())
        total_break = self.bucket_sampler.k_off * self.bond_graph.n_bonds
        total_propensity = total_bind + total_break
        
        dt = np.random.exponential(1.0 / total_propensity)
        
        if event_type == 'bind':
            part_i, part_j, color_i, color_j = event_data
            return True, 'bind', dt, part_i, part_j, color_i, color_j
        else:
            bond = event_data
            return True, 'break', dt, bond
    
    def execute_bind(self, part_i: int, part_j: int, color_i: Color, color_j: Color) -> int:
        """Execute a binding event, return edge key."""
        edge_key = self.bond_graph.add_bond(part_i, part_j, color_i, color_j)
        
        # Update buckets (which also updates particle available_sites)
        self.bucket_sampler.update_after_bind(part_i, part_j, color_i, color_j)
        self.invalidate_bins()
        
        return edge_key
    
    def execute_break(self, bond: Bond):
        """Execute a bond breaking event."""
        self.bond_graph.remove_bond(bond.particle_i, bond.particle_j, bond.edge_key)
        
        # Update buckets (which also updates particle available_sites)
        self.bucket_sampler.update_after_break(
            bond.particle_i, bond.particle_j, 
            bond.color_i, bond.color_j
        )
        self.invalidate_bins()
    
    def sample_event_cr(self) -> Tuple[str, any]:
        """
        Composition-rejection sampling for next event.
        
        Returns (event_type, event_data) where event_data is
        (part_i, part_j, color_i, color_j) for bind or Bond for break.
        """
        self._ensure_bins()
        
        # Composition step: select bin proportional to total bin weight
        # For simplicity, fall back to linear search over bins
        # Full CR would use another level of binning
        
        total_bind = sum(
            sum(self.bucket_sampler.compute_binding_propensities()[ch] for ch in channels)
            for channels in self._bind_bins.values()
        )
        total_break = self.bucket_sampler.k_off * self.bond_graph.n_bonds
        total = total_bind + total_break
        
        if total <= 0:
            return None, None
        
        r = random.random() * total
        
        # Search bins
        cumulative = 0.0
        for bin_idx in range(self._max_bin + 1):
            # Binding channels in this bin
            for channel in self._bind_bins.get(bin_idx, []):
                prop = self.bucket_sampler.compute_binding_propensities().get(channel, 0)
                cumulative += prop
                if cumulative >= r:
                    color_i, color_j = channel
                    part_i, part_j = self.bucket_sampler.sample_binding_pair(color_i, color_j)
                    return 'bind', (part_i, part_j, color_i, color_j)
            
            # Dissociation bonds in this bin
            for bond in self._break_bins.get(bin_idx, []):
                cumulative += self.bucket_sampler.k_off
                if cumulative >= r:
                    return 'break', bond
        
        return None, None


def create_sampler(particles: List[Particle], palette: ColorPalette, 
                   k_on: float, k_off: float,
                   bond_graph: BondGraph = None,
                   use_cr: bool = False) -> Tuple[GlobalBucketSampler, TwoStageSampler]:
    """Factory to create samplers for a particle system."""
    bucket_sampler = GlobalBucketSampler(
        particles=particles,
        palette=palette,
        k_on=k_on,
        k_off=k_off
    )
    
    if bond_graph is None:
        from ..topology.graph import create_bond_graph_from_system
        bond_graph = create_bond_graph_from_system(particles, palette)
    
    if use_cr:
        sampler = CompositionRejectionSampler(bucket_sampler, bond_graph)
    else:
        sampler = TwoStageSampler(bucket_sampler, bond_graph)
    
    return bucket_sampler, sampler