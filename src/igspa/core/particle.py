"""Particle definitions: blueprints, instances, and site management.

Particles are the fundamental units of the assembly system. Each particle has:
- A unique identifier
- A static blueprint defining total site capacities per port (K_i^c)
- Dynamic state tracking available sites (θ_i^c = available / capacity)
- Optional label for visualization/debugging
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Mapping, Union
from collections import Counter
import random

from .ports import PortPalette


@dataclass(frozen=True)
class ParticleArchetype:
    """
    Immutable blueprint defining a particle type's binding site capacities per port.
    
    This replaces the old ParticleBlueprint concept, using port-based semantics
    instead of color-based semantics.
    """
    port_counts: Dict[str, int]  # Port -> capacity
    label: Optional[str] = None
    
    def __post_init__(self):
        # Validate all counts are positive
        for port, count in self.port_counts.items():
            if count <= 0:
                raise ValueError(f"Site count for port '{port}' must be positive, got {count}")
    
    def __hash__(self):
        # Make the archetype hashable so it can be used in sets and frozensets
        return hash(tuple(sorted(self.port_counts.items())) + (hash(self.label),))
    
    @classmethod
    def from_port_list(cls, ports: List[str], label: Optional[str] = None) -> 'ParticleArchetype':
        """Create archetype from a list of ports (counts duplicates)."""
        counts = Counter(ports)
        return cls(port_counts=dict(counts), label=label)
    
    @classmethod
    def from_string_list(cls, port_strings: List[str], palette: PortPalette, 
                         label: Optional[str] = None) -> 'ParticleArchetype':
        """Create archetype from list of port strings."""
        # Validate ports are in palette
        for port_str in port_strings:
            if port_str.lower() not in palette.str_map:
                raise ValueError(f"Port '{port_str}' not in palette")
        ports = [palette.str_map[s.lower()] for s in port_strings]
        return cls.from_port_list(ports, label)
    
    @property
    def total_sites(self) -> int:
        return sum(self.port_counts.values())
    
    @property
    def ports_present(self) -> frozenset[str]:
        return frozenset(self.port_counts.keys())
    
    def __str__(self) -> str:
        parts = [f"{port}:{count}" for port, count in sorted(self.port_counts.items())]
        label_str = f" ({self.label})" if self.label else ""
        return f"Archetype[{', '.join(parts)}]{label_str}"
    
    def get_port_list(self) -> List[str]:
        """Get list of ports with duplicates (for compatibility)."""
        ports = []
        for port, count in self.port_counts.items():
            ports.extend([port] * count)
        return ports


@dataclass
class Particle:
    """
    Dynamic particle instance with mutable binding state.
    
    Tracks available sites per port (θ_i^c * K_i^c). When a site binds,
    available count decreases; when a bond breaks, it increases.
    
    Invariants:
    - 0 <= available_sites[port] <= archetype.port_counts[port] for all ports
    - Particle can only bind if it has available sites of the required port
    """
    archetype: ParticleArchetype
    particle_id: int
    available_sites: Dict[str, int] = field(init=False)
    label: str = field(init=False)
    
    def __post_init__(self):
        self.available_sites = self.archetype.port_counts.copy()
        self.label = self.archetype.label or f"P{self.particle_id}"
    
    @property
    def capacities(self) -> Mapping[str, int]:
        """Static site capacities (K_i^c)."""
        return self.archetype.port_counts
    
    @property
    def fractions(self) -> Dict[str, float]:
        """Availability fractions θ_i^c = available / capacity."""
        return {
            port: self.available_sites[port] / self.capacities[port]
            for port in self.capacities
        }
    
    def has_available(self, port: str) -> bool:
        """Check if particle has at least one available site of given port."""
        return self.available_sites.get(port, 0) > 0
    
    def available_count(self, port: str) -> int:
        """Get number of available sites of given port."""
        return self.available_sites.get(port, 0)
    
    def bind_port(self, port: str) -> bool:
        """
        Consume one site of given port for binding.
        Returns True if successful, False if no sites available.
        """
        if not self.has_available(port):
            return False
        self.available_sites[port] -= 1
        return True
    
    def release_port(self, port: str) -> bool:
        """
        Release one site of given port (bond broken).
        Returns True if successful, False if already at capacity.
        """
        capacity = self.capacities.get(port, 0)
        if self.available_sites.get(port, 0) >= capacity:
            return False
        self.available_sites[port] = self.available_sites.get(port, 0) + 1
        return True
    
    def get_active_ports(self) -> List[str]:
        """Get ports with at least one available site."""
        return [port for port, count in self.available_sites.items() if count > 0]
    
    def is_saturated(self) -> bool:
        """Check if all sites are bound (no available sites)."""
        return all(count == 0 for count in self.available_sites.values())
    
    def is_empty(self) -> bool:
        """Check if all sites are available (no bonds)."""
        return all(self.available_sites[port] == self.capacities[port] for port in self.capacities)
    
    def __str__(self) -> str:
        sites_str = ", ".join(
            f"{port}:{self.available_sites[port]}/{self.capacities[port]}"
            for port in sorted(self.capacities.keys())
        )
        return f"{self.label}[{sites_str}]"
    
    def __repr__(self) -> str:
        return f"Particle(id={self.particle_id}, label='{self.label}', sites={dict(self.available_sites)})"


# Backward compatibility alias
ParticleBlueprint = ParticleArchetype


def _normalize_probabilities(archetypes: List[ParticleArchetype], p: Optional[Union[List[float], Dict[ParticleArchetype, float]]]) -> List[float]:
    """Normalize probabilities to sum to 1."""
    if p is None:
        # Uniform distribution
        return [1.0 / len(archetypes)] * len(archetypes)
    
    if isinstance(p, list):
        if len(p) != len(archetypes):
            raise ValueError(f"Probability list length {len(p)} must match number of archetypes {len(archetypes)}")
        # Normalize the list
        total = sum(p)
        if total <= 0:
            raise ValueError("Probabilities must sum to a positive value")
        return [prob / total for prob in p]
    
    if isinstance(p, dict):
        # Convert dict to list in archetype order
        prob_list = []
        for archetype in archetypes:
            if archetype not in p:
                raise ValueError(f"Archetype {archetype} missing from probability dict")
            prob_list.append(p[archetype])
        # Normalize the list
        total = sum(prob_list)
        if total <= 0:
            raise ValueError("Probabilities must sum to a positive value")
        return [prob / total for prob in prob_list]
    
    raise TypeError("p must be None, a list, or a dict mapping archetypes to probabilities")


def create_particles_from_configuration(
    n_particles: int,
    config: 'SystemConfiguration',
    p: Optional[Union[List[float], Dict[ParticleArchetype, float]]] = None,
    seed: Optional[int] = None
) -> List[Particle]:
    """
    Factory function to create particle instances from a SystemConfiguration.
    
    Args:
        n_particles: Number of particles to create
        config: SystemConfiguration containing palette and archetypes
        p: Probability distribution for archetype selection:
           - None for uniform distribution
           - list of probabilities in archetype order
           - dict mapping archetype -> probability
        seed: Random seed for reproducibility
    
    Returns:
        List of Particle instances
    """
    if seed is not None:
        random.seed(seed)
    archetypes = list(config.archetypes)
    if not archetypes:
        raise ValueError("Configuration must contain at least one archetype")
    
    # Validate that all archetype ports are in palette (already done in SystemConfiguration.__post_init__)
    probabilities = _normalize_probabilities(archetypes, p)
    
    particles = []
    for i in range(n_particles):
        # Choose archetype based on probabilities
        archetype = random.choices(archetypes, weights=probabilities)[0]
        particle = Particle(archetype=archetype, particle_id=i)
        # Ensure available_sites has entries for all ports in palette
        for port in config.palette.ports:
            if port not in particle.available_sites:
                particle.available_sites[port] = 0
        particles.append(particle)
    
    return particles


def create_particles_from_palette(
    n_particles: int,
    palette: PortPalette,
    archetypes: List[ParticleArchetype],
    p: Optional[Union[List[float], Dict[ParticleArchetype, float]]] = None,
    seed: Optional[int] = None
) -> List[Particle]:
    """
    Factory function to create particle instances from palette and archetypes directly.
    
    Args:
        n_particles: Number of particles to create
        palette: PortPalette defining valid ports and pairs
        archetypes: List of ParticleArchetype instances
        p: Probability distribution for archetype selection (see create_particles_from_configuration)
        seed: Random seed for reproducibility
    
    Returns:
        List of Particle instances
    """
    # Create a temporary configuration
    from .config import SystemConfiguration
    config = SystemConfiguration(
        name="temp",
        palette=palette,
        archetypes=tuple(archetypes)
    )
    return create_particles_from_configuration(n_particles, config, p, seed)


# Backward compatibility function
def create_particles_from_blueprints(
    blueprints: List[ParticleArchetype],  # Using ParticleArchetype for backward compatibility
    palette: PortPalette = None,
    seed: Optional[int] = None
) -> List[Particle]:
    """
    Backward compatibility function - delegates to create_particles_from_palette.
    """
    if palette is None:
        palette = PortPalette.standard()
    return create_particles_from_palette(len(blueprints), palette, blueprints, seed=seed)
