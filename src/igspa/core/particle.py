"""
Particle definitions: blueprints, instances, and site management.

Particles are the fundamental units of the assembly system. Each particle has:
- A unique identifier
- A static blueprint defining total site capacities per color (K_i^c)
- Dynamic state tracking available sites (θ_i^c = available / capacity)
- Optional label for visualization/debugging
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Mapping
from collections import Counter

from .colors import Color, ColorPalette, COMPLEMENT_MAP, COLOR_STR_MAP


@dataclass(frozen=True)
class ParticleBlueprint:
    """
    Immutable blueprint defining a particle type's binding site capacities.
    
    This corresponds to K_i^c in the mathematical formulation - the static
    total capacity of color c sites on particle i.
    """
    site_counts: Dict[Color, int]  # Color -> capacity
    label: Optional[str] = None
    
    def __post_init__(self):
        # Validate all counts are positive
        for color, count in self.site_counts.items():
            if count <= 0:
                raise ValueError(f"Site count for {color} must be positive, got {count}")
    
    @classmethod
    def from_color_list(cls, colors: List[Color], label: Optional[str] = None) -> 'ParticleBlueprint':
        """Create blueprint from a list of colors (counts duplicates)."""
        counts = Counter(colors)
        return cls(site_counts=dict(counts), label=label)
    
    @classmethod
    def from_string_list(cls, color_strings: List[str], palette: ColorPalette, 
                         label: Optional[str] = None) -> 'ParticleBlueprint':
        """Create blueprint from list of color strings."""
        colors = [COLOR_STR_MAP[s.lower()] for s in color_strings]
        return cls.from_color_list(colors, label)
    
    @property
    def total_sites(self) -> int:
        return sum(self.site_counts.values())
    
    @property
    def colors_present(self) -> frozenset[Color]:
        return frozenset(self.site_counts.keys())
    
    def __str__(self) -> str:
        parts = [f"{color.name.lower()}:{count}" for color, count in sorted(self.site_counts.items(), key=lambda x: x[0].name)]
        label_str = f" ({self.label})" if self.label else ""
        return f"Blueprint[{', '.join(parts)}]{label_str}"


@dataclass
class Particle:
    """
    Dynamic particle instance with mutable binding state.
    
    Tracks available sites per color (θ_i^c * K_i^c). When a site binds,
    available count decreases; when a bond breaks, it increases.
    
    Invariants:
    - 0 <= available_sites[c] <= blueprint.site_counts[c] for all c
    - Particle can only bind if it has available sites of the required color
    """
    blueprint: ParticleBlueprint
    particle_id: int
    available_sites: Dict[Color, int] = field(init=False)
    label: str = field(init=False)
    
    def __post_init__(self):
        self.available_sites = self.blueprint.site_counts.copy()
        self.label = self.blueprint.label or f"P{self.particle_id}"
    
    @property
    def capacities(self) -> Mapping[Color, int]:
        """Static site capacities (K_i^c)."""
        return self.blueprint.site_counts
    
    @property
    def fractions(self) -> Dict[Color, float]:
        """Availability fractions θ_i^c = available / capacity."""
        return {
            color: self.available_sites[color] / self.capacities[color]
            for color in self.capacities
        }
    
    def has_available(self, color: Color) -> bool:
        """Check if particle has at least one available site of given color."""
        return self.available_sites.get(color, 0) > 0
    
    def available_count(self, color: Color) -> int:
        """Get number of available sites of given color."""
        return self.available_sites.get(color, 0)
    
    def bind_site(self, color: Color) -> bool:
        """
        Consume one site of given color for binding.
        Returns True if successful, False if no sites available.
        """
        if not self.has_available(color):
            return False
        self.available_sites[color] -= 1
        return True
    
    def release_site(self, color: Color) -> bool:
        """
        Release one site of given color (bond broken).
        Returns True if successful, False if already at capacity.
        """
        capacity = self.capacities.get(color, 0)
        if self.available_sites.get(color, 0) >= capacity:
            return False
        self.available_sites[color] = self.available_sites.get(color, 0) + 1
        return True
    
    def get_active_colors(self) -> List[Color]:
        """Get colors with at least one available site."""
        return [c for c, count in self.available_sites.items() if count > 0]
    
    def is_saturated(self) -> bool:
        """Check if all sites are bound (no available sites)."""
        return all(count == 0 for count in self.available_sites.values())
    
    def is_empty(self) -> bool:
        """Check if all sites are available (no bonds)."""
        return all(self.available_sites[c] == self.capacities[c] for c in self.capacities)
    
    def __str__(self) -> str:
        sites_str = ", ".join(
            f"{c.name.lower()}:{self.available_sites[c]}/{self.capacities[c]}"
            for c in sorted(self.capacities.keys(), key=lambda x: x.name)
        )
        return f"{self.label}[{sites_str}]"
    
    def __repr__(self) -> str:
        return f"Particle(id={self.particle_id}, label='{self.label}', sites={dict(self.available_sites)})"


def create_particles_from_blueprints(
    blueprints: List[ParticleBlueprint],
    palette: ColorPalette = None
) -> List[Particle]:
    """Factory function to create particle instances from blueprints."""
    if palette is None:
        palette = ColorPalette.standard()
    
    particles = []
    for idx, blueprint in enumerate(blueprints):
        # Validate blueprint colors are in palette
        for color in blueprint.colors_present:
            if color not in palette:
                raise ValueError(f"Blueprint color {color} not in palette")
        particle = Particle(blueprint=blueprint, particle_id=idx)
        # Ensure available_sites has entries for all colors in palette
        for color in palette.colors:
            if color not in particle.available_sites:
                particle.available_sites[color] = 0
        particles.append(particle)
    return particles