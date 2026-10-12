"""System configuration binding together port palette and archetypes."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from .particle import ParticleArchetype
from .ports import PortPalette


@dataclass(frozen=True)
class SystemConfiguration:
    """
    Complete system specification: port palette + particle archetypes.

    This replaces the older approach of having separate palette and blueprint
    definitions, bundling them into a single coherent configuration object.
    """
    name: str
    palette: PortPalette
    archetypes: tuple[ParticleArchetype, ...] = field(default_factory=tuple)

    def __post_init__(self):
        # Validate that every archetype's ports are present in the palette
        for archetype in self.archetypes:
            for port in archetype.port_counts:
                if port not in self.palette:
                    raise ValueError(
                        f"Archetype {archetype.label} uses port '{port}' "
                        f"which is not in palette {self.palette.ports}"
                    )

    @classmethod
    def standard(cls) -> 'SystemConfiguration':
        """Create the standard system with the default port pairs."""
        from .particle import ParticleArchetype  # Avoid circular import
        
        palette = PortPalette.standard()
        
        # Define standard archetypes matching legacy system
        archetypes = frozenset([
            ParticleArchetype.from_port_list(["white"], label="W"),
            ParticleArchetype.from_port_list(["black"], label="B"),
            ParticleArchetype.from_port_list(["red"], label="R"),
            ParticleArchetype.from_port_list(["green"], label="G"),
            ParticleArchetype.from_port_list(["blue"], label="U"),  # Using U for blue to avoid conflict
            ParticleArchetype.from_port_list(["yellow"], label="Y"),
        ])
        
        return cls(name="standard", palette=palette, archetypes=archetypes)

    @classmethod
    def from_spec(
        cls,
        name: str,
        port_names: Set[str],
        pair_specs: Set[tuple[str, str]],
        archetype_specs: List[List[str]]
    ) -> 'SystemConfiguration':
        """Create system configuration from raw specifications.

        Args:
            name: Configuration name
            port_names: Set of port label strings
            pair_specs: Set of tuples (port_a, port_b) representing valid unordered pairs
            archetype_specs: List of lists, where each inner list contains port strings
                           defining an archetype's port composition
        """
        palette = PortPalette.from_strings(port_names, pair_specs)
        
        archetypes = tuple(
            ParticleArchetype.from_port_list(ports, label=f"{ports[0].upper()}{i}")
            for i, ports in enumerate(archetype_specs)
        )
        
        return cls(name=name, palette=palette, archetypes=archetypes)

    def get_archetype_by_label(self, label: str) -> Optional[ParticleArchetype]:
        """Find an archetype by its label."""
        for archetype in self.archetypes:
            if archetype.label == label:
                return archetype
        return None

    def num_archetypes(self) -> int:
        """Get the number of distinct archetypes."""
        return len(self.archetypes)

    def num_ports(self) -> int:
        """Get the number of distinct ports in the palette."""
        return len(self.palette)

    def num_channels(self) -> int:
        """Get the number of distinct binding channels (valid pairs)."""
        return len(self.palette.valid_pairs)
