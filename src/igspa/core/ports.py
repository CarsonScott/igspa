"""Port palette definitions for generalized valid-pair binding.

The port palette defines a set of port labels and explicit valid binding pairs,
replacing the legacy complement-pair model with arbitrary symmetric binding rules.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Set, Tuple, List


@dataclass
class PortPalette:
    """
    Immutable port palette defining binding alphabet and valid pairs.

    The palette defines the complete set of ports available in the system
    and their valid binding relationships (symmetric, unordered pairs).
    """
    ports: FrozenSet[str]
    valid_pairs_tuple: Tuple[Tuple[str, str], ...] = field(default_factory=tuple)
    _valid_pairs_set: FrozenSet[FrozenSet[str]] = field(init=False)
    str_map: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        # Validate that all ports in valid_pairs are in the palette
        valid_pairs_set = set()
        for port_a, port_b in self.valid_pairs_tuple:
            if port_a not in self.ports:
                raise ValueError(f"Port {port_a} in pair ({port_a}, {port_b}) not in palette")
            if port_b not in self.ports:
                raise ValueError(f"Port {port_b} in pair ({port_a}, {port_b}) not in palette")
            valid_pairs_set.add(frozenset([port_a.lower(), port_b.lower()]))
        
        object.__setattr__(self, '_valid_pairs_set', frozenset(valid_pairs_set))
        
        # Build string lookup map (normalized to lower case)
        for port in self.ports:
            self.str_map[port.lower()] = port

    @property
    def valid_pairs(self) -> FrozenSet[FrozenSet[str]]:
        """Get the set of valid binding pairs (as frozensets of port names)."""
        return self._valid_pairs_set

    @classmethod
    def standard(cls) -> 'PortPalette':
        """Create the standard 6-port, 3-palette matching the legacy color complement pairs."""
        ports = frozenset(["white", "black", "red", "green", "blue", "yellow"])
        valid_pairs_tuple = (
            ("white", "black"),
            ("red", "green"),
            ("blue", "yellow")
        )
        return cls(ports=ports, valid_pairs_tuple=valid_pairs_tuple)

    @classmethod
    def from_strings(cls, port_names: Set[str], pair_specs: Set[Tuple[str, str]]) -> 'PortPalette':
        """Create palette from port names and explicit pair specifications.

        Args:
            port_names: Set of port label strings
            pair_specs: Set of tuples (port_a, port_b) representing valid unordered pairs
        """
        # Normalize port names to lower case for internal consistency, but keep original casing in ports set
        # We'll store ports as given, but use lower case for lookup
        ports = frozenset(port_names)
        # Convert pair_specs to a sorted tuple to ensure consistent ordering
        # Sort by first element, then second element for determinism
        sorted_pairs = tuple(sorted(pair_specs, key=lambda x: (x[0].lower(), x[1].lower())))
        return cls(ports=ports, valid_pairs_tuple=sorted_pairs)

    def get_channels(self) -> list[tuple[str, str]]:
        """Get unique binding channels (port_a, port_b) without duplication.

        Returns:
            List of tuples (port_a, port_b) in the order they were specified,
            with duplicates removed.
        """
        seen = set()
        channels = []
        for port_a, port_b in self.valid_pairs_tuple:
            pair_key = frozenset([port_a.lower(), port_b.lower()])
            if pair_key not in seen:
                seen.add(pair_key)
                channels.append((port_a, port_b))
        return channels

    def is_valid_pair(self, port_a: str, port_b: str) -> bool:
        """Check if two ports form a valid binding pair (unordered)."""
        return frozenset([port_a.lower(), port_b.lower()]) in self._valid_pairs_set

    def __contains__(self, port: str) -> bool:
        return port.lower() in self.str_map

    def __len__(self) -> int:
        return len(self.ports)


# Default standard palette
DEFAULT_PORT_PALETTE = PortPalette.standard()