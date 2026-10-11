"""
Port palette definitions for the IGSPA system.

This module defines a port-based palette that replaces the older
complementarity-only color model with an explicit set of valid unordered
binding pairs. This keeps the runtime complexity manageable while allowing
more flexible interaction graphs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, Iterable, List, Tuple


@dataclass(frozen=True)
class PortPalette:
    """
    Immutable palette defining a set of port labels and the valid interactions
    between them.

    The palette identifies:
    - ports: all port labels in the system
    - valid_pairs: unordered port pairs that may bind
    - str_map: string-to-port conversion
    """

    ports: FrozenSet[str]
    valid_pairs: FrozenSet[Tuple[str, str]]
    str_map: Dict[str, str] = field(default_factory=dict)

    def __post_init__(self):
        normalized_pairs = set()
        for p1, p2 in self.valid_pairs:
            a, b = sorted((str(p1), str(p2)))
            normalized_pairs.add((a, b))

        object.__setattr__(self, 'valid_pairs', frozenset(normalized_pairs))

        for port in self.ports:
            object.__setattr__(self, 'str_map', {**self.str_map, str(port).lower(): str(port)})

        # Ensure all pair members are known ports
        for p1, p2 in self.valid_pairs:
            if p1 not in self.ports or p2 not in self.ports:
                raise ValueError(f"Pair {p1}-{p2} references an unknown port")

    @property
    def colors(self):
        return self.ports

    @classmethod
    def standard(cls) -> "PortPalette":
        ports = frozenset({"A", "B", "C", "D", "E", "F"})
        valid_pairs = frozenset({("A", "B"), ("C", "D"), ("E", "F")})
        return cls(ports=ports, valid_pairs=valid_pairs)

    @classmethod
    def from_strings(
        cls,
        port_names: Iterable[str],
        pair_specs: Iterable[Tuple[str, str]],
    ) -> "PortPalette":
        ports = frozenset(str(name).lower() for name in port_names)
        valid_pairs = set()
        for a, b in pair_specs:
            pa, pb = str(a).lower(), str(b).lower()
            if pa not in ports or pb not in ports:
                raise ValueError(f"Pair ({pa}, {pb}) references ports not in the palette")
            valid_pairs.add(tuple(sorted((pa, pb))))
        return cls(ports=ports, valid_pairs=frozenset(valid_pairs))

    def get_channels(self) -> List[Tuple[str, str]]:
        return sorted(self.valid_pairs, key=lambda pair: (pair[0], pair[1]))

    def is_valid_pair(self, port_a: str, port_b: str) -> bool:
        return tuple(sorted((str(port_a).lower(), str(port_b).lower()))) in self.valid_pairs

    def __contains__(self, port):
        return str(port).lower() in self.ports

    def __len__(self):
        return len(self.ports)

    def __iter__(self):
        return iter(self.ports)


DEFAULT_PORT_PALETTE = PortPalette.standard()


__all__ = ["PortPalette", "DEFAULT_PORT_PALETTE"]
