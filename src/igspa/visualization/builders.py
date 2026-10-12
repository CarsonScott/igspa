"""Convenience builders for visualized particle systems."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Optional

import numpy as np

from ..core.config import SystemConfiguration
from ..core.particle import ParticleArchetype
from ..core.ports import PortPalette
from ..core.system import ParticleSystem, create_system_from_config


def create_palette(*valid_pairs: tuple[str, str]) -> PortPalette:
    """Build a port palette from its allowed binding pairs."""
    normalized_pairs = []
    port_names = set()
    for pair in valid_pairs:
        if len(pair) != 2:
            raise ValueError("Each valid pair must contain exactly two port names")
        port_a, port_b = pair
        normalized_pairs.append((port_a, port_b))
        port_names.update((port_a, port_b))
    return PortPalette.from_strings(port_names, set(normalized_pairs))


def create_archetypes(
    *port_lists: Iterable[str],
    palette: Optional[PortPalette] = None,
) -> list[ParticleArchetype]:
    """Build archetypes from port lists, optionally checking them against a palette."""
    archetypes = [ParticleArchetype.from_port_list(list(ports)) for ports in port_lists]
    if palette is not None:
        invalid_ports = {
            port
            for archetype in archetypes
            for port in archetype.port_counts
            if port not in palette
        }
        if invalid_ports:
            raise ValueError(f"Archetypes contain ports outside the palette: {sorted(invalid_ports)}")
    return archetypes


def create_config(
    palette: PortPalette,
    archetypes: Iterable[ParticleArchetype],
    name: str = "",
) -> SystemConfiguration:
    """Combine a palette and archetypes into a system configuration."""
    return SystemConfiguration(name=name, palette=palette, archetypes=tuple(archetypes))


def create_system(sys_def: Mapping) -> ParticleSystem:
    """Create a particle system from a compact dictionary specification.

    ``palette`` is a sequence of valid port pairs and ``archetypes`` is a
    sequence of port-name sequences. Existing ``PortPalette`` and
    ``ParticleArchetype`` objects can also be supplied directly.
    """
    required_keys = {"palette", "archetypes", "n_particles"}
    missing_keys = required_keys - sys_def.keys()
    if missing_keys:
        raise ValueError(f"System definition is missing required keys: {sorted(missing_keys)}")

    palette_spec = sys_def["palette"]
    if isinstance(palette_spec, PortPalette):
        palette = palette_spec
    else:
        palette = create_palette(*palette_spec)

    archetype_specs = sys_def["archetypes"]
    archetypes = []
    for spec in archetype_specs:
        if isinstance(spec, ParticleArchetype):
            archetypes.append(spec)
        else:
            ports = list(spec) if not isinstance(spec, str) else list(spec)
            archetypes.extend(create_archetypes(ports, palette=palette))

    config = create_config(palette, archetypes, name=sys_def.get("name", ""))
    return create_system_from_config(
        config,
        n_particles=sys_def["n_particles"],
        p=sys_def.get("probabilities"),
        k_on=sys_def.get("k_on", 1.0),
        k_off=sys_def.get("k_off", 0.1),
        seed=sys_def.get("seed"),
        use_composition_rejection=sys_def.get("use_composition_rejection", False),
    )


def random_palette(
    labels: Iterable[str],
    scalar_factor: float,
    seed: Optional[int] = None,
) -> PortPalette:
    """Generate a palette by independently selecting each possible port pair."""
    if not 0.0 <= scalar_factor <= 1.0:
        raise ValueError("scalar_factor must be between 0 and 1")
    input_labels = list(labels)
    if len(set(input_labels)) != len(input_labels):
        raise ValueError("Port labels must be unique")
    unique_labels = sorted(input_labels) if isinstance(labels, (set, frozenset)) else input_labels
    rng = np.random.default_rng(seed)
    pairs = [
        (port_a, port_b)
        for index, port_a in enumerate(unique_labels)
        for port_b in unique_labels[index + 1:]
        if rng.random() < scalar_factor
    ]
    return PortPalette.from_strings(set(unique_labels), set(pairs))


def random_archetypes(
    n_types: int,
    m_port_range: tuple[int, int],
    palette: PortPalette,
    seed: Optional[int] = None,
) -> list[ParticleArchetype]:
    """Generate archetypes with random port counts and assignments.

    ``m_port_range`` follows NumPy's convention: its upper bound is
    exclusive. Sampling with replacement permits multiple sites of one port.
    """
    minimum, maximum = m_port_range
    if n_types < 0:
        raise ValueError("n_types must be non-negative")
    if minimum < 1 or maximum <= minimum:
        raise ValueError("m_port_range must have a positive lower bound and a larger exclusive upper bound")
    if not palette.ports:
        raise ValueError("Cannot generate archetypes from an empty palette")

    rng = np.random.default_rng(seed)
    ports = sorted(palette.ports)
    archetypes = []
    for _ in range(n_types):
        count = int(rng.integers(minimum, maximum))
        selected = rng.choice(ports, size=count, replace=True).tolist()
        archetypes.append(ParticleArchetype.from_port_list(selected))
    return archetypes
