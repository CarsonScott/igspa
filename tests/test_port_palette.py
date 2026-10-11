"""Tests for the new port-palette system."""

import pytest
import random
from igspa.core.ports import PortPalette
from igspa.core.particle import ParticleArchetype, create_particles_from_configuration, create_particles_from_palette, create_particles_from_blueprints
from igspa.core.config import SystemConfiguration
from igspa.core.system import create_system_from_config, ParticleSystem


def test_port_palette_standard():
    """Test that the standard port palette matches the legacy color pairs."""
    palette = PortPalette.standard()
    assert len(palette) == 6
    assert "white" in palette
    assert "black" in palette
    assert "red" in palette
    assert "green" in palette
    assert "blue" in palette
    assert "yellow" in palette
    
    # Check valid pairs
    channels = palette.get_channels()
    assert len(channels) == 3
    assert ("white", "black") in channels
    assert ("red", "green") in channels
    assert ("blue", "yellow") in channels
    
    # Check validity
    assert palette.is_valid_pair("white", "black") == True
    assert palette.is_valid_pair("black", "white") == True  # unordered
    assert palette.is_valid_pair("red", "green") == True
    assert palette.is_valid_pair("green", "red") == True
    assert palette.is_valid_pair("blue", "yellow") == True
    assert palette.is_valid_pair("yellow", "blue") == True
    
    # Check invalid pairs
    assert palette.is_valid_pair("white", "red") == False
    assert palette.is_valid_pair("black", "blue") == False


def test_port_palette_from_strings():
    """Test creating a palette from port names and pair specifications."""
    port_names = {"A", "B", "C", "D"}
    pair_specs = {("A", "B"), ("C", "D")}
    palette = PortPalette.from_strings(port_names, pair_specs)
    
    assert len(palette) == 4
    assert "A" in palette
    assert "B" in palette
    assert "C" in palette
    assert "D" in palette
    
    channels = palette.get_channels()
    assert len(channels) == 2
    assert ("A", "B") in channels
    assert ("C", "D") in channels
    
    assert palette.is_valid_pair("A", "B") == True
    assert palette.is_valid_pair("B", "A") == True
    assert palette.is_valid_pair("C", "D") == True
    assert palette.is_valid_pair("D", "C") == True
    
    assert palette.is_valid_pair("A", "C") == False
    assert palette.is_valid_pair("B", "D") == False


def test_port_palette_invalid_pair():
    """Test that creating a palette with a port not in the set raises an error."""
    port_names = {"A", "B", "C"}
    pair_specs = {("A", "B"), ("B", "C"), ("C", "D")}  # D not in port_names
    with pytest.raises(ValueError):
        PortPalette.from_strings(port_names, pair_specs)


def test_particle_archetype_from_port_list():
    """Test creating an archetype from a list of ports."""
    archetype = ParticleArchetype.from_port_list(["white", "white", "red"], label="Test")
    assert archetype.port_counts["white"] == 2
    assert archetype.port_counts["red"] == 1
    assert archetype.total_sites == 3
    assert archetype.label == "Test"
    assert archetype.ports_present == frozenset(["white", "red"])


def test_particle_archetype_from_string_list():
    """Test creating an archetype from a list of port strings with a palette."""
    palette = PortPalette.standard()
    archetype = ParticleArchetype.from_string_list(["white", "red", "white"], palette, label="Test")
    assert archetype.port_counts["white"] == 2
    assert archetype.port_counts["red"] == 1
    assert archetype.total_sites == 3
    assert archetype.label == "Test"


def test_particle_archetype_invalid_port():
    """Test that creating an archetype with a port not in the palette raises an error."""
    palette = PortPalette.standard()
    with pytest.raises(ValueError):
        ParticleArchetype.from_string_list(["white", "invalid"], palette)


def test_create_particles_from_configuration_uniform():
    """Test creating particles from a configuration with uniform probabilities."""
    # Create a simple configuration
    palette = PortPalette.from_strings({"A", "B"}, {("A", "B")})
    archetype1 = ParticleArchetype.from_port_list(["A"], label="A1")
    archetype2 = ParticleArchetype.from_port_list(["B"], label="B1")
    config = SystemConfiguration("test", palette, frozenset([archetype1, archetype2]))
    
    # Create 10 particles with uniform probability (default)
    particles = create_particles_from_configuration(10, config, seed=42)
    assert len(particles) == 10
    
    # Check that we got a mix of archetypes
    labels = [p.archetype.label for p in particles]
    assert "A1" in labels
    assert "B1" in labels
    # With seed 42, we can check the exact distribution if we want, but for now just check both present


def test_create_particles_from_configuration_list_probabilities():
    """Test creating particles from a configuration with list probabilities."""
    palette = PortPalette.from_strings({"A", "B"}, {("A", "B")})
    archetype1 = ParticleArchetype.from_port_list(["A"], label="A1")
    archetype2 = ParticleArchetype.from_port_list(["B"], label="B1")
    config = SystemConfiguration("test", palette, frozenset([archetype1, archetype2]))
    
    # Create particles with 75% A1, 25% B1
    particles = create_particles_from_configuration(100, config, p=[0.75, 0.25], seed=42)
    assert len(particles) == 100
    
    labels = [p.archetype.label for p in particles]
    count_A1 = labels.count("A1")
    count_B1 = labels.count("B1")
    # Check that the counts are approximately correct (allowing for randomness)
    assert 70 <= count_A1 <= 80
    assert 20 <= count_B1 <= 30


def test_create_particles_from_configuration_dict_probabilities():
    """Test creating particles from a configuration with dict probabilities."""
    palette = PortPalette.from_strings({"A", "B"}, {("A", "B")})
    archetype1 = ParticleArchetype.from_port_list(["A"], label="A1")
    archetype2 = ParticleArchetype.from_port_list(["B"], label="B1")
    config = SystemConfiguration("test", palette, frozenset([archetype1, archetype2]))
    
    # Create particles with dict probabilities
    p_dict = {archetype1: 0.25, archetype2: 0.75}
    particles = create_particles_from_configuration(100, config, p=p_dict, seed=42)
    assert len(particles) == 100
    
    labels = [p.archetype.label for p in particles]
    count_A1 = labels.count("A1")
    count_B1 = labels.count("B1")
    # Check that the counts are approximately correct (allowing for randomness)
    assert 20 <= count_A1 <= 30
    assert 70 <= count_B1 <= 80


def test_create_particles_from_palette():
    """Test the helper function that creates particles from palette and archetypes directly."""
    palette = PortPalette.from_strings({"A", "B"}, {("A", "B")})
    archetype1 = ParticleArchetype.from_port_list(["A"], label="A1")
    archetype2 = ParticleArchetype.from_port_list(["B"], label="B1")
    
    particles = create_particles_from_palette(50, palette, [archetype1, archetype2], p=[0.5, 0.5], seed=42)
    assert len(particles) == 50
    
    labels = [p.archetype.label for p in particles]
    assert labels.count("A1") > 0
    assert labels.count("B1") > 0


def test_create_particles_from_blueprints_backward_compatibility():
    """Test that the backward compatibility function works."""
    palette = PortPalette.standard()
    # Use the old-style blueprints (which are now ParticleArchetype)
    blueprint1 = ParticleArchetype.from_port_list(["white"], label="W")
    blueprint2 = ParticleArchetype.from_port_list(["black"], label="B")

    particles = create_particles_from_blueprints([blueprint1, blueprint2], palette, seed=42)
    assert len(particles) == 2
    labels = {p.archetype.label for p in particles}
    assert labels == {"W", "B"}


def test_system_configuration_standard():
    """Test creating a standard system configuration."""
    config = SystemConfiguration.standard()
    assert config.name == "standard"
    assert isinstance(config.palette, PortPalette)
    assert len(config.archetypes) == 6
    # Check that we have archetypes for each port
    labels = {a.label for a in config.archetypes}
    assert labels == {"W", "B", "R", "G", "U", "Y"}  # U for blue, Y for yellow
    
    # Validate that the archetypes use ports from the palette
    for archetype in config.archetypes:
        for port in archetype.port_counts:
            assert port in config.palette


def test_system_configuration_from_spec():
    """Test creating a system configuration from raw specifications."""
    config = SystemConfiguration.from_spec(
        name="test",
        port_names={"A", "B", "C"},
        pair_specs={("A", "B"), ("B", "C")},
        archetype_specs=[["A"], ["B"], ["C"], ["A", "B"]]
    )
    
    assert config.name == "test"
    assert len(config.palette.ports) == 3
    assert "A" in config.palette
    assert "B" in config.palette
    assert "C" in config.palette
    
    channels = config.palette.get_channels()
    assert len(channels) == 2
    assert ("A", "B") in channels
    assert ("B", "C") in channels
    
    assert config.num_archetypes() == 4
    assert config.num_ports() == 3
    assert config.num_channels() == 2
    
    # Check archetypes
    labels = [a.label for a in config.archetypes]
    assert "A0" in labels
    assert "B1" in labels
    assert "C2" in labels
    assert "A3" in labels  # The last one is ["A", "B"] -> label A3
    
    # Find the archetype with both A and B
    ab_archetype = None
    for a in config.archetypes:
        if set(a.port_counts.keys()) == {"A", "B"}:
            ab_archetype = a
            break
    assert ab_archetype is not None
    assert ab_archetype.port_counts["A"] == 1
    assert ab_archetype.port_counts["B"] == 1


def test_system_configuration_get_archetype_by_label():
    """Test finding an archetype by its label."""
    config = SystemConfiguration.standard()
    archetype = config.get_archetype_by_label("W")
    assert archetype is not None
    assert archetype.label == "W"
    assert archetype.port_counts["white"] == 1
    
    # Test non-existent label
    assert config.get_archetype_by_label("NonExistent") is None


def test_create_system_from_config():
    """Test creating a ParticleSystem from a SystemConfiguration."""
    config = SystemConfiguration.from_spec(
        name="test",
        port_names={"A", "B"},
        pair_specs={("A", "B")},
        archetype_specs=[["A"], ["B"]]
    )
    
    system = create_system_from_config(config, n_particles=10, k_on=1.0, k_off=0.1, seed=42)
    assert isinstance(system, ParticleSystem)
    assert system.n_particles == 10
    assert system.palette == config.palette
    assert system.config.k_on == 1.0
    assert system.config.k_off == 0.1
    
    # Run a few steps to make sure it works
    initial_bonds = system.n_bonds
    for _ in range(5):
        system.step()
    # Should have some bonds now (or at least not crash)
    assert system.n_bonds >= initial_bonds


def test_create_system_from_config_with_probabilities():
    """Test creating a system with archetype probabilities."""
    config = SystemConfiguration.from_spec(
        name="test",
        port_names={"A", "B"},
        pair_specs={("A", "B")},
        archetype_specs=[["A"], ["B"], ["A", "B"]]
    )
    
    # Use dict probabilities: 50% A, 30% B, 20% AB
    p_dict = {}
    for archetype in config.archetypes:
        if archetype.label == "A0":
            p_dict[archetype] = 0.5
        elif archetype.label == "B1":
            p_dict[archetype] = 0.3
        else:  # "A2"
            p_dict[archetype] = 0.2
    
    system = create_system_from_config(
        config, 
        n_particles=100, 
        p=p_dict,
        k_on=1.0, 
        k_off=0.1, 
        seed=42
    )
    assert system.n_particles == 100
    
    # Count archetypes in the system
    counts = {"A0": 0, "B1": 0, "A2": 0}
    for p in system.particles:
        label = p.archetype.label
        if label in counts:
            counts[label] += 1
    
    # Check approximate counts (allowing for randomness)
    assert 40 <= counts["A0"] <= 60
    assert 20 <= counts["B1"] <= 40
    assert 10 <= counts["A2"] <= 30


if __name__ == '__main__':
    pytest.main([__file__, '-v'])