"""
Tests for IGSPA Core Components - Updated for Port-Palette API
"""

import pytest
import numpy as np
from collections import Counter

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from igspa.core.ports import PortPalette
from igspa.core.particle import ParticleArchetype, Particle, create_particles_from_blueprints
from igspa.core.config import SystemConfiguration
from igspa.core.system import ParticleSystem, create_system_from_config
from igspa.geometry.manifold import InformationManifold, FisherMetric
from igspa.topology.graph import BondGraph, Bond, ComplexAnalyzer
from igspa.algorithms.gillespie import (
    GlobalBucketSampler, TwoStageSampler, CompositionRejectionSampler
)
from igspa.utils.analysis import validate_system, SteadyStateAnalyzer, compute_thermodynamic_quantities


class TestPortPalette:
    """Test port palette system and validity."""

    def test_standard_palette(self):
        palette = PortPalette.standard()
        assert len(palette) == 6
        assert "white" in palette
        assert "black" in palette
        assert "red" in palette
        assert "green" in palette
        assert "blue" in palette
        assert "yellow" in palette

        # Check valid pairs (white-black, red-green, blue-yellow)
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

    def test_custom_palette(self):
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

    def test_invalid_palette_creation(self):
        """Test that creating a palette with a port not in the set raises an error."""
        port_names = {"A", "B", "C"}
        pair_specs = {("A", "B"), ("B", "C"), ("C", "D")}  # D not in port_names
        with pytest.raises(ValueError):
            PortPalette.from_strings(port_names, pair_specs)


class TestParticleArchetype:
    """Test particle archetype creation and validation."""

    def test_from_port_list(self):
        archetype = ParticleArchetype.from_port_list(["white", "white", "red"], label="Test")
        assert archetype.port_counts["white"] == 2
        assert archetype.port_counts["red"] == 1
        assert archetype.total_sites == 3
        assert archetype.label == "Test"
        assert archetype.ports_present == frozenset(["white", "red"])

    def test_from_string_list(self):
        palette = PortPalette.standard()
        archetype = ParticleArchetype.from_string_list(["white", "red", "white"], palette, label="Test")
        assert archetype.port_counts["white"] == 2
        assert archetype.port_counts["red"] == 1
        assert archetype.total_sites == 3
        assert archetype.label == "Test"

    def test_invalid_port(self):
        """Test that creating an archetype with a port not in the palette raises an error."""
        palette = PortPalette.standard()
        with pytest.raises(ValueError):
            ParticleArchetype.from_string_list(["white", "invalid"], palette)


class TestParticle:
    """Test dynamic particle state management."""

    def setup_method(self):
        self.archetype = ParticleArchetype.from_port_list(["white", "white", "red"], label="Test")
        self.particle = Particle(archetype=self.archetype, particle_id=0)

    def test_initial_state(self):
        assert self.particle.available_sites["white"] == 2
        assert self.particle.available_sites["red"] == 1
        assert self.particle.capacities["white"] == 2
        assert self.particle.capacities["red"] == 1

    def test_bind_port(self):
        assert self.particle.bind_port("white") == True
        assert self.particle.available_sites["white"] == 1
        assert self.particle.bind_port("white") == True
        assert self.particle.available_sites["white"] == 0
        assert self.particle.bind_port("white") == False  # Exhausted

    def test_release_port(self):
        self.particle.bind_port("white")
        self.particle.bind_port("white")
        assert self.particle.release_port("white") == True
        assert self.particle.available_sites["white"] == 1
        assert self.particle.release_port("white") == True
        assert self.particle.available_sites["white"] == 2
        assert self.particle.release_port("white") == False  # At capacity

    def test_fractions(self):
        frac = self.particle.fractions
        assert frac["white"] == 1.0
        assert frac["red"] == 1.0

        self.particle.bind_port("white")
        frac = self.particle.fractions
        assert frac["white"] == 0.5

    def test_is_saturated_empty(self):
        assert self.particle.is_empty() == True
        assert self.particle.is_saturated() == False

        self.particle.bind_port("white")
        self.particle.bind_port("white")
        self.particle.bind_port("red")

        assert self.particle.is_saturated() == True
        assert self.particle.is_empty() == False


class TestParticleSystem:
    """Test complete particle system."""

    def setup_method(self):
            self.system = create_system_from_config(
                n_particles=4,
                config=SystemConfiguration.from_spec(
                    name="test",
                    port_names={"white", "black", "red", "green", "blue", "yellow"},
                    pair_specs={("white", "black"), ("red", "green"), ("blue", "yellow")},
                    archetype_specs=[
                        ["white", "red"],
                        ["green", "blue"],
                        ["black", "green", "yellow"],
                        ["red", "blue"]
                    ]
                ),
                k_on=1.5, k_off=0.3,
                seed=42
            )

    def test_initialization(self):
        assert self.system.n_particles == 4
        assert len(self.system.palette.ports) == 6
        assert self.system.n_bonds == 0
        assert self.system.current_time == 0.0
        assert self.system.current_step == 0

    def test_step_execution(self):
        # First step should be a bind event (no bonds to break)
        result = self.system.step()
        assert result == True
        assert self.system.n_bonds == 1
        assert self.system.current_step == 1
        assert self.system.current_time > 0
        assert self.system.stats['bind_events'] == 1

    def test_multiple_steps(self):
        for _ in range(5):
            self.system.step()
        assert self.system.current_step == 5
        assert self.system.n_bonds >= 1

    def test_equilibrium(self):
        # Run until equilibrium or max steps
        for _ in range(100):
            if not self.system.step():
                break
        # Should not crash

    def test_complexes(self):
        for _ in range(10):
            self.system.step()
        complexes = self.system.get_complexes()
        assert isinstance(complexes, list)
        total_particles = sum(len(c) for c in complexes)
        assert total_particles == self.system.n_particles

    def test_bond_list(self):
        for _ in range(5):
            self.system.step()
        bonds = self.system.get_bond_list()
        assert len(bonds) == self.system.n_bonds
        for b in bonds:
            assert len(b) == 4  # i, j, port_i, port_j

    def test_availability_matrix(self):
            avail = self.system.get_availability_matrix()
            # Matrix should have entries for all particles (some may share labels)
            assert len(avail) >= 1
            for particle_data in avail.values():
                assert all(isinstance(v, int) for v in particle_data.values())

    def test_state_snapshot(self):
        for _ in range(5):
            self.system.step()
        snapshot = self.system.get_state_snapshot()
        assert snapshot.step == self.system.current_step
        assert snapshot.time == self.system.current_time
        assert len(snapshot.particles) == 4


class TestInformationManifold:
    """Test information manifold geometry."""

    def setup_method(self):
        self.system = create_system_from_config(
            n_particles=2,
            config=SystemConfiguration.from_spec(
                name="test",
                port_names={"white", "black", "red", "green", "blue", "yellow"},
                pair_specs={("white", "black"), ("red", "green"), ("blue", "yellow")},
                archetype_specs=[["white", "red"], ["green", "blue"]]
            ),
            k_on=1.0, k_off=0.1,
            seed=42
        )

    def test_manifold_creation(self):
        manifold = self.system.manifold
        assert manifold.n_particles == 2
        assert manifold.n_ports == 6

    def test_global_theta(self):
        theta = self.system.manifold.get_global_theta()
        assert theta.shape == (2 * 6,)  # 2 particles, 6 ports
        assert np.all(theta >= 0)
        assert np.all(theta <= 1)

    def test_channel_aggregates(self):
        agg = self.system.manifold.compute_channel_aggregates()
        assert len(agg) == 6
        # Initially all sites available
        assert agg["white"] == 1  # Only P0 has white
        assert agg["red"] == 1    # Only P0 has red
        assert agg["green"] == 1  # Only P1 has green
        assert agg["blue"] == 1   # Only P1 has blue

    def test_binding_propensities(self):
        props = self.system.manifold.compute_binding_propensities(1.0)
        # white-black channel: P0 has white, P1 has black (but P1 doesn't exist in 2-particle system)
        # Actually, in our 2-particle system we have P0: [white,red], P1: [green,blue]
        # So white-black should be 0 (no black available)
        assert props.get(("white", "black"), 0.0) == 0.0
        # red-green: P0 has red, P1 has green = 1*1 = 1
        assert props.get(("red", "green"), 0.0) == 1.0

    def test_dissociation_propensity(self):
        prop = self.system.manifold.compute_dissociation_propensity(0.1, 5)
        assert prop == 0.5

    def test_distance_to_saturation(self):
        dist = self.system.manifold.distance_to_saturation(0)
        # Initially all sites available, so distance should be large
        assert dist > 0


class TestBondGraph:
    """Test explicit graph topology."""

    def setup_method(self):
        self.system = create_system_from_config(
            n_particles=2,
            config=SystemConfiguration.from_spec(
                name="test",
                port_names={"white", "black", "red", "green", "blue", "yellow"},
                pair_specs={("white", "black"), ("red", "green"), ("blue", "yellow")},
                archetype_specs=[["white", "red"], ["green", "blue"]]
            ),
            k_on=1.0, k_off=0.1,
            seed=42
        )
        self.graph = self.system.bond_graph

    def test_add_remove_bond(self):
        key = self.graph.add_bond(0, 1, "white", "black")
        assert self.graph.n_bonds == 1
        assert key == 0

        bond = self.graph.remove_bond(0, 1, key)
        assert bond is not None
        assert bond.particle_i == 0
        assert bond.particle_j == 1
        assert self.graph.n_bonds == 0

    def test_multiple_bonds(self):
        self.graph.add_bond(0, 1, "white", "black")
        self.graph.add_bond(0, 1, "red", "green")
        assert self.graph.n_bonds == 2

        bonds = self.graph.get_bonds()
        assert len(bonds) == 2

    def test_bond_endpoint_orientation_is_preserved(self):
        key = self.graph.add_bond(1, 0, "white", "black")

        bond = self.graph.get_bonds()[0]
        assert (bond.particle_i, bond.port_i) == (1, "white")
        assert (bond.particle_j, bond.port_j) == (0, "black")
        assert self.graph.get_bonds_by_port("black")[0] == bond
        assert self.graph.get_particle_bonds(0)[0].port_i == "black"

        removed = self.graph.remove_bond(0, 1, key)
        assert (removed.particle_i, removed.port_i) == (1, "white")
        assert self.graph.get_port_degree(1, "white") == 0
        assert self.graph.get_port_degree(0, "black") == 0

    def test_port_degree(self):
        self.graph.add_bond(0, 1, "white", "black")
        assert self.graph.get_port_degree(0, "white") == 1
        assert self.graph.get_port_degree(1, "black") == 1
        assert self.graph.get_port_degree(0, "red") == 0

    def test_adjacency_matrix(self):
        self.graph.add_bond(0, 1, "white", "black")
        A = self.graph.get_adjacency_matrix("white")
        assert A[0, 1] == 1
        assert A[1, 0] == 0  # Directional: port_i is at particle_i

    def test_complex_analyzer(self):
        self.graph.add_bond(0, 1, "white", "black")
        analyzer = ComplexAnalyzer(self.graph)
        complexes = analyzer.find_complexes()
        assert len(complexes) == 1
        assert len(complexes[0]) == 2

        stats = analyzer.get_complex_statistics()
        assert stats['n_complexes'] == 1
        assert stats['largest_size'] == 2


class TestGlobalBucketSampler:
    """Test global bucket optimization."""

    def setup_method(self):
        self.system = create_system_from_config(
            n_particles=2,
            config=SystemConfiguration.from_spec(
                name="test",
                port_names={"white", "black", "red", "green"},
                pair_specs={("white", "black"), ("red", "green")},
                archetype_specs=[["white", "red"], ["black", "green"]]
            ),
            k_on=1.0, k_off=0.1,
            seed=42
        )
        self.sampler = self.system.bucket_sampler

    def test_bucket_initialization(self):
        # P0: white, red; P1: black, green
        assert 0 in self.sampler.buckets["white"]
        assert 0 in self.sampler.buckets["red"]
        assert 1 in self.sampler.buckets["black"]
        assert 1 in self.sampler.buckets["green"]
        # blue not in palette
        assert "blue" not in self.sampler.buckets or len(self.sampler.buckets.get("blue", set())) == 0

    def test_aggregate_sites(self):
        assert self.sampler.aggregate_sites["white"] == 1
        assert self.sampler.aggregate_sites["black"] == 1
        assert self.sampler.aggregate_sites["red"] == 1
        assert self.sampler.aggregate_sites["green"] == 1

    def test_binding_propensities(self):
        props = self.sampler.compute_binding_propensities()
        # white-black: P0 white * P1 black = 1*1 = 1
        assert props[("white", "black")] == 1.0
        # red-green: P0 red * P1 green = 1*1 = 1
        assert props[("red", "green")] == 1.0
        # Other channels not in palette

    def test_update_after_bind(self):
        # Simulate binding P0 white with P1 black
        self.sampler.update_after_bind(0, 1, "white", "black")

        # P0 white exhausted, P1 black exhausted
        assert 0 not in self.sampler.buckets["white"]
        assert 1 not in self.sampler.buckets["black"]
        assert self.sampler.aggregate_sites["white"] == 0
        assert self.sampler.aggregate_sites["black"] == 0

    def test_update_after_break(self):
        # First bind
        self.sampler.update_after_bind(0, 1, "white", "black")
        # Then break
        self.sampler.update_after_break(0, 1, "white", "black")

        # Should be back to initial state
        assert 0 in self.sampler.buckets["white"]
        assert 1 in self.sampler.buckets["black"]
        assert self.sampler.aggregate_sites["white"] == 1
        assert self.sampler.aggregate_sites["black"] == 1


class TestTwoStageSampler:
    """Test two-stage Gillespie sampler."""

    def setup_method(self):
        self.system = create_system_from_config(
            n_particles=2,
            config=SystemConfiguration.from_spec(
                name="test",
                port_names={"white", "black", "red", "green"},
                pair_specs={("white", "black"), ("red", "green")},
                archetype_specs=[["white", "red"], ["black", "green"]]
            ),
            k_on=1.0, k_off=0.1,
            seed=42
        )

    def test_step_bind(self):
        sampler = self.system.sampler
        # No bonds, so must bind
        success, event_type, dt, *rest = sampler.step()
        assert success == True
        assert event_type == 'bind'
        assert dt > 0
        assert len(rest) == 4  # part_i, part_j, port_i, port_j

    def test_step_break(self):
        # First create a bond
        self.system.step()  # bind
        assert self.system.n_bonds == 1

        sampler = self.system.sampler
        success, event_type, dt, *_ = sampler.step()
        assert success == True
        # Could be bind or break depending on propensities
        assert event_type in ('bind', 'break')
        assert dt > 0


class TestValidation:
    """Test system validation."""

    def test_valid_system(self):
        system = create_system_from_config(
            n_particles=2,
            config=SystemConfiguration.from_spec(
                name="test",
                port_names={"white", "black", "red", "green"},
                pair_specs={("white", "black"), ("red", "green")},
                archetype_specs=[["white", "red"], ["black", "green"]]
            ),
            k_on=1.0, k_off=0.1
        )
        system.run(max_steps=10, verbose=False)

        errors = validate_system(system)
        assert len(errors) == 0, f"Validation errors: {errors}"

    def test_conservation(self):
        system = create_system_from_config(
            n_particles=2,
            config=SystemConfiguration.from_spec(
                name="test",
                port_names={"white", "black", "red", "green"},
                pair_specs={("white", "black"), ("red", "green")},
                archetype_specs=[["white", "white", "red"], ["black", "black", "green"]]
            ),
            k_on=1.0, k_off=0.1,
            seed=42
        )
        system.run(max_steps=20, verbose=False)

        errors = validate_system(system)
        assert len(errors) == 0, f"Validation errors: {errors}"


class TestSteadyStateAnalyzer:
    """Test steady-state analysis."""

    def test_equilibrium_constants(self):
        system = create_system_from_config(
            n_particles=2,
            config=SystemConfiguration.from_spec(
                name="test",
                port_names={"white", "black", "red", "green"},
                pair_specs={("white", "black"), ("red", "green")},
                archetype_specs=[["white", "red"], ["black", "green"]]
            ),
            k_on=1.0, k_off=0.1
        )
        system.run(max_steps=50, verbose=False)

        analyzer = SteadyStateAnalyzer(system)
        k_eqs = analyzer.estimate_equilibrium_constants()

        # Should have entries for active channels (may be empty if no bonds formed)
        # Just verify it returns a dict
        assert isinstance(k_eqs, dict)

    def test_gel_fraction(self):
        system = create_system_from_config(
            n_particles=2,
            config=SystemConfiguration.from_spec(
                name="test",
                port_names={"white", "black", "red", "green"},
                pair_specs={("white", "black"), ("red", "green")},
                archetype_specs=[["white", "red"], ["black", "green"]]
            ),
            k_on=1.0, k_off=0.1
        )
        system.run(max_steps=20, verbose=False)

        analyzer = SteadyStateAnalyzer(system)
        gel = analyzer.compute_gel_fraction()
        assert 0 <= gel <= 1

    def test_thermodynamic_quantities(self):
        system = create_system_from_config(
            n_particles=2,
            config=SystemConfiguration.from_spec(
                name="test",
                port_names={"white", "black", "red", "green"},
                pair_specs={("white", "black"), ("red", "green")},
                archetype_specs=[["white", "red"], ["black", "green"]]
            ),
            k_on=1.0, k_off=0.1
        )
        system.run(max_steps=20, verbose=False)

        thermo = compute_thermodynamic_quantities(system)
        assert "entropy" in thermo
        assert "free_energy" in thermo
        assert thermo["total_bonds"] == system.n_bonds


class TestEnsemble:
    """Test ensemble runs."""

    def test_ensemble_run(self):
        def system_factory():
            return create_system_from_config(
                n_particles=4,
                config=SystemConfiguration.from_spec(
                    name="test",
                    port_names={"white", "black", "red", "green", "blue", "yellow"},
                    pair_specs={("white", "black"), ("red", "green"), ("blue", "yellow")},
                    archetype_specs=[
                        ["white", "red"],
                        ["green", "blue"],
                        ["black", "green", "yellow"],
                        ["red", "blue"]
                    ]
                ),
                k_on=1.5, k_off=0.3
            )

        from igspa.utils.analysis import run_ensemble, compute_ensemble_statistics

        results = run_ensemble(system_factory, n_runs=5, max_steps=20, verbose=False)
        assert len(results) == 5

        stats = compute_ensemble_statistics(results)
        assert "final_bonds" in stats
        assert "steps" in stats
        assert stats["final_bonds"]["n"] == 5


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_single_particle(self):
        system = create_system_from_config(
            n_particles=1,
            config=SystemConfiguration.from_spec(
                name="test",
                port_names={"white", "black"},
                pair_specs={("white", "black")},
                archetype_specs=[["white", "black"]]
            ),
            k_on=1.0, k_off=0.1
        )

        # Self-binding should be prevented
        for _ in range(10):
            system.step()
        assert system.n_bonds == 0  # No valid partner

    def test_no_valid_pairs(self):
        system = create_system_from_config(
            n_particles=2,
            config=SystemConfiguration.from_spec(
                name="test",
                port_names={"white", "red"},
                pair_specs=set(),  # No valid pairs
                archetype_specs=[["white", "red"], ["white", "red"]]
            ),
            k_on=1.0, k_off=0.1
        )

        for _ in range(10):
            if not system.step():
                break
        assert system.n_bonds == 0  # No binding possible

    def test_all_saturated(self):
        system = create_system_from_config(
            n_particles=2,
            config=SystemConfiguration.from_spec(
                name="test",
                port_names={"white", "black"},
                pair_specs={("white", "black")},
                archetype_specs=[["white"], ["black"]]
            ),
            k_on=100.0, k_off=0.001)  # Strong binding, weak unbinding

        # Should quickly saturate
        for _ in range(10):
            system.step()
        # At most 1 bond possible
        assert system.n_bonds <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
