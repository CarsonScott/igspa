"""
Tests for IGSPA Core Components
"""

import pytest
import numpy as np
from collections import Counter

from igspa.core.colors import Color, ColorPalette, COMPLEMENT_MAP, DEFAULT_PALETTE
from igspa.core.particle import Particle, ParticleBlueprint, create_particles_from_blueprints
from igspa.core.system import ParticleSystem, SimulationConfig, create_system_from_strings
from igspa.geometry.manifold import InformationManifold, FisherMetric, create_manifold_from_blueprints
from igspa.topology.graph import BondGraph, Bond, ComplexAnalyzer, create_bond_graph_from_system
from igspa.algorithms.gillespie import (
    GlobalBucketSampler, TwoStageSampler, CompositionRejectionSampler, create_sampler
)
from igspa.utils.analysis import validate_system, SteadyStateAnalyzer, compute_thermodynamic_quantities


class TestColors:
    """Test color system and complementarity."""
    
    def test_standard_palette(self):
        palette = DEFAULT_PALETTE
        assert len(palette) == 6
        assert Color.WHITE in palette
        assert Color.BLACK in palette
    
    def test_complementarity(self):
        assert COMPLEMENT_MAP[Color.WHITE] == Color.BLACK
        assert COMPLEMENT_MAP[Color.BLACK] == Color.WHITE
        assert COMPLEMENT_MAP[Color.RED] == Color.GREEN
        assert COMPLEMENT_MAP[Color.GREEN] == Color.RED
        assert COMPLEMENT_MAP[Color.BLUE] == Color.YELLOW
        assert COMPLEMENT_MAP[Color.YELLOW] == Color.BLUE
    
    def test_symmetric_complementarity(self):
        for c, comp in COMPLEMENT_MAP.items():
            assert COMPLEMENT_MAP[comp] == c
    
    def test_custom_palette(self):
        pairs = {'A': 'T', 'C': 'G'}
        palette = ColorPalette.from_strings(pairs)
        assert len(palette) == 4
        assert palette.complement_map[Color.WHITE] == Color.BLACK  # A->T maps to WHITE->BLACK


class TestParticleBlueprint:
    """Test particle blueprint creation and validation."""
    
    def test_from_color_list(self):
        bp = ParticleBlueprint.from_color_list([Color.WHITE, Color.WHITE, Color.RED])
        assert bp.site_counts[Color.WHITE] == 2
        assert bp.site_counts[Color.RED] == 1
        assert bp.total_sites == 3
    
    def test_from_string_list(self):
        bp = ParticleBlueprint.from_string_list(['white', 'red', 'white'], DEFAULT_PALETTE)
        assert bp.site_counts[Color.WHITE] == 2
        assert bp.site_counts[Color.RED] == 1
    
    def test_positive_counts(self):
        with pytest.raises(ValueError):
            ParticleBlueprint(site_counts={Color.WHITE: 0})
    
    def test_str_representation(self):
        bp = ParticleBlueprint.from_string_list(['white', 'red'], DEFAULT_PALETTE, label='Test')
        assert 'white:1' in str(bp)
        assert 'red:1' in str(bp)
        assert 'Test' in str(bp)


class TestParticle:
    """Test dynamic particle state management."""
    
    def setup_method(self):
        self.bp = ParticleBlueprint.from_string_list(['white', 'red', 'white'], DEFAULT_PALETTE)
        self.particle = Particle(blueprint=self.bp, particle_id=0)
    
    def test_initial_state(self):
        assert self.particle.available_sites[Color.WHITE] == 2
        assert self.particle.available_sites[Color.RED] == 1
        assert self.particle.capacities[Color.WHITE] == 2
        assert self.particle.capacities[Color.RED] == 1
    
    def test_bind_site(self):
        assert self.particle.bind_site(Color.WHITE) == True
        assert self.particle.available_sites[Color.WHITE] == 1
        assert self.particle.bind_site(Color.WHITE) == True
        assert self.particle.available_sites[Color.WHITE] == 0
        assert self.particle.bind_site(Color.WHITE) == False  # Exhausted
    
    def test_release_site(self):
        self.particle.bind_site(Color.WHITE)
        self.particle.bind_site(Color.WHITE)
        assert self.particle.release_site(Color.WHITE) == True
        assert self.particle.available_sites[Color.WHITE] == 1
        assert self.particle.release_site(Color.WHITE) == True
        assert self.particle.available_sites[Color.WHITE] == 2
        assert self.particle.release_site(Color.WHITE) == False  # At capacity
    
    def test_fractions(self):
        frac = self.particle.fractions
        assert frac[Color.WHITE] == 1.0
        assert frac[Color.RED] == 1.0
        
        self.particle.bind_site(Color.WHITE)
        frac = self.particle.fractions
        assert frac[Color.WHITE] == 0.5
    
    def test_is_saturated_empty(self):
        assert self.particle.is_empty() == True
        assert self.particle.is_saturated() == False
        
        self.particle.bind_site(Color.WHITE)
        self.particle.bind_site(Color.WHITE)
        self.particle.bind_site(Color.RED)
        
        assert self.particle.is_saturated() == True
        assert self.particle.is_empty() == False


class TestParticleSystem:
    """Test complete particle system."""
    
    def setup_method(self):
        self.system = create_system_from_strings([
            ['white', 'red'],
            ['green', 'blue'],
            ['black', 'green', 'yellow'],
            ['red', 'blue'],
        ], k_on=1.5, k_off=0.3)
    
    def test_initialization(self):
        assert self.system.n_particles == 4
        assert len(self.system.palette) == 6
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
            assert len(b) == 4  # i, j, color_i, color_j
    
    def test_availability_matrix(self):
        avail = self.system.get_availability_matrix()
        assert len(avail) == 4
        for particle_data in avail.values():
            assert all(isinstance(v, int) for v in particle_data.values())
    
    def test_state_snapshot(self):
        for _ in range(5):
            self.system.step()
        snapshot = self.system.get_state_snapshot()
        assert snapshot.step == self.system.current_step
        assert snapshot.time == self.system.current_time
        assert len(snapshot.particles) == 4


class TestManifold:
    """Test information manifold geometry."""
    
    def setup_method(self):
        self.system = create_system_from_strings([
            ['white', 'red'],
            ['green', 'blue'],
        ], k_on=1.0, k_off=0.1)
    
    def test_manifold_creation(self):
        manifold = self.system.manifold
        assert manifold.n_particles == 2
        assert manifold.n_colors == 6
    
    def test_global_theta(self):
        theta = self.system.manifold.get_global_theta()
        assert theta.shape == (2 * 6,)  # 2 particles, 6 colors
        assert np.all(theta >= 0)
        assert np.all(theta <= 1)
    
    def test_channel_aggregates(self):
        agg = self.system.manifold.compute_channel_aggregates()
        assert len(agg) == 6
        # Initially all sites available
        assert agg[Color.WHITE] == 1  # Only P0 has white
        assert agg[Color.RED] == 1    # Only P0 has red
        assert agg[Color.GREEN] == 1  # Only P1 has green
        assert agg[Color.BLUE] == 1   # Only P1 has blue
    
    def test_binding_propensities(self):
        props = self.system.manifold.compute_binding_propensities(1.0)
        # white-black channel: P0 has white, P2 has black
        # But P2 doesn't exist in this system
        # So white-black should be 0
        assert props.get((Color.WHITE, Color.BLACK), 0) == 0.0
    
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
        self.system = create_system_from_strings([
            ['white', 'red'],
            ['green', 'blue'],
        ], k_on=1.0, k_off=0.1)
        self.graph = self.system.bond_graph
    
    def test_add_remove_bond(self):
        key = self.graph.add_bond(0, 1, Color.WHITE, Color.BLACK)
        assert self.graph.n_bonds == 1
        assert key == 0
        
        bond = self.graph.remove_bond(0, 1, key)
        assert bond is not None
        assert bond.particle_i == 0
        assert bond.particle_j == 1
        assert self.graph.n_bonds == 0
    
    def test_multiple_bonds(self):
        self.graph.add_bond(0, 1, Color.WHITE, Color.BLACK)
        self.graph.add_bond(0, 1, Color.RED, Color.GREEN)
        assert self.graph.n_bonds == 2
        
        bonds = self.graph.get_bonds()
        assert len(bonds) == 2
    
    def test_color_degree(self):
        self.graph.add_bond(0, 1, Color.WHITE, Color.BLACK)
        assert self.graph.get_color_degree(0, Color.WHITE) == 1
        assert self.graph.get_color_degree(1, Color.BLACK) == 1
        assert self.graph.get_color_degree(0, Color.RED) == 0
    
    def test_adjacency_matrix(self):
        self.graph.add_bond(0, 1, Color.WHITE, Color.BLACK)
        A = self.graph.get_adjacency_matrix(Color.WHITE)
        assert A[0, 1] == 1
        assert A[1, 0] == 0  # Directional: color_i is at particle_i
    
    def test_complex_analyzer(self):
        self.graph.add_bond(0, 1, Color.WHITE, Color.BLACK)
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
        self.system = create_system_from_strings([
            ['white', 'red'],
            ['black', 'green'],
        ], k_on=1.0, k_off=0.1)
        self.sampler = self.system.bucket_sampler
    
    def test_bucket_initialization(self):
        # P0: white, red; P1: black, green
        assert 0 in self.sampler.buckets[Color.WHITE]
        assert 0 in self.sampler.buckets[Color.RED]
        assert 1 in self.sampler.buckets[Color.BLACK]
        assert 1 in self.sampler.buckets[Color.GREEN]
        assert len(self.sampler.buckets[Color.BLUE]) == 0
    
    def test_aggregate_sites(self):
        assert self.sampler.aggregate_sites[Color.WHITE] == 1
        assert self.sampler.aggregate_sites[Color.BLACK] == 1
        assert self.sampler.aggregate_sites[Color.RED] == 1
        assert self.sampler.aggregate_sites[Color.GREEN] == 1
    
    def test_binding_propensities(self):
        props = self.sampler.compute_binding_propensities()
        # white-black: P0 white * P1 black = 1*1 = 1
        assert props[(Color.WHITE, Color.BLACK)] == 1.0
        # red-green: P0 red * P1 green = 1*1 = 1
        assert props[(Color.RED, Color.GREEN)] == 1.0
        # Other channels: 0
        assert props[(Color.BLUE, Color.YELLOW)] == 0.0
    
    def test_update_after_bind(self):
        # Simulate binding P0 white with P1 black
        self.sampler.update_after_bind(0, 1, Color.WHITE, Color.BLACK)
        
        # P0 white exhausted, P1 black exhausted
        assert 0 not in self.sampler.buckets[Color.WHITE]
        assert 1 not in self.sampler.buckets[Color.BLACK]
        assert self.sampler.aggregate_sites[Color.WHITE] == 0
        assert self.sampler.aggregate_sites[Color.BLACK] == 0
    
    def test_update_after_break(self):
        # First bind
        self.sampler.update_after_bind(0, 1, Color.WHITE, Color.BLACK)
        # Then break
        self.sampler.update_after_break(0, 1, Color.WHITE, Color.BLACK)
        
        # Should be back to initial state
        assert 0 in self.sampler.buckets[Color.WHITE]
        assert 1 in self.sampler.buckets[Color.BLACK]
        assert self.sampler.aggregate_sites[Color.WHITE] == 1
        assert self.sampler.aggregate_sites[Color.BLACK] == 1


class TestTwoStageSampler:
    """Test two-stage Gillespie sampler."""
    
    def setup_method(self):
        self.system = create_system_from_strings([
            ['white', 'red'],
            ['black', 'green'],
        ], k_on=1.0, k_off=0.1)
    
    def test_step_bind(self):
        sampler = self.system.sampler
        # No bonds, so must bind
        success, event_type, dt, *rest = sampler.step()
        assert success == True
        assert event_type == 'bind'
        assert dt > 0
        assert len(rest) == 4  # part_i, part_j, color_i, color_j
    
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
        system = create_system_from_strings([
            ['white', 'red'],
            ['black', 'green'],
        ], k_on=1.0, k_off=0.1)
        system.run(max_steps=10, verbose=False)
        
        errors = validate_system(system)
        assert len(errors) == 0, f"Validation errors: {errors}"
    
    def test_conservation(self):
        system = create_system_from_strings([
            ['white', 'white', 'red'],
            ['black', 'black', 'green'],
        ], k_on=1.0, k_off=0.1)
        system.run(max_steps=20, verbose=False)
        
        errors = validate_system(system)
        assert len(errors) == 0, f"Validation errors: {errors}"


class TestSteadyStateAnalyzer:
    """Test steady-state analysis."""
    
    def test_equilibrium_constants(self):
        system = create_system_from_strings([
            ['white', 'red'],
            ['black', 'green'],
        ], k_on=1.0, k_off=0.1)
        system.run(max_steps=50, verbose=False)
        
        analyzer = SteadyStateAnalyzer(system)
        k_eqs = analyzer.estimate_equilibrium_constants()
        
        # Should have entries for active channels
        assert len(k_eqs) > 0
        for val in k_eqs.values():
            assert isinstance(val, float)
    
    def test_gel_fraction(self):
        system = create_system_from_strings([
            ['white', 'red'],
            ['black', 'green'],
        ], k_on=1.0, k_off=0.1)
        system.run(max_steps=20, verbose=False)
        
        analyzer = SteadyStateAnalyzer(system)
        gel = analyzer.compute_gel_fraction()
        assert 0 <= gel <= 1
    
    def test_thermodynamic_quantities(self):
        system = create_system_from_strings([
            ['white', 'red'],
            ['black', 'green'],
        ], k_on=1.0, k_off=0.1)
        system.run(max_steps=20, verbose=False)
        
        thermo = compute_thermodynamic_quantities(system)
        assert 'entropy' in thermo
        assert 'free_energy' in thermo
        assert thermo['total_bonds'] == system.n_bonds


class TestEnsemble:
    """Test ensemble runs."""
    
    def test_ensemble_run(self):
        from igspa.utils.analysis import create_standard_test_system, run_ensemble, compute_ensemble_statistics
        
        results = run_ensemble(create_standard_test_system, n_runs=5, max_steps=20, verbose=False)
        assert len(results) == 5
        
        stats = compute_ensemble_statistics(results)
        assert 'final_bonds' in stats
        assert 'steps' in stats
        assert stats['final_bonds']['n'] == 5


class TestEdgeCases:
    """Test edge cases and error conditions."""
    
    def test_single_particle(self):
        system = create_system_from_strings([
            ['white', 'black'],  # Can self-bind
        ], k_on=1.0, k_off=0.1)
        
        # Self-binding should be prevented
        for _ in range(10):
            system.step()
        assert system.n_bonds == 0  # No valid partner
    
    def test_no_complementary_colors(self):
        system = create_system_from_strings([
            ['white', 'red'],
            ['white', 'red'],  # Same colors, no complements
        ], k_on=1.0, k_off=0.1)
        
        for _ in range(10):
            if not system.step():
                break
        assert system.n_bonds == 0  # No binding possible
    
    def test_all_saturated(self):
        system = create_system_from_strings([
            ['white'],
            ['black'],
        ], k_on=100.0, k_off=0.001)  # Strong binding, weak unbinding
        
        # Should quickly saturate
        for _ in range(10):
            system.step()
        # At most 1 bond possible
        assert system.n_bonds <= 1


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
