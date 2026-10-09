"""
Core Particle System: Main simulation orchestrator.

The ParticleSystem class coordinates all components:
- Particles with dynamic binding state
- Information Manifold (continuous availability geometry)
- Bond Graph (explicit topology)
- Gillespie Engine (stochastic simulation)
- Sampling (global bucket optimization)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Tuple
from collections import Counter
import numpy as np
import time

from .colors import Color, ColorPalette, DEFAULT_PALETTE
from .particle import Particle, ParticleBlueprint, create_particles_from_blueprints
from ..geometry.manifold import InformationManifold, create_manifold_from_blueprints
from ..topology.graph import BondGraph, ComplexAnalyzer, create_bond_graph_from_system
from ..algorithms.gillespie import (
    GlobalBucketSampler, TwoStageSampler, CompositionRejectionSampler,
    create_sampler
)


@dataclass
class SimulationConfig:
    """Configuration parameters for the simulation."""
    k_on: float = 1.0
    k_off: float = 0.1
    max_steps: int = 1000
    max_time: float = float('inf')
    seed: Optional[int] = None
    use_composition_rejection: bool = False
    log_interval: int = 100
    track_complexes: bool = True
    track_manifold: bool = False
    callbacks: List[Callable] = field(default_factory=list)


@dataclass
class SimulationState:
    """Complete state snapshot for checkpointing/analysis."""
    step: int
    time: float
    particles: List[Particle]
    bond_graph: BondGraph
    manifold: InformationManifold
    complex_stats: Dict
    sampler_state: Dict
    
    def to_dict(self) -> Dict:
        return {
            'step': self.step,
            'time': self.time,
            'n_bonds': self.bond_graph.n_bonds,
            'complex_stats': self.complex_stats,
        }


class ParticleSystem:
    """
    Main simulation system integrating all NS-IGAA components.
    
    This is the primary user-facing class for running simulations.
    
    Example:
        blueprints = [
            ParticleBlueprint.from_string_list(['white', 'red'], palette),
            ParticleBlueprint.from_string_list(['green', 'blue'], palette),
        ]
        system = ParticleSystem(blueprints, k_on=1.5, k_off=0.3)
        system.run(1000)
        system.print_complexes()
    """
    
    def __init__(
        self,
        blueprints: List[ParticleBlueprint],
        palette: ColorPalette = None,
        config: SimulationConfig = None
    ):
        self.palette = palette or DEFAULT_PALETTE
        self.config = config or SimulationConfig()
        
        # Set random seed if provided
        if self.config.seed is not None:
            np.random.seed(self.config.seed)
            import random
            random.seed(self.config.seed)
        
        # Create particles from blueprints
        self.particles = create_particles_from_blueprints(blueprints, self.palette)
        
        # Create core components
        self.manifold = InformationManifold(
            particles=self.particles,
            palette=self.palette
        )
        
        self.bond_graph = create_bond_graph_from_system(self.particles, self.palette)
        self.complex_analyzer = ComplexAnalyzer(self.bond_graph)
        
        # Create samplers
        self.bucket_sampler, self.sampler = create_sampler(
            particles=self.particles,
            palette=self.palette,
            k_on=self.config.k_on,
            k_off=self.config.k_off,
            bond_graph=self.bond_graph,
            use_cr=self.config.use_composition_rejection
        )
        
        # Simulation state
        self.current_time = 0.0
        self.current_step = 0
        self.running = False
        self.history: List[SimulationState] = []
        
        # Statistics
        self.stats = {
            'bind_events': 0,
            'break_events': 0,
            'total_dt': 0.0,
            'time_per_step': [],
        }
    
    @property
    def n_particles(self) -> int:
        return len(self.particles)
    
    @property
    def n_bonds(self) -> int:
        return self.bond_graph.n_bonds
    
    def step(self) -> bool:
        """
        Execute one Gillespie SSA step.
        
        Returns True if step executed, False if no events possible (equilibrium).
        """
        if isinstance(self.sampler, TwoStageSampler):
            result = self.sampler.step()
            
            if not result[0]:
                return False
            
            event_type = result[1]
            dt = result[2]
            
            self.current_time += dt
            self.current_step += 1
            self.stats['total_dt'] += dt
            self.stats['time_per_step'].append(dt)
            
            if event_type == 'bind':
                part_i, part_j, color_i, color_j = result[3], result[4], result[5], result[6]
                self.sampler.execute_bind(part_i, part_j, color_i, color_j)
                self.stats['bind_events'] += 1
                
                # Run callbacks
                for cb in self.config.callbacks:
                    cb('bind', self.current_step, self.current_time, part_i, part_j, color_i, color_j)
                    
            elif event_type == 'break':
                bond = result[3]
                self.sampler.execute_break(bond)
                self.stats['break_events'] += 1
                
                for cb in self.config.callbacks:
                    cb('break', self.current_step, self.current_time, bond)
            
            # Invalidate CR bins if using composition-rejection
            if isinstance(self.sampler, CompositionRejectionSampler):
                self.sampler.invalidate_bins()
            
            # Invalidate complex cache
            self.complex_analyzer.invalidate_cache()
            
            return True
            
        else:
            # CompositionRejectionSampler path
            event_type, event_data = self.sampler.sample_event_cr()
            
            if event_type is None:
                return False
            
            # Sample time step
            total_bind = self.bucket_sampler.compute_total_binding_propensity()
            total_break = self.bucket_sampler.k_off * self.bond_graph.n_bonds
            total_prop = total_bind + total_break
            dt = np.random.exponential(1.0 / total_prop)
            
            self.current_time += dt
            self.current_step += 1
            
            if event_type == 'bind':
                part_i, part_j, color_i, color_j = event_data
                self.sampler.execute_bind(part_i, part_j, color_i, color_j)
                self.stats['bind_events'] += 1
            elif event_type == 'break':
                bond = event_data
                self.sampler.execute_break(bond)
                self.stats['break_events'] += 1
            
            self.sampler.invalidate_bins()
            self.complex_analyzer.invalidate_cache()
            
            return True
    
    def run(self, max_steps: int = None, max_time: float = None, 
            verbose: bool = True) -> Dict:
        """
        Run simulation until max_steps or max_time reached.
        
        Returns final statistics dictionary.
        """
        max_steps = max_steps or self.config.max_steps
        max_time = max_time or self.config.max_time
        
        self.running = True
        
        if verbose:
            print(f"Starting NS-IGAA Simulation")
            print(f"  Particles: {self.n_particles}")
            print(f"  Colors: {len(self.palette)}")
            print(f"  k_on: {self.config.k_on}, k_off: {self.config.k_off}")
            print(f"  Max steps: {max_steps}, Max time: {max_time}")
            print("-" * 50)
        
        start_time = time.time()
        
        for step in range(1, max_steps + 1):
            if not self.step():
                if verbose:
                    print(f"\nSystem reached equilibrium at step {self.current_step}")
                break
            
            if self.current_time >= max_time:
                if verbose:
                    print(f"\nReached max time {max_time} at step {self.current_step}")
                break
            
            # Logging
            if verbose and step % self.config.log_interval == 0:
                elapsed = time.time() - start_time
                print(f"Step {self.current_step}: t={self.current_time:.3f}, "
                      f"bonds={self.n_bonds}, "
                      f"binds={self.stats['bind_events']}, "
                      f"breaks={self.stats['break_events']}, "
                      f"{elapsed:.2f}s wall-time")
        
        self.running = False
        wall_time = time.time() - start_time
        
        # Final statistics
        final_stats = {
            'steps': self.current_step,
            'final_time': self.current_time,
            'final_bonds': self.n_bonds,
            'bind_events': self.stats['bind_events'],
            'break_events': self.stats['break_events'],
            'wall_time': wall_time,
            'steps_per_second': self.current_step / wall_time if wall_time > 0 else 0,
        }
        
        if self.config.track_complexes:
            complexes = self.complex_analyzer.get_complexes()
            final_stats['complexes'] = self.complex_analyzer.get_complex_statistics()
        
        if verbose:
            print("-" * 50)
            print(f"Simulation complete in {wall_time:.2f}s")
            print(f"Steps: {self.current_step}, Time: {self.current_time:.3f}")
            print(f"Bonds: {self.n_bonds}")
            print(f"Bind events: {self.stats['bind_events']}")
            print(f"Break events: {self.stats['break_events']}")
            if 'complexes' in final_stats:
                cs = final_stats['complexes']
                print(f"Complexes: {cs['n_complexes']}, Largest: {cs['largest_size']}")
        
        return final_stats
    
    def get_complexes(self) -> List[List[int]]:
        """Get current complexes as lists of particle indices."""
        return [list(c) for c in self.complex_analyzer.get_complexes()]
    
    def print_complexes(self, labels: bool = True):
        """Print current complex structure."""
        complexes = self.complex_analyzer.get_complexes()
        print(f"\n=== COMPLEXES (Time: {self.current_time:.3f}s, Step: {self.current_step}) ===")
        
        for idx, comp in enumerate(complexes):
            if labels:
                members = [self.particles[i].label for i in comp]
                print(f"Complex {idx + 1} (Size {len(members)}): {' -- '.join(members)}")
            else:
                print(f"Complex {idx + 1} (Size {len(comp)}): {sorted(comp)}")
    
    def get_bond_list(self) -> List[Tuple[int, int, str, str]]:
        """Get all bonds as (particle_i, particle_j, color_i, color_j)."""
        bonds = []
        for bond in self.bond_graph.get_bonds():
            bonds.append((
                bond.particle_i, 
                bond.particle_j, 
                bond.color_i.name.lower(), 
                bond.color_j.name.lower()
            ))
        return bonds
    
    def get_availability_matrix(self) -> Dict[str, List[int]]:
        """Get availability matrix: particle -> {color: available_count}."""
        result = {}
        for particle in self.particles:
            result[particle.label] = {
                c.name.lower(): particle.available_sites.get(c, 0)
                for c in self.palette.colors
            }
        return result
    
    def get_state_snapshot(self) -> SimulationState:
        """Get complete state snapshot for checkpointing."""
        complex_stats = self.complex_analyzer.get_complex_statistics() if self.config.track_complexes else {}
        
        return SimulationState(
            step=self.current_step,
            time=self.current_time,
            particles=self.particles.copy(),
            bond_graph=self.bond_graph,
            manifold=self.manifold,
            complex_stats=complex_stats,
            sampler_state={
                'buckets': {c.name: len(b) for c, b in self.bucket_sampler.buckets.items()},
                'aggregates': {c.name: a for c, a in self.bucket_sampler.aggregate_sites.items()},
            }
        )
    
    def __str__(self) -> str:
        return (f"ParticleSystem(n_particles={self.n_particles}, "
                f"n_bonds={self.n_bonds}, time={self.current_time:.3f}, "
                f"step={self.current_step})")


def create_system_from_strings(
    particle_strings: List[List[str]],
    k_on: float = 1.0,
    k_off: float = 0.1,
    palette: ColorPalette = None,
    **config_kwargs
) -> ParticleSystem:
    """
    Convenience factory to create system from string color lists.
    
    Example:
        system = create_system_from_strings([
            ['white', 'red'],
            ['green', 'blue'],
            ['black', 'green', 'yellow'],
        ], k_on=1.5, k_off=0.3)
    """
    if palette is None:
        palette = DEFAULT_PALETTE
    
    blueprints = [
        ParticleBlueprint.from_string_list(colors, palette)
        for colors in particle_strings
    ]
    
    config = SimulationConfig(k_on=k_on, k_off=k_off, **config_kwargs)
    return ParticleSystem(blueprints, palette, config)