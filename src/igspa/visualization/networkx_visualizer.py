"""
Real-time NetworkX/Matplotlib Visualizer for ParticleSystems.

Provides animated visualization of particle assembly dynamics using:
- matplotlib.animation.FuncAnimation for real-time updates
- NetworkX for graph representation
- ForceAtlas2-inspired force-directed layout with periodic updates
- Persistent node positions between simulation steps
- Edge-only updates for bond formation/breakage events
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Callable, Any, Set
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import networkx as nx
import warnings

from ..core.system import ParticleSystem
from ..core.particle import Particle
from ..topology.graph import BondGraph, Bond
from ..core.ports import PortPalette


@dataclass
class VisualizationConfig:
    """Configuration for the network visualizer."""
    figsize: Tuple[float, float] = (12, 12)
    node_size_range: Tuple[float, float] = (5,5)
    node_size_by: str = "degree"
    edge_width_range: Tuple[float, float] = (1.0,1.0)
    edge_width_by: str = "port"
    layout_update_interval: int = 1
    forceatlas2_iterations: int = 1
    forceatlas2_scaling: float = 1
    forceatlas2_gravity: float = -1.0
    forceatlas2_spring_strength: float = 1
    forceatlas2_dissipation: float = 0.1
    forceatlas2_prevent_overlap: bool = True
    forceatlas2_overlap_threshold: float = 1
    show_labels: bool = False
    label_fontsize: int = 8
    show_legend: bool = True
    background_color: str = "white"
    particle_color_map: Optional[Dict[str, str]] = None
    port_color_map: Optional[Dict[str, str]] = None
    animation_interval: int = 1
    repeat: bool = False
    save_path: Optional[str] = None
    save_fps: int = 10
    dpi: int = 100


@dataclass
class ForceAtlas2State:
    """Internal state for ForceAtlas2 layout computation."""
    positions: np.ndarray
    velocities: np.ndarray
    masses: np.ndarray
    forces: np.ndarray
    nodes: List[int]


class ForceAtlas2Layout:
    """
    ForceAtlas2-inspired force-directed layout algorithm.
    
    Simplified implementation based on the ForceAtlas2 algorithm (Jacomy et al., 2014).
    Features:
    - Attractive forces along edges (spring-like)
    - Repulsive forces between all nodes (Coulomb-like)
    - Gravity force pulling toward center
    - Velocity damping for stability
    - Overlap prevention (optional)
    """
    
    def __init__(
        self,
        n_nodes: int,
        scaling: float = 1.0,
        gravity: float = 1.0,
        spring_strength: float = 1,
        dissipation: float = 1.0,
        prevent_overlap: bool = True,
        overlap_threshold: float = 1.0,
        seed: Optional[int] = None,
    ):
        self.n_nodes = n_nodes
        self.scaling = scaling
        self.gravity = gravity
        self.spring_strength = spring_strength
        self.dissipation = dissipation
        self.prevent_overlap = prevent_overlap
        self.overlap_threshold = overlap_threshold
        
        if seed is not None:
            np.random.seed(seed)
        
        self.state = ForceAtlas2State(
            positions=np.random.randn(n_nodes, 2) * 1,
            velocities=np.zeros((n_nodes, 2)),
            masses=np.ones(n_nodes),
            forces=np.zeros((n_nodes, 2)),
            nodes=list(range(n_nodes)),
        )
    
    def set_positions(self, positions: np.ndarray):
        """Set node positions directly."""
        self.state.positions = positions.copy()
    
    def set_masses(self, masses: np.ndarray):
        """Set node masses (e.g., based on degree)."""
        self.state.masses = masses.copy()
    
    def compute_step(
        self,
        edges: List[Tuple[int, int]],
        edge_weights: Optional[np.ndarray] = None,
        fixed_nodes: Optional[Set[int]] = None,
    ) -> np.ndarray:
        """
        Compute one ForceAtlas2 iteration.
        
        Args:
            edges: List of (u, v) edge tuples
            edge_weights: Optional edge weights for spring strength
            fixed_nodes: Set of node indices to keep fixed
            
        Returns:
            Updated positions array
        """
        n = self.n_nodes
        pos = self.state.positions
        vel = self.state.velocities
        mass = self.state.masses
        forces = self.state.forces
        
        forces.fill(0.0)
        
        if edge_weights is None:
            edge_weights = np.ones(len(edges))
        
        for (u, v), w in zip(edges, edge_weights):
            if u == v:
                continue
            delta = pos[v] - pos[u]
            dist = np.linalg.norm(delta)
            angle = np.atan2(delta[1], delta[0])
            vec = np.array([np.cos(angle), np.sin(angle)])
            rad = 10
            force_vec = delta / 2
            forces[u] += force_vec
            forces[v] -= force_vec
        
        for i in range(n):
            for j in range(i + 1, n):
                delta = pos[j] - pos[i]
                angle = np.atan2(delta[1], delta[0])
                vec = np.array([np.cos(angle), np.sin(angle)])
                dist = np.linalg.norm(delta)
                force_vec = vec * 1 / (dist ** 2)
                forces[i] -= force_vec
                forces[j] += force_vec
        
        # center = np.mean(pos, axis=0)
        # for i in range(n):
        #     # if fixed_nodes and i in fixed_nodes:
        #     #     continue
        #     delta = center - pos[i]
        #     angle = np.atan2(delta[1], delta[0])
        #     vec = np.array(np.cos(angle), np.sin(angle))
        #     dist = np.linalg.norm(delta)
        #     force_mag = 100/(dist**1)
        #     force_vec = vec * force_mag
        #     forces[i] -= force_vec
        
        # vel *= self.dissipation
        vel += forces
        pos += vel * self.scaling
        
        self.state.positions = pos
        self.state.velocities = vel
        self.state.forces = forces
        
        return pos
    
    def run(
        self,
        edges: List[Tuple[int, int]],
        iterations: int = 50,
        edge_weights: Optional[np.ndarray] = None,
        fixed_nodes: Optional[Set[int]] = None,
        callback: Optional[Callable[[int, np.ndarray], None]] = None,
    ) -> np.ndarray:
        """Run ForceAtlas2 for multiple iterations."""
        for i in range(iterations):
            pos = self.compute_step(edges, edge_weights, fixed_nodes)
            if callback:
                callback(i, pos.copy())
        return self.state.positions.copy()
    
    def get_positions(self) -> np.ndarray:
        """Get current node positions."""
        return self.state.positions.copy()


class ParticleSystemVisualizer:
    """
    Real-time visualizer for ParticleSystem using NetworkX and Matplotlib.
    
    Features:
    - Step-wise simulation updates inside FuncAnimation
    - ForceAtlas2 layout updated periodically
    - Persistent node positions between steps
    - Edge-only updates for bond events
    - Color-coded particles by type/archetype
    - Port-specific edge colors
    """
    
    def __init__(
        self,
        system: ParticleSystem,
        config: Optional[VisualizationConfig] = None,
    ):
        self.system = system
        self.config = config or VisualizationConfig()
        
        self.n_particles = system.n_particles
        self.palette = system.palette
        
        self._setup_colors()
        self._setup_graph()
        self._setup_forceatlas2()
        self._setup_figure()
        
        self.step_count = 0
        self.layout_step_count = 0
        self.current_edges: Set[Tuple[int, int, str, str]] = set()
        self.animation: Optional[animation.FuncAnimation] = None
        self._running = False
        
        self._update_node_properties()
        self._initialize_positions()
    
    def _setup_colors(self):
        """Initialize color maps for particles and ports."""
        if self.config.particle_color_map is None:
            n_types = len(set(p.archetype for p in self.system.particles))
            cmap = plt.cm.tab20
            self.particle_colors = {
                arch: cmap(i % 20) for i, arch in enumerate(
                    sorted(set(p.archetype for p in self.system.particles), key=lambda a: a.label or '')
                )
            }
        else:
            self.particle_colors = self.config.particle_color_map
        
        if self.config.port_color_map is None:
            port_cmap = plt.cm.Set3
            self.port_colors = {
                port: port_cmap(i % 12) for i, port in enumerate(self.palette.ports)
            }
        else:
            self.port_colors = self.config.port_color_map
    
    def _setup_graph(self):
        """Initialize NetworkX graph representation."""
        self.nx_graph = nx.MultiGraph()
        self.nx_graph.add_nodes_from(range(self.n_particles))
        
        self.node_positions = np.random.rand(self.n_particles, 2) * 1
        
        self.edge_key_counter = 0
        self.edge_keys: Dict[Tuple[int, int, str, str], int] = {}
    
    def _setup_forceatlas2(self):
        """Initialize ForceAtlas2 layout engine."""
        masses = np.ones(self.n_particles)
        for i, particle in enumerate(self.system.particles):
            masses[i] = 1.0 + particle.archetype.total_sites * 0.5
        
        self.forceatlas2 = ForceAtlas2Layout(
            n_nodes=self.n_particles,
            scaling=self.config.forceatlas2_scaling,
            gravity=self.config.forceatlas2_gravity,
            spring_strength=self.config.forceatlas2_spring_strength,
            dissipation=self.config.forceatlas2_dissipation,
            prevent_overlap=self.config.forceatlas2_prevent_overlap,
            overlap_threshold=self.config.forceatlas2_overlap_threshold,
        )
        self.forceatlas2.set_masses(masses)
    
    def _setup_figure(self):
        """Initialize matplotlib figure and axes."""
        self.fig = plt.figure(figsize=self.config.figsize)
        self.fig.patch.set_facecolor(self.config.background_color)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor(self.config.background_color)
        self.ax.set_aspect('equal')
        self.ax.axis('off')
        
        self.node_collection = None
        self.edge_collections: List = []
        self.label_texts: List = []
        
        self.title_text = self.ax.text(
            0.5, 1.02, "", transform=self.ax.transAxes,
            ha='center', va='bottom', fontsize=12, fontweight='bold'
        )
        
        self.stats_text = self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes,
            ha='left', va='top', fontsize=9, fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        )
    
    def _initialize_positions(self):
        """Initialize layout with ForceAtlas2."""
        edges = []
        edge_weights = []
        for bond in self.system.bond_graph.get_bonds():
            edges.append((bond.particle_i, bond.particle_j))
            edge_weights.append(1.0)
        
        if edges:
            self.forceatlas2.run(
                edges=edges,
                edge_weights=np.array(edge_weights),
                iterations=self.config.forceatlas2_iterations,
            )
        self.node_positions = self.forceatlas2.get_positions()
        self._update_node_properties()
    
    def _update_node_properties(self):
        """Compute node colors and sizes based on current state."""
        self.node_colors = []
        self.node_sizes = []
        self.node_labels = {}
        
        degrees = dict(self.nx_graph.degree())
        max_degree = max(degrees.values()) if degrees else 1
        
        for i, particle in enumerate(self.system.particles):
            archetype = particle.archetype
            color = self.particle_colors.get(archetype, 'gray')
            self.node_colors.append(color)
            
            # if self.config.node_size_by == "degree":
            #     degree = degrees.get(i, 0)
            #     size = self.config.node_size_range[0] + \
            #            (self.config.node_size_range[1] - self.config.node_size_range[0]) * \
            #            (degree / max_degree if max_degree > 0 else 0)
            # elif self.config.node_size_by == "sites":
            #     total_sites = particle.archetype.total_sites
            #     max_sites = max(p.archetype.total_sites for p in self.system.particles)
            #     size = 1
            # else:
            size = 5
            
            self.node_sizes.append(size)
            
            if self.config.show_labels:
                self.node_labels[i] = particle.label
    
    def _sync_graph_with_system(self) -> Tuple[Set, Set]:
        """
        Synchronize NetworkX graph with ParticleSystem bond graph.
        
        Returns:
            (added_edges, removed_edges) as sets of (u, v, port_i, port_j)
        """
        system_bonds = set()
        for bond in self.system.bond_graph.get_bonds():
            system_bonds.add((bond.particle_i, bond.particle_j, bond.port_i, bond.port_j))
        
        added = system_bonds - self.current_edges
        removed = self.current_edges - system_bonds
        
        for u, v, port_i, port_j in added:
            edge_key = self.edge_key_counter
            self.edge_key_counter += 1
            self.nx_graph.add_edge(u, v, key=edge_key, port_i=port_i, port_j=port_j)
            self.edge_keys[(u, v, port_i, port_j)] = edge_key
        
        for u, v, port_i, port_j in removed:
            key = self.edge_keys.pop((u, v, port_i, port_j), None)
            if key is not None and self.nx_graph.has_edge(u, v, key=key):
                self.nx_graph.remove_edge(u, v, key=key)
        
        self.current_edges = system_bonds.copy()
        
        return added, removed
    
    def _update_forceatlas2(self):
        """Update layout using ForceAtlas2."""
        edges = []
        edge_weights = []
        for u, v, port_i, port_j in self.current_edges:
            edges.append((u, v))
            edge_weights.append(1.0)
        
        if edges:
            self.forceatlas2.run(
                edges=edges,
                edge_weights=np.array(edge_weights),
                iterations=self.config.forceatlas2_iterations,
            )
            self.node_positions = self.forceatlas2.get_positions()
    
    def _draw_nodes(self):
        """Draw or update node collection."""
        if self.node_collection is not None:
            self.node_collection.remove()
        
        self.node_collection = self.ax.scatter(
            self.node_positions[:, 0],
            self.node_positions[:, 1],
            s=self.node_sizes,
            c=self.node_colors,
            alpha=0.8,
            edgecolors='black',
            linewidths=0.5,
            zorder=3,
        )
        
        if self.config.show_labels:
            for text in self.label_texts:
                text.remove()
            self.label_texts = []
            for i, label in self.node_labels.items():
                x, y = self.node_positions[i]
                text = self.ax.text(
                    x, y, label,
                    ha='center', va='center',
                    fontsize=self.config.label_fontsize,
                    fontweight='bold',
                    zorder=4,
                )
                self.label_texts.append(text)
    
    def _draw_edges(self):
        """Draw or update edge collections."""
        for coll in self.edge_collections:
            coll.remove()
        self.edge_collections = []
        
        
        for port in self.palette.ports:
            port_edges = [(u, v) for u, v, pi, pj in self.current_edges if pi == port or pj == port]
            if not port_edges:
                continue
            
            xs, ys = [], []
            for u, v in port_edges:
                x1, y1 = self.node_positions[u]
                x2, y2 = self.node_positions[v]
                xs.extend([x1, x2, None])
                ys.extend([y1, y2, None])
            
            color = self.port_colors.get(port, 'gray')
            line, = self.ax.plot(
                xs, ys,
                color=color,
                linewidth=self.config.edge_width_range[0],
                alpha=0.6,
                zorder=2,
            )
            self.edge_collections.append(line)
    
    def _update_stats_text(self):
        """Update statistics display."""
        complexes = self.system.complex_analyzer.get_complexes()
        n_complexes = len(complexes)
        largest = max(len(c) for c in complexes) if complexes else 0
        
        stats = (
            f"Step: {self.system.current_step}\n"
            f"Time: {self.system.current_time:.3f}\n"
            f"Bonds: {self.system.n_bonds}\n"
            f"Complexes: {n_complexes}\n"
            f"Largest: {largest}\n"
            f"Binds: {self.system.stats['bind_events']}\n"
            f"Breaks: {self.system.stats['break_events']}"
        )
        self.stats_text.set_text(stats)
        
        self.title_text.set_text(
            f"NS-IGAA Particle Assembly | "
            f"Particles: {self.n_particles} | "
            f"Ports: {len(self.palette.ports)}"
        )
    
    def _animation_step(self, frame: int):
        """Single animation frame callback."""
        if not self._running:
            return
        
        self.step_count += 1
        
        stepped = self.system.step()
        
        added, removed = self._sync_graph_with_system()
        
        self._update_node_properties()
        
        self.layout_step_count += 1
        if self.layout_step_count >= self.config.layout_update_interval:
            self._update_forceatlas2()
            self.layout_step_count = 0
        
        self._draw_nodes()
        self._draw_edges()
        self._update_stats_text()
        
        if not stepped:
            self._running = False
            if self.config.save_path:
                self._save_animation()
        
        return [self.node_collection] + self.edge_collections + self.label_texts + [self.title_text, self.stats_text]
    
    def _init_animation(self):
        """Initialize animation frame."""
        self._draw_nodes()
        self._draw_edges()
        self._update_stats_text()
        return [self.node_collection] + self.edge_collections + self.label_texts + [self.title_text, self.stats_text]
    
    def _save_animation(self):
        """Save animation to file."""
        if self.config.save_path:
            writer = animation.PillowWriter(fps=self.config.save_fps)
            self.animation.save(self.config.save_path, writer=writer, dpi=self.config.dpi)
            print(f"Animation saved to {self.config.save_path}")
    
    def animate(
        self,
        max_frames: Optional[int] = None,
        max_time: Optional[float] = None,
    ) -> animation.FuncAnimation:
        """
        Start the real-time animation.
        
        Args:
            max_frames: Maximum number of animation frames (simulation steps)
            max_time: Maximum simulation time to run
            
        Returns:
            The FuncAnimation object
        """
        self._running = True
        self.step_count = 0
        self.layout_step_count = 0
        
        def frame_generator():
            frame = 0
            while self._running:
                if max_frames and frame >= max_frames:
                    break
                if max_time and self.system.current_time >= max_time:
                    break
                yield frame
                frame += 1
        
        self.animation = animation.FuncAnimation(
            self.fig,
            self._animation_step,
            frames=frame_generator(),
            init_func=self._init_animation,
            interval=self.config.animation_interval,
            blit=False,
            repeat=self.config.repeat,
            cache_frame_data=False,
        )
        
        plt.show()
        return self.animation
    
    def run_headless(
        self,
        max_steps: int,
        layout_update_interval: Optional[int] = None,
        callback: Optional[Callable[[int, np.ndarray, Dict], None]] = None,
    ) -> List[Dict]:
        """
        Run simulation headlessly (no display) and collect frame data.
        
        Args:
            max_steps: Number of simulation steps
            layout_update_interval: Override config layout update interval
            callback: Optional callback(step, positions, stats) called each step
            
        Returns:
            List of frame data dictionaries
        """
        interval = layout_update_interval or self.config.layout_update_interval
        frames_data = []
        
        for step in range(max_steps):
            stepped = self.system.step()
            added, removed = self._sync_graph_with_system()
            self._update_node_properties()
            
            self.layout_step_count += 1
            if self.layout_step_count >= interval:
                self._update_forceatlas2()
                self.layout_step_count = 0
            
            frame_data = {
                'step': self.system.current_step,
                'time': self.system.current_time,
                'positions': self.node_positions.copy(),
                'edges': list(self.current_edges),
                'node_colors': self.node_colors.copy(),
                'node_sizes': self.node_sizes.copy(),
                'n_bonds': self.system.n_bonds,
                'complexes': [list(c) for c in self.system.complex_analyzer.get_complexes()],
                'stats': {
                    'bind_events': self.system.stats['bind_events'],
                    'break_events': self.system.stats['break_events'],
                }
            }
            frames_data.append(frame_data)
            
            if callback:
                callback(step, self.node_positions.copy(), frame_data)
            
            if not stepped:
                break
        
        return frames_data
    
    def get_current_state(self) -> Dict:
        """Get current visualization state."""
        return {
            'positions': self.node_positions.copy(),
            'edges': list(self.current_edges),
            'node_colors': self.node_colors.copy(),
            'node_sizes': self.node_sizes.copy(),
            'node_labels': self.node_labels.copy(),
            'port_colors': self.port_colors.copy(),
        }
    
    def reset_layout(self, seed: Optional[int] = None):
        """Reset layout to random positions and re-run ForceAtlas2."""
        if seed is not None:
            np.random.seed(seed)
        self.node_positions = np.random.randn(self.n_particles, 2) * 1
        self.forceatlas2.set_positions(self.node_positions)
        self.forceatlas2.state.velocities.fill(0)
        self._initialize_positions()


def create_visualizer(
    system: ParticleSystem,
    **config_kwargs
) -> ParticleSystemVisualizer:
    """
    Factory function to create a ParticleSystemVisualizer with custom config.
    
    Example:
        system = create_system_from_strings([['white', 'red'], ['green', 'blue']])
        visualizer = create_visualizer(system, layout_update_interval=5, animation_interval=50)
        visualizer.animate(max_frames=200)
    """
    config = VisualizationConfig(**config_kwargs)
    return ParticleSystemVisualizer(system, config)


def animate_system(
    system: ParticleSystem,
    max_frames: int = 200,
    max_time: Optional[float] = None,
    layout_update_interval: int = 1,
    animation_interval: int = 1,
    figsize: Tuple[float, float] = (12, 10),
    save_path: Optional[str] = None,
    **kwargs
) -> animation.FuncAnimation:
    """
    Convenience function to quickly animate a ParticleSystem.
    
    Args:
        system: ParticleSystem to animate
        max_frames: Maximum animation frames
        max_time: Maximum simulation time
        layout_update_interval: ForceAtlas2 update frequency (steps)
        animation_interval: Milliseconds between frames
        figsize: Figure size
        save_path: Optional path to save animation (GIF)
        **kwargs: Additional VisualizationConfig parameters
        
    Returns:
        FuncAnimation object
    """
    config = VisualizationConfig(
        figsize=figsize,
        layout_update_interval=layout_update_interval,
        animation_interval=animation_interval,
        save_path=save_path,
        **kwargs
    )
    visualizer = ParticleSystemVisualizer(system, config)
    return visualizer.animate(max_frames=max_frames, max_time=max_time)


def run_headless_simulation(
    system: ParticleSystem,
    max_steps: int,
    layout_update_interval: int = 1,
    callback: Optional[Callable[[int, np.ndarray, Dict], None]] = None,
) -> List[Dict]:
    """
    Run simulation headlessly and collect frame data for custom rendering.
    
    Args:
        system: ParticleSystem to simulate
        max_steps: Number of simulation steps
        layout_update_interval: ForceAtlas2 update frequency
        callback: Optional callback(step, positions, frame_data)
        
    Returns:
        List of frame data dictionaries
    """
    config = VisualizationConfig(layout_update_interval=layout_update_interval)
    visualizer = ParticleSystemVisualizer(system, config)
    return visualizer.run_headless(max_steps, layout_update_interval, callback)


def plot_static_snapshot(
    system: ParticleSystem,
    ax: Optional[Axes] = None,
    config: Optional[VisualizationConfig] = None,
    layout_iterations: int = 100,
) -> Tuple[Figure, Axes]:
    """
    Create a static snapshot plot of the current system state.
    
    Args:
        system: ParticleSystem to plot
        ax: Optional matplotlib axes to plot on
        config: Optional VisualizationConfig
        layout_iterations: ForceAtlas2 iterations for layout
        
    Returns:
        (figure, axes) tuple
    """
    if config is None:
        config = VisualizationConfig()
    
    visualizer = ParticleSystemVisualizer(system, config)
    visualizer.forceatlas2.run(
        edges=[(b.particle_i, b.particle_j) for b in system.bond_graph.get_bonds()],
        iterations=layout_iterations,
    )
    visualizer.node_positions = visualizer.forceatlas2.get_positions()
    visualizer._update_node_properties()
    
    if ax is None:
        fig, ax = plt.subplots(figsize=config.figsize)
        fig.patch.set_facecolor(config.background_color)
        ax.set_facecolor(config.background_color)
    else:
        fig = ax.figure
    
    ax.scatter(
        visualizer.node_positions[:, 0],
        visualizer.node_positions[:, 1],
        s=visualizer.node_sizes,
        c=visualizer.node_colors,
        alpha=0.8,
        edgecolors='black',
        linewidths=0.5,
        zorder=3,
    )
    
    for u, v, port_i, port_j in visualizer.current_edges:
        x1, y1 = visualizer.node_positions[u]
        x2, y2 = visualizer.node_positions[v]
        color = visualizer.port_colors.get(port_i, 'gray')
        ax.plot([x1, x2], [y1, y2], color=color, linewidth=1.5, alpha=0.6, zorder=2)
    
    if config.show_labels:
        for i, label in visualizer.node_labels.items():
            x, y = visualizer.node_positions[i]
            ax.text(x, y, label, ha='center', va='center',
                   fontsize=config.label_fontsize, fontweight='bold', zorder=4)
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    complexes = system.complex_analyzer.get_complexes()
    n_complexes = len(complexes)
    largest = max(len(c) for c in complexes) if complexes else 0
    ax.set_title(
        f"NS-IGAA Snapshot | Step: {system.current_step} | "
        f"Time: {system.current_time:.3f} | Bonds: {system.n_bonds} | "
        f"Complexes: {n_complexes} | Largest: {largest}",
        fontsize=12
    )
    
    return fig, ax


__all__ = [
    "VisualizationConfig",
    "ForceAtlas2Layout",
    "ParticleSystemVisualizer",
    "create_visualizer",
    "animate_system",
    "run_headless_simulation",
    "plot_static_snapshot",
]