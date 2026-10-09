"""
Graph Topology: Explicit multi-graph representation of particle bonds.

The Explicit Graph Topology G(t) = (V, E(t)) tracks the actual physical
structure of the assembly. Bonds are colored edges between particles,
forming a multi-graph where multiple edges (of different colors) can exist
between the same pair of particles.

Key features:
- NetworkX MultiGraph for parallel edges with color attributes
- Efficient bond addition/removal with O(1) updates
- Complex detection via connected components
- Color-layer adjacency matrices for mathematical analysis
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional, Iterator, Mapping
from collections import defaultdict, Counter
import networkx as nx
import numpy as np
from numpy.typing import NDArray

from ..core.colors import Color, ColorPalette
from ..core.particle import Particle


@dataclass
class Bond:
    """A single bond between two particles at specific colored sites."""
    particle_i: int
    particle_j: int
    color_i: Color
    color_j: Color
    edge_key: int  # NetworkX edge key for multi-graph
    
    def __str__(self) -> str:
        return f"{self.particle_i}({self.color_i.name.lower()})--{self.particle_j}({self.color_j.name.lower()})"
    
    def reversed(self) -> 'Bond':
        """Return the same bond with particles swapped."""
        return Bond(
            particle_i=self.particle_j,
            particle_j=self.particle_i,
            color_i=self.color_j,
            color_j=self.color_i,
            edge_key=self.edge_key
        )


class BondGraph:
    """
    Explicit Multi-Graph Topology tracking all bonds.
    
    Uses NetworkX MultiGraph where:
    - Nodes = particle indices (0 to N-1)
    - Edges = bonds, with attributes: color_i, color_j, edge_key
    - Multiple edges between same nodes allowed (different colors or same color)
    
    This corresponds to the mathematical G(t) = (V, E(t)) with adjacency
    matrices A^c for each color layer.
    """
    
    def __init__(self, n_particles: int, palette: ColorPalette):
        self.n_particles = n_particles
        self.palette = palette
        self.graph = nx.MultiGraph()
        self.graph.add_nodes_from(range(n_particles))
        self.edge_counter = 0
        # Index: color -> set of (u, v, key) for fast lookups
        self._color_index: Dict[Color, Set[Tuple[int, int, int]]] = defaultdict(set)
        # Degree tracking per particle per color
        self._color_degree: Dict[int, Dict[Color, int]] = defaultdict(lambda: defaultdict(int))
    
    @property
    def n_bonds(self) -> int:
        return self.graph.number_of_edges()
    
    @property
    def n_nodes(self) -> int:
        return self.n_particles
    
    def add_bond(self, particle_i: int, particle_j: int, 
                 color_i: Color, color_j: Color) -> int:
        """
        Add a bond between particles at specified colors.
        
        Returns the edge key for future reference.
        """
        edge_key = self.edge_counter
        self.edge_counter += 1
        
        # Add to NetworkX graph
        self.graph.add_edge(
            particle_i, particle_j,
            key=edge_key,
            color_i=color_i,
            color_j=color_j
        )
        
        # Update color index
        self._color_index[color_i].add((particle_i, particle_j, edge_key))
        self._color_index[color_j].add((particle_j, particle_i, edge_key))
        
        # Update color degrees
        self._color_degree[particle_i][color_i] += 1
        self._color_degree[particle_j][color_j] += 1
        
        return edge_key
    
    def remove_bond(self, particle_i: int, particle_j: int, edge_key: int) -> Optional[Bond]:
        """
        Remove a bond by its edge key.
        
        Returns the removed bond info, or None if not found.
        """
        # Get bond data before removal
        if not self.graph.has_edge(particle_i, particle_j, key=edge_key):
            return None
        
        edge_data = self.graph.get_edge_data(particle_i, particle_j, key=edge_key)
        color_i = edge_data['color_i']
        color_j = edge_data['color_j']
        
        # Remove from NetworkX
        self.graph.remove_edge(particle_i, particle_j, key=edge_key)
        
        # Update color index
        self._color_index[color_i].discard((particle_i, particle_j, edge_key))
        self._color_index[color_j].discard((particle_j, particle_i, edge_key))
        
        # Update color degrees
        self._color_degree[particle_i][color_i] = max(0, self._color_degree[particle_i][color_i] - 1)
        self._color_degree[particle_j][color_j] = max(0, self._color_degree[particle_j][color_j] - 1)
        
        # Clean up empty degree entries
        if self._color_degree[particle_i][color_i] == 0:
            del self._color_degree[particle_i][color_i]
        if self._color_degree[particle_j][color_j] == 0:
            del self._color_degree[particle_j][color_j]
        
        return Bond(particle_i, particle_j, color_i, color_j, edge_key)
    
    def get_bonds(self) -> List[Bond]:
        """Get all bonds as list."""
        bonds = []
        for u, v, key, data in self.graph.edges(keys=True, data=True):
            bonds.append(Bond(u, v, data['color_i'], data['color_j'], key))
        return bonds
    
    def get_bonds_by_color(self, color: Color) -> List[Bond]:
        """Get all bonds involving a specific color."""
        bonds = []
        for u, v, key in self._color_index.get(color, set()):
            edge_data = self.graph.get_edge_data(u, v, key=key)
            if edge_data:
                bonds.append(Bond(u, v, edge_data['color_i'], edge_data['color_j'], key))
        return bonds
    
    def get_particle_bonds(self, particle_idx: int) -> List[Bond]:
        """Get all bonds for a specific particle."""
        bonds = []
        for neighbor, key, data in self.graph.edges(particle_idx, keys=True, data=True):
            if neighbor == particle_idx:
                # Self-loop
                bonds.append(Bond(particle_idx, particle_idx, data['color_i'], data['color_j'], key))
            else:
                # Determine which color belongs to this particle
                # The graph stores color_i for u, color_j for v
                if u == particle_idx:
                    bonds.append(Bond(u, v, data['color_i'], data['color_j'], key))
                else:
                    bonds.append(Bond(v, u, data['color_j'], data['color_i'], key))
        return bonds
    
    def get_color_degree(self, particle_idx: int, color: Color) -> int:
        """Get number of bonds of a specific color for a particle."""
        return self._color_degree[particle_idx].get(color, 0)
    
    def get_all_color_degrees(self, particle_idx: int) -> Dict[Color, int]:
        """Get all color degrees for a particle."""
        return dict(self._color_degree[particle_idx])
    
    def has_bond_between(self, u: int, v: int) -> bool:
        """Check if any bond exists between two particles."""
        return self.graph.has_edge(u, v)
    
    def get_adjacency_matrix(self, color: Color) -> NDArray[np.int32]:
        """
        Get adjacency matrix A^c for a specific color layer.
        
        A^c[i,j] = number of bonds of color c from i to j (counting color_i=c)
        """
        A = np.zeros((self.n_particles, self.n_particles), dtype=np.int32)
        for u, v, key in self._color_index.get(color, set()):
            edge_data = self.graph.get_edge_data(u, v, key=key)
            if edge_data and edge_data['color_i'] == color:
                A[u, v] += 1
            elif edge_data and edge_data['color_j'] == color:
                A[v, u] += 1
        return A
    
    def get_all_adjacency_matrices(self) -> Dict[Color, NDArray[np.int32]]:
        """Get adjacency matrices for all colors."""
        return {color: self.get_adjacency_matrix(color) for color in self.palette.colors}
    
    def get_laplacian(self, color: Color = None) -> NDArray[np.float64]:
        """
        Get graph Laplacian. If color specified, use that color layer.
        Otherwise, use total bond count (sum over colors).
        """
        if color is not None:
            A = self.get_adjacency_matrix(color)
        else:
            A = sum(self.get_all_adjacency_matrices().values())
        
        degree = A.sum(axis=1)
        L = np.diag(degree) - A
        return L
    
    def __len__(self) -> int:
        return self.n_bonds
    
    def __iter__(self) -> Iterator[Bond]:
        return iter(self.get_bonds())
    
    def __str__(self) -> str:
        return f"BondGraph(n_particles={self.n_particles}, n_bonds={self.n_bonds})"


class ComplexAnalyzer:
    """
    Analyzes connected components (complexes) in the bond graph.
    
    A "complex" is a connected component in the multi-graph, representing
    a physically connected assembly of particles.
    """
    
    def __init__(self, bond_graph: BondGraph):
        self.bond_graph = bond_graph
        self._complexes: Optional[List[Set[int]]] = None
        self._complex_id_map: Optional[Dict[int, int]] = None  # particle -> complex_id
    
    def find_complexes(self) -> List[Set[int]]:
        """Find all connected components (complexes)."""
        # Use underlying simple graph for connectivity
        simple_graph = nx.Graph(self.bond_graph.graph)
        self._complexes = list(nx.connected_components(simple_graph))
        self._complex_id_map = {}
        for cid, complex_nodes in enumerate(self._complexes):
            for node in complex_nodes:
                self._complex_id_map[node] = cid
        return self._complexes
    
    def get_complexes(self) -> List[Set[int]]:
        """Get cached complexes, computing if necessary."""
        if self._complexes is None:
            return self.find_complexes()
        return self._complexes
    
    def get_complex_id(self, particle_idx: int) -> int:
        """Get the complex ID containing a particle."""
        if self._complex_id_map is None:
            self.find_complexes()
        return self._complex_id_map.get(particle_idx, -1)
    
    def get_complex_sizes(self) -> List[int]:
        """Get sizes of all complexes."""
        return [len(c) for c in self.get_complexes()]
    
    def get_largest_complex(self) -> Set[int]:
        """Get the largest complex (by particle count)."""
        complexes = self.get_complexes()
        if not complexes:
            return set()
        return max(complexes, key=len)
    
    def get_complex_statistics(self) -> Dict:
        """Get summary statistics of complex distribution."""
        complexes = self.get_complexes()
        sizes = [len(c) for c in complexes]
        
        if not sizes:
            return {
                'n_complexes': 0,
                'total_particles': 0,
                'size_distribution': Counter(),
                'largest_size': 0,
                'mean_size': 0.0,
                'monomer_fraction': 1.0
            }
        
        size_dist = Counter(sizes)
        n_particles = sum(sizes)
        
        return {
            'n_complexes': len(complexes),
            'total_particles': n_particles,
            'size_distribution': size_dist,
            'largest_size': max(sizes),
            'mean_size': np.mean(sizes),
            'monomer_fraction': size_dist.get(1, 0) / len(complexes),
            'gel_fraction': max(sizes) / n_particles if n_particles > 0 else 0.0
        }
    
    def get_complex_bond_counts(self) -> Dict[int, Dict[Color, int]]:
        """
        Count bonds per color within each complex.
        
        Returns: complex_id -> {color: count}
        """
        complexes = self.get_complexes()
        result = {}
        
        for cid, complex_nodes in enumerate(complexes):
            color_counts = Counter()
            for bond in self.bond_graph:
                if bond.particle_i in complex_nodes and bond.particle_j in complex_nodes:
                    color_counts[bond.color_i] += 1
            result[cid] = dict(color_counts)
        
        return result
    
    def invalidate_cache(self):
        """Invalidate cached complexes (call after graph modification)."""
        self._complexes = None
        self._complex_id_map = None


def create_bond_graph_from_system(particles: List[Particle], palette: ColorPalette) -> BondGraph:
    """Factory to create empty bond graph for a particle system."""
    return BondGraph(n_particles=len(particles), palette=palette)