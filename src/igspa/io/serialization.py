"""
I/O and Serialization: Logging, checkpointing, and data export.

Provides utilities for saving/loading simulation state, exporting results
for analysis, and logging simulation trajectories.
"""

from __future__ import annotations
import json
import pickle
import numpy as np
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, TextIO
from pathlib import Path
from datetime import datetime

from ..core.system import ParticleSystem, SimulationState
from ..core.colors import Color, ColorPalette
from ..core.particle import Particle, ParticleBlueprint
from ..topology.graph import BondGraph, Bond, ComplexAnalyzer


class SimulationLogger:
    """
    Logs simulation trajectory to file or memory.
    
    Supports multiple output formats:
    - JSON Lines (streaming, human-readable)
    - NumPy .npz (efficient binary for numerical data)
    - Pickle (full Python object serialization)
    """
    
    def __init__(self, system: ParticleSystem, log_file: str = None):
        self.system = system
        self.log_file = log_file
        self.file_handle: Optional[TextIO] = None
        self.records: List[Dict] = []
        self._header_written = False
        
        if log_file:
            self.file_handle = open(log_file, 'w')
            self._write_header()
    
    def _write_header(self):
        """Write metadata header."""
        header = {
            'type': 'ns_igaa_simulation_log',
            'version': '1.0',
            'timestamp': datetime.now().isoformat(),
            'n_particles': self.system.n_particles,
            'n_colors': len(self.system.palette),
            'k_on': self.system.config.k_on,
            'k_off': self.system.config.k_off,
            'palette': {c.name: c.name.lower() for c in self.system.palette.colors},
        }
        if self.file_handle:
            self.file_handle.write(json.dumps(header) + '\n')
        self.records.append(header)
        self._header_written = True
    
    def log_step(self, event_type: str = None, **event_data):
        """Log current state after a step."""
        record = {
            'step': self.system.current_step,
            'time': self.system.current_time,
            'n_bonds': self.system.n_bonds,
            'event': event_type,
            'event_data': event_data,
        }
        
        if self.system.config.track_complexes:
            complexes = self.system.complex_analyzer.get_complexes()
            record['complexes'] = {
                'count': len(complexes),
                'sizes': [len(c) for c in complexes],
                'largest': max(len(c) for c in complexes) if complexes else 0,
            }
        
        # Particle availability summary
        avail = {}
        for p in self.system.particles:
            avail[p.label] = {c.name: p.available_sites.get(c, 0) for c in self.system.palette.colors}
        record['availability'] = avail
        
        if self.file_handle:
            self.file_handle.write(json.dumps(record, default=str) + '\n')
            self.file_handle.flush()
        
        self.records.append(record)
    
    def log_interval(self, interval: int = 100):
        """Log at specified intervals (call from callback)."""
        if self.system.current_step % interval == 0:
            self.log_step()
    
    def close(self):
        """Close log file."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def to_numpy(self) -> Dict[str, np.ndarray]:
        """Convert logged records to NumPy arrays for analysis."""
        steps = np.array([r['step'] for r in self.records[1:]], dtype=np.int32)
        times = np.array([r['time'] for r in self.records[1:]], dtype=np.float64)
        n_bonds = np.array([r['n_bonds'] for r in self.records[1:]], dtype=np.int32)
        
        # Complex stats
        complex_counts = np.array([r.get('complexes', {}).get('count', 0) for r in self.records[1:]], dtype=np.int32)
        largest_sizes = np.array([r.get('complexes', {}).get('largest', 0) for r in self.records[1:]], dtype=np.int32)
        
        return {
            'steps': steps,
            'times': times,
            'n_bonds': n_bonds,
            'complex_counts': complex_counts,
            'largest_complex_sizes': largest_sizes,
        }
    
    def save_numpy(self, filepath: str):
        """Save logged data as compressed NumPy archive."""
        data = self.to_numpy()
        np.savez_compressed(filepath, **data)
    
    def get_dataframe(self):
        """Convert to pandas DataFrame if available."""
        try:
            import pandas as pd
            return pd.DataFrame(self.records[1:])  # Skip header
        except ImportError:
            return None


class StateExporter:
    """
    Exports simulation state for analysis, visualization, or checkpointing.
    """
    
    def __init__(self, system: ParticleSystem):
        self.system = system
    
    def export_state(self, filepath: str, format: str = 'json'):
        """
        Export complete simulation state.
        
        Formats: 'json', 'pickle', 'npz'
        """
        state = self.system.get_state_snapshot()
        
        if format == 'json':
            self._export_json(state, filepath)
        elif format == 'pickle':
            self._export_pickle(state, filepath)
        elif format == 'npz':
            self._export_npz(state, filepath)
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def _export_json(self, state: SimulationState, filepath: str):
        """Export as JSON (human-readable, no NumPy arrays)."""
        data = {
            'step': state.step,
            'time': state.time,
            'n_particles': len(state.particles),
            'n_bonds': state.bond_graph.n_bonds,
            'particles': [
                {
                    'id': p.particle_id,
                    'label': p.label,
                    'blueprint': {c.name: p.blueprint.site_counts.get(c, 0) for c in p.blueprint.site_counts},
                    'available': {c.name: p.available_sites.get(c, 0) for c in p.available_sites},
                }
                for p in state.particles
            ],
            'bonds': [
                {
                    'particle_i': b.particle_i,
                    'particle_j': b.particle_j,
                    'color_i': b.color_i.name,
                    'color_j': b.color_j.name,
                    'edge_key': b.edge_key,
                }
                for b in state.bond_graph.get_bonds()
            ],
            'complexes': state.complex_stats,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def _export_pickle(self, state: SimulationState, filepath: str):
        """Export as pickle (full Python objects)."""
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def _export_npz(self, state: SimulationState, filepath: str):
        """Export as NumPy archive (efficient for numerical analysis)."""
        n = len(state.particles)
        colors = sorted(self.system.palette.colors, key=lambda c: c.name)
        n_colors = len(colors)
        color_idx = {c: i for i, c in enumerate(colors)}
        
        # Availability matrix: (n_particles, n_colors)
        availability = np.zeros((n, n_colors), dtype=np.int32)
        capacity = np.zeros((n, n_colors), dtype=np.int32)
        
        for i, p in enumerate(state.particles):
            for c, count in p.available_sites.items():
                availability[i, color_idx[c]] = count
            for c, count in p.blueprint.site_counts.items():
                capacity[i, color_idx[c]] = count
        
        # Bond list
        bonds = state.bond_graph.get_bonds()
        bond_data = np.zeros((len(bonds), 5), dtype=np.int32)  # i, j, color_i, color_j, key
        for idx, b in enumerate(bonds):
            bond_data[idx] = [b.particle_i, b.particle_j, color_idx[b.color_i], color_idx[b.color_j], b.edge_key]
        
        # Adjacency matrices per color
        adj_matrices = {}
        for c in colors:
            adj_matrices[c.name] = state.bond_graph.get_adjacency_matrix(c)
        
        np.savez_compressed(
            filepath,
            step=state.step,
            time=state.time,
            n_particles=n,
            n_colors=n_colors,
            colors=np.array([c.name for c in colors]),
            availability=availability,
            capacity=capacity,
            bonds=bond_data,
            **{f'adj_{k}': v for k, v in adj_matrices.items()},
            complex_stats=state.complex_stats,
        )
    
    @staticmethod
    def load_state(filepath: str, format: str = 'json') -> Dict:
        """Load exported state."""
        if format == 'json':
            with open(filepath, 'r') as f:
                return json.load(f)
        elif format == 'pickle':
            with open(filepath, 'rb') as f:
                return pickle.load(f)
        elif format == 'npz':
            return dict(np.load(filepath, allow_pickle=True))
        else:
            raise ValueError(f"Unknown format: {format}")
    
    def export_for_visualization(self, filepath: str):
        """
        Export in format suitable for graph visualization (GraphML, GEXF, etc).
        """
        # Create NetworkX graph with all attributes
        G = nx.MultiGraph()
        
        # Add nodes with particle attributes (flatten dicts to strings for GraphML)
        for p in self.system.particles:
            blueprint_str = ','.join(f"{c.name}:{p.blueprint.site_counts.get(c, 0)}" for c in p.blueprint.site_counts)
            available_str = ','.join(f"{c.name}:{p.available_sites.get(c, 0)}" for c in p.available_sites)
            G.add_node(
                p.particle_id,
                label=p.label,
                blueprint=blueprint_str,
                available=available_str,
            )
        
        # Add edges with bond attributes
        for bond in self.system.bond_graph.get_bonds():
            G.add_edge(
                bond.particle_i,
                bond.particle_j,
                key=bond.edge_key,
                color_i=bond.color_i.name,
                color_j=bond.color_j.name,
            )
        
        # Save as GraphML (supports multi-graph)
        nx.write_graphml(G, filepath)
    
    def export_complexes_csv(self, filepath: str):
        """Export complex membership as CSV."""
        import csv
        complexes = self.system.complex_analyzer.get_complexes()
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['complex_id', 'particle_id', 'particle_label', 'size'])
            for cid, comp in enumerate(complexes):
                for pid in comp:
                    p = self.system.particles[pid]
                    writer.writerow([cid, pid, p.label, len(comp)])
    
    def export_time_series_csv(self, filepath: str, logger: SimulationLogger):
        """Export time series data as CSV."""
        import csv
        data = logger.to_numpy()
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'time', 'n_bonds', 'complex_count', 'largest_complex'])
            for i in range(len(data['steps'])):
                writer.writerow([
                    data['steps'][i],
                    data['times'][i],
                    data['n_bonds'][i],
                    data['complex_counts'][i],
                    data['largest_complex_sizes'][i],
                ])


# Import networkx for visualization export
import networkx as nx