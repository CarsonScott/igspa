"""Information-Geometric Stochastic Particle Assembly (IGSPA)

A framework for simulating particle assembly using Gillespie SSA with
global-bucket optimization, Fisher Information Geometry, and explicit
graph topology tracking.

Mathematical Foundation:
- Implicit Information Manifold: Continuous availability parameters θ_i^p
  governed by Fisher Information Metric (Shahshahani geometry)
- Explicit Graph Topology: Multi-graph G(t) = (V, E(t)) with port-layer
- Adjacency matrices A^p
- Global Buckets: B_p = {i ∈ V | θ_i^p > 0} for O(|P| + |E|) propensity evaluation
- Two-Stage Gillespie Sampling: Channel selection → Weighted particle selection

Complexity: O(|P| + |E|) per step vs O(N²·|P|) standard
"""

__version__ = "1.0.0"
__author__ = "IGSPA Team"

from .core.particle import Particle, ParticleBlueprint, ParticleArchetype
from .core.system import ParticleSystem, create_system_from_strings, create_system_from_config, SimulationConfig
from .core.ports import PortPalette, DEFAULT_PORT_PALETTE
from .core.config import SystemConfiguration
from .algorithms.gillespie import (
    GlobalBucketSampler,
    TwoStageSampler,
    CompositionRejectionSampler,
)
from .geometry.manifold import InformationManifold, FisherMetric
from .topology.graph import BondGraph, ComplexAnalyzer
from .io.serialization import SimulationLogger, StateExporter
from .visualization.networkx_visualizer import (
    IGSPAVisualizer,
    VisualizationConfig,
    animate_system,
    run_headless_simulation,
    plot_static_snapshot,
)
from .visualization.builders import (
    create_palette,
    create_archetypes,
    create_config,
    create_system,
    random_palette,
    random_archetypes,
)
from .utils.analysis import (
    SteadyStateAnalyzer, analyze_manifold_geometry,
    compute_thermodynamic_quantities, run_ensemble, compute_ensemble_statistics,
    create_standard_test_system, create_linear_chain_system, create_branched_system
)

__all__ = [
    "Particle",
    "ParticleBlueprint",
    "ParticleArchetype",
    "ParticleSystem",
    "create_system_from_strings",
    "create_system_from_config",
    # "create_system_from_configuration",
    "SimulationConfig",
    "PortPalette",
    "DEFAULT_PORT_PALETTE",
    "SystemConfiguration",
    "GlobalBucketSampler",
    "TwoStageSampler",
    "CompositionRejectionSampler",
    "InformationManifold",
    "FisherMetric",
    "BondGraph",
    "ComplexAnalyzer",
    "SimulationLogger",
    "StateExporter",
    "SteadyStateAnalyzer",
    "analyze_manifold_geometry",
    "compute_thermodynamic_quantities",
    "run_ensemble",
    "compute_ensemble_statistics",
    "create_standard_test_system",
    "create_linear_chain_system",
    "create_branched_system",
    "VisualizationConfig",
    "animate_system",
    "run_headless_simulation",
    "plot_static_snapshot",
    "IGSPAVisualizer",
    "create_palette",
    "create_archetypes",
    "create_config",
    "create_system",
    "random_palette",
    "random_archetypes",
]
