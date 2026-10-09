# Information-Geometric Stochastic Particle Assembly (IGSPA) User Guide

## Information-Geometric Stochastic Particle Assembly (IGSPA)

A framework for simulating particle assembly using Gillespie SSA with global-bucket optimization, Fisher Information Geometry, and explicit graph topology tracking.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Core Concepts](#core-concepts)
3. [Installation](#installation)
4. [Basic Usage](#basic-usage)
5. [Advanced Features](#advanced-features)
6. [API Reference](#api-reference)
7. [Mathematical Background](#mathematical-background)
8. [Examples](#examples)

---

## Quick Start

```python
from igspa import create_system_from_strings

# Create a simple 2-particle system
system = create_system_from_strings([
    ['white', 'red'],
    ['green', 'blue'],
], k_on=1.5, k_off=0.3)

# Run simulation
stats = system.run(max_steps=100, verbose=True)

# Inspect results
system.print_complexes()
print(f"Final bonds: {stats['final_bonds']}")
```

---

## Core Concepts

### Particles & Binding Sites

Each particle has **binding sites** of specific **colors**. Colors come in complementary pairs:
- `white` ↔ `black`
- `red` ↔ `green`
- `blue` ↔ `yellow`

A particle can bind to another if it has an available site of a color complementary to the other particle's available site.

### Dual Representation

IGSPA maintains two synchronized representations:

1. **Implicit Information Manifold** (Continuous)
   - Tracks availability fractions θᵢᶜ ∈ [0, 1] for each particle and color
   - Equipped with Fisher Information Metric (Shahshahani geometry)
   - As θ → 0, informational distance → ∞ (scarcity geometry)

2. **Explicit Graph Topology** (Discrete)
   - Multi-graph G(t) = (V, E(t)) with colored edges
   - Tracks exact bond structure
   - Enables complex detection via connected components

### Global Bucket Optimization

Instead of O(N²) pairwise evaluation, IGSPA uses **global buckets**:
- B_c = {i | particle i has available sites of color c}
- Aggregate propensities: α_bind = k_on × (Σᵢ∈B_c availableᵢᶜ) × (Σⱼ∈B₋c availableⱼ₋c)
- Complexity: O(|C| + |E|) per step vs O(N²|C|) standard

---

## Installation

Dependencies:
- `numpy >= 1.24`
- `networkx >= 3.0`
- `scipy >= 1.10`
- `lxml >= 6.0` (for GraphML export)

---

## Basic Usage

### Creating a System

```python
from igspa import (
    ParticleBlueprint, 
    ParticleSystem, 
    ColorPalette,
    create_system_from_strings
)

# Method 1: Convenience factory (recommended)
system = create_system_from_strings([
    ['white', 'red'],           # Particle 0
    ['green', 'blue'],          # Particle 1
    ['black', 'green', 'yellow'], # Particle 2
    ['red', 'blue'],            # Particle 3
], k_on=1.5, k_off=0.3)

# Method 2: Explicit blueprints
palette = ColorPalette.standard()
blueprints = [
    ParticleBlueprint.from_string_list(['white', 'red'], palette),
    ParticleBlueprint.from_string_list(['green', 'blue'], palette),
]
config = SimulationConfig(k_on=1.0, k_off=0.1)
system = ParticleSystem(blueprints, palette, config)

# Method 3: Custom palette (e.g., DNA-like)
dna_palette = ColorPalette.from_strings({'A': 'T', 'C': 'G'})
system = create_system_from_strings([
    ['A', 'C'],
    ['T', 'G'],
], palette=dna_palette, k_on=2.0, k_off=0.5)
```

### Running Simulations

```python
# Simple run
stats = system.run(max_steps=1000, verbose=True)

# Run with time limit
stats = system.run(max_steps=1000, max_time=100.0, verbose=True)

# Silent run (for batch processing)
stats = system.run(max_steps=1000, verbose=False)

# With callbacks for custom logging
def my_callback(event_type, step, time, *args):
    print(f"Step {step}: {event_type} - {args}")

system.config.callbacks.append(my_callback)
system.run(max_steps=100, verbose=False)
```

### Inspecting Results

```python
# Print complexes
system.print_complexes()

# Get bond list
bonds = system.get_bond_list()
for bond in bonds:
    print(f"  {bond[0]}-{bond[1]} ({bond[2]}-{bond[3]})")

# Get availability matrix
avail = system.get_availability_matrix()
for label, sites in avail.items():
    print(f"  {label}: {sites}")

# Get state snapshot for checkpointing
state = system.get_state_snapshot()
```

---

## Advanced Features

### 1. Composition-Rejection Sampling (Large Systems)

For systems with >50 particles, enable O(1) sampling:

```python
from igspa import SimulationConfig, create_system_from_strings

config = SimulationConfig(
    k_on=1.0,
    k_off=0.1,
    max_steps=10000,
    use_composition_rejection=True,  # Enable CR sampler
    log_interval=500
)

system = create_system_from_strings(particle_strings, config=config)
stats = system.run(max_steps=5000, verbose=True)
```

### 2. Logging & Export

```python
from igspa import SimulationLogger, StateExporter

# Detailed logging
with SimulationLogger(system, 'simulation_log.jsonl') as logger:
    stats = system.run(max_steps=100, verbose=True)
    logger.save_numpy('simulation_data.npz')

# Export final state
exporter = StateExporter(system)
exporter.export_state('final_state.json', format='json')
exporter.export_state('final_state.npz', format='npz')
exporter.export_for_visualization('final_graph.graphml')  # For Gephi/Cytoscape
exporter.export_complexes_csv('complexes.csv')
exporter.export_time_series_csv('timeseries.csv', logger)
```

### 3. Manifold Geometry Analysis

```python
from igspa import analyze_manifold_geometry

system.run(max_steps=100, verbose=False)
geom = analyze_manifold_geometry(system)

print("Distances to saturation:")
for label, dist in geom['distances_to_saturation'].items():
    print(f"  {label}: {dist:.4f}")

print("Pairwise geodesic distances:")
for (l1, l2), dist in geom['pairwise_distances'].items():
    print(f"  {l1} <-> {l2}: {dist:.4f}")

print("Metric condition numbers:")
for label, cond in geom['metric_condition_numbers'].items():
    print(f"  {label}: {cond:.2f}")
```

### 4. Steady-State Analysis

```python
from igspa import SteadyStateAnalyzer, compute_thermodynamic_quantities

analyzer = SteadyStateAnalyzer(system)

# Equilibrium constants per channel
for channel, k_eq in analyzer.estimate_equilibrium_constants().items():
    c1, c2 = channel
    print(f"  {c1.name}-{c2.name}: K_eq ≈ {k_eq:.4f}")

# Gel fraction (largest complex fraction)
print(f"Gel fraction: {analyzer.compute_gel_fraction():.4f}")

# Mean cluster size
print(f"Mean cluster size: {analyzer.compute_mean_cluster_size():.4f}")

# Thermodynamic quantities
thermo = compute_thermodynamic_quantities(system)
for key, value in thermo.items():
    print(f"  {key}: {value}")
```

### 5. Ensemble Statistics

```python
from igspa import run_ensemble, compute_ensemble_statistics

def system_factory():
    return create_system_from_strings([
        ['white', 'red'],
        ['green', 'blue'],
        ['black', 'green', 'yellow'],
        ['red', 'blue'],
    ], k_on=1.5, k_off=0.3)

# Run 50 independent trajectories
ensemble_results = run_ensemble(system_factory, n_runs=50, max_steps=100)

# Compute statistics
ensemble_stats = compute_ensemble_statistics(ensemble_results)
for key, stat in ensemble_stats.items():
    print(f"  {key}: mean={stat['mean']:.2f}, std={stat['std']:.2f}")
```

### 6. Validation & Debugging

```python
from igspa import validate_system

# Check internal consistency
errors = validate_system(system)
if errors:
    for err in errors:
        print(f"ERROR: {err}")
else:
    print("System is valid!")
```

---

## API Reference

### Core Classes

| Class | Module | Description |
|-------|--------|-------------|
| `ParticleBlueprint` | `igspa.core.particle` | Immutable particle type definition |
| `Particle` | `igspa.core.particle` | Dynamic particle instance |
| `ParticleSystem` | `igspa.core.system` | Main simulation orchestrator |
| `SimulationConfig` | `igspa.core.system` | Configuration dataclass |
| `ColorPalette` | `igspa.core.colors` | Binding color alphabet |
| `Color` | `igspa.core.colors` | Enum of binding colors |

### Geometry

| Class | Module | Description |
|-------|--------|-------------|
| `InformationManifold` | `igspa.geometry.manifold` | Continuous availability manifold |
| `FisherMetric` | `igspa.geometry.manifold` | Fisher Information Metric |

### Topology

| Class | Module | Description |
|-------|--------|-------------|
| `BondGraph` | `igspa.topology.graph` | Multi-graph of bonds |
| `Bond` | `igspa.topology.graph` | Single colored bond |
| `ComplexAnalyzer` | `igspa.topology.graph` | Connected component analysis |

### Algorithms

| Class | Module | Description |
|-------|--------|-------------|
| `GlobalBucketSampler` | `igspa.algorithms.gillespie` | O(|C|+|E|) propensity eval |
| `TwoStageSampler` | `igspa.algorithms.gillespie` | Standard two-stage SSA |
| `CompositionRejectionSampler` | `igspa.algorithms.gillespie` | O(1) CR sampling |

### I/O

| Class | Module | Description |
|-------|--------|-------------|
| `SimulationLogger` | `igspa.io.serialization` | Trajectory logging |
| `StateExporter` | `igspa.io.serialization` | State export (JSON, NPZ, GraphML, CSV) |

### Analysis

| Function/Class | Module | Description |
|----------------|--------|-------------|
| `validate_system` | `igspa.utils.analysis` | Internal consistency check |
| `SteadyStateAnalyzer` | `igspa.utils.analysis` | Equilibrium analysis |
| `analyze_manifold_geometry` | `igspa.utils.analysis` | Fisher geometry analysis |
| `compute_thermodynamic_quantities` | `igspa.utils.analysis` | Entropy, free energy |
| `run_ensemble` | `igspa.utils.analysis` | Multi-trajectory runs |
| `compute_ensemble_statistics` | `igspa.utils.analysis` | Statistical analysis |

### Factory Functions

```python
from igspa import (
    create_system_from_strings,
    create_standard_test_system,
    create_linear_chain_system,
    create_branched_system
)
```

| Function | Description |
|----------|-------------|
| `create_system_from_strings(particle_strings, k_on, k_off, palette=None, **config_kwargs)` | Main factory - create from string color lists |
| `create_standard_test_system(k_on=1.5, k_off=0.3)` | Standard 4-particle test system |
| `create_linear_chain_system(n_particles, k_on=1.0, k_off=0.1)` | Linear chain forming system (alternating colors) |
| `create_branched_system(n_particles, k_on=1.0, k_off=0.1)` | Branching capable system (3 colors per particle) |

---

## Mathematical Background

### Fisher Information Manifold

Each particle i lives on a manifold ℳᵢ parameterized by availability fractions θᵢ = (θᵢ¹, ..., θᵢᵐ):

```
θᵢᶜ = available_sitesᵢᶜ / capacityᵢᶜ
```

The Fisher-Rao metric tensor is diagonal:
```
gᵢⱼ = δᵢⱼ / θᵢ
```

Geodesic distance between states p and q:
```
d(p, q) = arccos(Σ √(pᵢ qᵢ))
```

As θᵢᶜ → 0, distances diverge → ∞, geometrically encoding scarcity.

### Global Bucket Propensity

Binding propensity for channel (c, -c):
```
α_bind^{c,-c} = k_on × (Σᵢ∈B_c availableᵢᶜ) × (Σⱼ∈B₋c availableⱼ₋c)
```

Total system propensity:
```
a₀ = Σ_c α_bind^{c,-c} + k_off × |E|
```

### Two-Stage Sampling

1. **Channel Selection**: Sample (c, -c) with probability ∝ α_bind^{c,-c}
2. **Particle Selection**: 
   - Sample i ∈ B_c with P(i) ∝ availableᵢᶜ
   - Sample j ∈ B₋c with P(j) ∝ availableⱼ₋c

---

## Examples

### Example 1: Basic Simulation (`examples/basic_simulation.py`)

```bash
python examples/basic_simulation.py
```

Demonstrates standard system creation, running, and analysis.

### Example 2: Advanced Features (`examples/advanced_features.py`)

```bash
python examples/advanced_features.py
```

Demonstrates:
- Logging & export (JSONL, NPZ, GraphML, CSV)
- Composition-rejection sampling (100 particles)
- Ensemble statistics (20 runs)
- Manifold geometry analysis
- Custom DNA-like palette

### Example 3: Original Example (`example.py`)

```bash
python example.py
```

The original minimal example.

---

## Performance Tips

1. **Use global buckets** (default) for O(|C|+|E|) scaling
2. **Enable CR sampler** for N > 50 particles
3. **Disable complex tracking** if not needed: `config.track_complexes = False`
4. **Disable manifold tracking** if not needed: `config.track_manifold = False`
5. **Use NumPy export** (.npz) for large trajectory analysis

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: igspa` | Run `pip install -e .` from project root |
| `ImportError: cannot import name 'create_system_from_strings'` | Reinstall package after code changes |
| GraphML export fails | Install `lxml`: `pip install lxml` |
| Slow simulation with many particles | Enable `use_composition_rejection=True` |

### Validation

Always validate after modifications:
```python
from igspa import validate_system
errors = validate_system(system)
assert not errors, f"Validation failed: {errors}"
```

---

## Extending IGSPA

### Custom Color Palettes

```python
# 4-color, 2-pair system
custom_palette = ColorPalette.from_strings({
    'A': 'T',  # Pair 1
    'C': 'G',  # Pair 2
})
```

### Custom Analysis

```python
from igspa import SteadyStateAnalyzer

class CustomAnalyzer(SteadyStateAnalyzer):
    def my_custom_metric(self):
        # Access system internals
        return self.system.n_bonds / self.system.n_particles
```

### Callbacks for Real-Time Monitoring

```python
def monitor_gelation(event_type, step, time, *args):
    if event_type == 'bind':
        gel_frac = system.complex_analyzer.get_complex_statistics()['gel_fraction']
        if gel_frac > 0.5:
            print(f"Gelation detected at t={time:.2f}!")

system.config.callbacks.append(monitor_gelation)
```

---

## License

MIT License - See LICENSE file for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass: `python -m pytest tests/ -v`
5. Submit a pull request

---

## Citation

If you use IGSPA in research, please cite:

```
Information-Geometric Stochastic Particle Assembly (IGSPA)
Carson Scott, 2024
```