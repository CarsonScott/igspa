# Information-Geometric Stochastic Particle Assembly (IGSPA)

A framework for simulating interactive particle systems with no underlying geometrical structure, using gillespie SSA with global-bucket optimization, Fisher Information Geometry, and explicit graph topology tracking to derive a statistical distance metric.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-48_passing-brightgreen.svg)](tests/)

---

## Overview

Information-Geometric Stochastic Particle Assembly (IGSPA) simulates the self-assembly of particles with colored binding sites **without relying on spatial coordinates**. The system maintains a **dual geometric representation**:

| Representation | Type | Purpose |
|---|---|---|
| **Implicit Information Manifold** | Continuous | Tracks availability fractions θᵢᶜ with Fisher Information Metric |
| **Explicit Graph Topology** | Discrete | Tracks exact bond structure as a multi-graph |

### Key Innovation: Global Bucket Optimization

Standard Gillespie SSA requires O(N²|C|) pairwise evaluation. NS-IGAA achieves **O(|C| + |E|)** per step by:

1. **Global Buckets**: B_c = {i | particle i has available sites of color c}
2. **Aggregate Propensities**: α_bind^{c,-c} = k_on × (Σᵢ∈B_c availableᵢᶜ) × (Σⱼ∈B₋c availableⱼ₋c)
3. **Two-Stage Sampling**: Channel → Weighted particle selection

---

## Mathematical Foundation

### Information Manifold
Each particle i has availability fractions θᵢᶜ = available/capacity. The manifold uses the Shahshahani form of the Fisher Information Metric:
```
ds² = Σ_c (dθᶜ)² / θᶜ
```
As θ → 0, distances diverge → ∞, geometrically encoding scarcity.

### Explicit Graph Topology
Bonds form a multi-graph G(t) = (V, E(t)) with colored edges. Conservation law:
```
θᵢᶜ(t) = 1 - (Σⱼ Aᵢⱼᶜ(t)) / Kᵢᶜ
```

### Computational Complexity

|| Metric | Standard SSA | IGSPA (Global Bucket) |
||---|---|---|
|| Time/step | O(N²|C|) | **O(|C| + |E|)** |
|| Space | O(N²) | **O(N|C| + |E|)** |
|| Scaling | Quadratic | **Linear/Constant** |

---

**Dependencies**: `numpy>=1.24`, `networkx>=3.0`, `scipy>=1.10`, `lxml>=6.0` (for GraphML export)

---

## Quick Start

```python
from igspa import create_system_from_strings

# Create system from string color lists
system = create_system_from_strings([
    ['white', 'red'],               # Particle 0
    ['green', 'blue'],              # Particle 1
    ['black', 'green', 'yellow'],   # Particle 2
    ['red', 'blue'],                # Particle 3
], k_on=1.5, k_off=0.3)

# Run simulation
stats = system.run(max_steps=100, verbose=True)

# Inspect complexes
system.print_complexes()
```

**Output:**
```
=== COMPLEXES (Time: 7.878s, Step: 12) ===
Complex 1 (Size 1): P0
Complex 2 (Size 3): P1 -- P2 -- P3
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
NS-IGAA maintains two synchronized representations:

1. **Implicit Information Manifold** (Continuous)
   - Tracks availability fractions θᵢᶜ ∈ [0, 1] for each particle and color
   - Equipped with Fisher Information Metric (Shahshahani geometry)
   - As θ → 0, informational distance → ∞ (scarcity geometry)

2. **Explicit Graph Topology** (Discrete)
   - Multi-graph G(t) = (V, E(t)) with colored edges
   - Tracks exact bond structure
   - Enables complex detection via connected components

---

## Features

### Multiple Sampling Algorithms
- **TwoStageSampler** (default): O(|C| + |E|) two-stage Gillespie
- **CompositionRejectionSampler**: O(1) dyadic binning for large systems (N > 50)

### Comprehensive Analysis
- **SteadyStateAnalyzer**: Equilibrium constants, gel fraction, cluster sizes
- **Manifold Geometry**: Geodesic distances, saturation distances, metric condition numbers
- **Thermodynamics**: Entropy, free energy, chemical potentials
- **Ensemble Statistics**: Multi-trajectory statistical analysis

### Full I/O Pipeline
- **Logging**: JSONL streaming, NumPy .npz, Pandas DataFrames
- **Export**: JSON, Pickle, NPZ, GraphML (Gephi/Cytoscape), CSV
- **Checkpointing**: Full state snapshots for resumable simulations

### Flexible Color/Port Systems
- Standard 6-color palette (3 complementary pairs)
- Custom palettes (e.g., DNA: A↔T, C↔G)
- String-based color specification

---

## Architecture

```
src/igspa/
├── __init__.py              # Public API exports
├── core/
│   ├── colors.py            # Color enum, palette, complementarity
│   ├── particle.py          # ParticleBlueprint, Particle, factories
│   └── system.py            # ParticleSystem, SimulationConfig, factory
├── geometry/
│   └── manifold.py          # InformationManifold, FisherMetric
├── topology/
│   └── graph.py             # BondGraph, Bond, ComplexAnalyzer
├── algorithms/
│   └── gillespie.py         # GlobalBucketSampler, TwoStageSampler, CR Sampler
├── io/
│   └── serialization.py     # SimulationLogger, StateExporter
└── utils/
    └── analysis.py          # Validation, SteadyStateAnalyzer, ensemble tools
```

---

## Documentation

- **User Guide**: [USER_GUIDE.md](USER_GUIDE.md) - Complete usage documentation
- **API Reference**: See docstrings in each module
- **Mathematical Details**: This README (sections below)

---

## Detailed Mathematical Specifications

### 1. The Particle & Metric Manifold

Let a particle system be defined by the tuple $\mathcal{S} = (\mathcal{V}, \mathcal{C}, \mathbf{K}, \boldsymbol{\theta})$, where:

* $\mathcal{V} = \{1, \dots, N\}$ is the bounded set of particle vertices.
* $\mathcal{C} = \{c_1, c_2, \dots, c_m\}$ is the alphabet of binding site colors. A bijective mapping function $\text{comp}: \mathcal{C} \to \mathcal{C}$ satisfies comp(c) = -c and comp(-c) = c.
* $\mathbf{K}_i = [K_i^{c_1}, \dots, K_i^{c_m}]^T \in \mathbb{N}^m$ specifies the static total site capacities.
* $\boldsymbol{\theta}_i(t) = [\theta_i^{c_1}(t), \dots, \theta_i^{c_m}(t)]^T \in [0, 1]^m$ details the continuous tracking coordinate on the information manifold $\mathcal{M}_i$, representing the fraction of active, unbonded sites:
$$\theta_i^c(t) = \frac{\text{Unbound Sites of Color } c \text{ on Particle } i}{K_i^c}$$

The distance element on this manifold is governed by the localized Shahshahani form of the Fisher Information Metric Tensor ($g_{\mu\nu}$):
$$ds_i^2 = \sum_{c \in \mathcal{C}} \frac{(d\theta_i^c)^2}{\theta_i^c}$$
This metric guarantees that as a specific color resource approaches exhaustion ($\theta_i^c \to 0$), the informational distance stretches to infinity, accurately mirroring the extreme thermodynamic resistance of rare-state updates.

### 2. The Explicit Graph Geometry

The actual physical structure of the system is a combinatorial multi-graph $\mathcal{G}(t) = (\mathcal{V}, \mathcal{E}(t))$, parameterized by a set of directional adjacency matrices $A^c \in \mathbb{N}^{N \times N}$, tracking active bonds across individual color layers. Conservation laws mandate that:
$$\theta_i^c(t) = 1 - \frac{\sum_{j=1}^N A_{ij}^c(t)}{K_i^c}$$

### 3. Mathematical Optimization: Global Buckets

#### 3.1 Bypassing Quadratic Complexity
Standard stochastic simulation loops over all pairs $\mathcal{O}(N^2)$ to sum individual interaction parameters. This optimization establishes a set of global containers mapping colors to active sets of particle indices:
$$\mathcal{B}_c = \{ i \in \mathcal{V} \mid \theta_i^c > 0 \}$$
The global binding propensity ($\alpha_{\text{bind}}^{c, -c}$) for a color channel is evaluated instantly using aggregate states:
$$\alpha_{\text{bind}}^{c, -c} = k_{\text{on}} \cdot \left( \sum_{i \in \mathcal{B}_c} K_i^c \theta_i^c \right) \cdot \left( \sum_{j \in \mathcal{B}_{-c}} K_j^{-c} \theta_j^{-c} \right)$$

#### 3.2 Dissociation Channels
The global breaking propensity scales linearly with the cardinal volume of the active graph edges:
$$\alpha_{\text{break}} = \sum_{e \in \mathcal{E}} k_{\text{off}} = k_{\text{off}} \cdot |\mathcal{E}|$$
The sum of all active pathways yields the system parameter $a_0 = \sum \alpha_{\text{bind}} + \alpha_{\text{break}}$.

### 4. The Two-Stage Selection Routine

Each simulation iteration proceeds via the following sequence:

1. **Time Delta Evacuation**: Draw next event occurrence interval $\tau = \frac{1}{a_0} \ln\left(\frac{1}{r_1}\right)$, where r₁ ~ Uniform(0,1).
2. **Channel Selection**: Draw a random channel from the combined propensity array $\mathbf{\alpha}$ proportional to its total weight.
3. **Two-Stage Node Resolution** (If Binding Event Chosen):
   * Stage 3a: Extract an index i from bucket $\mathcal{B}_c$ with probability proportional to its localized residual capacity: $P(i) \propto K_i^c \theta_i^c$.
   * Stage 3b: Extract an index j from bucket $\mathcal{B}_{-c}$ with probability proportional to its localized residual capacity: $P(j) \propto K_j^{-c} \theta_j^{-c}$.
4. **State Mutation Execution**: Append an edge to $\mathcal{E}$ connecting (i, j). Update internal parameters $\theta_i^c$ and $\theta_j^{-c}$. If a parameter reaches zero, purge that node index from its corresponding bucket $\mathcal{B}$ in $\mathcal{O}(1)$ time.

---

## Architectural Extensions (Future Work)

### 6.1 Allosteric Coupling (Intra-Particle Interaction Matrix)
To introduce dependencies where a bond formation at one site alters the affinity of another site on the same particle, define an Allosteric Configuration Tensor ($\mathbf{W}_i$). When a binding event modifies coordinate $\theta_i^{c_1}$, it applies an instantaneous linear transformation to another color vector channel:
$$\theta_i^{c_2} \to \theta_i^{c_2} \times \mathbf{W}_i(c_1, c_2)$$
This dynamically updates the global buckets $\mathcal{B}_{c_2}$ without introducing spatial coordinates, allowing the model to simulate cooperative binding and enzyme-like feedback cascades.

### 6.2 Topological Constraints & Ring Closures
While the system is non-spatial, checking graph properties can introduce virtual geometry constraints. For instance, to penalize or favor the formation of closed molecular rings (cycles), the algorithm can check the shortest path distance between candidate nodes i and j in the explicit graph before confirming a bond. If a path already exists, the binding rate can be scaled by a cyclization factor γ:
$$\alpha_{\text{cyclic}} = \alpha_{\text{bind}} \times \gamma(\text{Graph Distance}(i, j))$$
This allows the simulation to capture the physical reality of loop closures and polymer entanglement using pure network topology.

---

## License

MIT License - See [LICENSE](LICENSE) file for details.

---

## Citation

```
Information-Geometric Stochastic Particle Assembly (IGSPA)
Carson Scott, 2024
```
