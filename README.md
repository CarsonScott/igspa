# Information-Geometric Stochastic Particle Assembly (IGSPA)

A framework for simulating interactive particle systems with no underlying geometrical structure, using gillespie SSA with global-bucket optimization, Fisher Information Geometry, and explicit graph topology tracking to derive a statistical distance metric by which to calculate interaction potentials.

Useful for modeling biomolecular dynamics in cellular simulations.

[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)](tests/)

---

## Documentation

- **User Guide (recommended)**: [USER_GUIDE.md](USER_GUIDE.md) 
- **API Reference**: See docstrings in each module
- **Mathematical Details**: This README (sections below)

---

## Overview

Information-Geometric Stochastic Particle Assembly (IGSPA) simulates the self-assembly of particles with labeled binding ports **without relying on spatial coordinates**. The system maintains a **dual geometric representation**:

| Representation | Type | Purpose |
|---|---|---|
| **Implicit Information Manifold** | Continuous | Tracks availability fractions θᵢₚ with Fisher Information Metric |
| **Explicit Graph Topology** | Discrete | Tracks exact bond structure as a multi-graph |

### Key Innovation: Global Bucket Optimization

Standard Gillespie SSA requires O(N²|P|) pairwise evaluation. NS-IGAA achieves **O(|P| + |E|)** per step by:

1. **Global Buckets**: Bₚ = {i | particle i has available sites of port p}
2. **Aggregate Propensities**: α_bind^{p,q} = k_on × (Σᵢ∈Bₚ availableᵢₚ) × (Σⱼ∈B_q availableⱼ_q) for valid pair (p,q)
3. **Two-Stage Sampling**: Valid Channel → Weighted particle selection

---

## Mathematical Foundation

### Information Manifold

Each particle i has availability fractions θᵢₚ = available/capacity. The manifold uses the Shahshahani form of the Fisher Information Metric:
```
ds² = Σ_p (dθᵖ)² / θᵖ
```
As θ → 0, distances diverge → ∞, geometrically encoding scarcity.

### Explicit Graph Topology

Bonds form a multi-graph G(t) = (V, E(t)) with labeled edges. Conservation law:
```
θᵢₚ(t) = 1 - (Σⱼ Aᵢⱼₚ(t)) / Kᵢₚ
```

### Computational Complexity

| Metric | Standard SSA | IGSPA (Global Bucket) |
|---|---|---|
| Time/step | O(N²|P|) | **O(|P| + |E|)** |
| Space | O(N²) | **O(N|P| + |E|)** |
| Scaling | Quadratic | **Linear/Constant** |

---

## Dependencies
`numpy>=1.24`, `networkx>=3.0`, `scipy>=1.10`, `lxml>=6.0` (for GraphML export)

---

## Quick Start

```python
from igspa import create_system_from_configuration

# Create system from port lists
system = create_system_from_configuration(
    n_particles=4,
    config=SystemConfiguration.from_spec(
        name='test',
        port_names={'white', 'black', 'red', 'green', 'blue', 'yellow'},
        pair_specs={('white','black'), ('red','green'), ('blue','yellow')},
        archetype_specs=[
            ['white', 'red'],           # Particle 0
            ['green', 'blue'],          # Particle 1
            ['black', 'green', 'yellow'], # Particle 2
            ['red', 'blue'],            # Particle 3
        ]
    ),
    k_on=1.5, k_off=0.3
)

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

Each particle has **binding sites** of specific **ports**. Valid binding occurs between specific port pairs:
- `white` ↔ `black`
- `red` ↔ `green`  
- `blue` ↔ `yellow`

A particle can bind to another if it has an available site of a port p and the other particle has an available site of port q, where (p,q) is a valid pair.

### Dual Representation

IGSPA maintains two synchronized representations:

1. **Implicit Information Manifold** (Continuous)
   - Tracks availability fractions θᵢₚ ∈ [0, 1] for each particle and port
   - Equipped with Fisher Information Metric (Shahshahani geometry)
   - As θ → 0, informational distance → ∞ (scarcity geometry)

2. **Explicit Graph Topology** (Discrete)
   - Multi-graph G(t) = (V, E(t)) with labeled edges
   - Tracks exact bond structure
   - Enables complex detection via connected components

---

## Features

### Multiple Sampling Algorithms
- **TwoStageSampler** (default): O(|P| + |E|) two-stage Gillespie
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

### Flexible Port Systems
- Standard 6-port palette (3 valid pairs)
- Custom palettes (e.g., DNA-like: A↔T, C↔G)
- String-based port specification
- Weighted archetype sampling

---

## Architecture

```
src/igspa/
├── __init__.py              # Public API exports
├── core/
│   ├── ports.py             # Port palette, valid pairs
│   ├── particle.py          # ParticleArchetype, Particle, factories
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

## Detailed Mathematical Specifications

### 1. The Particle & Metric Manifold

Let a particle system be defined by the tuple $\mathcal{S} = (\mathcal{V}, \mathcal{P}, \mathbf{K}, \boldsymbol{\theta})$, where:

* $\mathcal{V} = \{1, \dots, N\}$ is the bounded set of particle vertices.
* $\mathcal{P} = \{p_1, p_2, \dots, p_m\}$ is the alphabet of binding ports. A set $\mathcal{V}_{pairs} \subseteq \mathcal{P} \times \mathcal{P}$ specifies valid binding pairs (unordered, no self-pairs unless explicitly allowed).
* $\mathbf{K}_i = [K_i^{p_1}, \dots, K_i^{p_m}]^T \in \mathbb{N}^m$ specifies the static total site capacities.
* $\boldsymbol{\theta}_i(t) = [\theta_i^{p_1}(t), \dots, \theta_i^{p_m}(t)]^T \in [0, 1]^m$ details the continuous tracking coordinate on the information manifold $\mathcal{M}_i$, representing the fraction of active, unbonded sites:

$$\theta_i^p(t) = \frac{\text{Unbound Sites of Port } p \text{ on Particle } i}{K_i^p}$$

The distance element on this manifold is governed by the localized Shahshahani form of the Fisher Information Metric Tensor ($g_{\mu\nu}$):
$$ds_i^2 = \sum_{p \in \mathcal{P}} \frac{(d\theta_i^p)^2}{\theta_i^p}$$
This metric guarantees that as a specific port resource approaches exhaustion ($\theta_i^p \to 0$), the informational distance stretches to infinity, accurately mirroring the extreme thermodynamic resistance of rare-state updates.

### 2. The Explicit Graph Geometry

The actual physical structure of the system is a combinatorial multi-graph $\mathcal{G}(t) = (\mathcal{V}, \mathcal{E}(t))$, parameterized by a set of directional adjacency matrices $A^{p,q} \in \mathbb{N}^{N \times N}$, tracking active bonds across individual port layers for each valid pair $(p,q) \in \mathcal{V}_{pairs}$. Conservation laws mandate that:

$$\theta_i^p(t) = 1 - \frac{\sum_{j=1}^N \sum_{q: (p,q)\in\mathcal{V}_{pairs}} A_{ij}^{p,q}(t)}{K_i^p}$$

### 3. Mathematical Optimization: Global Buckets

#### 3.1 Bypassing Quadratic Complexity

Standard stochastic simulation loops over all pairs $\mathcal{O}(N^2)$ to sum individual interaction parameters. This optimization establishes a set of global containers mapping ports to active sets of particle indices:

$$\mathcal{B}_p = \{ i \in \mathcal{V} \mid \theta_i^p > 0 \}$$

The global binding propensity ($\alpha_{\text{bind}}^{p,q}$) for a valid port pair is evaluated instantly using aggregate states:

$$\alpha_{\text{bind}}^{p,q} = k_{\text{on}} \cdot \left( \sum_{i \in \mathcal{B}_p} K_i^p \theta_i^p \right) \cdot \left( \sum_{j \in \mathcal{B}_q} K_j^q \theta_j^q \right)$$

#### 3.2 Dissociation Channels

The global breaking propensity scales linearly with the cardinal volume of the active graph edges:

$$\alpha_{\text{break}} = \sum_{e \in \mathcal{E}} k_{\text{off}} = k_{\text{off}} \cdot |\mathcal{E}|$$

The sum of all active pathways yields the system parameter $a_0 = \sum \alpha_{\text{bind}} + \alpha_{\text{break}}$.

#### 4. The Two-Stage Selection Routine

Each simulation iteration proceeds via the following sequence:

1. **Time Delta Evacuation**: Draw next event occurrence interval $\tau = \frac{1}{a_0} \ln\left(\frac{1}{r_1}\right)$, where r₁ ~ Uniform(0,1).
2. **Channel Selection**: Draw a random valid channel from the combined propensity array $\mathbf{\alpha}$ proportional to its total weight.
3. **Two-Stage Node Resolution** (If Binding Event Chosen):
   * Stage 3a: Extract an index i from bucket $\mathcal{B}_p$ with probability proportional to its localized residual capacity: $P(i) \propto K_i^p \theta_i^p$.
   * Stage 3b: Extract an index j from bucket $\mathcal{B}_q$ with probability proportional to its localized residual capacity: $P(j) \propto K_j^q \theta_j^q$.
4. **State Mutation Execution**: Append an edge to $\mathcal{E}$ connecting (i, j). Update internal parameters $\theta_i^p$ and $\theta_j^q$. If a parameter reaches zero, purge that node index from its corresponding bucket $\mathcal{B}$ in $\mathcal{O}(1)$ time.

---

## Architectural Extensions (Future Work)

### 6.1 Allosteric Coupling (Intra-Particle Interaction Matrix)

To introduce dependencies where a bond formation at one port alters the affinity of another port on the same particle, define an Allosteric Configuration Tensor ($\mathbf{W}_i$). When a binding event modifies coordinate $\theta_i^{p_1}$, it applies an instantaneous linear transformation to another port vector channel:

$$\theta_i^{p_2} \to \theta_i^{p_2} \times \mathbf{W}_i(p_1, p_2)$$

This dynamically updates the global buckets $\mathcal{B}_{p_2}$ without introducing spatial coordinates, allowing the model to simulate cooperative binding and enzyme-like feedback cascades.

### 6.2 Topological Constraints & Ring Closures

While the system is non-spatial, checking graph properties can introduce virtual geometry constraints. For instance, to penalize or favor the formation of closed molecular rings (cycles), the algorithm can check the shortest path distance between candidate nodes i and j in the explicit graph before confirming a bond. If a path already exists, the binding rate can be scaled by a cyclization factor $\gamma$:

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