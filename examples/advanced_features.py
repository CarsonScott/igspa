"""
Example: Advanced NS-IGAA Features

Demonstrates:
- Composition-rejection sampling for O(1) event selection
- Manifold geometry analysis
- State logging and export
- Ensemble runs for statistics
"""

import numpy as np
from igspa import (
    ParticleBlueprint, ParticleSystem, ColorPalette,
    SimulationConfig, SimulationLogger, StateExporter,
    create_system_from_strings
)
from igspa.utils.analysis import (
    SteadyStateAnalyzer, analyze_manifold_geometry, 
    compute_thermodynamic_quantities, run_ensemble, compute_ensemble_statistics
)


def run_with_logging():
    """Run simulation with detailed logging."""
    print("=" * 60)
    print("Advanced Example: Logging & Export")
    print("=" * 60)
    
    system = create_system_from_strings([
        ['white', 'red'],
        ['green', 'blue'],
        ['black', 'green', 'yellow'],
        ['red', 'blue'],
    ], k_on=1.5, k_off=0.3)
    
    # Add logging callback
    log_data = []
    def log_callback(event_type, step, time, *args):
        log_data.append({
            'step': step,
            'time': time,
            'event': event_type,
            'args': args
        })
    
    system.config.callbacks.append(log_callback)
    system.config.log_interval = 5
    
    # Run with logger
    with SimulationLogger(system, 'simulation_log.jsonl') as logger:
        stats = system.run(max_steps=50, verbose=True)
        logger.save_numpy('simulation_data.npz')
    
    # Export final state
    exporter = StateExporter(system)
    exporter.export_state('final_state.json', format='json')
    exporter.export_state('final_state.npz', format='npz')
    exporter.export_for_visualization('final_graph.graphml')
    exporter.export_complexes_csv('complexes.csv')
    exporter.export_time_series_csv('timeseries.csv', logger)
    
    print(f"\nLogged {len(log_data)} events")
    print("Exported: final_state.json, final_state.npz, final_graph.graphml, complexes.csv, timeseries.csv")


def run_with_composition_rejection():
    """Run with composition-rejection sampler for large systems."""
    print("\n" + "=" * 60)
    print("Advanced Example: Composition-Rejection Sampling")
    print("=" * 60)
    
    # Create a larger system
    n_particles = 100
    particle_strings = []
    for i in range(n_particles):
        if i % 4 == 0:
            particle_strings.append(['white', 'red'])
        elif i % 4 == 1:
            particle_strings.append(['black', 'green'])
        elif i % 4 == 2:
            particle_strings.append(['blue', 'yellow'])
        else:
            particle_strings.append(['white', 'blue'])
    
    config = SimulationConfig(
        k_on=1.0,
        k_off=0.1,
        max_steps=1000,
        use_composition_rejection=True,  # Enable CR sampling
        log_interval=200
    )
    
    system = create_system_from_strings(particle_strings, k_on=1.0, k_off=0.1, use_composition_rejection=True, log_interval=200)
    
    print(f"System: {system.n_particles} particles, {len(system.palette)} colors")
    print(f"Using Composition-Rejection: {system.config.use_composition_rejection}")
    
    stats = system.run(max_steps=500, verbose=True)
    
    print(f"\nPerformance: {stats['steps_per_second']:.0f} steps/sec")
    print(f"Final bonds: {stats['final_bonds']}")


def run_ensemble_analysis():
    """Run ensemble for statistical analysis."""
    print("\n" + "=" * 60)
    print("Advanced Example: Ensemble Statistics")
    print("=" * 60)
    
    def system_factory():
        return create_system_from_strings([
            ['white', 'red'],
            ['green', 'blue'],
            ['black', 'green', 'yellow'],
            ['red', 'blue'],
        ], k_on=1.5, k_off=0.3)
    
    # Run 20 independent trajectories
    print("Running 20 independent trajectories...")
    ensemble_results = run_ensemble(system_factory, n_runs=20, max_steps=100, verbose=False)
    
    # Compute ensemble statistics
    ensemble_stats = compute_ensemble_statistics(ensemble_results)
    
    print("\nEnsemble Statistics (20 runs, 100 steps each):")
    for key, stat in ensemble_stats.items():
        print(f"  {key}: mean={stat['mean']:.2f}, std={stat['std']:.2f}, "
              f"min={stat['min']:.2f}, max={stat['max']:.2f}")


def analyze_manifold():
    """Analyze information manifold geometry."""
    print("\n" + "=" * 60)
    print("Advanced Example: Manifold Geometry Analysis")
    print("=" * 60)
    
    system = create_system_from_strings([
        ['white', 'red'],
        ['green', 'blue'],
        ['black', 'green', 'yellow'],
        ['red', 'blue'],
    ], k_on=1.5, k_off=0.3)
    
    system.run(max_steps=50, verbose=False)
    
    # Analyze manifold geometry
    geom = analyze_manifold_geometry(system)
    
    print("\nDistances to saturation (higher = more exhausted):")
    for label, dist in geom['distances_to_saturation'].items():
        print(f"  {label}: {dist:.4f}")
    
    print("\nPairwise geodesic distances:")
    for (l1, l2), dist in geom['pairwise_distances'].items():
        print(f"  {l1} <-> {l2}: {dist:.4f}")
    
    print("\nMetric condition numbers:")
    for label, cond in geom['metric_condition_numbers'].items():
        print(f"  {label}: {cond:.2f}")


def custom_palette_example():
    """Example with custom color palette."""
    print("\n" + "=" * 60)
    print("Advanced Example: Custom Color Palette")
    print("=" * 60)
    
    # Define custom 4-color, 2-pair palette (e.g., DNA-like: A-T, C-G)
    custom_pairs = {
        'A': 'T',
        'C': 'G',
    }
    
    palette = ColorPalette.from_strings(custom_pairs)
    print(f"Custom palette: {palette.colors}")
    print(f"Channels: {palette.get_channels()}")
    
    # Create system with custom palette
    system = create_system_from_strings([
        ['A', 'C'],   # Particle 0: A and C sites
        ['T', 'G'],   # Particle 1: T and G sites
        ['A', 'A'],   # Particle 2: Two A sites
        ['T', 'C'],   # Particle 3: T and C sites
    ], k_on=2.0, k_off=0.5, palette=palette)
    
    system.run(max_steps=30, verbose=True)
    system.print_complexes()


if __name__ == '__main__':
    run_with_logging()
    run_with_composition_rejection()
    run_ensemble_analysis()
    analyze_manifold()
    custom_palette_example()