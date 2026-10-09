"""
Example: Basic NS-IGAA Simulation

This reproduces the original example.py functionality with the new modular API.
"""

from igspa import (
    ParticleBlueprint, ParticleSystem, ColorPalette,
    create_system_from_strings
)

# Create the standard test system (same as original example.py)
system = create_system_from_strings([
    ['white', 'red'],               # P0
    ['green', 'blue'],              # P1
    ['black', 'green', 'yellow'],   # P2
    ['red', 'blue']                 # P3
], k_on=1.5, k_off=0.3)

print("NS-IGAA Simulation - Standard Test System")
print("=" * 50)
print(f"Particles: {system.n_particles}")
for p in system.particles:
    print(f"  {p}")
print()

# Run simulation
stats = system.run(max_steps=12, verbose=True)

# Print final complexes
system.print_complexes()

# Additional analysis
from igspa.utils.analysis import SteadyStateAnalyzer, compute_thermodynamic_quantities

analyzer = SteadyStateAnalyzer(system)
print("\nEquilibrium Constants:")
for channel, k_eq in analyzer.estimate_equilibrium_constants().items():
    c1, c2 = channel
    print(f"  {c1.name}-{c2.name}: K_eq ≈ {k_eq:.4f}")

print(f"\nGel fraction: {analyzer.compute_gel_fraction():.4f}")
print(f"Mean cluster size: {analyzer.compute_mean_cluster_size():.4f}")

print("\nThermodynamic Quantities:")
thermo = compute_thermodynamic_quantities(system)
for key, value in thermo.items():
    print(f"  {key}: {value}")