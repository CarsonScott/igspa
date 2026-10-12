"""
Example: Real-time visualization of ParticleSystem using NetworkX/Matplotlib.

This demonstrates the new visualization module with:
- FuncAnimation-based real-time animation
- ForceAtlas2 layout updating every N steps
- Persistent node positions with edge-only updates
- Color-coded particles and port-specific edges
"""

from igspa import SystemConfiguration, ParticleSystem
from igspa.visualization import animate_system, run_headless_simulation, plot_static_snapshot
import numpy as np
import matplotlib.pyplot as plt


def main():
    # Create a test system using the standard configuration
    config = SystemConfiguration.standard()
    
    system = ParticleSystem.create_from_config(
        config,
        n_particles=50,
        p=None,
        k_on=3.0,
        k_off=0.2,
        seed=42
    )
    
    print(f"Created system with {system.n_particles} particles")
    print(f"Ports: {system.palette.ports}")
    print()
    
    # Run a few steps first to build up some bonds
    system.run(max_steps=20)
    
    # Option 1: Real-time animation
    print("Starting real-time animation...")
    print("Close the window to continue to headless example.")
    
    anim = animate_system(
        system,
        max_frames=300,
        layout_update_interval=10,
        animation_interval=80,
        figsize=(14, 10),
        node_size_by="degree",
        forceatlas2_iterations=30,
        forceatlas2_scaling=2.0,
        forceatlas2_gravity=0.5,
        forceatlas2_spring_strength=0.2,
    )
    
    # Option 2: Headless simulation with custom callback
    print("\nRunning headless simulation...")
    
    def my_callback(step, positions, frame_data):
        if step % 50 == 0:
            print("  Step {}: t={:.3f}, bonds={}, complexes={}".format(
                step, frame_data['time'], frame_data['n_bonds'], len(frame_data['complexes'])
            ))
    
    frames = run_headless_simulation(
        system,
        max_steps=200,
        layout_update_interval=10,
        callback=my_callback,
    )
    
    print("\nCollected {} frames".format(len(frames)))
    print("Final state: step={}, time={:.3f}, bonds={}".format(
        frames[-1]['step'], frames[-1]['time'], frames[-1]['n_bonds']
    ))
    
    # Option 3: Static snapshot
    print("\nCreating static snapshot...")
    fig, ax = plot_static_snapshot(system, layout_iterations=100)
    plt.savefig("igspa_snapshot.png", dpi=150, bbox_inches='tight')
    print("Saved snapshot to igspa_snapshot.png")
    plt.show()


if __name__ == "__main__":
    main()