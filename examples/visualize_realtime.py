"""
Example: Real-time visualization of ParticleSystem using NetworkX/Matplotlib.

This demonstrates the new visualization module with:
- FuncAnimation-based real-time animation
- NetworkX spring layout updating every N steps
- Persistent node positions with edge-only updates
- Color-coded particles and port-specific edges
"""

from igspa import IGSPAVisualizer, create_system
from igspa.visualization import run_headless_simulation, plot_static_snapshot
import matplotlib.pyplot as plt


def main():
    system_definition = {
        "name": "live-demo",
        "palette": [("a", "b"), ("b", "c"), ("c", "a"), ("b", "d")],
        "archetypes": ["aab", "abb", "bbc", "bcc", "dab"],
        "n_particles": 50,
        "k_on": 0.1,
        "k_off": 0.01,
        "seed": 42,
    }
    system = create_system(system_definition)
    
    print(f"Created system with {system.n_particles} particles")
    print(f"Ports: {system.palette.ports}")
    print()
    
    # Option 1: Real-time animation
    print("Starting real-time animation...")
    print("Close the window to continue to headless example.")
    visualizer = IGSPAVisualizer(system, speed=1, steps_per_frame=1)
    visualizer.run(max_frames=300)
    
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
        layout_update_interval=1,
        callback=my_callback,
    )
    
    print("\nCollected {} frames".format(len(frames)))
    print("Final state: step={}, time={:.3f}, bonds={}".format(
        frames[-1]['step'], frames[-1]['time'], frames[-1]['n_bonds']
    ))
    
    # Option 3: Static snapshot
    print("\nCreating static snapshot...")
    fig, ax = plot_static_snapshot(system, layout_iterations=100)
    # plt.savefig("igspa_snapshot.png", dpi=150, bbox_inches='tight')
    # print("Saved snapshot to igspa_snapshot.png")
    plt.show()


if __name__ == "__main__":
    main()
