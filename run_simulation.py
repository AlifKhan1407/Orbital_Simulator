"""
Executable Entry Point for Planetary Orbit Simulations
"""

import matplotlib.pyplot as plt
import numpy as np
from src.orbital_sim import simulate_nbody, AU, YEAR, M_SOLAR, G


def plot_trajectories(
    trajectories: np.ndarray,
    labels: list[str],
    title: str,
    filename: str
) -> None:
    """Helper function to format and save trajectory plots in astronomical units."""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Draw central star
    ax.plot(0, 0, 'yo', markersize=12, label="Sun")

    num_bodies = trajectories.shape[0]
    for i in range(num_bodies):
        x_au = trajectories[i, 0, :] / AU
        y_au = trajectories[i, 1, :] / AU
        ax.plot(x_au, y_au, label=labels[i], linewidth=1.2)

    ax.set_aspect('equal', 'box')
    ax.set_xlabel("x [AU]")
    ax.set_ylabel("y [AU]")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right")
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close(fig)
    print(f"Saved plot: {filename}")


def run_single_planet_experiment() -> None:
    """Simulate Earth orbit under standard Euler vs Euler-Cromer."""
    v_earth = np.sqrt(G * M_SOLAR / AU)
    pos_earth = np.array([[AU, 0.0]])
    vel_earth = np.array([[0.0, v_earth]])
    mass_earth = np.array([5.972e24])

    dt = YEAR / 25000
    t_max = 5 * YEAR

    # Run Standard Euler
    traj_euler, _ = simulate_nbody(
        pos_earth, vel_earth, mass_earth, dt=dt, t_max=t_max, method="euler"
    )
    plot_trajectories(traj_euler, ["Earth (Standard Euler)"], "5-Year Earth Orbit (Forward Euler)", "earth_forward_euler.png")

    # Run Euler-Cromer (Energy Preserving)
    traj_cromer, _ = simulate_nbody(
        pos_earth, vel_earth, mass_earth, dt=dt, t_max=t_max, method="euler_cromer"
    )
    plot_trajectories(traj_cromer, ["Earth (Euler-Cromer)"], "5-Year Earth Orbit (Euler-Cromer)", "earth_euler_cromer.png")


def run_two_planet_interaction() -> None:
    """Simulate Earth and Venus interaction under perturbed mass scenario."""
    v_earth = np.sqrt(G * M_SOLAR / AU)
    r_venus = 0.72 * AU
    v_venus = np.sqrt(G * M_SOLAR / r_venus)

    pos = np.array([[AU, 0.0], [r_venus, 0.0]])
    vel = np.array([[0.0, v_earth], [0.0, v_venus]])

    m_earth = 5.972e24
    m_venus_perturbed = m_earth * 500
    masses = np.array([m_earth, m_venus_perturbed])

    dt = YEAR / 25000
    t_max = 5 * YEAR

    trajectories, _ = simulate_nbody(pos, vel, masses, dt=dt, t_max=t_max, method="euler_cromer")
    plot_trajectories(
        trajectories,
        ["Earth", "Perturbed Venus (500x Mass)"],
        "Two-Planet Interaction: Orbital Deviation under Massive Companion",
        "two_planet_interaction.png"
    )


def run_nbody_solar_system() -> None:
    """Simulate multi-body system (Earth, Mars, Jupiter) over 12 years."""
    names = ["Earth", "Jupiter", "Mars"]
    masses = np.array([5.97e24, 1.90e27, 6.39e23])
    pos = np.array([
        [1.5e11, 0.0],
        [7.7e11, 0.0],
        [2.2e11, 0.0]
    ])
    vel = np.array([
        [0.0, 29780.0],
        [0.0, 13070.0],
        [0.0, 24000.0]
    ])

    dt = YEAR / 25000
    t_max = 12 * YEAR

    trajectories, _ = simulate_nbody(pos, vel, masses, dt=dt, t_max=t_max, method="euler_cromer")
    plot_trajectories(trajectories, names, "12-Year Multi-Body Solar System Simulation", "solar_system_12yr.png")


if __name__ == "__main__":
    print("Running Single Planet Experiment...")
    run_single_planet_experiment()
    
    print("Running Two-Planet Interaction Experiment...")
    run_two_planet_interaction()

    print("Running 12-Year Solar System Simulation...")
    run_nbody_solar_system()