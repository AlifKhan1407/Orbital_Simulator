"""
Orbital Dynamics & N-Body Simulator
-----------------------------------
Vectorized numerical integration for planetary orbits around a central mass,
implementing standard Euler and Euler-Cromer numerical schemes.
"""

from typing import Tuple, Dict, Optional
import numpy as np

# Astronomical & Physical Constants
G: float = 6.6743e-11       # Gravitational constant [m^3 kg^-1 s^-2]
M_SOLAR: float = 1.989e30   # Solar mass [kg]
AU: float = 1.496e11        # Astronomical Unit [m]
YEAR: float = 3.154e7       # Seconds in one year [s]


def compute_accelerations(
    positions: np.ndarray,
    masses: np.ndarray,
    m_star: float = M_SOLAR,
    G_const: float = G
) -> np.ndarray:
    """
    Compute 2D acceleration vectors for N bodies due to central star and mutual gravity.

    Parameters
    ----------
    positions : np.ndarray
        Shape (N, 2) array of body positions [[x1, y1], [x2, y2], ...].
    masses : np.ndarray
        Shape (N,) array of body masses.
    m_star : float, default=M_SOLAR
        Mass of central star fixed at origin (0, 0).
    G_const : float, default=G
        Gravitational constant.

    Returns
    -------
    np.ndarray
        Shape (N, 2) acceleration vectors for each body [m/s^2].
    """
    positions = np.atleast_2d(positions)
    masses = np.atleast_1d(masses)
    num_bodies = positions.shape[0]

    # 1. Acceleration from central star at origin (0, 0)
    r_star = -positions  # Vector pointing from planet toward star
    dist_star = np.linalg.norm(positions, axis=1, keepdims=True)
    a_star = (G_const * m_star / (dist_star**3)) * r_star

    if num_bodies == 1:
        return a_star

    # 2. Pairwise acceleration between planets via array broadcasting
    # r_ij[i, j] = pos[j] - pos[i] (vector pointing from planet i to planet j)
    r_ij = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    dist_ij = np.linalg.norm(r_ij, axis=-1)

    # Prevent division by zero along diagonal (self-interaction)
    np.fill_diagonal(dist_ij, np.inf)

    # Pairwise acceleration on planet i from planet j: G * m_j / |r_ij|^3 * r_ij
    a_mutual = np.sum(
        (G_const * masses[np.newaxis, :, np.newaxis] / (dist_ij[:, :, np.newaxis] ** 3)) * r_ij,
        axis=1
    )

    return a_star + a_mutual


def simulate_nbody(
    initial_positions: np.ndarray,
    initial_velocities: np.ndarray,
    masses: np.ndarray,
    m_star: float = M_SOLAR,
    dt: float = YEAR / 25000,
    t_max: float = 5 * YEAR,
    method: str = "euler_cromer"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate trajectory history of N planetary bodies over a given time horizon.

    Parameters
    ----------
    initial_positions : np.ndarray
        Shape (N, 2) initial position vectors [m].
    initial_velocities : np.ndarray
        Shape (N, 2) initial velocity vectors [m/s].
    masses : np.ndarray
        Shape (N,) masses of planets [kg].
    m_star : float
        Mass of central star [kg].
    dt : float
        Integration time step [s].
    t_max : float
        Total duration of simulation [s].
    method : {"euler", "euler_cromer"}, default="euler_cromer"
        Numerical integration method. 'euler_cromer' conserves orbital energy.

    Returns
    -------
    trajectories : np.ndarray
        Shape (N, 2, num_steps) position coordinates over time.
    time_array : np.ndarray
        Shape (num_steps,) time values [s].
    """
    pos = np.array(initial_positions, dtype=np.float64, copy=True)
    vel = np.array(initial_velocities, dtype=np.float64, copy=True)
    masses = np.array(masses, dtype=np.float64)

    if pos.ndim == 1:
        pos = pos[np.newaxis, :]
        vel = vel[np.newaxis, :]

    num_steps = int(np.ceil(t_max / dt))
    num_bodies = pos.shape[0]

    trajectories = np.zeros((num_bodies, 2, num_steps), dtype=np.float64)
    time_array = np.linspace(0, t_max, num_steps)

    for t_idx in range(num_steps):
        trajectories[:, :, t_idx] = pos
        accel = compute_accelerations(pos, masses, m_star=m_star)

        if method == "euler":
            # Forward Euler: pos updated with current velocity
            pos_next = pos + vel * dt
            vel_next = vel + accel * dt
            pos, vel = pos_next, vel_next
        elif method == "euler_cromer":
            # Semi-Implicit Euler: vel updated first, then used for pos
            vel += accel * dt
            pos += vel * dt
        else:
            raise ValueError(f"Unknown integration method: '{method}'. Choose 'euler' or 'euler_cromer'.")

    return trajectories, time_array