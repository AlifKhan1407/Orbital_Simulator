# N-Body Planetary Orbit Simulator (`physics-numerical-methods`)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Vectorized N-body planetary orbit simulator implementing Forward Euler and Symplectic Euler-Cromer integration schemes.**

## Context & Acknowledgments

This project was originally developed as part of the UCL Practical Physics (`PHAS0007`) computational modeling curriculum, investigating numerical approximations of gravitational dynamics (awarded 74/100, First Class).

## Overview

This module simulates gravitational dynamics for multi-body systems in two dimensions around a central stellar mass. Developed as part of the UCL Practical Physics (`PHAS0007`) computational modeling curriculum[cite: 6].

### Physics & Numerical Background

The gravitational force exerted on planet $i$ by a central star $M$ and all other planets $j$ is given by:

$$
\mathbf{F}_i = -\frac{G M m_i}{\vert{}\mathbf{r}_i\vert{}^3} \mathbf{r}_i + \sum_{j \neq i} \frac{G m_i m_j}{\vert{}\mathbf{r}_j - \mathbf{r}_i\vert{}^3} (\mathbf{r}_j - \mathbf{r}_i)
$$

The simulator implements two time-stepping methods:
1. **Standard Forward Euler (Explicit):**
   
   $$
   \mathbf{r}_{n+1} = \mathbf{r}_n + \mathbf{v}_n \Delta t
   $$
   
   $$
   \mathbf{v}_{n+1} = \mathbf{v}_n + \mathbf{a}_n \Delta t
   $$
   
   *Note: Subject to unphysical energy amplification over extended orbital periods.*

2. **Euler-Cromer (Semi-Implicit / Symplectic):**
   
   $$
   \mathbf{v}_{n+1} = \mathbf{v}_n + \mathbf{a}_n \Delta t
   $$
   
   $$
   \mathbf{r}_{n+1} = \mathbf{r}_n + \mathbf{v}_{n+1} \Delta t
   $$
   
   *Note: Preserves phase-space area and eliminates spiral energy drift.*

---

## Project Architecture

```text
nbody-orbital-sim/
├── src/
│   ├── __init__.py
│   └── orbital_sim.py       # Core vectorized solver & constants
├── run_simulation.py        # Demo script for 1-body, 2-body, and N-body runs
├── README.md                # Project documentation
└── requirements.txt         # Dependencies (numpy, matplotlib)
