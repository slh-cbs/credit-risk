import numpy as np

def initial_state(n_paths, S0):
    return np.full(n_paths, S0)

def step(state, dt, mu, sigma, rng):
    z = rng.standard_normal(state.shape)
    drift = (mu - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * z
    return state * np.exp(drift + diffusion)
