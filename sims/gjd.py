import numpy as np

def initial_state(n_paths, S0):
    return np.full(n_paths, S0)

# --- Jump distributions ---

def lognormal_jump_dist(gamma, delta):
    # returns Y = log(1+eps) ~ N(gamma - 0.5*delta**2, delta**2)
    # mean is k = E[e^Y] - 1
    return {
        "sample": lambda n, rng: rng.normal(gamma - 0.5*delta**2, delta, size=n),
        "mean": lambda: np.exp(gamma) - 1,
    }


def constant_jump_dist(size):
    # size must take values in (-1, inf)
    # returns Y = log(1.0 + size)
    # mean is k = e^Y - 1 = size
    return {
        "sample": lambda n, rng: np.full(n, np.log(1.0 + size)),
        "mean": lambda: size,
    }


def double_exponential_jump_dist(p, eta1, eta2):
    # p must be in (0, 1)
    # eta1 > 1
    # eta2 > 0
    # returns Y = log(1+eps) ~ double exponential distributed
    # mean is k = E[e^Y] - 1
    return {
        "sample": lambda n, rng: np.where(
            rng.uniform(size=n) < p,
            rng.exponential(1/eta1, size=n),
            -rng.exponential(1/eta2, size=n)
        ),
        "mean": lambda: (
            p * eta1 / (eta1 - 1)
            + (1 - p) * eta2 / (eta2 + 1)
            - 1
        ),
    }

# --- Step function ---

def step(state, dt, mu_total, sigma, lambda_, jump_dist, rng):
    n_paths = state.shape[0]

    # --- Drift compensation ---
    k = jump_dist["mean"]()
    mu_sde = mu_total - lambda_ * k

    # --- Diffusion ---
    z = rng.standard_normal(n_paths)
    diffusion = np.exp(
        (mu_sde - 0.5 * sigma**2) * dt
        + sigma * np.sqrt(dt) * z
    )

    # --- Jumps ---
    n_jumps = rng.poisson(lambda_ * dt, size=n_paths)
    jump_sum = np.zeros(n_paths)

    jump_paths = n_jumps > 0
    if np.any(jump_paths):
        total_jumps = np.sum(n_jumps)
        logJ = jump_dist["sample"](total_jumps, rng)

        idx = 0
        for i in np.where(jump_paths)[0]:
            n = n_jumps[i]
            jump_sum[i] = np.sum(logJ[idx:idx+n])
            idx += n

    return state * diffusion * np.exp(jump_sum)