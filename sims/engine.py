import numpy as np
from collections import defaultdict

def run_monte_carlo(
    initial_state_fn,
    step_fn,
    times,
    n_paths,
    events,
    seed=None,
):
    """
    events: dict
        { time: { event_name: function(state) } }
    """

    rng = np.random.default_rng(seed)
    state = initial_state_fn(n_paths=n_paths)

    results = defaultdict(list)

    for i in range(1, len(times)):
        dt = times[i] - times[i - 1]
        t = times[i]

        state = step_fn(state=state, dt=dt, rng=rng)

        if t in events:
            for name, func in events[t].items():
                results[name].append(func(state))

    return results
