#!/usr/bin/env python
import numpy as np
from scipy import stats
from tqdm import tqdm
# MCMC MH sampler code. Lifted from
# https://github.com/tdhopper/mcmc/blob/master/Metropolis-Hastings%20Algorithm.ipynb


def transition(current_state, logpdf, dim, scaling):
    transition = stats.multivariate_normal.rvs(size=dim) * scaling
    transition = transition.squeeze()
    candidate = current_state + transition
    prev_log_likelihood = logpdf(current_state)
    candidate_log_likelihood = logpdf(candidate)
    if np.isinf(candidate_log_likelihood).any():
        return current_state
    diff = candidate_log_likelihood - prev_log_likelihood
    uniform_draw = np.log(stats.uniform(0, 1).rvs())
    return candidate if uniform_draw < diff else current_state


def generate_samples(initial_state, num_iterations, logpdf, scaling=0.01):
    """
    Generate samples using the MCMC MH algorithm.

    Args:
        initial_state (array-like): Initial state for the MCMC chain.
        num_iterations (int): Number of iterations to perform.
        logpdf (callable): A function that computes the log probability
                            density.

    Yields:
        array-like: The next state in the MCMC chain.
    """
    current_state = initial_state
    dim = 1 if isinstance(current_state, (float, int)) else len(current_state)

    # Wrap the loop in tqdm to show progress
    for _ in tqdm(range(num_iterations), desc="Sampling"):
        current_state = transition(current_state, logpdf, dim, scaling)
        yield current_state
