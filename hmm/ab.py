import numpy as np
from scipy.special import logsumexp


def _curr_u(n_samples, u, t, j, d):
    """
    Provides the current value of u checking whether the t-d is non-negative,
    t is less than n_samples, and d is greate than or equal to t - (n_samples - 1).
    Utilized by the ``_forward`` auxiliary function.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    u : np.ndarray
        Array of shape (n_samples, n_states, n_durations) containing the u values as produced by the ``_u_only`` function.
    t : int
        Current time step.
    j : int
        Current state.
    d : int
        Current duration.

    Returns
    -------
    curr_u : float
        Current value of u.

    See Also
    --------
    _forward : Function that computes the forward variable.
    base.HSMM._core_u_only : Method that computes the u values.
    """
    if t - d >= 0 and t < n_samples:
        return u[t, j, d]
    elif t - d < 0:
        return u[t, j, t]
    elif d >= t - (n_samples - 1):
        return u[n_samples - 1, j, (n_samples - 1) + d - t]
    else:
        return 0.0


def _forward(n_samples, n_states, n_durations,
             log_startprob,
             log_transmat,
             log_durprob,
             left_censor, right_censor,
             eta, u, xi):
    """
    Computes the forward variable alpha needed for the likelihood computation and the parameters re-estimation.
    Utilized by the ``HSMM._core_forward`` method.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_states : int
        Number of states.
    n_durations : int
        Number of durations.
    log_startprob : np.ndarray
        Array of shape (n_states,) containing the log of the initial state probabilities.
    log_transmat : np.ndarray
        Array of shape (n_states, n_states) containing the log of the transition probabilities.
    log_durprob : np.ndarray
        Array of shape (n_states, n_durations) containing the log of the duration probabilities.
    left_censor : int
        0 if no left censoring, 1 if left censoring (Default is 0).
    right_censor : int
        0 if no right censoring, 1 if right censoring (Default is 0).
    eta : np.ndarray
        Array of shape (n_samples, n_states, n_durations) containing the eta values.
    u : np.ndarray
        Array of shape (n_samples, n_states, n_durations) containing the u values as produced by the ``_u_only`` auxiliary
        function.
    xi : np.ndarray
        Array of shape (n_samples, n_states, n_states) containing the xi values.

    Returns
    -------
    alpha : np.ndarray
        Array of shape (n_states,) containing the alpha values.

    See Also
    --------
    base.HSMM._core_forward : Method that computes the forward variable.
    """
    # set number of iterations for t
    if right_censor != 0:
        t_iter = n_samples + n_durations - 1
    else:
        t_iter = n_samples
    alpha_addends = np.empty(n_durations)
    astar_addends = np.empty(n_states)
    alpha = np.empty(n_states)
    alphastar = np.empty((t_iter, n_states))

    for j in range(n_states):
        alphastar[0, j] = log_startprob[j]
    for t in range(t_iter):
        for j in range(n_states):
            for d in range(n_durations):
                # alpha summation
                if t - d >= 0:
                    alpha_addends[d] = alphastar[t - d, j] + log_durprob[j, d] + _curr_u(n_samples, u, t, j, d)
                elif left_censor != 0:
                    alpha_addends[d] = log_startprob[j] + log_durprob[j, d] + _curr_u(n_samples, u, t, j, d)
                else:
                    alpha_addends[d] = -np.inf
                eta[t, j, d] = alpha_addends[d]  # eta initial
            alpha[j] = logsumexp(alpha_addends)
        # alphastar summation
        for j in range(n_states):
            for i in range(n_states):
                astar_addends[i] = alpha[i] + log_transmat[i, j]
                if t < n_samples:
                    xi[t, i, j] = astar_addends[i]  # xi initial
            if t < t_iter - 1:
                alphastar[t + 1, j] = logsumexp(astar_addends)
    return alpha


def _backward(n_samples, n_states, n_durations,
              log_startprob,
              log_transmat,
              log_durprob,
              right_censor,
              beta, u, betastar):
    """
    Computes the backward variable beta needed for the likelihood computation and the parameters re-estimation.
    Utilized by the ``HSMM._core_backward`` method.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_states : int
        Number of states.
    n_durations : int
        Number of durations.
    log_startprob : np.ndarray
        Array of shape (n_states,) containing the log of the initial state probabilities.
    log_transmat : np.ndarray
        Array of shape (n_states, n_states) containing the log of the transition probabilities.
    log_durprob : np.ndarray
        Array of shape (n_states, n_durations) containing the log of the duration probabilities.
    right_censor : int
        0 if no right censoring, 1 if right censoring (Default is 0).
    beta : np.ndarray
        Array of shape (n_samples, n_states) containing the initialized beta values.
    u : np.ndarray
        Array of shape (n_samples, n_states, n_durations) containing the u values as produced by the ``_u_only`` auxiliary
        function.
    betastar : np.ndarray
        Array of shape (n_samples, n_states) containing the beta* values.

    Returns
    -------
    None

    See Also
    --------
    base.HSMM._core_backward : Method that computes the backward variable.

    Notes
    -----
    The beta values are computed inplace.
    """
    bstar_addends = np.empty(n_durations)
    beta_addends = np.empty(n_states)
    for j in range(n_states):
        beta[n_samples - 1, j] = 0.0
    for t in range(n_samples - 2, -2, -1):
        for j in range(n_states):
            for d in range(n_durations):
                # betastar summation
                if t + d + 1 < n_samples:
                    bstar_addends[d] = log_durprob[j, d] + _curr_u(n_samples, u, t + d + 1, j, d) + beta[t + d + 1, j]
                elif right_censor != 0:
                    bstar_addends[d] = log_durprob[j, d] + _curr_u(n_samples, u, t + d + 1, j, d)
                else:
                    bstar_addends[d] = -np.inf
            betastar[t + 1, j] = logsumexp(bstar_addends)
        if t > -1:
            # beta summation
            for j in range(n_states):
                for i in range(n_states):
                    beta_addends[i] = log_transmat[j, i] + betastar[t + 1, i]
                beta[t, j] = logsumexp(beta_addends)


def _u_only(n_samples, n_states, n_durations,
            log_obsprob, u):
    """
    Computes the u values needed for the forward variable computation.
    Utilized by the ``HSMM._core_u_only`` method.

    Parameters
    ----------
    n_samples : int
        Number of samples.
    n_states : int
        Number of states.
    n_durations : int
        Number of durations.
    log_obsprob : np.ndarray
        Array of shape (n_samples, n_states) containing the log of the observation probabilities.
    u : np.ndarray
        Array of shape (n_samples, n_states, n_durations) containing the u values.

    Returns
    -------
    None

    See Also
    --------
    base.HSMM._core_u_only : Method that computes the u values.

    Notes
    -----
    The u values are computed inplace.

    """
    for t in range(n_samples):
        for j in range(n_states):
            for d in range(n_durations):
                if t < 1 or d < 1:
                    u[t, j, d] = log_obsprob[t, j]
                else:
                    u[t, j, d] = u[t - 1, j, d - 1] + log_obsprob[t, j]
