import numpy as np
from scipy.special import logsumexp


def _curr_u(n_samples, u, t, j, d):
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


#
# def _smoothed(n_samples, n_states, n_durations,
#               beta, betastar,
#               right_censor,
#               eta, xi, gamma):
#
#     for t in range(n_samples - 1, -1, -1):
#         for i in range(n_states):
#             # eta computation
#             # note: if with right censor, then eta[t, :, :] for t >= n_samples will only
#             # be used for gamma computation. since beta[t, :] = 0 for t >= n_samples, hence
#             # no modifications to eta at t >= n_samples.
#             for d in range(n_durations):
#                 eta[t, i, d] = eta[t, i, d] + beta[t, i]
#             # xi computation
#             # note: at t == n_samples - 1, it is decided that xi[t, :, :] should be log(0),
#             # either with right censor or without, because there is no more next data.
#             for j in range(n_states):
#                 if t == n_samples - 1:
#                     xi[t, i, j] = -np.inf
#                 else:
#                     xi[t, i, j] = xi[t, i, j] + betastar[t + 1, j]
#             # gamma computation
#             # note: this is the slow "original" method. the paper provides a faster
#             # recursive method (using xi), but it requires subtraction and produced
#             # numerical inaccuracies from our initial tests.
#             gamma[t, i] = -np.inf
#             for d in range(n_durations):
#                 for h in range(n_durations):
#                     if h >= d and (t + d < n_samples or right_censor != 0):
#                         gamma[t, i] = np.logaddexp(gamma[t, i], eta[t + d, i, h])
#

def _u_only(n_samples, n_states, n_durations,
            log_obsprob, u):
    for t in range(n_samples):
        for j in range(n_states):
            for d in range(n_durations):
                if t < 1 or d < 1:
                    u[t, j, d] = log_obsprob[t, j]
                else:
                    u[t, j, d] = u[t - 1, j, d - 1] + log_obsprob[t, j]
