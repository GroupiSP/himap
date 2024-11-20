from scipy.stats import multivariate_normal
from scipy.special import logsumexp
from sklearn import cluster
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from itertools import zip_longest
import matplotlib.pyplot as plt
from math import ceil
import copy
from scipy.stats import geom
from scipy.signal import convolve
import scipy.stats as stats
from scipy.stats import norm
from ab import _forward, _backward, _u_only
import sys

from utils import baumwelch_method, fs_calculation,bs_calculation

import smoothed as core
from utils import *

np.seterr(invalid='ignore')


class HSMM:
    def __init__(self, n_states=2, n_durations=5, n_iter=20, tol=1e-2, left_to_right=False, obs_state_len=None,
                 f_value=None, random_state=None, name=""):

        if not n_states >= 2:
            raise ValueError("number of states (n_states) must be at least 2")
        if not n_durations >= 1:
            raise ValueError("number of durations (n_durations) must be at least 1")

        if obs_state_len is not None and f_value is not None:
            self.last_observed = True
        elif obs_state_len is not None and f_value is None:
            raise ValueError("provide the observed state's final value")
        elif obs_state_len is None and f_value is not None:
            raise ValueError("provide the observed state's length")
        else:
            self.last_observed = False

        self.max_len = None
        self.n_states = n_states
        self.n_durations = n_durations
        self.n_iter = n_iter
        self.tol = tol
        self.left_to_right = left_to_right
        self.obs_state_len = obs_state_len
        self.f_value = f_value
        self.random_state = random_state
        self.name = name
        self._print_name = ""
        self.oscillation = None
        self.score_per_iter = None
        self.score_per_sample = None
        self.bic_score = None
        self.left_censor = 0
        self.right_censor = 0

    # _init: initializes model parameters if there are none yet
    # if left_to_right is True then the first state has start probability=1 and the tmat has transition probability=1
    # only for i+1 state. Last state is observed
    def _init(self, X=None):
        if self.name != "" and self._print_name == "":
            self._print_name = f" ({self.name})"
        if not hasattr(self, "pi") and not self.left_to_right:
            self.pi = np.full(self.n_states, 1.0 / self.n_states)
        elif not hasattr(self, "pi") and self.left_to_right:
            self.pi = np.zeros(self.n_states)
            self.pi[0] = 1
        if not hasattr(self, "tmat") and not self.left_to_right:
            self.tmat = np.full((self.n_states, self.n_states), 1.0 / (self.n_states - 1))
            for i in range(self.n_states):
                self.tmat[i, i] = 0.0  # no self-transitions in EDHSMM
        elif not hasattr(self, "tmat") and self.left_to_right:
            self.tmat = np.zeros((self.n_states, self.n_states))

            for i in range(len(self.tmat)):
                for j in range(len(self.tmat[i]) - 1):
                    if i == j and j < len(self.tmat[i]) - 2:
                        self.tmat[i, j + 1] = 1

                    elif i == j and j == len(self.tmat[i]) - 2:
                        self.tmat[i, j + 1] = 1
            # self.tmat[-1, -1] = 1

        self._dur_init()  # duration

    # _check: check if properties of model parameters are satisfied
    def _check(self):
        # starting probabilities
        self.pi = np.asarray(self.pi)
        if self.pi.shape != (self.n_states,):
            raise ValueError("start probabilities (self.pi) must have shape ({},)".format(self.n_states))
        if not np.allclose(self.pi.sum(), 1.0):
            raise ValueError("start probabilities (self.pi) must add up to 1.0")
        # transition probabilities
        self.tmat = np.asarray(self.tmat)
        if self.tmat.shape != (self.n_states, self.n_states):
            raise ValueError("transition matrix (self.tmat) must have shape ({0}, {0})".format(self.n_states))
        if not np.allclose(self.tmat.sum(axis=1), 1.0) and not self.left_to_right:
            raise ValueError("transition matrix (self.tmat) must add up to 1.0")
        if not self.left_to_right:
            for i in range(self.n_states):
                if self.tmat[i, i] != 0.0:  # check for diagonals
                    raise ValueError("transition matrix (self.tmat) must have all diagonals equal to 0.0")
        # duration probabilities
        self._dur_check()

    # _dur_init: initializes duration parameters if there are none yet
    def _dur_init(self):
        """
        arguments: (self)
        return: None
        > initialize the duration parameters
        """
        pass  # implemented in subclass

    # _dur_check: checks if properties of duration parameters are satisfied
    def _dur_check(self):
        """
        arguments: (self)
        return: None
        > check the duration parameters
        """
        pass  # implemented in subclass

    # _dur_probmat: compute the probability per state of each duration
    def _dur_probmat(self):
        """
        arguments: (self)
        return: duration probability matrix
        """
        pass  # implemented in subclass

    # _dur_mstep: perform m-step for duration parameters
    def _dur_mstep(self):
        """
        arguments: (self, new_dur)
        return: None
        > compute the duration parameters
        """
        pass  # implemented in subclass

    # _emission_logl: compute the log-likelihood per state of each observation
    def _emission_logl(self):
        """
        arguments: (self, X)
        return: logframe
        """
        pass  # implemented in subclass

    # _emission_pre_mstep: prepare m-step for emission parameters
    def _emission_pre_mstep(self):
        """
        arguments: (self, gamma, emission_var)
        return: None
        > process gamma and save output to emission_var
        """
        pass  # implemented in subclass

    # _emission_mstep: perform m-step for emission parameters
    def _emission_mstep(self):
        """
        arguments: (self, X, emission_var)
        return: None
        > compute the emission parameters
        """
        pass  # implemented in subclass

    # _state_sample: generate observation for given state
    def _state_sample(self):
        """
        arguments: (self, state, random_state=None)
        return: np.ndarray of length equal to dimension of observation
        > generate sample from state
        """
        pass  # implemented in subclass

    # sample: generate random observation series
    def sample(self, n_samples=5, random_state=None):
        self._init(None)  # see "note for programmers" in init() in GaussianHSMM
        # self._check()
        # setup random state
        if random_state is None:
            random_state = self.random_state
        rnd_checked = np.random.default_rng(random_state)
        # adapted from hmmlearn 0.2.3 (see _BaseHMM.score function)
        pi_cdf = np.cumsum(self.pi)
        tmat_cdf = np.cumsum(self.tmat, axis=1)
        dur_cdf = np.cumsum(self._dur_probmat(), axis=1)
        # for first state
        currstate = (pi_cdf > rnd_checked.random()).argmax()  # argmax() returns only the first occurrence
        currdur = (dur_cdf[currstate] > rnd_checked.random()).argmax() + 1
        if currdur > n_samples:
            print(f"SAMPLE{self._print_name}: n_samples is too small to contain the first state duration.")
            return None
        state_sequence = [currstate] * currdur
        X = [self._state_sample(currstate, rnd_checked) for i in range(currdur)]  # generate observation
        ctr_sample = currdur
        # for next state transitions
        while ctr_sample < n_samples:
            currstate = (tmat_cdf[currstate] > rnd_checked.random()).argmax()
            currdur = (dur_cdf[currstate] > rnd_checked.random()).argmax() + 1
            # test if now in the end of generating samples
            if ctr_sample + currdur > n_samples:
                break  # else, do not include exceeding state duration
            state_sequence += [currstate] * currdur
            X += [self._state_sample(currstate, rnd_checked) for i in range(currdur)]  # generate observation
            ctr_sample += currdur
        return ctr_sample, np.atleast_2d(X), np.array(state_sequence, dtype=int)

    # _core_u_only: Python implementation
    def _core_u_only(self, logframe):
        n_samples = logframe.shape[0]
        u = np.empty((n_samples, self.n_states, self.n_durations))
        _u_only(n_samples, self.n_states, self.n_durations,
                logframe, u)
        return u

    # _core_forward: Python implementation
    def _core_forward(self, u, logdur):
        n_samples = u.shape[0]
        eta_samples = n_samples
        eta = np.empty((eta_samples + 1, self.n_states, self.n_durations))  # +1
        xi = np.empty((n_samples + 1, self.n_states, self.n_states))  # +1
        alpha = _forward(n_samples, self.n_states, self.n_durations,
                         log_mask_zero(self.pi),
                         log_mask_zero(self.tmat),
                         logdur, self.left_censor, self.right_censor, eta, u, xi)
        return eta, xi, alpha

    # _core_backward: Python implementation
    def _core_backward(self, u, logdur):
        n_samples = u.shape[0]
        beta = np.empty((n_samples, self.n_states))
        betastar = np.empty((n_samples, self.n_states))
        _backward(n_samples, self.n_states, self.n_durations,
                  log_mask_zero(self.pi),
                  log_mask_zero(self.tmat),
                  logdur, self.right_censor, beta, u, betastar)
        return beta, betastar

    # _core_smoothed: The SLOWEST fnc if implemented in python
    # still in Cython
    def _core_smoothed(self, beta, betastar, eta, xi):
        n_samples = beta.shape[0]
        gamma = np.empty((n_samples, self.n_states))
        core._smoothed(n_samples, self.n_states, self.n_durations,
                       beta, betastar, self.right_censor, eta, xi, gamma)
        return gamma

    # _core_viterbi: container for core._viterbi (for multiple observation sequences)
    def _core_viterbi(self, u, logdur):
        n_samples = u.shape[0]
        state_sequence, state_logl = core._viterbi(n_samples, self.n_states, self.n_durations,
                                                   log_mask_zero(self.pi),
                                                   log_mask_zero(self.tmat),
                                                   logdur, self.left_censor, self.right_censor, u)
        return state_sequence, state_logl

    # score: log-likelihood computation from observation series
    def score(self, X):
        self._init(X)
        # self._check()
        logdur = log_mask_zero(self._dur_probmat())  # build logdur
        # main computations
        score = 0

        logframe = self._emission_logl(X)  # build logframe
        u = self._core_u_only(logframe)
        _, betastar = self._core_backward(u, logdur)
        gammazero = log_mask_zero(self.pi) + betastar[0]
        score += logsumexp(gammazero)
        return score

    # predict: hidden state & duration estimation from observation series
    def predict(self, X):
        self._init(X)
        # self._check()
        logdur = log_mask_zero(self._dur_probmat())  # build logdur
        # main computations
        state_logl = 0  # math note: this is different from score() output
        state_sequence = np.empty(X.shape[0], dtype=int)  # total n_samples = X.shape[0]
        logframe = self._emission_logl(X)  # build logframe
        u = self._core_u_only(logframe)
        iter_state_sequence, iter_state_logl = self._core_viterbi(u, logdur)
        state_logl += iter_state_logl
        state_sequence = iter_state_sequence
        return state_sequence, state_logl

    # fit: parameter estimation from observation series
    def fit(self, X, save_iters=False):
        score_per_iter = []
        score_per_sample = []

        keys = list(X.keys())
        lens = []
        for traj in keys:
            lens.append(len(X[traj]))

        self.max_len = max(lens)
        init_history = X[keys[lens.index(max(lens))]]

        init_history = np.array(init_history).reshape((len(init_history), 1))

        self._init(init_history)  # initialization with the longest history
        self._check()

        # main computations
        for itera in range(self.n_iter):
            score = 0

            pi_num = mean_numerator = cov_numerator = denominator = np.full(self.n_states, -np.inf)
            tmat_num = dur_num = gamma_num = -np.inf

            for i in tqdm(range(len(X)), desc=f"Iters {itera + 1}/{self.n_iter}"):
                history = X[keys[i]]
                history = np.array(history).reshape((len(history), 1))
                emission_var = np.empty((history.shape[0], self.n_states))  # cumulative concatenation of gammas
                logdur = log_mask_zero(self._dur_probmat())  # build logdur
                j = len(history)

                logframe = self._emission_logl(history)  # build logframe
                logframe[logframe > 0] = 0  # necessary condition for histories with discrete observations; as the model
                # converges and calculates close-to-zero covariances, the probabilities of
                # observing the means get close to 1. So to avoid positive logframe values
                # we set them to 0 (exp(0)=1)

                u = self._core_u_only(logframe)
                eta, xi, alpha = self._core_forward(u, logdur)
                beta, betastar = self._core_backward(u, logdur)
                gamma = self._core_smoothed(beta, betastar, eta, xi)
                sample_score = logsumexp(gamma[0, :])
                score_per_sample.append(sample_score)  # this saves the scores of every history for every iter
                score += sample_score  # this is the total likelihood for all histories for current iter

                # preparation for reestimation / M-step
                if eta.shape[0] != j + 1:
                    eta = eta[:j + 1]
                if gamma.shape[0] != j + 1:
                    gamma = gamma[:j + 1]

                # normalization of each history's xi, eta and gamma with its likelihood
                norm_xi = np.subtract(xi, sample_score)
                norm_eta = np.subtract(eta, sample_score)
                norm_gamma = np.subtract(gamma, sample_score)

                ##############emission matrix estimation##############
                log_history = log_mask_zero(history)
                log_history[np.isnan(log_history)] = -np.inf
                mean_num = gamma + log_history  # numerator for mean re-estimation of current history
                mean_num = np.subtract(mean_num, sample_score)

                dist = history - self.mean[:, None]
                dist = np.square(dist.reshape((dist.shape[0], dist.shape[1])).T)
                log_dist = log_mask_zero(dist)
                log_dist[np.isnan(log_dist)] = -np.inf
                cov_num = gamma + log_dist  # numerator for covars re-estimation of current history
                cov_num = np.subtract(cov_num, sample_score)

                # add the mean numerator, covars numerator and denominator of prev history at the end of the current
                # ones
                mean_num_multiple_histories = np.vstack((mean_num, mean_numerator))
                cov_num_multiple_histories = np.vstack((cov_num, cov_numerator))
                denominator_multiple_histories = np.vstack((norm_gamma, denominator))

                # sum over time and histories
                mean_numerator = logsumexp(mean_num_multiple_histories, axis=0)
                cov_numerator = logsumexp(cov_num_multiple_histories, axis=0)
                denominator = logsumexp(denominator_multiple_histories, axis=0)
                ########################################################

                # append the previous sum of xi and eta to the last position of the new xi and eta
                norm_xi[j] = tmat_num
                norm_eta[j] = dur_num

                # Calculation of he total xi, eta and gamma variables for all the histories
                pi_num = logsumexp([pi_num, norm_gamma[0]], axis=0)
                tmat_num = logsumexp(norm_xi, axis=0)
                dur_num = logsumexp(norm_eta, axis=0)

            ############################################################################################################
            # check for loop break
            if itera > 0 and abs(abs(score) - abs(old_score)) < self.tol:
                print(f"\nFIT{self._print_name}: converged at loop {itera + 1} with score: {score}.")
                break
            elif itera > 0 and (np.isnan(score) or np.isinf(score)):
                print("\nThere is no possible solution. Try different parameters.")
                break
            # elif itera > 0 and score<old_score:
            #     self.oscillation = True
            else:
                score_per_iter.append(score)
                old_score = score

            # save the previous version of the model prior to updating
            if save_iters:
                with open('model_iter' + str(itera + 1) + '.txt', 'wb') as f:
                    pickle.dump(self, f)

            # emission parameters re-estimation
            weight = mean_numerator - denominator
            weight1 = cov_numerator - denominator

            mean = np.exp(weight)
            covmat = np.exp(weight1)

            for k in range(len(covmat)):
                if covmat[k,] == 0 or np.isnan(covmat[k,]):
                    covmat[k,] = 1e-30

            # reestimation of the rest of the model parameters and model update
            self.pi = np.exp(pi_num - logsumexp(pi_num))
            self.tmat = np.exp(tmat_num - logsumexp(tmat_num, axis=1)[None].T)
            self.dur = np.exp(dur_num - logsumexp(dur_num, axis=1)[None].T)
            self.mean = mean.reshape((mean.shape[0], 1))
            self.covmat = covmat.reshape((covmat.shape[0], 1, 1))

            # new
            self.tmat[-1, :] = np.zeros(self.n_states)
            #

            print(f"\nFIT{self._print_name}: re-estimation complete for loop {itera + 1} with score: {score}.")

        score_per_sample = np.array(score_per_sample).reshape((-1, len(X))).T
        score_per_iter = np.array(score_per_iter).reshape(len(score_per_iter), 1)
        print(sorted(range(len(self.mean.tolist())), key=self.mean.tolist().__getitem__))
        if self.last_observed:
            self.dur[-1, self.obs_state_len] = 0
            self.dur[-1, self.obs_state_len - 1] = 1

        if self.oscillation:
            print("\nOscillation in the convergence detected. Different parameters may give better results.")
        # return fitted model for joblib
        self.score_per_iter = score_per_iter
        self.score_per_sample = score_per_sample
        self.bic(X)

        return self

    def bic(self, train):
        if self.max_len is None:
            keys = list(train.keys())
            lens = []
            for traj in keys:
                lens.append(len(train[traj]))

            self.max_len = max(lens)

        if self.left_to_right and self.last_observed:
            n_params = (self.n_states - 1) * (self.n_durations - 1) + (self.n_states - 1) * 2

        elif self.left_to_right and not self.last_observed:
            n_params = (self.n_states) * (self.n_durations - 1) + (self.n_states) * 2

        elif not self.left_to_right:
            n_params = self.n_states + self.n_states ** 2 + (self.n_states) * (self.n_durations - 1) + (
                self.n_states) * 2

        best_ll = np.max(self.score_per_iter)
        n = self.max_len
        score = best_ll - 0.5 * n_params * np.log(n)
        self.bic_score = score

        return score

    def fit_bic(self, X, states, return_models=False):
        '''
        fit HSMM with the BIC criterion
        :param X: degradation histories
        :param states: list with number of states
        :param return_models: return all models in a dictionary
        '''

        bic = []
        models = {
            f"model_{i}": None for i in range(len(states))
        }
        for i, n_states in enumerate(states):
            hsmm = GaussianHSMM(n_states=n_states,
                                n_durations=self.n_durations,
                                n_iter=self.n_iter,
                                tol=self.tol,
                                f_value=self.f_value,
                                obs_state_len=self.obs_state_len,
                                left_to_right=self.left_to_right
                                )

            hsmm.fit(X)

            n = 0

            for k in range(len(X)):
                history = get_single_history(X, k)
                n += len(history)

            loglik = hsmm.score_per_iter[-1]
            hi_emission = 2 * hsmm.n_states
            hi_dur = (hsmm.n_states - 1) * hsmm.n_durations
            hi = hi_emission + hi_dur
            bic.append(loglik - (hi / 2) * np.log(n))

            models[f"model_{i}"] = hsmm.__dict__

        best_model = models[f"model_{np.argmax(np.asarray(bic))}"]
        print(f"Best model was the model with {best_model['n_states']} states.")
        self.__dict__.update(best_model)
        if return_models:
            return self, models, bic

        return self, bic

    def RUL(self, viterbi_states, max_samples, path, equation=1, plot_rul=False, index=None):
        """
        :param path:
        :param index:
        :param plot_rul:
        :param viterbi_states: Single history
        :param max_samples: maximum length of RUL (default: 3000)
        :param equation: 1=best (with reduction with sojourn time to both terms)
        :return:

        Works for a single state history.
        """

        RUL = np.zeros((len(viterbi_states), max_samples))
        mean_RUL, LB_RUL, UB_RUL = (np.zeros(len(viterbi_states)) for _ in range(3))
        dur = self.dur
        prev_state, stime = 0, 0
        n_states = self.n_states

        for i, state in enumerate(viterbi_states):
            first, second = (np.zeros_like(dur[0, :]) for _ in range(2))
            first[1] = second[1] = 1
            cdf_curr_state = np.cumsum(dur[state, :])
            if state == prev_state:
                stime += 1
            else:
                prev_state = state
                stime = 1

            if stime < len(cdf_curr_state):
                d_value = cdf_curr_state[stime]
            else:
                d_value = cdf_curr_state[-1]

            available_states = np.arange(state, n_states - 1)

            for j in available_states:
                if j != available_states[-1]:
                    first = np.convolve(first, dur[j, :])
                    second = np.convolve(second, dur[j + 1, :])

                else:
                    first = np.convolve(first, dur[j, :])

            if equation == 1:
                first_red = np.zeros_like(first)
                first_red = first[stime:]

                # make sure that after subtracting the soujourn time from the pmf of the first term, that it still sums to 1
                if first_red.sum() != 1:
                    first_red[0] = first_red[0] + (1 - first_red.sum())

            else:
                first_red = first

            first_red = first_red * (1 - d_value)
            second = second * d_value

            result = [sum(n) for n in zip_longest(first_red, second, fillvalue=0)]

            if available_states.size > 0 or not self.last_observed:

                RUL[i, :] = [sum(n) for n in zip_longest(RUL[i, :], result, fillvalue=0)]
                cdf_curr_RUL = np.cumsum(RUL[i, :])

                # UB RUL
                X, y = [], []
                for l, value in enumerate(cdf_curr_RUL):
                    if value > 0.05:
                        X = [cdf_curr_RUL[l - 1], value]
                        y = [l - 1, l]
                        break
                X = np.asarray(X).reshape(-1, 1)
                y = np.asarray(y).reshape(-1, 1)
                UB_RUL[i] = LinearRegression().fit(X, y).predict(np.asarray(0.05).reshape(-1, 1))

                # LB RUL
                X, y = [], []
                for l, value in enumerate(cdf_curr_RUL):
                    if value > 0.95:
                        X = [cdf_curr_RUL[l - 1], value]
                        y = [l - 1, l]
                        break
                X = np.asarray(X).reshape(-1, 1)
                y = np.asarray(y).reshape(-1, 1)
                LB_RUL[i] = LinearRegression().fit(X, y).predict(np.asarray(0.95).reshape(-1, 1))

                # mean RUL
                value = np.arange(0, RUL.shape[1])
                mean_RUL[i] = sum(RUL[i, :] * value)

            elif not available_states.size > 0 and self.last_observed:
                RUL[i, :], mean_RUL[i], UB_RUL[i], LB_RUL[i] = 0, 0, 0, 0
                mean_RUL = np.hstack((np.delete(mean_RUL, mean_RUL == 0), np.array((0))))
                UB_RUL = np.hstack((np.delete(UB_RUL, UB_RUL == 0), np.array((0))))
                LB_RUL = np.hstack((np.delete(LB_RUL, LB_RUL == 0), np.array((0))))
                break

        true_RUL_v = len(viterbi_states) - 1
        if plot_rul:
            fig, ax = plt.subplots(figsize=(19, 10))
            ax.plot([0, true_RUL_v], [true_RUL_v, 0], label='True RUL', color='black', linewidth=2)
            ax.plot(mean_RUL, '--', label='Mean Predicted RUL', color='tab:red', linewidth=2)
            ax.plot(UB_RUL, '-.', label='Lower Bound (90% CI)', color='tab:blue', linewidth=1)
            ax.plot(LB_RUL, '-.', label='Upper Bound (90% CI)', color='tab:blue', linewidth=1)
            ax.fill_between(np.arange(0, len(UB_RUL)), UB_RUL, LB_RUL, alpha=0.1, color='tab:blue')
            fig.suptitle('RUL')
            ax.legend(loc='best')
            plt.savefig(path + f'figures/RUL_plot_Signal_{index + 1}.png', dpi=300)
            plt.close()

        return RUL, mean_RUL, UB_RUL, LB_RUL, true_RUL_v

    def prognostics(self, data, technique, max_samples=None, plot_rul=False, equation=1):
        """
        :param data: degradation histories
        :param max_samples: maximum length of RUL (default: 3000)
        :param technique: 'cmapss' or 'mimic' for file name
        :param plot_rul: Display RUL plot for each sample
        :return: None, json files are saved for pdf_rul and mean_rul
        """
        if self.max_len is None:
            keys = list(data.keys())
            lens = []
            for traj in keys:
                lens.append(len(data[traj]))

            self.max_len = max(lens)
        path = f"edhsmm/results/{technique}/"
        data_list = []
        max_timesteps = self.max_len
        max_samples = ceil(max_timesteps * 10) if max_samples is None else max_samples
        keys = list(data.keys())
        for i in range(len(data)):
            data_list.append(data[keys[i]])

        viterbi_states_all = get_viterbi(self, data)  # this has the full length of the observed state

        viterbi_list = []
        for i in range(len(viterbi_states_all)):
            # this has a single timestep for the observed state - Ready for RUL
            viterbi_single_state = get_single_history_states(viterbi_states_all,
                                                             i,
                                                             last_state=self.n_states - 1)
            viterbi_list.append(viterbi_single_state)

        pdf_ruls_all = {
            f"traj_{j}": {
                f"timestep_{i}": np.zeros((max_samples,)) for i in range(len(viterbi_list[j]))
            }
            for j in range(len(viterbi_list))
        }

        mean_rul_per_step = {
            f"traj_{i}": np.zeros((len(viterbi_list[i], ))) for i in range(len(viterbi_list))
        }

        upper_rul_per_step = {
            f"traj_{i}": np.zeros((len(viterbi_list[i], ))) for i in range(len(viterbi_list))
        }

        lower_rul_per_step = {
            f"traj_{i}": np.zeros((len(viterbi_list[i], ))) for i in range(len(viterbi_list))
        }

        true_rul_v = {
            f"traj_{i}": ""
        }

        for i in range(len(viterbi_states_all)):
            viterbi_single_state = get_single_history_states(viterbi_states_all,
                                                             i,
                                                             last_state=self.n_states - 1
                                                             )
            # viterbi_single_state=np.array(viterbi_single_state).reshape((len(viterbi_single_state),1))
            RUL_pred, mean_RUL, UB_RUL, LB_RUL, true_rul = self.RUL(viterbi_single_state,
                                                                    max_samples=max_samples,
                                                                    equation=equation,
                                                                    plot_rul=plot_rul,
                                                                    index=i,
                                                                    path=path,
                                                                    )

            for j in range(RUL_pred.shape[0]):
                pdf_ruls_all[f"traj_{i}"][f"timestep_{j}"] = RUL_pred[j, :].copy()
                mean_rul_per_step[f"traj_{i}"] = mean_RUL.copy()
                upper_rul_per_step[f"traj_{i}"] = UB_RUL.copy()
                lower_rul_per_step[f"traj_{i}"] = LB_RUL.copy()
                true_rul_v[f"traj_{i}"] = true_rul

        path_mean_rul = path + f"prognostics/mean_rul_per_step_{technique}.json"
        path_pdf_rul = path + f"prognostics/pdf_ruls_{technique}.json"
        path_upper_rul = path + f"prognostics/upper_ruls_{technique}.json"
        path_lower_rul = path + f"prognostics/lower_ruls_{technique}.json"
        path_true_rul = path + f"prognostics/true_ruls_{technique}.json"

        with open(path_mean_rul, "w") as fp:
            json.dump(mean_rul_per_step, fp, cls=NumpyArrayEncoder)

        with open(path_pdf_rul, "w") as fp:
            json.dump(pdf_ruls_all, fp, cls=NumpyArrayEncoder)

        with open(path_upper_rul, "w") as fp:
            json.dump(upper_rul_per_step, fp, cls=NumpyArrayEncoder)

        with open(path_lower_rul, "w") as fp:
            json.dump(lower_rul_per_step, fp, cls=NumpyArrayEncoder)

        with open(path_true_rul, "w") as fp:
            json.dump(true_rul_v, fp, cls=NumpyArrayEncoder)

        print(f"Prognostics complete. Results saved to: {path}")

    def save_model(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load_model(self, path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            self.__dict__.update(obj)


# Sample Subclass: Explicit Duration HSMM with Gaussian Emissions
class GaussianHSMM(HSMM):
    def __init__(self, n_states=2, n_durations=5, n_iter=20, tol=1e-2, left_to_right=False, obs_state_len=None,
                 f_value=None, random_state=None, name="",
                 kmeans_init='k-means++', kmeans_n_init='auto'):
        super().__init__(n_states, n_durations, n_iter, tol, left_to_right, obs_state_len,
                         f_value, random_state, name)
        self.kmeans_init = kmeans_init
        self.kmeans_n_init = kmeans_n_init

    def _init(self, X=None):
        super()._init()
        # note for programmers: for every attribute that needs X in score()/predict()/fit(),
        # there must be a condition "if X is None" because sample() doesn't need an X, but
        # default attribute values must be initiated for sample() to proceed.
        if not hasattr(self, "mean") and not self.left_to_right and not self.last_observed:  # also set self.n_dim here
            if X is None:  # default for sample()
                self.n_dim = 1
                self.mean = np.arange(0., self.n_states)[:, None]  # = [[0.], [1.], [2.], ...]
            else:
                self.n_dim = X.shape[1]
                kmeans = cluster.KMeans(n_clusters=self.n_states, random_state=self.random_state,
                                        init=self.kmeans_init, n_init=self.kmeans_n_init)
                kmeans.fit(X)
                self.mean = kmeans.cluster_centers_

        if not hasattr(self, "mean") and not self.left_to_right and self.last_observed:  # also set self.n_dim here
            if X is None:  # default for sample()
                self.n_dim = 1
                self.mean = np.arange(0., self.n_states)[:, None]  # = [[0.], [1.], [2.], ...]
            else:
                self.n_dim = X.shape[1]
                kmeans = cluster.KMeans(n_clusters=self.n_states - 1, random_state=self.random_state,
                                        init=self.kmeans_init, n_init=self.kmeans_n_init)
                kmeans.fit(X)
                clusters = kmeans.cluster_centers_
                self.mean = np.vstack((clusters, [self.f_value]))

        elif not hasattr(self, "mean") and self.left_to_right:  # also set self.n_dim here
            if X is None:  # default for sample()
                self.n_dim = 1
                self.mean = np.arange(0., self.n_states)[:, None]  # = [[0.], [1.], [2.], ...]
            else:
                self.n_dim = X.shape[1]
                kmeans = cluster.KMeans(n_clusters=self.n_states - 1, random_state=self.random_state,
                                        init=self.kmeans_init, n_init=self.kmeans_n_init)
                kmeans.fit(X[:-self.obs_state_len])
                clusters_sorted = np.sort(kmeans.cluster_centers_, axis=0)
                self.mean = np.vstack((clusters_sorted, [self.f_value]))
        else:
            self.n_dim = self.mean.shape[1]  # also default for sample()
        if not hasattr(self, "covmat"):
            if X is None:  # default for sample()
                self.covmat = np.repeat(np.identity(self.n_dim)[None], self.n_states, axis=0)
            else:
                self.covmat = np.repeat(np.identity(self.n_dim)[None], self.n_states, axis=0)

    def _check(self):
        super()._check()
        # means
        self.mean = np.asarray(self.mean)
        if self.mean.shape != (self.n_states, self.n_dim):
            raise ValueError("means (self.mean) must have shape ({}, {})"
                             .format(self.n_states, self.n_dim))
        # covariance matrices
        self.covmat = np.asarray(self.covmat)
        if self.covmat.shape != (self.n_states, self.n_dim, self.n_dim):
            raise ValueError("covariance matrices (self.covmat) must have shape ({0}, {1}, {1})"
                             .format(self.n_states, self.n_dim))

    def _dur_init(self):
        # non-parametric duration
        if not hasattr(self, "dur") and not self.last_observed:
            self.dur = np.full((self.n_states, self.n_durations), 1.0 / self.n_durations)

        elif not hasattr(self, "dur") and self.last_observed:
            self.dur = np.zeros((self.n_states, self.n_durations))
            self.dur[:-1, 1:].fill(1.0 / (self.n_durations - 1))
            self.dur[-1, self.obs_state_len] = 1

    def _dur_check(self):
        self.dur = np.asarray(self.dur)
        if self.dur.shape != (self.n_states, self.n_durations):
            raise ValueError("duration probabilities (self.dur) must have shape ({}, {})"
                             .format(self.n_states, self.n_durations))
        if not np.allclose(self.dur.sum(axis=1), 1.0):
            raise ValueError("duration probabilities (self.dur) must add up to 1.0")

    def _dur_probmat(self):
        # non-parametric duration
        return self.dur

    def _dur_mstep(self, new_dur):
        # non-parametric duration
        self.dur = new_dur

    def _emission_logl(self, X):
        # abort EM loop if any covariance matrix is not symmetric, positive-definite.
        # adapted from hmmlearn 0.2.3 (see _utils._validate_covars function)
        for n, cv in enumerate(self.covmat):
            if (not np.allclose(cv, cv.T) or np.any(np.linalg.eigvalsh(cv) <= 0)):
                raise ValueError("component {} of covariance matrix is not symmetric, positive-definite."
                                 .format(n))
                # https://www.youtube.com/watch?v=tWoFaPwbzqE&t=1694s
        n_samples = X.shape[0]
        logframe = np.empty((n_samples, self.n_states))
        for i in range(self.n_states):
            # math note: since Gaussian distribution is continuous, probability density
            # is what's computed here. thus log-likelihood can be positive!
            multigauss = multivariate_normal(self.mean[i], self.covmat[i])
            for j in range(n_samples):
                logframe[j, i] = log_mask_zero(multigauss.pdf(X[j]))
        return logframe

    def _emission_mstep(self, X, emission_var, inplace=True):
        denominator = logsumexp(emission_var, axis=0)
        # denominator = emission_var
        weight_normalized = np.exp(emission_var - denominator)[None].T
        # compute means (from definition; weighted)
        mean = (weight_normalized * X).sum(1)
        # compute covariance matrices (from definition; weighted)
        dist = X - self.mean[:, None]
        covmat = ((dist * weight_normalized)[:, :, :, None] * dist[:, :, None]).sum(1)
        if inplace == False:
            return mean, covmat
        elif inplace == True:
            self.mean = mean
            self.covmat = covmat

    def _state_sample(self, state, random_state=None):
        rnd_checked = np.random.default_rng(random_state)
        return rnd_checked.multivariate_normal(self.mean[state], self.covmat[state])


######HMM

def calculate_expected_value(pmf_values):
    expected_value = sum(x * p for x, p in enumerate(pmf_values))
    return expected_value


def calculate_cdf(pmf, confidence_level):
    # Calculate the CDF
    cdf = np.cumsum(pmf)
    # Calculate the lower and upper percentiles
    lower_percentile = (1 - confidence_level) / 2
    upper_percentile = 1 - lower_percentile
    lower_value = np.argmax(cdf >= lower_percentile)
    upper_value = np.argmax(cdf >= upper_percentile)

    return lower_value, upper_value





class HMM:
    def __init__(self, n_states, n_obs_symbols, n_iter=20, tol=1e-2, left_to_right=False, name=""):
        if not n_states >= 2:
            raise ValueError("number of states (n_states) must be at least 2")
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.left_to_right = left_to_right
        self.n_obs_symbols = n_obs_symbols
        self.oscillation = None
        self.name = name
        self._print_name = ""

    def _init(self, X=None):
        if self.name != "" and self._print_name == "":
            self._print_name = f" ({self.name})"
        if not hasattr(self, "ini_tr") and not self.left_to_right:
            self.ini_tr = np.full((self.n_states, self.n_states), 1.0 / (self.n_states))

        elif not hasattr(self, "ini_tr") and self.left_to_right:
            self.ini_tr = np.zeros((self.n_states, self.n_states))
            for i in range(self.n_states):
                if i == self.n_states - 1:
                    self.ini_tr[i, i - 1:i + 1] = 0.5
                else:
                    self.ini_tr[i, i:i + 2] = 0.5

        if not hasattr(self, "ini_emi") and not self.left_to_right:
            self.ini_emi = np.full((self.n_states, self.n_obs_symbols), 1.0 / (self.n_obs_symbols))

        elif not hasattr(self, "ini_emi") and self.left_to_right:
            self.ini_emi = np.zeros((self.n_states, self.n_obs_symbols))
            prob = 1 / (self.n_obs_symbols - 1)
            for row in range(self.ini_emi.shape[0] - 1):
                for column in range(self.ini_emi.shape[1] - 1):
                    self.ini_emi[row, column] = prob
            self.ini_emi[self.n_states - 1, self.n_obs_symbols - 1] = 1

    def fit(self, X, return_all_scores=False, save_iters=False):
        self._init(X)
        tr = np.zeros(self.ini_tr.shape)
        emi = np.zeros(self.ini_emi.shape)
        score_per_iter = []
        score = 1
        calc_emi = self.ini_emi.copy()
        calc_tr = self.ini_tr.copy()
        converged = False
        for itera in range(self.n_iter):
            old_score = score
            score = 0
            old_emi = calc_emi.copy()
            old_tr = calc_tr.copy()
            for i in tqdm(range(len(X)), desc=f"Iters {itera + 1}/{self.n_iter}"):
                history = X[f'traj_{i}']
                _, logPseq, fs, bs, scale = self.decode(history, calc_emi, calc_tr)
                score += logPseq
                history = np.concatenate([np.array([0]), history])
                tr, emi = baumwelch_method(self.n_states, self.n_obs_symbols, logPseq, fs, bs, scale, score, history, tr, emi,
                                           calc_tr, calc_emi)
            total_emissions = np.sum(emi, axis=1)
            total_transitions = np.sum(tr, axis=1)

            calc_emi = emi / total_emissions[:, np.newaxis]
            calc_tr = tr / total_transitions[:, np.newaxis]

            calc_tr[np.isnan(calc_tr)] = 0
            calc_emi[np.isnan(calc_emi)] = 0

            score_per_iter.append(score)
            if (abs(score - old_score) / (1 + abs(old_score))) < self.tol and \
                    np.linalg.norm(calc_tr - old_tr, ord=np.inf) / self.n_states < self.tol and \
                    np.linalg.norm(calc_emi - old_emi, ord=np.inf) / self.n_obs_symbols < self.tol:
                print(f"\nFIT{self._print_name}: converged at loop {itera + 1} with score: {score}.")
                converged = True
                self.tr = calc_tr
                self.emi = calc_emi
                break

            if save_iters:
                with open('model_iter' + str(itera + 1) + '.txt', 'wb') as f:
                    pickle.dump(self, f)

        if not converged:
            print("\nThere is no possible solution. Try different parameters.")

        if return_all_scores:
            return self, score_per_iter
        return self

    def fit_bic(self, X, states, return_models=False):
        '''
        fit HSMM with the BIC criterion
        :param X: degradation histories
        :param states: list with number of states
        :param return_models: return all models in a dictionary
        '''

        bic = []
        models = {
            f"model_{i}": None for i in range(len(states))
        }

        n=0
        for key in X.keys():
            history = X[key]
            n += len(history)

        for i, n_states in enumerate(states):
            hmm_model = HMM(n_states=n_states,
                                n_obs_symbols=self.n_obs_symbols,
                                n_iter=self.n_iter,
                                tol=self.tol,
                                left_to_right=self.left_to_right
                                )

            _, score_iters = hmm_model.fit(X, return_all_scores=True)
            loglik = score_iters[-1]
            num_params_emi = np.count_nonzero(hmm_model.emi) - 1
            num_params_tr = np.count_nonzero(hmm_model.tr) - 1
            bic.append(loglik - ((num_params_tr + num_params_emi) / 2) * np.log(n))
            models[f"model_{i}"] = hmm_model

        best_model = models[f"model_{np.argmax(np.asarray(bic))}"]
        if return_models:
            return best_model, models, bic

        return best_model, bic

    def decode(self, history, calc_emi, calc_tr):
        history = np.concatenate([np.array([self.n_obs_symbols + 1]), history])
        end_traj = len(history)
        fs = np.zeros((self.n_states, end_traj))
        fs[0, 0] = 1  # assume that we start in state 1.
        s = np.zeros((1, end_traj))
        s[0, 0] = 1

        fs, s = fs_calculation(self.n_states, end_traj, fs, s, history, calc_emi, calc_tr)

        bs = np.ones((self.n_states, end_traj))
        bs = bs_calculation(self.n_states, end_traj, bs, s, history, calc_emi, calc_tr)
        pSeq = np.sum(np.log(s[0, 1:]))
        pStates = fs * bs

        # get rid of the column that we stuck in to deal with the f0 and b0
        pStates = np.delete(pStates, 0, axis=1)

        return pStates, pSeq, fs, bs, s

    def sample(self):

        history = []
        states = []
        trc = np.cumsum(self.tr, axis=1)
        ec = np.cumsum(self.emi, axis=1)
        trc = trc / np.tile(trc[:, -1:], (1, self.n_states))
        ec = ec / np.tile(ec[:, -1:], (1, self.n_obs_symbols))
        currentstate = 1
        while currentstate < self.n_states:
            stateVal = np.random.rand()
            state = 1
            for innerState in range(self.n_states - 2, -1, -1):
                if stateVal > trc[currentstate - 1, innerState]:
                    state = innerState + 2
                    break
            val = np.random.rand()
            emit = 1
            for inner in range(self.n_obs_symbols - 2, -1, -1):
                if val > ec[state - 1, inner]:
                    emit = inner + 2
                    break
            history.append(emit)
            states.append(state)
            currentstate = state
        for i in range(5):
            history.append(self.n_obs_symbols)
            states.append(self.n_states)
        return history, states

    def sample_dataset(self, n_samples):
        obs = {}
        states_all = {}
        for i in range(n_samples):
            sample = self.sample()
            history, states = sample
            obs[f'traj_{i}'] = history
            states_all[f'traj_{i}'] = states
        return obs, states_all

    def predict(self, history, return_score=False):
        end_traj = len(history)
        currentState = np.zeros(end_traj, dtype=int)
        if end_traj == 0:
            return currentState, float('-inf')

        logTR = np.log(self.tr)
        logE = np.log(self.emi)

        pTR = np.zeros((self.n_states, end_traj), dtype=int)
        v = -np.inf * np.ones(self.n_states)
        v[0] = 0
        vOld = np.copy(v)

        for count in range(end_traj):
            for state in range(self.n_states):
                bestVal = -np.inf
                bestPTR = 0
                for inner in range(self.n_states):
                    val = vOld[inner] + logTR[inner, state]
                    if val > bestVal:
                        bestVal = val
                        bestPTR = inner
                pTR[state, count] = bestPTR
                v[state] = logE[state, history[count] - 1] + bestVal
            vOld[:] = v
        logP, finalState = np.max(v), np.argmax(v)
        currentState[end_traj - 1] = finalState
        for count in range(end_traj - 2, -1, -1):
            currentState[count] = pTR[currentState[count + 1], count + 1]
            if currentState[count] == -1:
                raise ValueError(f"ZeroTransitionProbability: {currentState[count + 1]}")
        if return_score:
            return currentState + 1, logP
        return currentState + 1

    # method that given a history and a path of estimatedStates calculates the most likely transition and emission matrices
    def estimate(self, history, estimatedStates, return_matrices=False):
        tr = []
        emi = []
        end_traj = len(history)

        tr = np.zeros((self.n_states, self.n_states))
        emi = np.zeros((self.n_states, self.n_obs_symbols))

        for count in range(end_traj - 1):
            tr[estimatedStates[count] - 1, estimatedStates[count + 1] - 1] += 1

        for count in range(end_traj):
            emi[estimatedStates[count] - 1, history[count] - 1] += 1

        tr_sum = np.sum(tr, axis=1)
        emi_sum = np.sum(emi, axis=1)

        tr_sum[tr_sum == 0] = -np.inf
        emi_sum[emi_sum == 0] = -np.inf

        tr = tr / tr_sum[:, None]
        emi = emi / emi_sum[:, None]

        if return_matrices:
            return tr, emi
        else:
            self.tr = tr
            self.emi = emi
            return self

    def RUL(self, estimatedStates, time_sample, confidence=0.95):
        N = max(estimatedStates) - 1
        rul_matrix = np.zeros((len(estimatedStates), time_sample))
        prev_state = 0  # aux variable
        tau = 0
        for i in range(len(estimatedStates)):
            current_state = estimatedStates[i] - 1
            if current_state == N:
                rul_matrix[i, :] = np.zeros(time_sample)
            else:
                if prev_state == current_state:
                    tau += 1
                else:
                    prev_state = current_state
                    tau = 1
                a_ii = self.tr[current_state, current_state]
                a_next = self.tr[current_state + 1, current_state + 1]
                x_d_i = np.arange(0, time_sample)
                param_tau = geom.cdf(tau, 1 - a_ii)
                d_i = geom.pmf(x_d_i, 1 - a_ii)
                mod_d_i = np.zeros(len(d_i))
                mod_d_i[0:(len(d_i) - tau)] = d_i[tau:]
                added_prob = 0
                for timestep in range(tau + 1):
                    added_prob += d_i[timestep]
                mod_d_i[0] = added_prob
                normal_gaussian = norm.pdf(x_d_i, loc=1, scale=0.56999999)
                for j in range(current_state + 1, N):
                    d_j = geom.pmf(x_d_i, 1 - self.tr[j, j])
                    mod_d_i = convolve(mod_d_i, d_j, mode='full')
                mod_d_i = convolve(mod_d_i, normal_gaussian, mode='full')[:time_sample]
                sum_conv = geom.pmf(x_d_i, 1 - a_next)
                for j in range(current_state + 2, N):
                    d_j = geom.pmf(x_d_i, 1 - self.tr[j, j])
                    sum_conv = convolve(sum_conv, d_j, mode='full')
                sum_conv = convolve(sum_conv, normal_gaussian, mode='full')[:time_sample]
                if current_state == N - 1:
                    rul_matrix[i, :] = (1 - param_tau) * mod_d_i[:time_sample] + param_tau * normal_gaussian
                else:
                    first_term = (1 - param_tau) * mod_d_i[:time_sample]
                    second_term = param_tau * sum_conv[:time_sample]
                    rul_current = first_term + second_term
                    rul_matrix[i, :] = rul_current
        rul_mean = []
        rul_upper_bound = []
        rul_lower_bound = []
        for i in range(rul_matrix.shape[0]):
            rul_pdf_current = rul_matrix[i, :]
            rul_value = calculate_expected_value(rul_pdf_current)
            if np.isnan(rul_value) or rul_value == 0:
                rul_mean.append(0)
                rul_upper_bound.append(0)
                rul_lower_bound.append(0)
                break
            else:
                rul_mean.append(int(rul_value))
                lower_bound, upper_bound = calculate_cdf(rul_pdf_current, confidence)
                rul_upper_bound.append(upper_bound)
                rul_lower_bound.append(lower_bound)
        return rul_mean, rul_upper_bound, rul_lower_bound

    def prognostics(self, data, time_sample=2000, plot_rul=False):
        rul_mean_all = {}
        rul_upper_bound_all = {}
        rul_lower_bound_all = {}
        for k in data.keys():
            viterbi = self.predict(data[k])
            rul_mean, rul_upper, rul_lower = self.RUL(viterbi, time_sample)
            rul_mean_all[k] = rul_mean
            rul_upper_bound_all[k] = rul_upper
            rul_lower_bound_all[k] = rul_lower
            if plot_rul:
                plt.figure(figsize=(10, 6))
                plt.plot(rul_mean, linewidth=3, color='blue', label='Predicted')
                plt.plot(range(len(data[k]), 0, -1), color='black', linewidth=2.5, linestyle='dashed', label='Real')
                plt.fill_between(range(len(rul_mean)), rul_lower, rul_upper, color='blue', alpha=0.2)
                plt.xlim(0, max(range(len(rul_mean))))
                plt.ylim(0, max(rul_upper))
                plt.xlabel('Time [s]')
                plt.ylabel('RUL')
                plt.tick_params(axis='both', which='both')
                plt.legend()
                plt.show(block=True)
        return rul_mean_all, rul_upper_bound_all, rul_lower_bound_all

