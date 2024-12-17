import os

from scipy.stats import multivariate_normal, norm, geom
from scipy.special import logsumexp
from scipy.signal import convolve
from sklearn import cluster
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
from itertools import zip_longest
import matplotlib.pyplot as plt
from math import ceil
import copy
from ab import _forward, _backward, _u_only

import smoothed as core

from utils import *
from plot import *

np.seterr(invalid='ignore', divide='ignore')


class HSMM:
    def __init__(self, n_states=2, n_durations=5, n_iter=20, tol=1e-2, left_to_right=False, obs_state_len=None,
                 f_value=None, random_state=None, name=""):

        """
        :n_states (int): Number of hidden states. Must be ≥ 2.
        :n_durations (int): Number of duration categories per state. Must be ≥ 1.
        :n_iter (int): Maximum number of iterations for training.
        :tol (float): Convergence threshold for stopping the training.
        :left_to_right (bool): Indicates whether the model follows a left-to-right topology.
        :obs_state_len (int, optional): Length of the observed state (required if f_value is provided).
        :f_value (int/float, optional): Final observed value of the state (required if obs_state_len is provided).
        :random_state (int/None, optional): Seed for reproducibility.
        :name (str, optional): Name of the model. Defaults to "hsmm" if not provided.
        """

        if not n_states >= 2:
            raise ValueError("number of states (n_states) must be at least 2")
        if not n_durations >= 1:
            raise ValueError("number of durations (n_durations) must be at least 1")
        if len(name) == 0:
            name = f"hsmm"
            print(f"Model name not provided. Default name: {name}")

        if obs_state_len is not None and f_value is not None:
            self.last_observed = True
        elif obs_state_len is not None and f_value is None:
            raise ValueError("provide the observed state's final value")
        elif obs_state_len is None and f_value is not None:
            raise ValueError("provide the observed state's length")
        else:
            self.last_observed = False

        # create the folders
        create_folders()

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
        self.score_per_iter = None
        self.score_per_sample = None
        self.bic_score = None

    def _init(self, X=None):
        """
        Initializes model parameters if they are not already set.
        For left-to-right models:
        Sets the initial state to 1 (pi[0] = 1) and enforces forward transitions.
        For other topologies:
        Distributes probabilities evenly among states.
        Disallows self-transitions (tmat[i, i] = 0).
        
        Parameters
        ----------
        X : (dict): Observation dataset.

        Returns
        -------
        None.

        """
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


    def _init_mc(self):
        """
        Initialize the model parameters for MC sampling
        (implemented in child class (obs_state_len needs to be defined as well))
        :param X:
        :return:
        """
        pass  # implemented in subclass

    def _check(self):
        """
        Validates the initialized parameters:

        Ensures starting probabilities (pi) sum to 1.
        Checks transition matrix (tmat) shape and sums across rows.
        Verifies duration probabilities.

        Returns
        -------
        None.

        """
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

    def _dur_init(self, *args):
        """
        Ιnitializes duration parameters if there are none yet
        arguments: (self)
        return: None
        """
        pass  # implemented in subclass

    def _dur_check(self, *args):
        """
        Checks if properties of duration parameters are satisfied
        arguments: (self)
        return: None
        """
        pass  # implemented in subclass

    def _dur_probmat(self, *args):
        """
        Compute the probability per state of each duration
        arguments: (self)
        return: duration probability matrix
        """
        pass  # implemented in subclass

    # _dur_mstep: perform m-step for duration parameters
    def _dur_mstep(self, *args):
        """
        arguments: (self, new_dur)
        return: None
        > compute the duration parameters
        """
        pass  # implemented in subclass

    def _emission_logl(self, *args):
        """
        Compute the log-likelihood of each observation under each state.
        arguments: (self, X)
        return: logframe
        """
        pass  # implemented in subclass

    def _emission_pre_mstep(self, *args):
        """
        Prepare for emission parameters re-estimation.
        (process gamma and save output to emission_var)
        arguments: (self, gamma, emission_var)
        return: None
        """
        pass  # implemented in subclass

    def _emission_mstep(self, *args):
        """
        Compute the emission parameters.
        arguments: (self, X, emission_var)
        return: None
        """
        pass  # implemented in subclass

    def _state_sample(self, *args):
        """
        Genrate observation for given state
        arguments: (self, state, random_state=None)
        return: np.ndarray of length equal to dimension of observation
        """
        pass  # implemented in subclass

    def sample(self, n_samples=5, random_state=None):
        """
        Generates a sequence of observations and corresponding state sequences performing a random walk on the model
        (MC Sampling).

        Parameters
        ----------
        :n_samples (int): Number of observations to generate.
        :random_state (int/None): Seed for reproducibility.

        Returns
        -------
        :ctr_sample (int): Number of samples generated.
        :X (ndarray): Generated observation sequence.
        :state_sequence (ndarray): State sequence corresponding to the observations.
        """

        self._init(None)  # see "note for programmers" in init() in GaussianHSMM
        # self._check()
        # setup random state
        if random_state is None:
            random_state = self.random_state
        rnd_checked = np.random.default_rng(random_state)
        pi_cdf = np.cumsum(self.pi)
        tmat_cdf = np.cumsum(self.tmat, axis=1)
        dur_cdf = np.cumsum(self._dur_probmat(), axis=1)
        # for first state
        currstate = (pi_cdf > rnd_checked.random()).argmax()  # argmax() returns only the first occurrence
        currdur = (dur_cdf[currstate] > rnd_checked.random()).argmax() + 1
        if currdur > n_samples:
            print(f"SAMPLE{self.name}: n_samples is too small to contain the first state duration.")
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

    def MC_dataset(self, num, timesteps):
        """
        Generates a dataset of a number of observations and corresponding state sequences utilizing the sample method .

        Parameters
        ----------
        :num (int): Number of samples to generate.
        :timesteps (int): Number of maximum timesteps for each sample.

        Returns
        -------
        :obs (dict[str, List[int]]): A dictionary with trajectory observations.
        :states (dict[str, List[int]]): A dictionary with the corresponding states for each trajectory.
        """
        self._init_mc()
        obs, states = {}, {}

        for i in range(num):
            sample = self.sample(timesteps)
            _, obs1, states1 = sample

            for j in range(len(states1)):
                if states1[j] > states1[j + 1]:
                    idx = j
                    break
            obs.update({f'traj_{i + 1}': list(obs1[:idx + 1, 0])})
            states.update({f'traj_{i + 1}': list(states1[:idx + 1])})

        return obs, states

    def _core_u_only(self, logframe):
        """
        Computes intermediate matrix u for duration probabilities.
        
        Parameters
        ----------
        
        :logframe (numpy.ndarray): A 2D array of log-likelihood values for each observation under each state. Shape: (n_samples, n_states).
        
        Returns
        -------
        
        :u (numpy.ndarray): A 3D array of intermediate values computed for each sample, state, and duration. Shape: (n_samples, n_states, n_durations).
        
        """
        n_samples = logframe.shape[0]
        u = np.empty((n_samples, self.n_states, self.n_durations))
        _u_only(n_samples, self.n_states, self.n_durations,
                logframe, u)
        return u

    def _core_forward(self, u, logdur):
        """
        Performs the forward step of the HSMM algorithm using duration and transition probabilities.
        
        Parameters
        ----------
        :u (numpy.ndarray): Intermediate values computed from _core_u_only. Shape: (n_samples, n_states, n_durations).
        :logdur (numpy.ndarray): Logarithm of the duration probabilities for each state. Shape: (n_states, n_durations).
        
        Returns
        -------
        :eta (numpy.ndarray): Smoothed probabilities for states and durations at each sample. Shape: (n_samples + 1, n_states, n_durations).
        :xi (numpy.ndarray): Transition probabilities between states at each step. Shape: (n_samples + 1, n_states, n_states).
        :alpha (numpy.ndarray): Forward probabilities for each state at each sample. Shape: (n_samples, n_states).
        """

        n_samples = u.shape[0]
        eta_samples = n_samples
        eta = np.empty((eta_samples + 1, self.n_states, self.n_durations))  # +1
        xi = np.empty((n_samples + 1, self.n_states, self.n_states))  # +1
        alpha = _forward(n_samples, self.n_states, self.n_durations,
                         log_mask_zero(self.pi),
                         log_mask_zero(self.tmat),
                         logdur, 0, 0, eta, u, xi)
        return eta, xi, alpha

    def _core_backward(self, u, logdur):
        """
        Implements the backward algorithm for the HSMM.
        Computes backward probabilities and intermediate variables for scaling.

        Parameters
        ----------
        :u (numpy.ndarray): Scaled forward probabilities from _core_u_only.
        :logdur (numpy.ndarray): Logarithmic duration probability matrix.

        Returns
        -------
        :beta (numpy.ndarray): Backward probabilities for each state.
        :betastar (numpy.ndarray): Scaled backward probabilities.
        """

        n_samples = u.shape[0]
        beta = np.empty((n_samples, self.n_states))
        betastar = np.empty((n_samples, self.n_states))
        _backward(n_samples, self.n_states, self.n_durations,
                  log_mask_zero(self.pi),
                  log_mask_zero(self.tmat),
                  logdur, 0, beta, u, betastar)
        return beta, betastar

    # implemented in Cython
    def _core_smoothed(self, beta, betastar, eta, xi):
        """
        Combines forward and backward variables to compute the smoothed probabilities.
        Implemented in Cython with core._smoothed.

        Parameters
        ----------
        :beta (numpy.ndarray): Backward probabilities for each state.
        :betastar (numpy.ndarray): Scaled backward probabilities.
        :eta (numpy.ndarray): Transition probabilities.
        :xi (numpy.ndarray): Joint probabilities of transitions.

        Returns
        -------
        :gamma (numpy.ndarray): Smoothed probabilities.
        """

        n_samples = beta.shape[0]
        gamma = np.empty((n_samples, self.n_states))
        core._smoothed(n_samples, self.n_states, self.n_durations,
                       beta, betastar, 0, eta, xi, gamma)
        return gamma

    def _core_viterbi(self, u, logdur):
        """
        Implements the Viterbi algorithm for finding the most probable state sequence given the observations.

        Parameters
        ----------
        :u (numpy.ndarray): Scaled forward probabilities from _core_u_only.
        :logdur (numpy.ndarray): Logarithmic duration probability matrix.

        Returns
        -------
        :state_sequence (numpy.ndarray): The most probable sequence of states.
        :state_logl (float): Log-likelihood of the state sequence.
        """

        n_samples = u.shape[0]
        state_sequence, state_logl = core._viterbi(n_samples, self.n_states, self.n_durations,
                                                   log_mask_zero(self.pi),
                                                   log_mask_zero(self.tmat),
                                                   logdur, 0, 0, u)
        return state_sequence, state_logl

    def score(self, X):
        """
        Computes the log-likelihood of the observation sequences under the current model.

        Parameters
        ----------
        :X (numpy.ndarray): Observation sequences.

        Returns
        -------
        :score (float): Total log-likelihood of the observations.
        """

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

    def predict(self, X):
        """
        Predicts the most likely hidden state sequence for a given observation sequence using the Viterbi algorithm.

        Parameters
        ----------
        :X (numpy.ndarray): Observation sequences.

        Returns
        -------
        :state_sequence (numpy.ndarray): Predicted state sequence.
        :state_logl (float): Log-likelihood of the predicted state sequence.
        """

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

    def fit(self, X, save_iters=False):
        """
        Trains the model using the Expectation-Maximization (EM) algorithm.

        Parameters
        ----------
        :X (dict): A dictionary where keys are trajectory identifiers and values are observation sequences (numpy arrays).
        :save_iters (bool, optional): Whether to save the model after each iteration. Defaults to False.

        Returns
        -------
        :self (GaussianHSMM): The trained model.
        """

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
                print(f"\nFIT{self.name}: converged at loop {itera + 1} with score: {score}.")
                break
            elif itera > 0 and (np.isnan(score) or np.isinf(score)):
                print("\nThere is no possible solution. Try different parameters.")
                break

            else:
                score_per_iter.append(score)
                old_score = score

            # save the previous version of the model prior to updating
            if save_iters:
                path = os.path.join(os.getcwd(), 'results', 'models', f'{self.name}_iter_{str(itera + 1)}.txt')
                with open(path, 'wb') as f:
                    pickle.dump(self, f)
                    print(f"\nModel saved at {path}")

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

            print(f"\nFIT {self.name}: re-estimation complete for loop {itera + 1} with score: {score}.")

        score_per_sample = np.array(score_per_sample).reshape((-1, len(X))).T
        score_per_iter = np.array(score_per_iter).reshape(len(score_per_iter), 1)
        print(sorted(range(len(self.mean.tolist())), key=self.mean.tolist().__getitem__))
        if self.last_observed:
            self.dur[-1, self.obs_state_len] = 0
            self.dur[-1, self.obs_state_len - 1] = 1
        # return fitted model for joblib
        self.score_per_iter = score_per_iter
        self.score_per_sample = score_per_sample
        self.bic(X)

        return self

    def bic(self, train):
        """
        Computes the Bayesian Information Criterion (BIC) score to evaluate model performance.

        Parameters
        ----------
        :train (dict): Observation sequences used for training.

        Returns
        -------
        :score (float): The BIC score for the model.
        """

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
        """
        Fits multiple models with different numbers of states, evaluates them using BIC, and selects the best one.

        Parameters
        ----------
        :X (dict): Observation sequences (same format as fit).
        :states (list[int]): List of state counts to evaluate.
        :return_models (bool, optional): Whether to return all trained models. Defaults to False.

        Returns
        -------
        :self (GaussianHSMM): The best-performing model.
        :bic (list[float]): BIC scores for each fitted model.
        :(optional) models (dict): All trained models, returned if return_models=True.
        """

        bic = []
        keys = list(X.keys())
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
                history = X[keys[i]]
                history = np.array(history).reshape((len(history), 1))
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

    def RUL(self, viterbi_states, max_samples, equation=1):
        """
        Estimates the Remaining Useful Life (RUL) for a given state history using convolution of duration probabilities.

        Parameters
        ----------
        :viterbi_states (numpy.ndarray): Sequence of Viterbi states representing the history of hidden states.
        :max_samples (int): Maximum length of RUL to consider.
        :equation (int, optional): Equation type for RUL estimation. Default is 1.
        
        Returns
        -------
        :RUL (numpy.ndarray): RUL probability distribution for each timestep.
        :mean_RUL (numpy.ndarray): Mean RUL for each timestep.
        :UB_RUL (numpy.ndarray): Upper bound of the RUL distribution.
        :LB_RUL (numpy.ndarray): Lower bound of the RUL distribution.

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
                LB_RUL[i] = LinearRegression().fit(X, y).predict(np.asarray(0.05).reshape(-1, 1))

                # LB RUL
                X, y = [], []
                for l, value in enumerate(cdf_curr_RUL):
                    if value > 0.95:
                        X = [cdf_curr_RUL[l - 1], value]
                        y = [l - 1, l]
                        break
                X = np.asarray(X).reshape(-1, 1)
                y = np.asarray(y).reshape(-1, 1)
                UB_RUL[i] = LinearRegression().fit(X, y).predict(np.asarray(0.95).reshape(-1, 1))

                # mean RUL
                value = np.arange(0, RUL.shape[1])
                mean_RUL[i] = sum(RUL[i, :] * value)

            elif not available_states.size > 0 and self.last_observed:
                RUL[i, :], mean_RUL[i], UB_RUL[i], LB_RUL[i] = 0, 0, 0, 0
                mean_RUL = np.hstack((np.delete(mean_RUL, mean_RUL == 0), np.array((0))))
                UB_RUL = np.hstack((np.delete(UB_RUL, UB_RUL == 0), np.array((0))))
                LB_RUL = np.hstack((np.delete(LB_RUL, LB_RUL == 0), np.array((0))))
                break

        return RUL, mean_RUL, UB_RUL, LB_RUL

    def prognostics(self, data, max_samples=None, plot_rul=True, get_metrics=True, equation=1):
        """
        Performs prognostics for given degradation histories, estimating RUL and saving the results.

        Parameters
        ----------
        :data (dict): A dictionary where keys are trajectory IDs and values are degradation histories.
        :max_samples (int, optional): Maximum length of RUL. Defaults to 10x the maximum trajectory length.
        :plot_rul (bool, optional): Whether to plot RUL results for each sample. Default is True.
        :get_metrics (bool, optional): Whether to compute and save evaluation metrics. Default is True.
        :equation (int, optional): Equation type for RUL estimation. Default is 1.
        
        Returns
        -------
        No direct return. Saves:
        :PDF RUL distributions.
        :Mean RUL per step.
        :Upper and lower RUL bounds.
        :Evaluation metrics (if get_metrics=True).
        :RUL plots (if plot_rul=True).

        """

        if self.max_len is None:
            keys = list(data.keys())
            lens = []
            for traj in keys:
                lens.append(len(data[traj]))

            self.max_len = max(lens)
        path = os.path.join(os.getcwd(), 'results')
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

        pdf_ruls_all = {f"traj_{j}": {} for j in range(len(viterbi_list))}
        mean_rul_per_step, upper_rul_per_step, lower_rul_per_step = {}, {}, {}

        for i in range(len(viterbi_states_all)):
            viterbi_single_state = get_single_history_states(viterbi_states_all,
                                                             i,
                                                             last_state=self.n_states - 1
                                                             )
            # viterbi_single_state=np.array(viterbi_single_state).reshape((len(viterbi_single_state),1))
            RUL_pred, mean_RUL, UB_RUL, LB_RUL = self.RUL(viterbi_single_state,
                                                          max_samples=max_samples,
                                                          equation=equation,
                                                          )

            for j in range(RUL_pred.shape[0]):
                pdf_ruls_all[f"traj_{i}"][f"timestep_{j}"] = RUL_pred[j, :].copy()
                mean_rul_per_step[f"traj_{i}"] = mean_RUL.copy()
                upper_rul_per_step[f"traj_{i}"] = UB_RUL.copy()
                lower_rul_per_step[f"traj_{i}"] = LB_RUL.copy()
            if plot_rul:
                fig_path = os.path.join(path, 'figures', f'{self.name}_RUL_plot_traj_{i + 1}.png')
                plot_ruls(mean_RUL, UB_RUL, LB_RUL, fig_path)
            if get_metrics:
                true_rul_dict = {}
                for key in mean_rul_per_step.keys():
                    true_rul_dict[key] = len(mean_rul_per_step[key])
                df_results = evaluate_test_set(mean_rul_per_step, upper_rul_per_step, lower_rul_per_step, true_rul_dict)

        path_mean_rul = os.path.join(path, 'dictionaries', f"mean_rul_per_step_{self.name}.json")
        path_pdf_rul = os.path.join(path, 'dictionaries', f"pdf_ruls_{self.name}.json")
        path_upper_rul = os.path.join(path, 'dictionaries', f"upper_ruls_{self.name}.json")
        path_lower_rul = os.path.join(path, 'dictionaries', f"lower_ruls_{self.name}.json")

        with open(path_mean_rul, "w") as fp:
            json.dump(mean_rul_per_step, fp, cls=NumpyArrayEncoder)

        with open(path_pdf_rul, "w") as fp:
            json.dump(pdf_ruls_all, fp, cls=NumpyArrayEncoder)

        with open(path_upper_rul, "w") as fp:
            json.dump(upper_rul_per_step, fp, cls=NumpyArrayEncoder)

        with open(path_lower_rul, "w") as fp:
            json.dump(lower_rul_per_step, fp, cls=NumpyArrayEncoder)

        print(f"\nPrognostics complete. Results saved to: {path}")
        if plot_rul:
            print(f"\nRUL plots saved to: {os.path.join(path, 'dictionaries', 'figures')}")

        if get_metrics:
            df_results.to_csv(f'{path}/df_results.csv', index=False)
            print(f'\n Metrics saved to: {path}')
            print(f'\n {df_results}')

    def save_model(self):
        """
        Saves the current model state to a file.

        Returns
        -------
        None.
        """

        path = os.path.join(os.getcwd(), 'results', 'models', f'{self.name}.txt')
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)
            print(f"Model saved to {path}.")

    def load_model(self, model_name):
        """
        Loads a previously saved model state from a file.

        Parameters
        ----------
        model_name (str): Name of the model file to load (without extension).

        Returns
        -------
        None.
        """

        path = os.path.join(os.getcwd(), 'results', 'models', f'{model_name}.txt')
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            self.__dict__.update(obj)


class GaussianHSMM(HSMM):
    """
    The GaussianHSMM class models Hidden Semi-Markov processes with Gaussian-distributed emissions.
    It supports explicit duration modeling, and it can handle left-to-right or arbitrary state transitions.
    K-means clustering is used for initialization.
    """

    def __init__(self, n_states=2, n_durations=5, n_iter=100, tol=0.5, left_to_right=True, obs_state_len=None,
                 f_value=None, random_state=None, name="",
                 kmeans_init='k-means++', kmeans_n_init='auto'):
        """
        Parameters
        ----------
        :n_states (int): Number of hidden states in the model. Default is 2.
        :n_durations (int): Maximum duration for each state. Default is 5.
        :n_iter (int): Maximum number of iterations for model fitting. Default is 20.
        :tol (float): Convergence threshold for the EM algorithm. Default is 1e-2.
        :left_to_right (bool): If True, constrains transitions to progress in a left-to-right manner. Default is True for prognostics.
        :obs_state_len (int, optional): Length of observed state (relevant in specific configurations).
        :f_value (float, optional): Emission value for the final state, if applicable.
        :random_state (int or RandomState instance, optional): Seed or random state for reproducibility.
        :name (str): Name identifier for the model.
        :kmeans_init (str): Initialization method for K-means clustering ('k-means++' or 'random'). Default is 'k-means++'.
        :kmeans_n_init (int or str): Number of initializations for K-means clustering. Default is 'auto'.
        """

        super().__init__(n_states, n_durations, n_iter, tol, left_to_right, obs_state_len,
                         f_value, random_state, name)
        self.kmeans_init = kmeans_init
        self.kmeans_n_init = kmeans_n_init

    def _init(self, X=None):
        """
        Initializes model parameters based on input data X.
        
        Parameters
        ----------
        :X (numpy.ndarray, optional): Observations to initialize the model. If None, defaults to 1D Gaussian emissions.

        Returns
        -------
        None.
        """

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

    def _init_mc(self):
        """
        Initializes model parameters for the Monte Carlo Sampling example
        :param X:
        :return:
        """
        pi = np.zeros(self.n_states)
        pi[0] = 1

        # durations
        dur = np.zeros((self.n_states, self.n_durations))
        mean_dur = int(self.n_durations / 2)
        if mean_dur<=0:
            raise ValueError("Mean duration must be greater than 0. Try increasing the n_durations or decreasing the n_states.")
        std_dur = int(5 * (self.n_states - 2) + 1)

        for i in range(len(dur) - 1):
            x = np.linspace(0, mean_dur * 2, mean_dur * 2)
            for k in range(len(x)):
                dur[i, k] = norm(mean_dur, std_dur).pdf(x[k,])
                dur[i, ((x.shape[0] // 2) - 1)] += 1 - dur[i].sum()
            mean_dur -= 20
            std_dur -= 5

        dur[-1, self.obs_state_len-1] = 1

        for i in range(len(dur)):
            dur[i, ((dur.shape[1] // 2) - 1)] += 1 - dur[i].sum()

        tmat = np.zeros((self.n_states, self.n_states))
        for i in range(len(tmat)):
            for j in range(len(tmat[i]) - 1):
                if i == j and j < len(tmat[i]) - 2:
                    tmat[i, j + 1] = 1

                elif i == j and j == len(tmat[i]) - 2:
                    tmat[i, j + 1] = 1

        tmat[-1, -2] = 1

        mean_v = [i * 10 for i in range(1, self.n_states)]
        mean_v.append(mean_v[-1] + 15)
        mean = np.array(mean_v).reshape(-1, 1)  # shape should be (n_states, n_dim)

        covmat_v = [6. for i in range(self.n_states - 1)]
        covmat_v.append(0.1)
        covmat = np.array(covmat_v).reshape((self.n_states, 1, 1))

        self.f_value = mean_v[-1]
        self.pi = pi
        self.tmat = tmat
        self.mean = mean
        self.covmat = covmat
        self.dur = dur

    def _check(self):
        """
        Performs validation checks to ensure model parameters are consistent.

        Returns
        -------
        None.
        """

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
        """
        Initializes the duration probability matrix self.dur.


        Returns
        -------
        None.
        """

        # non-parametric duration
        if not hasattr(self, "dur") and not self.last_observed:
            self.dur = np.full((self.n_states, self.n_durations), 1.0 / self.n_durations)

        elif not hasattr(self, "dur") and self.last_observed:
            self.dur = np.zeros((self.n_states, self.n_durations))
            self.dur[:-1, 1:].fill(1.0 / (self.n_durations - 1))
            self.dur[-1, self.obs_state_len] = 1

    def _dur_check(self):
        """
        Validates the duration probability matrix self.dur.

        Returns
        -------
        None.
        """

        self.dur = np.asarray(self.dur)
        if self.dur.shape != (self.n_states, self.n_durations):
            raise ValueError("duration probabilities (self.dur) must have shape ({}, {})"
                             .format(self.n_states, self.n_durations))
        if not np.allclose(self.dur.sum(axis=1), 1.0):
            raise ValueError("duration probabilities (self.dur) must add up to 1.0")

    def _dur_probmat(self):
        """
        Returns the duration probability matrix self.dur.
        """
        # non-parametric duration
        return self.dur

    def _dur_mstep(self, new_dur):
        """
        Performs the M-step update for the duration probabilities.
        
        Parameters
        ----------
        new_dur (numpy.ndarray): Updated duration probabilities.

        Returns
        -------
        None.
        """

        # non-parametric duration
        self.dur = new_dur

    def _emission_logl(self, X):
        """
        Calculates the log-likelihood of the emissions given the observations.
        
        Parameters
        ----------
        :X (numpy.ndarray): Observations.
        
        Returns
        -------
        :logframe (numpy.ndarray): Log-likelihood of each observation under each state.
        """
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
        """
        Performs the M-step update for emission parameters.

        Parameters
        ----------
        :X (numpy.ndarray): Observations.
        :emission_var (numpy.ndarray): Responsibilities or posteriors for each observation-state pair.
        :inplace (bool, optional): If True, updates parameters in-place. If False, returns updated parameters.

        Returns (if inplace=False)
        -------
        :mean (numpy.ndarray): Updated means for each state.
        :covmat (numpy.ndarray): Updated covariance matrices for each state.
        """

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
        """
        Generates a sample from the Gaussian distribution of a specified state.
        
        Parameters
        ----------
        :state (int): Index of the state to sample from.
        :random_state (int or RandomState, optional): Random seed or state for reproducibility.
        
        Returns
        -------
        :sample (numpy.ndarray): Sampled observation.
        """

        rnd_checked = np.random.default_rng(random_state)
        return rnd_checked.multivariate_normal(self.mean[state], self.covmat[state])


######HMM
class HMM:
    def __init__(self, n_states=2, n_obs_symbols=30, n_iter=100, tol=1e-2, left_to_right=True, name=""):
        """
        :n_states (int): The number of states in the HMM. Must be ≥ 2.
        :n_obs_symbols (int): The number of observation symbols.
        :n_iter (int): The maximum number of iterations for training. Default is 20.
        :tol (float): Tolerance for convergence during training. Default is 1e-2.
        :left_to_right (bool): Whether the HMM uses a left-to-right structure. Default is True for use in prognostics.
        :name (str): Name of the model. Default is "hmm" if no name is provided.
        """

        create_folders()

        if not n_states >= 2:
            raise ValueError("number of states (n_states) must be at least 2")
        if len(name) == 0:
            name = f"hmm"
            print(f"Model name not provided. Default name: {name}")

        self.max_len = None
        self.n_states = n_states
        self.n_iter = n_iter
        self.tol = tol
        self.left_to_right = left_to_right
        self.n_obs_symbols = n_obs_symbols
        self.name = name

    def _init(self, X=None):
        """
        Initializes transition and emission matrices based on model structure (left_to_right).
        Parameters
        -------
        : X (dict): Dataset of trajectories for determining the maximum sequence length. The default is None.
        Returns
        -------
        None.

        """
        if not hasattr(self, "ini_tr") and not self.left_to_right:
            self.ini_tr = np.full((self.n_states, self.n_states), 1.0 / (self.n_states))

        elif not hasattr(self, "ini_tr") and self.left_to_right:
            self.ini_tr = np.zeros((self.n_states, self.n_states))
            for i in range(self.n_states - 1):
                if i == self.n_states - 1:
                    self.ini_tr[i, i - 1:i + 1] = 0.5
                else:
                    self.ini_tr[i, i:i + 2] = 0.5
            self.ini_tr[-1, -1] = 1

        if not hasattr(self, "ini_emi") and not self.left_to_right:
            self.ini_emi = np.full((self.n_states, self.n_obs_symbols), 1.0 / (self.n_obs_symbols))

        elif not hasattr(self, "ini_emi") and self.left_to_right:
            self.ini_emi = np.zeros((self.n_states, self.n_obs_symbols))
            prob = 1 / (self.n_obs_symbols - 1)
            for row in range(self.ini_emi.shape[0] - 1):
                for column in range(self.ini_emi.shape[1] - 1):
                    self.ini_emi[row, column] = prob
            self.ini_emi[self.n_states - 1, self.n_obs_symbols - 1] = 1

        if X is not None and self.max_len is None:
            keys = list(X.keys())
            lens = []
            for traj in keys:
                lens.append(len(X[traj]))

            self.max_len = max(lens)

    def _init_mc(self):
        """
        Initializes the model parameters for the Monte Carlo Sampling example.
        :return:
        """
        tr = np.zeros((self.n_states, self.n_states))
        for i in range(self.n_states):
            if i < self.n_states - 1:
                tr[i, i] = np.random.uniform(0.95, 0.97)
                tr[i, i + 1] = 1 - tr[i, i]
            else:
                tr[i, i] = 1.0  # Make the last state absorbing
            self.tr = tr

        emi = np.zeros((self.n_states, self.n_obs_symbols))
        segment_size = self.n_obs_symbols // (self.n_states - 1)

        for i in range(self.n_states - 1):
            start_idx = i * segment_size
            end_idx = start_idx + segment_size
            if i == self.n_states - 2:
                end_idx = self.n_obs_symbols - 1  # Leave room for the last observation
            obs_indices = np.arange(start_idx, end_idx)
            center = (start_idx + end_idx - 1) / 2
            gaussian = np.exp(-0.5 * ((obs_indices - center) ** 2) / ((end_idx - start_idx) / 4))
            emi[i, start_idx:end_idx] = gaussian
            emi[i, :] /= emi[i, :].sum()

            # Last row: all zeros except 1 at the end
        emi[-1, -1] = 1.0
        self.emi = emi

    def fit(self, X, return_all_scores=False, save_iters=False):
        """
        Trains the HMM using the Baum-Welch algorithm.

        Parameters
        ----------
        :X (dict): Observations organized as { "traj_<index>": [sequence] }.
        :return_all_scores (bool): If True, returns log-likelihood scores for all iterations.
        :save_iters (bool): If True, saves the model at each iteration.

        Returns
        -------
        Trained HMM instance, optionally with log-likelihood scores.
        """

        self._init(X)
        score_per_iter = []
        score = 1
        emi, calc_emi = self.ini_emi.copy(), self.ini_emi.copy()
        tr, calc_tr = self.ini_tr.copy(), self.ini_tr.copy()
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
                tr, emi = baumwelch_method(self.n_states, self.n_obs_symbols, logPseq, fs, bs, scale, score, history,
                                           tr, emi,
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
                print(f"\nFIT {self.name}: converged at loop {itera + 1} with score: {score}.")
                converged = True
                self.tr = calc_tr
                self.emi = calc_emi
                break

            if save_iters:
                path = os.path.join(os.getcwd(), 'results', 'models', f'{self.name}_iter_{str(itera + 1)}.txt')
                with open(path, 'wb') as f:
                    pickle.dump(self, f)
                    print(f"\nModel saved at {path}")

        if not converged:
            print("\nThere is no possible solution. Try different parameters.")

        if return_all_scores:
            return self, score_per_iter
        return self

    def fit_bic(self, X, states, return_models=False):
        """
        Fits multiple HMMs using the Bayesian Information Criterion (BIC) to select the best model.
        
        Parameters
        ----------
        :X (dict): Observation dataset.
        :states (list): List of candidate numbers of states.
        :return_models (bool): If True, returns all trained models and BIC scores.
        
        Returns
        -------
        :Best HMM model based on BIC and the BIC scores.
        """

        bic = []
        models = {
            f"model_{i}": None for i in range(len(states))
        }

        n = 0
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
        print(f"\nBest model: {best_model.name} with {best_model.n_states} states.")
        if return_models:
            return best_model, models, bic

        return best_model, bic

    def decode(self, history, calc_emi, calc_tr):
        """
        Computes forward (fs) and backward (bs) probabilities for a given sequence.

        Parameters
        ----------
        :history (list): Observation sequence.
        :calc_emi (array): Current emission matrix.
        :calc_tr (array): Current transition matrix.

        Returns
        -------
        :pStates: Posterior probabilities for states.
        :pSeq: Log-probability of the sequence.
        :fs: Forward probabilities.
        :bs: Backward probabilities.
        :s: Scaling factors.

        """

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
        """
        Generates a random observation sequence and state sequence, starting from the initial state and terminating when the last state is reached.
        
        Parameters
        ----------
        None

        Returns
        -------
        :history (list): A list containing the generated sequence of observations, where each observation corresponds to a state in the sequence.
        :states (list): A list containing the sequence of states visited during the process, where each state is represented by its index.
        """

        history = []
        states = []

        # Cumulative transition and emission probabilities
        trc = np.cumsum(self.tr, axis=1)
        ec = np.cumsum(self.emi, axis=1)

        trc = trc / trc[:, -1][:, None]
        ec = ec / ec[:, -1][:, None]

        currentstate = 0
        while currentstate < self.n_states - 1:
            # Sample observation
            observation = np.searchsorted(ec[currentstate], np.random.rand())
            history.append(observation + 1)
            states.append(currentstate + 1)

            # Sample next state
            currentstate = np.searchsorted(trc[currentstate], np.random.rand())

        # Terminal state emits its fixed observation
        terminal_observation = np.argmax(self.emi[self.n_states - 1])
        history.append(terminal_observation + 1)
        states.append(currentstate + 1)

        return history, states

    def sample_dataset(self, n_samples):
        """
        Generates a dataset of sampled observation sequences.

        Parameters
        ----------
        :n_samples (int): Number of sequences to generate.

        Returns
        -------
        :obs: Generated observation sequences.
        :states_all: Corresponding state sequences.

        """

        self._init_mc()
        obs = {}
        states_all = {}
        for i in range(n_samples):
            sample = self.sample()
            history, states = sample
            while len(states) > 130 or len(states) < 100:
                sample = self.sample()
                history, states = sample
            obs[f'traj_{i}'] = history
            states_all[f'traj_{i}'] = states
        return obs, states_all

    def predict(self, history, return_score=False):
        """
        Performs Viterbi decoding to predict the most likely state sequence.

        Parameters
        ----------
        :history (list): Observation sequence.
        :return_score (bool): If True, returns the log-probability of the best state sequence.

        Returns
        -------
        :currentState: Most likely state sequence.
        :logP (optional): Log-probability of the predicted sequence.

        """

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

    def estimate(self, history, estimatedStates, return_matrices=False):
        """
        Estimates transition and emission matrices based on observed sequences and states.

        Parameters
        ----------
        :history (list): Observation sequence.
        :estimatedStates (list): Corresponding state sequence.
        :return_matrices (bool): If True, returns the matrices instead of updating the model.

        Returns
        -------
        :Updated HMM instance or matrices (tr, emi).

        """

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

    def RUL(self, estimatedStates, max_samples, confidence=0.95):
        """
        Estimates the remaining useful life of a system based on state sequence.

        Parameters
        ----------
        :estimatedStates (list): Sequence of estimated states.
        :max_samples (int): Maximum number of timesteps for RUL estimation.
        :confidence (float): Confidence level for bounds.

        Returns
        -------
        :rul_mean: Mean RUL estimates.
        :rul_upper_bound: Upper confidence bounds.
        :rul_lower_bound: Lower confidence bounds.
        :rul_matrix: RUL probability distributions.

        """

        N = max(estimatedStates) - 1
        rul_matrix = np.zeros((len(estimatedStates), max_samples))
        prev_state = 0  # aux variable
        tau = 0
        for i in range(len(estimatedStates)):
            current_state = estimatedStates[i] - 1
            if current_state == N:
                rul_matrix[i, :] = np.zeros(max_samples)
            else:
                if prev_state == current_state:
                    tau += 1
                else:
                    prev_state = current_state
                    tau = 1
                a_ii = self.tr[current_state, current_state]
                a_next = self.tr[current_state + 1, current_state + 1]
                x_d_i = np.arange(0, max_samples)
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
                mod_d_i = convolve(mod_d_i, normal_gaussian, mode='full')[:max_samples]
                sum_conv = geom.pmf(x_d_i, 1 - a_next)
                for j in range(current_state + 2, N):
                    d_j = geom.pmf(x_d_i, 1 - self.tr[j, j])
                    sum_conv = convolve(sum_conv, d_j, mode='full')
                sum_conv = convolve(sum_conv, normal_gaussian, mode='full')[:max_samples]
                if current_state == N - 1:
                    rul_matrix[i, :] = (1 - param_tau) * mod_d_i[:max_samples] + param_tau * normal_gaussian
                else:
                    first_term = (1 - param_tau) * mod_d_i[:max_samples]
                    second_term = param_tau * sum_conv[:max_samples]
                    rul_current = first_term + second_term
                    rul_matrix[i, :] = rul_current
        rul_mean, rul_upper_bound, rul_lower_bound = [], [], []

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
        return rul_mean, rul_upper_bound, rul_lower_bound, rul_matrix

    def prognostics(self, data, max_samples=None, plot_rul=True, get_metrics=True):
        """
        Performs prognostics and evaluates model performance.

        Parameters
        ----------
        :data (dict): Observation data for multiple trajectories.
        :max_samples (int): Maximum timesteps for RUL. Default is 10× the max sequence length.
        :plot_rul (bool): If True, saves RUL plots.
        :get_metrics (bool): If True, evaluates RUL predictions with metrics.
        
        Returns
        -------
        :None. Saves RUL estimates and metrics to files.
        
        """

        path = os.path.join(os.getcwd(), 'results')
        max_samples = ceil(self.max_len * 10) if max_samples is None else max_samples
        rul_mean_all, rul_upper_bound_all, rul_lower_bound_all = {}, {}, {}
        pdf_ruls_all = {f"traj_{i}": {} for i in range(
            len(data))}  # different initialization due to the structure of the dictionary cointaining the timesteps
        for index, k in enumerate(data.keys()):
            viterbi = self.predict(data[k])
            rul_mean, rul_upper, rul_lower, rul_pdf = self.RUL(viterbi, max_samples)
            rul_mean_all[k] = rul_mean
            rul_upper_bound_all[k] = rul_upper
            rul_lower_bound_all[k] = rul_lower
            for j in range(len(rul_mean)):
                pdf_ruls_all[k][f'timestep_{j}'] = rul_pdf[j, :]

            if plot_rul:
                fig_path = os.path.join(path, 'figures', f'{self.name}_RUL_plot_traj_{index + 1}.png')
                plot_ruls(rul_mean, rul_upper, rul_lower, fig_path)
        if get_metrics:
            true_rul_dict = {}
            for key in rul_mean_all.keys():
                true_rul_dict[key] = len(rul_mean_all[key])
            df_results = evaluate_test_set(rul_mean_all, rul_upper_bound_all, rul_lower_bound_all, true_rul_dict)

        path_mean_rul = os.path.join(path, 'dictionaries', f"mean_rul_per_step_{self.name}.json")
        path_pdf_rul = os.path.join(path, 'dictionaries', f"pdf_ruls_{self.name}.json")
        path_upper_rul = os.path.join(path, 'dictionaries', f"upper_ruls_{self.name}.json")
        path_lower_rul = os.path.join(path, 'dictionaries', f"lower_ruls_{self.name}.json")

        with open(path_mean_rul, "w") as fp:
            json.dump(rul_mean_all, fp, cls=NumpyArrayEncoder)

        with open(path_pdf_rul, "w") as fp:
            json.dump(pdf_ruls_all, fp, cls=NumpyArrayEncoder)

        with open(path_upper_rul, "w") as fp:
            json.dump(rul_upper_bound_all, fp, cls=NumpyArrayEncoder)

        with open(path_lower_rul, "w") as fp:
            json.dump(rul_lower_bound_all, fp, cls=NumpyArrayEncoder)

        print(f"\nPrognostics complete. Results saved to: {os.path.join(path, 'dictionaries')}")
        if plot_rul:
            print(f"\nRUL plots saved to: {os.path.join(path, 'dictionaries', 'figures')}")

        if get_metrics:
            df_results.to_csv(f'{path}/df_results.csv', index=False)
            print(f'\n Metrics saved to: {path}')
            print(f'\n {df_results}')

    def save_model(self):
        """
        Saves the trained model to a file.

        Returns
        -------
        :None. File saved in results/models.

        """

        path = os.path.join(os.getcwd(), 'results', 'models', f'{self.name}.txt')
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)
            print(f"Model saved to {path}.")

    def load_model(self, model_name):
        """
        Loads a saved model.

        Parameters
        ----------
        :model_name (str): Name of the model to load.

        Returns
        -------
        None.

        """

        path = os.path.join(os.getcwd(), 'results', 'models', f'{model_name}.txt')
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            self.__dict__.update(obj)
