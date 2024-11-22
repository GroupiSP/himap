import numpy as np
import scipy.stats

#todo: documentation for all of the functions
#todo: clean up and simplify the functions, erase redundant code

def init4MC(hsmm_class):
    # initial probability
    hsmm_class.pi = np.zeros(10)
    hsmm_class.pi[0] = 1

    # durations
    hsmm_class.dur = np.zeros((10, 60))
    x = np.linspace(0, 60, 60)
    y = np.zeros_like(x)

    for i in range(len(hsmm_class.dur)):
        for j in range(len(hsmm_class.dur[i])):
            hsmm_class.dur[i, j] = scipy.stats.norm(30, 10).pdf(x[j,])

    for i in range(len(hsmm_class.dur)):
        hsmm_class.dur[i, 29] += 1 - hsmm_class.dur[i].sum()

    # transition matrix
    num_of_states = 10
    hsmm_class.tmat = np.zeros((num_of_states, num_of_states))

    for i in range(len(hsmm_class.tmat)):
        for j in range(len(hsmm_class.tmat[i]) - 1):
            if i == j and j < len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 0.6
                hsmm_class.tmat[i, j + 2] = 0.4
            elif i == j and j == len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 1

    hsmm_class.tmat[-1, -2] = 1

    hsmm_class.mean = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])  # shape should be (n_states, n_dim)
    hsmm_class.mean = np.reshape(hsmm_class.mean, (-1, 1))
    hsmm_class.covmat = np.array([  # shape should be (n_states, n_dim, n_dim) -> array of square matrices
        [[10.]],
        [[10.]],
        [[10.]],
        [[10.]],
        [[10.]],
        [[10.]],
        [[10.]],
        [[10.]],
        [[10.]],
        [[10.]],
    ])


"""
This is used for testing. The init4fit functions are implemented in the HSMM custom classes with init4mon boolean
argument
"""


def init4fit(hsmm_class):
    # initial probability
    hsmm_class.pi = np.zeros(5)
    hsmm_class.pi[0] = 1

    # transition matrix
    num_of_states = 5
    hsmm_class.tmat = np.zeros((num_of_states, num_of_states))

    for i in range(len(hsmm_class.tmat)):
        for j in range(len(hsmm_class.tmat[i]) - 1):
            if i == j and j < len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 0.6
                hsmm_class.tmat[i, j + 2] = 0.4
            elif i == j and j == len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 1

    hsmm_class.tmat[-1, -1] = 1

    for i in range(len(hsmm_class.tmat)):
        for j in range(len(hsmm_class.tmat[i])):
            if hsmm_class.tmat[i, j] == 0:
                hsmm_class.tmat[i, j] = 1e-30

    for i in range(len(hsmm_class.tmat)):
        hsmm_class.tmat[i, 3] -= 1 - hsmm_class.tmat[i].sum()


def init4shortMC(hsmm_class):
    # initial probability
    hsmm_class.pi = np.zeros(5)
    hsmm_class.pi[0] = 1

    # durations
    hsmm_class.dur = np.zeros((5, 30))
    x = np.linspace(0, 30, 30)
    y = np.zeros_like(x)

    for i in range(len(hsmm_class.dur)):
        for j in range(len(hsmm_class.dur[i])):
            hsmm_class.dur[i, j] = scipy.stats.norm(30, 10).pdf(x[j,])

    for i in range(len(hsmm_class.dur)):
        hsmm_class.dur[i, 14] += 1 - hsmm_class.dur[i].sum()

    # transition matrix
    num_of_states = 5
    hsmm_class.tmat = np.zeros((num_of_states, num_of_states))

    for i in range(len(hsmm_class.tmat)):
        for j in range(len(hsmm_class.tmat[i]) - 1):
            if i == j and j < len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 0.6
                hsmm_class.tmat[i, j + 2] = 0.4
            elif i == j and j == len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 1

    hsmm_class.tmat[-1, -2] = 1

    hsmm_class.mean = np.array([10, 20, 30, 40, 50])  # shape should be (n_states, n_dim)
    hsmm_class.mean = np.reshape(hsmm_class.mean, (-1, 1))
    hsmm_class.covmat = np.array([  # shape should be (n_states, n_dim, n_dim) -> array of square matrices
        [[10.]],
        [[10.]],
        [[10.]],
        [[10.]],
        [[10.]],

    ])


def init4longMC(hsmm_class):
    # initial probability
    hsmm_class.pi = np.zeros(10)
    hsmm_class.pi[0] = 1

    # durations
    hsmm_class.dur = np.zeros((10, 200))
    x = np.linspace(0, 200, 200)
    y = np.zeros_like(x)

    for i in range(len(hsmm_class.dur)):
        for j in range(len(hsmm_class.dur[i])):
            hsmm_class.dur[i, j] = scipy.stats.norm(100, 10).pdf(x[j,])

    for i in range(len(hsmm_class.dur)):
        hsmm_class.dur[i, 99] += 1 - hsmm_class.dur[i].sum()

    # transition matrix
    num_of_states = 10
    hsmm_class.tmat = np.zeros((num_of_states, num_of_states))

    for i in range(len(hsmm_class.tmat)):
        for j in range(len(hsmm_class.tmat[i]) - 1):
            if i == j and j < len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 0.6
                hsmm_class.tmat[i, j + 2] = 0.4
            elif i == j and j == len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 1

    hsmm_class.tmat[-1, -2] = 1

    hsmm_class.mean = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])  # shape should be (n_states, n_dim)
    hsmm_class.mean = np.reshape(hsmm_class.mean, (-1, 1))
    hsmm_class.covmat = np.array([  # shape should be (n_states, n_dim, n_dim) -> array of square matrices
        [[10.]],
        [[10.]],
        [[10.]],
        [[10.]],
        [[10.]],
        [[10.]],
        [[10.]],
        [[10.]],
        [[10.]],
        [[10.]],
    ])


def init4durval(hsmm_class):
    # initial probability
    # initial probability
    hsmm_class.pi = np.zeros(5)
    hsmm_class.pi[0] = 1

    # durations
    hsmm_class.dur = np.zeros((5, 260))
    mean = 10
    std = 1

    for i in range(len(hsmm_class.dur)):
        x = np.linspace(0, mean * 2, mean * 2)
        for k in range(len(x)):
            hsmm_class.dur[i, k] = scipy.stats.norm(mean, std).pdf(x[k,])
            hsmm_class.dur[i, ((x.shape[0] // 2) - 1)] += 1 - hsmm_class.dur[i].sum()
        mean += 30
        std = +5

    for i in range(len(hsmm_class.dur)):
        hsmm_class.dur[i, ((hsmm_class.dur.shape[1] // 2) - 1)] += 1 - hsmm_class.dur[i].sum()

    # transition matrix
    # num_of_states = 5
    # hsmm_class.tmat = np.zeros((num_of_states, num_of_states))
    #
    # for i in range(len(hsmm_class.tmat)):
    #     for j in range(len(hsmm_class.tmat[i]) - 1):
    #         if i == j and j < len(hsmm_class.tmat[i]) - 2:
    #             hsmm_class.tmat[i, j + 1] = 0.6
    #             hsmm_class.tmat[i, j + 2] = 0.4
    #         elif i == j and j == len(hsmm_class.tmat[i]) - 2:
    #             hsmm_class.tmat[i, j + 1] = 1
    #
    # hsmm_class.tmat[-1, -2] = 1

    num_of_states = 5
    hsmm_class.tmat = np.zeros((num_of_states, num_of_states))

    for i in range(len(hsmm_class.tmat)):
        for j in range(len(hsmm_class.tmat[i]) - 1):
            if i == j and j < len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 1

            elif i == j and j == len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 1

    hsmm_class.tmat[-1, -2] = 1
    hsmm_class.mean = np.array([5, 15, 25, 35, 45])  # shape should be (n_states, n_dim)
    hsmm_class.mean = np.reshape(hsmm_class.mean, (-1, 1))
    hsmm_class.covmat = np.array([  # shape should be (n_states, n_dim, n_dim) -> array of square matrices
        [[6.]],
        [[6.]],
        [[6.]],
        [[6.]],
        [[6.]],

    ])


def init4durval_decr(hsmm_class):
    # initial probability
    # initial probability
    hsmm_class.pi = np.zeros(5)
    hsmm_class.pi[0] = 1

    # durations
    hsmm_class.dur = np.zeros((5, 260))
    mean = 130
    std = 21

    for i in range(len(hsmm_class.dur)):
        x = np.linspace(0, mean * 2, mean * 2)
        for k in range(len(x)):
            hsmm_class.dur[i, k] = scipy.stats.norm(mean, std).pdf(x[k,])
            hsmm_class.dur[i, ((x.shape[0] // 2) - 1)] += 1 - hsmm_class.dur[i].sum()
        mean -= 30
        std -= 5

    for i in range(len(hsmm_class.dur)):
        hsmm_class.dur[i, ((hsmm_class.dur.shape[1] // 2) - 1)] += 1 - hsmm_class.dur[i].sum()

    # transition matrix
    # num_of_states = 5
    # hsmm_class.tmat = np.zeros((num_of_states, num_of_states))
    #
    # for i in range(len(hsmm_class.tmat)):
    #     for j in range(len(hsmm_class.tmat[i]) - 1):
    #         if i == j and j < len(hsmm_class.tmat[i]) - 2:
    #             hsmm_class.tmat[i, j + 1] = 0.6
    #             hsmm_class.tmat[i, j + 2] = 0.4
    #         elif i == j and j == len(hsmm_class.tmat[i]) - 2:
    #             hsmm_class.tmat[i, j + 1] = 1
    #
    # hsmm_class.tmat[-1, -2] = 1

    num_of_states = 5
    hsmm_class.tmat = np.zeros((num_of_states, num_of_states))

    for i in range(len(hsmm_class.tmat)):
        for j in range(len(hsmm_class.tmat[i]) - 1):
            if i == j and j < len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 1

            elif i == j and j == len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 1

    hsmm_class.tmat[-1, -2] = 1
    hsmm_class.mean = np.array([10, 20, 30, 40, 50])  # shape should be (n_states, n_dim)
    hsmm_class.mean = np.reshape(hsmm_class.mean, (-1, 1))
    hsmm_class.covmat = np.array([  # shape should be (n_states, n_dim, n_dim) -> array of square matrices
        [[10.]],
        [[10.]],
        [[10.]],
        [[10.]],
        [[10.]],
    ])


def init4durval_observ_state_15(hsmm_class):
    # initial probability
    # initial probability
    hsmm_class.pi = np.zeros(6)
    hsmm_class.pi[0] = 1

    # durations
    hsmm_class.dur = np.zeros((6, 260))
    mean = 130
    std = 21

    for i in range(len(hsmm_class.dur) - 1):
        x = np.linspace(0, mean * 2, mean * 2)
        for k in range(len(x)):
            hsmm_class.dur[i, k] = scipy.stats.norm(mean, std).pdf(x[k,])
            hsmm_class.dur[i, ((x.shape[0] // 2) - 1)] += 1 - hsmm_class.dur[i].sum()
        mean -= 30
        std -= 5

    hsmm_class.dur[-1, 10] = 1

    for i in range(len(hsmm_class.dur)):
        hsmm_class.dur[i, ((hsmm_class.dur.shape[1] // 2) - 1)] += 1 - hsmm_class.dur[i].sum()

    # transition matrix
    # num_of_states = 5
    # hsmm_class.tmat = np.zeros((num_of_states, num_of_states))
    #
    # for i in range(len(hsmm_class.tmat)):
    #     for j in range(len(hsmm_class.tmat[i]) - 1):
    #         if i == j and j < len(hsmm_class.tmat[i]) - 2:
    #             hsmm_class.tmat[i, j + 1] = 0.6
    #             hsmm_class.tmat[i, j + 2] = 0.4
    #         elif i == j and j == len(hsmm_class.tmat[i]) - 2:
    #             hsmm_class.tmat[i, j + 1] = 1
    #
    # hsmm_class.tmat[-1, -2] = 1

    num_of_states = 6
    hsmm_class.tmat = np.zeros((num_of_states, num_of_states))

    for i in range(len(hsmm_class.tmat)):
        for j in range(len(hsmm_class.tmat[i]) - 1):
            if i == j and j < len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 1

            elif i == j and j == len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 1

    hsmm_class.tmat[-1, -2] = 1
    hsmm_class.mean = np.array([15, 25, 35, 45, 55, 5])  # shape should be (n_states, n_dim)
    hsmm_class.mean = np.reshape(hsmm_class.mean, (-1, 1))
    hsmm_class.covmat = np.array([  # shape should be (n_states, n_dim, n_dim) -> array of square matrices
        [[6.]],
        [[6.]],
        [[6.]],
        [[6.]],
        [[6.]],
        [[1]],

    ])


def init4durval_observ_state_5(hsmm_class):
    # initial probability
    # initial probability
    hsmm_class.pi = np.zeros(6)
    hsmm_class.pi[0] = 1

    # durations
    hsmm_class.dur = np.zeros((6, 260))
    mean = 130
    std = 21

    for i in range(len(hsmm_class.dur) - 1):
        x = np.linspace(0, mean * 2, mean * 2)
        for k in range(len(x)):
            hsmm_class.dur[i, k] = scipy.stats.norm(mean, std).pdf(x[k,])
            hsmm_class.dur[i, ((x.shape[0] // 2) - 1)] += 1 - hsmm_class.dur[i].sum()
        mean -= 30
        std -= 5

    hsmm_class.dur[-1, 10] = 1

    for i in range(len(hsmm_class.dur)):
        hsmm_class.dur[i, ((hsmm_class.dur.shape[1] // 2) - 1)] += 1 - hsmm_class.dur[i].sum()

    # transition matrix
    # num_of_states = 5
    # hsmm_class.tmat = np.zeros((num_of_states, num_of_states))
    #
    # for i in range(len(hsmm_class.tmat)):
    #     for j in range(len(hsmm_class.tmat[i]) - 1):
    #         if i == j and j < len(hsmm_class.tmat[i]) - 2:
    #             hsmm_class.tmat[i, j + 1] = 0.6
    #             hsmm_class.tmat[i, j + 2] = 0.4
    #         elif i == j and j == len(hsmm_class.tmat[i]) - 2:
    #             hsmm_class.tmat[i, j + 1] = 1
    #
    # hsmm_class.tmat[-1, -2] = 1

    num_of_states = 6
    hsmm_class.tmat = np.zeros((num_of_states, num_of_states))

    for i in range(len(hsmm_class.tmat)):
        for j in range(len(hsmm_class.tmat[i]) - 1):
            if i == j and j < len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 1

            elif i == j and j == len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 1

    hsmm_class.tmat[-1, -2] = 1
    hsmm_class.mean = np.array([5, 15, 25, 35, 45, -1])  # shape should be (n_states, n_dim)
    hsmm_class.mean = np.reshape(hsmm_class.mean, (-1, 1))
    hsmm_class.covmat = np.array([  # shape should be (n_states, n_dim, n_dim) -> array of square matrices
        [[6.]],
        [[6.]],
        [[6.]],
        [[6.]],
        [[6.]],
        [[1]],

    ])


def init4durval_observ_state_1(hsmm_class):
    # initial probability
    # initial probability
    hsmm_class.pi = np.zeros(6)
    hsmm_class.pi[0] = 1

    # durations
    hsmm_class.dur = np.zeros((6, 260))
    mean = 130
    std = 21

    for i in range(len(hsmm_class.dur) - 1):
        x = np.linspace(0, mean * 2, mean * 2)
        for k in range(len(x)):
            hsmm_class.dur[i, k] = scipy.stats.norm(mean, std).pdf(x[k,])
            hsmm_class.dur[i, ((x.shape[0] // 2) - 1)] += 1 - hsmm_class.dur[i].sum()
        mean -= 20
        std -= 5

    hsmm_class.dur[-1, 9] = 1

    for i in range(len(hsmm_class.dur)):
        hsmm_class.dur[i, ((hsmm_class.dur.shape[1] // 2) - 1)] += 1 - hsmm_class.dur[i].sum()

    # transition matrix
    # num_of_states = 5
    # hsmm_class.tmat = np.zeros((num_of_states, num_of_states))
    #
    # for i in range(len(hsmm_class.tmat)):
    #     for j in range(len(hsmm_class.tmat[i]) - 1):
    #         if i == j and j < len(hsmm_class.tmat[i]) - 2:
    #             hsmm_class.tmat[i, j + 1] = 0.6
    #             hsmm_class.tmat[i, j + 2] = 0.4
    #         elif i == j and j == len(hsmm_class.tmat[i]) - 2:
    #             hsmm_class.tmat[i, j + 1] = 1
    #
    # hsmm_class.tmat[-1, -2] = 1

    num_of_states = 6
    hsmm_class.tmat = np.zeros((num_of_states, num_of_states))

    for i in range(len(hsmm_class.tmat)):
        for j in range(len(hsmm_class.tmat[i]) - 1):
            if i == j and j < len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 1

            elif i == j and j == len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 1

    hsmm_class.tmat[-1, -2] = 1
    hsmm_class.mean = np.array([10, 20, 30, 40, 50, 65])  # shape should be (n_states, n_dim)
    hsmm_class.mean = np.reshape(hsmm_class.mean, (-1, 1))
    hsmm_class.covmat = np.array([  # shape should be (n_states, n_dim, n_dim) -> array of square matrices
        [[6.]],
        [[6.]],
        [[6.]],
        [[6.]],
        [[6.]],
        [[0.1]],

    ])


def init4durval_observ_state_1_long(hsmm_class):
    # initial probability
    # initial probability
    hsmm_class.pi = np.zeros(6)
    hsmm_class.pi[0] = 1

    # durations
    hsmm_class.dur = np.zeros((6, 260))
    mean = 100
    std = 21

    for i in range(len(hsmm_class.dur) - 1):
        x = np.linspace(0, mean * 2, mean * 2)
        for k in range(len(x)):
            hsmm_class.dur[i, k] = scipy.stats.norm(mean, std).pdf(x[k,])
            hsmm_class.dur[i, ((x.shape[0] // 2) - 1)] += 1 - hsmm_class.dur[i].sum()
        mean -= 15
        std -= 5

    hsmm_class.dur[-1, 10] = 1

    for i in range(len(hsmm_class.dur)):
        hsmm_class.dur[i, ((hsmm_class.dur.shape[1] // 2) - 1)] += 1 - hsmm_class.dur[i].sum()

    # transition matrix
    # num_of_states = 5
    # hsmm_class.tmat = np.zeros((num_of_states, num_of_states))
    #
    # for i in range(len(hsmm_class.tmat)):
    #     for j in range(len(hsmm_class.tmat[i]) - 1):
    #         if i == j and j < len(hsmm_class.tmat[i]) - 2:
    #             hsmm_class.tmat[i, j + 1] = 0.6
    #             hsmm_class.tmat[i, j + 2] = 0.4
    #         elif i == j and j == len(hsmm_class.tmat[i]) - 2:
    #             hsmm_class.tmat[i, j + 1] = 1
    #
    # hsmm_class.tmat[-1, -2] = 1

    num_of_states = 6
    hsmm_class.tmat = np.zeros((num_of_states, num_of_states))

    for i in range(len(hsmm_class.tmat)):
        for j in range(len(hsmm_class.tmat[i]) - 1):
            if i == j and j < len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 1

            elif i == j and j == len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 1

    hsmm_class.tmat[-1, -2] = 1
    hsmm_class.mean = np.array([5, 15, 25, 35, 45, 60])  # shape should be (n_states, n_dim)
    hsmm_class.mean = np.reshape(hsmm_class.mean, (-1, 1))
    hsmm_class.covmat = np.array([  # shape should be (n_states, n_dim, n_dim) -> array of square matrices
        [[6.]],
        [[6.]],
        [[6.]],
        [[6.]],
        [[6.]],
        [[1]],

    ])


def init4durval_observ_state_up(hsmm_class):
    # initial probability
    # initial probability
    hsmm_class.pi = np.zeros(6)
    hsmm_class.pi[0] = 1

    # durations
    hsmm_class.dur = np.zeros((6, 260))
    mean = 130
    std = 21

    for i in range(len(hsmm_class.dur) - 1):
        x = np.linspace(0, mean * 2, mean * 2)
        for k in range(len(x)):
            hsmm_class.dur[i, k] = scipy.stats.norm(mean, std).pdf(x[k,])
            hsmm_class.dur[i, ((x.shape[0] // 2) - 1)] += 1 - hsmm_class.dur[i].sum()
        mean -= 30
        std -= 5

    hsmm_class.dur[-1, 1] = 1

    for i in range(len(hsmm_class.dur)):
        hsmm_class.dur[i, ((hsmm_class.dur.shape[1] // 2) - 1)] += 1 - hsmm_class.dur[i].sum()

    # transition matrix
    # num_of_states = 5
    # hsmm_class.tmat = np.zeros((num_of_states, num_of_states))
    #
    # for i in range(len(hsmm_class.tmat)):
    #     for j in range(len(hsmm_class.tmat[i]) - 1):
    #         if i == j and j < len(hsmm_class.tmat[i]) - 2:
    #             hsmm_class.tmat[i, j + 1] = 0.6
    #             hsmm_class.tmat[i, j + 2] = 0.4
    #         elif i == j and j == len(hsmm_class.tmat[i]) - 2:
    #             hsmm_class.tmat[i, j + 1] = 1
    #
    # hsmm_class.tmat[-1, -2] = 1

    num_of_states = 6
    hsmm_class.tmat = np.zeros((num_of_states, num_of_states))

    for i in range(len(hsmm_class.tmat)):
        for j in range(len(hsmm_class.tmat[i]) - 1):
            if i == j and j < len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 1

            elif i == j and j == len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 1

    hsmm_class.tmat[-1, -2] = 1
    hsmm_class.mean = np.array([5, 15, 25, 35, 45, 60])  # shape should be (n_states, n_dim)
    hsmm_class.mean = np.reshape(hsmm_class.mean, (-1, 1))
    hsmm_class.covmat = np.array([  # shape should be (n_states, n_dim, n_dim) -> array of square matrices
        [[6.]],
        [[6.]],
        [[6.]],
        [[6.]],
        [[6.]],
        [[1]],

    ])


def init4durval_observ_state_up_down(hsmm_class):
    # initial probability
    # initial probability
    hsmm_class.pi = np.zeros(7)
    hsmm_class.pi[0] = 1

    # durations
    hsmm_class.dur = np.zeros((7, 260))
    mean = 130
    std = 21

    for i in range(len(hsmm_class.dur) - 2):
        x = np.linspace(0, mean * 2, mean * 2)
        for k in range(len(x)):
            hsmm_class.dur[i, k] = scipy.stats.norm(mean, std).pdf(x[k,])
            hsmm_class.dur[i, ((x.shape[0] // 2) - 1)] += 1 - hsmm_class.dur[i].sum()
        mean -= 30
        std -= 5

    hsmm_class.dur[-1, 5] = 1
    hsmm_class.dur[-2, 5] = 1

    for i in range(len(hsmm_class.dur)):
        hsmm_class.dur[i, ((hsmm_class.dur.shape[1] // 2) - 1)] += 1 - hsmm_class.dur[i].sum()

    # transition matrix
    # num_of_states = 5
    # hsmm_class.tmat = np.zeros((num_of_states, num_of_states))
    #
    # for i in range(len(hsmm_class.tmat)):
    #     for j in range(len(hsmm_class.tmat[i]) - 1):
    #         if i == j and j < len(hsmm_class.tmat[i]) - 2:
    #             hsmm_class.tmat[i, j + 1] = 0.6
    #             hsmm_class.tmat[i, j + 2] = 0.4
    #         elif i == j and j == len(hsmm_class.tmat[i]) - 2:
    #             hsmm_class.tmat[i, j + 1] = 1
    #
    # hsmm_class.tmat[-1, -2] = 1

    num_of_states = 7
    hsmm_class.tmat = np.zeros((num_of_states, num_of_states))

    for i in range(len(hsmm_class.tmat)):
        for j in range(len(hsmm_class.tmat[i]) - 1):
            if i == j and j < len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 1

            elif i == j and j == len(hsmm_class.tmat[i]) - 2:
                hsmm_class.tmat[i, j + 1] = 1

    hsmm_class.tmat[-1, -2] = 1

    hsmm_class.mean = np.array([5, 15, 25, 35, 45, 60, 65])  # shape should be (n_states, n_dim)
    hsmm_class.mean = np.reshape(hsmm_class.mean, (-1, 1))
    hsmm_class.covmat = np.array([  # shape should be (n_states, n_dim, n_dim) -> array of square matrices
        [[6.]],
        [[6.]],
        [[6.]],
        [[6.]],
        [[6.]],
        [[1]],
        [[1]],

    ])
