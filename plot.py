import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from utils import get_single_history


def plot_observ_states(obs, states, means, title):
    fig, ax1 = plt.subplots(figsize=(15, 3))

    color = 'tab:red'
    ax1.set_xlabel('time')
    ax1.set_ylabel('observations', color=color)
    ax1.plot(obs, color=color, label="observations")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('states', color=color)  # we already handled the x-label with ax1
    ax2.plot(np.asarray(states), color=color, linewidth=2, alpha=.8, label='HI')
    ax2.tick_params(axis='y', labelcolor=color, )
    ax2.set_yticks([tick for tick in ax2.get_yticks() if tick % 1 == 0])

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.suptitle(title)
    # fig.legend()

    plt.show()
    plt.pause(1)


def plot_multiple_observ(obs, states, means, num2plot):
    '''
    plot multiple degradation histories from MC sampling
    :param obs: all histories
    :param states: params from MC sampling
    :param means: params from MC sampling
    :param num2plot: number of histories to plot
    :return:
    '''
    title = f'Degradation histories with MC sampling\n {num2plot} random samples out of {obs.shape[0]} total samples'
    fig, axs = plt.subplots(num2plot)
    for i in range(num2plot):
        k = np.random.randint(obs.shape[0])
        states4plot = states[k, :]
        obs4plot = obs[k, :]

        for j in range(len(states4plot)):
            if states4plot[j] > states4plot[j + 1]:
                states4plot = np.delete(states4plot, slice(j + 1, -1), axis=0)
                states4plot = np.delete(states4plot, j + 1, axis=0)
                idx = j
                break
        obs4plot = np.delete(obs4plot, slice(idx + 1, -1), axis=0)
        obs4plot = np.delete(obs4plot, idx + 1, axis=0)

        axs[i].plot(means[states4plot], 'r', linewidth=2, alpha=.8)
        axs[i].plot(obs4plot)
    fig.suptitle(title)
    plt.show()
    plt.pause(1)


def plot_multiple_observ_single_graph(obs, states, means, num2plot):
    title = f'Degradation histories with MC sampling\n {num2plot} random samples out of {obs.shape[0]} total samples'
    plt.figure()

    for i in range(num2plot):
        k = np.random.randint(obs.shape[0])
        states4plot = states[k, :]
        obs4plot = obs[k, :]

        for j in range(len(states4plot)):
            if states4plot[j] > states4plot[j + 1]:
                states4plot = np.delete(states4plot, slice(j + 1, -1), axis=0)
                states4plot = np.delete(states4plot, j + 1, axis=0)
                idx = j
                break
        obs4plot = np.delete(obs4plot, slice(idx + 1, -1), axis=0)
        obs4plot = np.delete(obs4plot, idx + 1, axis=0)

        plt.plot(means[states4plot], linewidth=1, alpha=1)
        # plt.plot(obs4plot)
    plt.title(title)
    plt.show()
    plt.pause(1)


def plot_viterbi(obs, index, HSMM, train=True):
    history = get_single_history(obs, index)

    if train:
        text = 'train'
    else:
        text = 'test'

    newstate = HSMM.predict(history)
    newstate, _ = newstate
    means = HSMM.mean
    plot_observ_states(history, newstate, means, title=f'Viterbi on {text} data, {index} sample')
    return newstate


def plot_multiple_observ_new(obs, states, means, num2plot):
    title = f'Degradation histories with MC sampling\n {num2plot} random samples out of {obs.shape[0]} total samples'
    fig, axs = plt.subplots(num2plot)
    for i in range(num2plot):
        k = np.random.randint(obs.shape[0])
        states4plot = states[k, :]
        obs4plot = obs[k, :]

        axs[i].plot(means[states4plot], 'r', linewidth=2, alpha=.8)
        axs[i].plot(obs4plot)
    fig.suptitle(title)
    plt.show()
    plt.pause(1)


def plot_rul(mean_RUL, UB_RUL=None, LB_RUL=None):
    true_rul = np.arange(len(mean_RUL) - 1, -1, -1)
    fig, ax = plt.subplots()
    ax.plot(true_rul, label='True RUL', color='black', linewidth=2)
    ax.plot(mean_RUL, '--', label='Mean Predicted RUL', color='tab:red', linewidth=2)
    if (UB_RUL, LB_RUL) is not None:
        ax.plot(UB_RUL, '-.', label='Lower Bound (90% CI)', color='tab:blue', linewidth=1)
        ax.plot(LB_RUL, '-.', label='Upper Bound (90% CI)', color='tab:blue', linewidth=1)
        ax.fill_between(np.arange(0, len(UB_RUL)), UB_RUL, LB_RUL, alpha=0.1, color='tab:blue')

    fig.suptitle('RUL')
    ax.legend(loc='best')

##TODO: change the rest of the plot functions to look like the plot_observ_states
