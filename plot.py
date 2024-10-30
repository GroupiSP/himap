import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from utils import get_single_history


def plot_observ_states(obs, states,title):

    fig, ax1 = plt.subplots(figsize=(15, 3))

    color = 'tab:red'
    ax1.set_xlabel('time')
    ax1.set_ylabel('observations', color=color)
    ax1.plot(obs, color=color,label="observations")
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:blue'
    ax2.set_ylabel('states', color=color)  # we already handled the x-label with ax1
    ax2.plot(np.asarray(states), color=color, linewidth=2, alpha=.8, label='HI')
    ax2.tick_params(axis='y', labelcolor=color,)
    ax2.set_yticks([tick for tick in ax2.get_yticks() if tick % 1 == 0])

    # fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.suptitle(title)
    # fig.legend()

    plt.show()
    plt.pause(1)



def plot_multiple_observ(obs, states, num2plot):
    '''
    plot multiple degradation histories from MC sampling
    :param obs: all histories
    :param states: params from MC sampling
    :param means: params from MC sampling
    :param num2plot: number of histories to plot
    :return:
    '''
    title = f'Degradation histories with MC sampling\n {num2plot} random samples out of {len(obs)} total samples'
    fig, axs = plt.subplots(num2plot)
    keys = list(obs.keys())
    for i in range(num2plot):
        k = np.random.randint(len(obs))
        states4plot = states[keys[k]]
        obs4plot = obs[keys[k]]

        ax1=axs[i]

        color = 'tab:red'
        ax1.set_xlabel('time')
        ax1.set_ylabel('observations', color=color)
        ax1.plot(obs4plot, color=color, label="observations")
        ax1.tick_params(axis='y', labelcolor=color)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        ax2.set_ylabel('states', color=color)  # we already handled the x-label with ax1
        ax2.plot(states4plot, color=color, linewidth=2, alpha=.8, label='HI')
        ax2.tick_params(axis='y', labelcolor=color, )
        ax2.set_yticks([tick for tick in ax2.get_yticks() if tick % 1 == 0])
        ax2.set_ylim(top=max(states4plot)+1)
        plt.xlim(left=0)
        plt.ylim(bottom=0)

    fig.suptitle(title)
    plt.show()