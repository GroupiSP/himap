import warnings

warnings.filterwarnings(action="ignore")
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.simplefilter(action="ignore", category=UserWarning)

from utils import *
from base import GaussianHSMM, HMM
import argparse


def run_process(args):
    """
    Run the process for the selected model
    :param args:
    :return:
    """
    hsmm = args.hsmm
    mc_sampling = args.mc_sampling
    bic_fit = args.bic_ft
    save = args.save
    metrics = args.metrics
    enable_visuals = args.enable_visuals
    num_histories = args.num_histories
    n_states = args.n_states

    if mc_sampling:
        if not hsmm:
            hmm_init = HMM(n_states=n_states, n_obs_symbols=30)
            obs, states = hmm_init.mc_dataset(num_histories)
            hmm_estim = HMM(n_states=n_states,
                            n_obs_symbols=hmm_init.n_obs_symbols,
                            )
            if bic_fit:
                hmm_estim, bic = hmm_estim.fit_bic(obs, states=list(np.arange(2, n_states + 4)))
            else:
                hmm_estim.fit(obs)
            if save:
                hmm_estim.save_model()
            hmm_estim.prognostics(obs, plot_rul=enable_visuals, get_metrics=metrics)
        else:
            hsmm_init = GaussianHSMM(n_states=n_states, n_durations=260, f_value=60, obs_state_len=10)
            obs, states = hsmm_init.mc_dataset(num_histories, timesteps=1000)
            hsmm_estim = GaussianHSMM(n_states=n_states,
                                      n_durations=hsmm_init.n_durations,
                                      f_value=hsmm_init.f_value,
                                      obs_state_len=hsmm_init.obs_state_len,
                                      )
            if bic_fit:
                hsmm_estim, bic = hsmm_estim.fit_bic(obs, states=list(np.arange(2, n_states + 4)))
            else:
                hsmm_estim.fit(obs)
            if save:
                hsmm_estim.save_model()
            hsmm_estim.prognostics(obs, plot_rul=enable_visuals, get_metrics=metrics)


    else:
        f_value = 21
        obs_state_len = 5
        seqs_train, seqs_test = load_data_cmapss(f_value=f_value, obs_state_len=obs_state_len)
        if not hsmm:
            hmm_c = HMM(n_states=n_states, n_obs_symbols=f_value)
            if bic_fit:
                hmm_c, bic = hmm_c.fit_bic(seqs_train, states=list(np.arange(2, n_states + 2)))
            else:
                hmm_c.fit(seqs_train, save_iters=False)
            if save:
                hmm_c.save_model()
            hmm_c.prognostics(seqs_test, plot_rul=enable_visuals, get_metrics=metrics)
        else:
            hsmm_c = GaussianHSMM(n_states=n_states, n_durations=200, f_value=f_value, obs_state_len=obs_state_len)
            if bic_fit:
                hsmm_c, bic = hsmm_c.fit_bic(seqs_train, states=list(np.arange(2, n_states + 2)))
            else:
                hsmm_c.fit(seqs_train)
            if save:
                hsmm_c.save_model()
            hsmm_c.prognostics(seqs_test, plot_rul=enable_visuals, get_metrics=metrics)



def himap_main(hsmm, mc_sampling, bic_fit, save, metrics, enable_visuals, num_histories, n_states):
    """
    Main function for running the HMM models
    :param hsmm: True for Hidden Semi-Markov Model
    :param mc_sampling: True for Monte-Carlo Sampling
    :param bic_fit: True for Bayesian Information Criterion fitting
    :param save: True for saving the fitted models
    :param metrics: True for calculating performance metrics
    :param enable_visuals: True for generating and saving figures
    :param num_histories: The number of generated histories via Monte Carlo Sampling
    :param n_states: The number of hidden states for Markov Model
    :return:
    """
    print(
        "This is the code for applying the hmm models to CMAPSS data or to Monte-Carlo Simulated data \n"
    )
    print("Code running...\n")

    parser = argparse.ArgumentParser()

    
    parser.add_argument(
        "--hsmm",
        default=hsmm,
        type=str,
        help="Use Hidden Semi-Markov Model (hsmm), default=True",
    )
    parser.add_argument(
        "--mc_sampling",
        default=mc_sampling,
        type=str2bool,
        help="Use Monte-Carlo generated data for the example, default=False",
    )

    parser.add_argument(
        "--bic_ft",
        default=bic_fit,
        type=str2bool,
        help="Enable Bayesian Information Criterion fitting for Markov Models, default=False",
    )
    parser.add_argument(
        "--save",
        default=save,
        type=str2bool,
        help="Enable saving of the fitted models, default=True",
    )

    parser.add_argument(
        "--metrics",
        default=metrics,
        type=str2bool,
        help="Enable calculation of performance metrics for RUL prediction, default=True",
    )

    parser.add_argument(
        "--enable_visuals",
        default=enable_visuals,
        type=str2bool,
        help="Enable generating and saving figures, default=True",
    )

    parser.add_argument(
        "--num_histories",
        default=num_histories,
        type=int,
        help="Select the number of generated histories via Monte Carlo Sampling, default=50",
    )

    parser.add_argument(
        "--n_states",
        default=n_states,
        type=int,
        help="Select the number of hidden states for Markov Model, default=6",
    )

    args = parser.parse_args()

    run_process(args)


if __name__ == "__main__":
    hsmm, mc_sampling, bic_fit, save, metrics, enable_visuals, num_histories, n_states = (
        False, False, True, True, True, True, 20, 6)
    himap_main(hsmm, mc_sampling, bic_fit, save, metrics, enable_visuals, num_histories, n_states)
