import numpy as np
import matplotlib.pyplot as plt

from model.model import Model
from plot_utils import plot_ax


def make_model():
    """
    Creates model with actor-only approach
    """
    # At the start ofeach simulation, the weights were initialized to:
    # Gpellet = Npellet = Gchow = Nchow = 0.1
    model = Model(num_of_states=1, num_of_actions=2,
                  alfa=0.1, epsilon=0.6327, lambd=0.0204,
                  g_init=0.1, n_init=0.1)

    return model


def train(model: Model, n: int, p_pelet: float, n_pelet: float, p_chow: float = 1, n_chow: float = 0):
    """
    Compute exposure on all stimuli without action choosing (in accordance with the publication)
    :param model:
        concrete model
    :param n:
        number of trials
    :param p_pelet:
        reward for pelet, a better food
    :param n_pelet:
        cost of obtaining the pelet, pushing the lewer by mouse during the experiment
    :param p_chow:
        reward for chow, default low quality food. Default is 1
    :param n_chow:
        cost of the chow. Default is no cost, meaning - it is always available
    :return:
        list of tuple of ActionValues for each state and action.
        ActionValues(name, g, n)
    """
    action_values = []
    for _ in range(n):
        # lewer push - cost
        model.reward(reward=n_pelet, state=0, action=0)
        # pellet obtained - payoff
        model.reward(reward=p_pelet, state=0, action=0)

        # chow cost (default 0)
        model.reward(reward=n_chow, state=0, action=1)
        # chow payoff (default 1)
        model.reward(reward=p_chow, state=0, action=1)

        action_values.append(model.states[0].get_values())

    return action_values


def test(model, n, p_pelet, n_pelet, p_chow=1, n_chow=0, kn: float = 1, da=0.5, sigma=None):
    """
    Compute test with training on model choose match expected outcome (in accordance with the publication)

    :param model:
        concrete model
    :param n:
        number of trials
    :param p_pelet:
        reward for pelet, a better food
    :param n_pelet:
        cost of obtaining the pelet, pushing the lewer by mouse during the experiment
    :param p_chow:
        reward for chow, default low quality food. Default is 1
    :param n_chow:
        cost of the chow. Default is no cost, meaning - it is always available
    :param kn:
        D2 receptor reduction of activity. It simulates haloperidol antgonist effect on D2R
        kn=1 means 100% activity of D2 (no haloperidol injected)
        kn=0.75 means 75% activity of D2 (haloperidol reduced activity of 25% of D2R)
    :param da:
        Dopamine baseline activity. Default is 0.5 (based on th publication)
    :param sigma:
        Standard deviation of the normal distribution which is added to activity taken by the model.
    :return:
        tuple of (chow_choose:int, pellet_chose:int, action_values)

        action_values is a list of tuple of ActionValues for each state and action.
        ActionValues(name, g, n)
    """
    chow_choose = 0
    pellet_choose = 0

    action_values = []
    for _ in range(n):
        t = model.act(state=0, kn=kn, da=da, sigma=sigma)

        # if no move choose
        if t is None:
            continue

        # pellet
        if t == 0:
            pellet_choose += 1
            # lewer push - cost
            model.reward(reward=n_pelet, state=0, action=0)
            # pellet obtained - payoff
            model.reward(reward=p_pelet, state=0, action=0)

        # chow
        else:
            chow_choose += 1
            # chow cost (default 0)
            model.reward(reward=n_chow, state=0, action=1)
            # chow payoff (default 1)
            model.reward(reward=p_chow, state=0, action=1)

        action_values.append(model.states[0].get_values())

    return pellet_choose, chow_choose, action_values


def plot_rat_action_values(result_data, rat_num, title):
    train = result_data[1][rat_num][0]

    fig, axes = plt.subplots(2, 1)
    ax0, ax1 = axes.flatten()
    fig.suptitle(title, fontsize=16)

    ax0.set_title("Action: Get Pellet")
    ax0.plot([i for i in range(len(train))], [a[0].g for a in train], label="GO")
    ax0.plot([i for i in range(len(train))], [a[0].n for a in train], label="NO-GO")
    ax0.legend()

    ax1.set_title("Action: Get Chow")
    ax1.plot([i for i in range(len(train))], [a[1].g for a in train], label="GO")
    ax1.plot([i for i in range(len(train))], [a[1].n for a in train], label="NO-GO")
    ax1.legend()


def compute(mouse_num: int, trials: int, p_pelet, n_pelet, p_chow, n_chow, kn_control, kn_depleted, sigma,
            da_baseline=0.5):
    """
    Compute single model run with control and D2 depleted paradigm.
    :param mouse_num:
        number of mouses to be averaged
    :param trials:
        number of trials
    :param p_pelet:
        reward for pelet, a better food
    :param n_pelet:
        cost of obtaining the pelet, pushing the lewer by mouse during the experiment
    :param p_chow:
        reward for chow, default low quality food. Default is 1
    :param n_chow:
        cost of the chow. Default is no cost, meaning - it is always available
    :param kn_control:
        D2 receptor reduction of activity. It simulates haloperidol antgonist effect on D2R
        kn=1 means 100% activity of D2 (no haloperidol injected)
        kn=0.75 means 75% activity of D2 (haloperidol reduced activity of 25% of D2R)
    :param kn_depleted:
        D2 receptor reduction of activity. It simulates haloperidol antgonist effect on D2R
        kn=1 means 100% activity of D2 (no haloperidol injected)
        kn=0.75 means 75% activity of D2 (haloperidol reduced activity of 25% of D2R)
    :param sigma:
        Standard deviation of the normal distribution which is added to activity taken by the model.
    :param da_baseline:
        Dopamine baseline activity. Default is 0.5 (based on th publication)
    :return:
        numpy array of averaged by muse number consumption of:
        (control_pellets, control_chow, depleted_pellets, depleted_chow, action_values_train, action_values_test)
    """
    model = make_model()

    ct_pellets = []
    ct_chows = []

    dp_pellets = []
    dp_chows = []

    action_values = []

    # 6 rats was simulated.
    for rat_num in range(mouse_num):
        # train
        train_av = train(model, n=trials, p_pelet=p_pelet, n_pelet=n_pelet, p_chow=p_chow, n_chow=n_chow)

        # test
        ct_pellet, ct_chow, test_av_ct = test(model, n=trials, kn=kn_control, da=da_baseline, sigma=sigma,
                                              p_pelet=p_pelet, n_pelet=n_pelet, p_chow=p_chow, n_chow=n_chow)
        dp_pellet, dp_chow, test_av_dp = test(model, n=trials, kn=kn_depleted, da=da_baseline, sigma=SIGMA,
                                              p_pelet=p_pelet, n_pelet=n_pelet, p_chow=p_chow, n_chow=n_chow)

        ct_pellets.append(ct_pellet)
        ct_chows.append(ct_chow)
        dp_pellets.append(dp_pellet)
        dp_chows.append(dp_chow)

        action_values.append( (train_av, test_av_ct, test_av_dp) )
    return np.average([ct_pellets, ct_chows, dp_pellets, dp_chows], axis=1), action_values


def print_results(avgs):
    print("Control")
    print("    pellet:", round(avgs[0], 4), "chow:", round(avgs[1], 4))
    print("Depleted D2")
    print("    pellet:", round(avgs[2], 4), "chow:", round(avgs[3], 4))


"""
Train Actor Uncertainty Model with detrministic reward.
"""
if __name__ == '__main__':
    ACTOR_ONLY = True

    P_PELET = 15.511751
    N_PELET = -14.510517

    SIGMA = 1.066246
    KN_DEPLETED = 0.7507

    fig1, axes = plt.subplots(2, 1)
    ax0, ax1 = axes.flatten()

    # No cost for pellet
    r0 = compute(mouse_num=6, trials=180, p_pelet=P_PELET, n_pelet=0, p_chow=1, n_chow=0, kn_control=1,
                 kn_depleted=KN_DEPLETED, da_baseline=0.5, sigma=SIGMA)
    avg0 = r0[0]
    plot_rat_action_values(result_data=r0, rat_num=0, title="No cost for obtaining the pellet")

    # With cost for pellet
    r1 = compute(mouse_num=6, trials=180, p_pelet=P_PELET, n_pelet=N_PELET, p_chow=1, n_chow=0, kn_control=1,
                 kn_depleted=KN_DEPLETED, da_baseline=0.5, sigma=SIGMA)
    avg1 = r1[0]
    plot_rat_action_values(result_data=r1, rat_num=0, title="With cost for obtaining the pellet")

    print("NO COST FOR PELLET:")
    print_results(avg0)
    print()
    print("WITH COST FOR PELLET:")
    print_results(avg1)

    plot_ax(ax0, avg0.reshape([2, 2]).T, title="B) Theory")
    plot_ax(ax1, avg1.reshape([2, 2]).T, title="D) Theory")

    plt.show()
