import numpy as np

from model.model import Model


def make_model():
    # At the start ofeach simulation, the weights were initialized to:
    # Gpellet = Npellet = Gchow = Nchow = 0.1
    model = Model(num_of_states=1, num_of_actions=2, actor_only=ACTOR_ONLY,
                  alfa=0.1, epsilon=0.6327, lambd=0.0204,
                  g_init=0.1, n_init=0.1)

    return model


def train(model, n, p_pelet, n_pelet, p_chow=1, n_chow=0):
    for _ in range(n):
        # lewer push - cost
        model.reward(reward=n_pelet, state=0, action=0)
        # pellet obtained - payoff
        model.reward(reward=p_pelet, state=0, action=0)
        # chow cost (default 0)
        model.reward(reward=n_chow, state=0, action=1)
        # chow payoff (default 1)
        model.reward(reward=p_chow, state=0, action=1)


def test(model, n, p_pelet, n_pelet, p_chow=1, n_chow=0, kn: float = 1, da=0.5, sigma=None):
    chow_choose = 0
    pellet_choose = 0

    for _ in range(n):
        t = model.act(state=0, kn=kn, da=da, sigma=sigma)

        # if no move choose
        if t is None:
            continue

        # pellet
        if t == 0:
            pellet_choose += 1
            if train:
                # lewer push - cost
                model.reward(reward=n_pelet, state=0, action=0)
                # pellet obtained - payoff
                model.reward(reward=p_pelet, state=0, action=0)

        # chow
        else:
            chow_choose += 1
            if train:
                # chow cost (default 0)
                model.reward(reward=n_chow, state=0, action=1)
                # chow payoff (default 1)
                model.reward(reward=p_chow, state=0, action=1)

    return pellet_choose, chow_choose


def compute():
    control_model = make_model()
    depleted_model = make_model()

    ct_pellets = []
    ct_chows = []

    dp_pellets = []
    dp_chows = []

    # 6 rats was simulated.
    for rat_num in range(6):

        # Each simulation consisted of 180 training
        train(control_model,  p_pelet=P_PELET, n_pelet=N_PELET, n=180)
        train(depleted_model, p_pelet=P_PELET, n_pelet=N_PELET, n=180)

        # 180 testing trials
        ct_pellet, ct_chow = test(control_model,  p_pelet=P_PELET, sigma=SIGMA, n_pelet=N_PELET, n=180, kn=1, da=0.5)
        dp_pellet, dp_chow = test(depleted_model, p_pelet=P_PELET, sigma=SIGMA, n_pelet=N_PELET, n=180, kn=KN_BLOCK, da=0.5)

        ct_pellets.append(ct_pellet)
        ct_chows.append(ct_chow)
        dp_pellets.append(dp_pellet)
        dp_chows.append(dp_chow)

    print("CONTROL:")
    print("pellet:", round(np.average(ct_pellets), 4), "chow:", round(np.average(ct_chows), 4))
    print("DEPLETED D2:")
    print("pellet:", round(np.average(dp_pellets), 4), "chow:", round(np.average(dp_chows), 4))


"""
Train Actor Uncertainty Model with detrministic reward.
"""
if __name__ == '__main__':
    ACTOR_ONLY = True
    P_PELET = 15.511751
    SIGMA = 1.066246
    KN_BLOCK = 0.7507  # corresponding to blocking ofD2 receptors with an efficiency ofroughly 25%

    print("No pellet cost")
    N_PELET = 0
    compute()

    print()
    print("Pellet cost")
    N_PELET = -14.510517
    compute()
