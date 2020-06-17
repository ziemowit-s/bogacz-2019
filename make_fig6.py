import numpy as np

from model.model import Model

"""
Train Actor Uncertainty Model with detrministic reward.
"""
if __name__ == '__main__':
    ACTOR_ONLY = True

    model = Model(num_of_states=1, num_of_actions=2, actor_only=ACTOR_ONLY,
                  alfa=0.4, lambd=0.1013, epsilon=0.519)

    pellet_action = model.states[0].actions[0]
    chow_action = model.states[0].actions[1]

    pellet_action._g = 15.1
    pellet_action._n = 13.0

    chow_action._g = 0.9
    chow_action._n = 0.0

    control = model.act(state=0, kn=1, da=0.5, raw=True)
    # κN = 0.7507, corresponding to blocking of D2 receptors with an efficiency ofroughly 25%
    depleted = model.act(state=0, kn=0.7507, da=0.5, raw=True)

    txt = f"""
    |                       |  Pellet | Chow |
    |-----------------------|---------|------|
    | Control               |         |      |
    |   Replication         | {round(control[0], 4)}    | {round(control[1], 4)} |
    |   Fig 6               | 1.0     | 0.4  |
    | --------------------- | ------- | ---- |
    | D2R Depleted          |         |      |
    |   Replication         | {round(depleted[0], 4)} | {round(depleted[1], 4)} |
    |   Fig 6               | -0.6    | 0.4  |
    """
    print(txt)
