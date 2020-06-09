import numpy as np

from model.model import Model


def train(model, n):
    for _ in range(n):
        # Control
        actions = model.act(state=0, kn=1, da=0.5, raw=True)
        act = np.argmax(actions)

        # Depleted
        actions = model.act(state=0, kn=0.7507, da=0.5, raw=True)
        act = np.argmax(actions)


def test(model, n):
    control = 0
    depleted = 0
    for _ in range(n):

        # Control
        actions = model.act(state=0, kn=1, da=0.5, raw=True)
        act = np.argmax(actions)
        if act == 0:
            control += 1

        # Depleted
        actions = model.act(state=0, kn=0.7507, da=0.5, raw=True)
        act = np.argmax(actions)
        if act == 1:
            depleted += 1
    return control, depleted


"""
Train Actor Uncertainty Model with detrministic reward.
"""
if __name__ == '__main__':
    ACTOR_ONLY = True

    model = Model(num_of_states=1, num_of_actions=2, actor_only=ACTOR_ONLY,
                  alfa=0.1, epsilon=0.6327, lambd=0.0204)

    pellet_action = model.states[0].actions[0]
    chow_action = model.states[0].actions[1]

    pellet_action._g = 0.1
    pellet_action._n = 0.1

    chow_action._g = 0.1
    chow_action._n = 0.1

    for rat_num in range(6):
        train(model, n=180)
        control, depleted = test(model, n=180)
        print(control, depleted)

