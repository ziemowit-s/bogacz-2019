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

    print("Control")
    actions = model.act(state=0, kn=1, da=0.5, raw=True)
    print("Computed:", "T[pellet]=%s" % round(actions[0], 4), "T[chow]=%s" % round(actions[1], 4))
    print("Fig 6: T[pellet]=1.0 T[chow]=0.4")

    print()
    # κN = 0.7507, corresponding to blocking ofD2 receptors with an efficiency ofroughly 25%
    actions = model.act(state=0, kn=0.7507, da=0.5, raw=True)
    print("DA Depleted")
    print("Computed:", "T[pellet]=%s" % round(actions[0], 4), "T[chow]=%s" % round(actions[1], 4))
    print("Fig 6: T[pellet]=-0.6 T[chow]=0.4")
