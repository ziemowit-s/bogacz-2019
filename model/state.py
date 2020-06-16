import numpy as np

from model.action import Action


class State:
    def __init__(self, num_of_actions: int = 1, alfa: float = 0.1,
                 lambd: float = None, actor_only: bool = False, epsilon=0.1, g_init: float = 0, n_init: float = 0):

        self.actions = []
        for _ in range(num_of_actions):
            a = Action(alfa=alfa, lambd=lambd, actor_only=actor_only, epsilon=epsilon, g_init=g_init, n_init=n_init)
            self.actions.append(a)

    def reward(self, reward, action: int = 0):
        self.actions[action].reward(r=reward)

    def act(self, kn=1.0, da=0.5, raw=False, sigma=None):
        ts = []
        for act in self.actions:
            t = act.act(kn=kn, da=da)
            if sigma:
                t = t + np.random.normal(loc=0, scale=sigma, size=1)[0]
            ts.append(t)

        if raw:
            return ts
        elif np.max(ts) > 0:
            return np.argmax(ts)
        else:
            return None

    def __repr__(self):
        return str(self.act())
