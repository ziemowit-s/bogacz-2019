import numpy as np

from model.action import AUAction


class AUState:
    def __init__(self, num_of_actions: int = 1, alfa: float = 0.1,
                 lambd: float = None, actor_only: bool = False, epsilon=0.1):

        self.actions = []
        for _ in range(num_of_actions):
            a = AUAction(alfa=alfa, lambd=lambd, actor_only=actor_only, epsilon=epsilon)
            self.actions.append(a)

    def reward(self, reward, action: int = 0):
        self.actions[action].reward(r=reward)

    def act(self, kn=1.0, da=0.5):
        probas = []
        for act in self.actions:
            pi = act.act(kn=kn, da=da)
            if pi > 0:
                probas.append(pi)

        if len(probas) > 0:
            return np.max(probas)
        else:
            return None

    def __repr__(self):
        return str(self.act())
