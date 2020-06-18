from model.state import State


class Model:
    def __init__(self, num_of_states: int = 1, num_of_actions: int = 1, alfa: float = 0.1,
                 lambd: float = 0.1, epsilon=0.1, g_init: float = 0, n_init: float = 0):
        """
        Actor only approch model
        """
        self.states = []
        for _ in range(num_of_states):
            state = State(num_of_actions=num_of_actions, alfa=alfa, lambd=lambd, epsilon=epsilon,
                          g_init=g_init, n_init=n_init)
            self.states.append(state)

    def reward(self, reward, state: int = 0, action: int = 0):
        self.states[state].reward(reward=reward, action=action)

    def act(self, state: int = 0, kn=1.0, da=0.5, raw=False, sigma=None):
        return self.states[state].act(kn=kn, da=da, raw=raw, sigma=sigma)
