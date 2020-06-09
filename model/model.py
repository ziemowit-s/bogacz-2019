from model.state import AUState


class AUModel:
    def __init__(self, num_of_states: int = 1, num_of_actions: int = 1, alfa: float = 0.1,
                 lambd: float = 0.1, actor_only: bool = False, epsilon=0.1):
        """
        Actor Uncertainty model
        """
        self.states = []
        for _ in range(num_of_states):
            state = AUState(num_of_actions=num_of_actions, alfa=alfa, lambd=lambd,
                            actor_only=actor_only, epsilon=epsilon)
            self.states.append(state)

    def reward(self, reward, state: int = 0, action: int = 0):
        self.states[state].reward(reward=reward, action=action)

    def act(self, state: int = 0, kn=1.0, da=0.5):
        probas = self.states[state].act(kn=kn, da=da)
        return probas