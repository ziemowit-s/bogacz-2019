

class AUAction:
    def __init__(self, alfa: float = 0.1, lambd: float = 0.1, actor_only: bool = True,
                 epsilon: float = 0.1):
        """
        Actor Uncertainty action
        :param alfa:
        :param lambd:
        :param actor_only:
        """
        self._alfa = alfa
        self._lambd = lambd
        self._epsilon = epsilon
        self.actor_only = actor_only

        self._g = 0
        self._n = 0
        self._delta = 0

    def g(self, r):
        self._g = self._alfa * self.f() - self._lambd * self._g

    def n(self, r):
        self._n = self._alfa * self.f() - self._lambd * self._n

    def delta(self, r):
        self._delta = r - 0.5 * (self._g - self._n)

    def f(self):
        if self._delta > 0:
            return self._delta
        else:
            return self._delta * self._epsilon

    def reward(self, r):
        self.g(r)
        self.n(r)
        self.delta(r)

    def act(self, kn: float = 1.0, da: float = 0.5):
        """
        Thalamic output
        :param kn:
            between 0 and 1
        :return:
        """
        return da * self._g - (1 - kn * da) * self._n
