class Action:
    def __init__(self, alfa: float = 0.1, lambd: float = 0.1, epsilon: float = 0.1,
                 g_init: float = 0, n_init: float = 0):
        """
        Actor Uncertainty action
        :param alfa:
        :param lambd:
        """
        self._alfa = alfa
        self._lambd = lambd
        self._epsilon = epsilon

        self._g = g_init
        self._n = n_init

    def g(self, delta):
        delta_g = self._alfa * self.f(delta) - self._lambd * self._g
        self._g += delta_g
        if self._g < 0:
            self._g = 0

    def n(self, delta):
        delta_n = self._alfa * self.f(-delta) - self._lambd * self._n
        self._n += delta_n
        if self._n < 0:
            self._n = 0

    def delta(self, r):
        return r - 0.5 * (self._g - self._n)

    def f(self, delta):
        if delta > 0:
            return delta
        else:
            return delta * self._epsilon

    def reward(self, r):
        delta = self.delta(r)
        self.g(delta)
        self.n(delta)

    def act(self, kn: float = 1.0, da: float = 0.5):
        """
        Thalamic output
        :param kn:
            between 0 and 1
        :param da:
            dopamine. 0.5 is a baseline
        :return:
        """
        return da * self._g - (1 - kn * da) * self._n

    def get_values(self):
        return self._g, self._n
