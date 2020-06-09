import numpy as np

from model.model import AUModel

"""
Train Actor Uncertainty Model with detrministic reward.
"""
if __name__ == '__main__':
    EPOCH = 1
    REWARD = 1
    BATCH_NUM = 10000
    ACTOR_ONLY = True
    STATE_AND_ACTION_NUM = 2

    accuracy = []
    for epoch in range(EPOCH):
        model = AUModel(num_of_states=STATE_AND_ACTION_NUM, num_of_actions=STATE_AND_ACTION_NUM, actor_only=ACTOR_ONLY,
                        alfa=0.4, lambd=0.1013, epsilon=0.519)

        N = [i for i in range(BATCH_NUM)]
        r = 0
        action = 0
        for i in N:
            state = np.random.randint(low=0, high=STATE_AND_ACTION_NUM, size=1)[0]
            actions = model.act(state=state, kn=1, da=0.5)
            action = int(np.argmax(actions))

            if state == action:
                r = REWARD
                accuracy.append(1)
            else:
                r = -REWARD
                accuracy.append(0)
            model.reward(reward=r, state=state, action=action)

            if len(accuracy) == round(len(N)/10):
                accuracy.pop(0)

            avg = np.average(accuracy)
            print('epoch:', epoch, 'i:', i, 'acc:', avg)
