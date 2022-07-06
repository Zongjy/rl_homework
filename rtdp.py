import numpy as np


class RTDP:
    def __init__(self, env):
        self.env = env
        self.iterations = 2077
        self.gamma = 1.
        self.actions = [a for a in range(21)]
        self.calc_policy()

    def calc_policy(self):
        # RTDP
        V = np.zeros(50 * 21)

        for i in range(self.iterations):
            obs = self.env.reset()
            while True:
                action = np.argmin(
                    [
                        self.env.prob[obs][a][0][3] + self.gamma * V[self.env.prob[obs][a][0][1]]
                        for a in self.actions
                    ]
                )
                V[obs] = self.env.prob[obs][action][0][3] + self.gamma * V[self.env.prob[obs][action][0][1]]
                obs, reward, cost, done, _ = self.env.step(action)
                if done:
                    break

        # plt.matshow(V.reshape(50, 21), cmap=plt.cm.Blues)
        # plt.show()
        self.V = V

    def policy(self, state):
        return np.argmin(
            [
                self.env.prob[state][a][0][3] + self.gamma * self.V[self.env.prob[state][a][0][1]]
                for a in self.actions
            ]
        )


def to_coordinate(state):
    x = int(state / 21)
    y = state - x * 21
    return x, y


def evaluate(rtdp, env):
    rewards = 0.
    state = env.reset()
    done = False
    steps = 0
    x = []
    y = []
    x.append(0)
    y.append(10)
    while not done:
        state, reward, cost, done, _ = env.step(rtdp.policy(state))
        tmpx, tmpy = to_coordinate(state)
        x.append(tmpx)
        y.append(tmpy)
        rewards += reward
        steps += 1
        if steps > 1e4:
            break
    x.append(50)
    y.append(10)
    xarray = np.asarray(x)
    yarray = np.asarray(y)
    return xarray, yarray
