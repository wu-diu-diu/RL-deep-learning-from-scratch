import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, arms=10):
        self.rate = np.random.rand(arms)  ## rate of win for each machine

    def play(self, arm):
        rate = self.rate[arm]
        if np.random.rand() < rate:
            return 1
        else:
            return 0


class Agent:
    def __init__(self, eps, arms=10):
        self.epsilon = eps
        self.Qs = np.zeros(arms)  ## value of each machine
        self.Ns = np.zeros(arms)  ## number of times each machine has been played

    def updata(self, action, reward):
        self.Ns[action] += 1
        self.Qs[action] += (reward - self.Qs[action]) / self.Ns[action]

    def get_action(self):
        if np.random.rand() < self.epsilon:  ## random select a machine with a certain probability
            return np.random.randint(len(self.Qs))  ## return a random idx of  machine
        else:
            return np.argmax(self.Qs)  ## return the index of the machine with the highest value


def main():
    steps = 1000
    eps = 0.1
    np.random.seed(0)

    bandit = Bandit()
    agent = Agent(eps)
    total_rewards = []
    total_reward = 0
    rates = []

    for step in range(steps):
        action = agent.get_action()
        reward = bandit.play(action)
        agent.updata(action, reward)
        total_reward += reward
        total_rewards.append(total_reward)
        rates.append(total_reward / (step + 1))

    print(total_reward)

    ## plot
    plt.ylabel('Total reward')
    plt.xlabel('Steps')
    plt.plot(total_rewards)
    plt.show()

    plt.ylabel('Rate')
    plt.xlabel('Steps')
    plt.plot(rates)
    plt.show()


if __name__ == '__main__':
    main()
