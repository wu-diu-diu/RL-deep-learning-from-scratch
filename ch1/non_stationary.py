import numpy as np
from bandit import Agent
import matplotlib.pyplot as plt


class NonStatBandit:
    def __init__(self, arms=10):
        self.arms = arms
        self.rate = np.random.rand(arms)  ## rate of win for each machine

    def play(self, arm):
        rate = self.rate[arm]
        self.rate += 0.1 * np.random.randn(self.arms)
        ## add random noise to the rate and noise follows normal distribution mean=0, std=0.1
        if np.random.rand() < rate:
            return 1
        else:
            return 0


class AlphaAgent:
    def __init__(self, eps, alpha, arms=10):
        self.epsilon = eps
        self.alpha = alpha
        self.Qs = np.zeros(arms)

    def updata(self, action, reward):
        self.Qs[action] += self.alpha * (reward - self.Qs[action])

    def get_action(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(len(self.Qs))
        else:
            return np.argmax(self.Qs)


def main():
    runs = 300
    steps = 1000
    eps = 0.1
    alpha = 0.8
    agent_type = ['sample avg', 'alpha const update']
    np.random.seed(10)
    results = {}
    for agent_type in agent_type:
        all_rates = np.zeros((runs, steps))  ## (300, 1000)
        for r in range(runs):
            if agent_type == 'sample avg':
                agent = Agent(eps)
                bandit = NonStatBandit()
            else:
                agent = AlphaAgent(eps, alpha)
                bandit = NonStatBandit()
            total_rewards = 0
            rates = []

            for s in range(steps):  ## play 1000 times in each run
                action = agent.get_action()
                reward = bandit.play(action)
                agent.updata(action, reward)
                total_rewards += reward
                rates.append(total_rewards / (s + 1))

            all_rates[r] = rates
        avg_rates = all_rates.mean(axis=0)
        results[agent_type] = avg_rates

    plt.figure()
    plt.ylabel('Average Rates')
    plt.xlabel('Steps')
    for key, avg_rates in results.items():
        plt.plot(avg_rates, label=key)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
