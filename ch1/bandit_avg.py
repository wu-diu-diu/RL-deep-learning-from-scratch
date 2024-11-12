from bandit import Bandit, Agent
import numpy as np
import matplotlib.pyplot as plt


def main():
    runs = 300
    steps = 1000
    eps = 0.1
    all_rates = np.zeros((runs, steps))
    np.random.seed(10)
    mean_rates = []
    for _ in range(3):
        for r in range(runs):  ## run 300 times
            bandit = Bandit()
            agent = Agent(eps)
            rates = []
            total_reward = 0
            for s in range(steps):  ## play 1000 times in each run
                action = agent.get_action()
                reward = bandit.play(action)
                agent.updata(action, reward)

                total_reward += reward
                rates.append(total_reward / (s + 1))

            all_rates[r] = rates
        eps += 0.1  ## the bigger eps means more random action

        mean_rates.append(all_rates.mean(axis=0))

    for i, rates in enumerate(mean_rates):
        plt.plot(rates, label=f'eps={0.1 + i * 0.1:.1f}')
    plt.ylabel('Rate')
    plt.xlabel('Steps')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
