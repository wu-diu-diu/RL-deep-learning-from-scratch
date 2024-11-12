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
    eps_list = [0.1, 0.3, 0.01]
    results = {}
    for eps in eps_list:
        all_rates = np.zeros((runs, steps))
        for r in range(runs):  ## run 300 times
            bandit = Bandit()  ## 300次试验，每次都初始化十台老虎机
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
        avg_rates = all_rates.mean(axis=0)
        results[str(eps)] = avg_rates

    for key, values in results.items():
        plt.plot(values, label=key)
    plt.ylabel('Rate')
    plt.xlabel('Steps')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main()
