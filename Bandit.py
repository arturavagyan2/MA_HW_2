from logs import *
from abc import ABC, abstractmethod

import csv
import numpy as np

logging.basicConfig
logger = logging.getLogger("MAB Application")


# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

ch.setFormatter(CustomFormatter())

logger.addHandler(ch)


class Bandit(ABC):
    @abstractmethod
    def __init__(self, p):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @abstractmethod
    def pull(self):
        pass

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def experiment(self):
        pass

    @abstractmethod
    def report(self):
        # store data in csv
        # log average reward (use f strings to make it informative)
        # log average regret (use f strings to make it informative)
        pass

#--------------------------------------#



class Visualization():

    def plot1(self):
        # Visualize the performance of each bandit: linear and log
        pass

    def plot2(self):
        # Compare E-greedy and thompson sampling cummulative rewards
        # Compare E-greedy and thompson sampling cummulative regrets
        pass


class EpsilonGreedy(Bandit):
    def __init__(self, probabilities, epsilon=0.1):
        self.probabilities = probabilities  # Success probabilities of bandits
        self.epsilon = epsilon
        self.counts = [0] * len(probabilities)  # Times each bandit was chosen
        self.values = [0.0] * len(probabilities)  # Estimated value of each bandit
        self.total_rewards = 0

    def __repr__(self):
        return f"EpsilonGreedy(epsilon={self.epsilon})"

    def pull(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(self.probabilities))  # Explore
        else:
            return np.argmax(self.values)  # Exploit

    def update(self, chosen_bandit, reward):
        self.counts[chosen_bandit] += 1
        # Update estimate of bandit's value
        n = self.counts[chosen_bandit]
        value = self.values[chosen_bandit]
        self.values[chosen_bandit] = ((n - 1) / float(n)) * value + (1 / float(n)) * reward
        self.total_rewards += reward

    def experiment(self, trials=1000):
        rewards = []
        for _ in range(trials):
            chosen_bandit = self.pull()
            reward = np.random.rand() < self.probabilities[chosen_bandit]
            self.update(chosen_bandit, reward)
            rewards.append(reward)
        return rewards
    
    def report(self, algorithm_name, trials=1000):
        average_reward = self.total_rewards / trials
        total_regret = trials - self.total_rewards  
        average_regret = total_regret / trials

        # Logging average reward and regret
        logging.info(f"{algorithm_name} - Average Reward: {average_reward}")
        logging.info(f"{algorithm_name} - Average Regret: {average_regret}")

        # Storing data in CSV file
        filename = f"{algorithm_name}_results.csv"
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Bandit", "Reward", "Algorithm"])

            for i, (value, count) in enumerate(zip(self.values, self.counts)):
                writer.writerow([i, value, algorithm_name])


class ThompsonSampling(Bandit):
    def __init__(self, probabilities):
        self.probabilities = probabilities
        self.successes = [0] * len(probabilities)
        self.failures = [0] * len(probabilities)
        self.total_rewards = 0

    def __repr__(self):
        return "ThompsonSampling()"

    def pull(self):
        samples = [np.random.beta(1 + s, 1 + f) for s, f in zip(self.successes, self.failures)]
        return np.argmax(samples)

    def update(self, chosen_bandit, reward):
        if reward:
            self.successes[chosen_bandit] += 1
        else:
            self.failures[chosen_bandit] += 1
        self.total_rewards += reward

    def experiment(self, trials=1000):
        rewards = []
        for _ in range(trials):
            chosen_bandit = self.pull()
            reward = np.random.rand() < self.probabilities[chosen_bandit]
            self.update(chosen_bandit, reward)
            rewards.append(reward)
        return rewards
    
    def report(self, algorithm_name, trials=1000):
        average_reward = self.total_rewards / trials
        total_regret = trials - self.total_rewards
        average_regret = total_regret / trials

        # Logging average reward and regret
        logging.info(f"{algorithm_name} - Average Reward: {average_reward}")
        logging.info(f"{algorithm_name} - Average Regret: {average_regret}")

        # Storing data in CSV file
        filename = f"{algorithm_name}_results.csv"
        with open(filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Bandit", "Reward", "Algorithm"])

            for i, (value, count) in enumerate(zip(self.values, self.counts)):
                writer.writerow([i, value, algorithm_name])




def comparison():
    # think of a way to compare the performances of the two algorithms VISUALLY and 
    pass

if __name__=='__main__':
   
    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")
