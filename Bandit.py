from logs import *
from abc import ABC, abstractmethod

import csv
import numpy as np
import matplotlib.pyplot as plt

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
        self.p = p
        self.rewards = []
        self.cumulative_reward = 0
        self.cumulative_regret = 0
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

    def calculate_cumulative_reward(self):
        """Calculate the cumulative reward of the bandit."""
        self.cumulative_reward = sum(self.rewards)

    def print_cumulative_reward(self):
        """Log the cumulative reward of the bandit."""
        logger.info(f'Cumulative Reward for Bandit (p={self.p}): {self.cumulative_reward}')

    def print_cumulative_regret(self):
        """Log the cumulative regret of the bandit."""
        logger.info(f'Cumulative Regret for Bandit (p={self.p}): {self.cumulative_regret}')

    def store_rewards_to_csv(self, algorithm_name):
        """Store the rewards obtained by the bandit to a CSV file without overwriting existing data.
        
        Args:
            algorithm_name (str): The name of the algorithm to label the data in the CSV file.
        """
        filename = 'bandits_rewards.csv'  
        mode = 'a'  
        header = False 

        try:
            with open(filename, 'r') as file:
                header = False
        except FileNotFoundError:
            header = True

        with open(filename, mode=mode, newline='') as file:
            writer = csv.writer(file)
            if header:
                writer.writerow(['Bandit', 'Reward', 'Algorithm'])

            for reward in self.rewards:
                writer.writerow([self.p, reward, algorithm_name])
        logger.info(f'Rewards data for {algorithm_name} bandit (p={self.p}) has been stored in {filename}.')


    def report(self, algorithm_name):
        logger.info(f"{algorithm_name} Bandit Algorithm")
        logger.info(f"True Mean: {self.p}")
        logger.info(f"Cumulative Regret: {self.cumulative_regret}")

class Visualization:
    """Class for visualizing bandit algorithms' performance through various plot types."""

    def plot_learning_process(self, bandits, num_trials, title):
        """Plot the learning process of bandit algorithms over a given number of trials.

        Args:
            bandits (list): A list of bandit instances.
            num_trials (int): Total number of trials to plot.
            title (str): Title of the plot.
        """
        plt.figure(figsize=(10, 5))
        for bandit in bandits:
            plt.plot(range(1, num_trials + 1), bandit.cumulative_average_rwrd, label=f'{bandit.__class__.__name__} Bandit {bandit.p:.2f}')
        plt.title(title)
        plt.xlabel('Number of Trials')
        plt.ylabel('Cumulative Average Reward')
        plt.legend()
        plt.show()

    def plot_epsilon_greedy(self, epsilon_greedy_bandits, num_trials, title):
        """Plot the estimated rewards of Epsilon Greedy bandits over trials.

        Args:
            epsilon_greedy_bandits (list): A list of Epsilon Greedy bandit instances.
            num_trials (int): Total number of trials to plot.
            title (str): Title of the plot.
        """
        plt.figure(figsize=(10, 5))
        for bandit in epsilon_greedy_bandits:
            plt.plot(bandit.cumulative_average_rwrd, label=f'Bandit {bandit.p:.2f}')
        plt.title(title)
        plt.xlabel('Number of Trials')
        plt.ylabel('Estimated Reward')
        plt.legend()
        plt.show()

    def plot_thompson_sampling(self, thompson_bandits, num_trials, title):
        """Plot the estimated rewards of Thompson Sampling bandits over trials.

        Args:
            thompson_bandits (list): A list of Thompson Sampling bandit instances.
            num_trials (int): Total number of trials to plot.
            title (str): Title of the plot.
        """
        plt.figure(figsize=(10, 5))
        for bandit in thompson_bandits:
            plt.plot(bandit.cumulative_average_rwrd, label=f'Bandit {bandit.p:.2f}')
        plt.title(title)
        plt.xlabel('Number of Trials')
        plt.ylabel('Estimated Reward')
        plt.legend()
        plt.show()

    def plot_cumulative_rewards(self, epsilon_greedy_bandits, thompson_bandits, num_trials):
        """Compare and plot cumulative rewards of Epsilon Greedy and Thompson Sampling bandits.

        Args:
            epsilon_greedy_bandits (list): A list of Epsilon Greedy bandit instances.
            thompson_bandits (list): A list of Thompson Sampling bandit instances.
            num_trials (int): Total number of trials to plot.
        """
        plt.figure(figsize=(10, 5))
        for bandit in epsilon_greedy_bandits:
            cumulative_rewards = np.cumsum(bandit.rewards)
            plt.plot(range(1, num_trials + 1), cumulative_rewards, label=f'Epsilon Greedy Bandit {bandit.p:.2f}')
        for bandit in thompson_bandits:
            cumulative_rewards = np.cumsum(bandit.rewards)
            plt.plot(range(1, num_trials + 1), cumulative_rewards, label=f'Thompson Sampling Bandit {bandit.p:.2f}')
        plt.title('Cumulative Rewards Comparison')
        plt.xlabel('Number of Trials')
        plt.ylabel('Cumulative Rewards')
        plt.legend()
        plt.show()

def compare_cumulative_regret(epsilon_greedy_bandits, thompson_bandits, num_trials):
    """Compare the cumulative regret of Epsilon Greedy and Thompson Sampling algorithms over trials.

    Args:
        epsilon_greedy_bandits (list): A list of Epsilon Greedy bandit instances.
        thompson_bandits (list): A list of Thompson Sampling bandit instances.
        num_trials (int): Total number of trials to compare.
    """
    plt.figure(figsize=(10, 5))
    for bandit in epsilon_greedy_bandits:
        plt.plot(range(1, num_trials + 1), bandit.cumulative_regret_dict.values(), label=f'Epsilon Greedy Bandit {bandit.p:.2f}')
    for bandit in thompson_bandits:
        plt.plot(range(1, num_trials + 1), bandit.cumulative_regret_dict.values(), label=f'Thompson Sampling Bandit {bandit.p:.2f}')
    plt.title('Comparison of Cumulative Regret')
    plt.xlabel('Number of Trials')
    plt.ylabel('Cumulative Regret')
    plt.legend()
    plt.show()



class EpsilonGreedy(Bandit):
    """Class representing the Epsilon Greedy bandit algorithm."""
    
    def __init__(self, true_mean, initial_epsilon):
        super().__init__(true_mean)
        self.p_estimate = 0
        self.N = 0
        self.cumulative_regret_dict = {}
        self.cumulative_average_rwrd = []
        self.epsilon = initial_epsilon

    def pull(self):
        """Return a reward based on the bandit's mean."""
        return np.random.randn() + self.p

    def update(self, res):
        """Update the bandit's state with the new result."""
        self.N += 1
        self.p_estimate = ((self.N - 1) * self.p_estimate + res) / self.N
        self.rewards.append(res)

    def experiment(self, num_trials):
        """Conduct a series of trials to update bandit's estimates and calculate regret."""
        for trial in range(num_trials):
            if np.random.random() < self.epsilon:
                action = np.random.randn() + self.p
            else:
                action = self.p_estimate

            reward = self.pull()
            self.update(reward)
            self.cumulative_regret += (self.p - action)
            self.cumulative_regret_dict[trial + 1] = self.cumulative_regret
            self.cumulative_average_rwrd.append(np.mean(self.rewards))

            # Decay epsilon by 1/t
            self.epsilon = 1 / (trial + 1)

    def __repr__(self):
        return f'EpsilonGreedy(true_mean={self.p}, initial_epsilon={self.epsilon})'



class ThompsonSampling(Bandit):
    """Class representing the Thompson Sampling bandit algorithm."""
    
    def __init__(self, true_mean):
        super().__init__(true_mean)
        self.alpha = 1
        self.beta = 1
        self.N = 0
        self.cumulative_regret_dict = {}
        self.cumulative_average_rwrd = []
    
    def pull(self):
        """Return a reward based on a beta distribution sampling."""
        return np.random.beta(self.alpha, self.beta)

    def update(self, res):
        """Update alpha or beta based on the result to influence future samples."""
        if res == 1:
            self.alpha += 1
        else:
            self.beta += 1
        self.rewards.append(res)

    def experiment(self, num_trials):
        """Conduct a series of trials to update bandit's estimates and calculate regret."""
        for trial in range(num_trials):
            action = self.pull()
            reward = np.random.randn() + self.p
            self.update(reward)
            self.cumulative_regret += (self.p - action)
            self.cumulative_regret_dict[trial + 1] = self.cumulative_regret
            self.cumulative_average_rwrd.append(np.mean(self.rewards))

    def __repr__(self):
        return f'ThompsonSampling(true_mean={self.p})'

if __name__=='__main__':
   
    bandit_rewards = [1, 2, 3, 4]
    num_trials = 20000
    initial_epsilon = 0.1  

    epsilon_greedy_bandits = [EpsilonGreedy(reward, initial_epsilon) for reward in bandit_rewards]

    thompson_bandits = [ThompsonSampling(reward) for reward in bandit_rewards]

    for bandit_type in [epsilon_greedy_bandits, thompson_bandits]:
        for bandit in bandit_type:
            bandit.experiment(num_trials)
            bandit.report(bandit.__class__.__name__)

    visualization = Visualization()
    visualization.plot_epsilon_greedy(epsilon_greedy_bandits, num_trials, 'Epsilon Greedy')
    visualization.plot_thompson_sampling(thompson_bandits, num_trials, 'Thompson Sampling')
    visualization.plot_learning_process(epsilon_greedy_bandits, num_trials, 'Learning Process: Epsilon Greedy')
    visualization.plot_learning_process(thompson_bandits, num_trials, 'Learning Process: Thompson Sampling')
    visualization.plot_cumulative_rewards(epsilon_greedy_bandits, thompson_bandits, num_trials)

    for bandit_type in [epsilon_greedy_bandits, thompson_bandits]:
        for bandit in bandit_type:
            bandit.experiment(num_trials)
            bandit.report(bandit.__class__.__name__)
            bandit.calculate_cumulative_reward()
            bandit.print_cumulative_reward()
            bandit.store_rewards_to_csv(bandit.__class__.__name__)

# Comparison of cumulative regrets
compare_cumulative_regret(epsilon_greedy_bandits, thompson_bandits, num_trials)