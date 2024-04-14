# Bandit Algorithms Project

This repository contains the implementation of two popular solutions to the multi-armed bandit problem: Epsilon Greedy and Thompson Sampling. These algorithms demonstrate the exploration-exploitation tradeoff, a key concept in reinforcement learning and decision-making processes.

## Features

- **Epsilon Greedy Algorithm**: Implements the epsilon-greedy strategy for balancing exploration and exploitation.
- **Thompson Sampling Algorithm**: Utilizes probability matching for decision-making, adapting based on observed rewards.
- **Cumulative Reward and Regret Calculation**: Functions to calculate and log the cumulative rewards and regrets during the simulation.
- **Performance Visualization**: Tools for visualizing the learning processes and performance comparisons of the algorithms across trials.

## Getting Started

### Prerequisites

To install the necessary dependencies, use the following command:

```bash
pip install -r requirements.txt
```

### Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/bandit-algorithms.git
cd bandit-algorithms
```

### Usage

To run the simulation and visualize the results, use the following commands:

```bash
python Bandit.py
```