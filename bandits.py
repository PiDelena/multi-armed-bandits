# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 00:42:01 2026

@author: Paola Delena

Independent implementation of the stationary multi-armed bandit examples in
Sutton and Barto (2018), Reinforcement Learning: An Introduction, Chapter 2.
"""
import numpy as np
import matplotlib.pyplot as plt

class Bandit:
    """
    Stationary k-armed bandit environment.

    Each action a has a TRUE value q*(a), generated from a Normal Distribution:
        q*(a) ~ Normal(true_mean, true_std)

    Parameters
    -------------------------
    k : int
        Number of arms (actions).
    true_mean : float
        Mean used to generate the true action values q*(a).
    true_std : float
        Standard deviation used to generate the true action values q*(a).
    reward_std : float
        Standard deviation of the reward distribution around q*(a).
    seed : int or None
        Random seed for reproducibility.
    """

    def __init__(self, k:int=10, true_mean:float=0.0, true_std:float=1.0, reward_std:float=1.0, seed:int|None = None):
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if true_std <= 0:
            raise ValueError("true_std must be positive.")
        if reward_std <= 0:
            raise ValueError("reward_std must be positive.")

        self.k = k
        self.true_mean = true_mean
        self.true_std = true_std
        self.reward_std = reward_std
        self.rng = np.random.default_rng(seed) # Random number generator

        # True action values q*(a) for each k
        self.q_star = self.rng.normal( loc=self.true_mean, scale=self.true_std,size=self.k) 

    def pull(self, action: int) -> float:
        """
        Pull the lever. Selects an action and receive a reward.

        Parameters
        ----------
        action : int
            Index of the selected lever (action) (0 to k-1).

        Returns
        ----------
        float
            Observed reward.
        """
        
        if not 0 <= action < self.k:
            raise ValueError(f"action must be in [0, {self.k - 1}]")

        reward = self.rng.normal(loc=self.q_star[action], scale=self.reward_std)
        return reward

    def optimal_action(self) -> int:
        """
        Return the index of the action with the highest true value q*(a).
        """
        return int(np.argmax(self.q_star))

    def info(self) -> None:
        """
        Print basic environment information.
        """
        print("Bandit environment")
        print(f"k = {self.k}")
        print(f"q_star = {self.q_star}")
        print(f"optimal action = {self.optimal_action()}")
        print(f"optimal value = {self.q_star[self.optimal_action()]:.4f}")
        
    def plot_distributions(self, n_samples: int = 2000):
        """
        Plot the reward distributions of all bandit arms using violin plots.
    
        Parameters
        ----------
        n_samples : int
            Number of reward samples drawn from each arm to estimate
            the distribution shape.
        """
        samples = []
    
        # Generate samples for each arm
        for a in range(self.k):
            rewards = self.rng.normal(loc=self.q_star[a], scale=self.reward_std, size=n_samples)
            samples.append(rewards)
    
        plt.figure(figsize=(10,5))
        plt.violinplot(samples, showmeans=True)
        plt.xticks(np.arange(1, self.k + 1), np.arange(1, self.k + 1))
        plt.xlabel("Action (Arm)")
        plt.ylabel("Reward distribution")
        plt.title("Reward Distributions of k-Armed Bandit")
    
        # mark the true means
        plt.scatter(np.arange(1, self.k + 1), self.q_star, color="red", label="True mean $q_*(a)$")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.show()
        
class GreedyAgent:
    
    def __init__(self, k:int, initial_value:float=0.0, alpha:float|None=None, seed:int|None=None):
        """
        Greedy agent for k-armed bandit.

        Parameters
        ----------
        k : int
            Number of actions.
        initial_value : float
            Initial estimate for Q(a). Use a high value for optimistic initialization.
        alpha : float or None
            Constant step-size. If None, use sample-average update.
        seed : int or None
            Random seed for reproducibility.
        """
        self.k = k
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.Q = np.full(k, initial_value, dtype=float) # Estimated action values # Array of lenght k with initial values
        self.N = np.zeros(k, dtype=int) # Action counts

    def select_action(self) -> int:
        """
        Select the action with the highest estimated value.
        Random tie-breaking.
        """
        max_value = np.max(self.Q)
        candidates = np.flatnonzero(self.Q == max_value)
        return int(self.rng.choice(candidates))

    def update(self, action: int, reward: float):
        """
        Update Q(a) using either sample-average or constant step-size.
        """
        self.N[action] += 1
        if self.alpha is None:
            self.Q[action] += (reward - self.Q[action]) / self.N[action] # Sample-average update
        else:
            self.Q[action] += self.alpha * (reward - self.Q[action]) # Constant step-size update

class EpsilonGreedyAgent:
    
    def __init__(self, k:int, epsilon:float=0.1, initial_value:float=0.0, alpha:float|None=None, seed:int|None=None):
        """
        Epsilon-greedy agent for k-armed bandit.
        
        Parameters
        ----------
        k : int
            Number of actions.
        epsilon : float
            Exploration probability.
        initial_value : float
            Initial estimate for Q(a).
        seed : int or None
            Random seed for reproducibility.
        """
        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be in [0, 1].")
        
        self.k = k
        self.epsilon = epsilon
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.Q = np.full(k, initial_value, dtype=float) # Estimated action values
        self.N = np.zeros(k, dtype=int) # Action counts

    def select_action(self) -> int:
        """
        Select an action using epsilon-greedy policy.
        """
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(0, self.k))
        
        max_value = np.max(self.Q)
        candidates = np.flatnonzero(self.Q == max_value)
        return int(self.rng.choice(candidates))

    def update(self, action: int, reward: float):
        """
        Update Q(a) using either sample-average or constant step-size.
        """
        self.N[action] += 1
        if self.alpha is None:
            self.Q[action] += (reward - self.Q[action]) / self.N[action] # Sample-average update
        else:
            self.Q[action] += self.alpha * (reward - self.Q[action]) # Constant step-size update

class UCBAgent:
    
    def __init__(self, k:int, c:float=2.0, alpha:float|None=None, seed:int|None=None):
        """
        Upper Confidence Bound agent.

        Parameters
        ----------
        k : int
            Number of actions.
        c : float
            Exploration parameter.
        alpha : float or None
            Constant step-size. If None, use sample-average update.
        seed : int or None
            Random seed.
        """
        self.k = k
        self.c = c
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.Q = np.zeros(k)
        self.N = np.zeros(k, dtype=int)
        self.t = 0

    def select_action(self) -> int:
        """
        Select action according to UCB rule.
        """
            
        self.t += 1 
        
        # Ensure every action is tried at least once
        
            # Random of untried actions
        untried = np.flatnonzero(self.N == 0)
        if len(untried) > 0:
            return int(self.rng.choice(untried))
        
            # Try in order
        # for a in range(self.k): # Tried in order, cound be CHANGED
        #     if self.N[a] == 0:
        #         return a
            
        ucb_values = self.Q + self.c * np.sqrt(np.log(self.t) / self.N)
        
        # As book equation: 
        # return int(np.argmax(ucb_values))
        
        # Equal probability to be selected, replacing argmax
        max_value = np.max(ucb_values)
        candidates = np.flatnonzero(ucb_values == max_value)
        return int(self.rng.choice(candidates))
        

    def update(self, action: int, reward: float):
        """
        Update Q(a) using either sample-average or constant step-size.
        """
        self.N[action] += 1
        if self.alpha is None:
            self.Q[action] += (reward - self.Q[action]) / self.N[action] # Sample-average update
        else:
            self.Q[action] += self.alpha * (reward - self.Q[action]) # Constant step-size update
            
class GradientBanditAgent:
    
    def __init__(self, k:int, alpha:float=0.1, use_baseline:bool=True, seed:int|None=None):
        """
        Gradient bandit agent with softmax action selection.

        Parameters
        ----------
        k : int
            Number of actions.
        alpha : float
            Step-size parameter.
        use_baseline : bool
            If True, use the average reward as baseline.
            If False, baseline is taken as 0.
        seed : int or None
            Random seed for reproducibility.
        """
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if alpha <= 0:
            raise ValueError("alpha must be positive.")
        
        self.k = k
        self.alpha = alpha
        self.use_baseline = use_baseline
        self.rng = np.random.default_rng(seed)
        self.H = np.zeros(k, dtype=float) # Action preferences H(a)
        self.pi = np.full(k, 1.0 / k, dtype=float) # Action probabilities pi(a)
        self.avg_reward = 0.0 # Average reward baseline
        self.t = 0 # Total number of steps experienced

    def _softmax(self) -> np.ndarray:
        """
        Compute softmax probabilities from preferences H.
        Numerical stabilization to avoid H values growing and creating overflow in exponential 
        """
        shifted_H = self.H - np.max(self.H) # - max(H) to apply numerical stabilization
        exp_H = np.exp(shifted_H) 
        return exp_H / np.sum(exp_H)

    def select_action(self) -> int:
        """
        Select an action according to softmax probabilities.
        """
        self.pi = self._softmax()
        return int(self.rng.choice(self.k, p=self.pi)) # Choice with probabilities based on pi

    def update(self, action:int, reward:float):
        """
        Update action preferences using the gradient bandit rule.
        
        High reward increases preference while others decrease.
        
        
        """
        self.t += 1
        
        # One-hot vector to update H of all actions at a time
        one_hot = np.zeros(self.k, dtype=float) 
        one_hot[action] = 1.0
        
        # Incremental update of average reward baseline
        self.avg_reward += (reward - self.avg_reward) / self.t
        
        if self.use_baseline:
            baseline = self.avg_reward
        else:
            baseline = 0.0

        # Gradient bandit update (vector form)
        self.H += self.alpha * (reward - baseline) * (one_hot - self.pi)

def run_bandit(bandit, agent, steps: int = 1000):
    """
    Run one bandit experiment for a given number of steps.

    Parameters
    ----------
    bandit : Bandit
        The environment.
    agent : GreedyAgent
        The agent.
    steps : int
        Number of interaction steps.

    Returns
    -------
    rewards : np.ndarray
        Reward obtained at each step.
    actions : np.ndarray
        Action selected at each step.
    optimal_actions : np.ndarray
        1 if the selected action was optimal, 0 otherwise.
    cumulative_rewards : np.ndarray
        Cumulative sum of rewards.
    average_rewards : np.ndarray
        Cumulative average reward up to each step.
    """

    rewards = np.zeros(steps, dtype=float)
    actions = np.zeros(steps, dtype=int)
    optimal_actions = np.zeros(steps, dtype=int)

    optimal_action = bandit.optimal_action()

    for t in range(steps):
        action = agent.select_action()
        reward = bandit.pull(action)
        agent.update(action, reward)

        rewards[t] = reward
        actions[t] = action
        optimal_actions[t] = int(action == optimal_action)

    cumulative_rewards = np.cumsum(rewards)
    average_rewards = cumulative_rewards / np.arange(1, steps + 1)

    return rewards, actions, optimal_actions, cumulative_rewards, average_rewards
    
def build_standard_agents(k:int, epsilon:float, c:float):
    """
    Build fresh agent instances for the standard 10-armed testbed.
    """
    agents = [
        ("Greedy", GreedyAgent(k=k)),
        ("Optimistic Greedy (Q1=5, α=0.1)", GreedyAgent(k=k, initial_value=5.0, alpha=0.1)),
        (f"Epsilon-Greedy (ε={epsilon})", EpsilonGreedyAgent(k=k, epsilon=epsilon)),
        (f"UCB (c={c})", UCBAgent(k=k, c=c)),
    ]
    return agents

def build_gradient_agents(k:int, alpha_grad:float):
    """
    Build fresh agent instances for the gradient bandit experiment.
    """
    agents = [
        (f"Gradient Bandit (α={alpha_grad}, baseline)", 
         GradientBanditAgent(k=k, alpha=alpha_grad, use_baseline=True)),
        (f"Gradient Bandit (α={alpha_grad}, no baseline)", 
         GradientBanditAgent(k=k, alpha=alpha_grad, use_baseline=False)),
    ]
    return agents

def run_testbed(n_runs:int, steps:int, k:int, agent_builder, true_mean:float=0.0, true_std:float=1.0, reward_std:float=1.0, base_seed:int|None=None):
    """
    Run the k-armed bandit testbed over many independent runs.

    Parameters
    ----------
    n_runs : int
        Number of independent runs (e.g., 2000 in the book).
    steps : int
        Number of steps per run (e.g., 1000 in the book).
    k : int
        Number of arms.
    agent_builder : callable
        Function that returns a fresh list of (name, agent) pairs.
    true_mean : float
        Mean of the distribution used to generate q*(a).
    true_std : float
        Std of the distribution used to generate q*(a).
    reward_std : float
        Reward noise std.
    base_seed : int or None
        Optional base seed for reproducibility.

    Returns
    -------
    results : dict
        For each agent name:
        - avg_reward_per_step
        - optimal_action_percentage
    """
    # Instantiate once to get names
    initial_agents = agent_builder()
    names = [name for name, _ in initial_agents]

    reward_sums = {name: np.zeros(steps, dtype=float) for name in names}
    optimal_sums = {name: np.zeros(steps, dtype=float) for name in names}

    # Independent streams for the problem, each agent, and its rewards.
    # A fixed base_seed reproduces both experiments and action tie-breaking.
    for run_seed in np.random.SeedSequence(base_seed).spawn(n_runs):
        streams = run_seed.spawn(1 + 2 * len(names))
        bandit = Bandit(
            k=k,
            true_mean=true_mean,
            true_std=true_std,
            reward_std=reward_std,
            seed=int(streams[0].generate_state(1)[0])
        )

        agents = agent_builder()
        for i, (name, agent) in enumerate(agents):
            agent.rng = np.random.default_rng(streams[1 + 2 * i])
            bandit.rng = np.random.default_rng(streams[2 + 2 * i])
            rewards, actions, optimal_flags, _, _ = run_bandit(bandit, agent, steps)

            reward_sums[name] += rewards
            optimal_sums[name] += optimal_flags

    results = {}
    for name in names:
        results[name] = {
            "avg_reward_per_step": reward_sums[name] / n_runs,
            "optimal_action_percentage": 100.0 * optimal_sums[name] / n_runs
        }

    return results

def plot_testbed_results(results, steps:int, title_prefix:str="10-armed Testbed"):
    """
    Plot average reward per step and % optimal action per step across runs.
    """
    x = np.arange(1, steps + 1)

    fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for name, data in results.items():
        ax[0].plot(x, data["avg_reward_per_step"], label=name)
        ax[1].plot(x, data["optimal_action_percentage"], label=name)

    ax[0].set_title(f"{title_prefix} - Average Reward")
    ax[0].set_ylabel("Average reward")
    ax[0].grid(alpha=0.3)
    ax[0].legend()

    ax[1].set_title(f"{title_prefix} - % Optimal Action")
    ax[1].set_ylabel("% optimal action")
    ax[1].set_xlabel("Steps")
    ax[1].set_ylim(0, 100)
    ax[1].grid(alpha=0.3)
    ax[1].legend()

    plt.tight_layout()
    plt.show()

def main():
    """Run the two stationary 10-armed bandit experiments."""
    k = 10
    steps = 1000
    n_runs = 2000
    epsilon = 0.1
    c = 2.0

    standard_results = run_testbed(
        n_runs=n_runs,
        steps=steps,
        k=k,
        agent_builder=lambda: build_standard_agents(k=k, epsilon=epsilon, c=c),
        true_mean=0.0,
        true_std=1.0,
        reward_std=1.0,
        base_seed=12345
    )
    plot_testbed_results(standard_results, steps, title_prefix="Standard 10-armed Testbed")

    # A subset of the parameter comparison in Figure 2.5 (alpha = 0.1).
    alpha_grad = 0.1
    gradient_results = run_testbed(
        n_runs=n_runs,
        steps=steps,
        k=k,
        agent_builder=lambda: build_gradient_agents(k=k, alpha_grad=alpha_grad),
        true_mean=4.0,
        true_std=1.0,
        reward_std=1.0,
        base_seed=54321
    )
    plot_testbed_results(gradient_results, steps, title_prefix="Gradient Bandit Testbed")


if __name__ == "__main__":
    main()
