import numpy as np
import matplotlib.pyplot as plt

# Categories
categories = ["Tech", "Sports", "Business", "Health", "Travel"]

# True user preferences (click-through rates)
true_means = [0.3, 0.5, 0.2, 0.1, 0.4]  # Corresponds to categories

# Simulate user clicking behavior
def simulate_user_response(action):
    return 1 if np.random.rand() < true_means[action] else 0

# ε-Greedy Strategy
def epsilon_greedy(epsilon=0.1, episodes=1000):
    Q = np.zeros(len(categories))       # Estimated values
    N = np.zeros(len(categories))       # Number of times each action was selected
    rewards = []
    action_counts = np.zeros(len(categories))

    for t in range(episodes):
        # ε-greedy decision
        if np.random.rand() < epsilon:
            action = np.random.randint(len(categories))  # Explore
        else:
            action = np.argmax(Q)                        # Exploit

        reward = simulate_user_response(action)

        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]   # Incremental update

        rewards.append(reward)
        action_counts[action] += 1

    return Q, action_counts, rewards

# Softmax Strategy
def softmax_strategy(tau=0.5, episodes=1000):
    Q = np.zeros(len(categories))       # Estimated values
    N = np.zeros(len(categories))       # Number of times each action was selected
    rewards = []
    action_counts = np.zeros(len(categories))

    for t in range(episodes):
        # Compute softmax probabilities
        exp_q = np.exp(Q / tau)
        probs = exp_q / np.sum(exp_q)

        action = np.random.choice(len(categories), p=probs)

        reward = simulate_user_response(action)

        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]

        rewards.append(reward)
        action_counts[action] += 1

    return Q, action_counts, rewards

# Run simulations
np.random.seed(42)
episodes = 1000

# ε-Greedy Results
eg_q, eg_counts, eg_rewards = epsilon_greedy(epsilon=0.1, episodes=episodes)

# Softmax Results
sm_q, sm_counts, sm_rewards = softmax_strategy(tau=0.5, episodes=episodes)

# Print Outputs
print("\n=== Epsilon-Greedy ===")
print("Estimated Q-values:", list(zip(categories, np.round(eg_q, 3))))
print("Selection counts:", list(zip(categories, eg_counts.astype(int))))
print("Total reward:", int(np.sum(eg_rewards)))

print("\n=== Softmax ===")
print("Estimated Q-values:", list(zip(categories, np.round(sm_q, 3))))
print("Selection counts:", list(zip(categories, sm_counts.astype(int))))
print("Total reward:", int(np.sum(sm_rewards)))

# Plotting

# Action distributions
plt.figure(figsize=(14, 5))

plt.subplot(1, 3, 1)
plt.bar(categories, eg_counts / episodes * 100, label='ε-Greedy', alpha=0.7)
plt.bar(categories, sm_counts / episodes * 100, label='Softmax', alpha=0.7)
plt.ylabel('Action Selection (%)')
plt.title('Action Distribution')
plt.legend()

# Average rewards over time
def moving_average(x, window=50):
    return np.convolve(x, np.ones(window)/window, mode='valid')

plt.subplot(1, 3, 2)
plt.plot(moving_average(eg_rewards), label='ε-Greedy')
plt.plot(moving_average(sm_rewards), label='Softmax')
plt.ylabel('Average Reward')
plt.title('Average Reward over Time')
plt.legend()

# Cumulative reward comparison
plt.subplot(1, 3, 3)
plt.plot(np.cumsum(eg_rewards), label='ε-Greedy')
plt.plot(np.cumsum(sm_rewards), label='Softmax')
plt.ylabel('Cumulative Reward')
plt.title('Cumulative Reward over Time')
plt.legend()

plt.tight_layout()
plt.show()
