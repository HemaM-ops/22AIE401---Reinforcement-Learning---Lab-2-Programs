import numpy as np
import matplotlib.pyplot as plt

# Define ad labels
ads = ["Ad A", "Ad B", "Ad C", "Ad D"]

# True CTRs for each ad (simulate user environment)
true_ctrs = [0.05, 0.10, 0.20, 0.15]  # Ad C is the best

# Simulate user click based on the true CTR of selected ad
def simulate_click(action):
    return 1 if np.random.rand() < true_ctrs[action] else 0

# ε-Greedy Strategy
def epsilon_greedy(epsilon=0.1, steps=2000):
    Q = np.zeros(len(ads))           # Estimated Q-values
    N = np.zeros(len(ads))           # Count of actions
    rewards = []
    action_counts = np.zeros(len(ads))

    for t in range(steps):
        if np.random.rand() < epsilon:
            action = np.random.randint(len(ads))  # Explore
        else:
            action = np.argmax(Q)                 # Exploit

        reward = simulate_click(action)

        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]  # Incremental mean
        rewards.append(reward)
        action_counts[action] += 1

    return Q, action_counts, rewards

# Softmax Strategy
def softmax_strategy(tau=0.2, steps=2000):
    Q = np.zeros(len(ads))           # Estimated Q-values
    N = np.zeros(len(ads))           # Count of actions
    rewards = []
    action_counts = np.zeros(len(ads))

    for t in range(steps):
        exp_q = np.exp(Q / tau)
        probs = exp_q / np.sum(exp_q)

        action = np.random.choice(len(ads), p=probs)
        reward = simulate_click(action)

        N[action] += 1
        Q[action] += (reward - Q[action]) / N[action]
        rewards.append(reward)
        action_counts[action] += 1

    return Q, action_counts, rewards

# Run Simulations
np.random.seed(1)
steps = 2000

# ε-Greedy
eg_q, eg_counts, eg_rewards = epsilon_greedy(epsilon=0.1, steps=steps)

# Softmax
sm_q, sm_counts, sm_rewards = softmax_strategy(tau=0.2, steps=steps)

# Print Outputs
print("\n=== Epsilon -Greedy Results ===")
print("Estimated Q-values:", list(zip(ads, np.round(eg_q, 3))))
print("Action Counts:", list(zip(ads, eg_counts.astype(int))))
print("Total Reward:", int(np.sum(eg_rewards)))

print("\n=== Softmax Results ===")
print("Estimated Q-values:", list(zip(ads, np.round(sm_q, 3))))
print("Action Counts:", list(zip(ads, sm_counts.astype(int))))
print("Total Reward:", int(np.sum(sm_rewards)))

# Plotting
def moving_average(data, window=100):
    return np.convolve(data, np.ones(window)/window, mode='valid')

plt.figure(figsize=(15, 5))

# Action distribution
plt.subplot(1, 3, 1)
plt.bar(ads, eg_counts / steps * 100, label='ε-Greedy', alpha=0.6)
plt.bar(ads, sm_counts / steps * 100, label='Softmax', alpha=0.6)
plt.ylabel('Action Selection (%)')
plt.title('Ad Selection Distribution')
plt.legend()

# Average reward over time
plt.subplot(1, 3, 2)
plt.plot(moving_average(eg_rewards), label='ε-Greedy')
plt.plot(moving_average(sm_rewards), label='Softmax')
plt.ylabel('Avg Reward (Moving Avg)')
plt.title('Average Reward Over Time')
plt.legend()

# Cumulative reward
plt.subplot(1, 3, 3)
plt.plot(np.cumsum(eg_rewards), label='ε-Greedy')
plt.plot(np.cumsum(sm_rewards), label='Softmax')
plt.ylabel('Cumulative Reward')
plt.title('Total Reward Over Time')
plt.legend()

plt.tight_layout()
plt.show()
