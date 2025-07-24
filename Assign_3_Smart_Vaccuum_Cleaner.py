import numpy as np
import matplotlib.pyplot as plt

# Zone labels
zones = ["Zone A", "Zone B", "Zone C", "Zone D", "Zone E"]

# Simulated true dirt probabilities (hidden from agent)
true_dirt_probs = [0.1, 0.3, 0.7, 0.2, 0.5]  # Zone C is dirtiest

# Simulate cleaning success (reward = 1 if dirty)
def simulate_cleaning(zone):
    return 1 if np.random.rand() < true_dirt_probs[zone] else 0

# ε-Greedy strategy
def epsilon_greedy(epsilon=0.1, episodes=1000):
    Q = np.zeros(len(zones))
    N = np.zeros(len(zones))
    rewards = []
    zone_counts = np.zeros(len(zones))

    for t in range(episodes):
        if np.random.rand() < epsilon:
            zone = np.random.randint(len(zones))  # Explore
        else:
            zone = np.argmax(Q)  # Exploit best known

        reward = simulate_cleaning(zone)

        N[zone] += 1
        Q[zone] += (reward - Q[zone]) / N[zone]
        rewards.append(reward)
        zone_counts[zone] += 1

    return Q, zone_counts, rewards

# Softmax strategy
def softmax_strategy(tau=0.2, episodes=1000):
    Q = np.zeros(len(zones))
    N = np.zeros(len(zones))
    rewards = []
    zone_counts = np.zeros(len(zones))

    for t in range(episodes):
        exp_q = np.exp(Q / tau)
        probs = exp_q / np.sum(exp_q)

        zone = np.random.choice(len(zones), p=probs)
        reward = simulate_cleaning(zone)

        N[zone] += 1
        Q[zone] += (reward - Q[zone]) / N[zone]
        rewards.append(reward)
        zone_counts[zone] += 1

    return Q, zone_counts, rewards

# Run simulations
np.random.seed(42)
episodes = 1000

eg_q, eg_counts, eg_rewards = epsilon_greedy(epsilon=0.1, episodes=episodes)
sm_q, sm_counts, sm_rewards = softmax_strategy(tau=0.2, episodes=episodes)

# Print results
print("\n=== Epsilon -Greedy Results ===")
print("Estimated Q-values:", list(zip(zones, np.round(eg_q, 3))))
print("Zone Selections:", list(zip(zones, eg_counts.astype(int))))
print("Total Reward (Cleaned Dirt):", int(np.sum(eg_rewards)))

print("\n=== Softmax Results ===")
print("Estimated Q-values:", list(zip(zones, np.round(sm_q, 3))))
print("Zone Selections:", list(zip(zones, sm_counts.astype(int))))
print("Total Reward (Cleaned Dirt):", int(np.sum(sm_rewards)))

# Plotting results
def moving_average(data, window=50):
    return np.convolve(data, np.ones(window)/window, mode='valid')

plt.figure(figsize=(15, 5))

# Zone selection bar chart
plt.subplot(1, 3, 1)
plt.bar(zones, eg_counts / episodes * 100, label='ε-Greedy', alpha=0.6)
plt.bar(zones, sm_counts / episodes * 100, label='Softmax', alpha=0.6)
plt.ylabel('Zone Selection (%)')
plt.title('Cleaning Zone Selection Distribution')
plt.legend()

# Average reward over time
plt.subplot(1, 3, 2)
plt.plot(moving_average(eg_rewards), label='ε-Greedy')
plt.plot(moving_average(sm_rewards), label='Softmax')
plt.ylabel('Avg Cleaning Reward')
plt.title('Average Reward Over Time')
plt.legend()

# Cumulative reward
plt.subplot(1, 3, 3)
plt.plot(np.cumsum(eg_rewards), label='ε-Greedy')
plt.plot(np.cumsum(sm_rewards), label='Softmax')
plt.ylabel('Cumulative Cleaned Dirt')
plt.title('Cumulative Reward Over Time')
plt.legend()

plt.tight_layout()
plt.show()
