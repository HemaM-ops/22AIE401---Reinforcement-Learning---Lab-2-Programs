import numpy as np 
import matplotlib.pyplot as plt 
np.random.seed(42)  # reproducibility 
n_arms = 10 
steps = 1000 
epsilon = 0.1 
true_values = np.random.normal(0, 1, n_arms)

#tau - temperature parameter for softmax
# A lower tau means more exploitation, a higher tau means more exploration
# Softmax function to convert Q-values to probabilities
def softmax(q_values, tau=0.5): 
    exp_q = np.exp(q_values / tau) 
    return exp_q / np.sum(exp_q) 
Q = np.zeros(n_arms) 
action_counts = np.zeros(n_arms) 
avg_rewards = [] 
for t in range(1, steps + 1): 
    probs = softmax(Q, tau=0.5) 
    action = np.random.choice(np.arange(n_arms), p=probs) 
    reward = np.random.normal(true_values[action], 1) 
    action_counts[action] += 1 
    Q[action] += (1 / action_counts[action]) * (reward - Q[action]) 
    avg_rewards.append(np.mean(avg_rewards[-1:] + [reward]) if 
avg_rewards else reward) 
action_percentages = 100 * action_counts / steps 
print("True Mean Rewards of Each Arm:\n", np.round(true_values, 3)) 
print("\nEstimated Q-values:\n", np.round(Q, 3)) 
print("\nAction Selection Percentage:\n", 
np.round(action_percentages, 1)) 
plt.plot(avg_rewards) 
plt.xlabel("Steps") 
plt.ylabel("Average Reward") 
plt.title("Softmax Average Reward over Time") 
plt.grid() 
plt.show()