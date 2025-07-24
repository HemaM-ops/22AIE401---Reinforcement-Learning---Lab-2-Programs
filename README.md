# 22AIE401---Reinforcement-Learning---Lab-2-Programs
Multi - Armed Bandit Problems. Epsilon - Greedy and Softmax techniques


<b>Assignment</b>

1. Personalized News Recommendation
Model a news website that shows one of 5 article categories (Tech, Sports, Business, Health,
Travel). The reward is 1 if the user clicks, 0 otherwise.
• Implement ε-Greedy and Softmax strategies.
• Simulate user preferences with predefined means.
• Plot and compare action distributions and average rewards.

<br>

2. Ad Placement on a Website
You have 4 ads with different click-through rates (CTRs).
• Use ε-Greedy and Softmax to identify the best ad.
• Simulate for 2000 steps.
• Analyze which method converges faster and more accurately.

<br>
3. Smart Vacuum Cleaner – Spot Cleaning Strategy (Anchor Example)
Your smart vacuum cleaner operates in a room with 5 predefined dirty zones. Each time it starts
cleaning, it must choose one spot to begin with. The likelihood of dirt (and hence reward) in
each zone varies.
• Model this as a 5-armed bandit problem where:
◦ Each arm represents a dirty zone (Zone A to Zone E).
◦ The reward is +1 if the zone is actually dirty and cleaned successfully, else 0.
◦ Each zone has a different true dirt probability (you can simulate these using a
normal or Bernoulli distribution).

<br>
Your Tasks:
• Implement both ε-Greedy and Softmax strategies for 1000 episodes.
• Track and compare:
◦ Estimated action-values,
◦ Percentage of times each zone is selected,
◦ Average reward over time.
• Which strategy learns to target the dirtiest zones more effectively?


<br>
4. LLM Integration Task
Use ChatGPT or any LLM to:
• Ask the LLM to critique their code (e.g., “How can I make my ε-Greedy implementation
more efficient?”)
• Ask the LLM to summarize differences between ε-Greedy and Softmax using their own
simulation outputs
• Prompt examples:
• “Explain why my ε-Greedy agent chose suboptimal arms more often.”
• “What could be the reason Softmax worked better in the pricing experiment?”
