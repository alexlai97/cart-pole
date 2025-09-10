# Understanding Rewards and Costs in Cart-Pole

## The Cart-Pole Reward System Explained

### Simple Reward Structure
Cart-Pole uses one of the simplest reward systems in reinforcement learning:

```
+1 reward for every timestep the pole stays balanced
0 reward when the episode ends (pole falls)
```

That's it! No complicated scoring, no bonuses for style - just **survival time**.

### Why This Matters
This **sparse reward** design teaches us important RL concepts:
- **Delayed gratification**: Good actions now may pay off much later
- **Credit assignment**: Which action caused the eventual success/failure?
- **Exploration vs exploitation**: How do you learn when feedback is so limited?

## Episode Termination (When You "Lose")

The episode ends when **any** of these conditions are met:

### 1. Pole Angle Too Large
```
|pole_angle| > 0.2095 radians (≈ 12 degrees)
```
**Why**: Beyond this angle, the pole is falling too fast to recover

### 2. Cart Position Too Far
```  
|cart_position| > 2.4 units
```
**Why**: Cart hits the edge of the track - nowhere left to go

### 3. Maximum Steps Reached
```
timesteps >= 500
```  
**Why**: You've "solved" Cart-Pole! This is actually winning, not losing.

## What "Solving" Cart-Pole Means

### Official Definition
Cart-Pole is considered "solved" when:
- **Average reward ≥ 475** over 100 consecutive episodes
- This means averaging 475+ timesteps per episode
- Since max episode length is 500, you need 95%+ success rate

### Why 475 and not 500?
- Allows for some imperfection (episodes ending at 490, 480, etc.)
- Real-world systems aren't perfect - this threshold is more realistic
- Still requires consistently excellent performance

## Cost Functions vs Rewards

### What's a Cost Function?
In some contexts, you'll see "cost functions" instead of rewards:
- **Reward**: +1 for good, 0 for bad (maximize)
- **Cost**: 0 for good, +1 for bad (minimize)

Cart-Pole traditionally uses rewards, but you could flip it:

```python
# Reward formulation (what we use)
reward = +1  # for each timestep survived

# Cost formulation (equivalent)
cost = 0     # for each timestep survived  
cost = 1     # when episode ends (penalty)
```

### Why Use One vs The Other?
- **Rewards**: Feel more intuitive ("getting points")
- **Costs**: Common in control theory ("minimizing error")  
- **Mathematically equivalent**: Just opposite signs

## Deep Dive: Why This Reward System is Challenging

### 1. Sparse Feedback
```
Episode: [+1, +1, +1, +1, ..., +1, END]
         timestep 1→2→3→4→...→23→ FAIL
```

You get the same +1 whether you're doing great or barely surviving. The agent must learn to differentiate based on **state information**, not rewards.

### 2. No Immediate Consequence
A bad action at timestep 5 might not cause failure until timestep 25. How does the agent know which action was actually bad?

### 3. No Guidance Toward Solution
The reward doesn't tell you **how** to improve:
- It doesn't say "move left" or "move right"
- It doesn't say "you're tilting too much"  
- It just says "you survived another timestep"

## How Different Agents Handle This

### Random Agent
**Strategy**: Ignore rewards entirely, just act randomly
**Problem**: No learning from feedback
**Result**: ~23 steps average (pure luck)

### Rule-Based Agent  
**Strategy**: Use domain knowledge, ignore numerical rewards
**Advantage**: Systematic approach beats randomness
**Limitation**: Fixed strategy, can't improve

### Learning Agents (Coming Soon)
**Strategy**: Use rewards to improve policy over time
**Q-Learning**: Learn which state-action pairs lead to higher total rewards
**DQN**: Use neural networks to handle continuous state space
**Policy Gradient**: Directly learn better action probabilities

## The Mathematics of Expected Return

### What Agents Actually Try to Maximize
Not just immediate reward, but **expected total return**:

```
Total Return = R₁ + R₂ + R₃ + ... + Rₜ
             = 1 + 1 + 1 + ... + 1  
             = T (episode length)
```

### Discounted Return (Advanced Concept)
Some algorithms use **discounted** returns to prioritize immediate rewards:

```
Discounted Return = R₁ + γR₂ + γ²R₃ + ... + γᵗ⁻¹Rₜ
```

Where γ (gamma) is the discount factor (0 < γ < 1).

**In Cart-Pole**: Usually γ = 0.99 or 1.0 (no discounting)
**Why discount?**: Makes learning more stable, prioritizes immediate gains

## Practical Implications for Learning

### What This Means for Training
1. **Need many episodes**: Sparse rewards require lots of experience
2. **Variance is high**: Episode lengths vary dramatically during learning
3. **Patience required**: Improvements may come in sudden jumps, not gradually

### Evaluation Challenges
- **Single episodes**: Meaningless (too much randomness)
- **Need statistics**: Average over 100+ episodes  
- **Track variance**: Consistent agents are often better than occasionally brilliant ones

## Alternative Reward Designs (Food for Thought)

### Dense Rewards (Not Used in Standard Cart-Pole)
```python
# Hypothetical dense reward
reward = 1.0 - abs(pole_angle) - abs(cart_position)
```
**Pros**: Provides guidance every step
**Cons**: Requires domain expertise, can bias learning

### Shaped Rewards  
```python
# Reward staying near center with pole upright  
reward = 1.0 + bonus_for_small_angle + bonus_for_center_position
```
**Pros**: Faster learning
**Cons**: May learn suboptimal policies

### Why Cart-Pole Doesn't Use These
The simple +1 reward tests whether algorithms can learn from minimal feedback - a key RL challenge!

## Summary: What This Teaches Us

### Key Insights
1. **Rewards shape behavior**: Simple rewards can lead to complex learned behaviors
2. **Sparsity is hard**: Limited feedback makes learning difficult  
3. **Episode thinking**: Agents must consider long-term consequences
4. **Statistical evaluation**: Individual episodes tell you little

### Why This Matters for RL
Cart-Pole's reward system is a perfect introduction to core RL challenges:
- **Temporal credit assignment**
- **Exploration vs exploitation**  
- **Sample efficiency**
- **Generalization**

Understanding these concepts through Cart-Pole prepares you for more complex environments with intricate reward structures.

---

**Next**: Try running the agents and see how their different approaches handle this challenging reward structure! The simple +1 reward leads to surprisingly rich learning dynamics.