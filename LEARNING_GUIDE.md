# REASAN Codebase Learning Guide
## A Step-by-Step Guide to Understanding Reinforcement Learning for Legged Robots

This guide will help you understand how this codebase trains locomotion, safety shield (filter), and navigation policies using reinforcement learning. We'll start from the entry scripts and dive deeper into each component.

---

## Table of Contents

1. [High-Level Architecture Overview](#1-high-level-architecture-overview)
2. [Understanding the Entry Scripts](#2-understanding-the-entry-scripts)
3. [The Training Pipeline Flow](#3-the-training-pipeline-flow)
4. [Step-by-Step Study Path](#4-step-by-step-study-path)
5. [Key Concepts Explained](#5-key-concepts-explained)
6. [Differences Between Tasks](#6-differences-between-tasks)

---

## 1. High-Level Architecture Overview

### The Big Picture

```
┌─────────────────────────────────────────────────────────────┐
│                    Training Script                          │
│              (train_loco.py, train_filter.py, etc.)         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Isaac Sim Application Launcher                 │
│              (Creates simulation environment)                │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Environment (Env)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  Scene       │  │  Observations│  │  Rewards     │     │
│  │  (Robot,     │  │  (State,     │  │  (Task-      │     │
│  │   Terrain)   │  │   Sensors)   │  │   specific)  │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              RSL-RL Runner (OnPolicyRunner)                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐     │
│  │  PPO         │  │  Actor-Critic│  │  Rollout     │     │
│  │  Algorithm   │  │  Networks    │  │  Storage    │     │
│  └──────────────┘  └──────────────┘  └──────────────┘     │
└─────────────────────────────────────────────────────────────┘
```

### Key Components:

1. **Environment (Env)**: The simulation world where the robot operates
   - Defines observations (what the robot sees)
   - Computes rewards (how well the robot performs)
   - Executes actions (what the robot does)

2. **Policy Network (Actor)**: Neural network that decides actions
   - Takes observations as input
   - Outputs actions (joint positions/torques)

3. **Value Network (Critic)**: Neural network that estimates value
   - Predicts expected future rewards
   - Used to compute advantages for training

4. **PPO Algorithm**: The learning algorithm
   - Collects experience (rollouts)
   - Updates the policy using collected data
   - Uses importance sampling and clipping

---

## 2. Understanding the Entry Scripts

### 2.1 Script Structure (train_loco.py example)

Let's break down `training/scripts/train_loco.py`:

```python
# Part 1: Setup and Configuration
- Import libraries (Isaac Lab, RSL-RL, etc.)
- Parse command-line arguments
- Set task name: "Unitree-Go2-Locomotion"
- Launch Isaac Sim application

# Part 2: Environment Creation
- Load environment configuration (env_cfg)
- Load agent/algorithm configuration (agent_cfg)
- Create gym environment: gym.make(task_name, cfg=env_cfg)
- Wrap environment for RSL-RL: RslRlVecEnvWrapper(env)

# Part 3: Runner Setup
- Create OnPolicyRunnerLoco
- Load checkpoint if resuming
- Save configuration files

# Part 4: Training
- Call runner.learn(num_learning_iterations)
- This runs the main training loop
```

### 2.2 Key Files to Study First

**Start here (in order):**

1. **`training/scripts/train_loco.py`** - Entry point for locomotion training
2. **`training/scripts/cli_args.py`** - How configurations are parsed
3. **`training/go2_lidar/go2_lidar/tasks/__init__.py`** - Task registration
4. **`training/go2_lidar/go2_lidar/tasks/go2_loco_env.py`** - Locomotion environment

---

## 3. The Training Pipeline Flow

### 3.1 Complete Training Flow

```
1. Script Initialization
   ├─ Parse arguments (num_envs, max_iterations, etc.)
   ├─ Launch Isaac Sim (creates simulation world)
   └─ Load configurations (env_cfg, agent_cfg)

2. Environment Creation
   ├─ Create scene (robot, terrain, sensors)
   ├─ Set up observation managers
   ├─ Set up reward functions
   └─ Register with gymnasium

3. Runner Initialization
   ├─ Create Actor-Critic networks
   ├─ Initialize PPO algorithm
   ├─ Set up rollout storage
   └─ Initialize optimizers

4. Training Loop (runner.learn())
   ├─ Rollout Phase (collect experience)
   │  ├─ Sample actions from policy
   │  ├─ Step environment
   │  ├─ Collect (obs, action, reward, done)
   │  └─ Repeat for num_steps_per_env
   │
   ├─ Learning Phase (update policy)
   │  ├─ Compute advantages
   │  ├─ Update policy (multiple epochs)
   │  ├─ Update value function
   │  └─ Update normalizers
   │
   └─ Logging
      ├─ Log metrics to wandb
      ├─ Save checkpoints
      └─ Print statistics
```

### 3.2 The Core Training Loop

Located in: `training/rsl_rl/rsl_rl/runners/on_policy_runner.py` (method `learn()`)

**Pseudocode:**
```python
for iteration in range(num_learning_iterations):
    # ROLLOUT PHASE: Collect experience
    for step in range(num_steps_per_env):
        actions = policy.act(observations)  # Sample actions
        obs, rewards, dones, infos = env.step(actions)  # Execute in simulation
        storage.add(obs, actions, rewards, dones)  # Store experience
    
    # LEARNING PHASE: Update policy
    advantages = compute_advantages(storage)  # Compute advantages
    for epoch in range(num_learning_epochs):
        for batch in minibatches:
            policy_loss = compute_policy_loss(batch, advantages)
            value_loss = compute_value_loss(batch)
            total_loss = policy_loss + value_loss
            optimizer.step(total_loss)
    
    # Reset storage for next iteration
    storage.reset()
```

---

## 4. Step-by-Step Study Path

### Phase 1: Understanding the Basics (Start Here!)

#### Step 1.1: Read the Entry Script
**File:** `training/scripts/train_loco.py`

**What to understand:**
- How the script initializes
- How configurations are loaded
- How the environment is created
- How the runner is set up

**Questions to answer:**
- What does `parse_env_cfg()` do?
- What does `parse_rsl_rl_cfg()` do?
- What is `RslRlVecEnvWrapper` and why is it needed?

#### Step 1.2: Understand Environment Registration
**File:** `training/go2_lidar/go2_lidar/tasks/__init__.py`

**What to understand:**
- How tasks are registered with gymnasium
- The relationship between task name and environment class
- Configuration entry points

**Questions to answer:**
- What happens when you call `gym.make("Unitree-Go2-Locomotion")`?
- How are environment and agent configs linked?

#### Step 1.3: Study the Locomotion Environment
**File:** `training/go2_lidar/go2_lidar/tasks/go2_loco_env.py`

**Key methods to understand:**
- `__init__()`: Environment setup
- `_reset_idx()`: Reset specific environments
- `_pre_physics_step()`: Apply actions before physics
- `_get_observations()`: Compute observations
- `_get_rewards()`: Compute rewards
- `_get_dones()`: Check termination conditions

**What to understand:**
- What observations does the robot receive?
- What actions does the robot take?
- How are rewards computed?
- When does an episode end?

---

### Phase 2: Understanding the Training Loop

#### Step 2.1: Study the Runner
**File:** `training/rsl_rl/rsl_rl/runners/on_policy_runner_loco.py`

**Key methods:**
- `__init__()`: Initialize networks, algorithm, storage
- `learn()`: Main training loop
- `act()`: Sample actions (delegated to algorithm)

**What to understand:**
- How the rollout phase works
- How the learning phase works
- How observations are normalized
- How checkpoints are saved

#### Step 2.2: Study the PPO Algorithm
**File:** `training/rsl_rl/rsl_rl/algorithms/ppo.py`

**Key methods:**
- `act()`: Sample actions from policy
- `process_env_step()`: Store transitions
- `update()`: Update policy and value networks
- `compute_returns_and_advantage()`: Compute advantages

**What to understand:**
- What is PPO? (Proximal Policy Optimization)
- How does policy clipping work?
- How are advantages computed?
- What is GAE (Generalized Advantage Estimation)?

**Learning Resources:**
- PPO Paper: https://arxiv.org/abs/1707.06347
- GAE Paper: https://arxiv.org/abs/1506.02438

#### Step 2.3: Study the Networks
**File:** `training/rsl_rl/rsl_rl/modules/actor_critic_recurrent.py`

**What to understand:**
- Actor network architecture (policy)
- Critic network architecture (value)
- Recurrent layers (LSTM/GRU) if used
- How actions are sampled from the policy

---

### Phase 3: Understanding Task-Specific Details

#### Step 3.1: Compare the Three Tasks

**Locomotion (`go2_loco_env.py`):**
- **Goal**: Learn to walk following velocity commands
- **Observations**: Robot state (joint positions, velocities, orientation, etc.)
- **Actions**: Joint position targets (12 joints)
- **Rewards**: Track velocity, maintain stability, avoid falling
- **Special**: Two-stage training (basic → advanced)

**Filter (`go2_filter_env.py`):**
- **Goal**: Learn to modify velocity commands to avoid obstacles
- **Observations**: Robot state + LiDAR rays
- **Actions**: Modified velocity commands (3D: vx, vy, yaw_rate)
- **Rewards**: Avoid collisions, follow commands, maintain stability
- **Special**: Uses pre-trained locomotion policy

**Navigation (`go2_nav_env.py`):**
- **Goal**: Navigate to goal while avoiding obstacles
- **Observations**: Robot state + LiDAR rays + goal information
- **Actions**: High-level velocity commands (3D)
- **Rewards**: Reach goal, avoid obstacles, efficiency
- **Special**: Uses both locomotion and filter policies

#### Step 3.2: Study Reward Functions

**Locomotion rewards** (in `go2_loco_env.py`):
- Tracking reward: How well robot follows velocity commands
- Orientation reward: Keep robot upright
- Action smoothness: Penalize jerky movements
- Contact reward: Encourage proper foot contact

**Filter rewards** (in `go2_filter_env.py`):
- Collision avoidance: Penalize getting too close to obstacles
- Command tracking: Follow high-level commands
- Stability: Maintain balance

**Navigation rewards** (in `go2_nav_env.py`):
- Goal reaching: Reward for getting closer to goal
- Collision avoidance: Penalize collisions
- Efficiency: Reward for making progress

---

### Phase 4: Advanced Topics

#### Step 4.1: Vectorized Environments
**Concept**: Multiple environments run in parallel on GPU

**Files to study:**
- `training/IsaacLab/source/isaaclab/isaaclab/envs/manager_based_rl_env.py`
- How `num_envs` parameter works
- How observations/actions are batched

#### Step 4.2: Domain Randomization
**Concept**: Randomize simulation parameters during training

**What to look for:**
- Mass randomization
- Friction randomization
- Terrain randomization
- Sensor noise

**Files:**
- `go2_loco_env.py`: `_randomize_mass()`, `_reset_physx_materials()`
- `go2_loco_env_cfg.py`: Randomization ranges

#### Step 4.3: Curriculum Learning
**Concept**: Gradually increase difficulty

**Examples:**
- Two-stage locomotion training
- Terrain difficulty progression
- Obstacle speed ranges

---

## 5. Key Concepts Explained

### 5.1 Reinforcement Learning Basics

**MDP (Markov Decision Process):**
- **State (s)**: Current situation (robot pose, joint angles, etc.)
- **Action (a)**: What the robot does (joint targets, velocities)
- **Reward (r)**: Feedback signal (how good/bad the action was)
- **Policy (π)**: Strategy for choosing actions

**Goal**: Learn a policy that maximizes cumulative reward

### 5.2 On-Policy vs Off-Policy

**This codebase uses On-Policy (PPO):**
- Collects new data with current policy
- Updates policy using only recent data
- More stable but less sample efficient

### 5.3 Actor-Critic Methods

**Actor (Policy Network):**
- Decides what action to take
- Outputs action distribution
- Trained to maximize expected reward

**Critic (Value Network):**
- Estimates value of current state
- Used to compute advantages
- Trained to predict future rewards

### 5.4 PPO Key Ideas

1. **Importance Sampling**: Reuse data from old policy
2. **Clipping**: Prevent large policy updates
3. **Multiple Epochs**: Update policy multiple times on same data
4. **GAE**: Better advantage estimation

---

## 6. Differences Between Tasks

### 6.1 Locomotion vs Filter vs Navigation

| Aspect | Locomotion | Filter | Navigation |
|--------|-----------|--------|------------|
| **Input** | Velocity commands | Velocity commands + LiDAR | Goal position + LiDAR |
| **Output** | Joint positions | Modified velocity | Velocity commands |
| **Policy Type** | Low-level control | Mid-level safety | High-level planning |
| **Uses Other Policies** | No | Uses locomotion | Uses locomotion + filter |
| **Observations** | Proprioception only | Proprioception + LiDAR | Proprioception + LiDAR + Goal |
| **Training Stages** | 2 stages | 2 stages | 1 stage |

### 6.2 Why This Modular Design?

1. **Locomotion**: Foundation - robot must walk
2. **Filter**: Safety layer - avoid obstacles
3. **Navigation**: High-level - reach goals

Each module can be trained independently and composed together!

---

## 7. Practical Exercises

### Exercise 1: Trace a Single Training Step
1. Start from `train_loco.py`
2. Follow the code until `runner.learn()` is called
3. Trace through one iteration of the training loop
4. Understand what happens at each step

### Exercise 2: Modify a Reward Function
1. Find reward computation in `go2_loco_env.py`
2. Add a new reward term
3. Understand how it affects training

### Exercise 3: Change Observation Space
1. Find `_get_observations()` in an environment
2. Add a new observation
3. Update network input size in config

### Exercise 4: Compare Configurations
1. Compare `go2_loco_ppo_cfg.py`, `go2_filter_ppo_cfg.py`, `go2_nav_ppo_cfg.py`
2. Understand why they differ
3. What hyperparameters matter most?

---

## 8. Recommended Reading Order

### For Complete Beginners:
1. Read this guide
2. Study `train_loco.py` line by line
3. Study `go2_loco_env.py` methods one by one
4. Study `on_policy_runner_loco.py` training loop
5. Study `ppo.py` algorithm

### For Those Familiar with RL:
1. Skip to Phase 2 (Training Loop)
2. Focus on PPO implementation
3. Study environment implementations
4. Understand the modular design

### For Those Familiar with Robotics:
1. Focus on environment implementations
2. Understand observation/action spaces
3. Study reward functions
4. Understand how policies compose

---

## 9. Key Files Reference

### Entry Points:
- `training/scripts/train_loco.py` - Locomotion training
- `training/scripts/train_filter.py` - Filter training
- `training/scripts/train_nav.py` - Navigation training

### Environments:
- `training/go2_lidar/go2_lidar/tasks/go2_loco_env.py` - Locomotion env
- `training/go2_lidar/go2_lidar/tasks/go2_filter_env.py` - Filter env
- `training/go2_lidar/go2_lidar/tasks/go2_nav_env.py` - Navigation env

### Configurations:
- `training/go2_lidar/go2_lidar/tasks/go2_loco_env_cfg.py` - Env config
- `training/go2_lidar/go2_lidar/tasks/go2_loco_ppo_cfg.py` - Agent config
- (Similar for filter and nav)

### Training Infrastructure:
- `training/rsl_rl/rsl_rl/runners/on_policy_runner_loco.py` - Locomotion runner
- `training/rsl_rl/rsl_rl/runners/on_policy_runner.py` - Standard runner
- `training/rsl_rl/rsl_rl/algorithms/ppo.py` - PPO algorithm
- `training/rsl_rl/rsl_rl/modules/actor_critic_recurrent.py` - Networks

---

## 10. Debugging Tips

### Common Issues:

1. **Out of Memory**: Reduce `num_envs`
2. **Training Not Converging**: Check reward scales, learning rate
3. **Policy Not Learning**: Check observations, rewards, network size
4. **Simulation Crashes**: Check action limits, physics settings

### Useful Debugging Tools:

1. **Wandb Logs**: Visualize training curves
2. **Checkpoint Inspection**: Load and test policies
3. **Play Scripts**: Test trained policies
4. **Print Statements**: Add debug prints in environments

---

## 11. Next Steps

After understanding this codebase:

1. **Experiment**: Modify rewards, observations, network architectures
2. **Extend**: Add new tasks or capabilities
3. **Deploy**: Understand deployment code in `deployment/` folder
4. **Read Papers**: 
   - REASAN paper (cited in README)
   - PPO paper
   - Related legged robot RL papers

---

## 12. Questions to Guide Your Study

As you read the code, ask yourself:

1. **What observations does the robot receive?**
2. **What actions can the robot take?**
3. **How are rewards computed?**
4. **When does an episode end?**
5. **How does the policy network work?**
6. **How does PPO update the policy?**
7. **Why use vectorized environments?**
8. **How does domain randomization help?**
9. **Why modular design (loco → filter → nav)?**
10. **How do the three policies work together?**

---

Good luck with your learning journey! Start with Phase 1 and work through systematically. Don't hesitate to experiment and modify the code to deepen your understanding.

