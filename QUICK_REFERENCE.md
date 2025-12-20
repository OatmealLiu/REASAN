# Quick Reference: REASAN Codebase

## File Navigation Cheat Sheet

### Entry Scripts → Core Components

```
train_loco.py
    ├─> cli_args.py (config parsing)
    ├─> go2_loco_env.py (environment)
    │   ├─> go2_loco_env_cfg.py (env config)
    │   └─> go2_loco_ppo_cfg.py (agent config)
    └─> on_policy_runner_loco.py (training loop)
        ├─> ppo.py (algorithm)
        ├─> actor_critic_recurrent.py (networks)
        └─> rollout_storage.py (experience buffer)
```

### Key Methods to Understand

#### Environment (`go2_loco_env.py`):
- `__init__()` - Setup robot, terrain, sensors
- `_reset_idx()` - Reset specific environments
- `_pre_physics_step()` - Apply actions before physics
- `_get_observations()` - Compute what robot sees
- `_get_rewards()` - Compute reward signal
- `_get_dones()` - Check if episode ended

#### Runner (`on_policy_runner_loco.py`):
- `__init__()` - Create networks, algorithm, storage
- `learn()` - Main training loop
  - Rollout phase: Collect experience
  - Learning phase: Update policy

#### PPO (`ppo.py`):
- `act()` - Sample actions from policy
- `process_env_step()` - Store transitions
- `update()` - Update policy using collected data
- `compute_returns_and_advantage()` - Compute advantages

## Training Flow (One Iteration)

```
1. Rollout Phase (collect experience):
   for step in range(num_steps_per_env):
       actions = policy.act(observations)
       obs, rewards, dones = env.step(actions)
       storage.add(obs, actions, rewards, dones)

2. Learning Phase (update policy):
   advantages = compute_advantages(storage)
   for epoch in range(num_learning_epochs):
       for batch in minibatches:
           loss = policy_loss + value_loss
           optimizer.step(loss)
```

## Key Configurations

### Environment Config (`*_env_cfg.py`):
- `num_envs`: Number of parallel environments
- `episode_length_s`: Episode duration
- `decimation`: Physics steps per env step
- `sim.dt`: Physics timestep

### Agent Config (`*_ppo_cfg.py`):
- `algorithm.num_learning_epochs`: PPO update epochs
- `algorithm.num_mini_batches`: Minibatch count
- `algorithm.clip_param`: PPO clipping parameter
- `algorithm.gamma`: Discount factor
- `algorithm.lam`: GAE lambda
- `policy.actor_hidden_dims`: Actor network size
- `policy.critic_hidden_dims`: Critic network size

## Observation Spaces

### Locomotion:
- Robot state: joint positions, velocities, orientation
- Command: desired velocity (vx, vy, yaw_rate)
- Previous actions: for smoothness

### Filter:
- Robot state: same as locomotion
- LiDAR rays: distance measurements
- Command: high-level velocity command
- Previous filter actions: for smoothness

### Navigation:
- Robot state: same as locomotion
- LiDAR rays: distance measurements
- Goal information: relative goal position/direction
- Command: (not used, policy outputs directly)

## Action Spaces

### Locomotion:
- 12 joint position targets (one per joint)

### Filter:
- 3D velocity command modification (vx, vy, yaw_rate)

### Navigation:
- 3D velocity command (vx, vy, yaw_rate)

## Reward Components

### Locomotion:
- `tracking_lin_vel`: Follow linear velocity command
- `tracking_ang_vel`: Follow angular velocity command
- `lin_vel_z`: Penalize vertical velocity
- `ang_vel_xy`: Penalize unwanted rotation
- `orientation`: Keep robot upright
- `torques`: Penalize large torques
- `action_rate`: Penalize action changes
- `action_smoothness`: Penalize jerky movements
- `dof_acc`: Penalize joint accelerations
- `collision`: Penalize unwanted contacts
- `stall_torque`: Penalize stalling

### Filter:
- `collision_avoidance`: Avoid getting too close
- `command_tracking`: Follow high-level commands
- `stability`: Maintain balance

### Navigation:
- `goal_reaching`: Get closer to goal
- `collision`: Avoid collisions
- `progress`: Reward forward progress

## Common Patterns

### 1. Environment Reset:
```python
def _reset_idx(self, env_ids):
    # Reset robot pose
    # Reset commands
    # Reset episode counters
    # Reset buffers
```

### 2. Action Application:
```python
def _pre_physics_step(self, actions):
    # Scale actions
    # Apply to robot
    # Update internal state
```

### 3. Observation Computation:
```python
def _get_observations(self):
    # Get robot state
    # Process sensors
    # Concatenate into vector
    return observations
```

### 4. Reward Computation:
```python
def _get_rewards(self):
    # Compute individual reward terms
    # Weight and sum
    return total_reward
```

## Debugging Checklist

- [ ] Check observation shapes match network input
- [ ] Check action shapes match network output
- [ ] Verify rewards are reasonable (not too large/small)
- [ ] Check if episodes terminate correctly
- [ ] Verify action scaling is correct
- [ ] Check for NaN/Inf in observations/rewards
- [ ] Verify checkpoint loading works
- [ ] Check wandb logs for training progress

## Useful Commands

### Training:
```bash
# Locomotion stage 1
python scripts/train_loco.py --run_name loco_new --num_envs 8 --max_iterations 5000 --wandb_proj go2_loco

# Locomotion stage 2
python scripts/train_loco.py --run_name loco_new --resume --load_run loco_new --num_envs 8 --max_iterations 5000 --wandb_proj go2_loco --second_stage
```

### Playing (testing):
```bash
# Test locomotion
python scripts/play_loco.py --load_run loco_1

# Test filter
python scripts/play_filter.py --load_run filter_1 --with_dyn_obst --confirm

# Test navigation
python scripts/play_nav.py --load_run nav_1 --confirm --with_dyn_obst
```

## Key Differences: Locomotion vs Filter vs Navigation

| Feature | Locomotion | Filter | Navigation |
|---------|-----------|--------|------------|
| **Runner** | `OnPolicyRunnerLoco` | `OnPolicyRunner` | `OnPolicyRunner` |
| **Uses Ray** | No | Yes (`use_ray=True`) | Yes (`use_ray=True`) |
| **Privileged Obs** | Separate critic obs | Same as actor | Same as actor |
| **Policy Composition** | Standalone | Uses loco policy | Uses loco + filter |

## Network Architecture Patterns

### Actor Network:
```
Input (obs) → MLP → LSTM → MLP → Output (action mean)
                              └→ Output (action std)
```

### Critic Network:
```
Input (obs or privileged_obs) → MLP → LSTM → MLP → Output (value)
```

## Important Constants

- Physics timestep: Usually `1/200` seconds
- Environment timestep: `physics_dt * decimation`
- Episode length: Usually 9-20 seconds
- Action scale: Usually `0.25` for locomotion

## Study Order (Quick Path)

1. `train_loco.py` - Entry point
2. `go2_loco_env.py` - Environment
3. `on_policy_runner_loco.py` - Training loop
4. `ppo.py` - Algorithm
5. Compare with `go2_filter_env.py` and `go2_nav_env.py`

## Questions to Answer While Reading

1. What is the observation space dimension?
2. What is the action space dimension?
3. How many reward terms are there?
4. What triggers episode termination?
5. How are actions scaled/applied?
6. What network architecture is used?
7. What PPO hyperparameters are used?
8. How does domain randomization work?

