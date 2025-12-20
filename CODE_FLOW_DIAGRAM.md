# Code Flow Diagram: REASAN Training

This document shows the exact sequence of function calls during training, making it easier to trace through the codebase.

## 1. Script Initialization Flow

```
train_loco.py (or train_filter.py, train_nav.py)
│
├─> Parse command-line arguments
│   └─> argparse.ArgumentParser
│
├─> Set task_name
│   ├─> "Unitree-Go2-Locomotion" (loco)
│   ├─> "Unitree-Go2-Filter" (filter)
│   └─> "Unitree-Go2-Navigation" (nav)
│
├─> Launch Isaac Sim
│   └─> AppLauncher(args_cli)
│       └─> Creates simulation_app
│
└─> main()
    │
    ├─> Load environment config
    │   └─> parse_env_cfg(task_name, ...)
    │       └─> Loads from registry: go2_loco_env_cfg.py
    │
    ├─> Load agent config
    │   └─> parse_rsl_rl_cfg(task_name, args_cli)
    │       └─> Loads from registry: go2_loco_ppo_cfg.py
    │
    ├─> Create environment
    │   └─> gym.make(task_name, cfg=env_cfg)
    │       └─> Calls registered entry_point
    │           └─> Go2LocoEnv.__init__(cfg)
    │               ├─> DirectRLEnv.__init__(cfg)
    │               ├─> Setup robot, terrain, sensors
    │               ├─> Initialize buffers
    │               └─> Reset environment
    │
    ├─> Wrap environment
    │   └─> RslRlVecEnvWrapper(env)
    │       └─> Wraps for RSL-RL compatibility
    │
    ├─> Create runner
    │   └─> OnPolicyRunnerLoco(env, agent_cfg, ...)
    │       └─> __init__()
    │           ├─> Get observations to determine dimensions
    │           ├─> Create Actor-Critic network
    │           ├─> Create PPO algorithm
    │           ├─> Create rollout storage
    │           └─> Initialize optimizers
    │
    └─> Start training
        └─> runner.learn(num_learning_iterations)
```

## 2. Environment Creation Flow (Go2LocoEnv.__init__)

```
Go2LocoEnv.__init__(cfg)
│
├─> DirectRLEnv.__init__(cfg)
│   ├─> Create simulation
│   ├─> Create scene (robot, terrain)
│   ├─> Create sensors
│   └─> Initialize buffers
│
├─> Setup robot-specific things
│   ├─> Get body/joint IDs
│   ├─> Initialize action buffers
│   ├─> Initialize command buffers
│   └─> Setup reward tracking
│
└─> Reset all environments
    └─> reset()
        └─> _reset_idx(all_env_ids)
```

## 3. Runner Initialization Flow (OnPolicyRunnerLoco.__init__)

```
OnPolicyRunnerLoco.__init__(env, train_cfg, ...)
│
├─> Get observation dimensions
│   └─> obs, extras = env.get_observations()
│       └─> Returns: (num_envs, obs_dim)
│
├─> Create policy network
│   └─> ActorCriticRecurrent(num_obs, num_privileged_obs, num_actions, ...)
│       ├─> Actor network (MLP + LSTM + MLP)
│       └─> Critic network (MLP + LSTM + MLP)
│
├─> Create PPO algorithm
│   └─> PPO(policy, ...)
│       ├─> Store policy reference
│       ├─> Create optimizer
│       └─> Initialize rollout storage (will be created later)
│
├─> Create observation normalizer
│   └─> EmpiricalNormalization(obs_dim)
│
└─> Load checkpoint if resuming
    └─> self.load(resume_path)
```

## 4. Main Training Loop Flow (runner.learn())

```
learn(num_learning_iterations)
│
├─> Initialize wandb
│   └─> wandb.init(project=wandb_project)
│
├─> Get initial observations
│   └─> obs, extras = env.get_observations()
│
├─> Initialize tracking variables
│   ├─> Episode info buffers
│   ├─> Reward buffers
│   └─> Episode length buffers
│
└─> for iteration in range(num_learning_iterations):
    │
    ├─> ROLLOUT PHASE (collect experience)
    │   └─> with torch.inference_mode():
    │       └─> for step in range(num_steps_per_env):
    │           │
    │           ├─> Normalize observations
    │           │   └─> obs = obs_normalizer(obs)
    │           │
    │           ├─> Sample actions
    │           │   └─> actions = alg.act(obs, privileged_obs)
    │           │       └─> PPO.act()
    │           │           ├─> policy.act_inference(obs)
    │           │           │   └─> ActorCriticRecurrent.act_inference()
    │           │           │       ├─> Forward through actor network
    │           │           │       ├─> Sample from action distribution
    │           │           │       └─> Return actions
    │           │           └─> Store in transition buffer
    │           │
    │           ├─> Step environment
    │           │   └─> obs, rewards, dones, infos = env.step(actions)
    │           │       └─> DirectRLEnv.step()
    │           │           ├─> _pre_physics_step(actions)
    │           │           │   └─> Go2LocoEnv._pre_physics_step()
    │           │           │       ├─> Scale actions
    │           │           │       ├─> Apply to robot
    │           │           │       └─> Update internal state
    │           │           │
    │           │           ├─> Physics simulation loop
    │           │           │   └─> for _ in range(decimation):
    │           │           │       ├─> sim.step()
    │           │           │       └─> scene.update()
    │           │           │
    │           │           ├─> Compute rewards
    │           │           │   └─> rewards = _get_rewards()
    │           │           │       └─> Go2LocoEnv._get_rewards()
    │           │           │           ├─> Compute individual reward terms
    │           │           │           └─> Sum weighted rewards
    │           │           │
    │           │           ├─> Check terminations
    │           │           │   └─> dones = _get_dones()
    │           │           │       └─> Go2LocoEnv._get_dones()
    │           │           │
    │           │           ├─> Reset terminated environments
    │           │           │   └─> if any(dones):
    │           │           │       └─> _reset_idx(terminated_env_ids)
    │           │           │           └─> Go2LocoEnv._reset_idx()
    │           │           │               ├─> Reset robot pose
    │           │           │               ├─> Reset commands
    │           │           │               └─> Reset buffers
    │           │           │
    │           │           └─> Get new observations
    │           │               └─> obs = _get_observations()
    │           │                   └─> Go2LocoEnv._get_observations()
    │           │                       ├─> Get robot state
    │           │                       ├─> Get sensor data
    │           │                       └─> Concatenate into vector
    │           │
    │           ├─> Move to device
    │           │   └─> obs, rewards, dones = ...to(device)
    │           │
    │           ├─> Process step
    │           │   └─> alg.process_env_step(rewards, dones, infos)
    │           │       └─> PPO.process_env_step()
    │           │           ├─> Store in transition buffer
    │           │           └─> Update episode statistics
    │           │
    │           └─> Update tracking
    │               ├─> cur_reward_sum += rewards
    │               └─> cur_episode_length += 1
    │
    ├─> LEARNING PHASE (update policy)
    │   └─> alg.update()
    │       └─> PPO.update()
    │           │
    │           ├─> Compute returns and advantages
    │           │   └─> compute_returns_and_advantage()
    │           │       ├─> Compute value predictions
    │           │       ├─> Compute returns (discounted rewards)
    │           │       └─> Compute advantages (GAE)
    │           │
    │           ├─> Normalize advantages
    │           │   └─> advantages = (advantages - mean) / std
    │           │
    │           └─> for epoch in range(num_learning_epochs):
    │               └─> for batch in minibatches:
    │                   │
    │                   ├─> Forward pass
    │                   │   └─> policy.evaluate(obs, actions)
    │                   │       ├─> Actor forward (get action log_probs)
    │                   │       └─> Critic forward (get values)
    │                   │
    │                   ├─> Compute policy loss
    │                   │   └─> compute_policy_loss()
    │                   │       ├─> Ratio = new_log_prob / old_log_prob
    │                   │       ├─> Clipped ratio
    │                   │       └─> Loss = -min(ratio * advantage, clip(ratio) * advantage)
    │                   │
    │                   ├─> Compute value loss
    │                   │   └─> compute_value_loss()
    │                   │       ├─> Value prediction error
    │                   │       └─> Clipped value loss
    │                   │
    │                   ├─> Compute entropy bonus
    │                   │   └─> entropy = -sum(prob * log(prob))
    │                   │
    │                   ├─> Total loss
    │                   │   └─> loss = policy_loss + value_loss_coef * value_loss - entropy_coef * entropy
    │                   │
    │                   └─> Backward and update
    │                       ├─> loss.backward()
    │                       ├─> clip_grad_norm()
    │                       └─> optimizer.step()
    │
    ├─> Update normalizers
    │   ├─> obs_normalizer.update(obs)
    │   └─> Update running statistics
    │
    ├─> Logging
    │   ├─> Log to wandb
    │   │   ├─> Episode rewards
    │   │   ├─> Episode lengths
    │   │   ├─> Policy loss
    │   │   ├─> Value loss
    │   │   └─> Learning rate
    │   │
    │   └─> Print statistics
    │       └─> Print every N iterations
    │
    └─> Save checkpoint
        └─> if iteration % checkpoint_interval == 0:
            └─> torch.save(...)
```

## 5. Environment Step Flow (Detailed)

```
env.step(actions)
│
├─> Convert actions to device
│   └─> actions = actions.to(device)
│
├─> PRE-PHYSICS STEP
│   └─> _pre_physics_step(actions)
│       └─> Go2LocoEnv._pre_physics_step()
│           ├─> Scale actions
│           │   └─> actions_scaled = actions * action_scale
│           │
│           ├─> Apply to robot
│           │   └─> _apply_action()
│           │       └─> robot.set_joint_position_target(targets)
│           │
│           └─> Update internal state
│               ├─> Store previous actions
│               └─> Update command tracking
│
├─> PHYSICS SIMULATION
│   └─> for _ in range(decimation):
│       ├─> sim.step()
│       │   └─> Advances physics by physics_dt
│       │
│       └─> scene.update(dt=physics_dt)
│           └─> Updates robot state, sensors, etc.
│
├─> POST-PHYSICS STEP
│   │
│   ├─> Compute rewards
│   │   └─> rewards = _get_rewards()
│   │       └─> Go2LocoEnv._get_rewards()
│   │           ├─> tracking_lin_vel = exp(-error^2)
│   │           ├─> tracking_ang_vel = exp(-error^2)
│   │           ├─> orientation = exp(-tilt^2)
│   │           ├─> action_rate = -|action - prev_action|
│   │           └─> total = sum(weighted_terms)
│   │
│   ├─> Check terminations
│   │   └─> dones = _get_dones()
│   │       └─> Go2LocoEnv._get_dones()
│   │           ├─> Check if robot fell
│   │           ├─> Check if episode timeout
│   │           └─> Return done flags
│   │
│   ├─> Reset terminated environments
│   │   └─> if any(dones):
│   │       └─> _reset_idx(terminated_env_ids)
│   │           └─> Go2LocoEnv._reset_idx()
│   │               ├─> Reset robot pose/velocity
│   │               ├─> Sample new commands
│   │               ├─> Reset terrain if needed
│   │               └─> Reset episode counters
│   │
│   └─> Get observations
│       └─> obs = _get_observations()
│           └─> Go2LocoEnv._get_observations()
│               ├─> Robot state
│               │   ├─> Joint positions
│               │   ├─> Joint velocities
│               │   ├─> Base orientation
│               │   └─> Base velocity
│               │
│               ├─> Commands
│               │   ├─> Desired linear velocity
│               │   └─> Desired angular velocity
│               │
│               ├─> Previous actions
│               │   └─> For smoothness
│               │
│               └─> Concatenate
│                   └─> torch.cat([...], dim=-1)
│
└─> Return
    └─> (obs, rewards, dones, infos)
```

## 6. PPO Update Flow (Detailed)

```
PPO.update()
│
├─> Get data from storage
│   └─> obs, actions, rewards, dones, old_log_probs, values = storage.get_all()
│
├─> Compute returns and advantages
│   └─> compute_returns_and_advantage()
│       │
│       ├─> Compute value predictions for last step
│       │   └─> last_values = critic(last_obs)
│       │
│       ├─> Compute returns (discounted cumulative rewards)
│       │   └─> returns = rewards + gamma * next_values * (1 - dones)
│       │       └─> Backward pass through time
│       │
│       └─> Compute advantages (GAE)
│           └─> advantages = returns - values
│               └─> Then apply GAE lambda smoothing
│
├─> Normalize advantages
│   └─> advantages = (advantages - mean) / (std + eps)
│
└─> for epoch in range(num_learning_epochs):
    └─> Shuffle data into minibatches
        └─> for batch in minibatches:
            │
            ├─> Forward pass
            │   └─> policy.evaluate(batch_obs, batch_actions)
            │       ├─> Actor forward
            │       │   ├─> MLP(obs)
            │       │   ├─> LSTM(hidden_state)
            │       │   ├─> MLP(lstm_out)
            │       │   ├─> Action distribution (mean, std)
            │       │   └─> log_prob = dist.log_prob(actions)
            │       │
            │       └─> Critic forward
            │           ├─> MLP(obs or privileged_obs)
            │           ├─> LSTM(hidden_state)
            │           ├─> MLP(lstm_out)
            │           └─> value = output
            │
            ├─> Compute policy loss
            │   └─> compute_policy_loss()
            │       ├─> ratio = exp(new_log_prob - old_log_prob)
            │       ├─> surr1 = ratio * advantages
            │       ├─> surr2 = clip(ratio, 1-eps, 1+eps) * advantages
            │       └─> loss = -min(surr1, surr2).mean()
            │
            ├─> Compute value loss
            │   └─> compute_value_loss()
            │       ├─> value_pred_clipped = old_value + clip(value - old_value, -eps, eps)
            │       ├─> value_loss1 = (value - returns)^2
            │       ├─> value_loss2 = (value_pred_clipped - returns)^2
            │       └─> loss = max(value_loss1, value_loss2).mean()
            │
            ├─> Compute entropy
            │   └─> entropy = -sum(prob * log(prob))
            │
            ├─> Total loss
            │   └─> loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            │
            └─> Update
                ├─> optimizer.zero_grad()
                ├─> loss.backward()
                ├─> clip_grad_norm(max_norm)
                └─> optimizer.step()
```

## 7. Key Data Structures

### Transition Buffer (RolloutStorage)
```
Storage contains:
- observations: (num_steps, num_envs, obs_dim)
- actions: (num_steps, num_envs, action_dim)
- rewards: (num_steps, num_envs)
- dones: (num_steps, num_envs)
- old_log_probs: (num_steps, num_envs)
- values: (num_steps, num_envs)
- advantages: (num_steps, num_envs) [computed during update]
- returns: (num_steps, num_envs) [computed during update]
```

### Environment State
```
Each environment has:
- Robot state: pose, velocity, joint positions/velocities
- Command: desired velocity (vx, vy, yaw_rate)
- Episode counter: current step in episode
- Reset buffer: which envs need reset
- Reward buffer: current step rewards
- Done buffer: termination flags
```

## 8. Function Call Hierarchy (Summary)

```
train_loco.py
  └─> main()
      ├─> parse_env_cfg() → Go2LocoEnvCfg
      ├─> parse_rsl_rl_cfg() → Go2LocoPPOCfg
      ├─> gym.make() → Go2LocoEnv.__init__()
      │   └─> DirectRLEnv.__init__()
      │       └─> reset() → _reset_idx()
      ├─> RslRlVecEnvWrapper()
      ├─> OnPolicyRunnerLoco.__init__()
      │   ├─> ActorCriticRecurrent()
      │   └─> PPO()
      └─> runner.learn()
          └─> for iteration:
              ├─> ROLLOUT:
              │   └─> for step:
              │       ├─> alg.act() → policy.act_inference()
              │       └─> env.step() → _pre_physics_step() → _get_rewards() → _get_observations()
              └─> LEARNING:
                  └─> alg.update() → compute_returns_and_advantage() → policy.evaluate() → optimizer.step()
```

This flow diagram should help you trace through the codebase systematically!

