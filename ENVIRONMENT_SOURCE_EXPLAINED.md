# Where Does the Training Environment Come From?

This document explains the complete path from `gym.make()` to the actual environment instance.

## The Complete Flow

```
train_loco.py (line 113)
    │
    │ task_name = "Unitree-Go2-Locomotion"
    │ env_cfg = <configuration object>
    │
    └─> gym.make("Unitree-Go2-Locomotion", cfg=env_cfg, render_mode=None)
        │
        │ gym.make() looks up the task in gymnasium's registry
        │ The registry was populated when we imported go2_lidar.tasks
        │
        └─> Finds registration in: go2_lidar/go2_lidar/tasks/__init__.py
            │
            │ gym.register(
            │     id="Unitree-Go2-Locomotion",
            │     entry_point="go2_lidar.tasks.go2_loco_env:Go2LocoEnv",
            │     ...
            │ )
            │
            └─> Calls entry_point: Go2LocoEnv.__init__(cfg, render_mode, **kwargs)
                │
                │ This is the actual environment class:
                │ File: go2_lidar/go2_lidar/tasks/go2_loco_env.py
                │ Class: Go2LocoEnv(DirectRLEnv)
                │
                └─> Go2LocoEnv.__init__() creates the environment instance
```

## Step-by-Step Breakdown

### Step 1: Import Triggers Registration

**File:** `training/scripts/train_loco.py` (line 53)

```python
import go2_lidar.tasks  # noqa: F401
```

This import statement executes `go2_lidar/tasks/__init__.py`, which contains:

```python
gym.register(
    id="Unitree-Go2-Locomotion",
    entry_point=f"{__name__}.go2_loco_env:Go2LocoEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_loco_env:Go2LocoEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.go2_loco_ppo_cfg:Go2LocoPPOCfg",
    },
)
```

**What happens:**
- The `gym.register()` call adds "Unitree-Go2-Locomotion" to gymnasium's environment registry
- It stores the entry point: `"go2_lidar.tasks.go2_loco_env:Go2LocoEnv"`
- This tells gymnasium: "When someone asks for 'Unitree-Go2-Locomotion', create an instance of `Go2LocoEnv`"

### Step 2: Task Name is Set

**File:** `training/scripts/train_loco.py` (line 36)

```python
task_name = "Unitree-Go2-Locomotion"
```

This string matches the `id` in the `gym.register()` call.

### Step 3: Environment Configuration is Loaded

**File:** `training/scripts/train_loco.py` (line 69-74)

```python
env_cfg = parse_env_cfg(
    task_name,
    device=args_cli.device,
    num_envs=args_cli.num_envs,
    use_fabric=True,
)
```

**What `parse_env_cfg()` does:**
1. Looks up "Unitree-Go2-Locomotion" in gym registry
2. Gets `env_cfg_entry_point` from the registration kwargs
3. Resolves: `"go2_lidar.tasks.go2_loco_env:Go2LocoEnvCfg"`
4. Imports and instantiates `Go2LocoEnvCfg` class
5. Overrides with CLI arguments (num_envs, device, etc.)
6. Returns the configuration object

**The config class is defined in:**
- File: `go2_lidar/go2_lidar/tasks/go2_loco_env_cfg.py`
- Class: `Go2LocoEnvCfg(DirectRLEnvCfg)`

### Step 4: gym.make() Creates the Environment

**File:** `training/scripts/train_loco.py` (line 113)

```python
env = gym.make(task_name, cfg=env_cfg, render_mode=None)
```

**What `gym.make()` does:**
1. Looks up "Unitree-Go2-Locomotion" in gymnasium's registry
2. Gets the `entry_point`: `"go2_lidar.tasks.go2_loco_env:Go2LocoEnv"`
3. Resolves the entry point:
   - Module: `go2_lidar.tasks.go2_loco_env`
   - Class: `Go2LocoEnv`
4. Imports: `from go2_lidar.tasks.go2_loco_env import Go2LocoEnv`
5. Instantiates: `Go2LocoEnv(cfg=env_cfg, render_mode=None, **kwargs)`

### Step 5: Environment Class Initialization

**File:** `go2_lidar/go2_lidar/tasks/go2_loco_env.py` (line 30-34)

```python
class Go2LocoEnv(DirectRLEnv):
    cfg: Go2LocoEnvCfg

    def __init__(self, cfg: Go2LocoEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        # ... environment-specific setup ...
```

**What happens in `Go2LocoEnv.__init__()`:**
1. Calls `DirectRLEnv.__init__()` (parent class)
   - Creates simulation
   - Creates scene (robot, terrain, sensors)
   - Initializes buffers
2. Sets up locomotion-specific components:
   - Robot body/joint IDs
   - Action buffers
   - Command buffers
   - Reward tracking
3. Resets all environments

## File Locations Summary

| Component | File Path | Class/Function |
|-----------|-----------|---------------|
| **Registration** | `go2_lidar/go2_lidar/tasks/__init__.py` | `gym.register()` |
| **Environment Class** | `go2_lidar/go2_lidar/tasks/go2_loco_env.py` | `Go2LocoEnv` |
| **Environment Config** | `go2_lidar/go2_lidar/tasks/go2_loco_env_cfg.py` | `Go2LocoEnvCfg` |
| **Agent Config** | `go2_lidar/go2_lidar/tasks/go2_loco_ppo_cfg.py` | `Go2LocoPPOCfg` |
| **Base Class** | `IsaacLab/.../direct_rl_env.py` | `DirectRLEnv` |
| **Usage** | `training/scripts/train_loco.py` | `gym.make()` |

## Understanding the Entry Point String

The entry point format is: `"module.path:ClassName"`

For locomotion:
```python
entry_point="go2_lidar.tasks.go2_loco_env:Go2LocoEnv"
```

This means:
- **Module**: `go2_lidar.tasks.go2_loco_env` (the file `go2_loco_env.py`)
- **Class**: `Go2LocoEnv` (the class defined in that file)

When gymnasium resolves this, it does:
```python
from go2_lidar.tasks.go2_loco_env import Go2LocoEnv
env = Go2LocoEnv(cfg=env_cfg, render_mode=None)
```

## Why This Design?

1. **Separation of Concerns**: 
   - Registration is separate from implementation
   - Configs are separate from environment logic

2. **Flexibility**:
   - Can register multiple versions of same task
   - Can swap implementations easily

3. **Standard Interface**:
   - Uses gymnasium's standard registration system
   - Compatible with other RL libraries

4. **Lazy Loading**:
   - Environment class only imported when needed
   - Config classes loaded separately

## Similar Pattern for Other Tasks

The same pattern applies to filter and navigation:

**Filter:**
- Registration: `go2_lidar/tasks/__init__.py` (line 23-31)
- Environment: `go2_lidar/tasks/go2_filter_env.py` → `Go2FilterEnv`
- Config: `go2_lidar/tasks/go2_filter_env_cfg.py` → `Go2FilterEnvCfg`

**Navigation:**
- Registration: `go2_lidar/tasks/__init__.py` (line 33-41)
- Environment: `go2_lidar/tasks/go2_nav_env.py` → `Go2NavEnv`
- Config: `go2_lidar/tasks/go2_nav_env_cfg.py` → `Go2NavEnvCfg`

## Key Takeaway

When you call:
```python
env = gym.make("Unitree-Go2-Locomotion", cfg=env_cfg)
```

You're actually calling:
```python
from go2_lidar.tasks.go2_loco_env import Go2LocoEnv
env = Go2LocoEnv(cfg=env_cfg)
```

The `gym.register()` call in `__init__.py` creates the mapping between the task name string and the actual Python class!

