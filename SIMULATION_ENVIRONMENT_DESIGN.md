# Where the Simulation Environment is Designed and Defined

This document explains where the actual physics simulation world is created - the robot model, terrain, sensors, and physics engine.

## Overview: The Complete Simulation Stack

```
┌─────────────────────────────────────────────────────────────┐
│  Config Files (WHAT to create)                             │
│  - Go2LocoEnvCfg: Defines robot, terrain, sensors          │
│  - UNITREE_GO2_CFG: Robot model definition                 │
│  - GO2_LOCO_TERRAIN_CFG: Terrain generation parameters     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  DirectRLEnv Base Class (HOW to create simulation)         │
│  - Creates SimulationContext (physics engine)               │
│  - Creates InteractiveScene (container for objects)        │
│  - Sets up physics parameters                               │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Go2LocoEnv._setup_scene() (WHEN to create objects)        │
│  - Instantiates robot from config                           │
│  - Generates terrain from config                            │
│  - Creates sensors from config                              │
│  - Adds everything to scene                                 │
└─────────────────────────────────────────────────────────────┘
```

## 1. Physics Simulation Engine

### Created By: `DirectRLEnv` Base Class

**File:** `training/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py` (line 99-100)

```python
# Create simulation context (physics engine)
if SimulationContext.instance() is None:
    self.sim: SimulationContext = SimulationContext(self.cfg.sim)
```

**What it does:**
- Creates the **PhysX physics engine** instance
- Sets up physics parameters from `cfg.sim`:
  - `dt`: Physics timestep (1/200 seconds)
  - `gravity`: Gravity vector
  - `physics_material`: Friction, restitution
  - `physx`: GPU settings

**Configuration Source:**
- `go2_loco_env_cfg.py` (line 134-144):
  ```python
  sim: SimulationCfg = SimulationCfg(
      dt=1 / 200,  # 200 Hz physics
      physics_material=sim_utils.RigidBodyMaterialCfg(...),
      physx=PhysxCfg(gpu_max_rigid_patch_count=4096 * 4096),
  )
  ```

## 2. Robot Model

### Defined In: Isaac Lab Assets

**File:** `training/IsaacLab/source/isaaclab_assets/isaaclab_assets/robots/unitree.py` (line 138-178)

```python
UNITREE_GO2_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/Go2/go2.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(...),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(...),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),  # Initial position
        joint_pos={...},      # Initial joint positions
        joint_vel={".*": 0.0}, # Initial joint velocities
    ),
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=23.5,
            stiffness=25.0,
            damping=0.5,
            ...
        ),
    },
)
```

**What this defines:**
- **Robot geometry**: Loaded from USD file (`go2.usd`)
  - This is a 3D model file containing the robot's mesh, joints, etc.
  - Located in Isaac Lab's asset directory
- **Actuators**: DC motor models for the 12 leg joints
- **Initial state**: Starting pose and joint positions
- **Physics properties**: Mass, inertia, collision shapes

**Used In:**
- `go2_loco_env_cfg.py` (line 185):
  ```python
  robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="/World/envs/env_.*/Robot")
  ```
  - Takes the base config and sets where to spawn the robot in the scene

**Instantiated In:**
- `go2_loco_env.py` (line 125-126):
  ```python
  def _setup_scene(self):
      self._robot = Articulation(self.cfg.robot)  # Creates robot instance
      self.scene.articulations["robot"] = self._robot  # Adds to scene
  ```

## 3. Terrain

### Defined In: Terrain Configuration

**File:** `training/go2_lidar/go2_lidar/terrain/train_terrain_cfg.py` (line 14-30)

```python
GO2_LOCO_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=0,
    size=(8.0, 8.0),           # Terrain patch size
    border_width=20.0,
    num_rows=10,                # Curriculum levels
    num_cols=20,                # Terrains per level
    horizontal_scale=0.1,       # Resolution
    vertical_scale=0.005,       # Height scale
    slope_threshold=0.75,
    use_cache=True,
    curriculum=True,            # Progressive difficulty
    sub_terrains={
        "random_rough": terrain_gen.HfRandomUniformTerrainCfg(
            proportion=1.0,
            noise_range=(0.01, 0.03),
            noise_step=0.01,
            border_width=0.1,
        ),
    },
)
```

**What this defines:**
- **Terrain generation algorithm**: Procedurally generates rough terrain
- **Terrain parameters**: Size, resolution, difficulty
- **Curriculum**: Multiple difficulty levels for training

**Used In:**
- `go2_loco_env_cfg.py` (line 165-183):
  ```python
  terrain = TerrainImporterCfg(
      prim_path="/World/ground",
      terrain_type="generator",
      terrain_generator=GO2_LOCO_TERRAIN_CFG,  # Uses the config above
      ...
  )
  ```

**Instantiated In:**
- `go2_loco_env.py` (line 138):
  ```python
  self._terrain: TerrainImporter = self.cfg.terrain.class_type(self.cfg.terrain)
  ```
  - Creates terrain generator instance
  - Generates terrain meshes procedurally

## 4. Sensors

### Contact Sensor

**Defined In:** `go2_loco_env_cfg.py` (line 187-192)

```python
contact_sensor = ContactSensorCfg(
    prim_path="/World/envs/env_.*/Robot/.*",
    history_length=3,
    track_air_time=True,
    track_pose=True,
)
```

**Instantiated In:** `go2_loco_env.py` (line 128-129)

```python
self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
self.scene.sensors["contact_sensor"] = self._contact_sensor
```

**What it does:**
- Detects collisions between robot parts and ground/obstacles
- Tracks which bodies are in contact
- Used for reward computation and safety checks

### LiDAR Ray Caster (Filter/Navigation)

**Defined In:** `go2_filter_env_cfg.py` / `go2_nav_env_cfg.py`

```python
raycaster = RayCasterCfgExtended(
    prim_path="/World/envs/env_.*/Robot/base",
    offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
    ...
)
```

**Instantiated In:** `go2_filter_env.py` / `go2_nav_env.py`

```python
self._raycaster = RayCasterDynamic(self.cfg.raycaster, sim_mid360=True)
self.scene.sensors["raycaster"] = self._raycaster
```

**What it does:**
- Simulates LiDAR sensor
- Casts rays from robot base
- Returns distance measurements
- Used as observations for filter/navigation policies

## 5. Scene Setup Flow

### Step-by-Step: How Everything Comes Together

**File:** `go2_loco_env.py` → `_setup_scene()` method (line 124-146)

```python
def _setup_scene(self):
    # 1. Create robot instance
    self._robot = Articulation(self.cfg.robot)
    self.scene.articulations["robot"] = self._robot
    
    # 2. Create contact sensor
    self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
    self.scene.sensors["contact_sensor"] = self._contact_sensor
    
    # 3. Configure terrain generator
    self.cfg.terrain.num_envs = self.scene.cfg.num_envs
    self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
    self.cfg.terrain.terrain_generator.num_rows = self.num_terrain_rows
    self.cfg.terrain.terrain_generator.num_cols = self.num_terrain_cols
    
    # 4. Create terrain instance
    self._terrain: TerrainImporter = self.cfg.terrain.class_type(self.cfg.terrain)
    
    # 5. Assign terrain patches to environments
    rand_cols = torch.randint(0, self.num_terrain_cols, size=(self.num_envs,), device=self.device)
    self._terrain.env_origins[:] = self._terrain.terrain_origins[0, rand_cols]
    
    # 6. Clone environments (create multiple parallel environments)
    self.scene.clone_environments(copy_from_source=False)
    
    # 7. Filter collisions (optimize physics)
    self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
    
    # 8. Add lighting
    sky_light_cfg = sim_utils.DomeLightCfg(...)
    sky_light_cfg.func("/World/skyLight", sky_light_cfg)
```

## 6. Base Class: DirectRLEnv

### What It Provides

**File:** `training/IsaacLab/source/isaaclab/isaaclab/envs/direct_rl_env.py`

**Key Responsibilities:**

1. **Creates Simulation Context** (line 99-100):
   ```python
   self.sim: SimulationContext = SimulationContext(self.cfg.sim)
   ```
   - This is the physics engine (PhysX)

2. **Creates Scene** (inherited from base):
   ```python
   self.scene: InteractiveScene = InteractiveScene(self.cfg.scene)
   ```
   - Container for all objects (robots, terrain, sensors)
   - Manages multiple parallel environments

3. **Sets Up Device** (line 105-106):
   ```python
   if "cuda" in self.device:
       torch.cuda.set_device(self.device)
   ```
   - Ensures GPU simulation if available

4. **Calls _setup_scene()**:
   - This is where child classes (Go2LocoEnv) add their specific objects

## 7. Complete File Locations

| Component | Definition Location | Instantiation Location |
|-----------|-------------------|----------------------|
| **Physics Engine** | `direct_rl_env.py` (DirectRLEnv.__init__) | `direct_rl_env.py` |
| **Robot Model** | `isaaclab_assets/robots/unitree.py` (UNITREE_GO2_CFG) | `go2_loco_env.py` (_setup_scene) |
| **Robot Config** | `go2_loco_env_cfg.py` (robot: ArticulationCfg) | `go2_loco_env.py` |
| **Terrain Config** | `terrain/train_terrain_cfg.py` (GO2_LOCO_TERRAIN_CFG) | `go2_loco_env.py` (_setup_scene) |
| **Terrain Usage** | `go2_loco_env_cfg.py` (terrain: TerrainImporterCfg) | `go2_loco_env.py` |
| **Contact Sensor** | `go2_loco_env_cfg.py` (contact_sensor: ContactSensorCfg) | `go2_loco_env.py` (_setup_scene) |
| **LiDAR Sensor** | `go2_filter_env_cfg.py` / `go2_nav_env_cfg.py` | `go2_filter_env.py` / `go2_nav_env.py` |
| **Scene Container** | `direct_rl_env.py` (InteractiveScene) | `direct_rl_env.py` |

## 8. Key Design Patterns

### 1. Configuration-Based Design
- **What**: Config files define WHAT to create
- **How**: Classes define HOW to create
- **When**: `_setup_scene()` defines WHEN to create

### 2. Separation of Concerns
- **Base class** (`DirectRLEnv`): Creates simulation infrastructure
- **Config classes**: Define parameters
- **Child classes** (`Go2LocoEnv`): Add task-specific objects

### 3. Procedural Generation
- **Terrain**: Generated at runtime, not pre-made
- **Robot**: Loaded from USD file (pre-made model)
- **Sensors**: Created from configs

### 4. Vectorization
- **Multiple environments**: Created by `scene.clone_environments()`
- **Parallel simulation**: All environments run simultaneously on GPU
- **Efficient**: Shared physics engine, separate state

## 9. Understanding the USD File

The robot model (`go2.usd`) is a **Universal Scene Description** file that contains:

- **3D meshes**: Visual appearance of robot parts
- **Collision shapes**: Simplified shapes for physics
- **Joints**: Revolute joints connecting parts
- **Actuators**: Motor definitions
- **Materials**: Visual materials and textures

**Location:** Isaac Lab's asset directory (downloaded with Isaac Lab)

**Created by:** Unitree Robotics (the robot manufacturer)

**Modified by:** Isaac Lab team (added physics properties, sensors)

## 10. Summary: The Complete Creation Flow

```
1. Script calls gym.make("Unitree-Go2-Locomotion", cfg=env_cfg)
   │
   └─> Go2LocoEnv.__init__(cfg)
       │
       ├─> DirectRLEnv.__init__(cfg)  [BASE CLASS]
       │   ├─> Creates SimulationContext (physics engine)
       │   ├─> Creates InteractiveScene (object container)
       │   └─> Sets up device, seed, etc.
       │
       └─> Go2LocoEnv-specific setup
           └─> Calls _setup_scene()  [CHILD CLASS]
               ├─> Creates robot: Articulation(cfg.robot)
               │   └─> Loads from: UNITREE_GO2_CFG → go2.usd file
               │
               ├─> Creates terrain: TerrainImporter(cfg.terrain)
               │   └─> Uses: GO2_LOCO_TERRAIN_CFG → Procedurally generates
               │
               ├─> Creates sensors: ContactSensor(cfg.contact_sensor)
               │   └─> Uses: ContactSensorCfg
               │
               └─> Adds everything to scene
                   └─> Clones for multiple environments
```

## Key Takeaways

1. **Physics Engine**: Created by `DirectRLEnv` base class
2. **Robot Model**: Defined in `UNITREE_GO2_CFG`, loaded from USD file
3. **Terrain**: Defined in `GO2_LOCO_TERRAIN_CFG`, procedurally generated
4. **Sensors**: Defined in config files, instantiated in `_setup_scene()`
5. **Scene**: Container managed by `DirectRLEnv`, populated by `_setup_scene()`

The actual simulation world is a combination of:
- **Pre-made assets** (robot USD file)
- **Procedural generation** (terrain)
- **Configuration-driven** (sensors, physics parameters)
- **Runtime instantiation** (everything created in `_setup_scene()`)

