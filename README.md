# Articulated Aerial Robot IsaacLab Environments
This is a project for Reinforcement Learning of Articulated Aerial Robot (AAR). Currently, the aerial robots are mainly from [DRAGON Lab](http://www.dragon.t.u-tokyo.ac.jp/). We hope to build a systematic project to solve motion planning and control problems of AARs.

## Research Work: 

### Learning Agile and Robust Omnidirectional Aerial Motion on Overactuated Tiltable-Quadrotors
![framework](/source/aerial_lab/docs/overframework.png)
```
@article{zhang2026omnirl,
  title   = {Learning Agile and Robust Omnidirectional Aerial Motion
             on Overactuated Tiltable-Quadrotors},
  author  = {Wentao Zhang and Zhaoqi Ma and Jinjie Li and Huayi Wang and Haokun Liu
             and Junichiro Sugihara and Chen Chen and Yicheng Chen and Moju Zhao},
  year={2026},
  eprint={2602.21583},
  archivePrefix={arXiv},
  primaryClass={cs.RO},
  url={https://arxiv.org/abs/2602.21583},
}
```
You can access platform-related code on [jsk_aerial_robot](https://github.com/ZWT006/jsk_aerial_robot), where you can try our controller in the Gazebo simulation by following the instructions in `robots/beetle_omni/policy/README.md`.

## Overview
This project/repository is an isolated environment, outside of the core Isaac Lab repository.
```yaml
Aerial_Lab
├── .vscode
├── logs    # training records
│   ├── rsl_rl
│   │   ├── TASK-NAME
│   │   │   ├── {year}-{month}-{day}_{hour}-{minute}-{second}
│   │   │   │   ├── exported    # exported neural network files
│   │   │   │   ├── git         # .diff files, helpful for fine-tuning parameters
│   │   │   │   ├── params      # rl algorithms parameters
│   │   │   │   ├── events.out.tfevents.xxxx    # tensorboard training data record
│   │   │   │   ├── model_xxxx.pt   # checkpoint NN files of training process
│   │   │   │   └── model_~~~~
│   │   ~   ~
│   ├── rl_games
│   └── skrl
├── outputs
├── scripts
│   ├── rl_games        # rl_games rl library
│   ├── rsl_rl          # rsl rl library (commonly used)
│   ├── skrl            # skrl rl library
│   ├── clean_trash.py  # clean all training records, use carefully
│   ├── debug_agent.py  # customed debug
│   ├── list_envs.py    # list all available envs
│   ├── random_agent.py # set action = rand, for debug
│   └── zero_agent.py   # set action = 0, for debug
├── source
│   └── aerial_lab
│       ├── aerial_lab
│       │   ├── actuators   # rotor class
│       │   ├── assets      # robot simulation
│       │   │   └── aerialrobot.py  # articulated robots (Beetle, Spidar, .etc)
│       │   ├── tasks  
│       │   │   ├── direct
│       │   │   │   ├── beetle
│       │   │   │   │   ├── agents  # rl algorithms configuration
│       │   │   │   │   │   ├── rl_games_ppo_cfg.yaml
│       │   │   │   │   │   ├── rsl_rl_ppo_cfg.py
│       │   │   │   │   │   └── skrl_rl_ppo_cfg.yaml
│       │   │   │   │   ├── __init__.py     # import task configuration
│       │   │   │   │   └── beetle_env.py   # rl environment
│       │   │   │   ├── spidar
│       │   │   │   └── xxxxxx
│       │   │   └── manager_based   # currently unused
│       │   ├── utility
│       │   │   ├── math.py         # math functions
│       │   │   └── noisemodel.py   # noise class
│       │   └── ui_extension_example.py # can be neglected
│       ├── aerial_lab.egg-info # can be neglected
│       ├── config              # can be neglected
│       ├── data
│       │   └── Robots          # robot urdf files
│       ├── doc
│       ├── pyproject.toml      # can be neglected
│       └── setup.py    # python dependences for project utilization
├── test                # unite debug and test code
├── .dockerignore
├── .flake8             # python format configuration
├── .gitattributes
├── .gitignore          # neglect unnecessary files
├── .pre-commit-config.yaml
├── README.md           # Project Overview
└── Trouble_shooting.md # Some common problems and how to handle
```

### Available Environments
| S. No. |         Task Name       |      Entry Point   |         Config        |
|--------|-------------------------|--------------------|-----------------------|
|   1    | Aerial-MiniQuadcopter-Pose-v0         | aerial_lab.tasks.direct.quadcopter.mini_quadcopter_env:MiniQuadcopterEnv                          | aerial_lab.tasks.direct.quadcopter.mini_quadcopter_env:MiniQuadcopterEnvCfg                              |
|   2    | Beetle-Pose-v0         | aerial_lab.tasks.direct.beetle.beetle_env:BeetleEnv                          | aerial_lab.tasks.direct.beetle.beetle_env:BeetleEnvCfg                              |
|   3    | Beetle-Omni-Pose-v0         | aerial_lab.tasks.direct.beetle.beetle_omni_env:BeetleEnv                          | aerial_lab.tasks.direct.beetle.beetle_omni_env:BeetleOmniEnvCfg                              |
<!-- |   1    | Aerial-Template-Direct-v0            | aerial_lab.tasks.direct.aerial_lab.aerial_lab_env:AerialLabEnv               | aerial_lab.tasks.direct.aerial_lab.aerial_lab_env_cfg:AerialLabEnvCfg               |
|   2    | Aerial-Template-Marl-Direct-v0       | aerial_lab.tasks.direct.aerial_lab_marl.aerial_lab_marl_env:AerialLabMarlEnv | aerial_lab.tasks.direct.aerial_lab_marl.aerial_lab_marl_env_cfg:AerialLabMarlEnvCfg |
|   3    | Beetle-Direct-Pose-v0         | aerial_lab.tasks.direct.beetle.beetle_env:BeetleEnv                          | aerial_lab.tasks.direct.beetle.beetle_env:BeetleEnvCfg                              |
|   4    | Beetle-Direct-Pose-DEBUG-v0   | aerial_lab.tasks.direct.beetle.beetle_env_debug:BeetleEnv                    | aerial_lab.tasks.direct.beetle.beetle_env_debug:BeetleEnvCfg                        |
|   5    | Dragon-Direct-v0              | aerial_lab.tasks.direct.dragon.dragon_env:DragonEnv                          | aerial_lab.tasks.direct.dragon.dragon_env:DragonEnvCfg                              |
|   6    | Quadcopter-Direct-v0          | aerial_lab.tasks.direct.quadcopter.quadcopter_env:QuadcopterEnv              | aerial_lab.tasks.direct.quadcopter.quadcopter_env:QuadcopterEnvCfg                  |
|   7    | Spidar-Direct-v0              | aerial_lab.tasks.direct.spidar.spidar_env:SpidarEnv                          | aerial_lab.tasks.direct.spidar.spidar_env:SpidarEnvCfg                              |
|   8    | Aerial-Velocity-Flat-Unitree-A1-v0   | isaaclab.envs:ManagerBasedRLEnv                                              | aerial_lab.tasks.manager_based.a1.flat_env_cfg:UnitreeA1FlatEnvCfg                  |
|   9    | Aerial-Velocity-Rough-Unitree-A1-v0  | isaaclab.envs:ManagerBasedRLEnv                                              | aerial_lab.tasks.manager_based.a1.rough_env_cfg:UnitreeA1RoughEnvCfg                |
|   10   | Aerial-Template-v0                   | isaaclab.envs:ManagerBasedRLEnv                                              | aerial_lab.tasks.manager_based.aerial_lab.aerial_lab_env_cfg:AerialLabEnvCfg        |
|   11   | Aerial-Velocity-Flat-Unitree-Go1-v0  | isaaclab.envs:ManagerBasedRLEnv                                              | aerial_lab.tasks.manager_based.go1.flat_env_cfg:UnitreeGo1FlatEnvCfg                |
|   12   | Aerial-Velocity-Rough-Unitree-Go1-v0 | isaaclab.envs:ManagerBasedRLEnv                                              | aerial_lab.tasks.manager_based.go1.rough_env_cfg:UnitreeGo1RoughEnvCfg             | -->

### Dependence
1. [Isaac Sim](https://isaac-sim.github.io/IsaacLab/release/v2.2.1/source/setup/installation/index.html) `5.0.0`
2. [Isaac Lab](https://github.com/isaac-sim/IsaacLab) `2.2.1`
3. [onnxruntime]`onnxruntime-1.16.3-cp38` for `{robot}_deploy.py`


**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** articulated, isaaclab

## Installation

### Nvidia Driver
Older NVIDIA driver versions may not support newer CUDA versions. It is recommended to install a GPU driver version `570` or higher.

### Isaac Lab
Install Isaac Lab by following the [installation guide : release 2.2.1](https://isaac-sim.github.io/IsaacLab/release/v2.2.1/source/setup/installation/index.html).
We recommend using the **conda or uv installation** as it simplifies calling Python scripts from the terminal.
The detailed installation process are as follows:

- Install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install). Be careful to the python version of your host machine. Choose `To download different version` and find host `python-version` and `system architecture`
- Create virtual environment, all the procedure below should be in `aeriallab` env
    ```bash
    conda create -n aeriallab python=3.11
    conda activate aeriallab
    ```

- Install dependencies
    ```bash
    pip install --upgrade pip
    pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
    pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com
    # Verifying Isaac Sim, it may consume a long time in first launch, also encounter `force quit`, just wait
    isaacsim
    ```

- Installing Isaac Lab
    ```bash
    git clone https://github.com/isaac-sim/IsaacLab.git
    cd IsaacLab
    git checkout -b aeriallab 47780cf02dae94410cfed81706c8c859eeeacd76
    sudo apt install cmake build-essential
    ./isaaclab.sh --install rl_games rsl_rl sb3 skrl robomimic # rl libraries
    # verifying Isaac Lab, it may consume a long time in first launch, also encounter `force quit`, just wait
    python scripts/tutorials/00_sim/create_empty.py
    ```

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory), install the library in editable mode using:
    ```bash
    cd aerial_lab
    git checkout -b release
    python -m pip install -e source/aerial_lab
    ```

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: If the task name changes, it may be necessary to update the search pattern `"Template-"`
        (in the `scripts/list_envs.py` file) so that it can be listed.

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/list_envs.py
        ```

    - Running a task:

        ```bash
        # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
        python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
        ```

    - Running a task with dummy agents:

        These include dummy agents that output zero or random agents. They are useful to ensure that the environments are configured correctly.

        - Zero-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/zero_agent.py --task=<TASK_NAME>
            ```
        - Random-action agent

            ```bash
            # use 'FULL_PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
            python scripts/random_agent.py --task=<TASK_NAME>
            ```

## Training
Currently, only `Beetle-Pose-v0`, `Beetle-Omni-Pose-v0` and `Aerial-MiniQuadcopter-Pose-v0` are available, while `rsl_rl` is recommended. If you want to train your own platform, pay attention to: `assets/aerialrobot.py`, `tasks/direct/xxx` and `data/Robots/xxx`. You can also access other AARs' model files on `data/Robots`

## Tutorials
If you want to learn the stack step by step, start with the Isaac Sim tutorials in [tutorials/README.md](tutorials/README.md).

Recommended order:

```bash
python tutorials/isaac_sim/00_create_empty.py --headless
python tutorials/isaac_sim/01_add_scene.py
python tutorials/isaac_sim/02_spawn_robot.py --robot mini_quad
python tutorials/isaac_sim/03_step_robot.py --robot mini_quad --steps 600
```

After that, move to the environment-oriented practice scripts in [scripts/tutorials/README.md](scripts/tutorials/README.md), then start RL training.

### Training specific task
Start training process w/o video record
```bash
# training with gui (time consuming, unrecommended!)
python scripts/rsl_rl/train.py --task=Beetle-Omni-Pose-v0
# training without gui
python scripts/rsl_rl/train.py --task=Beetle-Omni-Pose-v0 --headless
# training without gui , with video record
python scripts/rsl_rl/train.py --task=Beetle-Omni-Pose-v0 --headless --video --video_length 1000 --video_interval 10000
```
### Show Training Data
```bash
tensorboard --logdir=logs/rsl_rl/beetle_omni/ --port=6006
```
Then, in your web browser, view `http://localhost:6006`

### Review training result
Play trained policy w/o video record
```bash
# Play trained policy (latest checkpoint)
python scripts/rsl_rl/play.py --task=Beetle-Omni-Pose-v0 --num_envs=64
# Play trained policy with specific checkpoint
python scripts/rsl_rl/play.py --task=Beetle-Omni-Pose-v0 --checkpoint=logs/rsl_rl/beetle_omni/2025-10-31_13-34-44/model_8000.pt --num_envs=64
```
To facilitate evaluate policy, you can check outcomes in `exported/` and `video/`.
```bash
# Evaluate single policy
python scripts/rsl_rl/evaluate.py --task=Beetle-Omni-Pose-v0 --num_envs=64 --device=cpu --video --video_length=500 --checkpoint=logs/rsl_rl/beetle_omni/2025-12-01_10-46-19/model_10000.pt --headless
# Evaluate policy batch
python scripts/rsl_rl/eval_ckpts.py --task=Beetle-Omni-Pose-v0 --num_envs=64 --device=cpu --video --video_length=500 --run_dir=logs/rsl_rl/beetle_omni/2025-12-01_10-46-19 --headless
```

### Git Repository
To facilitate the git to store training code, just modify `envs/aeriallab/lib/python3.11/site-packages/rsl_rl/utils/utils.py/store_code_state` source code as:
```python
with open(diff_file_name, "x", encoding="utf-8") as f:
		# content = f"--- git status ---\n{repo.git.status()} \n\n\n--- git diff ---\n{repo.git.diff(t)}"
            content = ( f"--- git status ---\n{repo.git.status()} \n\n"
                        f"--- git log ---\n{repo.git.log('-1')} \n\n"
                        f"--- git diff ---\n{repo.git.diff(t)}")
```
which will store current commit.

Switch to log/git/diff
```bash
### exact diff file
awk '/^diff --git /,0' logs/rsl_rl/beetle_direct/2025-11-11_10-06-52/git/aerial_lab.diff > clean.diff
### apply diff
git apply clean.diff
```

### Set up IDE (recommended)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### URDF Visualizer
Isaac Sim provides an importer for `.urdf` to convert to `.usd`, but not for `.xacro`. Therefore, we recommend obtaining the `.urdf` file directly.
- convert `.xacro` to `.urdf`
    ```bash
    # 1. install tools
    sudo apt install liburdfdom-tools
    sudo apt install ros-noetic-urdf
    # 2. convert xacro to urdf
    rosrun xacro xacro robot.xacro > robot.urdf
    # 3. check urdf
    check_urdf robot.urdf
    # view urdf
    urdf_to_graphiz robot.urdf
    ```
- set package address in `.vscode/settings.json`
    ```json
        "urdf-visualizer.packages": {
            "beetle": "${workspaceFolder}/source/aerial_lab/data/Robots/beetle",
            "beetle_omni": "${workspaceFolder}/source/aerial_lab/data/Robots/beetle_omni",
            "beetle_hyper": "${workspaceFolder}/source/aerial_lab/data/Robots/beetle_hyper",
            "mini_quadrotor": "${workspaceFolder}/source/aerial_lab/data/Robots/mini_quadrotor",
            "spidar": "${workspaceFolder}/source/aerial_lab/data/Robots/spidar",
            "dragon": "${workspaceFolder}/source/aerial_lab/data/Robots/dragon",
            "hydrus": "${workspaceFolder}/source/aerial_lab/data/Robots/hydrus",
            "a1_description": "${workspaceFolder}/source/aerial_lab/data/Robots/a1_description",
            "go1_description": "${workspaceFolder}/source/aerial_lab/data/Robots/go1_description"
        }
    ```

- Click the `eye` symbol on the right corner of the `.urdf` or `.xacro` file.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/aerial_lab/aerial_lab/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

### Setup IsaacLab Python path
```json
"python.analysis.extraPaths": [
        "${workspaceFolder}/source/aerial_lab",
        "~/miniconda3/envs/aeriallab/lib/python3.11/site-packages",
        "<path_to_isaaclab>/IsaacLab/source",
        "<path_to_isaaclab>/IsaacLab/source/isaaclab_mimic",
        "<path_to_isaaclab>/IsaacLab/source/isaaclab",
        "<path_to_isaaclab>/IsaacLab/source/isaaclab_rl",
        "<path_to_isaaclab>/IsaacLab/source/isaaclab_assets",
        "<path_to_isaaclab>/IsaacLab/source/isaaclab_tasks"
    ],
```

## Code formatting

We have a pre-commit template to automatically format your code.
To install pre-commit:

```bash
pip install pre-commit
```

Then you can run pre-commit with:

```bash
pre-commit run --all-files
```
## [Troubleshooting](./trouble_shooting.md)

Research related questions, please contact Prof. Zhao at `chou@dragon.t.u-tokyo.ac.jp`
Code related questions, please contact `wentao-zhang@dragon.t.u-tokyo.ac.jp`
