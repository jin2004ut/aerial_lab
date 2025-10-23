# Articulated Aerial Robot IsaacLab Environments

## Overview

### Available Environments
| S. No. |         Task Name       |      Entry Point   |         Config        |  
|--------|-------------------------|--------------------|-----------------------|
|   1    | Aerial-Lab-Template-Direct-v0            | aerial_lab.tasks.direct.aerial_lab.aerial_lab_env:AerialLabEnv               | aerial_lab.tasks.direct.aerial_lab.aerial_lab_env_cfg:AerialLabEnvCfg               |
|   2    | Aerial-Lab-Template-Marl-Direct-v0       | aerial_lab.tasks.direct.aerial_lab_marl.aerial_lab_marl_env:AerialLabMarlEnv | aerial_lab.tasks.direct.aerial_lab_marl.aerial_lab_marl_env_cfg:AerialLabMarlEnvCfg |
|   3    | Aerial-Lab-Beetle-Direct-v0              | aerial_lab.tasks.direct.beetle.beetle_env:BeetleEnv                          | aerial_lab.tasks.direct.beetle.beetle_env:BeetleEnvCfg                              |
|   4    | Aerial-Lab-Quadcopter-Direct-v0          | aerial_lab.tasks.direct.quadcopter.quadcopter_env:QuadcopterEnv              | aerial_lab.tasks.direct.quadcopter.quadcopter_env:QuadcopterEnvCfg                  |
|   5    | Aerial-Lab-Velocity-Flat-Unitree-A1-v0   | isaaclab.envs:ManagerBasedRLEnv                                              | aerial_lab.tasks.manager_based.a1.flat_env_cfg:UnitreeA1FlatEnvCfg                  |
|   6    | Aerial-Lab-Velocity-Rough-Unitree-A1-v0  | isaaclab.envs:ManagerBasedRLEnv                                              | aerial_lab.tasks.manager_based.a1.rough_env_cfg:UnitreeA1RoughEnvCfg                |
|   7    | Aerial-Lab-Template-v0                   | isaaclab.envs:ManagerBasedRLEnv                                              | aerial_lab.tasks.manager_based.aerial_lab.aerial_lab_env_cfg:AerialLabEnvCfg        |
|   8    | Aerial-Lab-Velocity-Flat-Unitree-Go1-v0  | isaaclab.envs:ManagerBasedRLEnv                                              | aerial_lab.tasks.manager_based.go1.flat_env_cfg:UnitreeGo1FlatEnvCfg                |
|   9    | Aerial-Lab-Velocity-Rough-Unitree-Go1-v0 | isaaclab.envs:ManagerBasedRLEnv                                              | aerial_lab.tasks.manager_based.go1.rough_env_cfg:UnitreeGo1RoughEnvCfg              |

### Dependence
1. [Isaac Sim](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html) `5.0.0`
2. [Isaac Lab](https://github.com/isaac-sim/IsaacLab) `2.2.1`

This project/repository serves as a template for building projects or extensions based on Isaac Lab.
It allows you to develop in an isolated environment, outside of the core Isaac Lab repository.

**Key Features:**

- `Isolation` Work outside the core Isaac Lab repository, ensuring that your development efforts remain self-contained.
- `Flexibility` This template is set up to allow your code to be run as an extension in Omniverse.

**Keywords:** extension, template, isaaclab

## Installation

### Nvidia Driver
Older version nvidia-driver cannot support high version cuda. It is recommanded to install your gpu driber more than `570`

### Isaac Lab
Install Isaac Lab by following the [installation guide](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html).
We recommend using the conda or uv installation as it simplifies calling Python scripts from the terminal.

- Install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install). Be careful to the python version of your host machine.
- Create virtual environment
    ```bash
    conda create -n aeriallab python=3.11
    conda activate aeriallab
    ```

- Clone or copy this project/repository separately from the Isaac Lab installation (i.e. outside the `IsaacLab` directory):

- Using a python interpreter that has Isaac Lab installed, install the library in editable mode using:

    ```bash
    # use 'PATH_TO_isaaclab.sh|bat -p' instead of 'python' if Isaac Lab is not installed in Python venv or conda
    python -m pip install -e source/aerial_lab

- Verify that the extension is correctly installed by:

    - Listing the available tasks:

        Note: It the task name changes, it may be necessary to update the search pattern `"Template-"`
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

### Show Training Data
```bash
tensorboard --logdir=logs/rsl_rl/beetle_direct/ --port=6006
```

Start training process with video record
```bash
python source/standalone/workflows/rl_games/train.py --task=Isaac-Cartpole-v0 --headless --video --video_length 100 --video_interval 500
```
Play trained policy with specific checkpoint
```bash
python scripts/rsl_rl/play.py --task=Aerial-Lab-Beetle-Direct-DEBUG-v0 --checkpoint=logs/rsl_rl/beetle_direct_debug/2025-10-22_22-34-33/model_1000.pt  --num_envs=16
```

### URDF Visualizer
isaacsim provide importer for `.urdf` to convert to `.usd` but not for `.xacro`. So, we recommend to get `.urdf` file.
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
            "mini_quadrotor": "${workspaceFolder}/source/aerial_lab/data/Robots/mini_quadrotor",
            "spidar": "${workspaceFolder}/source/aerial_lab/data/Robots/spidar",
            "dragon": "${workspaceFolder}/source/aerial_lab/data/Robots/dragon",
            "hydrus": "${workspaceFolder}/source/aerial_lab/data/Robots/hydrus",
            "a1_description": "${workspaceFolder}/source/aerial_lab/data/Robots/a1_description",
            "go1_description": "${workspaceFolder}/source/aerial_lab/data/Robots/go1_description"
        }
    ```

- click `eye` symble on the right corner of `.urdf` or `.xacro` file.

### Set up IDE (Optional)

To setup the IDE, please follow these instructions:

- Run VSCode Tasks, by pressing `Ctrl+Shift+P`, selecting `Tasks: Run Task` and running the `setup_python_env` in the drop down menu.
  When running this task, you will be prompted to add the absolute path to your Isaac Sim installation.

If everything executes correctly, it should create a file .python.env in the `.vscode` directory.
The file contains the python paths to all the extensions provided by Isaac Sim and Omniverse.
This helps in indexing all the python modules for intelligent suggestions while writing code.

### Setup as Omniverse Extension (Optional)

We provide an example UI extension that will load upon enabling your extension defined in `source/aerial_lab/aerial_lab/ui_extension_example.py`.

To enable your extension, follow these steps:

1. **Add the search path of this project/repository** to the extension manager:
    - Navigate to the extension manager using `Window` -> `Extensions`.
    - Click on the **Hamburger Icon**, then go to `Settings`.
    - In the `Extension Search Paths`, enter the absolute path to the `source` directory of this project/repository.
    - If not already present, in the `Extension Search Paths`, enter the path that leads to Isaac Lab's extension directory directory (`IsaacLab/source`)
    - Click on the **Hamburger Icon**, then click `Refresh`.

2. **Search and enable your extension**:
    - Find your extension under the `Third Party` category.
    - Toggle it to enable your extension.

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

## Troubleshooting

### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/source/aerial_lab"
    ]
}
```

### Pylance Crash

If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
