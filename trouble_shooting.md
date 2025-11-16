#### Pylance Missing Indexing of Extensions

In some VsCode versions, the indexing of part of the extensions is missing.
In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`. (IsaacLab address is also needed)

```json
{
    "python.analysis.extraPaths": [
        "${workspaceFolder}/source/aerial_lab",
        "<path-to-ext-IsaacLab>/IsaacLab/source/isaaclab",
        "<path-to-ext-IsaacLab>/IsaacLab/source/isaaclab_tasks",
        "<path-to-ext-IsaacLab>/IsaacLab/source/isaaclab_assets",
        "<path-to-ext-IsaacLab>/IsaacLab/source/isaaclab_rl"
    ]
}
```

#### Pylance Crash

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

#### No space left on device
```bash
[Error] [carb] Failed to create change watch for `/{absolutePath}/miniconda3/envs/aeriallab/lib/python3.11/site-packages/isaacsim/extscache/omni.physx.fabric-107.3.18+107.3.1.lx64.r.cp311.u353/omni/physxfabric/scripts`: errno=28/No space left on device
```
Just too many mesh files are imported in IsaacLab, set more large capacity
```bash
echo "fs.inotify.max_user_watches=524288" | sudo tee -a /etc/sysctl.conf
sudo sysctl -p
```

#### isaaclab.sim.converters.UrdfConverter related warnings and errors
1. Make sure your `.urdf` files can correctly shown in `URDF.Visualizer` or `Rviz2`
2. Make sure all the related files' name only have `0~9`, `a~z`, `A~Z` and `_`, other symbols cannot be identified.
3. Make sure your `.dae` files don't have other symbols, if you got these from `SolidWorks`, don't use name with `NO ENGLISH`
