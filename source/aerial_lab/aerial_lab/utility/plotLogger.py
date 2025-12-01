import csv
import time
from collections import deque
from datetime import datetime
from pathlib import Path

# Plotting imports
import matplotlib
import numpy as np

matplotlib.use("Agg")  # Use non-interactive backend for thread safety
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


class ObservationLogger:
    """
    Logger for observation data with plotting capabilities.
    Saves plots with auto-incrementing filenames.
    """

    def __init__(self, save_dir: str = "plots", max_samples: int = 1000, plot_interval: float = 2.0):
        """
        Initialize observation logger.

        Args:
            save_dir: Directory to save plots (relative to script location)
            max_samples: Maximum number of samples to keep in memory
            plot_interval: Time interval (seconds) between plot updates
        """
        # Convert to Path object
        save_path = Path(save_dir)

        # Use absolute path if provided, otherwise make it absolute from current directory
        if save_path.is_absolute():
            self.save_dir = save_path
        else:
            self.save_dir = Path.cwd() / save_path

        # Create directory (including parent directories if needed)
        try:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            print(f"✓ Created/verified save directory: {self.save_dir}")
        except Exception as e:
            raise RuntimeError(f"Failed to create save directory {self.save_dir}: {e}")

        self.max_samples = max_samples
        self.plot_interval = plot_interval
        self.last_plot_time = time.time()

        # Data buffers with labels
        self.obs_labels = [
            "lin_vel_x",
            "lin_vel_y",
            "lin_vel_z",
            "ang_vel_x",
            "ang_vel_y",
            "ang_vel_z",
            "gravity_x",
            "gravity_y",
            "gravity_z",
            "goal_pos_x",
            "goal_pos_y",
            "goal_pos_z",
            "ang_err_x",
            "ang_err_y",
            "ang_err_z",
            "gimbal_0",
            "gimbal_1",
            "gimbal_2",
            "gimbal_3",
            "root_rot_0",
            "root_rot_1",
            "root_rot_2",
            "root_rot_3",
            "root_rot_4",
            "root_rot_5",
            "goal_rot_0",
            "goal_rot_1",
            "goal_rot_2",
            "goal_rot_3",
            "goal_rot_4",
            "goal_rot_5",
            "last_action_0",
            "last_action_1",
            "last_action_2",
            "last_action_3",
            "last_action_4",
            "last_action_5",
            "last_action_6",
            "last_action_7",
        ]

        self.action_labels = [
            "action_0",
            "action_1",
            "action_2",
            "action_3",
            "action_4",
            "action_5",
            "action_6",
            "action_7",
        ]

        self.timestamps = deque(maxlen=max_samples)
        self.obs_data = deque(maxlen=max_samples)
        self.action_data = deque(maxlen=max_samples)

        self.start_time = time.time()
        self.plot_counter = self._get_next_plot_number()

        print(f"ObservationLogger initialized. Plots will be saved to: {self.save_dir}")
        print(f"Starting plot counter at: {self.plot_counter}")

    def _get_next_plot_number(self) -> int:
        """Find the next available plot number."""
        existing_plots = list(self.save_dir.glob("obs_plot_*.png"))
        if not existing_plots:
            return 0

        numbers = []
        for p in existing_plots:
            try:
                num = int(p.stem.split("_")[-1])
                numbers.append(num)
            except ValueError:
                continue

        return max(numbers) + 1 if numbers else 0

    def log(self, obs: np.ndarray, action: np.ndarray):
        """
        Log observation and action data.

        Args:
            obs: Observation vector (39,)
            action: Action vector (8,)
        """
        current_time = time.time()

        self.timestamps.append(current_time)
        self.obs_data.append(obs.copy())
        self.action_data.append(action.copy())

        # Check if it's time to plot
        # if current_time - self.last_plot_time >= self.plot_interval:
        #     self.plot_and_save()
        #     self.last_plot_time = current_time

    def save_to_csv(self, filename: str = None) -> Path:
        """
        Save logged data to CSV file.

        Args:
            filename: Custom filename (optional). If None, auto-generated with timestamp.

        Returns:
            Path to saved CSV file
        """
        if len(self.timestamps) == 0:
            print("⚠ No data to save")
            return None

        # Generate filename
        if filename is None:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"obs_data_{timestamp_str}.csv"

        csv_path = self.save_dir / filename

        try:
            # Convert data to numpy arrays
            timestamps = np.array(self.timestamps)
            obs = np.array(self.obs_data)
            actions = np.array(self.action_data)

            # Create header
            header = ["timestamp", "time_elapsed"] + self.obs_labels + self.action_labels

            # Open CSV file
            with open(csv_path, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(header)

                # Write data rows
                start_time = timestamps[0]
                for i in range(len(timestamps)):
                    row = [
                        timestamps[i],  # Absolute timestamp
                        timestamps[i] - start_time,  # Relative time
                        *obs[i].tolist(),  # Observations
                        *actions[i].tolist(),  # Actions
                    ]
                    writer.writerow(row)

            print(f"✓ CSV saved: {csv_path} ({len(timestamps)} samples)")
            return csv_path

        except Exception as e:
            print(f"✗ Failed to save CSV {filename}: {e}")
            return None

    def plot_and_save(self):
        """Create plots and save to file."""
        if len(self.timestamps) < 10:
            return  # Need at least 10 samples to plot

        # Convert to numpy arrays
        t = np.array(self.timestamps)
        obs = np.array(self.obs_data)
        actions = np.array(self.action_data)

        if self.plot_counter == 0 and len(self.timestamps) > 50:
            # Skip initial transient samples for first plot
            skip_samples = 50
            t = t[skip_samples:] - t[skip_samples]  # Reset time to start from zero
            obs = obs[skip_samples:]
            actions = actions[skip_samples:]

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Define what to plot
        plot_configs = [
            # Row 1: Velocities
            {"idx": slice(0, 3), "title": "Linear Velocity (Body Frame)", "ylabel": "m/s", "labels": ["x", "y", "z"]},
            {
                "idx": slice(3, 6),
                "title": "Angular Velocity (Body Frame)",
                "ylabel": "rad/s",
                "labels": ["x", "y", "z"],
            },
            {"idx": slice(6, 9), "title": "Gravity Projection", "ylabel": "unit", "labels": ["x", "y", "z"]},
            # Row 2: Goal and Errors
            {"idx": slice(9, 12), "title": "Goal Position (Body Frame)", "ylabel": "m", "labels": ["x", "y", "z"]},
            {"idx": slice(12, 15), "title": "Angular Error", "ylabel": "rad", "labels": ["x", "y", "z"]},
            {"idx": slice(15, 19), "title": "Gimbal DOF", "ylabel": "rad", "labels": ["0", "1", "2", "3"]},
            # Row 3: Rotation vectors
            {
                "idx": slice(19, 25),
                "title": "Root Rotation Vector",
                "ylabel": "unit",
                "labels": ["0", "1", "2", "3", "4", "5"],
            },
            {
                "idx": slice(25, 31),
                "title": "Goal Rotation Vector",
                "ylabel": "unit",
                "labels": ["0", "1", "2", "3", "4", "5"],
            },
            {
                "idx": slice(31, 39),
                "title": "Last Action (in obs)",
                "ylabel": "unit",
                "labels": ["0", "1", "2", "3", "4", "5", "6", "7"],
            },
            # Row 4: Actions
            {
                "idx": slice(0, 4),
                "title": "Target Gimbal",
                "ylabel": "rad",
                "labels": ["0", "1", "2", "3"],
                "data": "action",
            },
            {
                "idx": slice(4, 8),
                "title": "Target Thrust",
                "ylabel": "N",
                "labels": ["0", "1", "2", "3"],
                "data": "action",
            },
            {"idx": None, "title": "Performance", "ylabel": "", "labels": []},  # Placeholder for text
        ]

        for i, config in enumerate(plot_configs):
            ax = fig.add_subplot(gs[i // 3, i % 3])

            if config["idx"] is None:
                # Performance metrics text
                ax.axis("off")
                mean_dt = np.mean(np.diff(t)) if len(t) > 1 else 0
                freq = 1.0 / mean_dt if mean_dt > 0 else 0
                info_text = (
                    f"Samples: {len(t)}, Duration: {t[-1]:.2f}s\nMean dt: {mean_dt*1000:.2f}ms, Freq:"
                    f" {freq:.1f}Hz\nMean vx: {np.mean(abs(obs[:,0])):.4f}, vy: {np.mean(abs(obs[:,1])):.4f}, vz:"
                    f" {np.mean(abs(obs[:,2])):.4f} m/s\nStd  vx: {np.std(obs[:,0]):.4f}, vy: {np.std(obs[:,1]):.4f},"
                    f" vz: {np.std(obs[:,2]):.4f} m/s\nMean wx: {np.mean(abs(obs[:,3])):.4f}, wy:"
                    f" {np.mean(abs(obs[:,4])):.4f}, wz: {np.mean(abs(obs[:,5])):.4f} rad/s\nStd  wx:"
                    f" {np.std(obs[:,3]):.4f}, wy: {np.std(obs[:,4]):.4f}, wz: {np.std(obs[:,5]):.4f} rad/s"
                )
                ax.text(-0.1, 0.5, info_text, fontsize=12, family="monospace", verticalalignment="center")
            else:
                # Plot data
                data = actions if config.get("data") == "action" else obs
                indices = config["idx"]

                for j, label in enumerate(config["labels"]):
                    ax.plot(t, data[:, indices.start + j], label=label, linewidth=1.5)

                ax.set_title(config["title"], fontsize=10, fontweight="bold")
                ax.set_xlabel("Time (s)", fontsize=9)
                ax.set_ylabel(config["ylabel"], fontsize=9)
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)

        # Add overall title
        fig.suptitle(
            f"Observation & Action Data - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            fontsize=14,
            fontweight="bold",
        )

        # Save figure
        filename = f"obs_plot_{self.plot_counter:04d}.png"
        save_path = self.save_dir / filename

        try:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Plot saved: {save_path}")
        except Exception as e:
            print(f"✗ Failed to save plot {filename}: {e}")
        finally:
            plt.close(fig)

        print(f"Plot saved: {filename}")
        self.plot_counter += 1

    def save_figure(self, saveDir: str, figName: str):
        """Save a given matplotlib figure to the save directory."""
        if len(self.timestamps) == 0:
            print("⚠ No data to save (timestamps empty). Skipping save_figure().")
            return
        # Convert to numpy arrays
        t = np.array(self.timestamps)
        t = t - t[0]
        obs = np.array(self.obs_data)
        actions = np.array(self.action_data)

        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        gs = GridSpec(4, 3, figure=fig, hspace=0.3, wspace=0.3)

        # Define what to plot
        plot_configs = [
            # Row 1: Velocities
            {"idx": slice(0, 3), "title": "Linear Velocity (Body Frame)", "ylabel": "m/s", "labels": ["x", "y", "z"]},
            {
                "idx": slice(3, 6),
                "title": "Angular Velocity (Body Frame)",
                "ylabel": "rad/s",
                "labels": ["x", "y", "z"],
            },
            {"idx": slice(6, 9), "title": "Gravity Projection", "ylabel": "unit", "labels": ["x", "y", "z"]},
            # Row 2: Goal and Errors
            {"idx": slice(9, 12), "title": "Goal Position (Body Frame)", "ylabel": "m", "labels": ["x", "y", "z"]},
            {"idx": slice(12, 15), "title": "Angular Error", "ylabel": "rad", "labels": ["x", "y", "z"]},
            {"idx": slice(15, 19), "title": "Gimbal DOF", "ylabel": "rad", "labels": ["0", "1", "2", "3"]},
            # Row 3: Rotation vectors
            {
                "idx": slice(19, 25),
                "title": "Root Rotation Vector",
                "ylabel": "unit",
                "labels": ["0", "1", "2", "3", "4", "5"],
            },
            {
                "idx": slice(25, 31),
                "title": "Goal Rotation Vector",
                "ylabel": "unit",
                "labels": ["0", "1", "2", "3", "4", "5"],
            },
            {
                "idx": slice(31, 39),
                "title": "Last Action (in obs)",
                "ylabel": "unit",
                "labels": ["0", "1", "2", "3", "4", "5", "6", "7"],
            },
            # Row 4: Actions
            {
                "idx": slice(0, 4),
                "title": "Target Gimbal",
                "ylabel": "rad",
                "labels": ["0", "1", "2", "3"],
                "data": "action",
            },
            {
                "idx": slice(4, 8),
                "title": "Target Thrust",
                "ylabel": "N",
                "labels": ["0", "1", "2", "3"],
                "data": "action",
            },
            {"idx": None, "title": "Performance", "ylabel": "", "labels": []},  # Placeholder for text
        ]

        for i, config in enumerate(plot_configs):
            ax = fig.add_subplot(gs[i // 3, i % 3])

            if config["idx"] is None:
                # Performance metrics text
                ax.axis("off")
                mean_dt = np.mean(np.diff(t)) if len(t) > 1 else 0
                freq = 1.0 / mean_dt if mean_dt > 0 else 0
                info_text = (
                    f"Samples: {len(t)}, Duration: {t[-1]:.2f}s\nMean dt: {mean_dt*1000:.2f}ms, Freq:"
                    f" {freq:.1f}Hz\nMean vx: {np.mean(abs(obs[:,0])):.4f}, vy: {np.mean(abs(obs[:,1])):.4f}, vz:"
                    f" {np.mean(abs(obs[:,2])):.4f} m/s\nStd  vx: {np.std(obs[:,0]):.4f}, vy: {np.std(obs[:,1]):.4f},"
                    f" vz: {np.std(obs[:,2]):.4f} m/s\nMean wx: {np.mean(abs(obs[:,3])):.4f}, wy:"
                    f" {np.mean(abs(obs[:,4])):.4f}, wz: {np.mean(abs(obs[:,5])):.4f} rad/s\nStd  wx:"
                    f" {np.std(obs[:,3]):.4f}, wy: {np.std(obs[:,4]):.4f}, wz: {np.std(obs[:,5]):.4f} rad/s"
                )
                ax.text(0.1, 0.5, info_text, fontsize=12, family="monospace", verticalalignment="center")
            else:
                # Plot data
                data = actions if config.get("data") == "action" else obs
                indices = config["idx"]

                for j, label in enumerate(config["labels"]):
                    ax.plot(t, data[:, indices.start + j], label=label, linewidth=1.5)

                ax.set_title(config["title"], fontsize=10, fontweight="bold")
                ax.set_xlabel("Time (s)", fontsize=9)
                ax.set_ylabel(config["ylabel"], fontsize=9)
                ax.legend(loc="upper right", fontsize=8)
                ax.grid(True, alpha=0.3)
                ax.tick_params(labelsize=8)

        # Add overall title
        fig.suptitle(
            f"Observation & Action Data - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            fontsize=14,
            fontweight="bold",
        )
        save_path = Path(saveDir) / figName
        try:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"✓ Figure saved: {save_path}")
        except Exception as e:
            print(f"✗ Failed to save figure {figName}: {e}")
        finally:
            plt.close()

    def save_final_plot(self):
        """Save final plot when shutting down."""
        self.plot_and_save()
        print(f"Final plot saved. Total plots: {self.plot_counter}")

    def shutdown(self):
        """Save all data and final plot before shutdown."""
        print("\n" + "=" * 30)
        print("Shutting down ObservationLogger...")
        print("=" * 30)

        # Save final plot
        # self.save_final_plot()

        # Save data to CSV
        # self.save_to_csv()

        print("=" * 30)
        print(f"✓ Shutdown complete. Files saved to: {self.save_dir}")
        print("=" * 30 + "\n")
