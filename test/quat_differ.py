from isaaclab.app import AppLauncher

# launch omniverse app in headless mode
simulation_app = AppLauncher(headless=True).app

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from isaaclab.utils.math import (
    compute_pose_error,
    matrix_from_quat,
    normalize,
    quat_apply,
    quat_conjugate,
    quat_error_magnitude,
    quat_from_angle_axis,
    quat_from_euler_xyz,
    quat_mul,
    sample_uniform,
    subtract_frame_transforms,
)


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    """
    Compute angular distance between two quaternions.

    Args:
        object_rot: quaternion (N, 4) in [w, x, y, z] format
        target_rot: quaternion (N, 4) in [w, x, y, z] format

    Returns:
        angular distance in radians (N,)
    """
    # Orientation alignment for the cube in hand and goal cube
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention


def rotation_error_vector(R_actual: torch.Tensor, R_desired: torch.Tensor) -> torch.Tensor:
    """
    Compute rotation error vector using the formula:
    e_R = vee(R_actual^T * R_desired - R_actual * R_desired^T) / 2

    Args:
        R_actual: current rotation matrix (N, 3, 3)
        R_desired: desired rotation matrix (N, 3, 3)

    Returns:
        error vector (N, 3)
    """
    # R_actual^T * R_desired
    term1 = torch.matmul(R_actual.transpose(-2, -1), R_desired)
    # R_actual * R_desired^T
    term2 = torch.matmul(R_actual, R_desired.transpose(-2, -1))
    # Skew-symmetric matrix
    skew = term1 - term2

    # Extract vector using vee operator
    # For skew matrix [[0, -c, b], [c, 0, -a], [-b, a, 0]]
    # Returns [a, b, c] = [skew[2,1], skew[0,2], skew[1,0]]
    error_vec = torch.stack([skew[:, 2, 1], skew[:, 0, 2], skew[:, 1, 0]], dim=-1) / 2.0

    return error_vec


def generate_random_quaternions(N: int, device: str = "cpu") -> torch.Tensor:
    """Generate N random unit quaternions."""
    q = torch.randn(N, 4, device=device)
    return normalize(q)


def quat_to_rotation_angle(quat: torch.Tensor) -> torch.Tensor:
    """
    Extract rotation angle from quaternion.

    Args:
        quat: quaternion (N, 4) in [w, x, y, z] format

    Returns:
        rotation angle in radians (N,)
    """
    # angle = 2 * arccos(w), clamped to avoid numerical issues
    w = quat[:, 0]
    angle = 2.0 * torch.acos(torch.clamp(w, -1.0, 1.0))
    return angle


def plot_quaternion_error_comparison(save_dir: str = "plots"):
    """
    Generate and plot comparison of different quaternion error metrics.

    Generates N random quaternion pairs and compares:
    1. True rotation angle (from rotation matrix)
    2. Rotation distance (using quaternion difference)
    3. Axis-angle error norm (from compute_pose_error)
    4. Rotation error vector norm (from matrix formula)
    """
    print("\n" + "=" * 70)
    print("Quaternion Error Metrics Comparison")
    print("=" * 70)

    # Generate test data
    N = 2000
    print(f"\nGenerating {N} random quaternion pairs...")

    # Identity quaternion and position (for reference)
    identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device="cpu").repeat(N, 1)
    zero_pos = torch.zeros(N, 3, device="cpu")

    # Generate random quaternions (representing rotations from identity)
    random_quats = generate_random_quaternions(N, device="cpu")

    # Compute true rotation angles
    print("Computing true rotation angles...")
    true_angles = quat_to_rotation_angle(random_quats)

    # Method 1: Rotation distance
    print("Computing rotation distance...")
    rot_distance = rotation_distance(random_quats, identity_quat)

    # Method 2: Axis-angle error from compute_pose_error
    print("Computing axis-angle error...")
    _, axis_angle_error = compute_pose_error(zero_pos, random_quats, zero_pos, identity_quat)
    axis_angle_norm = torch.linalg.norm(axis_angle_error, dim=1)

    # Method 3: Rotation error vector from matrix formula
    print("Computing rotation error vector...")
    R_actual = matrix_from_quat(random_quats)
    R_identity = matrix_from_quat(identity_quat)
    rot_error_vec = rotation_error_vector(R_actual, R_identity)
    rot_error_norm = torch.linalg.norm(rot_error_vec, dim=1)

    # Convert to numpy for plotting
    true_angles_np = true_angles.numpy()
    rot_distance_np = rot_distance.numpy()
    axis_angle_norm_np = axis_angle_norm.numpy()
    rot_error_norm_np = rot_error_norm.numpy()

    # Create plots
    print("\nGenerating plots...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # Plot 1: Rotation distance vs True angle
    axes[0, 0].scatter(true_angles_np, rot_distance_np, s=4, alpha=0.6, label="Rotation distance", color="blue")
    axes[0, 0].plot([0, np.pi], [0, np.pi], "r--", linewidth=2, label="y=x (ideal)")
    axes[0, 0].set_xlabel("True Rotation Angle (rad)", fontsize=12)
    axes[0, 0].set_ylabel("Rotation Distance (rad)", fontsize=12)
    axes[0, 0].set_title(
        "Method 1: Rotation Distance\n(2 * arcsin(||quat_diff[xyz]||))", fontsize=13, fontweight="bold"
    )
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([0, np.pi])
    axes[0, 0].set_ylim([0, np.pi])

    # Plot 2: Axis-angle error vs True angle
    axes[0, 1].scatter(true_angles_np, axis_angle_norm_np, s=4, alpha=0.6, label="Axis-angle error", color="green")
    axes[0, 1].plot([0, np.pi], [0, np.pi], "r--", linewidth=2, label="y=x (ideal)")
    axes[0, 1].set_xlabel("True Rotation Angle (rad)", fontsize=12)
    axes[0, 1].set_ylabel("Axis-Angle Error Norm (rad)", fontsize=12)
    axes[0, 1].set_title("Method 2: Axis-Angle Error\n(from compute_pose_error)", fontsize=13, fontweight="bold")
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([0, np.pi])
    axes[0, 1].set_ylim([0, np.pi])

    # Plot 3: Rotation error vector vs True angle
    axes[1, 0].scatter(true_angles_np, rot_error_norm_np, s=4, alpha=0.6, label="Rotation error vector", color="orange")
    axes[1, 0].plot([0, np.pi], [0, np.pi], "r--", linewidth=2, label="y=x (ideal)")
    axes[1, 0].set_xlabel("True Rotation Angle (rad)", fontsize=12)
    axes[1, 0].set_ylabel("Rotation Error Vector Norm (rad)", fontsize=12)
    axes[1, 0].set_title("Method 3: Rotation Error Vector\n(vee(R^T*R_d - R*R_d^T)/2)", fontsize=13, fontweight="bold")
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].set_xlim([0, np.pi])
    axes[1, 0].set_ylim([0, np.pi])

    # Plot 4: Error comparison (all methods)
    axes[1, 1].scatter(true_angles_np, rot_distance_np, s=3, alpha=0.5, label="Rotation distance", color="blue")
    axes[1, 1].scatter(true_angles_np, axis_angle_norm_np, s=3, alpha=0.5, label="Axis-angle error", color="green")
    axes[1, 1].scatter(true_angles_np, rot_error_norm_np, s=3, alpha=0.5, label="Rotation error vec", color="orange")
    axes[1, 1].plot([0, np.pi], [0, np.pi], "r--", linewidth=2, label="y=x (ideal)")
    axes[1, 1].set_xlabel("True Rotation Angle (rad)", fontsize=12)
    axes[1, 1].set_ylabel("Computed Error (rad)", fontsize=12)
    axes[1, 1].set_title("All Methods Comparison", fontsize=13, fontweight="bold")
    axes[1, 1].legend(fontsize=10)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].set_xlim([0, np.pi])
    axes[1, 1].set_ylim([0, np.pi])

    plt.tight_layout()

    # Save plot
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    plot_file = save_path / "quaternion_error_comparison.png"
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    print(f"\n✓ Plot saved to: {plot_file}")

    # Show plot
    plt.show()

    # Print statistics
    print("\n" + "=" * 70)
    print("Statistics Summary")
    print("=" * 70)

    error_rot_dist = np.abs(true_angles_np - rot_distance_np)
    error_axis_angle = np.abs(true_angles_np - axis_angle_norm_np)
    error_rot_vec = np.abs(true_angles_np - rot_error_norm_np)

    print(f"\nRotation Distance:")
    print(f"  Mean error:   {np.mean(error_rot_dist):.6f} rad ({np.degrees(np.mean(error_rot_dist)):.4f}°)")
    print(f"  Max error:    {np.max(error_rot_dist):.6f} rad ({np.degrees(np.max(error_rot_dist)):.4f}°)")
    print(f"  Std error:    {np.std(error_rot_dist):.6f} rad ({np.degrees(np.std(error_rot_dist)):.4f}°)")

    print(f"\nAxis-Angle Error:")
    print(f"  Mean error:   {np.mean(error_axis_angle):.6f} rad ({np.degrees(np.mean(error_axis_angle)):.4f}°)")
    print(f"  Max error:    {np.max(error_axis_angle):.6f} rad ({np.degrees(np.max(error_axis_angle)):.4f}°)")
    print(f"  Std error:    {np.std(error_axis_angle):.6f} rad ({np.degrees(np.std(error_axis_angle)):.4f}°)")

    print(f"\nRotation Error Vector:")
    print(f"  Mean error:   {np.mean(error_rot_vec):.6f} rad ({np.degrees(np.mean(error_rot_vec)):.4f}°)")
    print(f"  Max error:    {np.max(error_rot_vec):.6f} rad ({np.degrees(np.max(error_rot_vec)):.4f}°)")
    print(f"  Std error:    {np.std(error_rot_vec):.6f} rad ({np.degrees(np.std(error_rot_vec)):.4f}°)")

    print(f"\nTrue angle range: [{np.min(true_angles_np):.4f}, {np.max(true_angles_np):.4f}] rad")
    print(f"                  [{np.degrees(np.min(true_angles_np)):.2f}°, {np.degrees(np.max(true_angles_np)):.2f}°]")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    # Run visualization
    plot_quaternion_error_comparison(save_dir="plots")

    # Original test code (for quick verification)
    print("\n" + "=" * 70)
    print("Quick Verification Test")
    print("=" * 70)

    quat_1 = normalize(torch.randn(2, 4))
    quat_2 = normalize(torch.randn(2, 4))
    pos_1 = torch.randn(2, 3)
    pos_2 = torch.randn(2, 3)

    dist = rotation_distance(quat_1, quat_2)
    print("\nRotation distance:", dist)

    pose_err, axis_angle_error = compute_pose_error(
        pos_1,
        quat_1,
        pos_2,
        quat_2,
    )
    print("Axis angle error norm:", torch.linalg.norm(axis_angle_error, dim=1))
    print("Axis angle error:", axis_angle_error)

    R1 = matrix_from_quat(quat_1)
    R2 = matrix_from_quat(quat_2)
    rot_err_vec = rotation_error_vector(R1, R2)
    print("Rotation error vector:", rot_err_vec)
    print("Rotation error vector norm:", torch.linalg.norm(rot_err_vec, dim=1))
    print("=" * 70)
