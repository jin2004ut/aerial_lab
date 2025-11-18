import numpy as np
import matplotlib.pyplot as plt


# ===========================
#  Quaternion utilities (NumPy)
# ===========================
def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """
    Compute quaternion conjugate.
    
    Args:
        q: quaternion (N, 4) or (4,) in [w, x, y, z] format
    
    Returns:
        conjugate quaternion [w, -x, -y, -z]
    """
    q = np.asarray(q)
    if q.ndim == 1:
        return np.array([q[0], -q[1], -q[2], -q[3]])
    else:
        conj = q.copy()
        conj[:, 1:] *= -1
        return conj


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """
    Quaternion multiplication (Hamilton product).
    
    Args:
        q1: quaternion (N, 4) or (4,) in [w, x, y, z] format
        q2: quaternion (N, 4) or (4,) in [w, x, y, z] format
    
    Returns:
        q1 * q2
    """
    q1 = np.asarray(q1)
    q2 = np.asarray(q2)
    
    # Handle single quaternion case
    if q1.ndim == 1 and q2.ndim == 1:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
            w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
            w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
            w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
        ])
    
    # Batch case
    w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,  # w
        w1*x2 + x1*w2 + y1*z2 - z1*y2,  # x
        w1*y2 - x1*z2 + y1*w2 + z1*x2,  # y
        w1*z2 + x1*y2 - y1*x2 + z1*w2   # z
    ], axis=-1)


def angle_distance(object_rot: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
    """
    Compute angular distance between two quaternions.
    
    Args:
        object_rot: current quaternion (N, 4) or (4,) in [w, x, y, z] format
        target_rot: target quaternion (N, 4) or (4,) in [w, x, y, z] format
    
    Returns:
        angular distance in radians (N,) or scalar
    """
    # Compute quaternion difference: q_diff = object_rot * conj(target_rot)
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    
    # Extract vector part [x, y, z]
    if quat_diff.ndim == 1:
        vec_part = quat_diff[1:4]
    else:
        vec_part = quat_diff[:, 1:4]
    
    # Compute norm of vector part
    vec_norm = np.linalg.norm(vec_part, axis=-1)
    
    # Clamp to avoid numerical issues with arcsin
    vec_norm_clamped = np.clip(vec_norm, 0.0, 1.0)
    
    # Angular distance: 2 * arcsin(||vec_part||)
    return 2.0 * np.arcsin(vec_norm_clamped)


def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion to rotation matrix.
    
    Args:
        q: quaternion (N, 4) or (4,) in [w, x, y, z] format
    
    Returns:
        rotation matrix (3, 3) or (N, 3, 3)
    """
    q = np.asarray(q)
    
    if q.ndim == 1:
        w, x, y, z = q
        R = np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [2*(x*z - y*w),         2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ])
        return R
    else:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        R = np.zeros((q.shape[0], 3, 3))
        R[:, 0, 0] = 1 - 2*(y*y + z*z)
        R[:, 0, 1] = 2*(x*y - z*w)
        R[:, 0, 2] = 2*(x*z + y*w)
        R[:, 1, 0] = 2*(x*y + z*w)
        R[:, 1, 1] = 1 - 2*(x*x + z*z)
        R[:, 1, 2] = 2*(y*z - x*w)
        R[:, 2, 0] = 2*(x*z - y*w)
        R[:, 2, 1] = 2*(y*z + x*w)
        R[:, 2, 2] = 1 - 2*(x*x + y*y)
        return R


def vee_map(skew_matrix: np.ndarray) -> np.ndarray:
    """
    Vee operator: Extract 3-vector from skew-symmetric matrix.
    
    For skew-symmetric matrix:
        [  0  -c   b ]
        [  c   0  -a ]
        [ -b   a   0 ]
    
    Returns vector [a, b, c]
    
    Args:
        skew_matrix: skew-symmetric matrix (3, 3) or (N, 3, 3)
    
    Returns:
        vector (3,) or (N, 3)
    """
    skew_matrix = np.asarray(skew_matrix)
    
    if skew_matrix.ndim == 2:
        # Single matrix case
        return np.array([
            skew_matrix[2, 1],   # a = -skew[1, 2]
            skew_matrix[0, 2],   # b = -skew[2, 0]
            skew_matrix[1, 0]    # c = -skew[0, 1]
        ])
    else:
        # Batch case
        return np.stack([
            skew_matrix[:, 2, 1],
            skew_matrix[:, 0, 2],
            skew_matrix[:, 1, 0]
        ], axis=-1)


def rotation_error_vector(R_actual: np.ndarray, R_desired: np.ndarray) -> np.ndarray:
    """
    Compute rotation error vector using the formula:
    e_R = vee(R_actual^T * R_desired - R_actual * R_desired^T) / 2
    
    This gives the axis-angle representation of the rotation error.
    
    Args:
        R_actual: current rotation matrix (3, 3) or (N, 3, 3)
        R_desired: desired rotation matrix (3, 3) or (N, 3, 3)
    
    Returns:
        error vector (3,) or (N, 3)
    """
    R_actual = np.asarray(R_actual)
    R_desired = np.asarray(R_desired)
    
    if R_actual.ndim == 2:
        # Single matrix case
        # R_actual^T * R_desired
        term1 = R_actual.T @ R_desired
        # R_actual * R_desired^T
        term2 = R_actual @ R_desired.T
        # Skew-symmetric matrix
        skew = term1 - term2
    else:
        # Batch case
        # R_actual^T * R_desired
        term1 = np.matmul(R_actual.transpose(0, 2, 1), R_desired)
        # R_actual * R_desired^T
        term2 = np.matmul(R_actual, R_desired.transpose(0, 2, 1))
        # Skew-symmetric matrix
        skew = term1 - term2
    
    # Extract vector using vee operator
    error_vec = vee_map(skew) / 2.0
    
    return error_vec


def rotation_distance(object_rot: np.ndarray, target_rot: np.ndarray) -> np.ndarray:
    """
    Compute rotation error vector between two quaternions using rotation matrices.
    
    Formula: e_R = vee(R_actual^T * R_desired - R_actual * R_desired^T) / 2
    
    Args:
        object_rot: current quaternion (N, 4) or (4,) in [w, x, y, z] format
        target_rot: target quaternion (N, 4) or (4,) in [w, x, y, z] format
    
    Returns:
        rotation error vector (N, 3) or (3,)
    """
    R_actual = matrix_from_quat(object_rot)
    R_desired = matrix_from_quat(target_rot)
    
    return rotation_error_vector(R_actual, R_desired)


def quat_from_rotation_matrix(R: np.ndarray) -> np.ndarray:
    """
    Convert rotation matrix to quaternion [w, x, y, z].
    
    Args:
        R: rotation matrix (3, 3)
    
    Returns:
        quaternion (4,)
    """
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])


# ===========================
#  SO(3) Log Map: rotation matrix -> rot_vec
# ===========================
def so3_log(R: np.ndarray) -> np.ndarray:
    """
    Compute log map of rotation matrix R ∈ SO(3)
    Return rotation vector (axis * angle)
    """
    # Ensure np array
    R = np.asarray(R)

    # Rotation angle
    trace_val = np.trace(R)
    cos_theta = (trace_val - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)

    # Very small rotation case
    if theta < 1e-5:
        return 0.5 * np.array([R[2, 1] - R[1, 2],
                               R[0, 2] - R[2, 0],
                               R[1, 0] - R[0, 1]])

    # Near pi: special handling
    if np.abs(theta - np.pi) < 1e-4:
        # Find rotation axis
        A = (R + np.eye(3)) / 2.0
        axis = np.sqrt(np.maximum(np.diag(A), 0.0))
        axis = axis / np.linalg.norm(axis)
        return axis * theta

    # General case
    w = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ]) / (2 * np.sin(theta))

    return w * theta


# ===========================
# Generate random rotation matrix
# ===========================
def random_rotation_matrix():
    q = np.random.randn(4)
    q = q / np.linalg.norm(q)
    w, x, y, z = q

    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [2*(x*z - y*w),         2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ])
    return R


# ===========================
# Test and visualize
# ===========================
if __name__ == "__main__":
    N = 2000
    angles_true = []
    angles_log = []
    angles_distance = []
    angles_rotation_error = []

    identity_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion

    for _ in range(N):
        R = random_rotation_matrix()

        # True angle from rotation matrix
        cos_theta = (np.trace(R) - 1.0) / 2.0
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        theta_true = np.arccos(cos_theta)

        # Log map angle
        rot_vec = so3_log(R)
        theta_log = np.linalg.norm(rot_vec)

        # Angle distance using quaternion
        q = quat_from_rotation_matrix(R)
        theta_quat = angle_distance(q, identity_quat)

        # Rotation error vector norm
        error_vec = rotation_distance(q, identity_quat)
        theta_error = np.linalg.norm(error_vec)

        angles_true.append(theta_true)
        angles_log.append(theta_log)
        angles_distance.append(theta_quat)
        angles_rotation_error.append(theta_error)

    # Print a few examples
    print("\n=== Sample Outputs (first 5) ===")
    for i in range(min(5, N)):
        print(f"\nSample {i+1}:")
        print(f"  True angle:     {angles_true[i]:.6f} rad ({np.degrees(angles_true[i]):.2f}°)")
        print(f"  Log map:        {angles_log[i]:.6f} rad")
        print(f"  Quat distance:  {angles_distance[i]:.6f} rad")
        print(f"  Rotation error: {angles_rotation_error[i]:.6f} rad")

    # ===========================
    # Plot comparison
    # ===========================
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Log map vs True angle
    axes[0].scatter(angles_true, angles_log, s=6, alpha=0.6, label='Log map')
    axes[0].plot([0, np.pi], [0, np.pi], 'r--', label='y=x')
    axes[0].set_xlabel("True rotation angle (rad)")
    axes[0].set_ylabel("Log map rotation vector norm (rad)")
    axes[0].set_title("SO(3) Log Map Verification")
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Quaternion distance vs True angle
    axes[1].scatter(angles_true, angles_distance, s=6, alpha=0.6, label='Quat distance', color='green')
    axes[1].plot([0, np.pi], [0, np.pi], 'r--', label='y=x')
    axes[1].set_xlabel("True rotation angle (rad)")
    axes[1].set_ylabel("Quaternion angular distance (rad)")
    axes[1].set_title("Quaternion Distance Verification")
    axes[1].legend()
    axes[1].grid(True)

    # Plot 3: Rotation error vector norm vs True angle
    axes[2].scatter(angles_true, angles_rotation_error, s=6, alpha=0.6, label='Rotation error', color='orange')
    axes[2].plot([0, np.pi], [0, np.pi], 'r--', label='y=x')
    axes[2].set_xlabel("True rotation angle (rad)")
    axes[2].set_ylabel("Rotation error vector norm (rad)")
    axes[2].set_title("Rotation Error Vector Verification")
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.show()

    # Print statistics
    print("\n=== Statistics ===")
    print(f"Log map error (mean): {np.mean(np.abs(np.array(angles_true) - np.array(angles_log))):.6f} rad")
    print(f"Quat distance error (mean): {np.mean(np.abs(np.array(angles_true) - np.array(angles_distance))):.6f} rad")
    print(f"Rotation error error (mean): {np.mean(np.abs(np.array(angles_true) - np.array(angles_rotation_error))):.6f} rad")
    print(f"Max true angle: {np.max(angles_true):.4f} rad ({np.degrees(np.max(angles_true)):.2f}°)")
