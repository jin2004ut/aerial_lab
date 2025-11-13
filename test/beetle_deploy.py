#!/usr/bin/env python3
"""
Beetle Policy Deployment with ONNXRuntime in ROS Noetic

Features:
- Thread-safe ROS topic subscriptions (IMU, Odometry, Gimbal)
- High-precision control loop with rospy.Timer
- ONNX policy inference for real-time control
- Single lock design for efficient data synchronization
"""

import threading
import time
from collections import deque
from pathlib import Path
from typing import Tuple

import numpy as np
import onnxruntime as ort
import rospy
import tf.transformations as tf
from geometry_msgs.msg import Point, Pose, PoseStamped, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import JointState

# # # # spinal.msg::FourAxisCommand
# float32[3] angles
# float32[] base_thrust
from spinal.msg import FourAxisCommand, Imu
from std_msgs.msg import UInt8
from tf2_geometry_msgs import (
    do_transform_point,
    do_transform_pose,
    do_transform_vector3,
)

# 相对路径辅助：基于当前文件位置构造相对路径
# 使用示例：rel_path("models", "policy.onnx") -> 返回脚本目录下 models/policy.onnx 的绝对路径
HERE = Path(__file__).resolve().parent


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by quaternion q."""
    q_mat = tf.quaternion_matrix(q)
    v_rotated = q_mat[:3, :3] @ v
    return v_rotated


def quat_apply_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vector v by the inverse of quaternion q."""
    q_inv = tf.quaternion_inverse(q)
    q_mat = tf.quaternion_matrix(q_inv)
    v_rotated = q_mat[:3, :3] @ v
    return v_rotated


def quat_inv(q: np.ndarray) -> np.ndarray:
    """Return the inverse of quaternion q."""
    q_inv = tf.quaternion_inverse(q)
    return q_inv


def axis_angle_from_quat(quat: np.ndarray, eps: float = 1.0e-6) -> np.ndarray:
    """Convert quaternion to axis-angle representation"""
    # Modified to take in quat as [q_w, q_x, q_y, q_z]
    # Quaternion is [q_w, q_x, q_y, q_z] = [cos(theta/2), n_x * sin(theta/2), n_y * sin(theta/2), n_z * sin(theta/2)]
    # Axis-angle is [a_x, a_y, a_z] = [theta * n_x, theta * n_y, theta * n_z]

    # Ensure quaternion has positive w (choose shorter rotation path)
    if quat[0] < 0.0:
        quat = -quat

    # Calculate magnitude of imaginary part and half angle
    mag = np.linalg.norm(quat[1:])  # norm of [x, y, z]
    half_angle = np.arctan2(mag, quat[0])
    angle = 2.0 * half_angle

    # Check whether to apply Taylor approximation
    # When angle is small, sin(half_angle) / angle ≈ 1/2 - angle^2 / 48
    if np.abs(angle) > eps:
        sin_half_angles_over_angles = np.sin(half_angle) / angle
    else:
        sin_half_angles_over_angles = 0.5 - angle * angle / 48

    # Return axis-angle: [x, y, z] / (sin(half_angle) / angle)
    return quat[1:4] / sin_half_angles_over_angles


def subtract_frame_transforms(
    t01: np.ndarray, q01: np.ndarray, t02: np.ndarray, q02: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    # compute orientation
    q10 = quat_inv(q01)
    if q02 is not None:
        q12 = tf.quaternion_multiply(q10, q02)
    else:
        q12 = q10
    # compute translation
    if t02 is not None:
        t12 = quat_apply(q10, t02 - t01)
    else:
        t12 = quat_apply(q10, -t01)
    return t12, q12


def compute_pose_error(
    t01: np.ndarray,
    q01: np.ndarray,
    t02: np.ndarray,
    q02: np.ndarray,
    rot_error_type: str = "axis_angle",
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the position and orientation error between source and target frames"""
    # For unit quaternions, the inverse is the conjugate
    # q_inv = conj(q) = [w, -x, -y, -z]
    source_quat_inv = tf.quaternion_conjugate(q01)

    # Quaternion error: q_error = q_target * q_current_inv
    quat_error = tf.quaternion_multiply(q02, source_quat_inv)

    # Position error in world frame
    pos_error = t02 - t01

    # Return based on specified type
    if rot_error_type == "quat":
        return pos_error, quat_error
    elif rot_error_type == "axis_angle":
        # Convert to axis-angle error
        axis_angle_error = axis_angle_from_quat(quat_error)
        return pos_error, axis_angle_error
    else:
        raise ValueError(f"Unsupported orientation error type: {rot_error_type}. Valid: 'quat', 'axis_angle'.")


def rel_path(*parts) -> str:
    """Return an absolute path by joining parts relative to this script's directory."""
    return str(HERE.joinpath(*parts).resolve())


# # # # Usage:
# 1. Launch ROS core and necessary nodes to publish sensor data (IMU, etc.)
#    roslaunch gimbalrotor bringup.launch rm:=False sim:=True headless:=False
# 2. Run this script with the path to the ONNX model as a ROS parameter
#    e.g., rosrun your_package beetle_deploy.py _onnx_path:=/path/to/policy.onnx
# 3. set the desired Goal Pose by
#    python3 int
# 4. The script will subscribe to sensor topics, run inference, and publish actions
# # # #


class PolicyDeployer:
    def __init__(self, onnx_path: str, control_freq: float = 50.0):
        """
        Initialize the policy deployer.

        Args:
            onnx_path: Path to ONNX model file
            control_freq: Control loop frequency in Hz
        """
        rospy.init_node("beetle_policy_node", anonymous=True)

        # Control parameters
        self.control_dt = 1.0 / control_freq
        self.control_freq = control_freq

        class Scales:
            lin_vel = 0.2
            ang_vel = 0.4
            obs_clip = 100.0
            action_clip = 100.0
            gimbal_action_scale = 0.25
            thrust_action_scale = 1.25

        self.scales = Scales()

        # self.scales = class{
        #     'lin_vel': 0.2,
        #     'ang_vel': 0.4,
        #     'obs_clip': 100.0,
        #     'action_clip': 1.0,
        #     'gimbal_action_scale': 0.25,
        #     'thrust_action_scale': 1.25
        # }

        # Thread-safe data containers - use a single lock for all sensor data
        self.data_lock = threading.Lock()

        # Initialize sensor data structures
        self.imu_data = Imu()  # Will store dict with processed IMU data
        self.imu_catch = False
        self.odom_data = Odometry()  # Store odometry data
        self.odom_catch = False
        # self.gimbal_data = UInt8()  # Store gimbal state
        self.gimbal_data = JointState()  # Store gimbal state
        self.gimbal_catch = False

        self.desired_pose = PoseStamped(
            header=rospy.Header(frame_id="world"),
            pose=Pose(position=Point(x=0.0, y=0.0, z=1.2), orientation=Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)),
        )

        self.gimbal_pos = np.zeros(4, dtype=np.float32)
        self.body_quat = np.zeros(4, dtype=np.float32)
        self.body_pos = np.zeros(3, dtype=np.float32)
        # self.gimbal_pos_scale = 0.001533981
        self.gimbal_default_pos = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.thrust_default = 7.0

        # Load ONNX model
        self.session = self._load_onnx_model(onnx_path)
        self.obs_size, self.action_size = self._get_io_shapes()
        self.obs = np.zeros(self.obs_size, dtype=np.float32)
        self.action = np.zeros(self.action_size, dtype=np.float32)
        self.last_action = np.zeros(self.action_size, dtype=np.float32)
        rospy.loginfo(f"Observation size: {self.obs_size}, Action size: {self.action_size}")

        # ROS subscribers - they run in ROS's own thread pool, no need for separate threads
        self.imu_sub = rospy.Subscriber("/gimbalrotor/imu", Imu, self._imu_callback, queue_size=1)

        self.odom_sub = rospy.Subscriber(
            "/gimbalrotor/ground_truth",
            Odometry,
            self._odom_callback,
            queue_size=1,
            # "/gimbalrotor/uav/cog/odom", Odometry, self._odom_callback, queue_size=1
        )

        # self.gimbal_sub = rospy.Subscriber(
        #     "/gimbalrotor/gimbal_dof", UInt8, self._gimbal_callback, queue_size=1
        # )
        self.gimbal_sub = rospy.Subscriber("/gimbalrotor/joint_states", JointState, self._gimbal_callback, queue_size=1)

        self.goal_pose_sub = rospy.Subscriber(
            "/gimbalrotor/goal_pose", PoseStamped, self._goal_pose_callback, queue_size=1
        )

        # ROS publishers
        self.thrust_pub = rospy.Publisher("/gimbalrotor/four_axes/command", FourAxisCommand, queue_size=1)

        self.gimbal_pub = rospy.Publisher("/gimbalrotor/gimbals_ctrl", JointState, queue_size=1)

        # Start ROS timer for control loop
        self.control_timer = rospy.Timer(rospy.Duration(1.0 / control_freq), self._control_loop_callback)

        rospy.loginfo("Policy Deployer initialized")
        rospy.loginfo(f"Control frequency: {control_freq} Hz")
        rospy.loginfo(f"Observation size: {self.obs_size}, Action size: {self.action_size}")

    def _load_onnx_model(self, onnx_path: str) -> ort.InferenceSession:
        """Load ONNX model with optimized settings."""
        if not Path(onnx_path).exists():
            raise FileNotFoundError(f"ONNX model not found: {onnx_path}")

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 2  # Adjust based on CPU cores

        providers = ["CPUExecutionProvider"]  # Use CUDA if available
        session = ort.InferenceSession(onnx_path, sess_options, providers=providers)

        rospy.loginfo(f"ONNX model loaded from {onnx_path}")
        return session

    def _get_io_shapes(self) -> tuple[int, int]:
        """Get input/output dimensions from ONNX model."""
        input_shape = self.session.get_inputs()[0].shape
        output_shape = self.session.get_outputs()[0].shape

        # Handle dynamic batch dimension
        obs_size = input_shape[1] if len(input_shape) > 1 else input_shape[0]
        action_size = output_shape[1] if len(output_shape) > 1 else output_shape[0]

        return obs_size, action_size

    def _imu_callback(self, msg: Imu):
        """IMU data callback - processes and stores IMU data with thread safety."""
        with self.data_lock:
            self.imu_data = msg
            self.imu_catch = True

    def _odom_callback(self, msg: Odometry):
        """Odometry data callback - stores odometry data with thread safety."""
        with self.data_lock:
            self.odom_data = msg
            self.odom_catch = True

    # def _gimbal_callback(self, msg: UInt8):
    #     """Gimbal state callback - stores gimbal data with thread safety."""
    #     with self.data_lock:
    #         self.gimbal_data = msg
    #         for i in range(msg.size()):
    #             self.gimbal_pos[i] = (msg.position[i] - self.gimbal_default_pos[i]) * self.gimbal_pos_scale
    #         self.gimbal_catch = True

    def _gimbal_callback(self, msg: JointState):
        """Gimbal state callback - stores gimbal data with thread safety."""
        with self.data_lock:
            self.gimbal_data = msg
            for i in range(len(msg.name)):
                self.gimbal_pos[i] = msg.position[i]
            self.gimbal_catch = True

    def _goal_pose_callback(self, msg: PoseStamped):
        """Goal pose callback - stores desired goal pose with thread safety."""
        with self.data_lock:
            self.desired_pose = msg
        print("============ Received new goal pose ============ ")
        print("Position:", msg.pose.position)
        print("Angle:", tf.euler_from_quaternion(msg.pose.orientation, "szyx"))

    def _build_observation(self) -> np.ndarray:
        """
        Construct observation vector from sensor data.

        Returns:
            obs: (obs_size,) numpy array
        """
        with self.data_lock:
            if self.imu_catch is False:
                print("Warning: No IMU data received yet, useing zeros")
                return None
            else:
                obs_project_gravity_vec = np.array(
                    [self.imu_data.acc[0], self.imu_data.acc[1], self.imu_data.acc[2]]
                ).astype(np.float32)
            if self.odom_catch is False:
                print("Warning: No Odometry data received yet, using zeros")
                return None
            else:
                self.body_quat = np.array([
                    self.odom_data.pose.pose.orientation.x,
                    self.odom_data.pose.pose.orientation.y,
                    self.odom_data.pose.pose.orientation.z,
                    self.odom_data.pose.pose.orientation.w,
                ]).astype(np.float32)
                self.body_pos = np.array([
                    self.odom_data.pose.pose.position.x,
                    self.odom_data.pose.pose.position.y,
                    self.odom_data.pose.pose.position.z,
                ]).astype(np.float32)
                obs_pos = self.body_pos
                obs_quat = self.body_quat
                obs_lin_vel_b = np.array([
                    self.odom_data.twist.twist.linear.x,
                    self.odom_data.twist.twist.linear.y,
                    self.odom_data.twist.twist.linear.z,
                ]).astype(np.float32)
                obs_lin_vel_b = quat_apply_inv(obs_quat, obs_lin_vel_b)
                obs_ang_vel_b = np.array([
                    self.odom_data.twist.twist.angular.x,
                    self.odom_data.twist.twist.angular.y,
                    self.odom_data.twist.twist.angular.z,
                ]).astype(np.float32)
                obs_ang_vel_b = quat_apply_inv(obs_quat, obs_ang_vel_b)  # Should rotate by quaternion, not position
            if self.gimbal_catch is False:
                print("Warning: No Gimbal data received yet, using zeros")
                return None
            else:
                obs_gimbal_dof = self.gimbal_pos.copy()  # Already (4,) shape

            # Example observation construction (adjust to match training config)
            # obs = torch.cat(
            #     (
            #         self._robot.data.root_lin_vel_b * self.obsScales.lin_vel,
            #         self._robot.data.root_ang_vel_b * self.obsScales.ang_vel,
            #         self._robot.data.projected_gravity_b,
            #         goal_pos_b,
            #         angular_error,
            #         self._robot.data.joint_pos[:, self._gimbal_ids[0]] - self._gimbal_default_pos,
            #         root_rot_vec,
            #         goal_rot_vec,
            #         self._last_actions,
            #     ),
            #     dim=-1,
            # )
            goal_pos_b, angular_error = compute_pose_error(
                t01=obs_pos,
                q01=obs_quat,
                t02=np.array([
                    self.desired_pose.pose.position.x,
                    self.desired_pose.pose.position.y,
                    self.desired_pose.pose.position.z,
                ]).astype(np.float32),
                q02=np.array([
                    self.desired_pose.pose.orientation.x,
                    self.desired_pose.pose.orientation.y,
                    self.desired_pose.pose.orientation.z,
                    self.desired_pose.pose.orientation.w,
                ]).astype(np.float32),
            )

            root_rot_vec = tf.quaternion_matrix(obs_quat)[:3, :2]  # Fixed: should be [:3, :2] for 3x2 = 6 elements
            goal_rot_vec = tf.quaternion_matrix(
                np.array([
                    self.desired_pose.pose.orientation.x,
                    self.desired_pose.pose.orientation.y,
                    self.desired_pose.pose.orientation.z,
                    self.desired_pose.pose.orientation.w,
                ]).astype(np.float32)
            )[
                :3, :2
            ]  # Fixed: should be [:3, :2]

            obs = np.concatenate([
                obs_lin_vel_b * self.scales.lin_vel,  # (3,) 3
                obs_ang_vel_b * self.scales.ang_vel,  # (3,) 6
                obs_project_gravity_vec,  # (3,) 9
                goal_pos_b,  # (3,) 12
                angular_error,  # (3,) 15
                obs_gimbal_dof,  # (4,) 19
                root_rot_vec.flatten(),  # (6,) 25
                goal_rot_vec.flatten(),  # (6,) 31
                self.last_action,  # (8,) 39
                # Add more features as needed
            ])

            # Pad or truncate to match expected size
            if len(obs) != self.obs_size:
                print(f"Warning: Observation size mismatch. Expected {self.obs_size}, got {len(obs)}")
                return None

            return obs.astype(np.float32)

    def _infer_action(self, obs: np.ndarray) -> np.ndarray:
        """
        Run ONNX inference to get action.

        Args:
            obs: (obs_size,) observation vector

        Returns:
            action: (action_size,) action vector
        """
        # Reshape to (1, obs_size) for batch inference
        obs_batch = obs.reshape(1, -1)

        # Run inference
        input_name = self.session.get_inputs()[0].name
        output_name = self.session.get_outputs()[0].name

        action = self.session.run([output_name], {input_name: obs_batch})[0]

        return action.flatten()

    def _publish_thrust(self, target_thrust: np.ndarray):
        """
        Publish action to ROS topic.

        Args:
            action: (action_size,) action vector
                    Expected format: [angle1, angle2, angle3, base_thrust, gimbal_states...]
        """
        # Parse action vector according to your robot's control interface
        # Adjust indices based on your action space design

        # Publish four-axis command (roll, pitch, yaw angles + thrust)
        cmd_msg = FourAxisCommand()
        target_thrust = target_thrust * self.scales.thrust_action_scale + self.thrust_default
        cmd_msg.angles = [0.0, 0.0, 0.0]  # [roll, pitch, yaw] angles
        cmd_msg.base_thrust = target_thrust.tolist()  # base thrust
        self.thrust_pub.publish(cmd_msg)

    def _publish_gimbal(self, target_pos: np.ndarray):
        """
        Publish gimbal target positions to ROS topic.

        Args:
            target_pos: (num_gimbals,) target gimbal positions
        """
        gimbal_msg = JointState()
        gimbal_msg.header.stamp = rospy.Time.now()
        for i in range(len(target_pos)):
            gimbal_msg.name.append(f"gimbal_joint_{i+1}")
            gimbal_msg.position.append(target_pos[i] * self.scales.gimbal_action_scale + self.gimbal_default_pos[i])
        self.gimbal_pub.publish(gimbal_msg)

    def _control_loop_callback(self, event):
        """Control loop timer callback - runs at specified frequency."""
        start = rospy.Time.now()

        obs = self._build_observation()

        # Check if observation is valid
        if obs is None:
            rospy.logwarn_throttle(5.0, "Waiting for sensor data...")
            return

        action = self._infer_action(obs)

        target_gimbal = action[:4]
        target_thrust = action[4:]

        self._publish_thrust(target_thrust)
        self._publish_gimbal(target_gimbal)

        with self.data_lock:
            self.last_action = action

        # Log performance metrics
        elapsed = (rospy.Time.now() - start).to_sec()
        if elapsed > self.control_dt:
            rospy.logwarn_throttle(1.0, f"Control loop overrun: {elapsed * 1000:.1f}ms")

    def spin(self):
        """Keep the node running - ROS Timer will automatically trigger callbacks."""
        rospy.spin()


def main():
    """Main entry point."""
    # Get configuration from ROS parameters
    # onnx_path = rospy.get_param("~onnx_path", rel_path("models", "policy.onnx"))
    onnx_path = rel_path("policy", "policy.onnx")
    control_freq = 50.0

    try:
        deployer = PolicyDeployer(onnx_path, control_freq)
        deployer.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("ROS interrupt received, shutting down")
    except Exception as e:
        rospy.logerr(f"Error in policy deployer: {e}")
        raise


if __name__ == "__main__":
    main()
