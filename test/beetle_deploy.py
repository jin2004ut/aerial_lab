#!/usr/bin/env python3
"""
Beetle Policy Deployment with ONNXRuntime in ROS Noetic
Features:
- Multi-threaded topic subscription (IMU, etc.)
- High-precision control loop with threading.Timer
- ONNX policy inference
"""

import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import onnxruntime as ort
import rospy
from geometry_msgs.msg import Vector3Stamped
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray

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
        
        # Thread-safe data containers
        self.data_lock = threading.Lock()
        self.imu_data = None
        self.last_action = np.zeros(8, dtype=np.float32)
        
        # History buffers (for temporal features if needed)
        self.imu_history = deque(maxlen=10)
        
        # Load ONNX model
        self.session = self._load_onnx_model(onnx_path)
        self.obs_size, self.action_size = self._get_io_shapes()
        
        # ROS subscribers (run in separate threads)
        self.imu_sub = rospy.Subscriber(
            "/imu/data", Imu, self._imu_callback, queue_size=1
        )
        
        # ROS publishers
        self.action_pub = rospy.Publisher(
            "/beetle/action", Float32MultiArray, queue_size=1
        )
        
        # 启动 ROS 定时器（更 Pythonic）
        self.control_timer = rospy.Timer(
            rospy.Duration(1.0 / control_freq),
            self._control_loop_callback
        )

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
        """IMU data callback (runs in subscriber thread)."""
        with self.data_lock:
            self.imu_data = {
                "lin_acc": np.array([
                    msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z
                ], dtype=np.float32),
                "ang_vel": np.array([
                    msg.angular_velocity.x,
                    msg.angular_velocity.y,
                    msg.angular_velocity.z
                ], dtype=np.float32),
                "orientation": np.array([
                    msg.orientation.w,
                    msg.orientation.x,
                    msg.orientation.y,
                    msg.orientation.z
                ], dtype=np.float32),
            }
            self.imu_history.append(self.imu_data.copy())

    def _build_observation(self) -> np.ndarray:
        """
        Construct observation vector from sensor data.
        
        Returns:
            obs: (obs_size,) numpy array
        """
        with self.data_lock:
            if self.imu_data is None:
                rospy.logwarn("No IMU data received, using zeros")
                return np.zeros(self.obs_size, dtype=np.float32)
            
            # Example observation construction (adjust to match training config)
            obs = np.concatenate([
                self.imu_data["lin_acc"],           # (3,)
                self.imu_data["ang_vel"],           # (3,)
                self.imu_data["orientation"],       # (4,)
                self.last_action,                   # (8,)
                # Add more features as needed
            ])
            
            # Pad or truncate to match expected size
            if len(obs) < self.obs_size:
                obs = np.pad(obs, (0, self.obs_size - len(obs)))
            elif len(obs) > self.obs_size:
                obs = obs[:self.obs_size]
            
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

    def _publish_action(self, action: np.ndarray):
        """Publish action to ROS topic."""
        msg = Float32MultiArray()
        msg.data = action.tolist()
        self.action_pub.publish(msg)

    def _control_loop_callback(self, event):
        """定时器回调（自动触发，无需手动递归）"""
        start = rospy.Time.now()

        obs = self._build_observation()
        action = self._infer_action(obs)
        self._publish_action(action)

        with self.data_lock:
            self.last_action = action

        # 记录性能
        elapsed = (rospy.Time.now() - start).to_sec()
        if elapsed > self.control_dt:
            rospy.logwarn_throttle(1.0, f"Overrun: {elapsed*1000:.1f}ms")

    def spin(self):
        rospy.spin()  # ROS Timer 会自动触发，无需额外逻辑


def main():
    """Main entry point."""
    # Configuration
    onnx_path = rospy.get_param("~onnx_path", "/path/to/policy.onnx")
    control_freq = rospy.get_param("~control_freq", 50.0)

    try:
        deployer = PolicyDeployer(onnx_path, control_freq)
        deployer.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        if 'deployer' in locals():
            deployer.stop()


if __name__ == "__main__":
    main()
