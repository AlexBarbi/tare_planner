#!/usr/bin/env python3
"""
lio_base_odometry.py (ROS 2 Python)

Republishes FAST-LIO's /Odometry, which is the pose of the lidar (rslidar_lio in camera_init), as the pose of the
robot's base (lio_base, same frame and stamp) on /lio_base_odometry. explore_warehouse_launch.xml points TARE
(sub_state_estimation_topic_) and tare_finish_merge_map.py at it.

Why: the goal bridge and the controller place the robot by its base (/odom), TARE placed it by the lidar, 0.29 m ahead
of the base on the Aliengo. A TARE waypoint under the base was reached for the controller, which stood still, and
0.3 m away for TARE, which kept it until the lidar got there: the robot stood for good (warehouse, 2026-11-10, after
turning round with TARE's lookahead on the path node under the base). With the base on both sides they agree.

The lidar -> base offset is the static TF from the /Odometry child frame to base_frame (rslidar_lio -> lio_base from
fast_lio_slam_launch.xml), looked up once; nothing is published until it is known. Pose covariance and twist are
copied as they are (FAST-LIO leaves the twist zero).

Usage: ros2 run tare_planner lio_base_odometry.py --ros-args -p use_sim_time:=true
"""

import numpy as np
import rclpy
from geometry_msgs.msg import Quaternion
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformException, TransformListener


def quat_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def quat_multiply(a, b):
    return Quaternion(
        w=a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z,
        x=a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y,
        y=a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
        z=a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w,
    )


def compose(position, orientation, offset):
    """Pose (position (3,), orientation Quaternion) of a frame offset (Transform, in the frame of the given pose) from
    the given pose: the base from the lidar."""
    t = offset.translation
    return (position + quat_to_matrix(orientation) @ np.array([t.x, t.y, t.z]),
            quat_multiply(orientation, offset.rotation))


class LioBaseOdometry(Node):
    def __init__(self):
        super().__init__("lio_base_odometry")
        input_topic = self.declare_parameter("input_topic", "/Odometry").value
        output_topic = self.declare_parameter("output_topic", "/lio_base_odometry").value
        self.base_frame = self.declare_parameter("base_frame", "lio_base").value
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.offset = None  # Transform child_frame_id -> base_frame
        self.pub = self.create_publisher(Odometry, output_topic, 5)
        self.create_subscription(Odometry, input_topic, self.on_odometry, 5)

    def on_odometry(self, msg):
        if self.offset is None:
            try:
                self.offset = self.tf_buffer.lookup_transform(msg.child_frame_id, self.base_frame, Time()).transform
            except TransformException as e:
                self.get_logger().warn(f"waiting for the static TF {msg.child_frame_id} -> {self.base_frame}: {e}",
                                       throttle_duration_sec=5.0)
                return
            t = self.offset.translation
            self.get_logger().info(f"{msg.child_frame_id} -> {self.base_frame}: ({t.x:.3f}, {t.y:.3f}, {t.z:.3f})")
        p = msg.pose.pose.position
        position, orientation = compose(np.array([p.x, p.y, p.z]), msg.pose.pose.orientation, self.offset)
        out = Odometry()
        out.header = msg.header
        out.child_frame_id = self.base_frame
        out.pose.pose.position.x, out.pose.pose.position.y, out.pose.pose.position.z = map(float, position)
        out.pose.pose.orientation = orientation
        out.pose.covariance = msg.pose.covariance
        out.twist = msg.twist
        self.pub.publish(out)


def main():
    rclpy.init()
    node = LioBaseOdometry()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
