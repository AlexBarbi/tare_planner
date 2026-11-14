"""lio_base_odometry.py moves FAST-LIO's /Odometry from the lidar to the robot's base."""
import importlib.util
import math
import os

import numpy as np
import pytest
import rclpy
from geometry_msgs.msg import Quaternion, Transform, TransformStamped, Vector3
from nav_msgs.msg import Odometry
from tf2_ros import TransformException

os.environ["ROS_DOMAIN_ID"] = "93"

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts', 'lio_base_odometry.py')
_spec = importlib.util.spec_from_file_location('lio_base_odometry', _path)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

# rslidar_lio -> lio_base in fast_lio_slam_launch.xml
LIDAR_TO_BASE = Transform(translation=Vector3(x=-0.2835, y=0.0, z=-0.066), rotation=Quaternion(w=1.0))


def yaw_quaternion(yaw):
    return Quaternion(z=math.sin(yaw / 2.0), w=math.cos(yaw / 2.0))


def lidar_odometry(x, y, z, q):
    msg = Odometry()
    msg.header.frame_id = 'camera_init'
    msg.header.stamp.sec = 264
    msg.child_frame_id = 'rslidar_lio'
    msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = x, y, z
    msg.pose.pose.orientation = q
    msg.pose.covariance[0] = 0.01
    return msg


def test_base_behind_lidar_along_heading():
    """The 2026-11-10 warehouse stall: lidar at (17.31, -5.06) facing 154 deg, base 0.28 m behind it."""
    yaw = math.radians(154.4)
    position, orientation = script.compose(np.array([17.31, -5.06, -0.07]), yaw_quaternion(yaw), LIDAR_TO_BASE)
    assert position == pytest.approx([17.31 - 0.2835 * math.cos(yaw), -5.06 - 0.2835 * math.sin(yaw), -0.07 - 0.066])
    assert (orientation.x, orientation.y, orientation.z, orientation.w) == pytest.approx(
        (0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)))


def test_pitched_lidar():
    """A pitched body moves the base offset out of the horizontal plane, as the rigid transform says."""
    pitch = math.radians(10.0)
    q = Quaternion(y=math.sin(pitch / 2.0), w=math.cos(pitch / 2.0))
    position, _ = script.compose(np.zeros(3), q, LIDAR_TO_BASE)
    assert position == pytest.approx([-0.2835 * math.cos(pitch) - 0.066 * math.sin(pitch), 0.0,
                                      0.2835 * math.sin(pitch) - 0.066 * math.cos(pitch)])


def test_offset_rotation_is_applied():
    """A base frame turned against the lidar frame turns the output orientation by the same amount."""
    offset = Transform(translation=Vector3(), rotation=yaw_quaternion(math.radians(90.0)))
    _, orientation = script.compose(np.zeros(3), yaw_quaternion(math.radians(30.0)), offset)
    assert (orientation.z, orientation.w) == pytest.approx((math.sin(math.radians(60.0)), math.cos(math.radians(60.0))))


@pytest.fixture
def node():
    rclpy.init()
    n = script.LioBaseOdometry()
    published = []
    n.pub.publish = published.append
    yield n, published
    n.destroy_node()
    rclpy.shutdown()


def test_nothing_published_until_offset_known(node):
    n, published = node

    def no_tf(*args, **kwargs):
        raise TransformException('rslidar_lio does not exist')
    n.tf_buffer.lookup_transform = no_tf
    n.on_odometry(lidar_odometry(1.0, 2.0, 0.0, yaw_quaternion(0.0)))
    assert published == []


def test_relayed_message(node):
    n, published = node
    lookups = []

    def static_tf(target, source, time):
        lookups.append((target, source))
        return TransformStamped(transform=LIDAR_TO_BASE)
    n.tf_buffer.lookup_transform = static_tf
    for x in (1.0, 2.0):
        n.on_odometry(lidar_odometry(x, 2.0, 0.5, yaw_quaternion(0.0)))
    assert lookups == [('rslidar_lio', 'lio_base')]  # static: looked up once
    out = published[-1]
    assert out.header.frame_id == 'camera_init' and out.header.stamp.sec == 264
    assert out.child_frame_id == 'lio_base'
    p = out.pose.pose.position
    assert (p.x, p.y, p.z) == pytest.approx((2.0 - 0.2835, 2.0, 0.5 - 0.066))
    assert out.pose.covariance[0] == pytest.approx(0.01)
