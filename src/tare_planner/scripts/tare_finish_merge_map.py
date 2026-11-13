#!/usr/bin/env python3
"""
tare_finish_merge_map.py (ROS 2 Python)

Ends a TARE exploration unattended. Once TARE reports the exploration finished (/exploration_finish, std_msgs/Bool,
true from "Exploration completed, returning home" on), it
  0. with return_home, lets TARE and the goal bridge walk the robot back home first: home is where FAST-LIO had the
     robot's base (odometry_topic, /lio_base_odometry from lio_base_odometry.py, the topic TARE plans with) when this
     node started, which is TARE's home when both come from the same launch with the robot standing. It goes on when the robot is within home_tolerance of it (on the floor: FAST-LIO drifts in
     z), or after return_home_timeout, wherever the robot is then. SC-PGO and far_planner keep mapping on the way.
     Without return_home it goes on at once, where the exploration ended.
     With sweep (exit_inspection, explore_warehouse_launch.xml exits:=true), instead: stops TARE and its goal bridge at
     once and sends /exit_inspection/start_sweep (YAML: home x, y, yaw); exit_sign_sweep.py walks the robot to the
     viewpoints of the exit signs the camera has not looked at and then home, and answers /exit_inspection/sweep_done.
     It goes on then, or after sweep_timeout (sweep_nodes get SIGINT). Nobody listening on start_sweep: as without;
  1. sends SIGINT to the nodes in stop_nodes (TARE and the goal bridges): TARE would go on sending its home
     waypoint; the ones still alive after flush_time get SIGTERM;
  2. stops the robot: a /opti_pessi/goal at its /odom position (OptiPessi stands at a reached goal), sent again every
     tick until the merge starts, in case a bridge tick raced the SIGINT;
  3. after flush_time, once SC-PGO (pgo_node) has rewritten its pose files (1 Hz) with the last keyframes, runs
     makeMergedMap.py <save_directory> --no-vis --output <map_name> --dynamic-output <map_name stem>_dynamic.pcd with
     the python that has open3d, its output going to <map_name stem>.log next to the map. The merge leaves out what
     moved (people the people filter missed, carts): points other keyframes saw through (dynamic_removal.py next to
     it); the _dynamic.pcd holds them;
  4. at the same time, when graph_name is set, saves far_planner's visibility graph next to the map: publishes
     <save_directory>/<graph_name> on /save_file_dir (graph_decoder writes the last /robot_vgraph there), waits for the
     file, then sends SIGINT to graph_nodes (far_planner, graph_decoder; SIGTERM after flush_time). far_planner runs
     during the exploration only to build that graph (its /way_point remapped away from the goal bridges);
     far_load_graph.py loads it in the next missions;
  5. with prune_graph, once both are done, runs pruneFarGraph.py <graph> <map> <map_name stem>_dynamic.pcd (next to
     makeMergedMap.py): the obstacle nodes far_planner made of people that walked by go (the graph had them, its
     terrain maps saw them); the graph as saved stays as <graph_name stem>_unpruned.vgh, the log is
     <graph_name stem>_prune.log;
  6. with exits, once the merge has ended (SC-PGO's pose files are final then), runs finalize_exits.py
     <save_directory> (package exit_inspection): the exit sign sightings of exit_landmark_mapper re-anchored on the
     optimized poses, written to <save_directory>/exit_inspection/exits.yaml; log exit_inspection/finalize.log;
  7. exits when the merge, the graph save, its pruning and the exits end (a merge already started when this node is
     stopped still finishes).
     FAST-LIO and SC-PGO keep running.

Empty save_directory, python and merge_script (the defaults) mean: SC-PGO's save_directory parameter (a relative one
is relative to SC-PGO's working directory), <workspace>/.venv-yolo/bin/python and
<workspace>/src/FAST_LIO_SLAM_ROS2/SC_PGO_ROS2/utils/python/makeMergedMap.py, <workspace> being the first directory
above the tare_planner install that has a src/.

Usage: ros2 run tare_planner tare_finish_merge_map.py --ros-args -p map_name:=optimized_map.pcd
"""

import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import rclpy
from ament_index_python.packages import get_package_prefix
from nav_msgs.msg import Odometry
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter_client import AsyncParameterClient
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, String
from visualization_msgs.msg import Marker, MarkerArray


def node_pids(name):
    """PIDs of the processes running node name: started with -r __node:=name (ros2 launch, ros2 run --ros-args), or
    else from an executable or script called name (ros2 run without remapping)."""
    pids = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                args = f.read().decode(errors="replace").split("\0")
        except OSError:
            continue
        executables = {os.path.basename(arg) for arg in args[:2]}
        if f"__node:={name}" in args or name in executables or f"{name}.py" in executables:
            pids.append(int(entry))
    return pids


def workspace():
    """First directory above the tare_planner install that has a src/, or None."""
    for parent in Path(get_package_prefix("tare_planner")).resolve().parents:
        if (parent / "src").is_dir():
            return parent
    return None


class TareFinishMergeMap(Node):
    def __init__(self):
        super().__init__("tare_finish_merge_map")
        self.stop_nodes = self.declare_parameter(
            "stop_nodes", ["tare_planner_node", "tare_path_to_opti_pessi_goal", "way_point_to_opti_pessi_goal"]).value
        self.pgo_node = self.declare_parameter("pgo_node", "alaserPGO").value
        self.save_directory = self.declare_parameter("save_directory", "").value
        self.map_name = self.declare_parameter("map_name", "optimized_map.pcd").value
        self.python = self.declare_parameter("python", "").value
        self.merge_script = self.declare_parameter("merge_script", "").value
        self.flush_time = self.declare_parameter("flush_time", 5.0).value
        self.graph_name = self.declare_parameter("graph_name", "warehouse.vgh").value
        self.graph_nodes = self.declare_parameter("graph_nodes", ["far_planner", "graph_decoder"]).value
        self.graph_timeout = self.declare_parameter("graph_timeout", 10.0).value
        self.prune_graph = self.declare_parameter("prune_graph", True).value
        self.prune_script = self.declare_parameter("prune_script", "").value
        # exit_inspection: exit sign sightings -> exits.yaml at the end (needs exit_landmark_mapper during the run)
        self.exits = self.declare_parameter("exits", False).value
        self.exits_script = self.declare_parameter("exits_script", "").value
        self.return_home = self.declare_parameter("return_home", True).value
        # 1.0: TARE calls the robot home ~0.5 m short of where it started (0.53 m in the empty room, 2026-11-10)
        self.home_tolerance = self.declare_parameter("home_tolerance", 1.0).value
        # dynamic typing: a launch argument like return_home_timeout:=300 arrives as an int
        self.return_home_timeout = float(self.declare_parameter(
            "return_home_timeout", 600.0, ParameterDescriptor(dynamic_typing=True)).value)
        # exit sign sweep before going home (exit_inspection exit_sign_sweep.py)
        self.sweep = self.declare_parameter("sweep", False).value
        self.sweep_timeout = float(self.declare_parameter(
            "sweep_timeout", 900.0, ParameterDescriptor(dynamic_typing=True)).value)
        self.sweep_nodes = self.declare_parameter("sweep_nodes", ["exit_sweep_goal_bridge"]).value

        root = workspace()
        if root is not None:
            self.python = self.python or str(root / ".venv-yolo" / "bin" / "python")
            self.merge_script = self.merge_script or str(
                root / "src" / "FAST_LIO_SLAM_ROS2" / "SC_PGO_ROS2" / "utils" / "python" / "makeMergedMap.py")
        self.prune_script = self.prune_script or os.path.join(os.path.dirname(self.merge_script), "pruneFarGraph.py")
        if self.exits and not self.exits_script:
            try:
                self.exits_script = os.path.join(get_package_prefix("exit_inspection"), "lib", "exit_inspection",
                                                 "finalize_exits.py")
            except Exception:  # PackageNotFoundError
                self.get_logger().error("exits:=true but package exit_inspection not found: no exits.yaml")
                self.exits = False
        if self.exits and not os.path.isfile(self.exits_script):
            self.get_logger().error(f"exits_script '{self.exits_script}' not found: no exits.yaml")
            self.exits = False
        for name, path in (("python", self.python), ("merge_script", self.merge_script),
                           ("prune_script", self.prune_script if self.prune_graph and self.graph_name else self.python)):
            if not os.path.isfile(path):
                self.get_logger().error(f"{name} '{path}' not found: the map will not be merged")

        self.odom = None
        self.create_subscription(Odometry, "/odom", lambda msg: setattr(self, "odom", msg), 5)
        # FAST-LIO at the base, as TARE takes it (its sub_state_estimation_topic_): where home is and how far the robot
        # is from it. Not /Odometry, the lidar, 0.29 m ahead of the base: a robot home but turned round was up to
        # 0.58 m off there.
        self.odometry_topic = self.declare_parameter("odometry_topic", "/lio_base_odometry").value
        self.home = None
        self.home_yaw = 0.0
        self.position = None
        self.create_subscription(Odometry, self.odometry_topic, self.on_lio_odometry, 5)
        self.create_subscription(Bool, "/exploration_finish", self.on_exploration_finish, 5)
        self.goal_pub = self.create_publisher(MarkerArray, "/opti_pessi/goal", 1)
        self.save_graph_pub = self.create_publisher(String, "/save_file_dir", 5)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.sweep_pub = self.create_publisher(String, "/exit_inspection/start_sweep", 5)
        self.sweep_done = False
        self.sweep_since = None
        self.create_subscription(String, "/exit_inspection/sweep_done", self.on_sweep_done, latched)
        self.pgo_params = AsyncParameterClient(self, self.pgo_node)
        self.save_directory_future = None

        self.state = "exploring"  # -> sweeping or returning (return_home) -> stopping -> merging -> done
        self.return_since = None
        self.finished_at = None
        self.stop_goal = None
        self.merge = None
        self.map_path = None
        self.dynamic_path = None
        self.directory = None
        self.merged = False
        # graph save: idle -> saving -> stopping -> done (done at once without graph_name)
        self.graph_state = "idle" if self.graph_name else "done"
        self.graph_path = None
        self.graph_since = None
        self.graph_size = -1
        self.graph_saved = False
        # graph pruning, after the merge and the graph save: idle -> pruning -> done
        self.prune_state = "idle" if self.prune_graph and self.graph_name else "done"
        self.prune = None
        # exits.yaml, after the merge: idle -> running -> done (done at once without exits)
        self.exits_state = "idle" if self.exits else "done"
        self.exits_proc = None
        self.create_timer(0.5, self.tick)
        where = self.save_directory or f"{self.pgo_node}'s save_directory"
        then = "the robot back home, then " if self.return_home else ""
        self.get_logger().info(f"waiting for /exploration_finish, then {then}merging {self.map_name} in {where}")

    def on_lio_odometry(self, msg):
        self.position = (msg.pose.pose.position.x, msg.pose.pose.position.y)
        if self.home is None:
            self.home = self.position
            q = msg.pose.pose.orientation
            self.home_yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))

    def home_distance(self):
        return math.hypot(self.position[0] - self.home[0], self.position[1] - self.home[1])

    def on_sweep_done(self, msg):
        self.sweep_done = True

    def on_exploration_finish(self, msg):
        if not msg.data or self.state != "exploring":
            return
        if self.sweep:
            if self.sweep_pub.get_subscription_count() > 0:
                self.get_logger().info("exploration finished: stopping TARE, exit sign sweep, then home")
                self.signal_nodes(signal.SIGINT)
                home = None if self.home is None else [self.home[0], self.home[1], self.home_yaw]
                self.sweep_pub.publish(String(data=f"home: {home}" if home else "{}"))
                self.state, self.sweep_since = "sweeping", time.monotonic()
                return
            self.get_logger().warn("sweep:=true but nobody listens on /exit_inspection/start_sweep: no sweep")
        if self.return_home and self.home is None:
            self.get_logger().warn(f"exploration finished, but no {self.odometry_topic}: home unknown, ending it here")
        elif self.return_home and self.home_distance() > self.home_tolerance:
            self.get_logger().info(
                f"exploration finished {self.home_distance():.1f} m from home ({self.home[0]:.2f}, {self.home[1]:.2f}): "
                f"waiting for TARE to bring the robot back, at most {self.return_home_timeout:.0f} s")
            self.state, self.return_since = "returning", time.monotonic()
            return
        self.stop_exploration()

    def stop_exploration(self):
        self.get_logger().info(f"stopping {', '.join(self.stop_nodes)} and the robot")
        self.state, self.finished_at = "stopping", time.monotonic()
        self.signal_nodes(signal.SIGINT)
        if self.odom is None:
            self.get_logger().warn("no /odom: the robot is not stopped, it walks on to the last goal")
        else:
            # The robot's own yaw (about z only): acados turns to the yaw of its goal, w = 1 would turn it towards +x
            q = self.odom.pose.pose.orientation
            yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
            marker = Marker()
            marker.header.frame_id = self.odom.header.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = "goal"
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.position = self.odom.pose.pose.position
            marker.pose.orientation.z, marker.pose.orientation.w = math.sin(0.5 * yaw), math.cos(0.5 * yaw)
            marker.scale.x, marker.scale.y, marker.scale.z = 0.5, 0.1, 0.1
            marker.color.r, marker.color.a = 1.0, 1.0
            self.stop_goal = MarkerArray(markers=[marker])
            self.goal_pub.publish(self.stop_goal)
        if not self.save_directory:
            self.save_directory_future = self.pgo_params.get_parameters(["save_directory"])

    def finished(self):
        return (self.state == "done" and self.graph_state == "done" and self.prune_state == "done"
                and self.exits_state == "done")

    def signal_nodes(self, sig, names=None):
        for name in self.stop_nodes if names is None else names:
            for pid in node_pids(name):
                try:
                    os.kill(pid, sig)
                except ProcessLookupError:
                    continue
                self.get_logger().info(f"{signal.Signals(sig).name} sent to {name} (PID {pid})")

    def resolve_save_directory(self):
        """The directory SC-PGO saves to, or None."""
        if self.save_directory:
            return os.path.abspath(os.path.expanduser(self.save_directory))
        future = self.save_directory_future
        if future is None or not future.done() or future.result() is None:
            self.get_logger().error(f"no answer from /{self.pgo_node} about its save_directory (not running? "
                                    "pgo:=false?); set this node's save_directory")
            return None
        value = future.result().values[0]
        if value.type != ParameterType.PARAMETER_STRING:
            self.get_logger().error(f"/{self.pgo_node} has no save_directory parameter")
            return None
        directory = value.string_value
        if not os.path.isabs(directory):
            # Relative to where SC-PGO runs (ros2 launch's working directory), not to where this node runs
            pids = node_pids(self.pgo_node)
            directory = os.path.join(os.readlink(f"/proc/{pids[0]}/cwd") if pids else os.getcwd(), directory)
        return os.path.normpath(directory)

    def start_merge(self):
        directory = self.resolve_save_directory()
        if directory is None:
            return False
        self.directory = directory
        self.map_path = os.path.join(directory, self.map_name)
        log_path = os.path.splitext(self.map_path)[0] + ".log"
        self.dynamic_path = os.path.splitext(self.map_path)[0] + "_dynamic.pcd"
        command = [self.python, self.merge_script, directory, "--no-vis", "--output", self.map_name,
                   "--dynamic-output", self.dynamic_path]
        self.get_logger().info(f"merging the map: {' '.join(command)} (output in {log_path})")
        try:
            # Its own session: a Ctrl-C on the launch does not reach it, so a started merge still saves the map
            with open(log_path, "w") as log:
                self.merge = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        except OSError as e:
            self.get_logger().error(f"cannot run the merge: {e}")
            return False
        return True

    def tick(self):
        self.tick_graph()
        self.tick_state()
        self.tick_prune()
        self.tick_exits()

    def tick_state(self):
        if self.state == "sweeping":
            waited = time.monotonic() - self.sweep_since
            if self.sweep_done:
                self.get_logger().info(f"exit sign sweep done after {waited:.0f} s")
            elif waited > self.sweep_timeout:
                self.get_logger().warn(f"exit sign sweep not done after {waited:.0f} s (sweep_timeout): stopping it")
                self.signal_nodes(signal.SIGINT, self.sweep_nodes)
            else:
                return
            self.stop_exploration()
        elif self.state == "returning":
            distance, waited = self.home_distance(), time.monotonic() - self.return_since
            if distance <= self.home_tolerance:
                self.get_logger().info(f"robot back home after {waited:.0f} s")
            elif waited > self.return_home_timeout:
                self.get_logger().warn(f"robot still {distance:.1f} m from home after {waited:.0f} s "
                                       "(return_home_timeout): ending the exploration where it is")
            else:
                self.get_logger().info(f"returning home: {distance:.1f} m to go", throttle_duration_sec=10.0)
                return
            self.stop_exploration()
        elif self.state == "stopping":
            if self.stop_goal is not None:
                self.goal_pub.publish(self.stop_goal)
            if time.monotonic() - self.finished_at < self.flush_time:
                return
            self.signal_nodes(signal.SIGTERM)  # only the ones SIGINT did not end are still found
            self.state = "merging" if self.start_merge() else "done"
            if self.graph_state == "idle":
                self.start_graph_save()
        elif self.state == "merging":
            code = self.merge.poll()
            if code is None:
                return
            if code == 0 and os.path.isfile(self.map_path):
                size = os.path.getsize(self.map_path) / 1e6
                self.get_logger().info(f"map saved: {self.map_path} ({size:.1f} MB)")
                self.merged = True
            else:
                self.get_logger().error(f"merge failed (exit code {code}), see "
                                        f"{os.path.splitext(self.map_path)[0]}.log")
            self.state = "done"

    def start_graph_save(self):
        directory = self.resolve_save_directory()
        if directory is None:
            self.graph_state = "done"
            return
        if self.save_graph_pub.get_subscription_count() == 0:
            self.get_logger().error("nobody on /save_file_dir (graph_decoder not running?): the graph is not saved")
            self.graph_state = "done"
            return
        self.graph_path = os.path.join(directory, self.graph_name)
        if os.path.exists(self.graph_path):
            os.remove(self.graph_path)  # so the file found below is the new one
        self.get_logger().info(f"saving the FAR graph: {self.graph_path}")
        self.save_graph_pub.publish(String(data=self.graph_path))
        self.graph_state, self.graph_since = "saving", time.monotonic()

    def tick_graph(self):
        if self.graph_state == "saving":
            # graph_decoder writes the whole file in its callback: a non-empty file whose size holds is complete
            size = os.path.getsize(self.graph_path) if os.path.isfile(self.graph_path) else 0
            if size > 0 and size == self.graph_size:
                with open(self.graph_path) as f:
                    nodes = sum(1 for _ in f)
                self.get_logger().info(f"FAR graph saved: {self.graph_path} ({nodes} nodes)")
                self.graph_saved = True
            elif time.monotonic() - self.graph_since < self.graph_timeout:
                self.graph_size = size
                return
            else:
                self.get_logger().error(f"FAR graph not saved to {self.graph_path} within {self.graph_timeout:.0f} s "
                                        "(far_planner graph empty?)")
            self.signal_nodes(signal.SIGINT, self.graph_nodes)
            self.graph_state, self.graph_since = "stopping", time.monotonic()
        elif self.graph_state == "stopping" and time.monotonic() - self.graph_since >= self.flush_time:
            self.signal_nodes(signal.SIGTERM, self.graph_nodes)
            self.graph_state = "done"

    def tick_prune(self):
        if self.prune_state == "idle":
            if self.state != "done" or self.graph_state != "done":
                return
            if not (self.merged and self.graph_saved and os.path.isfile(self.dynamic_path)):
                self.get_logger().warn("no merged map or no FAR graph: the graph is not pruned")
                self.prune_state = "done"
                return
            log_path = os.path.splitext(self.graph_path)[0] + "_prune.log"
            command = [self.python, self.prune_script, self.graph_path, self.map_path, self.dynamic_path]
            self.get_logger().info(f"pruning the FAR graph: {' '.join(command)} (output in {log_path})")
            try:
                with open(log_path, "w") as log:
                    self.prune = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            except OSError as e:
                self.get_logger().error(f"cannot run the graph pruning: {e}")
                self.prune_state = "done"
                return
            self.prune_state = "pruning"
        elif self.prune_state == "pruning":
            code = self.prune.poll()
            if code is None:
                return
            if code == 0:
                with open(self.graph_path) as f:
                    nodes = sum(1 for _ in f)
                self.get_logger().info(f"FAR graph pruned: {self.graph_path} ({nodes} nodes)")
            else:
                self.get_logger().error(f"graph pruning failed (exit code {code}), the graph is as far_planner saved "
                                        f"it; see {os.path.splitext(self.graph_path)[0]}_prune.log")
            self.prune_state = "done"

    def tick_exits(self):
        if self.exits_state == "idle":
            if self.state != "done":
                return
            if self.directory is None:
                self.get_logger().warn("save_directory unknown: no exits.yaml")
                self.exits_state = "done"
                return
            log_dir = os.path.join(self.directory, "exit_inspection")
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "finalize.log")
            command = [sys.executable, self.exits_script, self.directory]
            self.get_logger().info(f"writing the exits: {' '.join(command)} (output in {log_path})")
            try:
                with open(log_path, "w") as log:
                    self.exits_proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                                       start_new_session=True)
            except OSError as e:
                self.get_logger().error(f"cannot run finalize_exits.py: {e}")
                self.exits_state = "done"
                return
            self.exits_state = "running"
        elif self.exits_state == "running":
            code = self.exits_proc.poll()
            if code is None:
                return
            log_path = os.path.join(self.directory, "exit_inspection", "finalize.log")
            if code == 0:
                with open(log_path) as f:
                    first = f.readline().strip()
                self.get_logger().info(f"exits written: {os.path.join(self.directory, 'exit_inspection', 'exits.yaml')}"
                                       f" ({first})")
            else:
                self.get_logger().error(f"finalize_exits.py failed (exit code {code}), see {log_path}")
            self.exits_state = "done"


def main():
    rclpy.init()
    node = TareFinishMergeMap()
    try:
        while rclpy.ok() and not node.finished():
            rclpy.spin_once(node, timeout_sec=0.5)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if node.merge is not None and node.merge.poll() is None:
            node.get_logger().info(f"the merge (PID {node.merge.pid}) goes on in the background: {node.map_path}")
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
