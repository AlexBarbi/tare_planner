#!/usr/bin/env python3
"""
tare_path_to_opti_pessi_goal.py (ROS 2 Python)

Bridge from TARE to OptiPessiController. OptiPessi walks straight to /opti_pessi/goal, with no local planner in between,
so the goal is chosen here and checked against the obstacles of /terrain_map (points more than obstacle_height above
the ground, farther than self_radius from the robot):

  1. /way_point, the paths in path_topics (/exploration_path: local path, /global_path_full: global TSP loop),
     /terrain_map (all in frame map) and the robot (/odom, base) are brought to goal_frame (odom, the only frame the
     controller takes) with the latest TF (odom -> lio_map -> camera_init -> map, see tare_explore_nav_launch.xml);
  2. the candidate goal is the first of
     - midpoint: the middle point between the robot and /way_point, if the straight line robot -> /way_point keeps
       clearance from every obstacle (walking to the midpoint again and again walks that whole line);
     - path: lookahead_distance ahead of the robot along the TARE path /way_point lies on (pure pursuit, see follow()),
       if the line robot -> it keeps clearance: the path goes round walls and through doors, but flickers when TARE
       replans, so it is only taken when the straight way is blocked;
     - detour: a point up to detour_distance away, in the direction of the path goal (or else of the midpoint) turned
       by the smallest multiple of detour_step (up to max_detour_angle each side) that keeps clearance; if there is
       none, the search is repeated with the distance halved down to min_detour_distance, then all over again with
       min_clearance (as the autonomy stack's localPlanner shrinks pathRange, then pathScale);
     - stop: the robot itself (OptiPessi stands at a reached goal). A stop is final while /way_point stays put: the
       robot stands, so nothing the choice depends on changes;
  3. the goal follows the candidate, but a candidate that jumps (more than jump_distance in one tick: TARE replanned,
     the path flickered, the choice above changed) is only taken once it has stayed put for switch_time, as long as
     the current goal is still clear and more than reach_distance away;
  4. it is published (one ADD ARROW marker in goal_frame, colored by where it comes from) when it has moved more than
     republish_distance or its yaw has turned by more than republish_yaw.

Every goal has a yaw, in the marker's orientation: on a path or far_planner's route its direction there in walking
direction (at a vertex the stretch after it, at the end the last stretch: see path_yaw()); a midpoint faces /way_point,
a detour the way from the robot to it; a stop, or the robot at the end of far_planner's route, keeps the robot's yaw
(/odom). acados_legged_controller turns to it near the goal (useGoalYaw, its relay opti_pessi_goal_relay.py);
OptiPessiController reads the position only.
A yaw hint (/goal_yaw_hint, geometry_msgs/PoseStamped, latched here until replaced; an empty frame_id clears it)
overrides that yaw for every goal within hint_radius of the hint's position: exit_inspection_pass.py sends one with
each viewpoint, so the robot ends there facing the door. Nobody sends hints in the TARE exploration.

Without a fresh /terrain_map nothing is checked: the path goal is taken, or else the midpoint. Until /way_point and
/odom are fresh the last goal is kept. kExtendWayPoint must stay false in the TARE config (the path goal needs
/way_point on the path). All distances can be changed at run time (ros2 param set).

With far_planner (route_topic:=/viz_path_topic, set by far_planner_launch.xml) the goal is taken on the route
far_planner has planned, the line it shows from the robot through its graph nodes to the goal (a LINE_STRIP Marker,
published on every planning loop, empty when it has no route), instead of step 2:
     - path: lookahead_distance ahead of the robot along the route (pure pursuit, see pursue()), its end at most,
       if the line robot -> it keeps clearance; if not, the same with the lookahead halved down to
       min_detour_distance, which stays on the route and stops short of what is in the way;
     - detour, then stop, as above, towards the first of those points.
/way_point is not used then. far_planner makes it for a local planner that steers in a direction: it is pushed to
local_planner_range (2.5 m) from the robot, past the next node and off the route, and it jumps whenever the route
changes. It is only used while far_planner publishes no route (goal reached: /way_point is the goal; no way found:
/way_point is the robot), with step 2 as it is.

The robot comes from /odom by message, not from the odom -> base TF, which OptiPessi's visualizer stamps with wall time.

Usage: ros2 run tare_planner tare_path_to_opti_pessi_goal.py --ros-args -p clearance:=0.3
"""

import numpy as np
import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped
from nav_msgs.msg import Odometry, Path
from rcl_interfaces.msg import FloatingPointRange, ParameterDescriptor
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

# Projections within this distance of the nearest one are the same place on the path: a waypoint is a path vertex, so
# it touches two segments, and a looped path passes the robot twice.
MATCH_SLACK = 0.15
# Where the robot may be on the path: TARE's path starts at the viewpoint nearest the lidar, up to ~0.6 m from base.
ROBOT_MATCH_RADIUS = 1.0
# Goal marker color (r, g, b) by where the goal comes from
COLORS = {"midpoint": (0.0, 1.0, 0.0), "path": (0.0, 0.5, 1.0), "detour": (1.0, 0.8, 0.0), "stop": (1.0, 0.0, 0.0)}


def quat_to_matrix(q):
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def yaw_of(q):
    """Yaw [rad] of quaternion q (geometry_msgs Quaternion); 0 for the zero quaternion of an unset orientation."""
    return float(np.arctan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)))


def angle_difference(a, b):
    """a - b [rad] wrapped to [-pi, pi)."""
    return float((a - b + np.pi) % (2.0 * np.pi) - np.pi)


def bearing(origin, target, fallback):
    """Yaw [rad] of the line origin (2,) -> target (2,), or fallback when target is on origin."""
    offset = target - origin
    return float(np.arctan2(offset[1], offset[0])) if np.linalg.norm(offset) > 1e-3 else fallback


def path_yaw(path, vertex_s, s, direction):
    """Yaw [rad] of path (n, 3) at arc length s, walked along it (direction 1) or back (-1): the segment s lies on; at a
    vertex the one after it in walking order, at the end of the walk the last one walked."""
    at_vertex = np.flatnonzero(np.abs(vertex_s - s) < 1e-9)
    s = vertex_s[at_vertex[0]] if len(at_vertex) else s
    if direction > 0:
        i = int(np.clip(np.searchsorted(vertex_s, s, side="right") - 1, 0, len(path) - 2))
        d = path[i + 1] - path[i]
    else:
        i = int(np.clip(np.searchsorted(vertex_s, s, side="left") - 1, 0, len(path) - 2))
        d = path[i] - path[i + 1]
    return float(np.arctan2(d[1], d[0]))


def project(point, path_xy):
    """Distance from point (2,) to each segment of the polyline path_xy (n, 2) and arc length of the projection on it,
    plus the arc length of each vertex. Segments must not be empty."""
    a, ab = path_xy[:-1], np.diff(path_xy, axis=0)
    length = np.linalg.norm(ab, axis=1)
    t = np.clip(np.einsum("ij,ij->i", point - a, ab) / length ** 2, 0.0, 1.0)
    distance = np.linalg.norm(a + t[:, None] * ab - point, axis=1)
    vertex_s = np.concatenate(([0.0], np.cumsum(length)))
    return distance, vertex_s[:-1] + t * length, vertex_s


def point_at(path, vertex_s, s):
    """Point at arc length s along path (n, 3)."""
    i = int(np.clip(np.searchsorted(vertex_s, s, side="right") - 1, 0, len(path) - 2))
    t = np.clip((s - vertex_s[i]) / (vertex_s[i + 1] - vertex_s[i]), 0.0, 1.0)
    return path[i] + t * (path[i + 1] - path[i])


def distance_to_segment(points, a, b):
    """Distance from points (m, 2) to the segment a -> b."""
    ab = b - a
    t = np.clip((points - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0)
    return np.linalg.norm(a + t[:, None] * ab - points, axis=1)


def follow(path, robot, waypoint, lookahead, waypoint_tolerance, max_waypoint_path_distance, max_chord_deviation):
    """Pure pursuit along path (n, 3) from robot (2,) towards waypoint (2,).

    The robot's projection on the path is the nearest one within max_waypoint_path_distance of the waypoint's along the
    path (a TARE path can loop back past the robot). The goal is lookahead further along the path, never past the
    waypoint, pulled back to the path vertex it would cut past when a vertex in between lies more than
    max_chord_deviation off the straight line robot -> goal.

    Returns the goal (3,) and the yaw of the path there in walking direction (see path_yaw()), or None if the waypoint
    is not on path.
    """
    xy = path[:, :2]
    waypoint_distance, waypoint_s, vertex_s = project(waypoint, xy)
    if waypoint_distance.min() > waypoint_tolerance:
        return None
    waypoint_s = waypoint_s[waypoint_distance <= waypoint_distance.min() + MATCH_SLACK]
    robot_distance, robot_s, _ = project(robot, xy)
    near = robot_distance <= max(robot_distance.min() + MATCH_SLACK, ROBOT_MATCH_RADIUS)
    robot_distance, robot_s = robot_distance[near], robot_s[near]
    # The robot is where the path is nearest, unless that is farther from the waypoint along the path than TARE puts
    # it: a TARE path loops back past the robot, and on its way out or back may pass closer than where TARE starts it.
    gap = np.abs(waypoint_s[None, :] - robot_s[:, None])  # (robot, waypoint) projections
    plausible = np.flatnonzero(gap.min(axis=1) <= max_waypoint_path_distance)
    i = plausible[np.argmin(robot_distance[plausible])] if len(plausible) else np.argmin(gap.min(axis=1))
    return ahead_on_path(path, vertex_s, robot_s[i], waypoint_s[np.argmin(gap[i])], robot, lookahead,
                         max_chord_deviation)


def ahead_on_path(path, vertex_s, start, end, robot, lookahead, max_chord_deviation):
    """Point (3,) lookahead along path (n, 3) from arc length start towards arc length end, end at most, pulled back
    to the path vertex it would cut past when a vertex in between lies more than max_chord_deviation off the straight
    line robot (2,) -> it; and the yaw of the path there in walking direction (see path_yaw())."""
    direction = 1.0 if end >= start else -1.0
    ahead = (vertex_s - start) * direction  # arc length of each vertex ahead of the robot's projection

    # Vertices passed walking min(lookahead, remaining) from the robot's projection, in walking order, and the point
    # reached, with their arc lengths
    length = min(lookahead, abs(end - start))
    passed = np.flatnonzero((ahead > 0.0) & (ahead < length))
    passed = passed if direction > 0 else passed[::-1]
    candidates = np.vstack([path[passed], point_at(path, vertex_s, start + direction * length)])
    candidates_s = np.concatenate([vertex_s[passed], [start + direction * length]])
    # Farthest candidate whose straight line from the robot stays within max_chord_deviation of the vertices before it
    k = next((k for k in range(len(candidates) - 1, 0, -1)
              if distance_to_segment(candidates[:k, :2], robot, candidates[k, :2]).max() <= max_chord_deviation), 0)
    return candidates[k], path_yaw(path, vertex_s, candidates_s[k], direction)


def pursue(path, robot, lookahead, max_chord_deviation):
    """Pure pursuit along a route (n, 3) that starts at the robot (2,) and ends at the goal, as far_planner's does:
    the point lookahead further along it than the robot, the end of the route at most, not cutting corners, and the
    yaw of the route there (see ahead_on_path()). The robot is where the route is nearest; if it passes the robot again
    later (out and back round the end of a wall), at the first of those places."""
    distance, s, vertex_s = project(robot, path[:, :2])
    start = s[distance <= distance.min() + MATCH_SLACK].min()
    return ahead_on_path(path, vertex_s, start, vertex_s[-1], robot, lookahead, max_chord_deviation)


def without_empty_segments(path):
    """path (n, 3) without the vertices that repeat the one before (TARE appends the robot or a loop's start again)."""
    return path[np.concatenate(([True], np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1) > 1e-6))]


def clear(robot, goal, obstacles, clearance):
    """Whether the segment robot -> goal (2,) keeps clearance from obstacles (m, 2). Obstacles behind the robot do not
    count: walking away from the robot does not bring it closer to them."""
    ab = goal - robot
    if ab @ ab < 1e-12:
        return True
    ahead = obstacles[(obstacles - robot) @ ab > 0.0]
    return len(ahead) == 0 or bool(distance_to_segment(ahead, robot, goal).min() >= clearance)


def detour(robot, target, obstacles, clearance, distance, step, max_angle):
    """Point at most distance (and at most as far as target (2,)) from robot, in the direction of target turned by the
    smallest multiple of step [rad] up to max_angle each side, whose segment from robot keeps clearance; or None."""
    offset = target - robot
    length = min(np.linalg.norm(offset), distance)
    if length < 1e-6:
        return None
    heading = np.arctan2(offset[1], offset[0])
    for k in range(int(round(max_angle / step)) + 1):
        for sign in (1.0, -1.0) if k else (1.0,):
            angle = heading + sign * k * step
            goal = robot + length * np.array([np.cos(angle), np.sin(angle)])
            if clear(robot, goal, obstacles, clearance):
                return goal
    return None


def relaxed(value, minimum):
    """value, then value halved while that stays above minimum, then minimum: the steps a limit is relaxed in."""
    values = [value]
    while values[-1] / 2.0 > minimum:
        values.append(values[-1] / 2.0)
    if minimum < values[-1]:
        values.append(minimum)
    return values


class TareGoalBridge(Node):
    def __init__(self):
        super().__init__("tare_path_to_opti_pessi_goal")

        def number(name, value, description, low, high):
            descriptor = ParameterDescriptor(description=description,
                                             floating_point_range=[FloatingPointRange(from_value=low, to_value=high)])
            self.declare_parameter(name, value, descriptor)

        number("clearance", 0.3, "distance the line robot -> goal keeps from obstacles [m]", 0.0, 2.0)
        number("obstacle_height", 0.2, "/terrain_map points higher above the ground are obstacles [m]", 0.0, 2.0)
        number("self_radius", 0.3, "obstacles nearer the robot are ignored (the robot's own legs, a wall it "
               "brushes) [m]", 0.0, 1.0)
        number("lookahead_distance", 3.0, "path goal distance ahead of the robot along the TARE path or far_planner's "
               "route [m]", 0.2, 5.0)
        number("max_chord_deviation", 0.2, "how far path vertices may lie off the line robot -> path goal [m]", 0.0,
               2.0)
        number("waypoint_tolerance", 0.3, "how far /way_point may lie off a path it is taken from [m]", 0.0, 2.0)
        number("max_waypoint_path_distance", 8.0, "farthest /way_point along the path from the robot "
               "(TARE: kLookAheadDistance plus a path segment) [m]", 1.0, 100.0)
        number("min_clearance", 0.2, "clearance a detour may shrink to before the robot stops [m]", 0.0, 2.0)
        number("detour_distance", 1.0, "farthest detour goal from the robot [m]", 0.2, 5.0)
        number("min_detour_distance", 0.3, "distance a detour, or the lookahead along far_planner's route, may shrink "
               "to before the robot stops [m]", 0.05, 5.0)
        number("detour_step", 15.0, "angle between detour directions [deg]", 1.0, 90.0)
        number("max_detour_angle", 90.0, "largest detour turn each side [deg]", 0.0, 180.0)
        number("jump_distance", 0.5, "candidate move in one tick that is a jump [m]", 0.0, 10.0)
        number("switch_time", 1.0, "how long a jumped candidate must stay put before it is taken [s]", 0.0, 10.0)
        number("reach_distance", 0.3, "goal distance at which the next candidate is taken at once [m]", 0.0, 2.0)
        number("republish_distance", 0.1, "goal change that is published [m]", 0.0, 1.0)
        number("republish_yaw", 0.1, "goal yaw change that is published, also at the same place [rad]", 0.0, 3.2)
        number("hint_radius", 0.5, "goals this close to the /goal_yaw_hint position take its yaw [m]", 0.0, 5.0)
        number("input_timeout", 3.0, "age after which a path, /way_point, /odom or /terrain_map is ignored [s]", 0.1,
               60.0)
        self.goal_frame = self.declare_parameter("goal_frame", "odom").value
        self.path_topics = self.declare_parameter("path_topics", ["/exploration_path", "/global_path_full"]).value
        # far_planner's route (LINE_STRIP Marker): when set and published, the goal is taken on it, not from /way_point
        self.route_topic = self.declare_parameter("route_topic", "").value
        rate = self.declare_parameter("rate", 10.0).value

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.goal_pub = self.create_publisher(MarkerArray, "/opti_pessi/goal", 1)

        # Latest message (for /terrain_map: its frame and obstacle points (n, 3)) and receive time per input
        self.inputs = {}

        def keep(key):
            return lambda msg: self.inputs.__setitem__(key, (msg, self.get_clock().now()))

        def keep_obstacles(msg):
            # terrainAnalysis: intensity is the height above the ground
            points = point_cloud2.read_points_numpy(msg, field_names=["x", "y", "z", "intensity"], skip_nans=True)
            obstacles = points[points[:, 3] > self.param("obstacle_height"), :3].astype(float)
            self.inputs["/terrain_map"] = ((msg.header.frame_id, obstacles), self.get_clock().now())

        for topic in self.path_topics:
            self.create_subscription(Path, topic, keep(topic), 1)
        if self.route_topic:
            self.create_subscription(Marker, self.route_topic, keep(self.route_topic), 1)
        self.create_subscription(PointStamped, "/way_point", keep("/way_point"), 5)
        self.create_subscription(Odometry, "/odom", keep("/odom"), 5)
        self.create_subscription(PointCloud2, "/terrain_map", keep_obstacles, 1)
        self.hint = None  # last /goal_yaw_hint (PoseStamped), None when cleared
        self.create_subscription(PoseStamped, "/goal_yaw_hint", self.on_hint, 5)

        self.goal = None  # (2,) in goal_frame, its yaw there, and where it comes from
        self.goal_yaw = None
        self.goal_source = None
        self.published_goal = None
        self.published_yaw = None
        self.candidate = None  # last candidate, and since when it has not jumped
        self.candidate_since = None
        self.last_way_point = None
        self.create_timer(1.0 / rate, self.tick)

    def param(self, name):
        return self.get_parameter(name).value

    def on_hint(self, msg):
        self.hint = msg if msg.header.frame_id else None
        if self.hint is None:
            self.get_logger().info("goal yaw hint cleared")
        else:
            p = msg.pose.position
            self.get_logger().info(f"goal yaw hint at ({p.x:.2f}, {p.y:.2f}) in {msg.header.frame_id}: "
                                   f"{np.degrees(yaw_of(msg.pose.orientation)):.0f} deg")

    def fresh(self, key):
        msg, received = self.inputs.get(key, (None, None))
        if msg is None or self.get_clock().now() - received > Duration(seconds=self.param("input_timeout")):
            return None
        return msg

    def to_goal_frame(self, frame, points):
        """points (n, 3) from frame to goal_frame, with the latest TF (a planner message may be stamped ahead of the
        odom tree)."""
        if not frame or frame == self.goal_frame:
            return points
        tf = self.tf_buffer.lookup_transform(self.goal_frame, frame, Time()).transform
        t = tf.translation
        return points @ quat_to_matrix(tf.rotation).T + np.array([t.x, t.y, t.z])

    def yaw_to_goal_frame(self, frame, yaw):
        """yaw [rad] from frame to goal_frame, with the latest TF (frames turned about z only)."""
        if not frame or frame == self.goal_frame:
            return yaw
        return yaw + yaw_of(self.tf_buffer.lookup_transform(self.goal_frame, frame, Time()).transform.rotation)

    def path_goal(self, paths, robot, waypoint):
        """Pure-pursuit goal (2,) along the first path /way_point lies on and the path's yaw there, or None."""
        for path in paths:
            path = without_empty_segments(path)
            if len(path) < 2:
                continue
            found = follow(path, robot, waypoint, self.param("lookahead_distance"), self.param("waypoint_tolerance"),
                           self.param("max_waypoint_path_distance"), self.param("max_chord_deviation"))
            if found is not None:
                return found[0][:2], found[1]
        return None

    def choose(self, robot, robot_yaw, waypoint, path_goal, obstacles):
        """Candidate goal (2,), its yaw and where it comes from (see the module docstring)."""
        midpoint = 0.5 * (robot + waypoint)
        midpoint_goal = (midpoint, bearing(robot, waypoint, robot_yaw), "midpoint")
        if path_goal is not None:
            path_point, yaw = path_goal
            path_goal = (path_point, yaw, "path")
        if obstacles is None:
            return midpoint_goal if path_goal is None else path_goal
        clearance = self.param("clearance")
        if clear(robot, waypoint, obstacles, clearance):
            return midpoint_goal
        if path_goal is not None and clear(robot, path_goal[0], obstacles, clearance):
            return path_goal
        targets = ([] if path_goal is None else [path_goal[0]]) + [midpoint]
        return self.detour_or_stop(robot, robot_yaw, targets, obstacles)

    def choose_on_route(self, robot, robot_yaw, route, obstacles):
        """Candidate goal (2,) on far_planner's route (n, 3), its yaw and where it comes from (see the module
        docstring)."""
        route = without_empty_segments(route)
        if len(route) < 2:  # the robot is at the goal
            return route[0, :2], robot_yaw, "path"
        goals = [pursue(route, robot, lookahead, self.param("max_chord_deviation"))
                 for lookahead in relaxed(self.param("lookahead_distance"), self.param("min_detour_distance"))]
        goals = [(point[:2], yaw, "path") for point, yaw in goals]
        if obstacles is None:
            return goals[0]
        for goal in goals:
            if clear(robot, goal[0], obstacles, self.param("clearance")):
                return goal
        return self.detour_or_stop(robot, robot_yaw, [goals[0][0]], obstacles)

    def detour_or_stop(self, robot, robot_yaw, targets, obstacles):
        """Detour towards one of targets facing the way to it, or the robot itself keeping its yaw (see
        relaxed_detour())."""
        goal = self.relaxed_detour(robot, targets, obstacles)
        return (robot, robot_yaw, "stop") if goal is None else (goal, bearing(robot, goal, robot_yaw), "detour")

    def relaxed_detour(self, robot, targets, obstacles):
        """First detour (2,) towards one of targets, from detour_distance at clearance down to min_detour_distance at
        min_clearance, or None. Relaxed before stopping: a stop is final, the robot stands and sees the same obstacles
        from then on."""
        step, max_angle = np.radians(self.param("detour_step")), np.radians(self.param("max_detour_angle"))
        for clearance in relaxed(self.param("clearance"), self.param("min_clearance")):
            for distance in relaxed(self.param("detour_distance"), self.param("min_detour_distance")):
                for target in targets:
                    goal = detour(robot, target, obstacles, clearance, distance, step, max_angle)
                    if goal is not None:
                        return goal
        return None

    def tick(self):
        odom, way_point = self.fresh("/odom"), self.fresh("/way_point")
        route_msg = self.fresh(self.route_topic) if self.route_topic else None
        if route_msg is not None and len(route_msg.points) < 2:
            route_msg = None  # far_planner has no route now: /way_point says where to stand
        if odom is None or (way_point is None and route_msg is None):
            self.get_logger().info("waiting for /odom and /way_point", throttle_duration_sec=10.0)
            return
        terrain = self.fresh("/terrain_map")
        p = odom.pose.pose.position
        paths = []  # (n, 3) in goal_frame
        obstacles = route = waypoint = hint = None
        try:
            if self.hint is not None:
                h = self.hint.pose.position
                hint = (self.to_goal_frame(self.hint.header.frame_id, np.array([[h.x, h.y, h.z]]))[0, :2],
                        self.yaw_to_goal_frame(self.hint.header.frame_id, yaw_of(self.hint.pose.orientation)))
            robot = self.to_goal_frame(odom.header.frame_id, np.array([[p.x, p.y, p.z]]))[0]
            robot_yaw = self.yaw_to_goal_frame(odom.header.frame_id, yaw_of(odom.pose.pose.orientation))
            if route_msg is not None:
                route = self.to_goal_frame(route_msg.header.frame_id,
                                           np.array([[q.x, q.y, q.z] for q in route_msg.points]))
            else:
                w = way_point.point
                waypoint = self.to_goal_frame(way_point.header.frame_id, np.array([[w.x, w.y, w.z]]))[0, :2]
                for topic in self.path_topics:
                    path_msg = self.fresh(topic)
                    if path_msg is not None and len(path_msg.poses) >= 2:
                        path = np.array([[q.pose.position.x, q.pose.position.y, q.pose.position.z]
                                         for q in path_msg.poses])
                        paths.append(self.to_goal_frame(path_msg.header.frame_id, path))
            if terrain is not None:
                obstacles = self.to_goal_frame(*terrain)[:, :2]
        except TransformException as e:
            self.get_logger().warn(f"no TF to {self.goal_frame}, goal kept: {e}", throttle_duration_sec=5.0)
            return
        robot_z, robot = robot[2], robot[:2]

        if obstacles is None:
            self.get_logger().warn("no /terrain_map, goal not checked for obstacles", throttle_duration_sec=5.0)
        else:
            # Only obstacles a goal (at most /way_point, lookahead_distance, detour_distance or the current goal away)
            # can come near
            reach = max(0.0 if waypoint is None else np.linalg.norm(waypoint - robot), self.param("lookahead_distance"),
                        self.param("detour_distance"),
                        0.0 if self.goal is None else np.linalg.norm(self.goal - robot)) + self.param("clearance")
            distance = np.linalg.norm(obstacles - robot, axis=1)
            obstacles = obstacles[(distance > self.param("self_radius")) & (distance < reach)]
        if route is not None:
            candidate, candidate_yaw, source = self.choose_on_route(robot, robot_yaw, route, obstacles)
        else:
            candidate, candidate_yaw, source = self.choose(robot, robot_yaw, waypoint,
                                                           self.path_goal(paths, robot, waypoint), obstacles)
        if source == "stop":
            towards = "towards /way_point" if route is None else "along far_planner's route"
            self.get_logger().warn(
                f"no way {towards} keeps {self.param('min_clearance'):.2f} m from /terrain_map obstacles "
                f"(detours down to {self.param('min_detour_distance'):.2f} m): standing until it changes",
                throttle_duration_sec=5.0)

        # A new TARE waypoint, in TARE's frame: in goal_frame it moves with every odom -> lio_map update
        if waypoint is not None and self.last_way_point != (round(w.x, 1), round(w.y, 1)):
            self.last_way_point = (round(w.x, 1), round(w.y, 1))
            self.get_logger().info(f"/way_point ({waypoint[0]:.2f}, {waypoint[1]:.2f}), "
                                   f"{np.linalg.norm(waypoint - robot):.2f} m away")

        now = self.get_clock().now()
        if self.candidate is None or np.linalg.norm(candidate - self.candidate) > self.param("jump_distance"):
            self.candidate_since = now
        self.candidate = candidate
        keep_goal = (self.goal is not None
                     and now - self.candidate_since < Duration(seconds=self.param("switch_time"))
                     and np.linalg.norm(self.goal - robot) > self.param("reach_distance")
                     and (obstacles is None or clear(robot, self.goal, obstacles, self.param("clearance"))))
        if not keep_goal:
            if source != self.goal_source:
                self.get_logger().info(f"goal from {source}")
            self.goal, self.goal_yaw, self.goal_source = candidate, candidate_yaw, source
        if hint is not None and np.linalg.norm(self.goal - hint[0]) <= self.param("hint_radius"):
            self.goal_yaw = hint[1]

        if (self.published_goal is not None
                and np.linalg.norm(self.goal - self.published_goal) <= self.param("republish_distance")
                and abs(angle_difference(self.goal_yaw, self.published_yaw)) <= self.param("republish_yaw")):
            return
        self.published_goal, self.published_yaw = self.goal, self.goal_yaw
        marker = Marker()
        marker.header.frame_id = self.goal_frame
        marker.header.stamp = now.to_msg()
        marker.ns = "goal"
        marker.type = Marker.ARROW  # the goal yaw: acados turns to it near the goal, OptiPessi reads the position only
        marker.action = Marker.ADD
        marker.pose.position.x, marker.pose.position.y = map(float, self.goal)
        marker.pose.position.z = float(robot_z)
        marker.pose.orientation.z = float(np.sin(0.5 * self.goal_yaw))
        marker.pose.orientation.w = float(np.cos(0.5 * self.goal_yaw))
        marker.scale.x, marker.scale.y, marker.scale.z = 0.5, 0.1, 0.1
        marker.color.r, marker.color.g, marker.color.b = COLORS[self.goal_source]
        marker.color.a = 1.0
        self.goal_pub.publish(MarkerArray(markers=[marker]))


def main():
    rclpy.init()
    node = TareGoalBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
