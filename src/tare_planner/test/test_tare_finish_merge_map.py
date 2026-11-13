"""When tare_finish_merge_map.py ends an exploration TARE reports finished: once the robot is back home, or at once."""
import importlib.util
import os
import signal

import pytest
import rclpy
from nav_msgs.msg import Odometry
from std_msgs.msg import Bool

os.environ["ROS_DOMAIN_ID"] = "93"  # the node publishes a stop goal: never on the domain of a robot

_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts', 'tare_finish_merge_map.py')
_spec = importlib.util.spec_from_file_location('tare_finish_merge_map', _path)
script = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(script)

RETURN_HOME_TIMEOUT = 60.0


def odometry(x, y, z=0.47):
    msg = Odometry()
    msg.header.frame_id = 'camera_init'
    msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z = float(x), float(y), float(z)
    return msg


class Finish:
    """The node with a clock the test moves, and nothing of it reaching other processes."""

    def __init__(self, monkeypatch, return_home, params=()):
        self.clock = 1000.0
        monkeypatch.setattr(script.time, 'monotonic', lambda: self.clock)
        extra = [a for p in params for a in ('-p', p)]
        rclpy.init(args=['--ros-args', '-p', f'return_home:={str(return_home).lower()}', '-p', 'home_tolerance:=0.5',
                         '-p', f'return_home_timeout:={RETURN_HOME_TIMEOUT}', '-p', 'flush_time:=5.0', *extra])
        self.node = script.TareFinishMergeMap()
        self.signals = []
        self.stop_goals = []
        self.node.signal_nodes = lambda sig, names=None: self.signals.append((sig, names))
        self.node.goal_pub.publish = self.stop_goals.append
        self.node.start_merge = lambda: False
        self.node.start_graph_save = lambda: setattr(self.node, 'graph_state', 'done')
        self.node.odom = odometry(0.0, 0.0)  # /odom, where the stop goal is put

    def robot_at(self, x, y, z=0.47):
        self.node.on_lio_odometry(odometry(x, y, z))

    def exploration_finished(self):
        self.node.on_exploration_finish(Bool(data=True))

    def tick(self, seconds=0.5):
        self.clock += seconds
        self.node.tick()

    def stopped_tare(self):
        return (signal.SIGINT, None) in self.signals


@pytest.fixture
def finish(monkeypatch):
    made = []

    def make(return_home=True, params=()):
        made.append(Finish(monkeypatch, return_home, params))
        return made[0]

    yield make
    for f in made:
        f.node.destroy_node()
    rclpy.try_shutdown()


def test_tare_keeps_running_when_the_exploration_finishes_away_from_home(finish):
    f = finish()
    f.robot_at(0.0, 0.0)  # where the robot stands when the launch starts: home
    f.robot_at(8.0, 3.0)
    f.exploration_finished()
    f.tick()

    assert f.node.state == 'returning'
    assert not f.stopped_tare()  # TARE and the goal bridge walk the robot home
    assert f.stop_goals == []


def test_it_stops_and_merges_once_the_robot_is_back_home(finish):
    f = finish()
    f.robot_at(0.0, 0.0)
    f.robot_at(8.0, 3.0)
    f.exploration_finished()
    f.tick()
    f.robot_at(2.0, 1.0)
    f.tick()
    assert f.node.state == 'returning'

    f.robot_at(0.3, -0.2, z=1.1)  # FAST-LIO drifts in z: home is a place on the floor
    f.tick()

    assert f.node.state == 'stopping'
    assert f.stopped_tare()
    assert len(f.stop_goals) >= 1


def test_it_stops_where_the_robot_is_when_home_is_not_reached_in_time(finish):
    f = finish()
    f.robot_at(0.0, 0.0)
    f.robot_at(8.0, 3.0)
    f.exploration_finished()
    f.tick(RETURN_HOME_TIMEOUT - 1.0)
    assert f.node.state == 'returning'

    f.tick(2.0)

    assert f.node.state == 'stopping'
    assert f.stopped_tare()


def test_an_exploration_that_finishes_at_home_stops_at_once(finish):
    f = finish()
    f.robot_at(0.0, 0.0)
    f.robot_at(0.2, 0.1)
    f.exploration_finished()
    f.tick()

    assert f.node.state == 'stopping'
    assert f.stopped_tare()


def test_without_fast_lio_odometry_it_stops_at_once(finish):
    f = finish()
    f.exploration_finished()
    f.tick()

    assert f.node.state == 'stopping'
    assert f.stopped_tare()


def test_return_home_off_stops_at_once_wherever_the_robot_is(finish):
    f = finish(return_home=False)
    f.robot_at(0.0, 0.0)
    f.robot_at(8.0, 3.0)
    f.exploration_finished()

    assert f.node.state == 'stopping'
    assert f.stopped_tare()
    assert len(f.stop_goals) == 1


def test_after_the_stop_the_merge_starts_as_before(finish):
    f = finish()
    f.robot_at(0.0, 0.0)
    f.exploration_finished()
    f.tick()
    assert f.node.state == 'stopping'

    f.tick(6.0)  # flush_time

    assert (signal.SIGTERM, None) in f.signals
    assert f.node.finished()  # start_merge and the graph save are stubbed out: nothing left to wait for


class FakeProcess:
    def __init__(self, code=0):
        self.code = code
        self.pid = 4242

    def poll(self):
        return self.code


def merged_and_saved(f, tmp_path, merge_code=0):
    """start_merge and the graph save stubbed as if both had written their files in tmp_path."""
    def start_merge():
        f.node.directory = str(tmp_path)
        f.node.map_path = str(tmp_path / 'optimized_map.pcd')
        f.node.dynamic_path = str(tmp_path / 'optimized_map_dynamic.pcd')
        for path in (f.node.map_path, f.node.dynamic_path):
            open(path, 'w').close()
        f.node.merge = FakeProcess(merge_code)
        return True

    def start_graph_save():
        f.node.graph_path = str(tmp_path / 'warehouse.vgh')
        with open(f.node.graph_path, 'w') as graph:
            graph.write('2 3 0 0 0 0 0 0 0 0 0 1 0 1 0 | | | \n')
        f.node.graph_saved = True
        f.node.graph_state = 'done'

    f.node.start_merge = start_merge
    f.node.start_graph_save = start_graph_save


def run_to_the_end(f):
    f.robot_at(0.0, 0.0)
    f.exploration_finished()
    f.tick()
    for _ in range(4):
        f.tick(6.0)


def test_the_graph_is_pruned_with_the_merged_map_and_its_dynamic_points(finish, tmp_path, monkeypatch):
    f = finish()
    merged_and_saved(f, tmp_path)
    started = []
    monkeypatch.setattr(script.subprocess, 'Popen', lambda command, **kwargs: started.append(command) or FakeProcess())

    run_to_the_end(f)

    assert len(started) == 1
    assert started[0][1].endswith('pruneFarGraph.py')
    assert started[0][2:] == [str(tmp_path / 'warehouse.vgh'), str(tmp_path / 'optimized_map.pcd'),
                              str(tmp_path / 'optimized_map_dynamic.pcd')]
    assert os.path.isfile(tmp_path / 'warehouse_prune.log')
    assert f.node.finished()


def test_no_pruning_when_the_merge_failed(finish, tmp_path, monkeypatch):
    f = finish()
    merged_and_saved(f, tmp_path, merge_code=1)
    started = []
    monkeypatch.setattr(script.subprocess, 'Popen', lambda command, **kwargs: started.append(command) or FakeProcess())

    run_to_the_end(f)

    assert started == []
    assert f.node.finished()


def test_a_failed_pruning_still_ends_the_node(finish, tmp_path, monkeypatch):
    f = finish()
    merged_and_saved(f, tmp_path)
    monkeypatch.setattr(script.subprocess, 'Popen', lambda command, **kwargs: FakeProcess(code=2))

    run_to_the_end(f)

    assert f.node.prune_state == 'done'
    assert f.node.finished()


def exits_script(tmp_path):
    path = tmp_path / 'finalize_exits.py'
    path.write_text('')
    return str(path)


def test_exits_are_written_after_the_merge_into_the_save_directory(finish, tmp_path, monkeypatch):
    f = finish(params=['exits:=true', f'exits_script:={exits_script(tmp_path)}'])
    merged_and_saved(f, tmp_path)
    started = []
    monkeypatch.setattr(script.subprocess, 'Popen', lambda command, **kwargs: started.append(command) or FakeProcess())
    (tmp_path / 'exit_inspection').mkdir()
    (tmp_path / 'exit_inspection' / 'finalize.log').write_text('2 door exits confirmed\n')

    run_to_the_end(f)

    finalize = [c for c in started if c[1].endswith('finalize_exits.py')]
    assert finalize == [[script.sys.executable, exits_script(tmp_path), str(tmp_path)]]
    assert f.node.exits_state == 'done'
    assert f.node.finished()


def test_exits_are_written_even_when_the_merge_failed(finish, tmp_path, monkeypatch):
    """The sightings are re-anchored on SC-PGO's pose files, not on the merged map."""
    f = finish(params=['exits:=true', f'exits_script:={exits_script(tmp_path)}'])
    merged_and_saved(f, tmp_path, merge_code=1)
    started = []
    monkeypatch.setattr(script.subprocess, 'Popen', lambda command, **kwargs: started.append(command) or FakeProcess())

    run_to_the_end(f)

    assert [c[1] for c in started] == [exits_script(tmp_path)]
    assert f.node.finished()


def test_no_exits_by_default(finish, tmp_path, monkeypatch):
    f = finish()
    merged_and_saved(f, tmp_path)
    started = []
    monkeypatch.setattr(script.subprocess, 'Popen', lambda command, **kwargs: started.append(command) or FakeProcess())

    run_to_the_end(f)

    assert not any(c[1].endswith('finalize_exits.py') for c in started)
    assert f.node.exits_state == 'done'


def test_a_missing_exits_script_turns_exits_off(finish, tmp_path):
    f = finish(params=['exits:=true', f'exits_script:={tmp_path / "missing.py"}'])
    assert not f.node.exits
    assert f.node.exits_state == 'done'


def test_a_failed_finalize_still_ends_the_node(finish, tmp_path, monkeypatch):
    f = finish(params=['exits:=true', f'exits_script:={exits_script(tmp_path)}'])
    merged_and_saved(f, tmp_path)
    monkeypatch.setattr(script.subprocess, 'Popen', lambda command, **kwargs: FakeProcess(code=1))

    run_to_the_end(f)

    assert f.node.exits_state == 'done'
    assert f.node.finished()


def test_the_stop_goal_keeps_the_robots_yaw(finish):
    """acados turns to the yaw of its goal: the stop goal at the robot carries the robot's own yaw (a rotation about z
    only, whatever roll and pitch /odom has), not the identity quaternion, which would turn it towards +x."""
    import math
    f = finish(return_home=False)
    yaw, roll = 0.7, 0.05
    # yaw then roll (ZYX): q = q_z(yaw) * q_x(roll)
    cy, sy, cr, sr = math.cos(yaw / 2), math.sin(yaw / 2), math.cos(roll / 2), math.sin(roll / 2)
    q = f.node.odom.pose.pose.orientation
    q.w, q.x, q.y, q.z = cy * cr, cy * sr, sy * sr, sy * cr
    f.exploration_finished()

    marker = f.stop_goals[-1].markers[0]
    assert marker.type == marker.ARROW
    o = marker.pose.orientation
    assert (o.x, o.y) == (0.0, 0.0)
    assert (o.z, o.w) == pytest.approx((math.sin(yaw / 2), math.cos(yaw / 2)))


def test_home_comes_from_the_base_odometry_tare_plans_with(finish):
    """TARE's home is the first pose on its sub_state_estimation_topic_ (/lio_base_odometry, the base): the same here,
    not the lidar's /Odometry, 0.29 m ahead of the base, which put a robot home but turned round up to 0.58 m off."""
    f = finish()
    topics = {s.topic_name for s in f.node.subscriptions}
    assert '/lio_base_odometry' in topics
    assert '/Odometry' not in topics


# ---------------------------------------------------------------- exit sign sweep (exit_inspection exit_sign_sweep.py)

def with_sweep_listener(f, listening=True):
    sent = []
    f.node.sweep_pub.get_subscription_count = lambda: 1 if listening else 0
    f.node.sweep_pub.publish = sent.append
    return sent


def test_with_sweep_tare_stops_at_once_and_the_sweep_starts_with_home(finish):
    f = finish(params=('sweep:=true', 'sweep_timeout:=100.0'))
    sent = with_sweep_listener(f)
    f.robot_at(1.0, 2.0)  # home
    f.robot_at(8.0, 3.0)
    f.exploration_finished()
    assert f.node.state == 'sweeping'
    assert f.stopped_tare()  # TARE would walk the robot home over the sweep
    assert len(sent) == 1 and sent[0].data.startswith('home: [1.0, 2.0,')
    f.tick(50.0)
    assert f.node.state == 'sweeping' and f.stop_goals == []
    f.node.on_sweep_done(None)
    f.tick()
    assert f.node.state == 'stopping' and len(f.stop_goals) >= 1


def test_a_sweep_that_does_not_end_is_stopped_after_sweep_timeout(finish):
    f = finish(params=('sweep:=true', 'sweep_timeout:=100.0'))
    with_sweep_listener(f)
    f.robot_at(0.0, 0.0)
    f.exploration_finished()
    f.tick(99.0)
    assert f.node.state == 'sweeping'
    f.tick(2.0)
    assert f.node.state == 'stopping'
    assert (signal.SIGINT, ['exit_sweep_goal_bridge']) in f.signals


def test_without_a_sweep_node_the_exploration_ends_as_before(finish):
    f = finish(params=('sweep:=true',))
    sent = with_sweep_listener(f, listening=False)
    f.robot_at(0.0, 0.0)
    f.robot_at(8.0, 3.0)
    f.exploration_finished()
    assert sent == [] and f.node.state == 'returning'  # TARE brings the robot home
