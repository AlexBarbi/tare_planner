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

    def __init__(self, monkeypatch, return_home):
        self.clock = 1000.0
        monkeypatch.setattr(script.time, 'monotonic', lambda: self.clock)
        rclpy.init(args=['--ros-args', '-p', f'return_home:={str(return_home).lower()}', '-p', 'home_tolerance:=0.5',
                         '-p', f'return_home_timeout:={RETURN_HOME_TIMEOUT}', '-p', 'flush_time:=5.0'])
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

    def make(return_home=True):
        made.append(Finish(monkeypatch, return_home))
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
