# Copyright (c) 2024 PAL Robotics S.L. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import json
from math import cos, sin
from math import pi as PI
import os
import sys
import unittest
from typing import List

from emorobcare_cv_msgs.msg import MultiUserEngagementLevel
from geometry_msgs.msg import TransformStamped
from hri_actions_msgs.msg import Intent
from hri_msgs.msg import EngagementLevel, IdsList
import pandas as pd
import rclpy
from rclpy.duration import Duration
from rclpy.executors import Executor, SingleThreadedExecutor, TimeoutException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.qos import DurabilityPolicy, qos_profile_system_default
from rclpy.time import Time
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String
from tf_transformations import quaternion_from_euler
import tf2_ros

from hri_multiuser_engagement.multiuser_engagement_node import MultiUserEngagementNode

sys.path.insert(0, os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..')))

PKG = 'hri_multiuser_engagement'

OBSERVATION_WINDOW = 10.
FOV_DEG = 60.
FOV = FOV_DEG * PI / 180.
REFERENCE_FRAME = 'sellion_link'
NODE_RATE = 10.


@dataclass(frozen=True)
class Position:
    x: float = 0.
    y: float = 0.
    z: float = 0.


@dataclass(frozen=True)
class Rotation:
    roll: float = 0.
    pitch: float = 0.
    yaw: float = 0.


# human head positions and rotations relative to the REFERENCE_FRAME (robot sellion link)
HEAD_POS_FRONT = Position(x=1.)
HEAD_POS_LEFT = Position(x=cos(FOV), y=sin(FOV))
HEAD_POS_RIGHT = Position(x=cos(FOV), y=sin(-FOV))
HEAD_POS_SLIGHTLY_LEFT = Position(x=cos(FOV / 4), y=sin(FOV / 4))
HEAD_POS_SLIGHTLY_RIGHT = Position(x=cos(FOV / 4), y=sin(-FOV / 4))
HEAD_ROT_FRONT = Rotation(yaw=PI)
HEAD_ROT_LEFT = Rotation(yaw=PI - FOV)
HEAD_ROT_RIGHT = Rotation(yaw=PI + FOV)
HEAD_ROT_SLIGHTLY_LEFT = Rotation(yaw=PI - FOV / 4)
HEAD_ROT_SLIGHTLY_RIGHT = Rotation(yaw=PI + FOV / 4)

# human gaze rotations relative to the human head frame
GAZE_ROT_FRONT = Rotation(roll=-PI / 2, yaw=-PI / 2)

# allowed engagement level transitions (previous level, next level)
ENG_LVL_ALLOWED_TRANS = [
    (EngagementLevel.UNKNOWN, EngagementLevel.UNKNOWN),
    (EngagementLevel.UNKNOWN, EngagementLevel.DISENGAGED),
    (EngagementLevel.DISENGAGED, EngagementLevel.DISENGAGED),
    (EngagementLevel.DISENGAGED, EngagementLevel.ENGAGING),
    (EngagementLevel.ENGAGING, EngagementLevel.ENGAGING),
    (EngagementLevel.ENGAGING, EngagementLevel.DISENGAGED),
    (EngagementLevel.ENGAGING, EngagementLevel.ENGAGED),
    (EngagementLevel.ENGAGED, EngagementLevel.ENGAGED),
    (EngagementLevel.ENGAGED, EngagementLevel.DISENGAGING),
    (EngagementLevel.DISENGAGING, EngagementLevel.DISENGAGING),
    (EngagementLevel.DISENGAGING, EngagementLevel.ENGAGED),
    (EngagementLevel.DISENGAGING, EngagementLevel.DISENGAGED),
]


@dataclass(frozen=True)
class PersonState:
    engaged: bool = True
    head_pos: Position = HEAD_POS_FRONT
    head_rot: Rotation = HEAD_ROT_FRONT
    gaze_rot: Rotation = GAZE_ROT_FRONT
    duration: float = 1.5 * OBSERVATION_WINDOW
    movement: str = 'static'


PersonData = List[PersonState]


def spin_some(executor: Executor, timeout=timedelta(seconds=1.)):
    start = datetime.now()
    # get first available task without waiting
    cb_iter = executor._wait_for_ready_callbacks(timeout_sec=0.)

    def spin_some_cb(timeout):
        nonlocal cb_iter, start
        while True:
            try:
                handler, *_ = next(cb_iter)
                handler()
                if handler.exception() is not None:
                    raise handler.exception()
            except TimeoutException:
                elapsed = datetime.now() - start
                if elapsed > timeout:
                    raise TimeoutException(
                        f'Time elapsed spinning {elapsed} with timeout {timeout}')
            except StopIteration:
                break

    # run twice to ensure simulated clock timers are executed:
    # clock is updated on the first run with a subscriber callback,
    # and timer callbacks are evaluated before subscriber callbacks
    # https://docs.ros.org/en/humble/Concepts/Intermediate/About-Executors.html#scheduling-semantics
    spin_some_cb(timeout)
    # spin_some_cb(timeout)


class SynthDataPublisher(Node):
    def __init__(self, node: Node):
        self.node = node
        self.run_count = 0

    def open(self, data: List[PersonData]):
        self.df = self.build_persons_data(data)
        self.persons_ids = self.df.columns.get_level_values(0).unique()
        self.time = Time()

        self.clock_pub = self.node.create_publisher(Clock, '/clock', 1)

        self.persons_tracked_pub = self.node.create_publisher(
            IdsList, '/humans/persons/tracked', 1)
        self.faces_tracked_pub = self.node.create_publisher(
            IdsList, '/humans/faces/tracked', 1)
        self.intents_gt_pub = self.node.create_publisher(
            Intent, '/intents/gt', len(self.persons_ids))
        self.tf_br = tf2_ros.TransformBroadcaster(self.node)

        qos_latched = qos_profile_system_default
        qos_latched.depth = 1
        qos_latched.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.face_id_pub = dict([])
        self.engagement_status_gt_pub = {}
        for id in self.persons_ids:
            self.face_id_pub[id] = self.node.create_publisher(
                String, f'/humans/persons/{id}/face_id', qos_latched)
            self.engagement_status_gt_pub[id] = self.node.create_publisher(
                EngagementLevel, f'/humans/persons/{id}/engagement_status/gt', qos_latched)

    @staticmethod
    def build_persons_data(data: List[PersonData]) -> pd.DataFrame:
        df = pd.DataFrame()
        df_persons = []
        for person_data in data:
            curr_timestamp = pd.Timestamp(0)
            df_states = []
            for state in person_data:
                end_timestamp = curr_timestamp + pd.Timedelta(seconds=state.duration)
                df_state = pd.json_normalize(asdict(state)).drop(columns=['duration'])
                df_state.index = [end_timestamp]
                if state.movement == 'static':
                    index = pd.date_range(
                        start=curr_timestamp, end=end_timestamp,
                        freq=pd.DateOffset(seconds=1 / NODE_RATE))
                    df_state = df_state.reindex(index=index, method='backfill')
                # TODO: add other kind of movement, for instance adding noise and/or interpolating
                # towards the next PersonState
                df_state['tracked'] = [True] * df_state.index.size
                start_check_timestamp = curr_timestamp + pd.Timedelta(
                    seconds=OBSERVATION_WINDOW * 1.1)
                df_state['tested'] = [
                    timestamp > start_check_timestamp for timestamp in df_state.index]
                df_states.append(df_state)
                curr_timestamp = end_timestamp
            df_persons.append(pd.concat(df_states))
        df = pd.concat(df_persons, axis=1, keys=[f'p{i}' for i in range(len(data))])
        return df

    def run(self):
        df_curr = self.df.iloc[self.run_count]
        ids_tracked = [
            id for id in self.persons_ids if df_curr[id, 'tracked'] is True]
        ids_tested = [id for id in self.persons_ids if df_curr[id, 'tested'] is True]
        ids_tracked_mgs = IdsList(ids=ids_tracked)
        self.persons_tracked_pub.publish(ids_tracked_mgs)
        self.faces_tracked_pub.publish(ids_tracked_mgs)
        for id in ids_tracked:
            df_person_state = df_curr[id]

            self.face_id_pub[id].publish(String(data=str(id)))
            tf = TransformStamped()

            tf.header.stamp = self.time.to_msg()
            tf.header.frame_id = REFERENCE_FRAME
            tf.child_frame_id = f'face_{id}'
            tf.transform.translation.x = df_person_state['head_pos.x']
            tf.transform.translation.y = df_person_state['head_pos.y']
            tf.transform.translation.z = df_person_state['head_pos.z']
            q = quaternion_from_euler(
                df_person_state['head_rot.roll'],
                df_person_state['head_rot.pitch'],
                df_person_state['head_rot.yaw'],
            )
            tf.transform.rotation.x = q[0]
            tf.transform.rotation.y = q[1]
            tf.transform.rotation.z = q[2]
            tf.transform.rotation.w = q[3]
            self.tf_br.sendTransform(tf)

            tf.header.stamp = self.time.to_msg()
            tf.header.frame_id = f'face_{id}'
            tf.child_frame_id = f'gaze_{id}'
            tf.transform.translation.x = 0.0
            tf.transform.translation.y = 0.0
            tf.transform.translation.z = 0.0
            q = quaternion_from_euler(
                df_person_state['gaze_rot.roll'],
                df_person_state['gaze_rot.pitch'],
                df_person_state['gaze_rot.yaw'],
            )
            tf.transform.rotation.x = q[0]
            tf.transform.rotation.y = q[1]
            tf.transform.rotation.z = q[2]
            tf.transform.rotation.w = q[3]
            self.tf_br.sendTransform(tf)

        for id in ids_tested:
            df_person_state = df_curr[id]
            if df_person_state['engaged'] is True:
                engagement_msg = EngagementLevel()
                engagement_msg.header.stamp = self.node.get_clock().now().to_msg()
                engagement_msg.level = EngagementLevel.ENGAGED
                self.engagement_status_gt_pub[id].publish(engagement_msg)

                intent_msg = Intent()
                intent_msg.intent = intent_msg.ENGAGE_WITH
                intent_msg.data = f'{{"recipient":"{id}"}}'
                self.intents_gt_pub.publish(intent_msg)
            else:
                engagement_msg = EngagementLevel()
                engagement_msg.header.stamp = self.node.get_clock().now().to_msg()
                engagement_msg.level = EngagementLevel.DISENGAGED
                self.engagement_status_gt_pub[id].publish(engagement_msg)

        self.clock_pub.publish(Clock(clock=self.time.to_msg()))
        self.time = self.time + Duration(seconds=(1/NODE_RATE))

        self.run_count += 1
        data_exhausted = self.run_count >= len(self.df.index)
        return data_exhausted

    def close(self):
        self.node.destroy_publisher(self.clock_pub)
        self.node.destroy_publisher(self.persons_tracked_pub)
        self.node.destroy_publisher(self.faces_tracked_pub)
        self.node.destroy_publisher(self.intents_gt_pub)
        del self.tf_br
        for publisher in self.face_id_pub.values():
            self.node.destroy_publisher(publisher)
        for publisher in self.engagement_status_gt_pub.values():
            self.node.destroy_publisher(publisher)


class PersonUnderTest:
    def __init__(self, id: String, node: Node, test_case: unittest.TestCase):
        self.id = id
        self.node = node
        self.test_case = test_case
        self.engagement_level = EngagementLevel.UNKNOWN
        self.engagement_level_gt = None
        self.is_intent_engaged = False
        self.is_intent_engaged_gt = None
        self.engagement_status_sub = self.node.create_subscription(
            EngagementLevel, f'/humans/persons/{id}/engagement_status',
            lambda msg: self._on_engagement_level(msg), 1)
        self.engagement_status_gt_sub = self.node.create_subscription(
            EngagementLevel, f'/humans/persons/{id}/engagement_status/gt',
            lambda msg: self._on_engagement_level_gt(msg), 1)
        self.intents_sub = self.node.create_subscription(
            Intent, '/intents', lambda msg: self._on_intents(msg), 10)
        self.intents_gt_sub = self.node.create_subscription(
            Intent, '/intents/gt', lambda msg: self._on_intents_gt(msg), 10)
        self.multiuser_engagement_sub = self.node.create_subscription(
            MultiUserEngagementLevel, '/humans/engagement/multiuser',
            lambda msg: self._on_multiuser_engagement(msg), 10)

    def close(self):
        self.node.destroy_subscription(self.engagement_status_sub)
        self.node.destroy_subscription(self.engagement_status_gt_sub)
        self.node.destroy_subscription(self.intents_sub)
        self.node.destroy_subscription(self.intents_gt_sub)
        self.node.destroy_subscription(self.multiuser_engagement_sub)

    def _on_engagement_level(self, msg: EngagementLevel):
        self.test_case.assertIn(
            (self.engagement_level, msg.level), ENG_LVL_ALLOWED_TRANS
        )
        self.engagement_level = msg.level

    def _on_engagement_level_gt(self, msg: EngagementLevel):
        self.engagement_level_gt = msg.level

    def _on_intents(self, msg: Intent):
        data = json.loads(msg.data)
        # Handle both single-user format (only 'recipient') and multiuser format ('recipient' + 'target')
        if msg.intent == Intent.ENGAGE_WITH and data.get('recipient') == str(self.id):
            self.is_intent_engaged = True

    def _on_intents_gt(self, msg: Intent):
        data = json.loads(msg.data)
        if msg.intent == Intent.ENGAGE_WITH and data.get('recipient') == str(self.id):
            self.is_intent_engaged_gt = True

    def _on_multiuser_engagement(self, msg: MultiUserEngagementLevel):
        # Verify multiuser engagement messages are correctly published
        if msg.person_id == str(self.id):
            # Verify that level matches the per-person engagement_status topic
            self.test_case.assertEqual(msg.level, self.engagement_level)


class EngagementTester:
    def __init__(self, node: Node, test_case: unittest.TestCase):
        self.node = node
        self.test_case = test_case
        self.persons_under_test = {}
        self.persons_tracked_sub = self.node.create_subscription(
            IdsList, '/humans/persons/tracked', lambda msg: self._on_persons_tracked(msg), 1)

    def run(self):
        for person in self.persons_under_test.values():
            if person.engagement_level_gt:
                self.test_case.assertEqual(person.engagement_level, person.engagement_level_gt)
                person.engagement_level_gt = None
            if person.is_intent_engaged_gt:
                self.test_case.assertEqual(person.is_intent_engaged, person.is_intent_engaged_gt)
                person.is_intent_engaged_gt = None

    def close(self):
        for person in self.persons_under_test.values():
            person.close()
        self.node.destroy_subscription(self.persons_tracked_sub)

    def _on_persons_tracked(self, msg: IdsList):
        ids_tracked_new = set(msg.ids)
        ids_tracked = self.persons_under_test.keys()
        ids_to_remove = ids_tracked - ids_tracked_new
        ids_to_add = ids_tracked_new - ids_tracked

        for id in ids_to_remove:
            self.persons_under_test[id].close()
            del self.persons_under_test[id]

        for id in ids_to_add:
            self.persons_under_test[id] = PersonUnderTest(id, self.node, self.test_case)


class GenericTestSequence(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rclpy.init()
        cls.hri_engagement_node = MultiUserEngagementNode()
        cls.hri_engagement_node.set_parameters([
            Parameter('observation_window', Parameter.Type.DOUBLE, OBSERVATION_WINDOW),
            Parameter('field_of_view', Parameter.Type.DOUBLE, FOV_DEG),
            Parameter('reference_frame', Parameter.Type.STRING, REFERENCE_FRAME),
            Parameter('rate', Parameter.Type.DOUBLE, NODE_RATE),
            Parameter('use_sim_time', Parameter.Type.BOOL, True)])
        cls.hri_engagement_executor = SingleThreadedExecutor()
        cls.hri_engagement_executor.add_node(cls.hri_engagement_node)
        cls.hri_engagement_node.trigger_configure()
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        rclpy.shutdown()
        return super().tearDownClass()

    def setUp(self) -> None:
        self.hri_engagement_node.trigger_activate()
        self.tester_node = rclpy.create_node('tester')
        self.tester_executor = SingleThreadedExecutor()
        self.tester_executor.add_node(self.tester_node)
        self.synth_pub = SynthDataPublisher(self.tester_node)
        self.tester = EngagementTester(self.tester_node, self)
        return super().setUp()

    def tearDown(self) -> None:
        self.hri_engagement_node.trigger_deactivate()
        self.synth_pub.close()
        self.tester.close()
        self.tester_executor.remove_node(self.tester_node)
        self.tester_node.destroy_node()
        del self.tester_node
        return super().tearDown()

    def _test(self, data):
        self.synth_pub.open(data)
        data_exhausted = False
        while (rclpy.ok() and not data_exhausted):
            data_exhausted = self.synth_pub.run()
            spin_some(self.hri_engagement_executor)
            spin_some(self.tester_executor)
            self.tester.run()


class TestSynthData(GenericTestSequence):
    def test_one_person_in_front(self):
        self._test([[PersonState()]])

    def test_one_person_look_slightly_left(self):
        self._test([[PersonState(head_rot=HEAD_ROT_SLIGHTLY_LEFT)]])

    def test_one_person_look_slightly_right(self):
        self._test([[PersonState(head_rot=HEAD_ROT_SLIGHTLY_RIGHT)]])

    def test_one_person_look_left(self):
        self._test([[PersonState(engaged=False, head_rot=HEAD_ROT_LEFT)]])

    def test_one_person_look_right(self):
        self._test([[PersonState(engaged=False, head_rot=HEAD_ROT_RIGHT)]])

    def test_one_person_out_fov_left(self):
        self._test([[PersonState(engaged=False, head_pos=HEAD_POS_LEFT, head_rot=HEAD_ROT_RIGHT)]])

    def test_one_person_out_fov_right(self):
        self._test([[PersonState(engaged=False, head_pos=HEAD_POS_RIGHT, head_rot=HEAD_ROT_LEFT)]])

    def test_two_people_in_front(self):
        self._test([
            [PersonState(head_pos=HEAD_POS_SLIGHTLY_LEFT, head_rot=HEAD_ROT_SLIGHTLY_RIGHT)],
            [PersonState(
                head_pos=HEAD_POS_SLIGHTLY_RIGHT, head_rot=HEAD_ROT_SLIGHTLY_LEFT, duration=8)]])

    def test_two_people_one_out_fov(self):
        self._test([
            [PersonState()],
            [PersonState(engaged=False, head_pos=HEAD_POS_LEFT, head_rot=HEAD_ROT_RIGHT)]])

    def test_two_people_one_look_away(self):
        self._test([
            [PersonState(engaged=False, head_rot=HEAD_ROT_LEFT)],
            [PersonState(head_pos=HEAD_POS_SLIGHTLY_LEFT, head_rot=HEAD_ROT_SLIGHTLY_RIGHT)]])

    def test_one_person_swipe_look(self):
        self._test([
            [PersonState(engaged=False, head_rot=HEAD_ROT_LEFT),
             PersonState(),
             PersonState(engaged=False, head_rot=HEAD_ROT_RIGHT)]])

    def test_one_person_move_across(self):
        self._test([
            [PersonState(engaged=False, head_pos=HEAD_POS_LEFT, head_rot=HEAD_ROT_RIGHT),
             PersonState(),
             PersonState(engaged=False, head_pos=HEAD_POS_RIGHT, head_rot=HEAD_ROT_LEFT)]])
