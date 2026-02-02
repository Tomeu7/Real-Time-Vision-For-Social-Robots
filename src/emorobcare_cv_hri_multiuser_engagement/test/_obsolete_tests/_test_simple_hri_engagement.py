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

import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

import rclpy
from rclpy.executors import Executor, SingleThreadedExecutor, TimeoutException
from rclpy.parameter import Parameter
from rclpy.time import Time
from rosbag2_py import ConverterOptions, SequentialReader, StorageOptions
from rosgraph_msgs.msg import Clock


from hri import HRIListener
from hri_msgs.msg import EngagementLevel, IdsList
from tf2_msgs.msg import TFMessage
from std_msgs.msg import String
from hri_engagement.engagement_node import EngagementNode, PersonEngagement


PKG = 'test_hri_engagement'


def spin_some(executor: Executor, timeout=timedelta(seconds=10.)):
    start = datetime.now()
    # get first available task without waiting
    cb_iter = executor._wait_for_ready_callbacks(timeout_sec=0.)
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


class TestEngagementMixin():

    @classmethod
    def setUpClass(cls) -> None:
        rclpy.init()
        cls.engagement_node = EngagementNode()
        cls.engagement_node.set_parameters([
            Parameter(name='reference_frame', value=cls.reference_frame),
            Parameter(name='max_distance', value=cls.max_distance),
            Parameter(name='field_of_view', value=cls.field_of_view),
            Parameter(name='use_sim_time', value=True)])
        cls.engagement_executor = SingleThreadedExecutor()
        cls.engagement_executor.add_node(cls.engagement_node)
        cls.engagement_node.trigger_configure()

        cls.latching_qos = rclpy.qos.QoSProfile(
            depth=1, durability=rclpy.qos.QoSDurabilityPolicy.TRANSIENT_LOCAL)
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        cls.engagement_node.destroy_node()
        rclpy.shutdown()
        return super().tearDownClass()

    def addPerson(self, person_id: str):

        self.tester_node.get_logger().info(
            f"New person {person_id}")
        self.persons_pub[person_id] = self.tester_node.create_publisher(
            String, f'/humans/persons/{person_id}/face_id', qos_profile=self.latching_qos)

        self.publishers_map[f'/humans/persons/{person_id}/face_id'] = self.persons_pub[person_id]

    def setUp(self) -> None:
        self.tester_node = rclpy.create_node('tester_node')
        self.clock_pub = self.tester_node.create_publisher(Clock, '/clock', 1)

        h_p_t_pub = self.tester_node.create_publisher(
            IdsList, '/humans/persons/tracked', 1)

        h_f_t_pub = self.tester_node.create_publisher(
            IdsList, '/humans/faces/tracked', 1)

        tf_pub = self.tester_node.create_publisher(
            TFMessage, '/tf', 1)

        self.persons_pub = dict()

        self.publishers_map = {
            '/humans/persons/tracked': h_p_t_pub,
            '/humans/faces/tracked': h_f_t_pub,
            '/tf': tf_pub,
        }

        self.faces = ["f00000", "f00001", "f00002"]

        for f in self.faces:
            self.addPerson("person_" + f)

        self.tester_executor = SingleThreadedExecutor()
        self.tester_executor.add_node(self.tester_node)

        self.hri_listener = HRIListener(
            'hri_listener_test_node', False)  # no auto-spin

        self.engagement_node.trigger_activate()

        # wait for the publishers to be ready
        time.sleep(0.5)

        # pre-publish all the faces and persons (linked to the faces) that we
        # expect to see
        h_f_t_pub.publish(IdsList(ids=self.faces))
        h_p_t_pub.publish(
            IdsList(ids=["person_" + f for f in self.faces]))

        for f in self.faces:
            self.persons_pub["person_" + f].publish(String(data=f))

        return super().setUp()

    def tearDown(self) -> None:

        self.tester_node.get_logger().info("Tearing down")

        self.engagement_node.trigger_deactivate()

        del self.hri_listener

        self.tester_executor.remove_node(self.tester_node)
        for pub in self.publishers_map.values():
            self.tester_node.destroy_publisher(pub)
        self.tester_node.destroy_node()

        return super().tearDown()

    def spin(self, time_ns=None):
        if time_ns is not None:
            self.clock_pub.publish(
                Clock(clock=Time(nanoseconds=time_ns).to_msg()))
        spin_some(self.engagement_executor)
        self.hri_listener.spin_some(timedelta(seconds=1.))
        spin_some(self.tester_executor)

    def _test(self, bag_path: Path, expected_engagement_timeline: dict[str, dict[int, EngagementLevel]]):
        bag_reader = SequentialReader()
        print(str(bag_path))
        bag_reader.open(StorageOptions(uri=str(bag_path)),
                        ConverterOptions('', ''))

        self.spin()

        expected_persons = ["person_" +
                            f for f in expected_engagement_timeline.keys()]

        self.assertCountEqual(self.faces, self.hri_listener.faces.keys())
        self.assertCountEqual(
            expected_persons, self.hri_listener.tracked_persons.keys())

        seen_active_persons = set()

        nb_engagement_samples = 0

        try:
            while bag_reader.has_next():
                topic, msg_raw, time_ns = bag_reader.read_next()

                self.publishers_map[topic].publish(msg_raw)
                self.spin(time_ns)

                seen_active_persons |= set(
                    self.engagement_node.active_persons.keys())

                print(
                    self.hri_listener.tracked_persons["person_f00000"].engagement_status)

                if "person_f00000" in self.engagement_node.active_persons:
                    print(len(
                        self.engagement_node.active_persons["person_f00000"].person_engagement_history))
        except Exception as e:
            # print full traceback
            import traceback
            traceback.print_exc()
            raise e

        print("Bag file complete")

        self.assertCountEqual(expected_persons, seen_active_persons)
        # self.assertEquals(voice.speech, expected_final)


class TestSimpleEngagement(TestEngagementMixin, unittest.TestCase):
    reference_frame = 'camera_link'
    max_distance = 4.0
    field_of_view = 60.0
    bags_path = Path().cwd() / 'test' / 'data'

    def test_full__engagement(self):
        self._test(self.bags_path / 'bag_1_engaged',
                   {"f00000": {0: EngagementLevel.ENGAGED},
                    "f00001": {0: EngagementLevel.UNKNOWN},
                    "f00002": {0: EngagementLevel.UNKNOWN}
                    })

    def test_full_disengagement(self):
        self._test(self.bags_path / 'bag_2_disengaged',
                   {"f00000": {0: EngagementLevel.DISENGAGED},
                    "f00001": {0: EngagementLevel.UNKNOWN},
                    "f00002": {0: EngagementLevel.UNKNOWN}
                    })

#
# class TestHRIEngagement(unittest.TestCase):
#
#    def setUp(self):
#        self.directory = os.path.dirname(
#            os.path.realpath(__file__)) + "/simple_bags/"
#        self.expected_result = None
#        self.engagement_value = EngagementLevel.UNKNOWN
#        self.person_tracked = dict()
#        self.person_id = dict()
#        self.face_id = dict()
#        self.reference_frame = "camera_link"
#        engagement_history_size = NODE_RATE * BUFFER_DURATION
#        self.frame_to_skip = engagement_history_size + 10
#        self.visual_social_engagement_thr = VISUAL_SOCIAL_ENGAGEMENT_THR
#        self.rosbag_files = list()
#        self.tfBuffer = tf2_ros.Buffer()
#        self.listener = tf2_ros.TransformListener(self.tfBuffer)
#        # static tf
#        t_camera = geometry_msgs.msg.TransformStamped()
#        t_camera.header.stamp = rospy.Time.now()
#        t_camera.header.frame_id = "camera_link"
#        t_camera.child_frame_id = "camera_color_optical_frame"
#        t_camera.transform.translation.x = 0
#        t_camera.transform.translation.y = 0
#        t_camera.transform.translation.z = 0
#        t_camera.transform.rotation.x = -0.5
#        t_camera.transform.rotation.y = 0.5
#        t_camera.transform.rotation.z = -0.5
#        t_camera.transform.rotation.w = 0.5
#        self.publish_static_tf(t_camera.header.frame_id,
#                               t_camera.child_frame_id, t_camera)
#
#    def tearDown(self):
#        self.tfBuffer = None
#        self.listener = None
#
#    @staticmethod
#    def publish_static_tf(source_frame, target_frame, transform):
#        """
#        method that publishes the tf from reference_frame to child_frame
#        """
#        br = tf2_ros.StaticTransformBroadcaster()
#        print(f'static transformation from {source_frame} to {target_frame}')
#        br.sendTransform(transform)
#
#    @staticmethod
#    def publish_tf(source_frame, target_frame, transform):
#        """
#        it publishes the transform from the source_frame to target_frame
#        """
#        br = tf2_ros.TransformBroadcaster()
#        print(f' transformation from {source_frame} to {target_frame}')
#        br.sendTransform(transform)
#
#    def test_engagement_one_person(self):
#        """
#        it tests the engagement of one person
#        """
#        f = open(self.directory + 'list_of_bags.json')
#        data = json.load(f)
#        engaged_data = data['single_person']['engaged']
#
#        # extract person_id, rosbag_name, and expected result
#        for person in engaged_data:
#            if person == "person_id":
#                self.persons_id = engaged_data[person]
#            if person == "rosbag":
#                self.rosbag_files = engaged_data[person]
#            if person == 'expected_result':
#                self.expected_result = \
#                    [(int(engaged_data[person]))][0]
#
#        assert len(self.persons_id) == len(self.rosbag_files), \
#            f'n_persons need to be equal to n_rosbag_files'
#
#        person_instance = pyhri.Person("",
#                                       self.tfBuffer,
#                                       self.reference_frame)
#        face_instance = pyhri.Face("",
#                                   self.tfBuffer,
#                                   self.reference_frame)
#
#        face_id = ""
#        tf_rcamera_to_hface = None
#        tf_hface_to_hgaze = None
#
#        # run the unittest for one rosbag
#        for i in range(len(self.rosbag_files)):
#            engagement_node = \
#                EngagementNode(self.visual_social_engagement_thr,
#                               self.reference_frame)
#            person_instance.id = (self.persons_id[i])
#            rospy.loginfo(f'Processing file ...'
#                          f'{self.directory + self.rosbag_files[i]}')
#            bag = rosbag.Bag(self.directory + self.rosbag_files[i],
#                             'r', allow_unindexed=True)
#            iterator = bag.read_messages(topics=["/humans/persons/tracked",
#                                                 "/humans/persons/"
#                                                 + self.persons_id[i] +
#                                                 "/face_id", "/tf",
#                                                 "/tf_static"])
#            self.tf_rcamera_to_hface = None
#            self.tf_hface_to_hgaze = None
#
#            # we need to skip the first NODE_RATE * BUFFER_DURATION frames as
#            # the algorithm is still buffering plus the addional 4 frame to give it
#            # the possibility to swich from UNKOWN to ENGAGED.
#            frame_counter = 0
#
#            msgs = [m for m in iterator]
#            prev_timestamp = msgs[0].timestamp
#
#            for (topic, msg, t) in msgs:
#                # get the tracked persons from the topic
#                if topic == "/humans/persons/tracked":
#                    self.person_tracked = msg
#                    for id in self.person_tracked.ids:
#                        engagement_node.hri_listener._tracked_persons[id] = \
#                            person_instance
#
#                # for each tracked person get the face_id
#                # (we need that to compute the engagement)
#                if topic == "/humans/persons/" + self.persons_id[i] \
#                        + "/face_id":
#                    face_id = msg
#                    face_instance.id = face_id.data
#                    face_instance.frame = "face_" + face_id.data
#                    face_instance.gaze_frame = "gaze_" + face_id.data
#                    person_instance.face = face_instance
#
#                # get the tfs and republish them
#                if topic == "/tf":
#                    if msg.transforms[0].header.frame_id == \
#                            "camera_color_optical_frame" and \
#                            msg.transforms[0].child_frame_id == "face_" \
#                            + face_id.data:
#                        # compute the transformation
#                        tf_rcamera_to_hface = msg.transforms[0]
#                    if msg.transforms[0].header.frame_id == "face_" \
#                            + face_id.data and \
#                            msg.transforms[0].child_frame_id == "gaze_" \
#                            + face_id.data:
#                        # compute the transformation
#                        tf_hface_to_hgaze = msg.transforms[0]
#
#                    if tf_rcamera_to_hface is not None and \
#                            tf_hface_to_hgaze is not None:
#                        tf_rcamera_to_hface.header.stamp = rospy.Time.now()
#                        self.publish_tf(tf_rcamera_to_hface.header.frame_id,
#                                        tf_rcamera_to_hface.child_frame_id,
#                                        tf_rcamera_to_hface)
#                        tf_hface_to_hgaze.header.stamp = rospy.Time.now()
#                        self.publish_tf(tf_hface_to_hgaze.header.frame_id,
#                                        tf_hface_to_hgaze.child_frame_id,
#                                        tf_hface_to_hgaze)
#
#                        rospy.sleep(t-prev_timestamp)
#                        prev_timestamp = t
#
#                        try:
#                            engagement_node.get_tracked_humans()
#                            if frame_counter < self.frame_to_skip and engagement_node.tracked_persons_in_the_scene:
#                                frame_counter += 1
#                            else:
#                                self.assertEqual(self.expected_result,
#                                                 engagement_node.active_persons[self.persons_id[i]].person_current_engagement_level)
#
#                        except (tf2_ros.LookupException,
#                                tf2_ros.ConnectivityException,
#                                tf2_ros.ExtrapolationException) as e:
#                            rospy.logwarn(f'Exception in tf2 {e}')
#
#            bag.close()
#
#    def test_disengagement_one_person(self):
#        """
#        it tests the disengagement of one person
#        """
#        f = open(self.directory + 'list_of_bags.json')
#        data = json.load(f)
#        disengaged_data = data['single_person']['disengaged']
#
#        # extract person_id, rosbag_name, and expected result
#        for person in disengaged_data:
#            if person == "person_id":
#                self.persons_id = disengaged_data[person]
#            if person == "rosbag":
#                self.rosbag_files = disengaged_data[person]
#            if person == 'expected_result':
#                self.expected_result = \
#                    [(int(disengaged_data[person]))][0]
#
#        assert len(self.persons_id) == len(self.rosbag_files), \
#            f'n_persons need to be equal to n_rosbag_files'
#
#        person_instance = pyhri.Person("",
#                                       self.tfBuffer,
#                                       self.reference_frame)
#        face_instance = pyhri.Face("",
#                                   self.tfBuffer,
#                                   self.reference_frame)
#
#        face_id = ""
#        tf_rcamera_to_hface = None
#        tf_hface_to_hgaze = None
#
#        # run the unittest for one rosbag
#        for i in range(len(self.rosbag_files)):
#            engagement_node = \
#                EngagementNode(self.visual_social_engagement_thr,
#                               self.reference_frame)
#            person_instance.id = (self.persons_id[i])
#            rospy.loginfo(f'Processing file ...'
#                          f'{self.directory + self.rosbag_files[i]}')
#            bag = rosbag.Bag(self.directory + self.rosbag_files[i],
#                             'r', allow_unindexed=True)
#            iterator = bag.read_messages(topics=["/humans/persons/tracked",
#                                                 "/humans/persons/"
#                                                 + self.persons_id[i] +
#                                                 "/face_id", "/tf",
#                                                 "/tf_static"])
#
#            # we need to skip the first NODE_RATE * BUFFER_DURATION frames as
#            # the algorithm is still buffering plus the addional 4 frame to give it
#            # the possibility to swich from UNKOWN to ENGAGED.
#            frame_counter = 0
#
#            msgs = [m for m in iterator]
#            prev_timestamp = msgs[0].timestamp
#
#            for (topic, msg, t) in msgs:
#                # get the tracked persons from the topic
#                if topic == "/humans/persons/tracked":
#                    self.person_tracked = msg
#                    for id in self.person_tracked.ids:
#                        engagement_node.hri_listener._tracked_persons[id] = \
#                            person_instance
#
#                # for each tracked person get the face_id
#                # (we need that to compute the engagement)
#                if topic == "/humans/persons/" + self.persons_id[i] \
#                        + "/face_id":
#                    face_id = msg
#                    face_instance.id = face_id.data
#                    face_instance.frame = "face_" + face_id.data
#                    face_instance.gaze_frame = "gaze_" + face_id.data
#                    person_instance.face = face_instance
#
#                # get the tfs and republish them
#                if topic == "/tf":
#                    if msg.transforms[0].header.frame_id == \
#                            "camera_color_optical_frame" and \
#                            msg.transforms[0].child_frame_id == "face_" \
#                            + face_id.data:
#                        # compute the transformation
#                        tf_rcamera_to_hface = msg.transforms[0]
#                    if msg.transforms[0].header.frame_id == "face_" \
#                            + face_id.data and \
#                            msg.transforms[0].child_frame_id == "gaze_" \
#                            + face_id.data:
#                        # compute the transformation
#                        tf_hface_to_hgaze = msg.transforms[0]
#
#                    if tf_rcamera_to_hface is not None and \
#                            tf_hface_to_hgaze is not None:
#                        tf_rcamera_to_hface.header.stamp = rospy.Time.now()
#                        self.publish_tf(tf_rcamera_to_hface.header.frame_id,
#                                        tf_rcamera_to_hface.child_frame_id,
#                                        tf_rcamera_to_hface)
#                        tf_hface_to_hgaze.header.stamp = rospy.Time.now()
#                        self.publish_tf(tf_hface_to_hgaze.header.frame_id,
#                                        tf_hface_to_hgaze.child_frame_id,
#                                        tf_hface_to_hgaze)
#
#                        rospy.sleep(t-prev_timestamp)
#                        prev_timestamp = t
#
#                        try:
#                            engagement_node.get_tracked_humans()
#                            if frame_counter < self.frame_to_skip:
#                                frame_counter += 1
#                            else:
#                                self.assertEqual(self.expected_result,
#                                                 engagement_node.active_persons
#                                                 [self.persons_id[i]].
#                                                 person_current_engagement_level)
#
#                        except (tf2_ros.LookupException,
#                                tf2_ros.ConnectivityException,
#                                tf2_ros.ExtrapolationException) as e:
#                            rospy.logwarn(f'Exception in tf2 {e}')
#
#            bag.close()


if __name__ == '__main__':
    unittest.main()
