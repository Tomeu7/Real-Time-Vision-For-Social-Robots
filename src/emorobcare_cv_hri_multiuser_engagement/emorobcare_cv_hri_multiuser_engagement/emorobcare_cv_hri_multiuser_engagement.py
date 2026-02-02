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
# Based on https://github.com/ros4hri/hri_engagement

from collections import deque
import json
import math

from diagnostic_msgs.msg import DiagnosticArray, DiagnosticStatus, KeyValue
from builtin_interfaces.msg import Time, Duration
from emorobcare_cv_msgs.msg import EngagementDetail, MultiUserEngagementLevel, EngagementGeometry, ProcessingTime
from geometry_msgs.msg import TransformStamped
from hri import HRIListener, Person
from hri_actions_msgs.msg import Intent
from hri_msgs.msg import EngagementLevel
from lifecycle_msgs.msg import State
from rcl_interfaces.msg import ParameterDescriptor
import rclpy
from rclpy.executors import SingleThreadedExecutor, ExternalShutdownException
from rclpy.lifecycle import Node, TransitionCallbackReturn
from rclpy.lifecycle.node import LifecycleState
from std_msgs.msg import Header
import tf2_ros
from tf2_ros import TransformException
import time

ENABLE_PROCESSING_TIME = False

EngagementStatus = {
    # unknown: no information is provided about the engagement level
    EngagementLevel.UNKNOWN: 'UNKNOWN',
    # disengaged: the person has not looked in the direction of the robot
    EngagementLevel.DISENGAGED: 'DISENGAGED',
    # engaging: the person has started to look in the direction of the robot
    EngagementLevel.ENGAGING: 'ENGAGING',
    # engaged: the person is fully engaged with the robot
    EngagementLevel.ENGAGED: 'ENGAGED',
    # disengaging: the person has started to look away from the robot
    EngagementLevel.DISENGAGING: 'DISENGAGING'
}

# diagnostic message publish rate in Hertz
DIAG_PUB_RATE = 1


class PersonEngagement(object):
    """
    Auxiliary class checking the engagement status of a person.

    Auxiliary class that given a Person identified with their person_id,
    it publishes their engagement status on the topic:
    /humans/persons/<human_id>/engagement_status.
    To compute the engagement, their gaze direction is estimated with
    respect to the robot.
    The engagement statuses in which a person can be, are the following:
    # UNKNOWN: no information is provided about the engagement level
    # DISENGAGED: the person has not looked in the direction of the robot
    # ENGAGING: the person has started to look in the direction of the robot
    # ENGAGED: the person is fully engaged with the robot
    # DISENGAGING: the person has started to look away from the robot
    """

    def __init__(self,
                 node: Node,
                 person: Person,
                 reference_frame: str,
                 max_distance: float,
                 field_of_view_rad: float,
                 robot_field_of_view_rad: float,
                 tablet_field_of_view_rad: float,
                 robot_engagement_threshold: float,
                 person_engagement_threshold: float,
                 tablet_engagement_threshold: float,
                 engagement_mode: str,
                 min_engagement_score: float,
                 rate: float,
                 observation_window: float,
                 others=None,
                 tablet_frame="tablet_link",
                 tf_buffer=None,
                 debug_mode=False,
                 skip_anonymous_engagement=True,
                 engagement_cache=None
                 ):

        self.node = node

        self.person = person

        self.others = others or {}

        self.reference_frame = reference_frame
        self.max_distance = max_distance
        self.field_of_view_rad = field_of_view_rad
        self.robot_field_of_view_rad = robot_field_of_view_rad
        self.tablet_field_of_view_rad = tablet_field_of_view_rad
        self.robot_engagement_threshold = robot_engagement_threshold
        self.person_engagement_threshold = person_engagement_threshold
        self.tablet_engagement_threshold = tablet_engagement_threshold
        self.engagement_mode = engagement_mode
        self.min_engagement_score = min_engagement_score
        self.tablet_frame = tablet_frame
        self.tf_buffer = tf_buffer
        self.debug_mode = debug_mode
        self.skip_anonymous_engagement = skip_anonymous_engagement
        self.engagement_cache = engagement_cache if engagement_cache is not None else {}

        # Cached tablet transform (set externally to avoid redundant TF lookups)
        self.cached_tablet_transform = None

        # number of samples used to infer the user's engagement
        self.engagement_history_size = int(rate * observation_window)
        # start publishing the engagement status after half the buffer duration
        self.min_samples = int(0.5 * self.engagement_history_size)

        self.is_registered = True

        # time vars for computing the tf transform availability
        self.current_time_from_tf = self.node.get_clock().now()
        self.start_time_from_tf = self.node.get_clock().now()
        # timeout after which the engagement status is set to unknown
        self.timeout_tf = 10

        # publisher for the engagement status of the person
        try:
            self.engagement_status_pub = self.node.create_publisher(
                EngagementLevel,
                self.person.ns +
                '/engagement_status',
                10,
            )
            self.intent_pub = self.node.create_publisher(
                Intent,
                '/intents',
                10)

        except AttributeError:
            self.get_logger().warn(
                f'cannot create a pub as the value of self.person_id is {self.person.id}')

        # Separate engagement histories per target
        # Format: {target_id: [+1, -1, +1, ...]}
        self.engagement_histories = {}

        # Store current raw engagement scores for debugging
        self.current_engagement_scores = {}  # {target_id: raw_score}

        # current engagement level
        self.person_current_engagement_level = EngagementLevel.UNKNOWN

        # Track primary target (who they're most engaged with)
        self.primary_target_id = None

        # publish the engagement status as soon as the Person is created
        self.publish_engagement_status()

    def get_logger(self):
        return self.node.get_logger()

    def unregister(self):
        """Unregister the Person engagement_status_pub and the face_id_sub."""
        self.person_current_engagement_level = EngagementLevel.UNKNOWN
        self.publish_engagement_status()
        self.node.destroy_publisher(self.engagement_status_pub)
        self.node.destroy_publisher(self.intent_pub)

    ##############################################################
    # Engagement helper functions (optimized - no numpy allocations)
    ##############################################################

    def _rotate_vector_by_quat_inverse(self, vx, vy, vz, qx, qy, qz, qw):
        """Rotate vector (vx,vy,vz) by inverse of quaternion (qx,qy,qz,qw).

        Uses conjugate quaternion q* = (-qx, -qy, -qz, qw) for inverse rotation.
        Pure Python - no numpy allocations.
        """
        # Conjugate quaternion (inverse rotation)
        qx, qy, qz = -qx, -qy, -qz

        # v' = v + 2*w*(q × v) + 2*(q × (q × v))
        # First cross product: q × v
        cx = qy * vz - qz * vy
        cy = qz * vx - qx * vz
        cz = qx * vy - qy * vx

        return (
            vx + 2 * (qw * cx + qy * cz - qz * cy),
            vy + 2 * (qw * cy + qz * cx - qx * cz),
            vz + 2 * (qw * cz + qx * cy - qy * cx),
        )

    def _compute_inverse_transform_fast(self, trans, rot):
        """Compute inverse translation using quaternion math (no matrices).

        Given transform T with translation t and rotation R,
        computes the translation part of T^{-1} = -R^T * t
        """
        return self._rotate_vector_by_quat_inverse(
            -trans.x, -trans.y, -trans.z,
            rot.x, rot.y, rot.z, rot.w
        )

    def compute_relative_positions(self, transform_a, transform_b):
        """Compute BOTH relative positions in one shot (no double inverse).

        Given:
            transform_a: ^R T_A (A's gaze in reference frame)
            transform_b: ^R T_B (B's gaze in reference frame)

        Returns:
            (pos_b_in_a, pos_a_in_b): Tuples of (x, y, z) for both directions

        This replaces relative_transform_person2person + _compute_inverse_transform
        with a single efficient computation.
        """
        # Extract positions in reference frame
        pa = transform_a.transform.translation
        pb = transform_b.transform.translation
        ra = transform_a.transform.rotation
        rb = transform_b.transform.rotation

        # Vector from A to B in reference frame
        diff_x = pb.x - pa.x
        diff_y = pb.y - pa.y
        diff_z = pb.z - pa.z

        # Position of B in A's frame: rotate (B-A) by inverse of A's rotation
        pos_b_in_a = self._rotate_vector_by_quat_inverse(
            diff_x, diff_y, diff_z,
            ra.x, ra.y, ra.z, ra.w
        )

        # Position of A in B's frame: rotate (A-B) by inverse of B's rotation
        pos_a_in_b = self._rotate_vector_by_quat_inverse(
            -diff_x, -diff_y, -diff_z,
            rb.x, rb.y, rb.z, rb.w
        )

        return pos_b_in_a, pos_a_in_b

    ##############################################################
    # Engagement main functions
    ##############################################################

    def compute_engagement_for_robot(self, transform: TransformStamped):
        """Compute engagement metric for robot (uses fast inverse).

        For robot engagement, we only have one transform (person's gaze in robot frame),
        so we need to compute the inverse to get both directions.
        """
        if not transform or transform == TransformStamped():
            return 0.0

        trans = transform.transform.translation
        rot = transform.transform.rotation

        # Fast inverse using quaternion math (no numpy)
        tx, ty, tz = self._compute_inverse_transform_fast(trans, rot)

        d_ab = math.sqrt(tx ** 2 + ty ** 2 + tz ** 2)

        if d_ab > self.max_distance:
            if self.debug_mode:
                self.get_logger().info(
                    f"[{self.person.id} → robot] Distance {d_ab:.2f}m > max ({self.max_distance}m) → disengaged.",
                    throttle_duration_sec=3
                )
            return 0.0

        # Mixed convention: person (+Z fwd) and robot (+X fwd)
        xb, yb, zb = tz, tx, ty  # Person gaze (optical)
        xa, ya, za = trans.x, trans.y, trans.z  # Robot gaze (sellion link, robot frame)

        return self._compute_engagement_score(d_ab, xa, ya, za, xb, yb, zb, "robot")

    def compute_engagement_from_positions(self, pos_b_in_a, pos_a_in_b, target_label: str):
        """Compute engagement directly from pre-computed positions (no inverse needed).

        Args:
            pos_b_in_a: (x, y, z) position of B in A's frame
            pos_a_in_b: (x, y, z) position of A in B's frame
            target_label: Label for logging ("tablet" or person_id)

        This is the optimized path for person-to-person and tablet engagement.
        Both positions are computed in one shot by compute_relative_positions().
        """
        # Distance (same either way)
        d_ab = math.sqrt(pos_a_in_b[0]**2 + pos_a_in_b[1]**2 + pos_a_in_b[2]**2)

        if d_ab > self.max_distance:
            if self.debug_mode:
                self.get_logger().info(
                    f"[{self.person.id} → {target_label}] Distance {d_ab:.2f}m > max ({self.max_distance}m) → disengaged.",
                    throttle_duration_sec=3
                )
            return 0.0

        # Both use optical frame convention (+Z forward)
        # pos_b_in_a = B's position in A's frame (for gaze_ab: how A looks at B)
        # pos_a_in_b = A's position in B's frame (for gaze_ba: how B looks at A)
        xb, yb, zb = pos_b_in_a[2], pos_b_in_a[0], pos_b_in_a[1]  # A looking at B
        xa, ya, za = pos_a_in_b[2], pos_a_in_b[0], pos_a_in_b[1]  # B looking at A

        return self._compute_engagement_score(d_ab, xa, ya, za, xb, yb, zb, target_label)

    def _compute_engagement_score(self, d_ab, xa, ya, za, xb, yb, zb, target_label):
        """Core engagement calculation (Webb & Lemaignan formula).

        Args:
            d_ab: Distance between A and B
            xa, ya, za: Coordinates for gaze_ba calculation (A's position from B's view)
            xb, yb, zb: Coordinates for gaze_ab calculation (B's position from A's view)
            target_label: Label for logging and FOV selection
        """
        # gaze_ab: how much A is looking at B
        gaze_ab = 0.0
        if xb > 0:
            numerator = math.sqrt(yb ** 2 + zb ** 2)
            denominator = math.tan(self.field_of_view_rad) * xb
            gaze_ab = max(0, 1 - (numerator / denominator))

        # gaze_ba: how much B is looking at A
        gaze_ba = 0.0
        log_d_ab = 0.0
        if xa > 0:
            numerator = math.sqrt(ya ** 2 + za ** 2)
            # Select FOV based on target type
            if target_label == "robot":
                target_fov = self.robot_field_of_view_rad
            elif target_label == "tablet":
                target_fov = self.tablet_field_of_view_rad
            else:
                target_fov = self.field_of_view_rad

            denominator = math.tan(target_fov) * xa
            gaze_ba = max(0, 1 - (numerator / denominator))
            log_d_ab = math.log(-d_ab + self.max_distance + 1) / math.log(self.max_distance + 1)

        m_ab = gaze_ab * gaze_ba
        s_ab = min(1, m_ab * log_d_ab)

        if self.debug_mode:
            self.get_logger().info(
                f"[{self.person.id} → {target_label}] "
                f"dAB={d_ab:.2f}, log(dAB)={log_d_ab:.2f}, "
                f"gazeAB={gaze_ab:.2f}, gazeBA={gaze_ba:.2f}, "
                f"M_AB={m_ab:.2f}, S_AB={s_ab:.2f}",
                throttle_duration_sec=10
            )
            self.publish_geometry(target_label, d_ab, log_d_ab, gaze_ab, gaze_ba, m_ab, s_ab, xa, ya, za, xb, yb, zb)

        return s_ab

    def assess_tablet_engagement(self, face, cached_tablet_transform=None):
        """Compute engagement between a person and the tablet.

        Args:
            face: The person's face with gaze transform
            cached_tablet_transform: Optional pre-looked-up tablet transform to avoid redundant TF lookups
        """
        if not self.tf_buffer:
            self.get_logger().debug(f"[{self.person.id}] TF buffer not available for tablet engagement", throttle_duration_sec=5)
            return

        # Use cached transform if provided, otherwise look it up
        if cached_tablet_transform is not None:
            tablet_transform = cached_tablet_transform
        else:
            try:
                tablet_transform = self.tf_buffer.lookup_transform(
                    self.reference_frame,
                    self.tablet_frame,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.1)
                )
            except (TransformException, Exception) as e:
                self.get_logger().warn(f"[{self.person.id}] Failed to lookup tablet transform '{self.tablet_frame}' → '{self.reference_frame}': {e}", throttle_duration_sec=5)
                return

        try:
            pos_tablet_in_person, pos_person_in_tablet = self.compute_relative_positions(
                face.gaze_transform, tablet_transform
            )
        except Exception as e:
            self.get_logger().warn(f"[{self.person.id}] Failed to compute person→tablet transform: {e}")
            return

        s_ab_tablet = self.compute_engagement_from_positions(
            pos_tablet_in_person, pos_person_in_tablet, target_label="tablet"
        )
        self.current_engagement_scores["tablet"] = s_ab_tablet

        self.get_logger().debug(
            f"[{self.person.id}] Tablet engagement: score={s_ab_tablet:.3f}, threshold={self.tablet_engagement_threshold:.3f}, "
            f"engaged={'YES' if s_ab_tablet > self.tablet_engagement_threshold else 'NO'}",
            throttle_duration_sec=5
        )

        if "tablet" not in self.engagement_histories:
            self.engagement_histories["tablet"] = deque(maxlen=self.engagement_history_size)

        if s_ab_tablet > self.tablet_engagement_threshold:
            self.engagement_histories["tablet"].append(1)
        else:
            self.engagement_histories["tablet"].append(-1)
        
    def assess_robot_engagement(self, face):
        # compute the person's position 'viewed' from the robot's 'gaze'
        person_from_robot = face.gaze_transform

        # Use optimized method (fast quaternion inverse, no numpy)
        s_ab_robot = self.compute_engagement_for_robot(person_from_robot)

        # Store raw score for debugging
        self.current_engagement_scores["robot"] = s_ab_robot

        # Initialize history if needed
        if "robot" not in self.engagement_histories:
            self.engagement_histories["robot"] = deque(maxlen=self.engagement_history_size)

        # Append +1 or -1 to robot's history (use robot-specific threshold)
        if s_ab_robot > self.robot_engagement_threshold:
            self.engagement_histories["robot"].append(1)
        else:
            self.engagement_histories["robot"].append(-1)
    
    def assess_persons_engagement(self, face):
        if not getattr(self, "others", None):
            return

        for other_id, other_person in self.others.items():
            if not other_person or not other_person.face:
                continue

            # Skip anonymous persons if configured
            if self.skip_anonymous_engagement and "anonymous" in other_id:
                continue

            face_b = other_person.face
            if not face_b.gaze_transform:
                continue

            # Check cache first (performance optimization - avoids duplicate computation)
            cache_key = (min(self.person.id, other_id), max(self.person.id, other_id))
            if cache_key in self.engagement_cache:
                # Use cached value (already computed in get_tracked_humans)
                s_ab_person = self.engagement_cache[cache_key]
            else:
                # Cache miss - compute now using optimized method (no double inverse)
                try:
                    pos_b_in_a, pos_a_in_b = self.compute_relative_positions(
                        face.gaze_transform, face_b.gaze_transform
                    )
                    s_ab_person = self.compute_engagement_from_positions(
                        pos_b_in_a, pos_a_in_b, target_label=other_id
                    )
                except Exception as e:
                    self.get_logger().info(f"Transform A→B unavailable: {e}")
                    continue

            # Store raw score for debugging
            self.current_engagement_scores[other_id] = s_ab_person

            # Initialize history if needed
            if other_id not in self.engagement_histories:
                self.engagement_histories[other_id] = deque(maxlen=self.engagement_history_size)

            # Append +1 or -1 to this person's history (use person-specific threshold)
            if s_ab_person > self.person_engagement_threshold:
                self.engagement_histories[other_id].append(1)
            else:
                self.engagement_histories[other_id].append(-1)

    def assess_engagement(self):
        """
        Compute the current 'visual social engagement' metric.

        Computes the current 'visual social engagement' metric as defined in
        "Measuring Visual Social Engagement from Proxemics and Gaze" (by Webb
        and Lemaignan).

        If the person's engagement metric is above
        engagement_threshold, we add +1 in the engagement_history, if
        not we add a -1. The vector will be then used by the Person class to
        estimate the human engagement over the BUFFER_DURATION.
        """
        # get (and keep!) the hri::Face pointer to the current face
        face = self.person.face

        # --- Gaze availability check (shared timeout) ---
        if not face or not face.gaze_transform or face.gaze_transform == TransformStamped():
            if self.debug_mode:
                if not face:
                    self.get_logger().debug(f"[{self.person.id}] No face detected — cannot compute engagement.")
                elif not face.gaze_transform:
                    self.get_logger().debug(f"[{self.person.id}] Face detected but no gaze direction available.")
                else:
                    self.get_logger().debug(f"[{self.person.id}] Invalid gaze transform received.")
            self.current_time_from_tf = (self.node.get_clock().now() - self.start_time_from_tf).nanoseconds / 1e9

            if self.current_time_from_tf >= self.timeout_tf:
                if self.debug_mode:
                    self.get_logger().debug(
                        f"[{self.person.id}] Timeout: no gaze data for {self.timeout_tf}s → setting UNKNOWN."
                    )
                self.person_current_engagement_level = EngagementLevel.UNKNOWN
                self.publish_engagement_status()
                self.start_time_from_tf = self.node.get_clock().now()

            return  # stop here entirely if gaze missing

        ##########################################################
        # 1. Engagement with the robot
        ##########################################################
        self.assess_robot_engagement(face)

        ##########################################################
        # 2. Engagement with other persons
        ##########################################################
        self.assess_persons_engagement(face)

        ##########################################################
        # 3. Engagement with tablet
        ##########################################################
        # Use cached tablet transform if available (set by parent node)
        self.assess_tablet_engagement(face, cached_tablet_transform=self.cached_tablet_transform)

        # time of the last successful tf
        self.start_time_from_tf = self.node.get_clock().now()

    def compute_engagement(self):
        """
        Compute the engagement level of the person based on highest-average target.

        Status can be "unknown", "disengaged", "engaging", "engaged", "disengaging".
        Supports three modes:
        1. Winner mode (engagement_mode='winner'): Pick highest score above target-specific threshold
        2. Relative mode (engagement_mode='relative'): Compares raw scores, picks highest
        3. Absolute mode (engagement_mode='absolute'): Uses thresholds for each target type
        """
        # Note: No need to manually clean up histories - deque with maxlen handles this automatically

        if self.engagement_mode == 'winner':
            # === WINNER MODE: Pick highest score (like relative), use absolute thresholds ===
            # Get the most recent raw scores for each target (no threshold filtering)
            target_raw_scores = {}
            for target_id in self.engagement_histories.keys():
                if target_id in self.current_engagement_scores:
                    score = self.current_engagement_scores[target_id]
                    # Only consider targets with minimum engagement (like relative mode)
                    if score >= self.min_engagement_score:
                        target_raw_scores[target_id] = score

            if not target_raw_scores:
                return  # No targets meet minimum engagement

            # Find target with highest score (like relative mode)
            primary_target = max(target_raw_scores, key=target_raw_scores.get)
            max_score = target_raw_scores[primary_target]

            # Use raw score directly for state machine with absolute thresholds
            engagement_value = max_score

            # Log all scores
            all_raw_scores_str = ", ".join([f"{tid}: {score:.2f}" for tid, score in self.current_engagement_scores.items()])
            self.get_logger().debug(f'[{self.person.id}] [WINNER] All raw scores: {all_raw_scores_str}')

            # Log qualified scores (above min_engagement_score)
            qualified_scores_str = ", ".join([f"{tid}: {score:.2f}" for tid, score in target_raw_scores.items()])
            self.get_logger().debug(f'[{self.person.id}] [WINNER] Above min threshold: {qualified_scores_str}')

            # Log primary target
            self.get_logger().debug(f'[{self.person.id}] [WINNER] PRIMARY: {primary_target}, Score: {max_score:.2f}')

        elif self.engagement_mode == 'relative':
            # === RELATIVE MODE: Compare raw engagement scores ===
            # Get the most recent raw scores for each target
            target_raw_scores = {}
            for target_id in self.engagement_histories.keys():
                if target_id in self.current_engagement_scores:
                    score = self.current_engagement_scores[target_id]
                    # Only consider targets with minimum engagement
                    if score >= self.min_engagement_score:
                        target_raw_scores[target_id] = score

            if not target_raw_scores:
                return  # No targets meet minimum threshold

            # Find target with highest raw score
            primary_target = max(target_raw_scores, key=target_raw_scores.get)
            max_score = target_raw_scores[primary_target]

            # Convert raw score to engagement value for state machine
            # Map raw score (0-1) to engagement value (-1 to 1) for compatibility
            engagement_value = (max_score * 2) - 1

            # Log ALL raw scores (including those below threshold)
            all_raw_scores_str = ", ".join([f"{tid}: {score:.2f}" for tid, score in self.current_engagement_scores.items()])
            self.get_logger().debug(f'[{self.person.id}] [RELATIVE] All raw scores: {all_raw_scores_str}')

            # Log scores that meet minimum threshold
            qualified_scores_str = ", ".join([f"{tid}: {score:.2f}" for tid, score in target_raw_scores.items()])
            self.get_logger().debug(f'[{self.person.id}] [RELATIVE] Qualified scores (>={self.min_engagement_score:.2f}): {qualified_scores_str}')

            # Log primary target
            self.get_logger().debug(f'[{self.person.id}] [RELATIVE] PRIMARY: {primary_target}, Raw Score: {max_score:.2f}, Engagement Value: {engagement_value:.2f}')

        else:
            # === ABSOLUTE MODE: Use threshold-based detection ===
            # Compute average for each target using +1/-1 history
            target_averages = {}
            for target_id, history in self.engagement_histories.items():
                if len(history) >= self.min_samples:
                    target_averages[target_id] = sum(history) / len(history)

            if not target_averages:
                return  # Need at least one target with enough samples

            # Find primary target (highest average engagement)
            primary_target = max(target_averages, key=target_averages.get)
            engagement_value = target_averages[primary_target]

            self.get_logger().debug(f'[ABSOLUTE] Target averages: {target_averages}')
            self.get_logger().debug(f'[ABSOLUTE] Primary: {primary_target}, Mean: {engagement_value:.2f}')

        # Update primary target (same for all modes)
        self.primary_target_id = primary_target

        # Define adaptive thresholds based on engagement mode
        if self.engagement_mode == 'relative':
            # Relative mode: scores centered around 0 (-1 to 1)
            thresh_disengaged_to_engaging = 0.0
            thresh_engaging_to_engaged = 0.2
            thresh_engaged_to_disengaging = 0.1
            thresh_disengaging_to_engaged = 0.4
            thresh_engaging_to_disengaged = -0.3
            thresh_disengaging_to_disengaged = -0.2
        elif self.engagement_mode == 'winner':
            # Winner/Absolute mode: use target-specific thresholds (0 to 1)
            if primary_target == "robot":
                base_threshold = self.robot_engagement_threshold
            elif primary_target == "tablet":
                base_threshold = self.tablet_engagement_threshold
            else:
                base_threshold = self.person_engagement_threshold

            # Define thresholds as percentages of base threshold
            thresh_disengaged_to_engaging = base_threshold * 0.5   # 50% of threshold
            thresh_engaging_to_engaged = base_threshold            # At threshold
            thresh_engaged_to_disengaging = base_threshold * 0.8   # 80% of threshold
            thresh_disengaging_to_engaged = base_threshold * 1.2   # 120% of threshold
            thresh_engaging_to_disengaged = base_threshold * 0.3   # 30% of threshold
            thresh_disengaging_to_disengaged = base_threshold * 0.4  # 40% of threshold
        else:
            # self.engagement_mode == 'absolute'
            # Fallback to original hardcoded thresholds
            thresh_disengaged_to_engaging = -0.4
            thresh_engaging_to_engaged = 0.5
            thresh_engaged_to_disengaging = 0.4
            thresh_disengaging_to_engaged = 0.6
            thresh_engaging_to_disengaged = -0.6
            thresh_disengaging_to_disengaged = -0.5

        # State machine with adaptive thresholds
        next_level = EngagementLevel.UNKNOWN

        if self.person_current_engagement_level == EngagementLevel.UNKNOWN:
            next_level = EngagementLevel.DISENGAGED

        elif self.person_current_engagement_level == EngagementLevel.DISENGAGED:
            if engagement_value > thresh_disengaged_to_engaging:
                next_level = EngagementLevel.ENGAGING

        elif self.person_current_engagement_level == EngagementLevel.ENGAGING:
            if engagement_value < thresh_engaging_to_disengaged:
                next_level = EngagementLevel.DISENGAGED
            elif engagement_value > thresh_engaging_to_engaged:
                next_level = EngagementLevel.ENGAGED

        elif self.person_current_engagement_level == EngagementLevel.ENGAGED:
            if engagement_value < thresh_engaged_to_disengaging:
                next_level = EngagementLevel.DISENGAGING

        elif self.person_current_engagement_level == EngagementLevel.DISENGAGING:
            if engagement_value > thresh_disengaging_to_engaged:
                next_level = EngagementLevel.ENGAGED
            elif engagement_value < thresh_disengaging_to_disengaged:
                next_level = EngagementLevel.DISENGAGED

        if (
            next_level != EngagementLevel.UNKNOWN and
            next_level != self.person_current_engagement_level
        ):
            self.person_current_engagement_level = next_level
            self.get_logger().debug('Engagement status for {} is: {} (primary target: {})'.format(
                self.person.id,
                EngagementStatus[self.person_current_engagement_level],
                primary_target,
            ),
            )

            # if we just became engaged, publish an ENGAGE_WITH intent
            if self.person_current_engagement_level == EngagementLevel.ENGAGED:
                intent_msg = Intent()
                intent_msg.intent = intent_msg.ENGAGE_WITH
                intent_msg.data = json.dumps({'recipient': self.person.id, 'target': primary_target})
                self.intent_pub.publish(intent_msg)

    def publish_geometry(self, target_label, distance, log_distance, gaze_ab, gaze_ba, mutual_gaze, score, xa, ya, za, xb, yb, zb):
        """Publish detailed geometry metrics."""
        msg = EngagementGeometry()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.person_id = self.person.id
        msg.target_id = target_label
        msg.distance = float(distance)
        msg.log_distance = float(log_distance)
        msg.gaze_ab = float(gaze_ab)
        msg.gaze_ba = float(gaze_ba)
        msg.mutual_gaze = float(mutual_gaze)
        msg.score = float(score)
        msg.xa = float(xa)
        msg.ya = float(ya)
        msg.za = float(za)
        msg.xb = float(xb)
        msg.yb = float(yb)
        msg.zb = float(zb)

        self.node.geometry_pub.publish(msg)

    def publish_engagement_status(self):
        """Publish the engagement_status of the person."""
        engagement_msg = EngagementLevel()
        engagement_msg.header.stamp = self.node.get_clock().now().to_msg()
        engagement_msg.level = self.person_current_engagement_level
        self.engagement_status_pub.publish(engagement_msg)

    def publish_multiuser_engagement(self):
        """Publish multi-user engagement with all target details."""
        if not self.primary_target_id:
            return  # No primary target yet

        # Create message
        msg = MultiUserEngagementLevel()
        msg.header.stamp = self.node.get_clock().now().to_msg()
        msg.person_id = self.person.id
        msg.primary_target_id = self.primary_target_id
        msg.level = self.person_current_engagement_level

        # Get max score (raw score of primary target)
        msg.max_score = self.current_engagement_scores.get(self.primary_target_id, 0.0)

        # Compute history averages for all targets
        target_averages = {}
        for target_id, history in self.engagement_histories.items():
            if len(history) > 0:
                target_averages[target_id] = sum(history) / len(history)

        # Add all engagement details for debugging
        for target_id, raw_score in self.current_engagement_scores.items():
            detail = EngagementDetail()
            detail.target_id = target_id
            detail.score = float(raw_score)
            # Add history average (defaults to 0.0 if not available)
            detail.history_average = float(target_averages.get(target_id, 0.0))
            msg.engagement_details.append(detail)

        # Publish
        self.node.multiuser_engagement_pub.publish(msg)

    def run(self):
        """
        Execute the periodic logic.

        It calls the engaged_person method that computes the
        engagement status and the callback that publishes the
        status on the topic /humans/persons/<human_id>/engagement_status.
        """
        # if we do not have the face id of the person we just return
        if not self.person.id:
            self.get_logger().debug(
                'there is no face_id for the person {}'.format(
                    self.person.id), throttle_duration_sec=1
            )
            return
        else:
            self.assess_engagement()
            self.compute_engagement()
            self.publish_engagement_status()
            self.publish_multiuser_engagement()


class MultiUserEngagementNode(Node):
    """
    This node detects the tracked persons who are in the field of view of the robot's camera.

    Among those persons, it selects those who are active, that is, those whose
    visual social engagement metric (as defined in "Measuring Visual Social
    Engagement from Proxemics and Gaze" by Webb and Lemaignan) is above 0.5.

    For each of the active persons, it creates a Person object from which
    the engagement status is computed and published in a topic.
    """

    def __init__(
            self,
    ):
        super().__init__('emorobcare_cv_hri_multiuser_engagement')

        self.declare_parameter(
            'reference_frame', 'sellion_link', ParameterDescriptor(
                description="Robot's reference point, used to compute the distance and mutual "
                            'gaze.'))

        self.declare_parameter(
            'max_distance', 4.0, ParameterDescriptor(
                description='People further away than this distance (in meters) are considered as '
                            'disengaged.'))

        self.declare_parameter(
            'field_of_view', 60., ParameterDescriptor(
                description='Field of view (in degrees) for humans. '
                            'Use to compute mutual gaze between humans and targets.'))

        self.declare_parameter(
            'robot_field_of_view', 60., ParameterDescriptor(
                description='Field of view (in degrees) for the robot camera. '
                            'Can be different from human field_of_view.'))

        self.declare_parameter(
            'tablet_field_of_view', 60., ParameterDescriptor(
                description='Field of view (in degrees) for the tablet camera/screen. '
                            'Can be different from human and robot field_of_view.'))

        self.declare_parameter(
            'robot_engagement_threshold', 0.6, ParameterDescriptor(
                description='Threshold for robot engagement. Set higher to compensate for '
                            'better detection due to camera alignment.'))

        self.declare_parameter(
            'person_engagement_threshold', 0.4, ParameterDescriptor(
                description='Threshold for person-to-person engagement. Set lower to '
                            'compensate for side-angle detection difficulties.'))

        self.declare_parameter(
            'tablet_engagement_threshold', 0.5, ParameterDescriptor(
                description='Threshold for person-to-tablet engagement. Set moderately to '
                            'detect when people are actively looking at the tablet.'))

        self.declare_parameter(
            'engagement_mode', 'relative', ParameterDescriptor(
                description='Engagement computation mode: '
                            '"absolute" - use target-specific thresholds directly, '
                            '"relative" - normalize scores relative to other targets, '
                            '"winner" - engage with highest-scoring target if above threshold (recommended)'))

        self.declare_parameter(
            'person2person_engagement_enabled', False, ParameterDescriptor(
                description='Enable person2person.'))

        self.declare_parameter(
            'person2robot_engagement_enabled', True, ParameterDescriptor(
                description='Enable person2robot engagement computation.'))

        self.declare_parameter(
            'person2tablet_engagement_enabled', True, ParameterDescriptor(
                description='Enable person2tablet.'))

        self.declare_parameter(
            'min_engagement_score', 0.3, ParameterDescriptor(
                description='Minimum raw engagement score required for any target to be considered. '
                            'Prevents engagement detection with very low scores.'))

        self.declare_parameter(
            'observation_window', 1., ParameterDescriptor(
                description='The time window (in sec.) used to compute the engagement level of a '
                            'person.'))

        self.declare_parameter(
            'rate', 10., ParameterDescriptor(
                description='Engagement level computation and publication rate (in Hz).'))

        self.declare_parameter(
            'tablet_frame', 'tablet_link', ParameterDescriptor(
                description="Tablet's reference frame for computing engagement."
            ))

        self.declare_parameter(
            'debug_mode', False, ParameterDescriptor(
                description='Enable detailed debug logging and geometry message publishing. '
                            'Set to true for development/debugging.'))

        self.declare_parameter(
            'skip_anonymous_engagement', True, ParameterDescriptor(
                description='Skip person-to-person engagement computation for persons labeled as "anonymous".'))

        self.get_logger().info('State: Unconfigured.')

    def on_configure(self, state: LifecycleState) -> TransitionCallbackReturn:

        self.get_logger().info('='*60)
        self.get_logger().info('CONFIGURE: Loading parameters...')
        self.get_logger().info('='*60)

        self.max_distance = self.get_parameter('max_distance').value
        self.reference_frame = self.get_parameter('reference_frame').value
        self.field_of_view_rad = self.get_parameter('field_of_view').value * math.pi / 180
        self.robot_field_of_view_rad = self.get_parameter('robot_field_of_view').value * math.pi / 180
        self.tablet_field_of_view_rad = self.get_parameter('tablet_field_of_view').value * math.pi / 180
        self.robot_engagement_threshold = self.get_parameter('robot_engagement_threshold').value
        self.person_engagement_threshold = self.get_parameter('person_engagement_threshold').value
        self.tablet_engagement_threshold = self.get_parameter('tablet_engagement_threshold').value
        self.engagement_mode = self.get_parameter('engagement_mode').value
        self.min_engagement_score = self.get_parameter('min_engagement_score').value
        self.observation_window = self.get_parameter('observation_window').value
        self.rate = self.get_parameter('rate').value
        self.use_sim_time = self.get_parameter('use_sim_time').value
        self.tablet_frame = self.get_parameter('tablet_frame').value
        self.debug_mode = self.get_parameter('debug_mode').value
        self.skip_anonymous_engagement = self.get_parameter('skip_anonymous_engagement').value

        # Log all loaded parameters
        self.get_logger().info(f'  max_distance: {self.max_distance}')
        self.get_logger().info(f'  reference_frame: {self.reference_frame}')
        self.get_logger().info(f'  field_of_view: {self.field_of_view_rad * 180 / math.pi}°')
        self.get_logger().info(f'  robot_field_of_view: {self.robot_field_of_view_rad * 180 / math.pi}°')
        self.get_logger().info(f'  robot_engagement_threshold: {self.robot_engagement_threshold}')
        self.get_logger().info(f'  person_engagement_threshold: {self.person_engagement_threshold}')
        self.get_logger().info(f'  tablet_engagement_threshold: {self.tablet_engagement_threshold}')
        self.get_logger().info(f'  engagement_mode: {self.engagement_mode}')
        self.get_logger().info(f'  min_engagement_score: {self.min_engagement_score}')
        self.get_logger().info(f'  observation_window: {self.observation_window}')
        self.get_logger().info(f'  rate: {self.rate}')
        self.get_logger().info(f'  tablet_frame: {self.tablet_frame}')
        self.get_logger().info(f'  debug_mode: {self.debug_mode}')
        self.get_logger().info(f'  skip_anonymous_engagement: {self.skip_anonymous_engagement}')
        self.get_logger().info('='*60)

        self.get_logger().info('State: Inactive.')
        return super().on_configure(state)

    def on_activate(self, state: LifecycleState) -> TransitionCallbackReturn:

        # get the list of IDs of the currently visible persons
        self.tracked_persons_in_the_scene = None

        # those humans who are actively detected and are considered as 'engaged'
        self.active_persons = dict()

        # Shared cache for person-to-person engagement (optimization to avoid duplicate computations)
        self.person_engagement_cache = {}

        self.hri_listener = HRIListener('hri_engagement_listener', True, self.use_sim_time)
        self.hri_listener.set_reference_frame(self.reference_frame)

        # Initialize TF2 buffer and listener for tablet transform lookups
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.proc_timer = self.create_timer(
            1/self.rate, self.get_tracked_humans, clock=self.get_clock())

        self.diag_pub = self.create_publisher(
            DiagnosticArray, '/diagnostics', 1)
        self.diag_timer = self.create_timer(
            1/DIAG_PUB_RATE, self.do_diagnostics, clock=self.get_clock())

        # Create shared multi-user engagement publisher
        self.multiuser_engagement_pub = self.create_publisher(
            MultiUserEngagementLevel,
            '/humans/engagement/multiuser',
            10
        )
        self.get_logger().info('Created multi-user engagement topic: /humans/engagement/multiuser')

        # Create geometry details publisher (for debugging/analysis)
        self.geometry_pub = self.create_publisher(
            EngagementGeometry,
            '/humans/engagement/geometry',
            10
        )
        self.get_logger().info('Created engagement geometry topic: /humans/engagement/geometry')

        if ENABLE_PROCESSING_TIME:
            self.processing_time_pub = self.create_publisher(
                ProcessingTime, '/processing_time/multiuser_engagement', 10)

        self.get_logger().info('State: Active.')
        return super().on_activate(state)

    def on_cleanup(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.internal_cleanup()
        self.get_logger().info('State: Unconfigured.')
        return super().on_cleanup(state)

    def on_deactivate(self, state: LifecycleState) -> TransitionCallbackReturn:
        self.internal_deactivate()
        self.get_logger().info('State: Inactive.')
        return super().on_deactivate(state)

    def on_shutdown(self, state: LifecycleState) -> TransitionCallbackReturn:
        if state.state_id == State.PRIMARY_STATE_ACTIVE:
            self.internal_deactivate()
        if state.state_id in [State.PRIMARY_STATE_ACTIVE, State.PRIMARY_STATE_INACTIVE]:
            self.internal_cleanup()
        self.get_logger().info('State: Finalized.')
        return super().on_shutdown(state)

    def internal_cleanup(self):
        pass

    def internal_deactivate(self):
        self.destroy_timer(self.diag_timer)
        self.destroy_publisher(self.diag_pub)
        self.destroy_publisher(self.multiuser_engagement_pub)
        self.destroy_publisher(self.geometry_pub)
        if ENABLE_PROCESSING_TIME:
            self.destroy_publisher(self.processing_time_pub)
        self.destroy_timer(self.proc_timer)
        for _, person in self.active_persons.items():
            person.unregister()
        del self.hri_listener
        del self.active_persons
        del self.tracked_persons_in_the_scene

    def get_tracked_humans(self):
        """Update self.active_persons, the dictionary of tracked humans PersonEngagement."""
        if ENABLE_PROCESSING_TIME:
            t_in = time.time()

        self.tracked_persons_in_the_scene = self.hri_listener.tracked_persons

        # Look up tablet transform ONCE per cycle (performance optimization)
        # This avoids redundant TF lookups for each person
        cached_tablet_transform = None
        try:
            cached_tablet_transform = self.tf_buffer.lookup_transform(
                self.reference_frame,
                self.tablet_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1)
            )
        except (TransformException, Exception) as e:
            if self.debug_mode:
                self.get_logger().debug(f"Failed to lookup tablet transform in cycle: {e}")

        # Clear engagement cache for this cycle (performance optimization to avoid duplicate computations)
        self.person_engagement_cache = {}

        # Pre-compute person-to-person engagements for all pairs (avoid duplicate computations)
        # Each pair is computed once and stored with canonical key (min_id, max_id)
        person_ids = list(self.tracked_persons_in_the_scene.keys())
        for i, id_a in enumerate(person_ids):
            # Skip anonymous if configured
            if self.skip_anonymous_engagement and "anonymous" in id_a:
                continue

            person_a = self.tracked_persons_in_the_scene[id_a]
            if not person_a or not person_a.face or not person_a.face.gaze_transform:
                continue

            for id_b in person_ids[i+1:]:  # Only compute each pair once
                # Skip anonymous if configured
                if self.skip_anonymous_engagement and "anonymous" in id_b:
                    continue

                person_b = self.tracked_persons_in_the_scene[id_b]
                if not person_b or not person_b.face or not person_b.face.gaze_transform:
                    continue

                try:
                    # Use optimized method (no double inverse)
                    if self.active_persons.get(id_a):
                        temp_person_eng = self.active_persons[id_a]
                        pos_b_in_a, pos_a_in_b = temp_person_eng.compute_relative_positions(
                            person_a.face.gaze_transform,
                            person_b.face.gaze_transform
                        )
                        s_ab = temp_person_eng.compute_engagement_from_positions(
                            pos_b_in_a, pos_a_in_b, target_label=id_b
                        )

                        # Store with canonical key
                        cache_key = (min(id_a, id_b), max(id_a, id_b))
                        self.person_engagement_cache[cache_key] = s_ab
                except Exception as e:
                    if self.debug_mode:
                        self.get_logger().debug(f"Failed to pre-compute engagement {id_a}↔{id_b}: {e}")

        # check if the current active persons are
        # still active otherwise: unregister them and remove from the dict
        if self.active_persons:
            for active_human in list(self.active_persons.keys()):
                if active_human not in self.tracked_persons_in_the_scene.keys():
                    self.active_persons[
                        active_human
                    ].person_current_engagement_level = EngagementLevel.UNKNOWN
                    self.active_persons[active_human].publish_engagement_status(
                    )
                    self.active_persons[active_human].unregister()
                    del self.active_persons[active_human]
        else:
            self.get_logger().info('There are no active people around', throttle_duration_sec=1)

        # check whether the active persons are new
        # if so create a new instance of a Person
        for person_id, person_instance in self.tracked_persons_in_the_scene.items():
            others = {
                pid: pinst
                for pid, pinst in self.tracked_persons_in_the_scene.items()
                if pid != person_id
            }

            if person_id not in self.active_persons:
                self.active_persons[person_id] = PersonEngagement(
                    self,
                    person_instance,
                    self.reference_frame,
                    self.max_distance,
                    self.field_of_view_rad,
                    self.robot_field_of_view_rad,
                    self.tablet_field_of_view_rad,
                    self.robot_engagement_threshold,
                    self.person_engagement_threshold,
                    self.tablet_engagement_threshold,
                    self.engagement_mode,
                    self.min_engagement_score,
                    self.rate,
                    self.observation_window,
                    others=others,
                    tablet_frame=self.tablet_frame,
                    tf_buffer=self.tf_buffer,
                    debug_mode=self.debug_mode,
                    skip_anonymous_engagement=self.skip_anonymous_engagement,
                    engagement_cache=self.person_engagement_cache,
                )
            else:
                self.active_persons[person_id].others = others

            # Set cached tablet transform for this cycle (performance optimization)
            self.active_persons[person_id].cached_tablet_transform = cached_tablet_transform

            self.active_persons[person_id].run()

        if ENABLE_PROCESSING_TIME:
            self.get_logger().debug(f"MultiUserEngagementNode cycle time: {time.time() - t_in:.4f} seconds", throttle_duration_sec=1)
            self.process_time_wall(t_in)

    def process_time_wall(self, t_in_wall: float):
        t_out_wall = time.time()
        delta_s = t_out_wall - t_in_wall

        processing_time_msg = ProcessingTime()
        processing_time_msg.id = "multiuser_engagement"

        processing_time_msg.t_in = Time(sec=int(t_in_wall), nanosec=int((t_in_wall % 1)*1e9))
        processing_time_msg.t_out = Time(sec=int(t_out_wall), nanosec=int((t_out_wall % 1)*1e9))

        delta = Duration()
        delta.sec = int(delta_s)
        delta.nanosec = int((delta_s - int(delta_s)) * 1e9)

        processing_time_msg.delta = delta
        self.processing_time_pub.publish(processing_time_msg)

    def do_diagnostics(self):
        now = self.get_clock().now()
        arr = DiagnosticArray(header=Header(stamp=now.to_msg()))
        msg = DiagnosticStatus(
            name='/social_perception/engagement/emorobcare_cv_hri_multiuser_engagement', hardware_id='none')

        msg.level = DiagnosticStatus.OK

        msg.values = [
            KeyValue(key='Module name', value='emorobcare_cv_hri_multiuser_engagement'),
            KeyValue(key='Current engagement levels:',
                     value=str({k: EngagementStatus[v.person_current_engagement_level]
                                for k, v in self.active_persons.items()})),
        ]

        arr.status = [msg]
        self.diag_pub.publish(arr)


def main(args=None):
    rclpy.init(args=args)
    node = MultiUserEngagementNode()
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        node.destroy_node()


if __name__ == '__main__':
    main()
