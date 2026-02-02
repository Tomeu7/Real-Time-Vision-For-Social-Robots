# Multi-User Engagement Detection

A ROS2 node for detecting and tracking visual social engagement between a robot and multiple persons, as well as engagement between persons themselves. Based on ROS4HRI engagement https://github.com/ros4hri/hri_engagement.

## Overview

This package implements the visual social engagement metric from:
> Webb, A. M., & Lemaignan, S. (2017). "Measuring Visual Social Engagement from Proxemics and Gaze"

The system computes engagement based on:
- **Mutual gaze** between entities (robot↔person, person↔person)
- **Proximity** (distance between entities)
- **Field of view** constraints

## Key Features

- Robot-to-person engagement tracking
- Person-to-person engagement tracking
- Person-to-tablet engagement tracking

---

## Reference Frames

### Coordinate System Conventions

The system uses two different coordinate frame conventions:

#### 1. Robot Frame (`sellion_link`)
- **Convention**: Standard robotics frame
- **Forward axis**: `+X`
- **Right axis**: `+Y`
- **Up axis**: `+Z`

#### 2. Person Gaze Frame (Optical frame)
- **Convention**: Camera/optical frame
- **Forward axis**: `+Z`
- **Right axis**: `+X`
- **Down axis**: `+Y`

### Transform Handling

#### Robot-to-Person Engagement
When computing engagement between robot and person:

```python
# Person's view (optical frame: +Z forward)
xb = tz  # Forward component from inverse transform
yb = tx  # Horizontal offset
zb = ty  # Vertical offset

# Robot's view (robot frame: +X forward)
xa = trans.x  # Forward component
ya = trans.y  # Horizontal offset
za = trans.z  # Vertical offset
```

#### Person-to-Person Engagement
When computing engagement between two persons:

```python
# Both use optical frame (+Z forward)
# Person A's view (from inverse transform)
xb = tz  # Forward
yb = tx  # Horizontal
zb = ty  # Vertical

# Person B's view (from direct transform)
xa = trans.z  # Forward
ya = trans.x  # Horizontal
za = trans.y  # Vertical
```

### Relative Transform Computation

For person-to-person engagement:

```
T_A→B = T_world→A^(-1) × T_world→B
```

Where:
- `T_world→A`: Transform from world to Person A's gaze frame
- `T_world→B`: Transform from world to Person B's gaze frame
- `T_A→B`: Transform from Person A to Person B

---

## Webb & Lemaignan Formula

### Engagement Score Computation

The engagement score `s_AB` between entity A and entity B is computed as:

```
s_AB = min(1, m_AB × log_d_AB)
```

Where:

#### Mutual Gaze Component (`m_AB`)

```
m_AB = gaze_AB × gaze_BA
```

- `gaze_AB`: Does A see B? (0.0 to 1.0)
- `gaze_BA`: Does B see A? (0.0 to 1.0)

#### Individual Gaze Score

For entity A looking at entity B:

```
gaze_AB = max(0, 1 - (√(y_B² + z_B²)) / (tan(FOV) × x_B))
```

Where:
- `x_B`: Forward distance from A to B (in A's frame)
- `y_B`: Horizontal offset from A to B
- `z_B`: Vertical offset from A to B
- `FOV`: Field of view angle (default: 80°)

**Interpretation**:
- `gaze_AB = 1.0`: B is directly in A's gaze center
- `gaze_AB = 0.0`: B is outside A's field of view
- `gaze_AB > 0 only if x_B > 0` (B must be in front of A)

#### Distance Attenuation (`log_d_AB`)

```
d_AB = √(tx² + ty² + tz²)
log_d_AB = log(-d_AB + max_distance + 1) / log(max_distance + 1)
```

Where:
- `d_AB`: Euclidean distance between A and B
- `max_distance`: Maximum engagement distance (default: 4.0m)

**Interpretation**:
- `log_d_AB = 1.0`: Very close (d ≈ 0)
- `log_d_AB = 0.0`: At maximum distance
- `log_d_AB < 0`: Beyond maximum distance → `s_AB = 0`

### Example Calculation

Given:
- Distance: `d_AB = 2.0m`
- Max distance: `max_distance = 4.0m`
- FOV: `80°`
- Person B is slightly off-center from A

```
# Gaze scores
gaze_AB = 0.8  # A sees B pretty well
gaze_BA = 0.9  # B sees A very well

# Mutual gaze
m_AB = 0.8 × 0.9 = 0.72

# Distance factor
log_d_AB = log(4.0 - 2.0 + 1) / log(5.0) = log(3) / log(5) ≈ 0.68

# Final engagement score
s_AB = min(1, 0.72 × 0.68) = 0.49
```

---

## Engagement State Machine

### States

```
UNKNOWN (0)
    ↓
DISENGAGED (1) ←→ ENGAGING (2) ←→ ENGAGED (3) ←→ DISENGAGING (4)
```

### State Transitions

Based on average engagement value over observation window (default: 10 seconds):

| Current State | Condition | Next State |
|--------------|-----------|------------|
| UNKNOWN | Always | DISENGAGED |
| DISENGAGED | avg > -0.4 | ENGAGING |
| ENGAGING | avg < -0.6 | DISENGAGED |
| ENGAGING | avg > 0.5 | ENGAGED |
| ENGAGED | avg < 0.4 | DISENGAGING |
| DISENGAGING | avg > 0.6 | ENGAGED |
| DISENGAGING | avg < -0.5 | DISENGAGED |

### Hysteresis

The state machine includes hysteresis to prevent oscillation:
- Entering ENGAGED requires `avg > 0.5`
- Staying ENGAGED only requires `avg > 0.4`
- This 0.1 buffer prevents rapid state changes

---

## Multi-Target Engagement Logic

### Separate Histories Per Target

Instead of a single engagement history, we maintain **separate histories for each target**:

```python
engagement_histories = {
    "robot": [+1, -1, +1, +1, +1, ...],
    "person_abc": [-1, -1, +1, +1, -1, ...],
    "person_xyz": [-1, -1, -1, -1, -1, ...],
    "tablet": [-1, -1, -1, +1, +1, ...],
}
```

### Binary Engagement Values

For each frame and each target, we use **separate thresholds** to compensate for detection bias:

```python
# For robot engagement (higher threshold due to better detection)
if s_AB_robot > robot_engagement_threshold:  # default = 0.6
    append +1  # Engaged with robot
else:
    append -1  # Not engaged with robot

# For person-to-person engagement (lower threshold to compensate for side angles)
if s_AB_person > person_engagement_threshold:  # default = 0.4
    append +1  # Engaged with this person
else:
    append -1  # Not engaged with this person
```

### Primary Target Selection

```python
# Compute average for each target
target_averages = {
    "robot": sum(robot_history) / len(robot_history),
    "person_abc": sum(person_abc_history) / len(person_abc_history),
    "person_xyz": sum(person_xyz_history) / len(person_xyz_history),
}

# Select target with highest average
primary_target = max(target_averages, key=target_averages.get)
engagement_value = target_averages[primary_target]

# Use this value in the state machine
```

### Why This Approach?

#### Problem with Raw Scores
Using raw Webb & Lemaignan scores directly would create bias:
- Robot typically has **better gaze detection** (frontal camera)
- Robot has **more stable transforms**
- Robot would almost always win, even if person is looking at another person

#### Solution: Normalized Binary Histories with Target-Specific Thresholds
- Each target gets fair evaluation: "engaged or not?"
- **Robot threshold (0.6)** is higher to compensate for better detection
- **Person threshold (0.4)** is lower to compensate for side-angle detection
- Detection quality differences are balanced by threshold adjustment
- We still track WHO they're engaged with (highest average)

---

## Parameters

### Configurable Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `reference_frame` | string | `sellion_link` | Robot's reference frame |
| `max_distance` | float | `4.0` | Maximum engagement distance (m) |
| `field_of_view` | float | `60.0` | Field of view angle for humans (degrees) |
| `robot_field_of_view` | float | `60.0` | Field of view angle for robot camera (degrees) |
| `robot_engagement_threshold` | float | `0.6` | Threshold for robot engagement (higher = more conservative) |
| `person_engagement_threshold` | float | `0.4` | Threshold for person-to-person engagement (lower compensates for side angles) |
| `tablet_frame` | string | `tablet_link` | TF frame name for the tablet |
| `observation_window` | float | `10.0` | History window size (seconds) |
| `rate` | float | `10.0` | Computation rate (Hz) |

### Example Launch

```bash
ros2 launch hri_multiuser_engagement multiuser_engagement.launch.py \
    reference_frame:=base_link \
    max_distance:=5.0 \
    field_of_view:=70.0 \
    robot_field_of_view:=70.0 \
    robot_engagement_threshold:=0.7 \
    person_engagement_threshold:=0.3 \
    tablet_frame:=tablet_link
```

---

## Published Topics

### `/humans/engagement/multiuser` (emorobcare_cv_msgs/MultiUserEngagementLevel)

Single topic containing all engagement information:

```yaml
header:
  stamp: ...
person_id: "person_abc"
primary_target_id: "robot"  # Who they're most engaged with
level: 3  # ENGAGED
max_score: 0.78  # Raw score with primary target
engagement_details:
  - target_id: "robot"
    score: 0.78
  - target_id: "person_xyz"
    score: 0.23
  - target_id: "person_def"
    score: 0.15
```

### `/humans/persons/{person_id}/engagement_status` (hri_msgs/EngagementLevel)

Traditional per-person engagement status (for backward compatibility):

```yaml
header:
  stamp: ...
level: 3  # ENGAGED
```

### `/intents` (hri_actions_msgs/Intent)

Published when person becomes ENGAGED:

```yaml
intent: ENGAGE_WITH
data: '{"recipient": "person_abc", "target": "robot"}'
```

---

## Subscribed Topics

The node automatically subscribes to HRI topics via `pyhri`:

- `/humans/persons/{id}/face/gaze` - Gaze transforms
- `/humans/persons/{id}/face/roi` - Face regions
- `/humans/tracked` - List of tracked persons

---

## Algorithm Flow

```
┌─────────────────────────────────────────────┐
│ For each tracked person:                    │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ 1. Assess Robot Engagement                  │
│    - Get person→robot transform             │
│    - Compute s_AB using Webb & Lemaignan   │
│    - Append +1 or -1 to robot history      │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ 2. Assess Person-Person Engagement          │
│    For each other person:                   │
│    - Compute relative transform             │
│    - Compute s_AB using Webb & Lemaignan   │
│    - Append +1 or -1 to person history     │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ 3. Assess Tablet Engagement                 │
│    - Lookup tablet transform from TF        │
│    - Compute relative transform             │
│    - Compute s_AB using Webb & Lemaignan   │
│    - Append +1 or -1 to tablet history     │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ 4. Compute Overall Engagement               │
│    - Compute average for each target        │
│    - Select primary target (max average)    │
│    - Run state machine on primary average   │
└─────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────┐
│ 5. Publish Results                          │
│    - MultiUserEngagementLevel (all data)    │
│    - EngagementLevel (backward compat)      │
│    - Intent (if newly engaged)              │
└─────────────────────────────────────────────┘
```


## References

1. Webb, A. M., & Lemaignan, S. (2017). *Measuring Visual Social Engagement from Proxemics and Gaze*.

2. ROS4HRI: https://wiki.ros.org/hri

3. Original engagement implementation: https://github.com/ros4hri/hri_engagement

---

## License

Apache License 2.0

## Authors

- Antonio Andriella
- Séverin Lemaignan
- Luka Juricic
- Multi-user extensions: EmoRobCare team

## Maintainer

PAL Robotics (<bartomeu.pou@iiia.csic.es>)
