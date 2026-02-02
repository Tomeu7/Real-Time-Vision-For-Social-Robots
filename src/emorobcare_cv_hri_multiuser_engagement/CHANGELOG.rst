^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Changelog for package hri_engagement
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

2.0.2 (2025-07-09)
------------------
* doc
* Remove blank line
* Add diagnostic analyzers
* ensure we keep the same pointer to the current face
  Avoid a crash happening if the face disappears while assessing engagement
* Contributors: Noel Jimenez, Séverin Lemaignan

2.0.1 (2024-08-21)
------------------
* rename diagnostics msg to match documentation (and diagnostic_aggregator) categories
* Contributors: Séverin Lemaignan

2.0.0 (2024-08-21)
------------------

Port to ROS 2

* fix test dependency
* workaroud missing dependecy in tf_transformations
* improve lifecycle shutdown handling
* linting
* add license
* exclude obsolete tests from linters
* cleanup dependencies
* port synthetic data tests
* use sim time in listener; porting fixes
* refactor parameter variables
* enable synthetic and linting tests; relocate obsolete tests
* fix not properly destroyed interfaces on deactivate and test tear down
* launch: use get_pal_configuration from launch_pal
* wip unittest
* WIP on the unit tests
* upstream pyhri bug fixed; can properly destroy hri_listener
* test runs without crash, replay bag files, but does not actually test hri_engagement
* fix some crashers
* initial scaffolding for ROS2 unit-tests
  Unit-testing infrastructure mostly borrowed from asr_vosk
* convert test bag files to ROS2
* Initial ROS2 port
  Node runs; no testing yet, as tests need to be ported first
  While here:
  - add diagnostics publishing
  - expose parameters for observation window duration and note rate
* Contributors: Luka Juricic, Séverin Lemaignan

1.0.1 (2024-01-10)
------------------
* add missing dep on rostest
* Contributors: Séverin Lemaignan

1.0.0 (2023-12-05)
------------------
* [test] disabled simple 'disengagement' bagfiles
  The current version of the algorithm consider these people to be
  partially engaged, thus failing the test
* [doc] fix/update/complete README
* tune the algorithm for more plausible results
  Tested with myself in front of the webcam
  While here, only publish the 'ENGAGE_WITH' intent when we first become ENGAGED
* change how distance is used to compute engagement
  Instead of simply dividing mutual gaze by distance, use a more complex log-based version of the distance that goes down 'gently' towards zero
* add param to set a maximum distance to even consider engagement
  By default, 4m
* {hri_engagement.launch -> monitor.launch}
* {engagement_node.py -> engagement_monitor}
* add synthetic tests
* Add two unittest for hri_engagement
  - Add a new test folder called complex_bags with bags that do not pass
  the test
  - Add a new test folder called simple_bags with bags that need to pass
  the test
  - Code refactoring
* migrate code to pyhri
* add ROS parameters to configure field of view and social engagement threshold
* reimplement the core algorithm to use the Visual Social Engagement metric
  From: Measuring Visual Social Engagement from Proxemics and Gaze (Webb and Lemaignan 2022)
  While here, adjusted several constant for more reliable behaviour
* Contributors: Luka Juricic, Séverin Lemaignan, antonioandriella, saracooper

0.1.0 (2022-08-02)
------------------
* Add ReadMe
* remove engagement callback rate in params
* [DEBUG] duplicate call to method compute_engagement
* code refactoring - improving code structure
  - add sellion_link as robot_gaze_frame
  - add timer to track tf availability (10 secs)
  - set engagement to unknown when person not active
  - remove timer callback for engagement status
* code refactoring - design new class structure
  - define an aux Person class that computes the engagement_status
  and publish it on a topic.
  - edit EngagementNode class that now only checks for the active
  persons and for each of them initialises an instance of the class Person.
  - migrate code from tf to tf2
  - add var is_registered to Person
  - simplify compute_engagement method
* [feat] engagement node working with >= 1 human
  the hri_person_manager recognises a human and assign
  them a unique id. A dedicated callback has been introduced to
  publish the engagement status at a given frame/rate.
* [feat] engagement node working with 1 human
  It publishes the status of an active human as:
  ['disengaged', 'disengaging', 'engaging', 'engaged']
  based on their previous status history
* Changed reference frame for resting head position
* Publishing PoseStamped on /look_to_wildcard
  /look_to_wildcard topic expects PoseStamped object, not
  PointStamped. Removed the topic latcher since it is no more
  required
* Reduced publishing rate on /look_at
  The node now publishes a PointStamped object on /look_at only if:
  -- The engaged face has moved more than a specific threshold value
  from the last position the eyes have moved for
  -- The state machine goes from state 2 (Retrieving engagement) to
  state 0 (No engagement). In this case the robot will go back
  looking in front of it and eyes need to be repositioned.
* Fixing attention geometry and new debug tools
  -- Debugged formula in the cone-of-attention section
  -- It is now possible to visualize the current state machine
  value as a semaphore in rviz
* Publishing engagement information for head turning
  Through topic_tools relay, the information about the point the
  robot should look at gets published also on the /look_at_wildcard
  topic, involved in head turning.
* introduced robot head frame parameter
* code refactoring
  -- removed comments and unused dependencies from package.xml
  -- removed comments and unused dependencies from CMakeLists.txt
* fixed active users publishing
  -- added function update_active_users to track all the ids that
  have been active at least once in the last time_thr sec
  -- code refactoring
* code refactoring
* Modifying the structure representing the close faces
* publishing ids of the active users
* Initial commit
* Contributors: Antonio Andriella, Séverin Lemaignan, Lorenzo Ferrini
