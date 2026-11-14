1.4
Data Collection Summary
For each trial, the following data streams are simultaneously recorded:
2
Data TypeParametersSampling Rate
IMU (7 sensors: pelvis, thighs,
lower legs, feet)9 channels per sensor
(LAcc/AAcc/Quaternion)190 Hz
TreadmillSpeed range: 0–10 km/h, Ramp degree:
0–15 degrees

3.0.2
IMU
The IMU system consists of 7 sensors, wired connected to a Personal Computer (PC) that collects and
processes the data from the sensors. The IMU sensors are made in-house, and the model of IMU chip is
ICM20948. Each IMU provides 9 channels of data per sensor (3 groups × 3 axes):
• LAcc (Linear Acceleration): 3-axis linear acceleration (X/Y/Z), gravity not removed, unit: pre-
sumably g
• AAcc (Angular Velocity): 3-axis angular velocity (Roll/Pitch/Yaw), unit: deg/s
• Quaternion: 4-axis quaternion (W/X/Y/Z), unit: dimensionless
The 7 sensors are placed on: pelvis, bilateral thighs, bilateral lower legs, bilateral feet.
3.1Sensor Placement
3.1.1IMU Placement
Based on the Xsens Awinda sensor placement guidelines and usual practice of IMU placement in relavent
exoskeleton, the following IMU placements are used to capture lower limb motion across all tasks (see
Figure 1).
• Upper Leg (Thigh): middle of lateral thigh (less fatty tissue, strap stays in place)
• Lower Leg (Shank): proximal lower leg, just below the knee
• Pelvis: tracking hip movement
• Foot: dorsum of the foot (instep), capturing ankle and foot dynamics
All IMU sensors have its own reference frame and number index, and its own local xyz coordinate system
and index number is labelled. Note the placement orientation of the IMU sensors should strickly follow the
above description below.
IMU Configuration (all tasks):
3(a) Right Leg view (b) Left Leg view
(sensors 2, 4, 6)
(sensors 3, 5, 7)
Figure 1: IMU sensor placement on the lower limb. Seven wired IMU sensors (numbered 1–7 across both
legs) are strapped at the pelvis, bilateral thighs, bilateral lower legs, and bilateral feet.
• 1 × Pelvis (IMU 1; x-axis’s positive direction should point toward the ground when subject is standing)
• 2 × Thigh (Left: IMU 3, Right: IMU 2; y-axis’s positive direction should point out from subject)
• 2 × Lower Leg (Left: IMU 5, Right: IMU 4; mounted on the proximal lower leg, just below the knee; y-axis’s positive direction should point out from subject)
• 2 × Foot (Left: IMU 7, Right: IMU 6; mounted on the dorsum of the foot/instep; x-axis’s positive direction should point toward the subject)
All IMU sensors have its own reference frame, and its own local xyz coordinate system is labelled.

4
Procedures and Data Collection: Tasks
Four activities of daily living (ADLs) were performed by participants: level walking, sit-to-stand (STS), ramp
walking, and stair ascent/descent. Depending on the task, trials were conducted using different exercise
equipment (e.g., treadmill, stairs, chair) and task-specific parameters (e.g., speed, incline, step height).
44.1
Standard Procedures Before All Tasks
Step 1: Ensure that all sensors (IMU and EMG) are securely attached to the participant.
Step 2: Open two software applications: MATLAB for real-time EMG data collection and ZCANPRO for
IMU data collection.
Step 3: Check that all cables are connected properly. For the IMU: connect the CAN card to the computer
first, then connect the battery to the MCU. Confirm the EMG system is powered on.
Step 4: Check the EMG signals and IMU data streams, including the synchronization between the two
systems: Observe the synchronization channel on both IMU and EMG have simultaneous high voltage or
shown in the CAN data stream to confirm data streams are aligned.
4.2
Standard Procedures for All Task Trials
Step 1: Modify the parameter (if have any) on the relevant equipment (e.g., treadmill speed, ramp incline,
chair height) according to the certain task and trial condition.
Step 2: Before the actual data collection for each trial, modify the file name in both MATLAB and
ZCANPRO.
Step 3: Click the “Start” button in both software to begin data collection.
Step 4: Press button A on the key fob, indicating the real start of the trial.
Step 5: After the trial is completed, press button D on the key fob to indicate the end of the data collection.
4.3
Task A - Level Walking
Level walking is performed on a treadmill(Figure 2a).
Experimental Design:
• Speed: 3 levels (3.5 km/h, 3.8 km/h, 4.1 km/h)
• Duration per condition: 1 min per trial
• Trials per speed: 3 trials
• Total trials: 3 speeds × 3 trials = 9 trials
