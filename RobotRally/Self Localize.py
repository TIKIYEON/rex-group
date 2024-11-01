import cv2
import particle
import camera
import numpy as np
import time
from timeit import default_timer as timer
import sys
import math
import functions as f

# Flags
showGUI = True  # Whether or not to open GUI windows
onRobot = True  # Whether or not we are running on the Arlo robot

def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
      You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot

if isRunningOnArlo():
    sys.path.append("./")
    print("On Arlo!")

try:
    import robot
    onRobot = True
    arlo = robot.Robot()
except ImportError:
    print("selflocalize.py: robot module not present - forcing not running on Arlo!")
    onRobot = False

try:
    import picamera2
    print("Camera.py: Using picamera2 module")
except ImportError:
    print("Camera.py: picamera2 module not available")
    exit(-1)

# Landmarks.
# The robot knows the position of 2 landmarks. Their coordinates are in the unit centimeters [cm].
landmarkIDs = [6, 8, 2, 3]
landmarks = {
    6: (0.0, 0.0),  # Coordinates for landmark 1
    8: (500.0, 0.0),
    3: (500.0, 500.0),
    2: (0.0, 5000.0),
    #4: (500.0, 0.0)
}
landmark_colors = [f.CRED, f.CGREEN, f.CBLUE, f.CCYAN] # Colors used when drawing the landmarks

# Robot movement controls
sleep360 = 2.45 * 2
sleep1 = sleep360 / 360
leftSpeed = 46.2
rightSpeed = 42

# Define the standard deviations for distance and angle (you can tweak these)
sigma_d = 5
sigma_theta = np.pi / 12

# Camera calibration
camera_matrix = np.array([[3606/2, 0, 960], [0, 3606/2, 540], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# Main program #
try:
    if showGUI:
        # Open windows
        WIN_RF1 = "Robot view"
        cv2.namedWindow(WIN_RF1)
        cv2.moveWindow(WIN_RF1, 50, 50)

        WIN_World = "World view"
        cv2.namedWindow(WIN_World)
        cv2.moveWindow(WIN_World, 500, 50)

    # Initialize particles
    num_particles = 1000
    particles = f.initialize_particles(num_particles)

    est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

    # Driving parameters
    velocity = 0.0 # cm/sec
    angular_velocity = 0.0 # radians/sec

    # Allocate space for world map
    world = np.zeros((500,500,3), dtype=np.uint8) 

    # Draw map
    f.draw_world(est_pose, particles, world, landmarks, landmarkIDs, landmark_colors)

    print("Opening and initializing camera")
    if isRunningOnArlo():
        cam = camera.Camera(0, robottype='arlo', useCaptureThread=False)
    else:
        cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=False)

    arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
    arucoParams = cv2.aruco.DetectorParameters_create()

    curr_id_index = 0
    #target_ids = [1, 2, 3, 4, 1]
    target_ids = [6, 8, 3, 2, 6]

    while True:
        # Handle user input for control
        image = cam.get_next_frame()


        (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

        # Time step (dt)
        dt = 0.1

        if ids is None or (ids is not None and target_ids[curr_id_index] not in ids):
            arlo.go_diff(leftSpeed, rightSpeed, 1, 0)
            time.sleep(sleep360 / 16)
            arlo.stop()
            time.sleep(sleep360 / 8)
        else:
            #### TVEC FILTERING BEGIN ####
            print("Found correct target id")
            id_list = [i for i, x in enumerate(ids) if x == target_ids[curr_id_index]]
            print(f"ID: {target_ids[curr_id_index]} found at indicies: {id_list}")

            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 18, camera_matrix, dist_coeffs)

            print(f"tvec from cv2: {tvec}")
            
            target_vec, orientation_vec = f.target_id_tvec(tvec, rvec, id_list)

            print(f"tvec for ID{target_ids[curr_id_index]}: {target_vec}")

            #### TVEC FILTERING END ####
            # now we have the copy we want for our target/marker, the closer one :)

            # Update each particle's state
            for p in particles:
                # Compute the change in x, y, and theta
                delta_x = velocity * dt * np.cos(p.getTheta())
                delta_y = velocity * dt * np.sin(p.getTheta())
                delta_theta = angular_velocity * dt

            p = particle.move_particle(p, delta_x, delta_y, delta_theta)
            # Add uncertainty to the particle's pose
            particle.add_uncertainty([p], 0.5, 0.1)  # Motion noise

            objectIDs, dists, angles = cam.detect_aruco_objects(image)

            est_pose = particle.estimate_pose(particles)

            # Update particle weights
            if objectIDs is not None:
                print("Object ID = ", objectIDs, ", Distance = ", dists, ", angle = ", angles)
                # removing the duplicates for each ID and picking the shortest distance for each
                objects_dict = {}
                for i in range(len(objectIDs)):
                    # add back when we want to filter out the aruco markers we don't want the particles' weights to be updated by
                    #if objectIDs[i] < 5:
                    if objectIDs[i] in objects_dict.keys():
                        if dists[i] < objects_dict[objectIDs[i]][0]:
                                objects_dict[objectIDs[i]] = [dists[i], angles[i]]
                    else:
                        objects_dict[objectIDs[i]] = [dists[i], angles[i]]

                objectIDs = list(objects_dict.keys())
                dists = [value[0] for value in objects_dict.values()]
                angles = [value[1] for value in objects_dict.values()]

                print("Object ID = ", objectIDs, ", Distance = ", dists, ", angle = ", angles)
                for p in particles:
                    log_total_weight = 0.0
                    for i in range(len(objectIDs)):
                        detected_id = objectIDs[i]
                        measured_dist = dists[i]
                        measured_angle = angles[i]

                        weight = f.compute_particle_weight(p, detected_id, measured_dist, measured_angle, landmarks, sigma_d, sigma_theta)
                        log_weight = np.log(weight + 1e-300)  # Avoid log(0)
                        log_total_weight += log_weight
                        # Use log weights to avoid numerical underflow when multiplying many small probabilities
                    p.setWeight(np.exp(log_total_weight))

                # Resample particles
                particles = f.SIR_resample_particles(particles)


            ### ANGLE CORRECTION BEGIN ###
            # TODO Fix
            # why do we have a for loop for the driving to ONE LANDMARK routine?
            distance_to_landmark  = objects_dict[target_ids[curr_id_index]][0]
            distance_per_second = 41.5
        #for i in range(len(ids)):
            print(f"Angle correction loop-size: {len(ids)}")
            print(f"Target vec: {target_vec}")
            marker_x, marker_y, marker_z = target_vec[0][0]
            print(f"Marker x: {marker_x}, Marker y: {marker_y}, Marker z: {marker_z}")
            rotation_matrix, _ = cv2.Rodrigues(orientation_vec[0])
            yaw = np.arcsin(-rotation_matrix[2, 0])
            yaw_degrees = np.degrees(yaw)
            print(f"Yaw: {yaw_degrees}")
            time.sleep(1)
            move_speed_left = 46.4
            move_speed_right = 42

            threshold = 5.0
            #threshold = 10.0
            if yaw_degrees > 45:
                new_angle = 45 - (yaw_degrees - 45)
            elif yaw_degrees < -45:
                new_angle = -45 - (yaw_degrees + 45)
            else:
                new_angle = yaw_degrees

            if yaw_degrees > 0:
                new_angle = 90 - yaw_degrees
            else:
                new_angle = -90 - yaw_degrees
            print(f"New angle: {new_angle}")

            if abs(yaw_degrees) > threshold:
                if yaw_degrees > 0:
                    yaw_degrees = yaw_degrees - 5
                    print(f"Rotate left by {yaw_degrees} degrees")
                    arlo.go_diff(leftSpeed, rightSpeed, 0, 1)
                    sign = +1
                else:
                    print(f"Rotate right by {yaw_degrees} degrees")
                    arlo.go_diff(leftSpeed, rightSpeed, 1, 0)
                    sign = -1

                time.sleep(sleep1 * abs(new_angle))
                arlo.stop()
                arlo.go_diff(move_speed_left, move_speed_right, 1, 1)
                time.sleep(1/42 * marker_z * math.cos(math.radians(abs(new_angle))))
                if sign == 1:
                    arlo.go_diff(leftSpeed, rightSpeed, 1, 0)
                else:
                    arlo.go_diff(leftSpeed, rightSpeed, 0, 1)
                time.sleep(sleep1 * 94)
                arlo.stop()
            else:
                print("No need to rotate")

            #print(f"tvec: {tvec}")

            ### ANGLE CORRECTION END ###

            # Ready to drive
            drive = True
            time.sleep(2)
            print("Going to landmark")

            # start timer when we start driving to landmark
            last_execution_time = time.time()
            
            ### DRIVE TO LANDMARK BEGIN ###
            while drive:
                current_time = time.time()
    
                # Check if one second has elapsed
                if current_time - last_execution_time >= 1:
                    distance_to_landmark -= distance_per_second
                    last_execution_time = current_time 
                    print(f"A second later... Distance to landmark: {distance_to_landmark}")

                image = cam.get_next_frame()
                (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
                print(f"IDs: {ids}")

                # IF CONDITION TO MAKE SURE THE LANDMARK IS IN HINDSIGHT
                if ids is not None:
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 18, camera_matrix, dist_coeffs)
                    driving_tvec, driving_rvec = f.target_id_tvec(tvec, rvec, id_list)
                    print(f"tvec: {driving_tvec}")
                    marker_x = driving_tvec[0][0][0]
                    print(f"Marker x: {marker_x}")
                    distance_threshold = 0.2
                    base_turn_speed = 2.5
                    move_speed_left = leftSpeed
                    move_speed_right = rightSpeed

                    # Dynamic speed adjustment based on the magnitude of
                    # marker_x.
                    # Scale factor determines how sharp the turn is based on
                    # the offset
                    scale_factor = 10
                    dynamic_turn_speed = min(abs(marker_x * scale_factor), base_turn_speed * 3)

                    if marker_x > 0.05:
                        print("Turning right")
                        arlo.go_diff((move_speed_left + dynamic_turn_speed), (move_speed_right - base_turn_speed), 1, 1)
                    elif marker_x < -0.05:
                        print("Turning left")
                        arlo.go_diff((move_speed_left - base_turn_speed), (move_speed_right + dynamic_turn_speed), 1, 1)
                    else:
                        print("Driving forward")
                        arlo.go_diff(move_speed_left, move_speed_right, 1, 1)
                
                # WRONG IF CONDITION DETECTED BEGIN

                else:
                    if arlo.read_front_ping_sensor()/10 > 40 or distance_to_landmark > 60:
                        arlo.go_diff(leftSpeed, rightSpeed, 1, 1)
            
                if arlo.read_front_ping_sensor()/10 < 40 or distance_to_landmark <= 40 :
                    print("No IDs, we are now using distance and sensors to navigate :))")
                    print(f"We should stopppp! we are {distance_to_landmark} cms away from the landmark!!")
                    drive = False
                    arlo.stop()
            
            arlo.stop()
            
            # TODO Keep loooking with camera after inital loss of landmark
            #while arlo.read_front_ping_sensor()/10 > 40:
                #arlo.go_diff(leftSpeed, rightSpeed, 1, 1)

            ### DRIVE TO LANDMARK END ###

            for p in particles:
                delta_x = velocity * 2.0 * np.cos(p.getTheta())
                delta_y = velocity * 2.0 * np.sin(p.getTheta())
                p = particle.move_particle(p, delta_x, delta_y, 0.0)
                particle.add_uncertainty([p], 2.0, 0.05)

            # We have reached our goal - increment target id
            if (curr_id_index < len(target_ids)):
                curr_id_index += 1
                print(f"New target ID: {target_ids[curr_id_index]}")
            else:
                print("visited all landmarks")

            arlo.stop()
            #end of driving to landmark

        # Update particle velocities
        velocity = leftSpeed

        # Re-estimate pose after movement
        est_pose = particle.estimate_pose(particles)

        if showGUI:
            # Draw map
            f.draw_world(est_pose, particles, world, landmarks, landmarkIDs, landmark_colors)

            # Show frame
            cv2.imshow(WIN_RF1, image)

            # Show world
            cv2.imshow(WIN_World, world)

finally:
    # Make sure to clean up even if an exception occurred

    # Close all windows
    cv2.destroyAllWindows()

    # Clean-up capture thread
    cam.terminateCaptureThread()
