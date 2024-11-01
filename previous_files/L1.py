import cv2
import time
import robot
import math
import numpy as np
from pprint import *
from time import sleep

# Create a robot object and initialize
arlo = robot.Robot()
sleep(1)

sleep360 = 2.45 * 2
sleep1 = sleep360 / 360
leftSpeed = 42
rightSpeed = 42

try:
    import picamera2
    print("Camera.py: Using picamera2 module")
except ImportError:
    print("Camera.py: picamera2 module not available")
    exit(-1)

# Open a camera device for capturing
imageSize = (1280, 720)
FPS = 30
cam = picamera2.Picamera2()
frame_duration_limit = int(1/FPS * 1000000)  # Microseconds
picam2_config = cam.create_video_configuration({"size": imageSize, "format": 'RGB888'}, controls={"FrameDurationLimits": (frame_duration_limit, frame_duration_limit)}, queue=False)
cam.configure(picam2_config)
cam.start(show_preview=False)

time.sleep(1)  # wait for camera to setup

# Camera calibration
camera_matrix = np.array([[3606/2, 0, 960], [0, 3606/2, 540], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# Open a window for camera feed
WIN_RF = "Camera Feed"
cv2.namedWindow(WIN_RF)
cv2.moveWindow(WIN_RF, 100, 100)

# Create a blank canvas for plotting landmarks
canvas = np.ones((500, 750, 3), dtype=np.uint8) * 255  # 500x500 white canvas

arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
arucoParams = cv2.aruco.DetectorParameters_create()

landmarks = []
separateLandmark = []

curr_id_index = 0
target_ids = [1, 2, 3, 4, 1]

while True:
    image = cam.capture_array("main")
    #cv2.imshow(WIN_RF, image)

    (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

    #if ids is None or (ids is not None and ids[0][0] != target_ids[curr_id_index]):
    if ids is None or (ids is not None and target_ids[curr_id_index] not in ids[0]):
        # Lookging for landmark
        arlo.go_diff(leftSpeed, rightSpeed, 1, 0)
        sleep(sleep360 / 16)
        arlo.stop()
        sleep(sleep360 / 8)
    else:
        # Test case
        print("Found correct target id")
        # [i for i, x in enumerate(a) if x == 2]
        ids_list = [i for i, x in enumerate(ids) if x == target_ids[curr_id_index]]
        print(ids_list)


        # Decide which tvec to use
        rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 18, camera_matrix, dist_coeffs)

        if len(ids_list) > 1:
            print(ids_list[0])
            print(ids_list[1])
            if tvec[0][ids_list[0]][2] < tvec[0][ids_list[1]][2]:
                # We choose the first tvec
                tvec = [[tvec[0][ids_list[0]]]]
            else:
                # We choose the latter
                tvec = [[tvec[0][ids_list[1]]]]

            # min(tvec[0][ids_list[0]][2], [tvec[0][ids_list[1]][2])

        cv2.drawFrameAxes(image, camera_matrix, dist_coeffs, rvec[0], tvec[0], 18)

        print(tvec)

        # Adjust scaling factors and center position to fit the canvas
        scale_factor = 3  # Adjust the scaling factor to fit the canvas better
        offset = 250  # Offset to center around the middle of the canvas

        for i in range(len(ids)):
            marker_x, marker_y, marker_z = tvec[i][0]  # x, y, z coordinates of the marker

            # Scale and shift the coordinates to fit the canvas
            # Scale and shift the coordinates to fit the canvas
            #canvas_x = int(offset + marker_x * scale_factor)
            #canvas_y = int(offset - marker_z * scale_factor)  # Invert z-axis to match screen coordinates

            # Debug print to check values
            #print(f"Marker ID {ids[i][0]} at (marker_x, marker_z): ({marker_x}, {marker_z}) -> (canvas_x, canvas_y): ({canvas_x}, {canvas_y})")

            rotation_matrix, _ = cv2.Rodrigues(rvec[i][0])
            yaw = np.arcsin(-rotation_matrix[2, 0])
            #print(f"Rotation matrix: {rotation_matrix}")
            #print(f"Yaw: {yaw})")
            yaw_degrees = np.degrees(yaw)
            #print("GRADER FORHÅBENTLIG:")
            print(yaw_degrees)
            sleep(1)
            move_speed_left = 60
            move_speed_right = 60

            threshold = 5.0 # degrees
            # the newAngle is 0 to 45 and from 45 and back to 0 if the angle is really from 46 to 90
            if yaw_degrees > 45:
                newAngle = 45 - (yaw_degrees - 45)
            elif yaw_degrees < -45:
                print("HOW")
                print(-45 - (yaw_degrees + 45))
                newAngle = -45 - (yaw_degrees + 45)
            else:
                newAngle = yaw_degrees

            if yaw_degrees > 0:
                newAngle = 90 - yaw_degrees
            else:
                newAngle = -90 - yaw_degrees

            print(f"NewAngle: {newAngle}")
            if abs(yaw_degrees) > threshold:
                if yaw_degrees > 0:
                    yaw_degrees = yaw_degrees - 5
                    print(f"Rotate Left by {yaw_degrees} degrees")
                    arlo.go_diff(leftSpeed, rightSpeed, 0, 1)
                    sign = +1
                else:
                    print(f"Rotate Right by {abs(yaw_degrees)} degrees")
                    arlo.go_diff(leftSpeed, rightSpeed, 1, 0)
                    sign = -1

                sleep(sleep1 * abs(newAngle))
                print(sleep1 * abs(newAngle))
                arlo.stop()
                arlo.go_diff(move_speed_left, move_speed_right, 1, 1) # drive straight
                sleep(1/42 * marker_z * math.cos(math.radians(abs(newAngle))))
                if sign == 1:
                    arlo.go_diff(leftSpeed, rightSpeed, 1, 0)
                else:
                    arlo.go_diff(leftSpeed, rightSpeed, 0, 1)
                sleep(sleep1 * 94)
                arlo.stop()

            else:

                print("No significant rotation needed")


            print(f"tvec: {tvec}")

            drive = True
            sleep(2)
            print("Going to landmark")

            while drive:
                image = cam.capture_array("main")
                (corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)
                print(f"ids: {ids}")
                if ids is not None:
                    rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 18, camera_matrix, dist_coeffs)
                    print(f"tvec: {tvec}")
                    marker_x = tvec[0][0][0]
                    print(f"Marker x: {marker_x}")
                    distance_threshold = 0.2
                    turn_speed = 2.5
                    move_speed_left = leftSpeed
                    move_speed_right = rightSpeed
                    if marker_x > 0.05:
                        print("Turning right")
                        arlo.go_diff((move_speed_left + turn_speed), (move_speed_right - turn_speed), 1, 1) # turn right
                    elif marker_x < -0.05:
                        print("Turning left")
                        arlo.go_diff(move_speed_left - turn_speed, move_speed_right + turn_speed, 1, 1)  # turn left
                    else:
                        arlo.go_diff(move_speed_left, move_speed_right, 1, 1)  # go straight
                else:
                    print("No IDS!")
                    drive = False
                    print("slut")
                    arlo.stop()

            arlo.stop()

            while arlo.read_front_ping_sensor()/10 > 40:
                arlo.go_diff(leftSpeed, rightSpeed, 1, 1)
                print(arlo.read_front_ping_sensor())

            # We have reached our goal - increament target id
            if (curr_id_index < len(target_ids)):
                curr_id_index += 1
                print(f"New target ID: {target_ids[curr_id_index]}")
            else:
                print("Visited all landmarks")

            arlo.stop()

            """
            # Store the landmark coordinates and ID
            landmarks.append((ids[i][0], (marker_x, marker_y, marker_z)))
            distance_threshold = 1000.2
            turn_speed = 2.5
            move_speed_left = 58
            move_speed_right = 50
            if abs(marker_z) > distance_threshold:
                if marker_x > 0.05:
                    arlo.go_diff((move_speed_left + turn_speed), (move_speed_right - turn_speed), 1, 1) # turn left
                elif marker_x < -0.05:
                    arlo.go_diff(move_speed_left - turn_speed, move_speed_right + turn_speed, 1, 1)  # turn right
                else:
                    arlo.go_diff(move_speed_left, move_speed_right, 1, 1)  # turn right

                separateLandmark.append(landmarks)
                print(f"Landmarks stored: {landmarks}")
                print(f"Landmarks persisted: {separateLandmark}")

                print(f"Translation vector (x, y, z): {tvec[0][0]}")
            print("færdig")"""
        # Display the camera feed
        #cv2.imshow(WIN_RF, image)

        # Display the canvas with the landmark plot
        cv2.imshow('Landmark Map', canvas)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release resources
cam.stop()
cv2.destroyAllWindows()

