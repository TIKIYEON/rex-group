import cv2
import particle
import camera
import numpy as np
import time
from timeit import default_timer as timer
import sys
import math

# Flags
showGUI = True  # Whether or not to open GUI windows
onRobot = True  # Whether or not we are running on the Arlo robot

def isRunningOnArlo():
    """Return True if we are running on Arlo, otherwise False.
      You can use this flag to switch the code from running on you laptop to Arlo - you need to do the programming here!
    """
    return onRobot

if isRunningOnArlo():
    # XXX: You need to change this path to point to where your robot.py file is located
    sys.path.append("./")
    print("On Arlo!")

arlo = None

try:
    import robot
    onRobot = True
    arlo = robot.Robot()
except ImportError:
    print("selflocalize.py: robot module not present - forcing not running on Arlo!")
    onRobot = False

# Some color constants in BGR format
CRED = (0, 0, 255)
CGREEN = (0, 255, 0)
CBLUE = (255, 0, 0)
CCYAN = (255, 255, 0)
CYELLOW = (0, 255, 255)
CMAGENTA = (255, 0, 255)
CWHITE = (255, 255, 255)
CBLACK = (0, 0, 0)

# Landmarks.
# The robot knows the position of 2 landmarks. Their coordinates are in the unit centimeters [cm].
landmarkIDs = [6, 4]
landmarks = {
    6: (0.0, 0.0),  # Coordinates for landmark 1
    4: (300.0, 0.0)  # Coordinates for landmark 2
}
landmark_colors = [CRED, CGREEN] # Colors used when drawing the landmarks

# Robot movement controls
turn_speed = 2.5
turn_time = 0.1
move_speed_left = 58
move_speed_right = 50
forward_speed = 42 # Sleep(1)

def jet(x):
    """Colour map for drawing particles. This function determines the colour of
    a particle from its weight."""
    r = (x >= 3.0/8.0 and x < 5.0/8.0) * (4.0 * x - 3.0/2.0) + (x >= 5.0/8.0 and x < 7.0/8.0) + (x >= 7.0/8.0) * (-4.0 * x + 9.0/2.0)
    g = (x >= 1.0/8.0 and x < 3.0/8.0) * (4.0 * x - 1.0/2.0) + (x >= 3.0/8.0 and x < 5.0/8.0) + (x >= 5.0/8.0 and x < 7.0/8.0) * (-4.0 * x + 7.0/2.0)
    b = (x < 1.0/8.0) * (4.0 * x + 1.0/2.0) + (x >= 1.0/8.0 and x < 3.0/8.0) + (x >= 3.0/8.0 and x < 5.0/8.0) * (-4.0 * x + 5.0/2.0)

    return (255.0*r, 255.0*g, 255.0*b)

def draw_world(est_pose, particles, world):
    """Visualization.
    This functions draws robots position in the world coordinate system."""
    # Fix the origin of the coordinate system
    offsetX = 100
    offsetY = 250

    # Constant needed for transforming from world coordinates to screen coordinates (flip the y-axis)
    ymax = world.shape[0]

    world[:] = CWHITE # Clear background to white

    # Find largest weight
    max_weight = 0
    for particle in particles:
        max_weight = max(max_weight, particle.getWeight())

    # Draw particles
    for particle in particles:
        x = int(particle.getX() + offsetX)
        y = ymax - (int(particle.getY() + offsetY))
        colour = jet(particle.getWeight() / max_weight)
        cv2.circle(world, (x,y), 2, colour, 2)
        b = (int(particle.getX() + 15.0*np.cos(particle.getTheta()))+offsetX,
                                     ymax - (int(particle.getY() + 15.0*np.sin(particle.getTheta()))+offsetY))
        cv2.line(world, (x,y), b, colour, 2)

    # Draw landmarks
    for i in range(len(landmarkIDs)):
        ID = landmarkIDs[i]
        lm = (int(landmarks[ID][0] + offsetX), int(ymax - (landmarks[ID][1] + offsetY)))
        cv2.circle(world, lm, 5, landmark_colors[i], 2)

    # Draw estimated robot pose
    a = (int(est_pose.getX())+offsetX, ymax-(int(est_pose.getY())+offsetY))
    b = (int(est_pose.getX() + 15.0*np.cos(est_pose.getTheta()))+offsetX,
                                 ymax-(int(est_pose.getY() + 15.0*np.sin(est_pose.getTheta()))+offsetY))
    cv2.circle(world, a, 5, CMAGENTA, 2)
    cv2.line(world, a, b, CMAGENTA, 2)

def initialize_particles(num_particles):
    particles = []
    for i in range(num_particles):
        # Random starting points.
        p = particle.Particle(600.0*np.random.ranf() - 100.0, 600.0*np.random.ranf() - 250.0, np.mod(2.0*np.pi*np.random.ranf(), 2.0*np.pi), 1.0/num_particles)
        particles.append(p)
    return particles

# Compute particle weights
def compute_particle_weight(particle, detected_id, measured_dist, measured_angle, landmarks, sigma_d, sigma_theta):
    # Get the landmark's position
    landmark_pos = landmarks[detected_id]

    # Compute the distance from the particle to the landmark
    dx = landmark_pos[0] - particle.getX()
    dy = landmark_pos[1] - particle.getY()
    predicted_dist = np.sqrt(dx**2 + dy**2)

    # e_theta for all particles
    e_theta = np.array([(np.cos(particle.getTheta())),(np.sin(particle.getTheta()))])

    #e_l for all particles
    e_l = np.array([(dx,dy)])  / predicted_dist

    # Dot product
    e_dot = np.dot(e_theta, e_l.flatten())
    # Cross product
    e_cross = np.cross(e_theta, e_l)

    # Predicted angle
    predicted_angle = np.arctan2(e_cross, e_dot)

    # Compute the angle difference
    angle_diff = measured_angle - predicted_angle
    angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))  # Normalize

    # Distance weight (Gaussian likelihood without normalizing constant)
    dist_weight = np.exp(-0.5 * ((measured_dist - predicted_dist)**2 / sigma_d**2))

    # Orientation weight (Gaussian likelihood without normalizing constant)
    angle_weight = np.exp(-0.5 * (angle_diff**2 / sigma_theta**2))

    # Total weight is the product of both likelihoods
    return dist_weight * angle_weight

def SIR_resample_particles(particles):
    # Step 1: Normalize the weights
    total_weight = sum([p.getWeight() for p in particles])
    if total_weight == 0:
        # If all weights are zero, set equal weights
        for p in particles:
            p.setWeight(1.0 / len(particles))
        total_weight = 1.0
    else:
        for p in particles:
            p.setWeight(p.getWeight() / total_weight)

    normalized_weights = [p.getWeight() for p in particles]

    # Step 2: Generate cumulative distribution of weights
    cumulative_sum = np.cumsum(normalized_weights)

    # Step 3: Resampling using the cumulative distribution
    new_particles = []
    for _ in range(len(particles)):
        r = np.random.uniform(0, 1 - 1e-10)  # Avoid r == 1.0
        index = np.searchsorted(cumulative_sum, r)
        if index >= len(particles):
            index = len(particles) - 1  # Ensure index is within bounds
        # Clone the selected particle and give it equal weight
        new_particle = particle.Particle(
            particles[index].getX(),
            particles[index].getY(),
            particles[index].getTheta(),
            1.0 / len(particles)  # Equal weight after resampling
        )
        new_particles.append(new_particle)

    return new_particles

# Define the standard deviations for distance and angle (you can tweak these)
sigma_d = 5
sigma_theta = np.pi / 12

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
    particles = initialize_particles(num_particles)

    est_pose = particle.estimate_pose(particles) # The estimate of the robots current pose

    # Driving parameters
    velocity = 0.0 # cm/sec
    angular_velocity = 0.0 # radians/sec

    #center_x = 150  # Center point (between landmarks at (0, 0) and (300, 0))
    #center_y = 0
    center_x = (landmarks[6][0] + landmarks[4][0]) / 2
    center_y = (landmarks[6][1] + landmarks[4][1]) / 2

    # Initialize the robot (XXX: You do this)
    arlo = arlo.Robot()

    # Allocate space for world map
    world = np.zeros((500,500,3), dtype=np.uint8)

    # Draw map
    draw_world(est_pose, particles, world)

    print("Opening and initializing camera")
    if isRunningOnArlo():
        #cam = camera.Camera(0, robottype='arlo', useCaptureThread=True)
        cam = camera.Camera(0, robottype='arlo', useCaptureThread=False)
    else:
        #cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=True)
        cam = camera.Camera(0, robottype='macbookpro', useCaptureThread=False)

    while True:
        # Handle user input for control
        action = cv2.waitKey(10)
        if action == ord('q'):  # Quit
            break

        if not isRunningOnArlo():
            if action == ord('w'):  # Forward
                velocity += 4.0
            elif action == ord('x'):  # Backwards
                velocity -= 4.0
            elif action == ord('s'):  # Stop
                velocity = 0.0
                angular_velocity = 0.0
            elif action == ord('a'):  # Left
                angular_velocity += 0.2
            elif action == ord('d'):  # Right
                angular_velocity -= 0.2

        # Use motor controls to update particles
        # XXX: Make the robot drive
        # XXX: You do this

        #  Rotate to detect both landmarks
        arlo.go_diff(40, 40, 0, 1)
        time.sleep(turn_time)
        arlo.stop()
        time.sleep(turn_time)

        # Fetch next frame
        colour = cam.get_next_frame()

        # Time step (dt)
        dt = 0.1
        # This could be a large value if we wanted to simulate a faster robot
        # Or smaller if we wanted to simulate a slower robot

        #particle.add_uncertainty(particles, 10, 5)  # Motion noise

        # Update each particle's state
        for p in particles:
            # Compute the change in x, y, and theta
            delta_x = velocity * dt * np.cos(p.getTheta())
            delta_y = velocity * dt * np.sin(p.getTheta())
            delta_theta = angular_velocity * dt

            # Move the particle
            #p.setX(p.getX() + delta_x)
            #p.setY(p.getY() + delta_y)
            #p.setTheta(p.getTheta() + delta_theta)

            # p.setX(p.getX() + np.random.normal(0, 5))  # Motion noise sigma
            # p.setY(p.getY() + np.random.normal(0, 5))
            # p.setTheta(p.getTheta() + np.random.normal(0, 2.5))  # Motion noise theta sigma


            p = particle.move_particle(p, delta_x, delta_y, delta_theta)

            # I Add uncertainty to the particle's pose
            particle.add_uncertainty([p], 0.5, 0.1)  # Motion noise


        # Detect objects
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        
        # TODO: I estimate the pose of the robot
        est_pose = particle.estimate_pose(particles)
        
        # Update particle weights
        if objectIDs is not None:
            # removing the duplicates for each ID and picking the shortest distance for each
            objects_dict = {}
            for i in range(len(objectIDs)):
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

                    weight = compute_particle_weight(p, detected_id, measured_dist, measured_angle, landmarks, sigma_d, sigma_theta)
                    log_weight = np.log(weight + 1e-300)  # Avoid log(0)
                    log_total_weight += log_weight
                    # Use log weights to avoid numerical underflow when multiplying many small probabilities
                p.setWeight(np.exp(log_total_weight))

            # Resample particles
            particles = SIR_resample_particles(particles)
        else:
            # No observations; continue with current weights
            pass

        # I Center between the landmarks
        # Distance and angle to center
        dx = center_x - est_pose.getX()
        dy = center_y - est_pose.getY()
        desired_theta = np.arctan2(dy, dx)
        delta_theta = desired_theta - est_pose.getTheta()
        # Normalise angle to be between [-pi, pi]
        delta_theta = np.arctan2(np.sin(delta_theta), np.cos(delta_theta))
        center_dist = np.sqrt(dx**2 + dy**2)

        # I Rotate the robot to face the center
        turn_time = abs(delta_theta) / turn_speed
        if delta_theta > 0:
            arlo.go_diff(turn_speed, turn_speed, 0, 1)
        else:
            arlo.go_diff(turn_speed, turn_speed, 1, 0)
        time.sleep(turn_time)
        arlo.stop()
        time.sleep(turn_time)

        # I Update particles after rotation
        angular_velocity = delta_theta / turn_time
        for p in particles:
            p = particle.move_particle(p, 0, 0, delta_theta)
            particle.add_uncertainty([p], 0.5, 0.1)

        # I Re-estimate pose
        est_pose = particle.estimate_pose(particles)

        # I Recalculate dx and dy
        dx = center_x - est_pose.getX()
        dy = center_y - est_pose.getY()
        center_dist = np.sqrt(dx**2 + dy**2)
        move_time = center_dist / forward_speed

        #  I Move the robot
        arlo.go_diff(move_speed_left, move_speed_right, 1, 1)
        time.sleep(move_time)
        arlo.stop()
        time.sleep(0.1)

        # I Update particle velocities
        velocity = forward_speed

        # I Update particles after movement
        for p in particles:
            delta_x = velocity * move_time * np.cos(p.getTheta())
            delta_y = velocity * move_time * np.sin(p.getTheta())
            p = particle.move_particle(p, delta_x, delta_y, 0.0)
            particle.add_uncertainty([p], 2.0, 0.05)

        # I Re-estimate pose after movement
        est_pose = particle.estimate_pose(particles)

        # I Update the world map
        draw_world(est_pose, particles, world)

        if showGUI:
            # Draw map
            draw_world(est_pose, particles, world)

            # Show frame
            cv2.imshow(WIN_RF1, colour)

            # Show world
            cv2.imshow(WIN_World, world)

finally:
    # Make sure to clean up even if an exception occurred

    # Close all windows
    cv2.destroyAllWindows()

    # Clean-up capture thread
    cam.terminateCaptureThread()
