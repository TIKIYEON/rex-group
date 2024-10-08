import cv2
import particle
import camera
import numpy as np
import time
from timeit import default_timer as timer
import sys

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
    sys.path.append("../../../../Arlo/python")

try:
    import robot
    onRobot = True
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
landmarkIDs = [1, 2]
landmarks = {
    1: (0.0, 0.0),  # Coordinates for landmark 1
    2: (300.0, 0.0)  # Coordinates for landmark 2
}
landmark_colors = [CRED, CGREEN] # Colors used when drawing the landmarks

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

    # Compute the predicted angle to the landmark
    predicted_angle = np.arctan2(dy, dx) - particle.getTheta()

    # Normalize the angle to the range [-pi, pi]
    predicted_angle = np.arctan2(np.sin(predicted_angle), np.cos(predicted_angle))

    # Distance weight (Gaussian likelihood)
    dist_weight = (1.0 / np.sqrt(2 * np.pi * sigma_d**2)) * np.exp(-0.5 * ((measured_dist - predicted_dist)**2 / sigma_d**2))

    # Orientation weight (Gaussian likelihood)
    angle_weight = (1.0 / np.sqrt(2 * np.pi * sigma_theta**2)) * np.exp(-0.5 * ((measured_angle - predicted_angle)**2 / sigma_theta**2))

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
    normalized_weights = [p.getWeight() / total_weight for p in particles]

    # Step 2: Generate cumulative distribution of weights
    cumulative_sum = np.cumsum(normalized_weights)

    # Step 3: Resampling using the cumulative distribution
    new_particles = []
    for _ in range(len(particles)):
        r = np.random.uniform(0, 1)
        index = np.searchsorted(cumulative_sum, r)
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
sigma_d = 10
sigma_theta = np.pi / 8

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

    # Initialize the robot (XXX: You do this)

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

        # Fetch next frame
        colour = cam.get_next_frame()

        # Time step (dt)
        dt = 0.1  # Adjust this based on your control loop timing

        # Update each particle's state
        for p in particles:
            # Compute the change in x, y, and theta
            delta_x = velocity * dt * np.cos(p.getTheta())
            delta_y = velocity * dt * np.sin(p.getTheta())
            delta_theta = angular_velocity * dt

            # Move the particle
            p.setX(p.getX() + delta_x)
            p.setY(p.getY() + delta_y)
            p.setTheta(p.getTheta() + delta_theta)

            # Optionally, add uncertainty to the particle motion
            p.setX(p.getX() + np.random.normal(0, 0.5))  # Motion noise sigma
            p.setY(p.getY() + np.random.normal(0, 0.5))
            p.setTheta(p.getTheta() + np.random.normal(0, 0.05))  # Motion noise theta sigma

        # Detect objects
        objectIDs, dists, angles = cam.detect_aruco_objects(colour)
        if objectIDs is not None:
            for i in range(len(objectIDs)):
                print("Object ID = ", objectIDs[i], ", Distance = ", dists[i], ", angle = ", angles[i])
            # For each particle, compute the weight
            for p in particles:
                total_weight = 1.0
                for i in range(len(objectIDs)):
                    detected_id = objectIDs[i]
                    measured_dist = dists[i]
                    measured_angle = angles[i]

                    weight = compute_particle_weight(p, detected_id, measured_dist, measured_angle, landmarks, sigma_d, sigma_theta)
                    total_weight *= weight
                p.setWeight(total_weight)

            # Resample particles
            particles = SIR_resample_particles(particles)
        else:
            # No observations; continue with current weights
            pass

        # Estimate pose
        est_pose = particle.estimate_pose(particles)

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
