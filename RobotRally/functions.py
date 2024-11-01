import cv2
import numpy as np
import particle


# Some color constants in BGR format
CRED = (0, 0, 255)
CGREEN = (0, 255, 0)
CBLUE = (255, 0, 0)
CCYAN = (255, 255, 0)
CYELLOW = (0, 255, 255)
CMAGENTA = (255, 0, 255)
CWHITE = (255, 255, 255)
CBLACK = (0, 0, 0)


def jet(x):
    """Colour map for drawing particles. This function determines the colour of
    a particle from its weight."""
    r = (
        (x >= 3.0 / 8.0 and x < 5.0 / 8.0) * (4.0 * x - 3.0 / 2.0)
        + (x >= 5.0 / 8.0 and x < 7.0 / 8.0)
        + (x >= 7.0 / 8.0) * (-4.0 * x + 9.0 / 2.0)
    )
    g = (
        (x >= 1.0 / 8.0 and x < 3.0 / 8.0) * (4.0 * x - 1.0 / 2.0)
        + (x >= 3.0 / 8.0 and x < 5.0 / 8.0)
        + (x >= 5.0 / 8.0 and x < 7.0 / 8.0) * (-4.0 * x + 7.0 / 2.0)
    )
    b = (
        (x < 1.0 / 8.0) * (4.0 * x + 1.0 / 2.0)
        + (x >= 1.0 / 8.0 and x < 3.0 / 8.0)
        + (x >= 3.0 / 8.0 and x < 5.0 / 8.0) * (-4.0 * x + 5.0 / 2.0)
    )

    return (255.0 * r, 255.0 * g, 255.0 * b)


def draw_world(est_pose, particles, world, landmarks, landmarkIDs, landmark_colors):
    """Visualization.
    This functions draws robots position in the world coordinate system."""
    # Fix the origin of the coordinate system
    offsetX = 100
    offsetY = 250

    # Constant needed for transforming from world coordinates to screen coordinates (flip the y-axis)
    ymax = world.shape[0]

    world[:] = CWHITE  # Clear background to white

    # Find largest weight
    max_weight = 0
    for particle in particles:
        max_weight = max(max_weight, particle.getWeight())

    # Draw particles
    for particle in particles:
        x = int(particle.getX() + offsetX)
        y = ymax - (int(particle.getY() + offsetY))
        colour = jet(particle.getWeight() / max_weight)
        cv2.circle(world, (x, y), 2, colour, 2)
        b = (
            int(particle.getX() + 15.0 * np.cos(particle.getTheta())) + offsetX,
            ymax
            - (int(particle.getY() + 15.0 * np.sin(particle.getTheta())) + offsetY),
        )
        cv2.line(world, (x, y), b, colour, 2)

    # Draw landmarks
    print(f"Landmark IDs: {landmarkIDs}")
    for i in range(len(landmarkIDs)):
        ID = landmarkIDs[i]
        # print(f"LandmarkID[I][0]: {landmarks[ID][0]}, LandmarkID[I][1]: {landmarks[ID][1]}")
        lm = (int(landmarks[ID][0] + offsetX), int(ymax - (landmarks[ID][1] + offsetY)))
        # lm = (int (landmarks[ID][0]), int(landmarks[ID][1]))
        print(f"Landmark {ID} at {lm}")
        cv2.circle(world, lm, 5, landmark_colors[i], 2)

    # Draw estimated robot pose
    a = (int(est_pose.getX()) + offsetX, ymax - (int(est_pose.getY()) + offsetY))
    b = (
        int(est_pose.getX() + 15.0 * np.cos(est_pose.getTheta())) + offsetX,
        ymax - (int(est_pose.getY() + 15.0 * np.sin(est_pose.getTheta())) + offsetY),
    )
    cv2.circle(world, a, 5, CMAGENTA, 2)
    cv2.line(world, a, b, CMAGENTA, 2)


def initialize_particles(num_particles):
    particles = []
    for i in range(num_particles):
        # Random starting points.
        p = particle.Particle(
            600.0 * np.random.ranf() - 100.0,
            600.0 * np.random.ranf() - 250.0,
            np.mod(2.0 * np.pi * np.random.ranf(), 2.0 * np.pi),
            1.0 / num_particles,
        )
        particles.append(p)
    return particles


# Compute particle weights
def compute_particle_weight(
    particle,
    detected_id,
    measured_dist,
    measured_angle,
    landmarks,
    sigma_d,
    sigma_theta,
):
    # Get the landmark's position
    landmark_pos = landmarks[detected_id]

    # Compute the distance from the particle to the landmark
    dx = landmark_pos[0] - particle.getX()
    dy = landmark_pos[1] - particle.getY()
    predicted_dist = np.sqrt(dx**2 + dy**2)

    # e_theta for all particles
    e_theta = np.array([(np.cos(particle.getTheta())), (np.sin(particle.getTheta()))])

    # e_l for all particles
    e_l = np.array([(dx, dy)]) / predicted_dist

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
    dist_weight = np.exp(-0.5 * ((measured_dist - predicted_dist) ** 2 / sigma_d**2))

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
            1.0 / len(particles),  # Equal weight after resampling
        )
        new_particles.append(new_particle)

    return new_particles


def target_id_tvec(tvec, rvec, id_list):
    print("INSIDE TARGET ID TVEC")
    print(f"ID List: {len(id_list)}")
    print(f"tvec: {tvec}")
    print(f"rvec: {rvec}")

    if len(id_list) > 1 and len(tvec[0]) > 1:
        print(id_list[0])
        print(id_list[1])

        if tvec[0][id_list[0]][2] < tvec[0][id_list[1]][2]:
            return [[tvec[0][id_list[0]]]], [[rvec[0][id_list[0]]]]
        else:
            return [[tvec[0][id_list[1]]]], [[rvec[0][id_list[1]]]]
    else:
        print("poop")
        return tvec, rvec

# if len(id_list) > 1 and len(tvec[0]) > 1:
#     print(id_list[0])
#     print(id_list[1])


#     if tvec[0][id_list[0]][2] < tvec[0][id_list[1]][2]:
#         tvec = [[tvec[0][id_list[0]]]]
#         rvec = [[rvec[0][id_list[0]]]]
#     else:
#         tvec = [[tvec[0][id_list[1]]]]
#         rvec = [[rvec[0][id_list[1]]]]