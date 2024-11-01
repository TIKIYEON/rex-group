import cv2
import numpy as np
from occupancy_grid import occupancy_grid, cell_size, grid_size, add_custom_landmark
import robot
from time import sleep
import math

# Create a robot object and initialize
arlo = robot.Robot()
sleep(1)

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
picam2_config = cam.create_video_configuration(
    {"size": imageSize, "format": 'RGB888'},
    controls={"FrameDurationLimits": (frame_duration_limit, frame_duration_limit)},
    queue=False)
cam.configure(picam2_config)
cam.start(show_preview=False)

# time.sleep(1)  # wait for camera to setup

# Camera calibration
camera_matrix = np.array([[3606/2.1, 0, 640], [0, 3606/2.1, 360], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1), dtype=np.float32)

# ArUco marker detection parameters
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
arucoParams = cv2.aruco.DetectorParameters_create()

class Node:
    def __init__(self, pos, theta=0.0):
        self.pos = np.array(pos)
        self.path = []
        self.parent = None
        self.theta = 0.0

    def calc_distance_to(self, other_node):
        dx = self.pos[0] - other_node.pos[0]
        dy = self.pos[1] - other_node.pos[1]
        return np.hypot(dx, dy)
    def ret_theta():
        return theta


class RRT:
    def __init__(self, start, goal, expand_dis=1, path_resolution=0.05, goal_sample_rate=0.1, max_iter=500, scale_factor=10):
        self.start = Node(start)
        self.end = Node(goal)
        self.expand_dis = expand_dis
        self.path_resolution = path_resolution
        self.goal_sample_rate = goal_sample_rate
        self.max_iter = max_iter
        self.node_list = [self.start]
        self.scale_factor = scale_factor
        self.image_width = int(grid_size[0] * self.scale_factor)
        self.image_height = int(grid_size[1] * self.scale_factor)

    def planning(self, animation=True):
        for i in range(self.max_iter):
            rnd_node = self.sample_random_node()
            nearest_ind = self.get_nearest_node_index(self.node_list, rnd_node)
            nearest_node = self.node_list[nearest_ind]

            new_node = self.steer(nearest_node, rnd_node, self.expand_dis)

            if self.check_collision_free(new_node):
                self.node_list.append(new_node)

            if animation:
                self.draw_graph(rnd_node)

            # Check if the new_node is close to the goal
            if new_node.calc_distance_to(self.end) < self.expand_dis:
                final_node = self.steer(new_node, self.end, self.expand_dis)
                if self.check_collision_free(final_node):
                    print(self.generate_final_path(final_node))
                    return self.generate_final_path(final_node)
        return None  # Path not found

    def steer(self, from_node, to_node, extend_length):
        direction = to_node.pos - from_node.pos
        distance = np.linalg.norm(direction)
        if distance == 0:
            return None
        direction = direction / distance  # Normalize direction

        new_pos = from_node.pos + direction * min(extend_length, distance)
        new_node = Node(new_pos)
        new_node.parent = from_node
        new_node.path = [from_node.pos, new_node.pos]

        return new_node


    def sample_random_node(self):
        if np.random.rand() > self.goal_sample_rate:
            rnd = Node(np.random.uniform(
                low=(-grid_size[0] * cell_size / 2, -grid_size[1] * cell_size / 2),
                high=(grid_size[0] * cell_size / 2, grid_size[1] * cell_size / 2),
                size=2))
        else:
            rnd = Node(self.end.pos)
        return rnd

    def world_to_image_coords(self, x, y):
        x_shifted = x + grid_size[0] * cell_size / 2
        y_shifted = y + grid_size[1] * cell_size / 2
        ix = int(x_shifted * self.scale_factor / cell_size)
        iy = int(y_shifted * self.scale_factor / cell_size)
        # Invert y to match image coordinates
        iy = self.image_height - iy
        return ix, iy

    def draw_graph(self, rnd_node=None, path=None):
        # Create a blank image with white background
        image = np.ones((self.image_height, self.image_width, 3), dtype=np.uint8) * 255  # White background

        # Draw occupancy grid
        occupancy_grid_resized = cv2.resize(((1 - occupancy_grid.T ) * 255).astype(np.uint8),
                                            (self.image_width, self.image_height),
                                            interpolation=cv2.INTER_NEAREST)
        # Invert y-axis since image y increases downward
        occupancy_grid_resized = cv2.flip(occupancy_grid_resized, 0)
        # Convert occupancy grid to 3-channel image
        occupancy_grid_image = cv2.cvtColor(occupancy_grid_resized, cv2.COLOR_GRAY2BGR)
        # Overlay the occupancy grid onto the image
        image = cv2.addWeighted(image, 0.7, occupancy_grid_image, 0.3, 0)

        # Draw the random node if provided
        if rnd_node is not None:
            ix, iy = self.world_to_image_coords(rnd_node.pos[0], rnd_node.pos[1])
            cv2.circle(image, (ix, iy), 5, (0, 0, 0), -1)  # Black dot

            # Draw all nodes and paths
        for node in self.node_list:
            if node.parent:
                x1, y1 = self.world_to_image_coords(node.parent.pos[0], node.parent.pos[1])
                x2, y2 = self.world_to_image_coords(node.pos[0], node.pos[1])
                cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green line

        # Draw start and end positions
        x_end, y_end = self.world_to_image_coords(self.end.pos[0], self.end.pos[1])
        x_start, y_start = self.world_to_image_coords(self.start.pos[0], self.start.pos[1])
        cv2.circle(image, (x_start, y_start), 10, (0, 0, 255), -1)  # Red circle for start
        cv2.circle(image, (x_end, y_end), 10, (0, 0, 255), -1)     # Red circle for goal

                # Draw final path if provided
        if path is not None:
            for i in range(len(path) - 1):
                x1, y1 = self.world_to_image_coords(path[i][0], path[i][1])
                x2, y2 = self.world_to_image_coords(path[i + 1][0], path[i + 1][1])
                cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red line for final path

                        # Show the image
        cv2.imshow("RRT Path Planning", image)
        cv2.waitKey(1)

    def get_nearest_node_index(self, node_list, rnd_node):
        dlist = [node.calc_distance_to(rnd_node) for node in node_list]
        return dlist.index(min(dlist))

    def check_collision_free(self, node):
        '''
        TO-DO :: we need to make sure it avoids collisions by considering the target's radius AND the robot's dimensions as well.
        '''
        if node is None:
            return False
        for p in node.path:
            x, y = p
            grid_x = int(x / cell_size + grid_size[0] // 2)
            grid_y = int(y / cell_size + grid_size[1] // 2)
            if grid_x < 0 or grid_x >= grid_size[0] or grid_y < 0 or grid_y >= grid_size[1]:
                return False
            if occupancy_grid[grid_x, grid_y]:
                return False
        return True

    def generate_final_path(self, goal_node):                           # this function starts the path from end to start
        path = [[self.end.pos[0], self.end.pos[1]]]                     # appends end node (x,y) coordinates first
        node = goal_node                                                # considers the last node its beginning
        while node.parent is not None:                                  # keeps iterating until it reaches root or start
                path.append([node.pos[0], node.pos[1]])                 # adds nodes as it travels back
                node = node.parent                                      # makes the parent, the next node
        path.append([self.start.pos[0], self.start.pos[1]])             # lastly it appends the starting point onto the path list
        return path

    def calcAngle(self,cod1,cod2):
        # Coordinates of current node and next node
        x1, y1 = cod1[0], cod1[1]
        x2, y2 = cod2[0], cod2[1]

        # Calculate differences in x and y
        delta_x = x2 - x1
        delta_y = y2 - y1

        # Calculate the angle in radians between point A and point B
        angle_radians = math.atan2(delta_y, delta_x)

        # Optionally, convert to degrees for better understanding
        angle_degrees = math.degrees(angle_radians)
        
        return(angle_degrees)

    def pathAngles(self,path0):
        path = path0[::-1]
        robotAngle = 0.0
        angleList = []
        for i in range(len(path)-1):
            print(f"path {i} : {path[i]}")
            nextAngle = self.calcAngle(path[i],path[i + 1]) - robotAngle
            robotAngle = self.calcAngle(path[i],path[i + 1])
            if nextAngle < -180:
                nextAngle = nextAngle + 360
            angleList = angleList + [nextAngle]
        return angleList

    def drive_final_path(self, path):
        leftSpeed = 65
        rightSpeed = 60.8
        sleep360 = 2.7
        turn_leftSpeed = 45
        turn_rightSpeed = 40.8
        sleepStepSize = 2.1
        final_angles = self.pathAngles(path)
        print(f"Coordinates: {path[::-1]}")
        print(f"Final angles: {final_angles}")
        
        
        for i in range(len(final_angles)):
            if final_angles[i] < 0:
                # turn right
                arlo.go_diff(turn_leftSpeed, turn_rightSpeed, 1, 0)
            else :
                # turn left
                arlo.go_diff(turn_leftSpeed, turn_rightSpeed, 0, 1)
            turn_duration = (abs(final_angles[i])/360) * sleep360
            sleep(turn_duration)
            arlo.stop()
            # sleep(1)
            # start moving towards goal
            arlo.go_diff(leftSpeed, rightSpeed, 1, 1)
            # sleep for the distance that needs to be driven, so if distance calculated = 1.40 meters
            # then we need to sleep for distance/avg_step (avg step is like 41.5 cms I believe, so 1.40/0.415)
            # drive fixed step length, which we approximate to be 2.3 seconds of sleep
            sleep(sleepStepSize)
            arlo.stop()
            # repeat
        

if __name__ == '__main__':
    path_res = 0.05
    start = [0, 0]
    goal = [2, 0]
    running = True
    
    print("Looking for landmarks...\n")

    while running:
        # Capture the frame from the camera
        image = cam.capture_array("main")

        # Detect ArUco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters=arucoParams)

        if ids is not None:
            print("Found ids: ")
            print(ids)

            # Get tvec 
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 14.5, camera_matrix, dist_coeffs)
            print(tvec)
            
            for i in range(len(ids)):
                # x_coor = (tvec[0][i][0]/10) + 5
                # y_coor = (tvec[0][i][2]/100) + 5
                x_coor = -((tvec[i][0][0]/50) + 5)
                y_coor = (tvec[i][0][2]/100) + 5
                print(x_coor)
                print(y_coor)
                # Plot the landmark
                add_custom_landmark(x_coor, y_coor, occupancy_grid, cell_size=0.1, radius=2)
                
                # for row in occupancy_grid:
                #     grid_row = [int(b) for b in row]
                #     print(grid_row)
                
            running = False                
    # Initialize the map and the RRT algorithm
    rrt = RRT(start=start, goal=goal, expand_dis=0.5, path_resolution=path_res)
    path = rrt.planning(animation=True)
    rrt.drive_final_path(path)

    if path is None:
        print("Cannot find path")
    else:
        print("Found path!")
        # Draw final path
        rrt.draw_graph(path=path)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Release resources
cam.stop()

# Keep the window open until it is closed
cv2.waitKey(0)
cv2.destroyAllWindows()
