
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import numpy as np
import cv2
import pyrealsense2 as rs


import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from geometry_msgs.msg import Point, PoseStamped

from .arm_control import ArmControl

def draw_landmarks_on_image(rgb_image, detection_result):

    MARGIN = 10  # pixels
    FONT_SIZE = 1
    FONT_THICKNESS = 1
    HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        # Draw the hand landmarks.
        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
        annotated_image,
        hand_landmarks_proto,
        solutions.hands.HAND_CONNECTIONS,
        solutions.drawing_styles.get_default_hand_landmarks_style(),
        solutions.drawing_styles.get_default_hand_connections_style())

        # Get the top left corner of the detected hand's bounding box.
        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - MARGIN

        # Draw handedness (left or right hand) on the image.
        cv2.putText(annotated_image, f"{handedness[0].category_name}",
                    (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

    return annotated_image


class GestureTrackerNode(Node):
    def __init__(self, camera_rate=10, control_rate=50):
        super().__init__('realsense_node')
        self.bridge = CvBridge()

        # Configure RealSense pipeline
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # Set camera resolution (adjust as needed)
        self.width = 640
        self.height = 480
        self.config.enable_stream(rs.stream.color, self.width, self.height, rs.format.bgr8, 30)

        # Start streaming
        self.pipeline.start(self.config)
        #self.pipeline.start()

        self.robot_control = ArmControl()
        self.target_orientation = np.array([0., 1., 0., 0.]) # face down

        # Publishers
        self.image_pub = self.create_publisher(Image, 'camera/color/image_raw', 10)
        self.hand_pos_pub = self.create_publisher(Point, '/hand/position', 5)
        self.robot_pose_pub = self.create_publisher(PoseStamped, '/robot/pose', 5)
        
        # Robot pose msg
        self.pose_msg = PoseStamped()
        self.pose_msg.pose.orientation.w = self.target_orientation[0]
        self.pose_msg.pose.orientation.x = self.target_orientation[1]
        self.pose_msg.pose.orientation.y = self.target_orientation[2]
        self.pose_msg.pose.orientation.z = self.target_orientation[3]
        self.pose_msg.header.frame_id = 'attachment_site'
        
        # Mediapipe setup
        base_options = python.BaseOptions(model_asset_path='src/gesture_tracker_ros/resource/hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1, running_mode=vision.RunningMode.LIVE_STREAM,
                                               result_callback=self.detection_cb)
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.latest_detection_timestamp = None
        self.latest_detection = None

        self.timer = self.create_timer(1.0/camera_rate, self.capture)  
        self.timer = self.create_timer(1.0/control_rate, self.robot_control.step)
        self.robot_control.start()
        print("Node init finished")

    def capture(self):
        """
        Capture a frame from the relasense
        """
        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("no frame")
            return

        timestamp = self.get_clock().now()
        
        if (self.latest_detection_timestamp is not None and self.latest_detection_timestamp.nanoseconds < timestamp.nanoseconds - 0.2*1e9):
            print("lagging, skip frame")
            return
        # Convert image to OpenCV format
        color_image = np.asanyarray(color_frame.get_data())
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image)
        self.detector.detect_async(mp_img, int(timestamp.nanoseconds / 1e6))
            

    def detection_cb(self, result: mp.tasks.vision.HandLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
        self.latest_detection_timestamp = rclpy.time.Time(nanoseconds=timestamp_ms * 1e6)
        
        self.map_hand_to_robot_coords(result)

        annotated_image = draw_landmarks_on_image(output_image.numpy_view(), result)

        img = annotated_image  
        # Create ROS Image message
        image_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
        image_msg.header.frame_id = 'camera_color_frame'
        image_msg.header.stamp = self.latest_detection_timestamp.to_msg()

        self.image_pub.publish(image_msg)
        #print(image_msg)

    def map_hand_to_robot_coords(self, result: mp.tasks.vision.HandLandmarkerResult):
        """
        Map the normalized landmarks in [0 1] image height, width to the robot arm
        """
        if len(result.handedness) == 0:
            return
        img_bounds = np.array([[0, 1], [0, 1], [0, 1]])
        robot_bounds = np.array([[0.15, 0.4], [-0.2, 0.2], [0.2, 0.2]])
        # 8 is index finger tip. Get its XYZ coords
        hand_pos = np.array([getattr(result.hand_landmarks[0][8], dim) for dim in ('x', 'y', 'z')])
        robot_pos = np.empty(3)
        
        robot_pos = (hand_pos - img_bounds.min(axis=1)) * (robot_bounds[:, 1] - robot_bounds[:, 0]) / (img_bounds[:, 1] - img_bounds[:, 0]) + robot_bounds.min(axis=1)
        self.hand_pos_pub.publish(Point(x=hand_pos[0], y=hand_pos[1], z=hand_pos[2]))

        self.pose_msg.header.stamp = self.get_clock().now().to_msg()
        self.pose_msg.pose.position.x = robot_pos[0]
        self.pose_msg.pose.position.y = robot_pos[1]
        self.pose_msg.pose.position.z = robot_pos[2]
        self.robot_pose_pub.publish(self.pose_msg)

        self.robot_control.update_target(robot_pos, self.target_orientation)
        
        
    def control_loop(self):
        res = self.robot_control.step()
        if res is not None:
            print(f"Error: {res}")
            self.robot_control.stop()
        # TODO pulbish joint angles

    def destroy_node(self, args):
        self.robot_control.stop()
        self.pipeline.stop()
        super().destroy_node(*args)

def main():

    rclpy.init()

    node = GestureTrackerNode()

    rclpy.spin(node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
