
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

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
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
    def __init__(self, rate=30):
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

        # ROS2 publisher for image topic
        self.image_pub = self.create_publisher(Image, 'camera/color/image_raw', 10)
        self.timer = self.create_timer(1.0/rate, self.capture_and_publish)  # Adjust timer based on desired frame rate

        # mediapipe setup
        base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
        options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=1)
        self.detector = vision.HandLandmarker.create_from_options(options)
        print("Node init finished")
        #self.capture_and_publish()

    def capture_and_publish(self):
        #while rclpy.ok():
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                print("no frame")
                return
                #continue

            # Convert image to OpenCV format
            color_image = np.asanyarray(color_frame.get_data())
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=color_image)
            detection_result = self.detector.detect(mp_img)
            print(detection_result)
            
            annotated_image = draw_landmarks_on_image(mp_img.numpy_view(), detection_result)
            
            img = annotated_image 
            # Create ROS Image message
            image_msg = self.bridge.cv2_to_imgmsg(img, encoding="bgr8")
            image_msg.header.frame_id = 'camera_color_frame'
            image_msg.header.stamp = self.get_clock().now().to_msg()

            self.image_pub.publish(image_msg)

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
