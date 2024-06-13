import rclpy
from rclpy.node import Node

from sensor_msgs.msg import Image
from std_msgs.msg import String

from cv_bridge import CvBridge, CvBridgeError
import cv2
import time
import io

bridge = CvBridge ()

class Publisher(Node):

    def __init__(self):
        super().__init__('publisher')
        self.publisher_ = self.create_publisher(Image, 'images', 10)
        timer_period = 4
        self.timer = self.create_timer (timer_period, self.timer_callback)
        self.i = 1

    def timer_callback(self):
        image_path = '/root/images/'
        try:
            file_name = f'{image_path}{self.i}.jpg'
            # Read in the images using open cv2 library imread function
            img = cv2.imread(file_name)
            # convert the cv2 image to sensor_msgs Image type using CV Bridge function cv2_to_imgmsg
            cv_img = bridge.cv2_to_imgmsg(img, encoding='passthrough')
            self.publisher_.publish (cv_img)
            self.get_logger().info('Publishing: "%d"' % self.i)
            self.i += 1
        except Exception as e:
            print (e)


def main(args=None):
    rclpy.init(args=args)
    publisher = Publisher()
    rclpy.spin(publisher)
    publisher.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
