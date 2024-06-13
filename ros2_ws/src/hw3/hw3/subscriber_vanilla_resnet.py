#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import time
import PIL
import json
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.io import read_image
import torch

bridge = CvBridge()

# Load the resnet 50 model and the weights here

weights = ResNet50_Weights.DEFAULT
model = resnet50(weights=weights, progress=False)

transforms = weights.transforms()
model = model.eval ()

def get_prediction():
    d = read_image('camera_image.jpeg')
    # Write the code here:
    # Transform the image to the input format
    # Get the output from model
    # Get the classes names
    # return the found objects
    img_processed = transforms(d)

    img_tensor = torch.unsqueeze(img_processed, 0)
    prediction = model(img_tensor).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    return category_name

class Subscriber (Node):

    def __init__ (self):
        super ().__init__ ("subscriber")
        self.subscription = self.create_subscription (
                Image,
                "images",
                self.listener_callback,
                10)
        self.subscription

    def listener_callback (self, msg):
        try:
            # pass
            # Read in the images in cv2 format from topic using bridge imgmsg_to_cv2 function
            cv_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except CvBridgeError as e:
            print (e)
        else:
            # pass
            # write cv2 image in a local file 'camera_image.jpeg' using imwrite function
            cv2.imwrite('camera_image.jpeg', cv_image)
        ans = get_prediction () 
        print (ans)

def main (args = None):
    rclpy.init (args = args)
    node = Subscriber()
    rclpy.spin (node)
    node.destroy_node ()
    rclpy.shutdown ()

if __name__ == '__main__':
    main ()

