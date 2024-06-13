# Objects-Detection-via-ROS2-Topic-Subscription

## Introduction
This project involves understanding how deep neural networks work in vision and using a trained neural network for inference with PyTorch. The objective is to choose models, write inference code for image classification, and handle ROS2 topic subscription and publishing.

**Features**
  Image Acquisition and Publishing:
    Capture images using a monocular camera and publish them as ROS2 topics.
    Convert OpenCV images to ROS sensor messages.
  Model Inference:
    Utilize pretrained Mask RCNN ResNet 50 and vanilla ResNet models for object detection.
    Format images for model input and perform inference to detect objects.
  ROS2 Integration:
    Subscribe to image topics and process them for object detection.
    Play and subscribe to rosbag recordings for sequential image processing.
