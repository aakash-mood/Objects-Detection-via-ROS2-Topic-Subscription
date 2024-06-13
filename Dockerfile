FROM osrf/ros:humble-desktop-full

RUN apt-get update && apt-get install -y git wget python3-pip vim
RUN pip3 install setuptools==58.2.0
RUN pip3 install torch torchvision

COPY ros2_ws/ /root/ros2_ws/
COPY images/ /root/images/
COPY recordings/ /root/recordings/

ENV DISPLAY=novnc:0.0
