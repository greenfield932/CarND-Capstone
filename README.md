# Programming a Real Self-Driving Car

This is the project repo for the final project of the Udacity Self-Driving Car Nanodegree: Programming a Real Self-Driving Car. 

[image1]: ./imgs/output_video.gif
[image2]: ./imgs/overview.png

# Result video

The following video contains testing of the project using simulator

![alt text][image1]

# Project overview

This project require to implement ROS nodes (partially) to drive self driving car. It includes implementation of waypoint updater, twist controller, traffic light detector and classifier. The project consists of the following architecture:

![alt text][image2]

## Waypoint updater

Waypoint updater gets information about vehicle pose and performs feeding waypoint follower with 200 points ahead using reference points from waypoint loader. It implements finite state machine for handling start/stop logic according to 
a nearest traffic light state. It also contains code for updating waypoints linear velocity and calculating optimal deceleration and acceleration speeds.

## Traffic light detector and classifier

Traffic light detector and classifier is implemented as a deep neural network for object detection. It performs object detection and classification at the same time.
For simulated traffic lights neural net was implemented using [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). [Here](https://github.com/EdjeElectronics/TensorFlow-Object-Detection-API-Tutorial-Train-Multiple-Objects-Windows-10) is a pretty good tutorial of how to do that.
Original network used for training is [faster_rcnn_inception_v2_coco](http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_v2_coco_2018_01_28.tar.gz). 
For real traffic light detection and classification [YOLO implementation](https://github.com/experiencor/keras-yolo3) was used.
Training images were extracted from simulator and ROSBAGs and manually annotated using [labelImg software](https://github.com/tzutalin/labelImg).
Images and annotations can be found [here](https://github.com/greenfield932/CarND-Capstone/tree/master/ros/src/tl_detector/images) and here[here](https://github.com/greenfield932/CarND-Capstone/tree/master/ros/src/tl_detector/images_real)
## DBW (drive-by-wire) node

This node gets twist data as input and generates brake/throttle/steering messages using PID controllers.

## Waypoint follower

Performs target velocities calculation based on pure-persuit algorithm to follow the trajectory.

## Waypoint loader

Performs loading reference trajectory points and generates linear and angular velocities for each point based on desired velocity.

# Installation
## Native Installation

* Be sure that your workstation is running Ubuntu 16.04 Xenial Xerus or Ubuntu 14.04 Trusty Tahir. [Ubuntu downloads can be found here](https://www.ubuntu.com/download/desktop).
* If using a Virtual Machine to install Ubuntu, use the following configuration as minimum:
  * 2 CPU
  * 2 GB system memory
  * 25 GB of free hard drive space

  The Udacity provided virtual machine has ROS and Dataspeed DBW already installed, so you can skip the next two steps if you are using this.

* Follow these instructions to install ROS
  * [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu) if you have Ubuntu 16.04.
  * [ROS Indigo](http://wiki.ros.org/indigo/Installation/Ubuntu) if you have Ubuntu 14.04.
* [Dataspeed DBW](https://bitbucket.org/DataspeedInc/dbw_mkz_ros)
  * Use this option to install the SDK on a workstation that already has ROS installed: [One Line SDK Install (binary)](https://bitbucket.org/DataspeedInc/dbw_mkz_ros/src/81e63fcc335d7b64139d7482017d6a97b405e250/ROS_SETUP.md?fileviewer=file-view-default)
* Download the [Udacity Simulator](https://github.com/udacity/CarND-Capstone/releases).

## Usage

1. Clone the project repository
```bash
git clone https://github.com/udacity/CarND-Capstone.git
```

2. Install python dependencies
```bash
cd CarND-Capstone
pip install -r requirements.txt
```
3. Make and run styx
```bash
cd ros
catkin_make
source devel/setup.sh
roslaunch launch/styx.launch
```
4. Run the simulator


# Contributors
| Name | E-mail | 
| ------ | ------ | 
| Dmitry Tyugin | dtyugin@gmail.com |