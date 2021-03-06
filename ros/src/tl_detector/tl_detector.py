#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from std_msgs.msg import Bool
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from scipy.spatial import KDTree
import numpy as np

STATE_COUNT_THRESHOLD = 3
DETECT_RATE_DELAY = 0.1 #each seconds detect
class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.recording = False
        self.show_detection = True
        self.debug_detection = False
        self.detection_timestamp = rospy.get_time()
        
        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.is_site = self.config['is_site']
        #self.show_detection = True

        if self.show_detection:
            self.detection_result_pub = rospy.Publisher('/traffic_detection2', Image, queue_size=1)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        
        
        self.light_classifier = TLClassifier(self.is_site)
        self.listener = tf.TransformListener()

        self.state = TrafficLight.RED
        self.last_state = TrafficLight.RED
        self.light_detected_state = TrafficLight.RED
        self.last_wp = -1
        self.state_count = STATE_COUNT_THRESHOLD
        self.img_cnt = 0
        self.img_written_cnt = 0
        
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb, queue_size=1)
        
        sub9 = rospy.Subscriber('/record_imgs', Bool, self.record_cb)

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def record_cb(self, msg):
        self.recording = msg.data

    def waypoints_cb(self, waypoints):
        if not self.waypoints_2d:
            self.waypoints_2d = []
            for waypoint in waypoints.waypoints:
                self.waypoints_2d.append([waypoint.pose.pose.position.x, waypoint.pose.pose.position.y])
            self.waypoint_tree = KDTree(self.waypoints_2d)

            self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    
    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        #code for image collection from simulator, can be controlled outside by 
        #for single image: rostopic pub /record_imgs std_msgs/Bool True 
        #for multiple shots 3 times per sec: rostopic -f 3 pub /record_imgs std_msgs/Bool True
        
        if self.recording == True:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv2.imwrite("./images_new1/"+str(self.img_written_cnt)+".png", cv_image)
            self.img_written_cnt+=1

        #rosbag debug
        if self.debug_detection == True:
            state = self.get_light_state(None)

        self.has_image = True
        self.camera_image = msg
        
        if self.waypoint_tree is None:
            return
        
        light_wp, state = self.process_traffic_lights()
        if(light_wp == -2):
            return
        
        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if (state == TrafficLight.RED or state == TrafficLight.YELLOW) else -1
            self.last_wp = light_wp
            rospy.loginfo("TL: Update traffic light with state:"+self.light_state_name(state) +  " and wp: "+str(light_wp))
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
            rospy.loginfo("TL: Update traffic light with state:"+self.light_state_name(state) +  " and lastwp: "+str(self.last_wp))

        self.state_count += 1

    def get_closest_waypoint(self, x, y):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        closest_idx = self.waypoint_tree.query([x,y], 1)[1]

        #check is found point ahead or behind the vehicle
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]
        
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x,y])
        
        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)
        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)

        #TODO implement
        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        
        """
        #if(not self.has_image):
        #    self.prev_light_loc = None
        #    return False

        
        if self.detection_timestamp and rospy.get_time() - self.detection_timestamp >=DETECT_RATE_DELAY and self.camera_image:
            cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")
            self.light_detected_state, out_image = self.light_classifier.get_classification(cv_image)
            rospy.loginfo("TL: Detector got state:"+self.light_state_name(self.light_detected_state))

            if self.show_detection:
                if self.light_detected_state!=TrafficLight.UNKNOWN:
                    cv_image = self.bridge.cv2_to_imgmsg(out_image, "bgr8")
                else:
                    cv_image = self.camera_image
                self.detection_result_pub.publish(cv_image)
            self.detection_timestamp = rospy.get_time()
            
        #Get classification
        #return self.light_classifier.get_classification(cv_image)
        
        return self.light_detected_state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        closest_light = None
        line_wp_idx = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose and self.lights is not None):
            car_wp_idx = self.get_closest_waypoint(self.pose.pose.position.x, self.pose.pose.position.y)
            diff = len(self.waypoints.waypoints)
            for i, light in enumerate(self.lights):
                line = stop_line_positions[i]
                temp_wp_idx = self.get_closest_waypoint(line[0], line[1])
                d = temp_wp_idx - car_wp_idx
                if d>=0 and d<diff:
                    diff = d
                    closest_light = light
                    line_wp_idx = temp_wp_idx

            #TODO find the closest visible traffic light (if one exists)

            if closest_light:
                state = self.get_light_state(closest_light)
                rospy.loginfo("TL: Light detected:"+self.light_state_name(state) + " at wp:" + str(line_wp_idx))
                return line_wp_idx, state

            #self.waypoints = None
            rospy.loginfo("TL: No light WP")
            return -1, TrafficLight.UNKNOWN

        if(not self.pose):
            rospy.loginfo("TL: No pose in tl detector yet")
        elif(self.lights == None):
            rospy.loginfo("TL: No traffic lights loaded in tl detector yet")
            
        return -2, TrafficLight.UNKNOWN

    def light_state_name(self, state):
        if state == TrafficLight.RED:
            return "red"
        elif state == TrafficLight.GREEN:
            return "green"
        elif state == TrafficLight.YELLOW:
            return "yellow"
        return "unknown"

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
