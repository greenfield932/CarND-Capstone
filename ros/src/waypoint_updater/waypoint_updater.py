#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import math
from scipy.spatial import KDTree
from std_msgs.msg import Int32
import numpy as np
'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_ACCEL = 10 #m/s^2

#State machine states for performing start/stop/accel/deccel logic depending on traffic light
DRIVE_STATE_BREAK = "STATE_BREAK"
DRIVE_STATE_BREAKING = "STATE_BREAKING"
DRIVE_STATE_ACCEL = "STATE_ACCEL"
DRIVE_STATE_ACCELERATING = "STATE_ACCELERATING"
DRIVE_STATE_DRIVING = "STATE_DRIVING"


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater', log_level=rospy.DEBUG)

        rospy.loginfo("WaypointUpdater init start")

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        
        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        
        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoint_tree = None
        self.accel_end_wp_idx = -1
        
        # TODO: Add other member variables you need below

        self.drive_state = DRIVE_STATE_BREAKING
        self.red_light = False
        self.closest_light_wp = -1
        self.max_vel = None
        self.loop()
    
    def loop(self):
        rate = rospy.Rate(50)
        while not rospy.is_shutdown():
            #rospy.loginfo("loop")
            if self.pose and self.base_waypoints:
                #rospy.loginfo("find waypoint")
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()

    #def out_of_change_speed(self):
    #    current_wp_idx = self.get_closest_waypoint_idx()
    #    if self.accel_start_wp_idx==-1 or self.accel_end_wp_idx == -1:
    #        return True
    #    return !(self.accel_start_wp_idx <= current_wp_idx and current_wp_idx <= self.accel_end_wp_idx)

    def out_of_accelerating(self):
        current_wp_idx = self.get_closest_waypoint_idx()
        return  self.accel_end_wp_idx < current_wp_idx

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
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
        #rospy.loginfo("found waypoint:"+str(closest_idx))

        return closest_idx

    def publish_waypoints(self, closest_idx):
        lane = Lane()
        lane.header = self.base_waypoints.header
        lane.waypoints = self.base_waypoints.waypoints[closest_idx:closest_idx + LOOKAHEAD_WPS]
        #rospy.loginfo("Closest idx: "+str(closest_idx))
        if len(lane.waypoints) < 10 and closest_idx > 10:
            for i in range(0, 10):
                lane.waypoints.append(self.base_waypoints.waypoints[i])

        #Check if we overlap with acceleration trajectory like we are on second lap
        #in this case we may stop because of previously set zero speed points
        #if 
        #for i in range(0, len(lane.waypoints)):
        #    if self.drive_state == DRIVE_STATE_DRIVING:
        #        self.set_waypoint_velocity(lane.waypoints, i, self.max_vel)

        #rospy.loginfo("Final waypoints count: "+str(len(lane.waypoints)))
        
        self.final_waypoints_pub.publish(lane)

    def pose_cb(self, msg):
        # TODO: Implement
        
        self.pose = msg
        current_wp_idx = self.get_closest_waypoint_idx()
        if current_wp_idx == 0:
            wp_count = len(self.base_waypoints.waypoints)
            self.set_waypoint_velocity(self.base_waypoints.waypoints, wp_count-1, self.max_vel)
        else:
            self.set_waypoint_velocity(self.base_waypoints.waypoints, current_wp_idx-1, self.max_vel)
        self.process_state_machine()
        pass

    def waypoints_cb(self, waypoints):
        # TODO: Implement

        if not self.waypoints_2d:
            self.waypoints_2d = []
            for waypoint in waypoints.waypoints:
                self.waypoints_2d.append([waypoint.pose.pose.position.x, waypoint.pose.pose.position.y])
            self.waypoint_tree = KDTree(self.waypoints_2d)

        self.base_waypoints = waypoints
        self.max_vel = 0.
        for wp in waypoints.waypoints:
            self.max_vel = max(self.max_vel, wp.twist.twist.linear.x)

        #find out max velocity from loaded waypoints
        rospy.loginfo("max vel:" + str(self.max_vel))

        #for i in range(0, len(waypoints.waypoints)):
        #    rospy.loginfo("Velocity for: " + str(i) + " is " +str(self.get_waypoint_velocity(self.base_waypoints.waypoints[i])))
        pass

    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        light_wp_idx = msg.data
        #rospy.loginfo("Traffic cb")
        if self.base_waypoints is not None and self.pose is not None:
            if light_wp_idx is not None and light_wp_idx!=-1:
                current_wp_idx = self.get_closest_waypoint_idx()
                if current_wp_idx is not None and current_wp_idx !=-1:
                    dist = self.distance(self.base_waypoints.waypoints, current_wp_idx, light_wp_idx)
                    if self.is_break_optimal(light_wp_idx) or dist < 20:
                        #rospy.loginfo("target light distance: "+str(dist))
                        #if dist <= self.get_optimal_break_distance():
                        #    rospy.loginfo("update traffic light to RED")
                        self.red_light = True
                        self.closest_light_wp = light_wp_idx
                    else:
                        self.red_light = False
                        rospy.loginfo("red light but too far to stop")
            else:
                self.red_light = False
                rospy.loginfo("update traffic light to GREEN")

            self.process_state_machine()

    #check if we should start braking depending on desired deceleration and distance to target stop point
    def is_break_optimal(self, target_idx):
        current_wp_idx = self.get_closest_waypoint_idx()
        current_vel = self.get_waypoint_velocity_by_idx(current_wp_idx)
        dist = self.distance(self.base_waypoints.waypoints, current_wp_idx, target_idx)
        #rospy.loginfo("dist to target: "+str(dist))
        if dist < 5 and current_vel < 0.1:
            return True

        dt = dist/current_vel
        if dt < 0.01:
            return True

        accel = current_vel/dt
        accel_diff = MAX_ACCEL - accel
        #rospy.loginfo("accel to target: "+str(accel))
        if accel >= MAX_ACCEL/2.:
        #    rospy.loginfo("Brake is optimal, start braking")
            return True

        return False

    #update waypoints for acceleration from current position to target velocity
    def accelerate(self):
        rospy.loginfo("accelerate")
        current_wp_idx = self.get_closest_waypoint_idx()
        current_vel = self.get_waypoint_velocity(self.base_waypoints.waypoints[current_wp_idx])
        self.accel_start_wp_idx = current_wp_idx
        next_vel = current_vel
        if next_vel < 1:
            next_vel = 1
        waypoints_count = len(self.base_waypoints.waypoints)
        for idx in range(current_wp_idx, waypoints_count):
            dist = self.distance(self.base_waypoints.waypoints, idx -1, idx)
            next_vel = next_vel + MAX_ACCEL * dist / next_vel
            if next_vel >=  self.max_vel:
                next_vel = self.max_vel
                #rospy.loginfo("max speed reached at idx:"+str(idx))
            #else:
            #rospy.loginfo("set vel to idx:"+str(idx) + " vel:" + str(next_vel))

            self.set_waypoint_velocity(self.base_waypoints.waypoints, idx, next_vel)
        for idx in range(0, current_wp_idx-1):
            self.set_waypoint_velocity(self.base_waypoints.waypoints, idx, self.max_vel)

        #TODO don't care about second lap for now
        #for idx in range(0, current_wp_idx - 1):
        #    dist = distance(self.base_waypoints.waypoints, idx -1, idx)
        #    next_vel = prev_vel + max_accel * dist / prev_vel
        #    self.set_waypoint_velocity(self.base_waypoints.waypoints, idx, next_vel)

    #update waypoints for deceleration from current position to target waypoint
    def decelerate(self, target_wp_idx):
        rospy.loginfo("decelerate")
        current_wp_idx = self.get_closest_waypoint_idx()
        waypoints_count = abs(target_wp_idx - current_wp_idx)
        if waypoints_count <=2:
            self.set_waypoint_velocity(self.base_waypoints.waypoints, current_wp_idx, 0.)
        else:
            current_vel = self.get_waypoint_velocity(self.base_waypoints.waypoints[current_wp_idx])
            target_vel = current_vel
            deccel_step = current_vel/waypoints_count
            #rospy.loginfo("deccel step: "+str(deccel_step))
            #rospy.loginfo("current_vel: "+str(current_vel))
            #rospy.loginfo("waypoints_count: "+str(waypoints_count))
            
            for idx in range(current_wp_idx, target_wp_idx+1):
                i = idx - current_wp_idx
                target_vel = current_vel - deccel_step * i
                #if target_vel < 3:
                #    target_vel = 3
                if idx >= target_wp_idx - 4:
                    target_vel = 0
                self.set_waypoint_velocity(self.base_waypoints.waypoints, idx, target_vel)

#            for idx in range(current_wp_idx, target_wp_idx+1):
#                rospy.loginfo("set vel to idx:"+str(idx) + " vel:" + str(self.get_waypoint_velocity(self.base_waypoints.waypoints[idx])))


    #state machine for start/stop logic
    def process_state_machine(self):
        prev_state = self.drive_state
        if self.drive_state == DRIVE_STATE_ACCEL:
            self.accelerate()
            #self.drive_state = DRIVE_STATE_ACCELERATING
            self.drive_state = DRIVE_STATE_DRIVING


        #elif self.drive_state == DRIVE_STATE_ACCELERATING and self.out_of_accelerating():
        #    self.drive_state = DRIVE_STATE_DRIVING
        #    self.accel_end_wp_idx = -1
        elif self.drive_state == DRIVE_STATE_BREAK:
            self.decelerate(self.closest_light_wp)
            self.drive_state = DRIVE_STATE_BREAKING

        elif self.drive_state == DRIVE_STATE_BREAKING and self.red_light == False:
            self.drive_state = DRIVE_STATE_ACCEL

        #elif (self.drive_state == DRIVE_STATE_DRIVING or self.drive_state == DRIVE_STATE_ACCELERATING) and self.red_light == True:
        elif self.drive_state == DRIVE_STATE_DRIVING and self.red_light == True:
            self.drive_state = DRIVE_STATE_BREAK

        if prev_state!=self.drive_state:
            rospy.loginfo("state machine in: "+prev_state + " out: " + self.drive_state)

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x
    
    def get_waypoint_velocity_by_idx(self, idx):
        return self.base_waypoints.waypoints[idx].twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint_idx, velocity):
        waypoints[waypoint_idx].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
