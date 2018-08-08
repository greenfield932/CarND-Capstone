from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, *args, **kwargs):
        # TODO: Implement
        self.vehicle_mass = args[0]
        self.fuel_capacity = args[1]
        self.brake_deadband = args[2]
        self.decel_limit = args[3]
        self.accel_limit = args[4]
        self.wheel_radius = args[5]
        self.wheel_base = args[6]
        self.steer_ratio = args[7]
        self.max_lat_accel = args[8]
        self.max_steer_angle = args[9]
        
        self.yaw_ctrl = YawController(self.wheel_base, self.steer_ratio, 0.1, self.max_lat_accel, self.max_steer_angle)
        self.throttle_ctrl = PID(kp = 0.3, ki = 0.1, kd = 0., mn = 0., mx = 0.2)
        self.vel_lpf = LowPassFilter(tau = 0.5, ts = 0.02)
        self.last_time = rospy.get_time()
        
        pass

    def control(self, *args, **kwargs):
        # TODO: Change the arg, kwarg list to suit your needs
        # Return throttle, brake, steer
        linear_vel = args[0]
        angular_vel = args[1]
        current_vel = args[2]
        dbw_enabled = args[3]
        if not dbw_enabled:
            self.throttle_ctrl.reset()
            return 0.,0., 0.
        current_vel = self.vel_lpf.filt(current_vel)
        steering = self.yaw_ctrl.get_steering(linear_vel, angular_vel, current_vel)
        vel_error = linear_vel - current_vel
        current_time = rospy.get_time()
        sample_time = current_time - self.last_time
        self.last_time = current_time
        throttle = self.throttle_ctrl.step(vel_error, sample_time)
        brake = 0
        
        if linear_vel == 0. and current_vel < 0.1:
            throttle = 0
            brake = 400
        elif throttle < .1 and vel_error < 0:
            throttle = 0
            decel = max(vel_error, self.decel_limit)
            brake = abs(decel)*self.vehicle_mass*self.wheel_radius
        return throttle, brake, steering