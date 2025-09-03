#!/usr/bin/env python3

import rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, BoolStamped, LanePose
from std_msgs.msg import Header, Int8, Empty, Bool

from lane_controller.controller import LaneController

from duckietown_msgs.msg import AprilTagDetectionArray, AprilTagDetection


class LaneControllerNode(DTROS):
    """Computes control action.
    The node compute the commands in form of linear and angular velocities, by processing the estimate error in
    lateral deviationa and heading.
    The configuration parameters can be changed dynamically while the node is running via ``rosparam set`` commands.
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node that ROS will use
    Configuration:
        ~v_bar (:obj:`float`): Nominal velocity in m/s
        ~k_d (:obj:`float`): Proportional term for lateral deviation
        ~k_theta (:obj:`float`): Proportional term for heading deviation
        ~k_Id (:obj:`float`): integral term for lateral deviation
        ~k_Iphi (:obj:`float`): integral term for lateral deviation
        ~d_thres (:obj:`float`): Maximum value for lateral error
        ~theta_thres (:obj:`float`): Maximum value for heading error
        ~d_offset (:obj:`float`): Goal offset from center of the lane
        ~integral_bounds (:obj:`dict`): Bounds for integral term
        ~d_resolution (:obj:`float`): Resolution of lateral position estimate
        ~phi_resolution (:obj:`float`): Resolution of heading estimate
        ~omega_ff (:obj:`float`): Feedforward part of controller
        ~verbose (:obj:`bool`): Verbosity level (0,1,2)
        ~stop_line_slowdown (:obj:`dict`): Start and end distances for slowdown at stop lines

    Publisher:
        ~car_cmd (:obj:`Twist2DStamped`): The computed control action
    Subscribers:
        ~lane_pose (:obj:`LanePose`): The lane pose estimate from the lane filter
        ~intersection_navigation_pose (:obj:`LanePose`): The lane pose estimate from intersection navigation
        ~wheels_cmd_executed (:obj:`WheelsCmdStamped`): Confirmation that the control action was executed
        ~stop_line_reading (:obj:`StopLineReading`): Distance from stopline, to reduce speed
        ~obstacle_distance_reading (:obj:`stop_line_reading`): Distancefrom obstacle virtual stopline, to reduce speed
    """

    def __init__(self, node_name):

        # Initialize the DTROS parent class
        super(LaneControllerNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        # Add the node parameters to the parameters dictionary
        self.params = dict()
        self.params["~v_bar"] = DTParam("~v_bar", param_type=ParamType.FLOAT, min_value=0.0, max_value=1.0)
        self.params["~k_d"] = DTParam("~k_d", param_type=ParamType.FLOAT, min_value=-10.0, max_value=10.0)
        self.params["~k_theta"] = DTParam("~k_theta", param_type=ParamType.FLOAT, min_value=-10.0, max_value=10.0)
        self.params["~k_Id"] = DTParam("~k_Id", param_type=ParamType.FLOAT, min_value=-10.0, max_value=10.0)
        self.params["~k_Iphi"] = DTParam("~k_Iphi", param_type=ParamType.FLOAT, min_value=-10.0, max_value=10.0)
        self.params["~d_thres"] = DTParam("~d_thres", param_type=ParamType.FLOAT, min_value=0.0, max_value=2.0)
        self.params["~theta_thres"] = DTParam("~theta_thres", param_type=ParamType.FLOAT, min_value=0.0, max_value=np.pi/2)
        self.params["~d_offset"] = DTParam("~d_offset", param_type=ParamType.FLOAT, min_value=-1.0, max_value=1.0)
        self.params["~integral_bounds"] = DTParam("~integral_bounds", param_type=ParamType.DICT)
        self.params["~d_resolution"] = DTParam("~d_resolution", param_type=ParamType.FLOAT, min_value=0.0, max_value=1.0)
        self.params["~phi_resolution"] = DTParam("~phi_resolution", param_type=ParamType.FLOAT, min_value=0.0, max_value=np.pi/2)
        self.params["~omega_ff"] = DTParam("~omega_ff", param_type=ParamType.FLOAT, min_value=0.0, max_value=10.0)
        self.params["~verbose"] = DTParam("~verbose", param_type=ParamType.BOOL)
        self.params["~stop_line_slowdown"] = DTParam("~stop_line_slowdown", param_type=ParamType.DICT)
        self.params["~use_LEDs"] = DTParam("~use_LEDs", param_type=ParamType.BOOL)

        # default integral control bounds
        self.params["~integral_bounds"].value = dict()
        self.params["~integral_bounds"].value["top"] = 10.0
        self.params["~integral_bounds"].value["bottom"] = -10.0

        # default stop line slowdown parameters
        self.params["~stop_line_slowdown"].value = dict()
        self.params["~stop_line_slowdown"].value["start"] = 0.6
        self.params["~stop_line_slowdown"].value["end"] = 0.2

        # initialize controller
        self.controller = LaneController(self.params)
        self.controller.verbose = self.params["~verbose"].value

        # initialize variables
        self.last_omega = 0.0
        self.last_s = None
        self.stop_line_distance = -1
        self.at_stop_line = False
        self.obstacle_stop_line_distance = -1
        self.obstacle_stop_line_detected = False
        self.at_obstacle_stop_line = False
        self.stop_line_slowdown_active = False
        self.april_tags_active = False
        self.stop_line_car_cmd = Twist2DStamped()
        self.stop_line_car_cmd.v = 0.0
        self.stop_line_car_cmd.omega = 0.0
        self.pose_msg = None
        self.intersection_navigation_pose_msg = None
        self.intersection_navigation_active = False
        self.intersection_navigation_pose_msg = LanePose()
        self.intersection_navigation_pose_msg.header.stamp = rospy.Time.now()
        self.intersection_navigation_pose_msg.header.frame_id = "lane_filter"
        self.intersection_navigation_pose_msg.d = 0.0
        self.intersection_navigation_pose_msg.phi = 0.0
        self.intersection_navigation_pose_msg.d_ref = 0.0
        self.intersection_navigation_pose_msg.phi_ref = 0.0
        self.intersection_navigation_pose_msg.d_phi_covariance = [0.0, 0.0, 0.0, 0.0]
        self.intersection_navigation_pose_msg.curvature = 0.0
        self.intersection_navigation_pose_msg.curvature_ref = 0.0
        self.intersection_navigation_pose_msg.v_ref = 0.0
        self.intersection_navigation_pose_msg.status = 0
        self.intersection_navigation_pose_msg.in_lane = True
        self.intersection_navigation_pose_msg = None
        self.intersection_navigation_active = False
        self.intersection_navigation_pose_msg = LanePose()
        self.intersection_navigation_pose_msg.header.stamp = rospy.Time.now()
        self.intersection_navigation_pose_msg.header.frame_id = "lane_filter"
        self.intersection_navigation_pose_msg.d = 0.0
        self.intersection_navigation_pose_msg.phi = 0.0
        self.intersection_navigation_pose_msg.d_ref = 0.0
        self.intersection_navigation_pose_msg.phi_ref = 0.0
        self.intersection_navigation_pose_msg.d_phi_covariance = [0.0, 0.0, 0.0, 0.0]
        self.intersection_navigation_pose_msg.curvature = 0.0
        self.intersection_navigation_pose_msg.curvature_ref = 0.0
        self.intersection_navigation_pose_msg.v_ref = 0.0
        self.intersection_navigation_pose_msg.status = 0
        self.intersection_navigation_pose_msg.in_lane = True
        self.wheels_cmd_executed = WheelsCmdStamped()
        self.wheels_cmd_executed.vel_left = 0.0
        self.wheels_cmd_executed.vel_right = 0.0
        self.drive_running = False

        # initialize times
        self.t_delta_s = 0.0
        self.t_prev_s = rospy.Time.now().to_sec()

        # subscribers
        self.sub_lane_reading = rospy.Subscriber("~lane_pose", LanePose, self.cb_pose, queue_size=1)
        self.sub_intersection_navigation_pose = rospy.Subscriber("~intersection_navigation_pose", LanePose, self.cb_intersection_navigation_pose, queue_size=1)
        self.sub_wheels_cmd_executed = rospy.Subscriber("~wheels_cmd_executed", WheelsCmdStamped, self.cb_wheels_cmd_executed, queue_size=1)
        self.sub_stop_line_reading = rospy.Subscriber("~stop_line_reading", StopLineReading, self.cb_stop_line_reading, queue_size=1)
        self.sub_obstacle_distance_reading = rospy.Subscriber("~obstacle_distance_reading", StopLineReading, self.cb_obstacle_distance_reading, queue_size=1)
        self.sub_vehicle_centers = rospy.Subscriber("~vehicle_centers", VehicleCorners, self.cb_vehicle_centers, queue_size=1)
        self.sub_apriltag_detections = rospy.Subscriber("~apriltag_detections", AprilTagDetectionArray, self.cb_apriltag_detections, queue_size=1)
        self.sub_turn_type = rospy.Subscriber("~turn_type", Int8, self.cb_turn_type, queue_size=1)
        self.sub_intersection_go = rospy.Subscriber("~intersection_go", BoolStamped, self.cb_intersection_go, queue_size=1)
        self.sub_vehicle_detected = rospy.Subscriber("~vehicle_detected", BoolStamped, self.cb_vehicle_detected, queue_size=1)
        self.sub_obstacle_detected = rospy.Subscriber("~obstacle_detected", BoolStamped, self.cb_obstacle_detected, queue_size=1)

        # publishers
        self.pub_car_cmd = rospy.Publisher("~car_cmd", Twist2DStamped, queue_size=1)

        # services
        if self.params["~use_LEDs"].value:
            rospy.loginfo(f"[{self.node_name}] Waiting for LED emitter service...")
            rospy.wait_for_service("~set_pattern")
            self.led_svc = rospy.ServiceProxy("~set_pattern", ChangePattern)

        self.loginfo("Initialized")

    def cb_pose(self, pose_msg):
        self.pose_msg = pose_msg

    def cb_intersection_navigation_pose(self, pose_msg):
        self.intersection_navigation_pose_msg = pose_msg

    def cb_wheels_cmd_executed(self, msg):
        self.wheels_cmd_executed = msg

    def cb_stop_line_reading(self, msg):
        self.stop_line_distance = msg.stop_line_point.x
        self.at_stop_line = msg.at_stop_line
        self.stop_line_detected = msg.stop_line_detected

    def cb_obstacle_distance_reading(self, msg):
        self.obstacle_stop_line_distance = msg.stop_line_point.x
        self.at_obstacle_stop_line = msg.at_stop_line
        self.obstacle_stop_line_detected = msg.stop_line_detected

    def cb_vehicle_centers(self, msg):
        self.vehicle_centers = msg.centers
        self.vehicle_detected = len(self.vehicle_centers) > 0

    def cb_apriltag_detections(self, msg):
        self.april_tags_active = len(msg.detections) > 0
        self.april_tags = msg.detections

    def cb_turn_type(self, msg):
        self.turn_type = msg.data

    def cb_intersection_go(self, msg):
        self.intersection_go = msg.data

    def cb_vehicle_detected(self, msg):
        self.vehicle_detected = msg.data

    def cb_obstacle_detected(self, msg):
        self.obstacle_detected = msg.data

    def publishCmd(self, car_cmd_msg):
        self.pub_car_cmd.publish(car_cmd_msg)

    def drive(self):
        # Get current time
        current_s = rospy.Time.now().to_sec()
        # Get elapsed time
        self.t_delta_s = current_s - self.t_prev_s
        # Save time
        self.t_prev_s = current_s

        # Get pose msg
        pose_msg = self.pose_msg

        # Check if we have a pose msg
        if pose_msg is None:
            return

        # Check if we are in intersection navigation mode
        if self.intersection_navigation_pose_msg is not None:
            pose_msg = self.intersection_navigation_pose_msg
            self.intersection_navigation_active = True
        else:
            self.intersection_navigation_active = False

        # Check if we are at a stop line
        if self.stop_line_distance > 0:
            self.stop_line_slowdown_active = True
        else:
            self.stop_line_slowdown_active = False

        # Check if we are at an obstacle stop line
        if self.obstacle_stop_line_distance > 0:
            self.obstacle_stop_line_detected = True
        else:
            self.obstacle_stop_line_detected = False

        # Check if we are at a vehicle stop line
        if self.vehicle_detected:
            self.vehicle_stop_line_detected = True
        else:
            self.vehicle_stop_line_detected = False

        # Check if we are at an obstacle stop line
        if self.obstacle_detected:
            self.obstacle_stop_line_detected = True
        else:
            self.obstacle_stop_line_detected = False

        # Check if we are at a stop line
        if self.at_stop_line:
            # Stop the car
            car_stop_msg = Twist2DStamped()
            car_stop_msg.v = 0
            car_stop_msg.omega = 0

            # Stop turn
            self.publishCmd(car_stop_msg)
            rospy.loginfo("    Stopping turn")

            # Change the LED back to the driving state
            if self.params["~use_LEDs"].value:
                rospy.loginfo(f"    Changing LEDS back to driving mode")
                if self.led_svc is not None:
                    msg = ChangePatternRequest(String("CAR_DRIVING"))
                    try:
                        resp = self.led_svc(msg)
                    except rospy.ServiceException as e:
                        rospy.logwarn(f"could not set LEDs: {e}")

        if not self.at_obstacle_stop_line:  # Lane following
            # Compute errors
            d_err = pose_msg.d - self.params["~d_offset"].value
            phi_err = pose_msg.phi

            # We cap the error if it grows too large
            if np.abs(d_err) > self.params["~d_thres"].value:
                self.log("d_err too large, thresholding it!", "error")
                d_err = np.sign(d_err) * self.params["~d_thres"].value

            wheels_cmd_exec = [self.wheels_cmd_executed.vel_left, self.wheels_cmd_executed.vel_right]
            if self.obstacle_stop_line_detected:
                v, omega = self.controller.compute_control_action(
                    d_err, phi_err, self.t_delta_s, wheels_cmd_exec,
                    self.obstacle_stop_line_distance, pose_msg
                )
                # TODO: This is a temporarily fix to avoid vehicle image
                # detection latency caused unable to stop in time.
                v = v * 0.25
                omega = omega * 0.25

            else:
                v, omega = self.controller.compute_control_action(
                    d_err, phi_err, self.t_delta_s, wheels_cmd_exec,
                    self.stop_line_distance, pose_msg
                )

            # Initialize car control msg, add header from input message
            car_control_msg = Twist2DStamped()
            car_control_msg.header = pose_msg.header

            # Add commands to car message
            car_control_msg.v = v
            car_control_msg.omega = omega

            self.publishCmd(car_control_msg)

        # Set the current time stamp, needed for lane following
        # Important: this needs to be set whether we're doing lane following or
        # intersection navigation, otherwise when we go back to lane following
        # from intersection navigation the first step of lane following will
        # break
        self.last_s = current_s
        self.drive_running = False


if __name__ == "__main__":
    # Initialize the node
    rospy.sleep(5)
    node = LaneControllerNode(node_name="lane_controller_node")
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        node.drive()
        rate.sleep()