#!/usr/bin/env python3

import rospy
from std_msgs.msg import String, Bool
from sensor_msgs.msg import Joy
from duckietown_msgs.msg import FSMState, BoolStamped
from geometry_msgs.msg import Twist

class FSMNode:
    def __init__(self):
        rospy.init_node('fsm_node', anonymous=False)
        
        # Get parameters
        self.vehicle_name = rospy.get_param('~veh', 'blueduckie')
        self.initial_state = rospy.get_param('~initial_state', 'LANE_FOLLOWING')
        
        # Current state
        self.current_state = self.initial_state
        self.joystick_override = False
        
        # Publishers
        self.pub_mode = rospy.Publisher('~mode', FSMState, queue_size=1, latch=True)
        
        # Subscribers
        self.sub_joy = rospy.Subscriber('joy', Joy, self.cb_joy, queue_size=1)
        self.sub_button = rospy.Subscriber('button_driver_node/event', BoolStamped, self.cb_button, queue_size=1)
        
        # State machine timer
        self.timer = rospy.Timer(rospy.Duration(0.1), self.update_state)
        
        # Publish initial state
        self.publish_state()
        
        rospy.loginfo(f"[{rospy.get_name()}] FSM Node initialized with state: {self.current_state}")

    def cb_joy(self, msg):
        """Handle joystick input for state transitions"""
        if len(msg.buttons) > 0:
            # Button 0 (usually 'A' or 'X') toggles between LANE_FOLLOWING and JOYSTICK_CONTROL
            if len(msg.buttons) > 0 and msg.buttons[0] == 1:
                if self.current_state == 'LANE_FOLLOWING':
                    self.transition_to('JOYSTICK_CONTROL')
                elif self.current_state == 'JOYSTICK_CONTROL':
                    self.transition_to('LANE_FOLLOWING')
            
            # Check if any joystick input is active (for override detection)
            axes_active = any(abs(axis) > 0.1 for axis in msg.axes)
            buttons_active = any(button == 1 for button in msg.buttons[1:])  # Skip button 0
            
            self.joystick_override = axes_active or buttons_active

    def cb_button(self, msg):
        """Handle physical button press for state transitions"""
        if msg.data:
            if self.current_state == 'LANE_FOLLOWING':
                self.transition_to('JOYSTICK_CONTROL')
            elif self.current_state == 'JOYSTICK_CONTROL':
                self.transition_to('LANE_FOLLOWING')

    def transition_to(self, new_state):
        """Transition to a new state"""
        if new_state != self.current_state:
            rospy.loginfo(f"[{rospy.get_name()}] State transition: {self.current_state} -> {new_state}")
            self.current_state = new_state
            self.publish_state()

    def update_state(self, event):
        """Periodic state update and validation"""
        # Auto-transition logic can go here
        # For now, just ensure we're publishing the current state
        self.publish_state()

    def publish_state(self):
        """Publish the current FSM state"""
        msg = FSMState()
        msg.header.stamp = rospy.Time.now()
        msg.state = self.current_state
        self.pub_mode.publish(msg)

if __name__ == '__main__':
    try:
        node = FSMNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass