#!/usr/bin/env python3

import rospy
from duckietown_msgs.msg import FSMState

def fsm_callback(msg):
    print(f"FSM State: {msg.state} at {msg.header.stamp}")

if __name__ == '__main__':
    rospy.init_node('fsm_test')
    
    # Subscribe to FSM state
    rospy.Subscriber('/blueduckie/fsm_node/mode', FSMState, fsm_callback)
    
    print("Listening for FSM states... Press Ctrl+C to exit")
    rospy.spin()