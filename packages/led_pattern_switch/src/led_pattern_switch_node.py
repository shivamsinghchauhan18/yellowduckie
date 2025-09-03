#!/usr/bin/env python3
import rospy
import time
from duckietown_msgs.msg import FSMState
from duckietown_msgs.srv import ChangePattern
from std_msgs.msg import String


class LEDPatternSwitchNode:
    def __init__(self):
        self.node_name = rospy.get_name()
        # Read parameters
        self.mappings = rospy.get_param("~mappings")
        source_topic_dict = rospy.get_param("~source_topics")
        self.current_src_name = "joystick"  # default if FSM is missing

        # --- log throttling state ---
        self._last_pattern = None
        self._last_fsm_state = None
        self._last_log_ts = 0.0
        self._log_throttle_sec = rospy.get_param("~log_throttle_sec", 2.0)

        # Service proxy to LED emitter (scoped to this node's namespace)
        self.changePattern = rospy.ServiceProxy("~set_pattern", ChangePattern)

        # Subscribers
        self.sub_fsm_state = rospy.Subscriber(
            rospy.get_param("~mode_topic"),
            FSMState,
            self.cbFSMState,
        )

        self.sub_dict = {}
        for src_name, topic_name in list(source_topic_dict.items()):
            self.sub_dict[src_name] = rospy.Subscriber(
                topic_name,
                String,
                self.msgincb,
                callback_args=src_name,
            )

        rospy.loginfo(f"[{self.node_name}] Initialized.")

    def cbFSMState(self, fsm_state_msg: FSMState):
        """
        Switch LED pattern source based on FSM state. Log only when pattern/state
        actually changes or after a throttle window to avoid log spam.
        """
        self.current_src_name = self.mappings.get(fsm_state_msg.state)
        if self.current_src_name is None:
            rospy.logwarn(
                f"[{self.node_name}] FSMState {fsm_state_msg.state} not handled. "
                f"No message will pass through the switch."
            )
            return

        now = time.monotonic()
        changed = (
            self.current_src_name != self._last_pattern
            or fsm_state_msg.state != self._last_fsm_state
        )
        throttled = (now - self._last_log_ts) >= self._log_throttle_sec

        if changed or throttled:
            rospy.loginfo(
                f"[{self.node_name}] LED pattern switched to {self.current_src_name} "
                f"in state {fsm_state_msg.state}."
            )
            self._last_pattern = self.current_src_name
            self._last_fsm_state = fsm_state_msg.state
            self._last_log_ts = now

    def msgincb(self, msg: String, src_name: str):
        """
        Forward the selected source's message to the LED changePattern service.
        """
        if src_name == self.current_src_name:
            # Only forward when this source is the active selection
            self.changePattern(msg)

    def on_shutdown(self):
        rospy.loginfo(f"[{self.node_name}] Shutting down.")


if __name__ == "__main__":
    rospy.init_node("LED_pattern_switch_node", anonymous=False)
    node = LEDPatternSwitchNode()
    rospy.on_shutdown(node.on_shutdown)
    rospy.spin()