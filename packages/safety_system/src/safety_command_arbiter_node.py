#!/usr/bin/env python3

import rospy
import numpy as np
from threading import Lock
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, BoolStamped
from std_msgs.msg import String, Header


class SafetyCommandArbiterNode(DTROS):
    """
    Safety Command Arbiter Node for Duckietown Safety System
    
    Validates and arbitrates between different command sources, ensuring all commands
    meet safety constraints. Provides priority-based command selection and safety overrides.
    
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node
    
    Configuration:
        ~max_linear_velocity (:obj:`float`): Maximum allowed linear velocity in m/s
        ~max_angular_velocity (:obj:`float`): Maximum allowed angular velocity in rad/s
        ~safety_override_priority (:obj:`int`): Priority level for safety overrides
        ~command_timeout (:obj:`float`): Timeout for command validity in seconds
        ~enable_safety_limits (:obj:`bool`): Enable safety constraint enforcement
    
    Subscribers:
        ~lane_controller_cmd (:obj:`Twist2DStamped`): Commands from lane controller
        ~navigation_cmd (:obj:`Twist2DStamped`): Commands from navigation system
        ~safety_override (:obj:`Twist2DStamped`): Emergency safety override commands
        ~safety_status (:obj:`String`): Current safety system status
        ~collision_risk (:obj:`String`): Current collision risk level
    
    Publishers:
        ~car_cmd (:obj:`Twist2DStamped`): Final arbitrated car command
        ~command_status (:obj:`String`): Status of command arbitration
        ~safety_violations (:obj:`String`): Safety constraint violations
    """
    
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(SafetyCommandArbiterNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )
        
        # Parameters
        self.max_linear_velocity = DTParam(
            "~max_linear_velocity",
            param_type=ParamType.FLOAT,
            default=0.5,
            min_value=0.1,
            max_value=2.0
        )
        
        self.max_angular_velocity = DTParam(
            "~max_angular_velocity",
            param_type=ParamType.FLOAT,
            default=2.0,
            min_value=0.5,
            max_value=5.0
        )
        
        self.safety_override_priority = DTParam(
            "~safety_override_priority",
            param_type=ParamType.INT,
            default=10,
            min_value=1,
            max_value=10
        )
        
        self.command_timeout = DTParam(
            "~command_timeout",
            param_type=ParamType.FLOAT,
            default=0.5,
            min_value=0.1,
            max_value=2.0
        )
        
        self.enable_safety_limits = DTParam(
            "~enable_safety_limits",
            param_type=ParamType.BOOL,
            default=True
        )
        
        # Command sources with priorities (higher number = higher priority)
        self.command_sources = {
            "lane_controller": {"priority": 1, "cmd": None, "timestamp": None},
            "navigation": {"priority": 2, "cmd": None, "timestamp": None},
            "safety_override": {"priority": 10, "cmd": None, "timestamp": None}
        }
        
        # Safety state
        self.current_safety_status = "SAFE"
        self.current_collision_risk = "NONE"
        self.safety_constraints_active = False
        
        # Thread safety
        self.state_lock = Lock()
        
        # Command validation statistics
        self.command_stats = {
            "total_commands": 0,
            "safety_violations": 0,
            "overrides_applied": 0,
            "commands_blocked": 0
        }
        
        # Publishers
        self.pub_car_cmd = rospy.Publisher(
            "~car_cmd",
            Twist2DStamped,
            queue_size=1,
            dt_topic_type=TopicType.CONTROL
        )
        
        self.pub_command_status = rospy.Publisher(
            "~command_status",
            String,
            queue_size=1
        )
        
        self.pub_safety_violations = rospy.Publisher(
            "~safety_violations",
            String,
            queue_size=1
        )
        
        # Subscribers
        self.sub_lane_controller_cmd = rospy.Subscriber(
            "~lane_controller_cmd",
            Twist2DStamped,
            self.cb_lane_controller_cmd,
            queue_size=1
        )
        
        self.sub_navigation_cmd = rospy.Subscriber(
            "~navigation_cmd",
            Twist2DStamped,
            self.cb_navigation_cmd,
            queue_size=1
        )
        
        self.sub_safety_override = rospy.Subscriber(
            "~safety_override",
            Twist2DStamped,
            self.cb_safety_override,
            queue_size=1
        )
        
        self.sub_safety_status = rospy.Subscriber(
            "~safety_status",
            String,
            self.cb_safety_status,
            queue_size=1
        )
        
        self.sub_collision_risk = rospy.Subscriber(
            "~collision_risk",
            String,
            self.cb_collision_risk,
            queue_size=1
        )
        
        # Start command arbitration timer
        self.timer = rospy.Timer(
            rospy.Duration(0.05),  # 20 Hz arbitration rate
            self.arbitrate_commands
        )
        
        self.log("Safety Command Arbiter initialized")
    
    def cb_lane_controller_cmd(self, msg):
        """
        Callback for lane controller commands
        
        Args:
            msg (:obj:`Twist2DStamped`): Lane controller command
        """
        with self.state_lock:
            self.command_sources["lane_controller"]["cmd"] = msg
            self.command_sources["lane_controller"]["timestamp"] = rospy.Time.now()
    
    def cb_navigation_cmd(self, msg):
        """
        Callback for navigation system commands
        
        Args:
            msg (:obj:`Twist2DStamped`): Navigation command
        """
        with self.state_lock:
            self.command_sources["navigation"]["cmd"] = msg
            self.command_sources["navigation"]["timestamp"] = rospy.Time.now()
    
    def cb_safety_override(self, msg):
        """
        Callback for safety override commands
        
        Args:
            msg (:obj:`Twist2DStamped`): Safety override command
        """
        with self.state_lock:
            self.command_sources["safety_override"]["cmd"] = msg
            self.command_sources["safety_override"]["timestamp"] = rospy.Time.now()
            
            # Log safety override activation
            self.log(f"Safety override activated: v={msg.v:.3f}, omega={msg.omega:.3f}", "warn")
            self.command_stats["overrides_applied"] += 1
    
    def cb_safety_status(self, msg):
        """
        Callback for safety system status
        
        Args:
            msg (:obj:`String`): Safety status message
        """
        with self.state_lock:
            # Parse safety status (format: SAFETY_LEVEL:LEVEL:...)
            try:
                if "SAFETY_LEVEL:" in msg.data:
                    parts = msg.data.split(":")
                    self.current_safety_status = parts[1]
                    
                    # Activate safety constraints for elevated safety levels
                    self.safety_constraints_active = self.current_safety_status in ["WARNING", "DANGER", "CRITICAL"]
                    
            except (ValueError, IndexError):
                self.log(f"Failed to parse safety status: {msg.data}", "warn")
    
    def cb_collision_risk(self, msg):
        """
        Callback for collision risk assessment
        
        Args:
            msg (:obj:`String`): Collision risk message
        """
        with self.state_lock:
            # Parse collision risk (format: LEVEL:TTC:ACTION)
            try:
                parts = msg.data.split(":")
                self.current_collision_risk = parts[0]
                
            except (ValueError, IndexError):
                self.log(f"Failed to parse collision risk: {msg.data}", "warn")
    
    def arbitrate_commands(self, event):
        """
        Main command arbitration function
        
        Args:
            event: Timer event
        """
        with self.state_lock:
            # Get the highest priority valid command
            selected_command = self.select_highest_priority_command()
            
            if selected_command is None:
                # No valid commands available, send stop command
                self.publish_stop_command("no_valid_commands")
                return
            
            # Validate command against safety constraints
            validated_command = self.validate_and_constrain_command(selected_command)
            
            # Publish the final command
            self.publish_command(validated_command)
            
            # Update statistics
            self.command_stats["total_commands"] += 1
    
    def select_highest_priority_command(self):
        """
        Select the highest priority valid command from available sources
        
        Returns:
            dict: Selected command info or None if no valid commands
        """
        current_time = rospy.Time.now()
        timeout_threshold = self.command_timeout.value
        
        valid_commands = []
        
        # Check each command source
        for source_name, source_info in self.command_sources.items():
            cmd = source_info["cmd"]
            timestamp = source_info["timestamp"]
            priority = source_info["priority"]
            
            # Check if command is valid (not None and not timed out)
            if cmd is not None and timestamp is not None:
                age = (current_time - timestamp).to_sec()
                if age <= timeout_threshold:
                    valid_commands.append({
                        "source": source_name,
                        "command": cmd,
                        "priority": priority,
                        "age": age
                    })
        
        # Sort by priority (highest first)
        valid_commands.sort(key=lambda x: x["priority"], reverse=True)
        
        # Return highest priority command
        return valid_commands[0] if valid_commands else None
    
    def validate_and_constrain_command(self, command_info):
        """
        Validate command against safety constraints and apply limits
        
        Args:
            command_info (dict): Command information
            
        Returns:
            Twist2DStamped: Validated and constrained command
        """
        original_cmd = command_info["command"]
        source = command_info["source"]
        
        # Create output command
        validated_cmd = Twist2DStamped()
        validated_cmd.header.stamp = rospy.Time.now()
        validated_cmd.v = original_cmd.v
        validated_cmd.omega = original_cmd.omega
        
        violations = []
        
        # Apply basic velocity limits
        if self.enable_safety_limits.value:
            # Linear velocity limits
            if abs(validated_cmd.v) > self.max_linear_velocity.value:
                violations.append(f"linear_velocity_limit:{validated_cmd.v:.3f}>{self.max_linear_velocity.value}")
                validated_cmd.v = np.sign(validated_cmd.v) * self.max_linear_velocity.value
            
            # Angular velocity limits
            if abs(validated_cmd.omega) > self.max_angular_velocity.value:
                violations.append(f"angular_velocity_limit:{validated_cmd.omega:.3f}>{self.max_angular_velocity.value}")
                validated_cmd.omega = np.sign(validated_cmd.omega) * self.max_angular_velocity.value
        
        # Apply safety-based constraints
        if self.safety_constraints_active:
            safety_factor = self.get_safety_constraint_factor()
            
            # Reduce velocities based on safety level
            validated_cmd.v *= safety_factor
            validated_cmd.omega *= safety_factor
            
            if safety_factor < 1.0:
                violations.append(f"safety_constraint_applied:factor_{safety_factor:.2f}")
        
        # Apply collision risk-based constraints
        if self.current_collision_risk in ["HIGH", "CRITICAL"]:
            # Severely limit or stop movement for high collision risk
            if self.current_collision_risk == "CRITICAL":
                validated_cmd.v = 0.0
                validated_cmd.omega = 0.0
                violations.append("collision_risk_critical:movement_blocked")
                self.command_stats["commands_blocked"] += 1
            elif self.current_collision_risk == "HIGH":
                validated_cmd.v *= 0.3  # Reduce to 30% speed
                validated_cmd.omega *= 0.5  # Reduce angular velocity
                violations.append("collision_risk_high:speed_reduced")
        
        # Log violations if any occurred
        if violations:
            self.command_stats["safety_violations"] += len(violations)
            violation_msg = String()
            violation_msg.data = f"{source}:{','.join(violations)}"
            self.pub_safety_violations.publish(violation_msg)
        
        return validated_cmd
    
    def get_safety_constraint_factor(self):
        """
        Get velocity constraint factor based on current safety level
        
        Returns:
            float: Constraint factor (0.0 to 1.0)
        """
        if self.current_safety_status == "CRITICAL":
            return 0.0  # Complete stop
        elif self.current_safety_status == "DANGER":
            return 0.3  # 30% of normal speed
        elif self.current_safety_status == "WARNING":
            return 0.6  # 60% of normal speed
        elif self.current_safety_status == "CAUTION":
            return 0.8  # 80% of normal speed
        else:
            return 1.0  # Normal speed
    
    def publish_command(self, command):
        """
        Publish the final arbitrated command
        
        Args:
            command (Twist2DStamped): Command to publish
        """
        self.pub_car_cmd.publish(command)
        
        # Publish command status
        status_msg = String()
        status_msg.data = f"CMD_PUBLISHED:v_{command.v:.3f}:omega_{command.omega:.3f}:safety_{self.current_safety_status}"
        self.pub_command_status.publish(status_msg)
    
    def publish_stop_command(self, reason):
        """
        Publish a stop command with specified reason
        
        Args:
            reason (str): Reason for stop command
        """
        stop_cmd = Twist2DStamped()
        stop_cmd.header.stamp = rospy.Time.now()
        stop_cmd.v = 0.0
        stop_cmd.omega = 0.0
        
        self.pub_car_cmd.publish(stop_cmd)
        
        # Publish status
        status_msg = String()
        status_msg.data = f"STOP_COMMAND:{reason}"
        self.pub_command_status.publish(status_msg)
        
        self.command_stats["commands_blocked"] += 1
    
    def get_command_statistics(self):
        """
        Get command arbitration statistics
        
        Returns:
            dict: Command statistics
        """
        with self.state_lock:
            return self.command_stats.copy()
    
    def reset_statistics(self):
        """
        Reset command statistics
        """
        with self.state_lock:
            self.command_stats = {
                "total_commands": 0,
                "safety_violations": 0,
                "overrides_applied": 0,
                "commands_blocked": 0
            }


if __name__ == "__main__":
    # Create and run the safety command arbiter node
    safety_arbiter_node = SafetyCommandArbiterNode("safety_command_arbiter_node")
    rospy.spin()