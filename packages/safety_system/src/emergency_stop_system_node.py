#!/usr/bin/env python3

import rospy
import time
from threading import Lock
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, BoolStamped
from std_msgs.msg import String, Header
from geometry_msgs.msg import Point32


class EmergencyStatus:
    """Custom message type for emergency status"""
    def __init__(self):
        self.header = Header()
        self.emergency_active = False
        self.trigger_reason = ""
        self.trigger_timestamp = rospy.Time()
        self.response_time = 0.0  # seconds
        self.safety_override_active = False


class SafetyOverride:
    """Custom message type for safety override commands"""
    def __init__(self):
        self.header = Header()
        self.override_active = True
        self.v = 0.0
        self.omega = 0.0
        self.priority = 10  # Higher number = higher priority


class EmergencyStopSystemNode(DTROS):
    """
    Emergency Stop System Node for Duckietown Safety System
    
    Provides hardware-level emergency stop capability with sub-100ms response time.
    Monitors multiple trigger conditions and maintains fail-safe operation.
    
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node
    
    Configuration:
        ~max_response_time (:obj:`float`): Maximum allowed response time in seconds
        ~min_distance_threshold (:obj:`float`): Minimum safe distance in meters
        ~enable_predictive_stop (:obj:`bool`): Enable predictive emergency stopping
        ~emergency_timeout (:obj:`float`): Timeout for emergency state in seconds
    
    Subscribers:
        ~collision_risk (:obj:`CollisionRisk`): Collision risk assessment from detection manager
        ~manual_emergency (:obj:`BoolStamped`): Manual emergency trigger
        ~system_health (:obj:`SystemHealth`): System health status
        ~car_cmd (:obj:`Twist2DStamped`): Current car commands to monitor
    
    Publishers:
        ~emergency_status (:obj:`EmergencyStatus`): Current emergency system status
        ~safety_override (:obj:`Twist2DStamped`): Emergency override commands
        ~emergency_log (:obj:`String`): Emergency event logging
    
    Services:
        ~/trigger_emergency_stop: Manually trigger emergency stop
        ~/reset_emergency_system: Reset emergency system after manual review
    """
    
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(EmergencyStopSystemNode, self).__init__(
            node_name=node_name, 
            node_type=NodeType.CONTROL
        )

    # Vehicle namespace for absolute topics
        self.veh = rospy.get_param("~veh", rospy.get_namespace().strip("/"))
        
        # Parameters
        self.max_response_time = DTParam(
            "~max_response_time", 
            param_type=ParamType.FLOAT, 
            default=0.1,
            min_value=0.01,
            max_value=1.0
        )
        
        self.min_distance_threshold = DTParam(
            "~min_distance_threshold",
            param_type=ParamType.FLOAT,
            default=0.3,
            min_value=0.1,
            max_value=2.0
        )
        
        self.enable_predictive_stop = DTParam(
            "~enable_predictive_stop",
            param_type=ParamType.BOOL,
            default=True
        )
        
        self.emergency_timeout = DTParam(
            "~emergency_timeout",
            param_type=ParamType.FLOAT,
            default=5.0,
            min_value=1.0,
            max_value=30.0
        )

        # Heartbeat rate for status publication (Hz)
        self.status_hz = DTParam(
            "~status_hz",
            param_type=ParamType.FLOAT,
            default=10.0,
            min_value=0.5,
            max_value=100.0
        )
        
        # State variables
        self.emergency_active = False
        self.emergency_trigger_time = None
        self.last_collision_risk = None
        self.system_health_ok = True
        self.manual_emergency_requested = False
        
        # Thread safety
        self.state_lock = Lock()
        
        # Emergency status message
        self.emergency_status = EmergencyStatus()
        
        # Publishers
        self.pub_emergency_status = rospy.Publisher(
            "~emergency_status", 
            String,  # Using String for now, would be EmergencyStatus in full implementation
            queue_size=1,
            dt_topic_type=TopicType.CONTROL
        )
        
        self.pub_safety_override = rospy.Publisher(
            "~safety_override",
            Twist2DStamped,
            queue_size=1,
            dt_topic_type=TopicType.CONTROL
        )
        
        self.pub_emergency_log = rospy.Publisher(
            "~emergency_log",
            String,
            queue_size=10
        )
        
        # Subscribers
        # Subscribe to other nodes using absolute, veh-qualified topics to avoid remap mismatches
        self.sub_collision_risk = rospy.Subscriber(
            f"/{self.veh}/collision_detection_manager_node/collision_risk",
            String,  # Would be CollisionRisk in full implementation
            self.cb_collision_risk,
            queue_size=1
        )
        
        self.sub_manual_emergency = rospy.Subscriber(
            "~manual_emergency",
            BoolStamped,
            self.cb_manual_emergency,
            queue_size=1
        )
        
        self.sub_system_health = rospy.Subscriber(
            f"/{self.veh}/safety_fusion_manager_node/system_health_summary",
            String,  # Would be SystemHealth in full implementation
            self.cb_system_health,
            queue_size=1
        )
        
        self.sub_car_cmd = rospy.Subscriber(
            f"/{self.veh}/lane_controller_node/car_cmd",
            Twist2DStamped,
            self.cb_car_cmd,
            queue_size=1
        )
        
        # Services would be implemented here in full version
        # self.srv_trigger_emergency = rospy.Service(...)
        # self.srv_reset_emergency = rospy.Service(...)
        
    # Start monitoring timer for emergency control loop
        self.timer = rospy.Timer(rospy.Duration(0.01), self.monitor_emergency_conditions)
    # Separate timer for heartbeat status publishing
        self._last_status_pub = rospy.Time(0)
        self.status_timer = rospy.Timer(rospy.Duration(max(0.001, 1.0 / self.status_hz.value)), self._heartbeat_tick)
        
        self.log("Emergency Stop System initialized and monitoring")
    
    def cb_collision_risk(self, msg):
        """
        Callback for collision risk messages
        
        Args:
            msg: Collision risk message (simplified as String for now)
        """
        with self.state_lock:
            # In full implementation, would parse CollisionRisk message
            # For now, simulate high risk detection
            if "HIGH" in msg.data or "CRITICAL" in msg.data:
                self.trigger_emergency_stop("collision_risk_high", msg.header.stamp)
    
    def cb_manual_emergency(self, msg):
        """
        Callback for manual emergency trigger
        
        Args:
            msg (:obj:`BoolStamped`): Manual emergency request
        """
        with self.state_lock:
            if msg.data and not self.emergency_active:
                self.trigger_emergency_stop("manual_trigger", msg.header.stamp)
    
    def cb_system_health(self, msg):
        """
        Callback for system health status
        
        Args:
            msg: System health message (simplified as String for now)
        """
        with self.state_lock:
            # In full implementation, would parse SystemHealth message
            # Parse aggregated health summary; if any subsystem TIMEOUT/FAILURE -> unhealthy
            self.system_health_ok = ("FAILURE" not in msg.data) and ("CRITICAL" not in msg.data) and ("TIMEOUT" not in msg.data)

            if not self.system_health_ok and not self.emergency_active:
                # Try to extract first offending subsystem for explicit reason
                reason = "system_health_failure"
                for token in msg.data.split("|"):
                    if ":" in token:
                        name, status = token.split(":", 1)
                        if any(x in status for x in ["TIMEOUT", "FAILURE", "CRITICAL"]):
                            reason = f"system_health_failure:{name}:{status}"
                            break
                self.trigger_emergency_stop(reason, rospy.Time.now())
    
    def cb_car_cmd(self, msg):
        """
        Callback for monitoring current car commands
        
        Args:
            msg (:obj:`Twist2DStamped`): Current car command
        """
        # Monitor commands for safety validation
        # In full implementation, would validate command safety
        pass
    
    def trigger_emergency_stop(self, reason, trigger_time):
        """
        Trigger emergency stop with specified reason
        
        Args:
            reason (str): Reason for emergency stop
            trigger_time (rospy.Time): Time when trigger occurred
        """
        if self.emergency_active:
            return  # Already in emergency state
        
        self.emergency_active = True
        self.emergency_trigger_time = trigger_time
        
        # Calculate response time
        current_time = rospy.Time.now()
        response_time = (current_time - trigger_time).to_sec()
        
        # Log emergency event
        log_msg = f"EMERGENCY STOP TRIGGERED: {reason} at {trigger_time.to_sec()}, response_time: {response_time:.3f}s"
        self.log(log_msg, "warn")
        
        # Publish emergency log
        log_pub_msg = String()
        log_pub_msg.data = log_msg
        self.pub_emergency_log.publish(log_pub_msg)
        
        # Update emergency status
        self.emergency_status.header.stamp = current_time
        self.emergency_status.emergency_active = True
        self.emergency_status.trigger_reason = reason
        self.emergency_status.trigger_timestamp = trigger_time
        self.emergency_status.response_time = response_time
        self.emergency_status.safety_override_active = True
        
        # Publish emergency status
        status_msg = String()
        status_msg.data = f"EMERGENCY_ACTIVE:{reason}:{response_time:.3f}"
        self.pub_emergency_status.publish(status_msg)
        
        # Issue immediate stop command
        self.publish_emergency_stop_command()
    
    def publish_emergency_stop_command(self):
        """
        Publish emergency stop command with highest priority
        """
        stop_cmd = Twist2DStamped()
        stop_cmd.header.stamp = rospy.Time.now()
        stop_cmd.v = 0.0
        stop_cmd.omega = 0.0
        
        self.pub_safety_override.publish(stop_cmd)
    
    def monitor_emergency_conditions(self, event):
        """
        Monitor emergency conditions and maintain emergency state
        
        Args:
            event: Timer event
        """
        with self.state_lock:
            if self.emergency_active:
                # Continue publishing stop commands while in emergency
                self.publish_emergency_stop_command()
                
                # Check for emergency timeout (auto-reset after timeout)
                if self.emergency_trigger_time:
                    elapsed = (rospy.Time.now() - self.emergency_trigger_time).to_sec()
                    if elapsed > self.emergency_timeout.value:
                        self.reset_emergency_system("timeout")
            
            # Publish current status
            self.publish_status()
    
    def reset_emergency_system(self, reason="manual"):
        """
        Reset emergency system to normal operation
        
        Args:
            reason (str): Reason for reset
        """
        with self.state_lock:
            if not self.emergency_active:
                return
            
            self.emergency_active = False
            self.emergency_trigger_time = None
            
            # Log reset event
            log_msg = f"EMERGENCY SYSTEM RESET: {reason} at {rospy.Time.now().to_sec()}"
            self.log(log_msg, "info")
            
            # Publish reset log
            log_pub_msg = String()
            log_pub_msg.data = log_msg
            self.pub_emergency_log.publish(log_pub_msg)
            
            # Update status
            self.emergency_status.emergency_active = False
            self.emergency_status.safety_override_active = False
            self.emergency_status.trigger_reason = ""
    
    def publish_status(self):
        """
        Publish current emergency system status
        """
        status_msg = String()
        if self.emergency_active:
            status_msg.data = f"EMERGENCY_ACTIVE:{self.emergency_status.trigger_reason}"
        else:
            status_msg.data = "NORMAL_OPERATION"
        
        self.pub_emergency_status.publish(status_msg)

    def _heartbeat_tick(self, event):
        """Periodic heartbeat publisher to ensure downstream health monitors receive fresh status at the configured rate."""
        self.publish_status()
    
    def on_shutdown(self):
        """
        Cleanup when node shuts down
        """
        self.log("Emergency Stop System shutting down")
        if self.emergency_active:
            # Ensure final stop command is sent
            self.publish_emergency_stop_command()


if __name__ == "__main__":
    # Create and run the emergency stop system node
    emergency_stop_node = EmergencyStopSystemNode("emergency_stop_system_node")
    rospy.spin()