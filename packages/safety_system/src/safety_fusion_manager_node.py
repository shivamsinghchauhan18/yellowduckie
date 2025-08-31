#!/usr/bin/env python3

import rospy
import numpy as np
from threading import Lock
from collections import deque
from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, BoolStamped
from std_msgs.msg import String, Header


class SafetyStatus:
    """Custom message type for overall safety status"""
    def __init__(self):
        self.header = Header()
        self.emergency_active = False
        self.safety_systems_health = {}  # Dict of system_name: health_status
        self.active_risks = []  # List of active collision risks
        self.safety_margins = {}  # Dict of margin_type: current_value
        self.last_safety_event = ""
        self.overall_safety_level = 0  # 0=SAFE, 1=CAUTION, 2=WARNING, 3=DANGER, 4=CRITICAL


class SafetyFusionManagerNode(DTROS):
    """
    Safety Fusion Manager Node for Duckietown Safety System
    
    Coordinates multiple safety systems and provides unified safety decision making.
    Implements safety data fusion algorithms and decision arbitration logic.
    
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node
    
    Configuration:
        ~fusion_update_rate (:obj:`float`): Safety fusion update rate in Hz
        ~health_check_timeout (:obj:`float`): Timeout for system health checks in seconds
        ~safety_margin_buffer (:obj:`float`): Additional safety margin buffer
        ~decision_confidence_threshold (:obj:`float`): Minimum confidence for safety decisions
    
    Subscribers:
        ~emergency_status (:obj:`String`): Emergency system status
        ~collision_risk (:obj:`String`): Collision risk assessment
        ~system_health_reports (:obj:`String`): Health reports from various systems
        ~sensor_health (:obj:`String`): Sensor health status
        ~actuator_health (:obj:`String`): Actuator health status
    
    Publishers:
        ~safety_status (:obj:`String`): Overall safety system status
        ~safety_recommendations (:obj:`String`): Safety recommendations for other systems
        ~safety_alerts (:obj:`String`): Critical safety alerts
        ~system_health_summary (:obj:`String`): Summary of all system health
    """
    
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(SafetyFusionManagerNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )
        
        # Parameters
        self.fusion_update_rate = DTParam(
            "~fusion_update_rate",
            param_type=ParamType.FLOAT,
            default=10.0,
            min_value=1.0,
            max_value=50.0
        )
        
        self.health_check_timeout = DTParam(
            "~health_check_timeout",
            param_type=ParamType.FLOAT,
            default=2.0,
            min_value=0.5,
            max_value=10.0
        )
        
        self.safety_margin_buffer = DTParam(
            "~safety_margin_buffer",
            param_type=ParamType.FLOAT,
            default=0.1,
            min_value=0.0,
            max_value=1.0
        )
        
        self.decision_confidence_threshold = DTParam(
            "~decision_confidence_threshold",
            param_type=ParamType.FLOAT,
            default=0.8,
            min_value=0.1,
            max_value=1.0
        )
        
        # State variables
        self.safety_status = SafetyStatus()
        self.system_health_reports = {}
        self.collision_risks = deque(maxlen=10)  # Keep recent collision risks
        self.emergency_status = {"active": False, "reason": ""}
        self.sensor_health = {}
        self.actuator_health = {}
        
        # Safety level constants
        self.SAFETY_SAFE = 0
        self.SAFETY_CAUTION = 1
        self.SAFETY_WARNING = 2
        self.SAFETY_DANGER = 3
        self.SAFETY_CRITICAL = 4
        
        # Thread safety
        self.state_lock = Lock()
        
        # Health check tracking
        self.last_health_updates = {}
        
        # Publishers
        self.pub_safety_status = rospy.Publisher(
            "~safety_status",
            String,  # Would be SafetyStatus in full implementation
            queue_size=1,
            dt_topic_type=TopicType.CONTROL
        )
        
        self.pub_safety_recommendations = rospy.Publisher(
            "~safety_recommendations",
            String,
            queue_size=1,
            dt_topic_type=TopicType.CONTROL
        )
        
        self.pub_safety_alerts = rospy.Publisher(
            "~safety_alerts",
            String,
            queue_size=1,
            dt_topic_type=TopicType.CONTROL
        )
        
        self.pub_system_health_summary = rospy.Publisher(
            "~system_health_summary",
            String,
            queue_size=1
        )
        
        # Subscribers
        self.sub_emergency_status = rospy.Subscriber(
            "~emergency_status",
            String,
            self.cb_emergency_status,
            queue_size=1
        )
        
        self.sub_collision_risk = rospy.Subscriber(
            "~collision_risk",
            String,
            self.cb_collision_risk,
            queue_size=1
        )
        
        self.sub_system_health_reports = rospy.Subscriber(
            "~system_health_reports",
            String,
            self.cb_system_health_reports,
            queue_size=5
        )
        
        self.sub_sensor_health = rospy.Subscriber(
            "~sensor_health",
            String,
            self.cb_sensor_health,
            queue_size=1
        )
        
        self.sub_actuator_health = rospy.Subscriber(
            "~actuator_health",
            String,
            self.cb_actuator_health,
            queue_size=1
        )
        
        # Start safety fusion timer
        self.timer = rospy.Timer(
            rospy.Duration(1.0 / self.fusion_update_rate.value),
            self.update_safety_fusion
        )
        
        # Start health monitoring timer
        self.health_timer = rospy.Timer(
            rospy.Duration(1.0),  # Check health every second
            self.monitor_system_health
        )
        
        self.log("Safety Fusion Manager initialized")
    
    def cb_emergency_status(self, msg):
        """
        Callback for emergency system status
        
        Args:
            msg (:obj:`String`): Emergency status message
        """
        with self.state_lock:
            # Parse emergency status (simplified)
            if "EMERGENCY_ACTIVE" in msg.data:
                parts = msg.data.split(":")
                self.emergency_status = {
                    "active": True,
                    "reason": parts[1] if len(parts) > 1 else "unknown",
                    "timestamp": rospy.Time.now()
                }
            else:
                self.emergency_status = {"active": False, "reason": ""}
            
            self.last_health_updates["emergency_system"] = rospy.Time.now()
    
    def cb_collision_risk(self, msg):
        """
        Callback for collision risk assessment
        
        Args:
            msg (:obj:`String`): Collision risk message
        """
        with self.state_lock:
            # Parse collision risk (simplified format: LEVEL:TTC:ACTION)
            try:
                parts = msg.data.split(":")
                risk_info = {
                    "level": parts[0],
                    "time_to_collision": float(parts[1]) if len(parts) > 1 else -1,
                    "action": parts[2] if len(parts) > 2 else "CONTINUE",
                    "timestamp": rospy.Time.now()
                }
                self.collision_risks.append(risk_info)
                
            except (ValueError, IndexError):
                self.log(f"Failed to parse collision risk message: {msg.data}", "warn")
            
            self.last_health_updates["collision_detection"] = rospy.Time.now()
    
    def cb_system_health_reports(self, msg):
        """
        Callback for system health reports
        
        Args:
            msg (:obj:`String`): System health report
        """
        with self.state_lock:
            # Parse system health report (format: SYSTEM_NAME:STATUS:DETAILS)
            try:
                parts = msg.data.split(":", 2)
                system_name = parts[0]
                status = parts[1]
                details = parts[2] if len(parts) > 2 else ""
                
                self.system_health_reports[system_name] = {
                    "status": status,
                    "details": details,
                    "timestamp": rospy.Time.now()
                }
                
                self.last_health_updates[system_name] = rospy.Time.now()
                
            except (ValueError, IndexError):
                self.log(f"Failed to parse system health report: {msg.data}", "warn")
    
    def cb_sensor_health(self, msg):
        """
        Callback for sensor health status
        
        Args:
            msg (:obj:`String`): Sensor health message
        """
        with self.state_lock:
            # Parse sensor health (format: SENSOR_NAME:STATUS)
            try:
                parts = msg.data.split(":")
                sensor_name = parts[0]
                status = parts[1]
                
                self.sensor_health[sensor_name] = {
                    "status": status,
                    "timestamp": rospy.Time.now()
                }
                
                self.last_health_updates[f"sensor_{sensor_name}"] = rospy.Time.now()
                
            except (ValueError, IndexError):
                self.log(f"Failed to parse sensor health message: {msg.data}", "warn")
    
    def cb_actuator_health(self, msg):
        """
        Callback for actuator health status
        
        Args:
            msg (:obj:`String`): Actuator health message
        """
        with self.state_lock:
            # Parse actuator health (format: ACTUATOR_NAME:STATUS)
            try:
                parts = msg.data.split(":")
                actuator_name = parts[0]
                status = parts[1]
                
                self.actuator_health[actuator_name] = {
                    "status": status,
                    "timestamp": rospy.Time.now()
                }
                
                self.last_health_updates[f"actuator_{actuator_name}"] = rospy.Time.now()
                
            except (ValueError, IndexError):
                self.log(f"Failed to parse actuator health message: {msg.data}", "warn")
    
    def update_safety_fusion(self, event):
        """
        Main safety fusion update function
        
        Args:
            event: Timer event
        """
        with self.state_lock:
            # Update safety status
            self.safety_status.header.stamp = rospy.Time.now()
            
            # Determine overall safety level
            overall_safety_level = self.calculate_overall_safety_level()
            self.safety_status.overall_safety_level = overall_safety_level
            
            # Update safety status fields
            self.safety_status.emergency_active = self.emergency_status["active"]
            self.safety_status.safety_systems_health = self.get_systems_health_summary()
            self.safety_status.active_risks = self.get_active_risks()
            self.safety_status.safety_margins = self.calculate_safety_margins()
            
            # Generate safety recommendations
            recommendations = self.generate_safety_recommendations(overall_safety_level)
            
            # Publish safety information
            self.publish_safety_status()
            self.publish_safety_recommendations(recommendations)
            
            # Check for critical alerts
            if overall_safety_level >= self.SAFETY_DANGER:
                self.publish_safety_alert(overall_safety_level)
    
    def calculate_overall_safety_level(self):
        """
        Calculate overall safety level based on all inputs
        
        Returns:
            int: Overall safety level
        """
        max_safety_level = self.SAFETY_SAFE
        
        # Check emergency status
        if self.emergency_status["active"]:
            max_safety_level = max(max_safety_level, self.SAFETY_CRITICAL)
        
        # Check collision risks
        for risk in self.collision_risks:
            if risk["level"] == "CRITICAL":
                max_safety_level = max(max_safety_level, self.SAFETY_CRITICAL)
            elif risk["level"] == "HIGH":
                max_safety_level = max(max_safety_level, self.SAFETY_DANGER)
            elif risk["level"] == "MEDIUM":
                max_safety_level = max(max_safety_level, self.SAFETY_WARNING)
            elif risk["level"] == "LOW":
                max_safety_level = max(max_safety_level, self.SAFETY_CAUTION)
        
        # Check system health
        for system_name, health_info in self.system_health_reports.items():
            if health_info["status"] == "CRITICAL_FAILURE":
                max_safety_level = max(max_safety_level, self.SAFETY_CRITICAL)
            elif health_info["status"] == "FAILURE":
                max_safety_level = max(max_safety_level, self.SAFETY_DANGER)
            elif health_info["status"] == "DEGRADED":
                max_safety_level = max(max_safety_level, self.SAFETY_WARNING)
        
        # Check sensor health
        critical_sensors_failed = 0
        for sensor_name, health_info in self.sensor_health.items():
            if health_info["status"] == "FAILED":
                critical_sensors_failed += 1
        
        if critical_sensors_failed >= 2:
            max_safety_level = max(max_safety_level, self.SAFETY_CRITICAL)
        elif critical_sensors_failed >= 1:
            max_safety_level = max(max_safety_level, self.SAFETY_DANGER)
        
        # Check actuator health
        for actuator_name, health_info in self.actuator_health.items():
            if health_info["status"] == "FAILED":
                max_safety_level = max(max_safety_level, self.SAFETY_CRITICAL)
            elif health_info["status"] == "DEGRADED":
                max_safety_level = max(max_safety_level, self.SAFETY_WARNING)
        
        return max_safety_level
    
    def get_systems_health_summary(self):
        """
        Get summary of all system health statuses
        
        Returns:
            dict: System health summary
        """
        health_summary = {}
        
        # Add system health reports
        for system_name, health_info in self.system_health_reports.items():
            health_summary[system_name] = health_info["status"]
        
        # Add sensor health
        for sensor_name, health_info in self.sensor_health.items():
            health_summary[f"sensor_{sensor_name}"] = health_info["status"]
        
        # Add actuator health
        for actuator_name, health_info in self.actuator_health.items():
            health_summary[f"actuator_{actuator_name}"] = health_info["status"]
        
        return health_summary
    
    def get_active_risks(self):
        """
        Get list of currently active risks
        
        Returns:
            list: Active collision risks
        """
        active_risks = []
        current_time = rospy.Time.now()
        
        # Get recent collision risks
        for risk in self.collision_risks:
            if (current_time - risk["timestamp"]).to_sec() < 2.0:  # Recent risks only
                if risk["level"] in ["MEDIUM", "HIGH", "CRITICAL"]:
                    active_risks.append(risk["level"])
        
        return active_risks
    
    def calculate_safety_margins(self):
        """
        Calculate current safety margins
        
        Returns:
            dict: Safety margins
        """
        margins = {}
        
        # Calculate collision margin based on recent risks
        min_ttc = float('inf')
        for risk in self.collision_risks:
            if risk["time_to_collision"] > 0:
                min_ttc = min(min_ttc, risk["time_to_collision"])
        
        margins["collision_time"] = min_ttc if min_ttc != float('inf') else -1
        
        # Calculate system health margin
        healthy_systems = sum(1 for health in self.system_health_reports.values() 
                            if health["status"] == "HEALTHY")
        total_systems = len(self.system_health_reports)
        margins["system_health"] = healthy_systems / max(total_systems, 1)
        
        return margins
    
    def generate_safety_recommendations(self, safety_level):
        """
        Generate safety recommendations based on current safety level
        
        Args:
            safety_level (int): Current overall safety level
            
        Returns:
            list: Safety recommendations
        """
        recommendations = []
        
        if safety_level >= self.SAFETY_CRITICAL:
            recommendations.append("IMMEDIATE_STOP")
            recommendations.append("ACTIVATE_EMERGENCY_PROTOCOLS")
        elif safety_level >= self.SAFETY_DANGER:
            recommendations.append("REDUCE_SPEED_50_PERCENT")
            recommendations.append("INCREASE_SAFETY_MARGINS")
        elif safety_level >= self.SAFETY_WARNING:
            recommendations.append("REDUCE_SPEED_25_PERCENT")
            recommendations.append("ENHANCED_MONITORING")
        elif safety_level >= self.SAFETY_CAUTION:
            recommendations.append("MAINTAIN_EXTRA_VIGILANCE")
        
        # Add specific recommendations based on active risks
        for risk in self.get_active_risks():
            if risk == "CRITICAL":
                recommendations.append("EMERGENCY_EVASION")
            elif risk == "HIGH":
                recommendations.append("IMMEDIATE_SLOWDOWN")
        
        return recommendations
    
    def monitor_system_health(self, event):
        """
        Monitor system health and detect timeouts
        
        Args:
            event: Timer event
        """
        with self.state_lock:
            current_time = rospy.Time.now()
            timeout_threshold = self.health_check_timeout.value
            
            # Check for health update timeouts
            for system_name, last_update in self.last_health_updates.items():
                if (current_time - last_update).to_sec() > timeout_threshold:
                    self.log(f"Health update timeout for {system_name}", "warn")
                    
                    # Mark system as unhealthy due to timeout
                    if system_name not in self.system_health_reports:
                        self.system_health_reports[system_name] = {}
                    
                    self.system_health_reports[system_name]["status"] = "TIMEOUT"
                    self.system_health_reports[system_name]["timestamp"] = current_time
    
    def publish_safety_status(self):
        """
        Publish overall safety status
        """
        safety_levels = ["SAFE", "CAUTION", "WARNING", "DANGER", "CRITICAL"]
        
        status_msg = String()
        status_msg.data = f"SAFETY_LEVEL:{safety_levels[self.safety_status.overall_safety_level]}:EMERGENCY_{self.safety_status.emergency_active}"
        
        self.pub_safety_status.publish(status_msg)
        
        # Publish system health summary
        health_summary = []
        for system, status in self.safety_status.safety_systems_health.items():
            health_summary.append(f"{system}:{status}")
        
        health_msg = String()
        health_msg.data = "|".join(health_summary)
        self.pub_system_health_summary.publish(health_msg)
    
    def publish_safety_recommendations(self, recommendations):
        """
        Publish safety recommendations
        
        Args:
            recommendations (list): List of safety recommendations
        """
        if recommendations:
            rec_msg = String()
            rec_msg.data = "|".join(recommendations)
            self.pub_safety_recommendations.publish(rec_msg)
    
    def publish_safety_alert(self, safety_level):
        """
        Publish critical safety alert
        
        Args:
            safety_level (int): Current safety level
        """
        safety_levels = ["SAFE", "CAUTION", "WARNING", "DANGER", "CRITICAL"]
        
        alert_msg = String()
        alert_msg.data = f"SAFETY_ALERT:{safety_levels[safety_level]}:IMMEDIATE_ATTENTION_REQUIRED"
        
        self.pub_safety_alerts.publish(alert_msg)


if __name__ == "__main__":
    # Create and run the safety fusion manager node
    safety_fusion_node = SafetyFusionManagerNode("safety_fusion_manager_node")
    rospy.spin()