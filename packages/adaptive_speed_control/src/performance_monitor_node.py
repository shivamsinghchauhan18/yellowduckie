#!/usr/bin/env python3

import rospy
import numpy as np
from threading import Lock
from collections import deque
from typing import Dict, List

from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import Twist2DStamped
from std_msgs.msg import String, Header


class PerformanceMonitorNode(DTROS):
    """
    Performance Monitor Node for Adaptive Speed Control System
    
    Monitors the performance of the adaptive speed control system and its
    integration with safety systems, providing metrics and alerts.
    
    Args:
        node_name (:obj:`str`): a unique, descriptive name for the node
    
    Subscribers:
        ~speed_commands (:obj:`Twist2DStamped`): Speed commands from adaptive controller
        ~safety_overrides (:obj:`String`): Safety override events
        ~performance_stats (:obj:`String`): Performance statistics from controller
    
    Publishers:
        ~performance_report (:obj:`String`): Performance monitoring report
        ~alerts (:obj:`String`): Performance alerts and warnings
    """
    
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(PerformanceMonitorNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.DIAGNOSTICS
        )
        
        # Performance tracking
        self.speed_command_history = deque(maxlen=1000)
        self.safety_override_history = deque(maxlen=100)
        self.performance_metrics = {
            "total_commands": 0,
            "average_speed": 0.0,
            "speed_variance": 0.0,
            "safety_override_rate": 0.0,
            "emergency_stop_count": 0,
            "system_uptime": 0.0
        }
        
        # Thread safety
        self.state_lock = Lock()
        
        # Timing
        self.start_time = rospy.Time.now()
        self.last_report_time = rospy.Time.now()
        
        # Alert thresholds
        self.alert_thresholds = {
            "max_safety_override_rate": 0.1,  # 10%
            "max_speed_variance": 0.05,       # 0.05 m/s variance
            "min_average_speed": 0.05,        # Minimum operational speed
            "max_emergency_stops_per_minute": 2
        }
        
        # Publishers
        self.pub_performance_report = rospy.Publisher(
            "~performance_report",
            String,
            queue_size=1
        )
        
        self.pub_alerts = rospy.Publisher(
            "~alerts",
            String,
            queue_size=1
        )
        
        # Subscribers
        self.sub_speed_commands = rospy.Subscriber(
            "~speed_commands",
            Twist2DStamped,
            self.cb_speed_commands,
            queue_size=10
        )
        
        self.sub_safety_overrides = rospy.Subscriber(
            "~safety_overrides",
            String,
            self.cb_safety_overrides,
            queue_size=10
        )
        
        self.sub_performance_stats = rospy.Subscriber(
            "~performance_stats",
            String,
            self.cb_performance_stats,
            queue_size=1
        )
        
        # Performance monitoring timer
        self.monitor_timer = rospy.Timer(
            rospy.Duration(5.0),  # 5-second monitoring interval
            self.monitor_performance
        )
        
        # Report generation timer
        self.report_timer = rospy.Timer(
            rospy.Duration(30.0),  # 30-second report interval
            self.generate_performance_report
        )
        
        self.log("Performance Monitor initialized")
    
    def cb_speed_commands(self, msg):
        """Callback for speed command monitoring."""
        with self.state_lock:
            command_data = {
                "timestamp": msg.header.stamp.to_sec(),
                "linear_velocity": msg.v,
                "angular_velocity": msg.omega
            }
            
            self.speed_command_history.append(command_data)
            self.performance_metrics["total_commands"] += 1
    
    def cb_safety_overrides(self, msg):
        """Callback for safety override monitoring."""
        with self.state_lock:
            override_data = {
                "timestamp": rospy.Time.now().to_sec(),
                "message": msg.data
            }
            
            self.safety_override_history.append(override_data)
            
            # Check for emergency stops
            if "emergency" in msg.data.lower() or "stop" in msg.data.lower():
                self.performance_metrics["emergency_stop_count"] += 1
    
    def cb_performance_stats(self, msg):
        """Callback for performance statistics updates."""
        with self.state_lock:
            # Parse performance statistics
            try:
                # Expected format: "total:X,env_adj:Y,follow_adj:Z,safety_override:W,emergency:V"
                parts = msg.data.split(",")
                for part in parts:
                    if ":" in part:
                        key, value = part.split(":")
                        if key == "total":
                            self.performance_metrics["total_commands"] = int(value)
                        elif key == "emergency":
                            self.performance_metrics["emergency_stop_count"] = int(value)
                            
            except (ValueError, IndexError) as e:
                rospy.logwarn(f"Failed to parse performance stats: {e}")
    
    def monitor_performance(self, event):
        """Monitor system performance and generate alerts."""
        with self.state_lock:
            current_time = rospy.Time.now()
            
            # Update system uptime
            self.performance_metrics["system_uptime"] = (current_time - self.start_time).to_sec()
            
            # Calculate performance metrics
            self._calculate_speed_metrics()
            self._calculate_safety_metrics()
            
            # Check for performance alerts
            self._check_performance_alerts()
    
    def _calculate_speed_metrics(self):
        """Calculate speed-related performance metrics."""
        if not self.speed_command_history:
            return
        
        # Extract recent speed commands (last 100)
        recent_commands = list(self.speed_command_history)[-100:]
        speeds = [cmd["linear_velocity"] for cmd in recent_commands]
        
        if speeds:
            self.performance_metrics["average_speed"] = np.mean(speeds)
            self.performance_metrics["speed_variance"] = np.var(speeds)
    
    def _calculate_safety_metrics(self):
        """Calculate safety-related performance metrics."""
        current_time = rospy.Time.now().to_sec()
        
        # Calculate safety override rate (last 5 minutes)
        recent_overrides = [
            override for override in self.safety_override_history
            if current_time - override["timestamp"] <= 300  # 5 minutes
        ]
        
        recent_commands = [
            cmd for cmd in self.speed_command_history
            if current_time - cmd["timestamp"] <= 300  # 5 minutes
        ]
        
        if len(recent_commands) > 0:
            self.performance_metrics["safety_override_rate"] = len(recent_overrides) / len(recent_commands)
        else:
            self.performance_metrics["safety_override_rate"] = 0.0
    
    def _check_performance_alerts(self):
        """Check for performance issues and generate alerts."""
        alerts = []
        
        # Check safety override rate
        if self.performance_metrics["safety_override_rate"] > self.alert_thresholds["max_safety_override_rate"]:
            alerts.append(f"HIGH_SAFETY_OVERRIDE_RATE:{self.performance_metrics['safety_override_rate']:.3f}")
        
        # Check speed variance (smoothness)
        if self.performance_metrics["speed_variance"] > self.alert_thresholds["max_speed_variance"]:
            alerts.append(f"HIGH_SPEED_VARIANCE:{self.performance_metrics['speed_variance']:.3f}")
        
        # Check minimum operational speed
        if (self.performance_metrics["average_speed"] > 0 and 
            self.performance_metrics["average_speed"] < self.alert_thresholds["min_average_speed"]):
            alerts.append(f"LOW_AVERAGE_SPEED:{self.performance_metrics['average_speed']:.3f}")
        
        # Check emergency stop frequency
        uptime_minutes = self.performance_metrics["system_uptime"] / 60.0
        if uptime_minutes > 1.0:  # Only check after 1 minute of operation
            emergency_rate = self.performance_metrics["emergency_stop_count"] / uptime_minutes
            if emergency_rate > self.alert_thresholds["max_emergency_stops_per_minute"]:
                alerts.append(f"HIGH_EMERGENCY_STOP_RATE:{emergency_rate:.3f}")
        
        # Publish alerts if any
        if alerts:
            alert_msg = String()
            alert_msg.data = ",".join(alerts)
            self.pub_alerts.publish(alert_msg)
            
            rospy.logwarn(f"Performance alerts: {alert_msg.data}")
    
    def generate_performance_report(self, event):
        """Generate comprehensive performance report."""
        with self.state_lock:
            current_time = rospy.Time.now()
            
            # Update metrics before reporting
            self._calculate_speed_metrics()
            self._calculate_safety_metrics()
            
            # Generate report
            report_data = {
                "timestamp": current_time.to_sec(),
                "uptime_seconds": self.performance_metrics["system_uptime"],
                "total_commands": self.performance_metrics["total_commands"],
                "average_speed": self.performance_metrics["average_speed"],
                "speed_variance": self.performance_metrics["speed_variance"],
                "safety_override_rate": self.performance_metrics["safety_override_rate"],
                "emergency_stop_count": self.performance_metrics["emergency_stop_count"],
                "command_frequency": self._calculate_command_frequency(),
                "system_health": self._assess_system_health()
            }
            
            # Format report message
            report_msg = String()
            report_msg.data = self._format_report(report_data)
            self.pub_performance_report.publish(report_msg)
            
            # Log summary
            if rospy.get_param("~verbose", False):
                self.log(f"Performance Report: {report_msg.data}")
    
    def _calculate_command_frequency(self):
        """Calculate command frequency (Hz)."""
        if len(self.speed_command_history) < 2:
            return 0.0
        
        # Calculate frequency from recent commands
        recent_commands = list(self.speed_command_history)[-50:]  # Last 50 commands
        if len(recent_commands) < 2:
            return 0.0
        
        time_span = recent_commands[-1]["timestamp"] - recent_commands[0]["timestamp"]
        if time_span > 0:
            return (len(recent_commands) - 1) / time_span
        else:
            return 0.0
    
    def _assess_system_health(self):
        """Assess overall system health."""
        health_score = 100.0  # Start with perfect health
        
        # Deduct points for performance issues
        if self.performance_metrics["safety_override_rate"] > 0.05:
            health_score -= 20.0  # High safety override rate
        
        if self.performance_metrics["speed_variance"] > 0.03:
            health_score -= 15.0  # High speed variance
        
        if self.performance_metrics["emergency_stop_count"] > 0:
            uptime_minutes = self.performance_metrics["system_uptime"] / 60.0
            if uptime_minutes > 0:
                emergency_rate = self.performance_metrics["emergency_stop_count"] / uptime_minutes
                if emergency_rate > 1.0:
                    health_score -= 30.0  # Frequent emergency stops
        
        # Command frequency check
        command_freq = self._calculate_command_frequency()
        if command_freq < 5.0:  # Less than 5 Hz
            health_score -= 10.0
        
        return max(0.0, health_score)
    
    def _format_report(self, report_data):
        """Format performance report as string."""
        return (f"uptime:{report_data['uptime_seconds']:.1f}s,"
                f"commands:{report_data['total_commands']},"
                f"avg_speed:{report_data['average_speed']:.3f},"
                f"speed_var:{report_data['speed_variance']:.4f},"
                f"safety_rate:{report_data['safety_override_rate']:.3f},"
                f"emergency_stops:{report_data['emergency_stop_count']},"
                f"cmd_freq:{report_data['command_frequency']:.1f}Hz,"
                f"health:{report_data['system_health']:.1f}%")
    
    def get_performance_summary(self):
        """Get current performance summary."""
        with self.state_lock:
            return self.performance_metrics.copy()
    
    def reset_performance_metrics(self):
        """Reset performance metrics."""
        with self.state_lock:
            self.speed_command_history.clear()
            self.safety_override_history.clear()
            self.performance_metrics = {
                "total_commands": 0,
                "average_speed": 0.0,
                "speed_variance": 0.0,
                "safety_override_rate": 0.0,
                "emergency_stop_count": 0,
                "system_uptime": 0.0
            }
            self.start_time = rospy.Time.now()


if __name__ == "__main__":
    # Create and run the performance monitor node
    performance_monitor = PerformanceMonitorNode("adaptive_speed_performance_monitor")
    rospy.spin()