# Emergency Stop System Core Logic
# Provides reusable emergency stop functionality

import rospy
import time
from threading import Lock


class EmergencyStopSystem:
    """
    Core emergency stop system logic
    
    Provides reusable emergency stop functionality that can be integrated
    into different nodes or used standalone.
    """
    
    def __init__(self, max_response_time=0.1, min_distance_threshold=0.3):
        """
        Initialize emergency stop system
        
        Args:
            max_response_time (float): Maximum allowed response time in seconds
            min_distance_threshold (float): Minimum safe distance in meters
        """
        self.max_response_time = max_response_time
        self.min_distance_threshold = min_distance_threshold
        
        self.emergency_active = False
        self.emergency_trigger_time = None
        self.emergency_reason = ""
        
        self.state_lock = Lock()
    
    def check_emergency_conditions(self, collision_risk=None, system_health=None, manual_trigger=False):
        """
        Check if emergency conditions are met
        
        Args:
            collision_risk (dict): Current collision risk assessment
            system_health (dict): Current system health status
            manual_trigger (bool): Manual emergency trigger
            
        Returns:
            bool: True if emergency stop should be triggered
        """
        with self.state_lock:
            # Check manual trigger
            if manual_trigger:
                return True
            
            # Check collision risk
            if collision_risk and collision_risk.get('level') in ['HIGH', 'CRITICAL']:
                return True
            
            # Check system health
            if system_health and not system_health.get('healthy', True):
                return True
            
            return False
    
    def trigger_emergency_stop(self, reason="unknown"):
        """
        Trigger emergency stop
        
        Args:
            reason (str): Reason for emergency stop
            
        Returns:
            dict: Emergency stop response information
        """
        with self.state_lock:
            if self.emergency_active:
                return {"already_active": True}
            
            trigger_time = rospy.Time.now()
            self.emergency_active = True
            self.emergency_trigger_time = trigger_time
            self.emergency_reason = reason
            
            response_time = (rospy.Time.now() - trigger_time).to_sec()
            
            return {
                "triggered": True,
                "reason": reason,
                "trigger_time": trigger_time,
                "response_time": response_time
            }
    
    def reset_emergency_system(self):
        """
        Reset emergency system to normal operation
        
        Returns:
            bool: True if reset successful
        """
        with self.state_lock:
            if not self.emergency_active:
                return False
            
            self.emergency_active = False
            self.emergency_trigger_time = None
            self.emergency_reason = ""
            
            return True
    
    def is_emergency_active(self):
        """
        Check if emergency is currently active
        
        Returns:
            bool: True if emergency is active
        """
        with self.state_lock:
            return self.emergency_active
    
    def get_emergency_info(self):
        """
        Get current emergency information
        
        Returns:
            dict: Emergency information
        """
        with self.state_lock:
            return {
                "active": self.emergency_active,
                "reason": self.emergency_reason,
                "trigger_time": self.emergency_trigger_time
            }