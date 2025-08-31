# Collision Detection Manager Core Logic
# Provides reusable collision detection functionality

import numpy as np
import math
from threading import Lock


class CollisionDetectionManager:
    """
    Core collision detection logic
    
    Provides reusable collision detection and risk assessment functionality.
    """
    
    def __init__(self, min_safe_distance=0.3, time_horizon=3.0):
        """
        Initialize collision detection manager
        
        Args:
            min_safe_distance (float): Minimum safe distance in meters
            time_horizon (float): Time horizon for predictions in seconds
        """
        self.min_safe_distance = min_safe_distance
        self.time_horizon = time_horizon
        
        self.detected_objects = []
        self.risk_history = []
        
        self.state_lock = Lock()
        
        # Risk level constants
        self.RISK_NONE = 0
        self.RISK_LOW = 1
        self.RISK_MEDIUM = 2
        self.RISK_HIGH = 3
        self.RISK_CRITICAL = 4
    
    def update_detections(self, detections):
        """
        Update object detections
        
        Args:
            detections (list): List of detected objects
        """
        with self.state_lock:
            self.detected_objects = detections
    
    def assess_collision_risk(self, current_velocity=None):
        """
        Assess collision risk based on current detections
        
        Args:
            current_velocity (dict): Current vehicle velocity
            
        Returns:
            dict: Collision risk assessment
        """
        with self.state_lock:
            if not self.detected_objects:
                return {
                    'risk_level': self.RISK_NONE,
                    'time_to_collision': -1,
                    'recommended_action': 'CONTINUE'
                }
            
            max_risk = self.RISK_NONE
            min_ttc = float('inf')
            
            for obj in self.detected_objects:
                risk_info = self._assess_object_risk(obj, current_velocity)
                
                if risk_info['risk_level'] > max_risk:
                    max_risk = risk_info['risk_level']
                
                if risk_info['time_to_collision'] < min_ttc:
                    min_ttc = risk_info['time_to_collision']
            
            return {
                'risk_level': max_risk,
                'time_to_collision': min_ttc if min_ttc != float('inf') else -1,
                'recommended_action': self._get_recommended_action(max_risk)
            }
    
    def _assess_object_risk(self, obj, current_velocity):
        """
        Assess risk for a single object
        
        Args:
            obj (dict): Object information
            current_velocity (dict): Current velocity
            
        Returns:
            dict: Risk assessment for object
        """
        distance = obj.get('distance', float('inf'))
        
        # Distance-based risk
        if distance < 0.15:
            risk_level = self.RISK_CRITICAL
        elif distance < self.min_safe_distance:
            risk_level = self.RISK_HIGH
        elif distance < self.min_safe_distance * 1.5:
            risk_level = self.RISK_MEDIUM
        elif distance < self.min_safe_distance * 2.0:
            risk_level = self.RISK_LOW
        else:
            risk_level = self.RISK_NONE
        
        # Time to collision calculation
        ttc = float('inf')
        if current_velocity and current_velocity.get('v', 0) > 0.01:
            ttc = distance / current_velocity['v']
            
            # Adjust risk based on TTC
            if ttc < 1.0 and risk_level < self.RISK_HIGH:
                risk_level = self.RISK_HIGH
            elif ttc < 2.0 and risk_level < self.RISK_MEDIUM:
                risk_level = self.RISK_MEDIUM
        
        return {
            'risk_level': risk_level,
            'time_to_collision': ttc
        }
    
    def _get_recommended_action(self, risk_level):
        """
        Get recommended action based on risk level
        
        Args:
            risk_level (int): Current risk level
            
        Returns:
            str: Recommended action
        """
        if risk_level >= self.RISK_CRITICAL:
            return 'STOP'
        elif risk_level >= self.RISK_HIGH:
            return 'SLOW'
        elif risk_level >= self.RISK_MEDIUM:
            return 'CAUTION'
        else:
            return 'CONTINUE'
    
    def get_risk_statistics(self):
        """
        Get collision risk statistics
        
        Returns:
            dict: Risk statistics
        """
        with self.state_lock:
            return {
                'total_objects': len(self.detected_objects),
                'risk_history_length': len(self.risk_history)
            }