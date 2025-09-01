#!/usr/bin/env python3

import numpy as np
import rospy
from typing import Dict, List, Optional, Tuple
from collections import deque


class FollowingDistanceController:
    """
    Controls following distance and speed based on detected vehicles ahead.
    
    Implements time-based following distance calculation with adaptive speed control
    to maintain safe following distances while ensuring smooth speed transitions.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the following distance controller.
        
        Args:
            config (Dict): Configuration parameters
        """
        self.config = config
        
        # Following distance parameters
        self.time_based_distance = config.get("following_distance", {}).get("time_based_distance", 2.0)
        self.min_distance = config.get("following_distance", {}).get("min_distance", 0.5)
        self.max_distance = config.get("following_distance", {}).get("max_distance", 4.0)
        self.speed_reduction_factor = config.get("following_distance", {}).get("speed_reduction_factor", 0.5)
        
        # Control parameters
        self.update_rate = config.get("control", {}).get("update_rate", 20)
        self.smoothing_window = config.get("control", {}).get("smoothing_window", 5)
        self.prediction_horizon = config.get("control", {}).get("prediction_horizon", 1.0)
        
        # State tracking
        self.current_following_vehicle = None
        self.following_distance_history = deque(maxlen=self.smoothing_window)
        self.speed_command_history = deque(maxlen=self.smoothing_window)
        self.last_update_time = None
        
        # Vehicle tracking
        self.tracked_vehicles = {}
        self.vehicle_timeout = 2.0  # seconds
        
        rospy.loginfo("Following Distance Controller initialized")
    
    def update_vehicle_detections(self, detected_vehicles: List[Dict]) -> None:
        """
        Update tracked vehicles with new detections.
        
        Args:
            detected_vehicles (List[Dict]): List of detected vehicles with positions
        """
        current_time = rospy.Time.now().to_sec()
        
        # Update existing vehicles and add new ones
        for vehicle in detected_vehicles:
            vehicle_id = vehicle.get("id", f"vehicle_{len(self.tracked_vehicles)}")
            
            self.tracked_vehicles[vehicle_id] = {
                "position": vehicle.get("position", {"x": 0, "y": 0, "z": 0}),
                "distance": vehicle.get("distance", float('inf')),
                "velocity": vehicle.get("velocity", {"x": 0, "y": 0}),
                "confidence": vehicle.get("confidence", 0.5),
                "last_seen": current_time,
                "bbox": vehicle.get("bbox", {}),
                "relative_position": vehicle.get("relative_position", "unknown")
            }
        
        # Remove vehicles that haven't been seen recently
        vehicles_to_remove = []
        for vehicle_id, vehicle_data in self.tracked_vehicles.items():
            if current_time - vehicle_data["last_seen"] > self.vehicle_timeout:
                vehicles_to_remove.append(vehicle_id)
        
        for vehicle_id in vehicles_to_remove:
            del self.tracked_vehicles[vehicle_id]
    
    def find_leading_vehicle(self, current_velocity: float) -> Optional[Dict]:
        """
        Find the vehicle directly ahead that should be followed.
        
        Args:
            current_velocity (float): Current vehicle velocity in m/s
            
        Returns:
            Optional[Dict]: Leading vehicle data or None if no vehicle to follow
        """
        if not self.tracked_vehicles:
            return None
        
        # Filter vehicles that are ahead and in the same lane
        ahead_vehicles = []
        
        for vehicle_id, vehicle_data in self.tracked_vehicles.items():
            # Check if vehicle is ahead (positive x in robot frame)
            position = vehicle_data["position"]
            if position["x"] > 0:
                # Check if vehicle is in the same lane (small lateral offset)
                if abs(position["y"]) < 0.5:  # 0.5m lane tolerance
                    ahead_vehicles.append({
                        "id": vehicle_id,
                        "data": vehicle_data
                    })
        
        if not ahead_vehicles:
            return None
        
        # Find the closest vehicle ahead
        closest_vehicle = min(ahead_vehicles, key=lambda v: v["data"]["distance"])
        
        # Only follow if vehicle is within reasonable following range
        if closest_vehicle["data"]["distance"] <= self.max_distance:
            return closest_vehicle["data"]
        
        return None
    
    def calculate_desired_following_distance(self, current_velocity: float, 
                                           leading_vehicle_velocity: float = 0.0) -> float:
        """
        Calculate desired following distance based on current speed.
        
        Args:
            current_velocity (float): Current vehicle velocity in m/s
            leading_vehicle_velocity (float): Leading vehicle velocity in m/s
            
        Returns:
            float: Desired following distance in meters
        """
        # Time-based following distance
        time_based_dist = current_velocity * self.time_based_distance
        
        # Add extra distance if leading vehicle is slower (relative velocity consideration)
        relative_velocity = current_velocity - leading_vehicle_velocity
        if relative_velocity > 0:  # We're approaching
            # Add extra distance proportional to relative velocity
            extra_distance = relative_velocity * 0.5  # 0.5 second buffer
            time_based_dist += extra_distance
        
        # Ensure distance is within bounds
        desired_distance = np.clip(time_based_dist, self.min_distance, self.max_distance)
        
        return desired_distance
    
    def calculate_following_speed_command(self, current_velocity: float, 
                                        leading_vehicle: Dict,
                                        base_speed_command: float) -> Tuple[float, Dict]:
        """
        Calculate speed command for vehicle following behavior.
        
        Args:
            current_velocity (float): Current vehicle velocity in m/s
            leading_vehicle (Dict): Leading vehicle data
            base_speed_command (float): Base speed command without following adjustment
            
        Returns:
            Tuple[float, Dict]: Adjusted speed command and following status
        """
        current_distance = leading_vehicle["distance"]
        leading_velocity = np.sqrt(
            leading_vehicle["velocity"]["x"]**2 + leading_vehicle["velocity"]["y"]**2
        )
        
        # Calculate desired following distance
        desired_distance = self.calculate_desired_following_distance(
            current_velocity, leading_velocity
        )
        
        # Calculate distance error
        distance_error = current_distance - desired_distance
        
        # Smooth the distance measurements
        self.following_distance_history.append(current_distance)
        smoothed_distance = np.mean(list(self.following_distance_history))
        smoothed_error = smoothed_distance - desired_distance
        
        # Calculate speed adjustment based on distance error
        speed_adjustment = self._calculate_speed_adjustment(
            smoothed_error, current_velocity, leading_velocity
        )
        
        # Apply speed adjustment to base command
        adjusted_speed = base_speed_command + speed_adjustment
        
        # Ensure speed is non-negative and reasonable
        adjusted_speed = max(0.0, min(adjusted_speed, base_speed_command * 1.2))
        
        # Smooth speed commands
        self.speed_command_history.append(adjusted_speed)
        final_speed = np.mean(list(self.speed_command_history))
        
        # Following status information
        following_status = {
            "is_following": True,
            "leading_vehicle_id": leading_vehicle.get("id", "unknown"),
            "current_distance": current_distance,
            "desired_distance": desired_distance,
            "distance_error": distance_error,
            "speed_adjustment": speed_adjustment,
            "leading_vehicle_velocity": leading_velocity,
            "following_safety_factor": self._calculate_safety_factor(smoothed_error)
        }
        
        return final_speed, following_status
    
    def _calculate_speed_adjustment(self, distance_error: float, 
                                  current_velocity: float,
                                  leading_velocity: float) -> float:
        """
        Calculate speed adjustment based on following distance error.
        
        Args:
            distance_error (float): Difference between current and desired distance
            current_velocity (float): Current vehicle velocity
            leading_velocity (float): Leading vehicle velocity
            
        Returns:
            float: Speed adjustment in m/s (positive = speed up, negative = slow down)
        """
        # Proportional control based on distance error
        kp_distance = 0.3  # Proportional gain for distance error
        
        # Derivative control based on relative velocity
        kd_velocity = 0.2  # Derivative gain for velocity difference
        relative_velocity = current_velocity - leading_velocity
        
        # Calculate base adjustment
        proportional_adjustment = kp_distance * distance_error
        derivative_adjustment = -kd_velocity * relative_velocity
        
        speed_adjustment = proportional_adjustment + derivative_adjustment
        
        # Apply limits to prevent aggressive changes
        max_adjustment = current_velocity * 0.3  # Max 30% speed change
        speed_adjustment = np.clip(speed_adjustment, -max_adjustment, max_adjustment)
        
        # Additional safety: if too close, always reduce speed
        if distance_error < -0.2:  # 20cm too close
            safety_reduction = min(0.1, abs(distance_error) * 0.5)
            speed_adjustment = min(speed_adjustment, -safety_reduction)
        
        return speed_adjustment
    
    def _calculate_safety_factor(self, distance_error: float) -> float:
        """
        Calculate safety factor based on following distance.
        
        Args:
            distance_error (float): Distance error in meters
            
        Returns:
            float: Safety factor (0.0 = unsafe, 1.0 = very safe)
        """
        if distance_error >= 0:
            # At or beyond desired distance - safe
            return min(1.0, 0.8 + distance_error * 0.1)
        else:
            # Closer than desired distance - reduce safety factor
            safety_factor = 0.8 + distance_error * 0.5  # Linear decrease
            return max(0.1, safety_factor)
    
    def get_following_distance_command(self, current_velocity: float,
                                     detected_vehicles: List[Dict],
                                     base_speed_command: float) -> Tuple[float, Dict]:
        """
        Main function to get speed command considering following distance.
        
        Args:
            current_velocity (float): Current vehicle velocity in m/s
            detected_vehicles (List[Dict]): List of detected vehicles
            base_speed_command (float): Base speed command without following adjustment
            
        Returns:
            Tuple[float, Dict]: Final speed command and following status
        """
        # Update vehicle tracking
        self.update_vehicle_detections(detected_vehicles)
        
        # Find leading vehicle to follow
        leading_vehicle = self.find_leading_vehicle(current_velocity)
        
        if leading_vehicle is None:
            # No vehicle to follow, return base command
            self.current_following_vehicle = None
            following_status = {
                "is_following": False,
                "leading_vehicle_id": None,
                "current_distance": float('inf'),
                "desired_distance": 0.0,
                "distance_error": 0.0,
                "speed_adjustment": 0.0,
                "leading_vehicle_velocity": 0.0,
                "following_safety_factor": 1.0
            }
            return base_speed_command, following_status
        
        # Calculate following speed command
        self.current_following_vehicle = leading_vehicle
        return self.calculate_following_speed_command(
            current_velocity, leading_vehicle, base_speed_command
        )
    
    def is_following_vehicle(self) -> bool:
        """
        Check if currently following a vehicle.
        
        Returns:
            bool: True if following a vehicle
        """
        return self.current_following_vehicle is not None
    
    def get_current_following_distance(self) -> Optional[float]:
        """
        Get current following distance if following a vehicle.
        
        Returns:
            Optional[float]: Current following distance in meters or None
        """
        if self.current_following_vehicle:
            return self.current_following_vehicle["distance"]
        return None
    
    def emergency_stop_for_collision(self, collision_distance: float) -> bool:
        """
        Determine if emergency stop is needed based on collision distance.
        
        Args:
            collision_distance (float): Distance to potential collision in meters
            
        Returns:
            bool: True if emergency stop is needed
        """
        # Emergency stop if collision is imminent
        emergency_threshold = 0.3  # 30cm emergency threshold
        
        if collision_distance < emergency_threshold:
            rospy.logwarn(f"Emergency stop triggered: collision distance {collision_distance:.2f}m")
            return True
        
        return False
    
    def reset_following_state(self):
        """Reset following controller state."""
        self.current_following_vehicle = None
        self.tracked_vehicles.clear()
        self.following_distance_history.clear()
        self.speed_command_history.clear()
        self.last_update_time = None