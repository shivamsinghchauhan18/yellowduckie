#!/usr/bin/env python3

import numpy as np
import rospy
from typing import Dict, List, Optional, Tuple
from collections import deque
import time


class AccelerationProfileManager:
    """
    Manages smooth acceleration and deceleration profiles with jerk limiting.
    
    Ensures comfortable and safe speed transitions by limiting acceleration,
    deceleration, and jerk (rate of acceleration change) while providing
    emergency braking capabilities when needed.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the acceleration profile manager.
        
        Args:
            config (Dict): Configuration parameters
        """
        self.config = config
        
        # Acceleration limits
        self.max_acceleration = config.get("acceleration", {}).get("max_acceleration", 0.3)
        self.max_deceleration = config.get("acceleration", {}).get("max_deceleration", 0.5)
        self.max_jerk = config.get("acceleration", {}).get("max_jerk", 0.5)
        self.comfort_factor = config.get("acceleration", {}).get("comfort_factor", 0.8)
        
        # Emergency braking parameters
        self.emergency_max_deceleration = config.get("emergency_braking", {}).get("max_deceleration", 1.0)
        self.emergency_trigger_distance = config.get("emergency_braking", {}).get("trigger_distance", 0.3)
        self.reaction_time = config.get("emergency_braking", {}).get("reaction_time", 0.1)
        
        # Control parameters
        self.update_rate = config.get("control", {}).get("update_rate", 20)
        self.dt = 1.0 / self.update_rate
        
        # State tracking
        self.current_velocity = 0.0
        self.current_acceleration = 0.0
        self.target_velocity = 0.0
        self.last_update_time = None
        
        # History for smoothing and jerk calculation
        self.velocity_history = deque(maxlen=10)
        self.acceleration_history = deque(maxlen=5)
        self.command_history = deque(maxlen=3)
        
        # Emergency state
        self.emergency_braking_active = False
        self.emergency_start_time = None
        
        rospy.loginfo("Acceleration Profile Manager initialized")
    
    def update_current_state(self, current_velocity: float) -> None:
        """
        Update current velocity and calculate acceleration.
        
        Args:
            current_velocity (float): Current vehicle velocity in m/s
        """
        current_time = time.time()
        
        # Calculate time delta
        if self.last_update_time is not None:
            dt = current_time - self.last_update_time
        else:
            dt = self.dt
        
        # Calculate current acceleration
        if self.velocity_history:
            prev_velocity = self.velocity_history[-1]
            self.current_acceleration = (current_velocity - prev_velocity) / dt
        else:
            self.current_acceleration = 0.0
        
        # Update state
        self.current_velocity = current_velocity
        self.last_update_time = current_time
        
        # Update history
        self.velocity_history.append(current_velocity)
        self.acceleration_history.append(self.current_acceleration)
    
    def calculate_smooth_speed_command(self, target_velocity: float,
                                     emergency_stop: bool = False) -> Tuple[float, Dict]:
        """
        Calculate smooth speed command with acceleration and jerk limits.
        
        Args:
            target_velocity (float): Desired target velocity in m/s
            emergency_stop (bool): True if emergency stop is required
            
        Returns:
            Tuple[float, Dict]: Smooth speed command and profile status
        """
        self.target_velocity = target_velocity
        
        # Handle emergency braking
        if emergency_stop:
            return self._handle_emergency_braking()
        
        # Reset emergency state if not in emergency
        if self.emergency_braking_active:
            self.emergency_braking_active = False
            self.emergency_start_time = None
        
        # Calculate velocity difference
        velocity_error = target_velocity - self.current_velocity
        
        # Determine if accelerating or decelerating
        if velocity_error > 0:
            # Accelerating
            max_allowed_accel = self.max_acceleration * self.comfort_factor
            acceleration_command = min(velocity_error / self.dt, max_allowed_accel)
        elif velocity_error < 0:
            # Decelerating
            max_allowed_decel = self.max_deceleration * self.comfort_factor
            acceleration_command = max(velocity_error / self.dt, -max_allowed_decel)
        else:
            # At target velocity
            acceleration_command = 0.0
        
        # Apply jerk limiting
        jerk_limited_acceleration = self._apply_jerk_limiting(acceleration_command)
        
        # Calculate new velocity command
        new_velocity = self.current_velocity + jerk_limited_acceleration * self.dt
        
        # Ensure velocity is non-negative and within reasonable bounds
        new_velocity = max(0.0, min(new_velocity, target_velocity * 1.1))
        
        # Apply command smoothing
        smoothed_velocity = self._apply_command_smoothing(new_velocity)
        
        # Profile status
        profile_status = {
            "target_velocity": target_velocity,
            "current_velocity": self.current_velocity,
            "commanded_velocity": smoothed_velocity,
            "acceleration": jerk_limited_acceleration,
            "jerk": self._calculate_current_jerk(),
            "emergency_braking": False,
            "acceleration_limited": abs(acceleration_command) > abs(jerk_limited_acceleration),
            "jerk_limited": self._is_jerk_limited(acceleration_command, jerk_limited_acceleration),
            "comfort_factor_applied": self.comfort_factor < 1.0
        }
        
        return smoothed_velocity, profile_status
    
    def _handle_emergency_braking(self) -> Tuple[float, Dict]:
        """
        Handle emergency braking with maximum deceleration.
        
        Returns:
            Tuple[float, Dict]: Emergency stop command and status
        """
        if not self.emergency_braking_active:
            self.emergency_braking_active = True
            self.emergency_start_time = time.time()
            rospy.logwarn("Emergency braking activated!")
        
        # Calculate emergency deceleration
        emergency_decel = -self.emergency_max_deceleration
        
        # Calculate new velocity with emergency deceleration
        new_velocity = self.current_velocity + emergency_decel * self.dt
        
        # Ensure we don't go negative
        new_velocity = max(0.0, new_velocity)
        
        # Emergency status
        emergency_status = {
            "target_velocity": 0.0,
            "current_velocity": self.current_velocity,
            "commanded_velocity": new_velocity,
            "acceleration": emergency_decel,
            "jerk": self._calculate_current_jerk(),
            "emergency_braking": True,
            "emergency_duration": time.time() - self.emergency_start_time if self.emergency_start_time else 0.0,
            "acceleration_limited": False,
            "jerk_limited": False,
            "comfort_factor_applied": False
        }
        
        return new_velocity, emergency_status
    
    def _apply_jerk_limiting(self, desired_acceleration: float) -> float:
        """
        Apply jerk limiting to acceleration command.
        
        Args:
            desired_acceleration (float): Desired acceleration in m/s²
            
        Returns:
            float: Jerk-limited acceleration in m/s²
        """
        if not self.acceleration_history:
            return desired_acceleration
        
        # Calculate maximum allowed acceleration change (jerk limit)
        max_accel_change = self.max_jerk * self.dt
        
        # Get previous acceleration
        prev_acceleration = self.acceleration_history[-1]
        
        # Calculate desired acceleration change
        accel_change = desired_acceleration - prev_acceleration
        
        # Limit acceleration change
        limited_accel_change = np.clip(accel_change, -max_accel_change, max_accel_change)
        
        # Calculate final acceleration
        final_acceleration = prev_acceleration + limited_accel_change
        
        return final_acceleration
    
    def _apply_command_smoothing(self, velocity_command: float) -> float:
        """
        Apply smoothing to velocity commands to reduce noise.
        
        Args:
            velocity_command (float): Raw velocity command
            
        Returns:
            float: Smoothed velocity command
        """
        # Add to command history
        self.command_history.append(velocity_command)
        
        # Apply moving average smoothing
        if len(self.command_history) >= 3:
            # Weighted average with more weight on recent commands
            weights = [0.2, 0.3, 0.5]  # Oldest to newest
            smoothed = sum(w * cmd for w, cmd in zip(weights, self.command_history))
        else:
            smoothed = velocity_command
        
        return smoothed
    
    def _calculate_current_jerk(self) -> float:
        """
        Calculate current jerk (rate of acceleration change).
        
        Returns:
            float: Current jerk in m/s³
        """
        if len(self.acceleration_history) < 2:
            return 0.0
        
        # Calculate jerk as change in acceleration over time
        accel_change = self.acceleration_history[-1] - self.acceleration_history[-2]
        jerk = accel_change / self.dt
        
        return jerk
    
    def _is_jerk_limited(self, desired_accel: float, actual_accel: float) -> bool:
        """
        Check if jerk limiting was applied.
        
        Args:
            desired_accel (float): Desired acceleration
            actual_accel (float): Actual applied acceleration
            
        Returns:
            bool: True if jerk limiting was applied
        """
        return abs(desired_accel - actual_accel) > 0.01  # 0.01 m/s² threshold
    
    def calculate_stopping_distance(self, current_velocity: float,
                                  deceleration: Optional[float] = None) -> float:
        """
        Calculate stopping distance for given velocity and deceleration.
        
        Args:
            current_velocity (float): Current velocity in m/s
            deceleration (Optional[float]): Deceleration rate, uses max if None
            
        Returns:
            float: Stopping distance in meters
        """
        if deceleration is None:
            deceleration = self.max_deceleration
        
        # Add reaction time distance
        reaction_distance = current_velocity * self.reaction_time
        
        # Calculate braking distance: v² / (2 * a)
        braking_distance = (current_velocity ** 2) / (2 * deceleration)
        
        total_stopping_distance = reaction_distance + braking_distance
        
        return total_stopping_distance
    
    def calculate_emergency_stopping_distance(self, current_velocity: float) -> float:
        """
        Calculate emergency stopping distance with maximum deceleration.
        
        Args:
            current_velocity (float): Current velocity in m/s
            
        Returns:
            float: Emergency stopping distance in meters
        """
        return self.calculate_stopping_distance(current_velocity, self.emergency_max_deceleration)
    
    def is_safe_to_proceed(self, obstacle_distance: float,
                          current_velocity: float) -> bool:
        """
        Check if it's safe to proceed given obstacle distance.
        
        Args:
            obstacle_distance (float): Distance to obstacle in meters
            current_velocity (float): Current velocity in m/s
            
        Returns:
            bool: True if safe to proceed
        """
        stopping_distance = self.calculate_stopping_distance(current_velocity)
        safety_margin = 0.2  # 20cm safety margin
        
        return obstacle_distance > (stopping_distance + safety_margin)
    
    def get_comfort_acceleration_profile(self, velocity_profile: List[float],
                                       time_steps: List[float]) -> List[float]:
        """
        Generate comfort-optimized acceleration profile for velocity trajectory.
        
        Args:
            velocity_profile (List[float]): Desired velocity profile
            time_steps (List[float]): Time steps for profile
            
        Returns:
            List[float]: Comfort-optimized acceleration profile
        """
        if len(velocity_profile) < 2 or len(time_steps) != len(velocity_profile):
            return [0.0] * len(velocity_profile)
        
        acceleration_profile = []
        
        for i in range(len(velocity_profile)):
            if i == 0:
                # First point, use zero acceleration
                acceleration_profile.append(0.0)
            else:
                # Calculate desired acceleration
                dt = time_steps[i] - time_steps[i-1]
                dv = velocity_profile[i] - velocity_profile[i-1]
                desired_accel = dv / dt if dt > 0 else 0.0
                
                # Apply comfort and jerk limits
                if i > 1:
                    prev_accel = acceleration_profile[i-1]
                    max_accel_change = self.max_jerk * dt
                    accel_change = desired_accel - prev_accel
                    limited_change = np.clip(accel_change, -max_accel_change, max_accel_change)
                    comfort_accel = prev_accel + limited_change
                else:
                    comfort_accel = desired_accel
                
                # Apply acceleration limits
                max_comfort_accel = self.max_acceleration * self.comfort_factor
                max_comfort_decel = self.max_deceleration * self.comfort_factor
                
                comfort_accel = np.clip(comfort_accel, -max_comfort_decel, max_comfort_accel)
                acceleration_profile.append(comfort_accel)
        
        return acceleration_profile
    
    def reset_profile_state(self):
        """Reset acceleration profile manager state."""
        self.current_velocity = 0.0
        self.current_acceleration = 0.0
        self.target_velocity = 0.0
        self.last_update_time = None
        self.velocity_history.clear()
        self.acceleration_history.clear()
        self.command_history.clear()
        self.emergency_braking_active = False
        self.emergency_start_time = None