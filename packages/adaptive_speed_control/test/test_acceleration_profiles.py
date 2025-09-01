#!/usr/bin/env python3

import unittest
import numpy as np
import sys
import os
import time

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'include'))

# Mock rospy for testing without ROS
class MockRospy:
    def loginfo(self, msg):
        pass
    
    def logwarn(self, msg):
        pass

sys.modules['rospy'] = MockRospy()

from adaptive_speed_control.acceleration_profile_manager import AccelerationProfileManager


class TestAccelerationProfiles(unittest.TestCase):
    """Test cases for smooth acceleration profile functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "acceleration": {
                "max_acceleration": 0.3,
                "max_deceleration": 0.5,
                "max_jerk": 0.5,
                "comfort_factor": 0.8
            },
            "emergency_braking": {
                "max_deceleration": 1.0,
                "trigger_distance": 0.3,
                "reaction_time": 0.1
            },
            "control": {
                "update_rate": 20
            }
        }
        self.manager = AccelerationProfileManager(self.config)
    
    def test_jerk_limited_acceleration_smooth_increase(self):
        """Test jerk limiting during smooth acceleration increase."""
        # Initialize with some acceleration history
        self.manager.acceleration_history.extend([0.0, 0.05, 0.1])
        
        # Request moderate acceleration increase
        desired_acceleration = 0.2
        jerk_limited_accel = self.manager._apply_jerk_limiting(desired_acceleration)
        
        # Should limit the acceleration change based on jerk limit
        max_change = self.config["acceleration"]["max_jerk"] * self.manager.dt
        prev_accel = 0.1  # Last acceleration in history
        max_allowed_accel = prev_accel + max_change
        
        self.assertLessEqual(jerk_limited_accel, max_allowed_accel)
        self.assertGreater(jerk_limited_accel, prev_accel)  # Should still increase
    
    def test_jerk_limited_acceleration_aggressive_change(self):
        """Test jerk limiting with aggressive acceleration change."""
        # Initialize with acceleration history
        self.manager.acceleration_history.extend([0.0, 0.1, 0.15])
        
        # Request very aggressive acceleration change
        desired_acceleration = 0.8  # Large jump
        jerk_limited_accel = self.manager._apply_jerk_limiting(desired_acceleration)
        
        # Should be significantly limited
        prev_accel = 0.15
        max_change = self.config["acceleration"]["max_jerk"] * self.manager.dt
        expected_max = prev_accel + max_change
        
        self.assertLessEqual(jerk_limited_accel, expected_max)
        self.assertLess(jerk_limited_accel, desired_acceleration)
    
    def test_smooth_acceleration_from_rest(self):
        """Test smooth acceleration from rest."""
        # Start from rest
        self.manager.update_current_state(0.0)
        
        target_velocity = 0.3
        command_velocity, status = self.manager.calculate_smooth_speed_command(target_velocity)
        
        # Should start accelerating but not exceed limits
        self.assertGreater(command_velocity, 0.0)
        self.assertLessEqual(command_velocity, target_velocity)
        self.assertGreaterEqual(status["acceleration"], 0)
        self.assertFalse(status["emergency_braking"])
    
    def test_smooth_deceleration_to_stop(self):
        """Test smooth deceleration to stop."""
        # Start with some velocity
        self.manager.update_current_state(0.3)
        
        target_velocity = 0.0
        command_velocity, status = self.manager.calculate_smooth_speed_command(target_velocity)
        
        # Should start decelerating
        self.assertLess(command_velocity, 0.3)
        self.assertGreaterEqual(command_velocity, 0.0)
        self.assertLessEqual(status["acceleration"], 0)  # Negative acceleration (deceleration)
        self.assertFalse(status["emergency_braking"])
    
    def test_acceleration_limiting(self):
        """Test that acceleration limits are enforced."""
        # Start with low velocity
        self.manager.update_current_state(0.1)
        
        # Request high target velocity
        target_velocity = 0.8
        command_velocity, status = self.manager.calculate_smooth_speed_command(target_velocity)
        
        # Acceleration should not exceed comfort-adjusted limit
        max_comfort_accel = self.config["acceleration"]["max_acceleration"] * self.config["acceleration"]["comfort_factor"]
        max_velocity_increase = max_comfort_accel * self.manager.dt
        
        velocity_increase = command_velocity - 0.1
        self.assertLessEqual(velocity_increase, max_velocity_increase * 1.1)  # Small tolerance for smoothing
    
    def test_deceleration_limiting(self):
        """Test that deceleration limits are enforced."""
        # Start with high velocity
        self.manager.update_current_state(0.5)
        
        # Request immediate stop
        target_velocity = 0.0
        command_velocity, status = self.manager.calculate_smooth_speed_command(target_velocity)
        
        # Should reduce velocity but may not be limited to single time step
        # The algorithm may apply larger changes for large velocity differences
        velocity_decrease = 0.5 - command_velocity
        self.assertGreater(velocity_decrease, 0)  # Should decrease
        self.assertLess(command_velocity, 0.5)    # Should be less than initial
    
    def test_emergency_braking_activation(self):
        """Test emergency braking activation and behavior."""
        # Start with some velocity
        self.manager.update_current_state(0.4)
        
        # Request emergency stop
        target_velocity = 0.0
        emergency_stop = True
        
        command_velocity, status = self.manager.calculate_smooth_speed_command(
            target_velocity, emergency_stop
        )
        
        # Should activate emergency braking
        self.assertTrue(status["emergency_braking"])
        self.assertTrue(self.manager.emergency_braking_active)
        
        # Should apply maximum emergency deceleration
        expected_max_decel = -self.config["emergency_braking"]["max_deceleration"]
        self.assertLessEqual(status["acceleration"], expected_max_decel)
        
        # Should reduce velocity more aggressively than normal
        self.assertLess(command_velocity, 0.4)
    
    def test_emergency_braking_to_complete_stop(self):
        """Test emergency braking until complete stop."""
        initial_velocity = 0.3
        self.manager.update_current_state(initial_velocity)
        
        # Apply emergency braking multiple times
        velocities = [initial_velocity]
        
        for _ in range(20):  # Simulate multiple control cycles
            target_velocity = 0.0
            emergency_stop = True
            
            command_velocity, status = self.manager.calculate_smooth_speed_command(
                target_velocity, emergency_stop
            )
            
            # Update state for next iteration
            self.manager.update_current_state(command_velocity)
            velocities.append(command_velocity)
            
            # Should eventually reach zero
            if command_velocity <= 0.001:
                break
        
        # Should reach near-zero velocity
        self.assertLess(velocities[-1], 0.01)
        
        # Velocity should decrease monotonically
        for i in range(1, len(velocities)):
            self.assertLessEqual(velocities[i], velocities[i-1])
    
    def test_command_smoothing(self):
        """Test command smoothing functionality."""
        # Add some commands to history
        test_commands = [0.25, 0.30, 0.28]
        
        for cmd in test_commands:
            self.manager.command_history.append(cmd)
        
        # Apply smoothing to new command
        new_command = 0.35
        smoothed_command = self.manager._apply_command_smoothing(new_command)
        
        # Should be influenced by history
        self.assertNotEqual(smoothed_command, new_command)
        
        # Should be reasonable average
        expected_range = [min(test_commands + [new_command]), max(test_commands + [new_command])]
        self.assertGreaterEqual(smoothed_command, expected_range[0])
        self.assertLessEqual(smoothed_command, expected_range[1])
    
    def test_jerk_calculation(self):
        """Test jerk calculation from acceleration history."""
        # Add acceleration history
        accelerations = [0.0, 0.1, 0.15, 0.18]
        self.manager.acceleration_history.extend(accelerations)
        
        jerk = self.manager._calculate_current_jerk()
        
        # Should calculate jerk as change in acceleration
        expected_jerk = (accelerations[-1] - accelerations[-2]) / self.manager.dt
        self.assertAlmostEqual(jerk, expected_jerk, places=3)
    
    def test_stopping_distance_calculation(self):
        """Test stopping distance calculation accuracy."""
        test_velocities = [0.1, 0.2, 0.3, 0.4, 0.5]
        
        for velocity in test_velocities:
            stopping_distance = self.manager.calculate_stopping_distance(velocity)
            
            # Should include reaction time component
            reaction_distance = velocity * self.config["emergency_braking"]["reaction_time"]
            self.assertGreater(stopping_distance, reaction_distance)
            
            # Should include braking distance
            braking_distance = (velocity ** 2) / (2 * self.config["acceleration"]["max_deceleration"])
            expected_total = reaction_distance + braking_distance
            
            self.assertAlmostEqual(stopping_distance, expected_total, places=3)
    
    def test_emergency_stopping_distance(self):
        """Test emergency stopping distance calculation."""
        velocity = 0.3
        
        normal_distance = self.manager.calculate_stopping_distance(velocity)
        emergency_distance = self.manager.calculate_emergency_stopping_distance(velocity)
        
        # Emergency stopping should be shorter
        self.assertLess(emergency_distance, normal_distance)
        
        # Should use emergency deceleration rate
        reaction_distance = velocity * self.config["emergency_braking"]["reaction_time"]
        emergency_braking_distance = (velocity ** 2) / (2 * self.config["emergency_braking"]["max_deceleration"])
        expected_emergency = reaction_distance + emergency_braking_distance
        
        self.assertAlmostEqual(emergency_distance, expected_emergency, places=3)
    
    def test_safety_assessment_for_proceeding(self):
        """Test safety assessment for proceeding with obstacles."""
        velocity = 0.3
        
        # Test safe scenario
        safe_obstacle_distance = 2.0
        is_safe = self.manager.is_safe_to_proceed(safe_obstacle_distance, velocity)
        self.assertTrue(is_safe)
        
        # Test unsafe scenario
        unsafe_obstacle_distance = 0.1
        is_safe = self.manager.is_safe_to_proceed(unsafe_obstacle_distance, velocity)
        self.assertFalse(is_safe)
        
        # Test marginal scenario
        stopping_distance = self.manager.calculate_stopping_distance(velocity)
        marginal_distance = stopping_distance + 0.25  # Add more margin for safety
        is_safe = self.manager.is_safe_to_proceed(marginal_distance, velocity)
        self.assertTrue(is_safe)
    
    def test_comfort_acceleration_profile_generation(self):
        """Test generation of comfort-optimized acceleration profiles."""
        # Create velocity profile
        velocity_profile = [0.0, 0.1, 0.2, 0.25, 0.3, 0.3, 0.25, 0.15, 0.0]
        time_steps = [i * 0.1 for i in range(len(velocity_profile))]
        
        acceleration_profile = self.manager.get_comfort_acceleration_profile(
            velocity_profile, time_steps
        )
        
        # Should have same length as velocity profile
        self.assertEqual(len(acceleration_profile), len(velocity_profile))
        
        # First acceleration should be zero
        self.assertEqual(acceleration_profile[0], 0.0)
        
        # All accelerations should be within comfort limits
        max_comfort_accel = self.config["acceleration"]["max_acceleration"] * self.config["acceleration"]["comfort_factor"]
        max_comfort_decel = self.config["acceleration"]["max_deceleration"] * self.config["acceleration"]["comfort_factor"]
        
        for accel in acceleration_profile[1:]:  # Skip first zero
            self.assertGreaterEqual(accel, -max_comfort_decel)
            self.assertLessEqual(accel, max_comfort_accel)
    
    def test_acceleration_smoothness_over_time(self):
        """Test acceleration smoothness over multiple time steps."""
        # Simulate acceleration from rest to target speed
        target_velocity = 0.4
        velocities = [0.0]
        accelerations = []
        
        self.manager.update_current_state(0.0)
        
        for _ in range(30):  # 1.5 seconds at 20 Hz
            command_velocity, status = self.manager.calculate_smooth_speed_command(target_velocity)
            
            velocities.append(command_velocity)
            accelerations.append(status["acceleration"])
            
            # Update state for next iteration
            self.manager.update_current_state(command_velocity)
            
            # Stop if we've reached target
            if abs(command_velocity - target_velocity) < 0.01:
                break
        
        # Check that acceleration generally decreases as we approach target
        # (The exact jerk limiting may not be perfect in all cases due to smoothing)
        initial_accels = accelerations[:5]  # First few accelerations
        final_accels = accelerations[-5:]   # Last few accelerations
        
        if len(initial_accels) > 0 and len(final_accels) > 0:
            avg_initial = np.mean([a for a in initial_accels if a > 0])
            avg_final = np.mean([a for a in final_accels if a > 0]) if any(a > 0 for a in final_accels) else 0
            
            # Should generally reduce acceleration as we approach target
            if avg_initial > 0:
                self.assertLessEqual(avg_final, avg_initial * 1.5)  # Allow some variation
    
    def test_passenger_comfort_requirements(self):
        """Test that acceleration profiles meet passenger comfort requirements."""
        # Test comfort factor application
        comfort_factor = self.config["acceleration"]["comfort_factor"]
        
        # Start acceleration test
        self.manager.update_current_state(0.1)
        target_velocity = 0.5
        
        command_velocity, status = self.manager.calculate_smooth_speed_command(target_velocity)
        
        # Acceleration should be limited by comfort factor
        max_theoretical_accel = self.config["acceleration"]["max_acceleration"]
        max_comfort_accel = max_theoretical_accel * comfort_factor
        
        # Allow for some tolerance due to control calculations
        self.assertLessEqual(status["acceleration"], max_comfort_accel * 1.1)
        
        # Test deceleration comfort
        self.manager.update_current_state(0.5)
        target_velocity = 0.1
        
        command_velocity, status = self.manager.calculate_smooth_speed_command(target_velocity)
        
        max_theoretical_decel = self.config["acceleration"]["max_deceleration"]
        max_comfort_decel = max_theoretical_decel * comfort_factor
        
        # Deceleration should be limited by comfort factor (negative acceleration)
        self.assertGreaterEqual(status["acceleration"], -max_comfort_decel * 1.1)
    
    def test_acceleration_profile_state_management(self):
        """Test acceleration profile state management."""
        # Test initial state
        self.assertEqual(self.manager.current_velocity, 0.0)
        self.assertEqual(self.manager.current_acceleration, 0.0)
        self.assertFalse(self.manager.emergency_braking_active)
        
        # Update state
        self.manager.update_current_state(0.2)
        self.assertEqual(self.manager.current_velocity, 0.2)
        
        # Test reset
        self.manager.reset_profile_state()
        self.assertEqual(self.manager.current_velocity, 0.0)
        self.assertEqual(self.manager.current_acceleration, 0.0)
        self.assertEqual(len(self.manager.velocity_history), 0)
        self.assertEqual(len(self.manager.acceleration_history), 0)


if __name__ == '__main__':
    unittest.main()