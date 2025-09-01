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
    
    class Time:
        @staticmethod
        def now():
            class MockTime:
                def to_sec(self):
                    return time.time()
            return MockTime()

sys.modules['rospy'] = MockRospy()

from adaptive_speed_control.following_distance_controller import FollowingDistanceController


class TestFollowingDistanceControl(unittest.TestCase):
    """Test cases for following distance control functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "following_distance": {
                "time_based_distance": 2.0,
                "min_distance": 0.5,
                "max_distance": 4.0,
                "speed_reduction_factor": 0.5
            },
            "control": {
                "update_rate": 20,
                "smoothing_window": 5,
                "prediction_horizon": 1.0
            }
        }
        self.controller = FollowingDistanceController(self.config)
    
    def test_time_based_following_distance_calculation(self):
        """Test time-based following distance calculation."""
        # Test various speeds
        test_speeds = [0.1, 0.2, 0.3, 0.4, 0.5]
        time_distance = self.config["following_distance"]["time_based_distance"]
        
        for speed in test_speeds:
            desired_distance = self.controller.calculate_desired_following_distance(speed)
            expected_min_distance = speed * time_distance
            
            # Should be at least the time-based distance
            self.assertGreaterEqual(desired_distance, expected_min_distance)
            
            # Should be within bounds
            self.assertGreaterEqual(desired_distance, self.config["following_distance"]["min_distance"])
            self.assertLessEqual(desired_distance, self.config["following_distance"]["max_distance"])
    
    def test_following_distance_with_relative_velocity(self):
        """Test following distance calculation with relative velocity consideration."""
        current_velocity = 0.3
        
        # Test with slower leading vehicle (we're approaching)
        slower_leading_velocity = 0.2
        distance_approaching = self.controller.calculate_desired_following_distance(
            current_velocity, slower_leading_velocity
        )
        
        # Test with same speed leading vehicle
        same_leading_velocity = 0.3
        distance_same_speed = self.controller.calculate_desired_following_distance(
            current_velocity, same_leading_velocity
        )
        
        # Should require larger distance when approaching
        self.assertGreater(distance_approaching, distance_same_speed)
    
    def test_vehicle_tracking_and_timeout(self):
        """Test vehicle tracking and timeout functionality."""
        # Add test vehicles
        detected_vehicles = [
            {
                "id": "vehicle_1",
                "position": {"x": 2.0, "y": 0.1, "z": 0.0},
                "distance": 2.0,
                "velocity": {"x": 0.2, "y": 0.0},
                "confidence": 0.8
            }
        ]
        
        self.controller.update_vehicle_detections(detected_vehicles)
        
        # Should have tracked the vehicle
        self.assertEqual(len(self.controller.tracked_vehicles), 1)
        self.assertIn("vehicle_1", self.controller.tracked_vehicles)
        
        # Simulate timeout by setting old timestamp
        self.controller.tracked_vehicles["vehicle_1"]["last_seen"] = time.time() - 5.0
        
        # Update with empty list (no new detections)
        self.controller.update_vehicle_detections([])
        
        # Vehicle should be removed due to timeout
        self.assertEqual(len(self.controller.tracked_vehicles), 0)
    
    def test_leading_vehicle_identification(self):
        """Test identification of leading vehicle to follow."""
        # Set up multiple vehicles
        self.controller.tracked_vehicles = {
            "vehicle_ahead_close": {
                "position": {"x": 1.5, "y": 0.0, "z": 0.0},
                "distance": 1.5,
                "velocity": {"x": 0.2, "y": 0.0},
                "confidence": 0.9,
                "last_seen": time.time()
            },
            "vehicle_ahead_far": {
                "position": {"x": 3.0, "y": 0.1, "z": 0.0},
                "distance": 3.0,
                "velocity": {"x": 0.3, "y": 0.0},
                "confidence": 0.8,
                "last_seen": time.time()
            },
            "vehicle_behind": {
                "position": {"x": -1.0, "y": 0.0, "z": 0.0},
                "distance": 1.0,
                "velocity": {"x": 0.1, "y": 0.0},
                "confidence": 0.7,
                "last_seen": time.time()
            },
            "vehicle_side": {
                "position": {"x": 1.0, "y": 1.0, "z": 0.0},
                "distance": 1.4,
                "velocity": {"x": 0.2, "y": 0.0},
                "confidence": 0.6,
                "last_seen": time.time()
            }
        }
        
        leading_vehicle = self.controller.find_leading_vehicle(0.3)
        
        # Should find the closest vehicle ahead in the same lane
        self.assertIsNotNone(leading_vehicle)
        self.assertEqual(leading_vehicle["distance"], 1.5)
    
    def test_following_speed_command_too_close(self):
        """Test speed command when following vehicle is too close."""
        current_velocity = 0.3
        base_speed = 0.3
        
        # Create leading vehicle that's too close
        leading_vehicle = {
            "distance": 0.4,  # Too close for 2-second following distance at 0.3 m/s
            "velocity": {"x": 0.2, "y": 0.0},
            "id": "close_vehicle"
        }
        
        adjusted_speed, status = self.controller.calculate_following_speed_command(
            current_velocity, leading_vehicle, base_speed
        )
        
        # Should reduce speed when too close
        self.assertLess(adjusted_speed, base_speed)
        self.assertTrue(status["is_following"])
        self.assertLess(status["distance_error"], 0)  # Negative error means too close
        self.assertLess(status["following_safety_factor"], 0.8)  # Reduced safety
    
    def test_following_speed_command_good_distance(self):
        """Test speed command when following distance is appropriate."""
        current_velocity = 0.3
        base_speed = 0.3
        
        # Create leading vehicle at good distance
        leading_vehicle = {
            "distance": 0.6,  # Good distance for 2-second following at 0.3 m/s
            "velocity": {"x": 0.3, "y": 0.0},
            "id": "good_distance_vehicle"
        }
        
        adjusted_speed, status = self.controller.calculate_following_speed_command(
            current_velocity, leading_vehicle, base_speed
        )
        
        # Should maintain similar speed at good distance
        self.assertAlmostEqual(adjusted_speed, base_speed, delta=0.1)
        self.assertTrue(status["is_following"])
        self.assertGreater(status["following_safety_factor"], 0.7)  # Good safety
    
    def test_following_speed_command_too_far(self):
        """Test speed command when following vehicle is too far."""
        current_velocity = 0.2
        base_speed = 0.3
        
        # Create leading vehicle that's far away
        leading_vehicle = {
            "distance": 1.0,  # Far for 2-second following at 0.2 m/s (should be 0.4)
            "velocity": {"x": 0.2, "y": 0.0},
            "id": "far_vehicle"
        }
        
        adjusted_speed, status = self.controller.calculate_following_speed_command(
            current_velocity, leading_vehicle, base_speed
        )
        
        # Should allow speed increase when too far
        self.assertGreaterEqual(adjusted_speed, current_velocity)
        self.assertTrue(status["is_following"])
        self.assertGreater(status["distance_error"], 0)  # Positive error means too far
    
    def test_speed_adjustment_calculation(self):
        """Test speed adjustment calculation logic."""
        # Test proportional control for distance error
        distance_error = -0.2  # 20cm too close
        current_velocity = 0.3
        leading_velocity = 0.2
        
        speed_adjustment = self.controller._calculate_speed_adjustment(
            distance_error, current_velocity, leading_velocity
        )
        
        # Should reduce speed when too close
        self.assertLess(speed_adjustment, 0)
        
        # Test with positive distance error (too far)
        distance_error = 0.3  # 30cm too far
        speed_adjustment = self.controller._calculate_speed_adjustment(
            distance_error, current_velocity, leading_velocity
        )
        
        # Should increase speed when too far
        self.assertGreater(speed_adjustment, 0)
    
    def test_safety_factor_calculation(self):
        """Test safety factor calculation based on following distance."""
        # Test safe distance (at or beyond desired)
        safe_distance_error = 0.1  # 10cm beyond desired distance
        safety_factor = self.controller._calculate_safety_factor(safe_distance_error)
        self.assertGreater(safety_factor, 0.8)
        
        # Test unsafe distance (too close)
        unsafe_distance_error = -0.3  # 30cm too close
        safety_factor = self.controller._calculate_safety_factor(unsafe_distance_error)
        self.assertLess(safety_factor, 0.7)
        
        # Safety factor should always be between 0 and 1
        self.assertGreaterEqual(safety_factor, 0.1)
        self.assertLessEqual(safety_factor, 1.0)
    
    def test_emergency_collision_detection(self):
        """Test emergency stop decision for collision scenarios."""
        # Test with imminent collision
        collision_distance = 0.2  # Very close
        emergency_needed = self.controller.emergency_stop_for_collision(collision_distance)
        self.assertTrue(emergency_needed)
        
        # Test with safe distance
        safe_distance = 1.0
        emergency_needed = self.controller.emergency_stop_for_collision(safe_distance)
        self.assertFalse(emergency_needed)
        
        # Test at threshold
        threshold_distance = 0.29  # Just below emergency threshold
        emergency_needed = self.controller.emergency_stop_for_collision(threshold_distance)
        self.assertTrue(emergency_needed)  # Should trigger below threshold
    
    def test_following_distance_command_integration(self):
        """Test integrated following distance command functionality."""
        current_velocity = 0.3
        base_speed_command = 0.3
        
        # Test with no vehicles
        detected_vehicles = []
        final_speed, status = self.controller.get_following_distance_command(
            current_velocity, detected_vehicles, base_speed_command
        )
        
        # Should return base command when no vehicles to follow
        self.assertEqual(final_speed, base_speed_command)
        self.assertFalse(status["is_following"])
        
        # Test with vehicles to follow
        detected_vehicles = [
            {
                "id": "vehicle_1",
                "position": {"x": 1.0, "y": 0.0, "z": 0.0},
                "distance": 1.0,
                "velocity": {"x": 0.2, "y": 0.0},
                "confidence": 0.8
            }
        ]
        
        final_speed, status = self.controller.get_following_distance_command(
            current_velocity, detected_vehicles, base_speed_command
        )
        
        # Should modify speed when following
        self.assertTrue(status["is_following"])
        self.assertIsNotNone(status["leading_vehicle_id"])
    
    def test_following_distance_smoothing(self):
        """Test smoothing of following distance measurements."""
        # Add multiple distance measurements
        distances = [1.0, 1.1, 0.9, 1.05, 0.95]
        
        for distance in distances:
            self.controller.following_distance_history.append(distance)
        
        # Should have smoothing effect
        smoothed_distance = np.mean(list(self.controller.following_distance_history))
        expected_smoothed = np.mean(distances)
        
        self.assertAlmostEqual(smoothed_distance, expected_smoothed, places=3)
    
    def test_vehicle_following_state_management(self):
        """Test following state management."""
        # Initially not following
        self.assertFalse(self.controller.is_following_vehicle())
        self.assertIsNone(self.controller.get_current_following_distance())
        
        # Set following vehicle
        self.controller.current_following_vehicle = {
            "distance": 1.5,
            "id": "test_vehicle"
        }
        
        # Should be following
        self.assertTrue(self.controller.is_following_vehicle())
        self.assertEqual(self.controller.get_current_following_distance(), 1.5)
        
        # Reset state
        self.controller.reset_following_state()
        self.assertFalse(self.controller.is_following_vehicle())
        self.assertIsNone(self.controller.get_current_following_distance())
    
    def test_following_distance_accuracy_requirements(self):
        """Test that following distance meets accuracy requirements."""
        # Test accuracy of time-based distance calculation
        test_velocities = [0.1, 0.2, 0.3, 0.4, 0.5]
        time_factor = 2.0  # 2-second following distance
        
        for velocity in test_velocities:
            desired_distance = self.controller.calculate_desired_following_distance(velocity)
            expected_distance = velocity * time_factor
            
            # Should be accurate within reasonable tolerance
            self.assertGreaterEqual(desired_distance, expected_distance)
            # Allow for minimum distance constraint
            max_expected = max(expected_distance * 1.5, self.config["following_distance"]["min_distance"])
            self.assertLessEqual(desired_distance, max_expected)
    
    def test_collision_avoidance_accuracy(self):
        """Test collision avoidance accuracy."""
        # Test various collision scenarios
        collision_scenarios = [
            {"distance": 0.1, "should_stop": True},   # Very close
            {"distance": 0.25, "should_stop": True},  # At threshold
            {"distance": 0.35, "should_stop": False}, # Just safe
            {"distance": 1.0, "should_stop": False}   # Safe
        ]
        
        for scenario in collision_scenarios:
            emergency_needed = self.controller.emergency_stop_for_collision(scenario["distance"])
            self.assertEqual(emergency_needed, scenario["should_stop"],
                           f"Failed for distance {scenario['distance']}")


if __name__ == '__main__':
    unittest.main()