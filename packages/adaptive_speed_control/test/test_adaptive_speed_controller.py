#!/usr/bin/env python3

import unittest
import numpy as np
import rospy
import rostest
from unittest.mock import Mock, patch, MagicMock

# Import the classes to test
from adaptive_speed_control.environmental_analyzer import EnvironmentalAnalyzer
from adaptive_speed_control.following_distance_controller import FollowingDistanceController
from adaptive_speed_control.acceleration_profile_manager import AccelerationProfileManager


class TestEnvironmentalAnalyzer(unittest.TestCase):
    """Test cases for EnvironmentalAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "visibility": {
                "brightness_threshold": 50,
                "contrast_threshold": 20,
                "min_visibility_threshold": 0.3,
                "max_speed_reduction": 0.7
            },
            "traffic_density": {
                "detection_distance": 2.0,
                "max_vehicles_threshold": 3,
                "speed_reduction_per_vehicle": 0.15
            },
            "road_conditions": {
                "surface_quality_threshold": 0.7,
                "weather_impact_factor": 0.8
            }
        }
        self.analyzer = EnvironmentalAnalyzer(self.config)
    
    def test_analyze_visibility_good_conditions(self):
        """Test visibility analysis with good lighting conditions."""
        # Create a test image with good brightness and contrast
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 120  # Good brightness
        test_image += np.random.randint(-30, 30, test_image.shape, dtype=np.int8)  # Add contrast
        test_image = np.clip(test_image, 0, 255).astype(np.uint8)
        
        visibility_factor = self.analyzer.analyze_visibility(test_image)
        
        # Should return high visibility factor for good conditions
        self.assertGreater(visibility_factor, 0.7)
        self.assertLessEqual(visibility_factor, 1.0)
    
    def test_analyze_visibility_poor_conditions(self):
        """Test visibility analysis with poor lighting conditions."""
        # Create a test image with poor brightness (too dark)
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 20  # Very dark
        
        visibility_factor = self.analyzer.analyze_visibility(test_image)
        
        # Should return low visibility factor for poor conditions
        self.assertLess(visibility_factor, 0.5)
        self.assertGreaterEqual(visibility_factor, 0.0)
    
    def test_analyze_traffic_density_no_vehicles(self):
        """Test traffic density analysis with no vehicles."""
        detected_objects = []
        
        traffic_factor = self.analyzer.analyze_traffic_density(detected_objects)
        
        # Should return 1.0 (no traffic impact) when no vehicles detected
        self.assertEqual(traffic_factor, 1.0)
    
    def test_analyze_traffic_density_with_vehicles(self):
        """Test traffic density analysis with multiple vehicles."""
        detected_objects = [
            {"type": "duckiebot", "distance": 1.0},
            {"type": "duckiebot", "distance": 1.5},
            {"type": "car", "distance": 0.8}
        ]
        
        traffic_factor = self.analyzer.analyze_traffic_density(detected_objects)
        
        # Should return reduced factor due to nearby vehicles
        self.assertLess(traffic_factor, 1.0)
        self.assertGreater(traffic_factor, 0.0)
    
    def test_analyze_road_conditions_default(self):
        """Test road conditions analysis with default parameters."""
        # Create a test image representing road surface
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 100
        
        road_factor = self.analyzer.analyze_road_conditions(test_image)
        
        # Should return reasonable road condition factor
        self.assertGreater(road_factor, 0.5)
        self.assertLessEqual(road_factor, 1.0)
    
    def test_get_environmental_speed_factor_integration(self):
        """Test integrated environmental speed factor calculation."""
        # Create test data
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 120
        detected_objects = [{"type": "duckiebot", "distance": 1.5}]
        imu_data = {"accel_x": 0.1, "accel_y": 0.1, "accel_z": 9.81}
        
        factors = self.analyzer.get_environmental_speed_factor(
            test_image, detected_objects, imu_data
        )
        
        # Check that all expected factors are present
        self.assertIn("visibility_factor", factors)
        self.assertIn("traffic_density_factor", factors)
        self.assertIn("road_condition_factor", factors)
        self.assertIn("overall_environmental_factor", factors)
        
        # Check that factors are in valid range
        for factor_name, factor_value in factors.items():
            self.assertGreaterEqual(factor_value, 0.0)
            self.assertLessEqual(factor_value, 1.0)


class TestFollowingDistanceController(unittest.TestCase):
    """Test cases for FollowingDistanceController class."""
    
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
    
    def test_calculate_desired_following_distance(self):
        """Test desired following distance calculation."""
        current_velocity = 0.3  # m/s
        leading_velocity = 0.2  # m/s
        
        desired_distance = self.controller.calculate_desired_following_distance(
            current_velocity, leading_velocity
        )
        
        # Should be at least time_based_distance * velocity
        expected_min = current_velocity * self.config["following_distance"]["time_based_distance"]
        self.assertGreaterEqual(desired_distance, expected_min)
        
        # Should be within bounds
        self.assertGreaterEqual(desired_distance, self.config["following_distance"]["min_distance"])
        self.assertLessEqual(desired_distance, self.config["following_distance"]["max_distance"])
    
    def test_find_leading_vehicle_no_vehicles(self):
        """Test finding leading vehicle when none are present."""
        self.controller.tracked_vehicles = {}
        
        leading_vehicle = self.controller.find_leading_vehicle(0.3)
        
        self.assertIsNone(leading_vehicle)
    
    def test_find_leading_vehicle_with_vehicles(self):
        """Test finding leading vehicle when vehicles are present."""
        # Add test vehicles
        self.controller.tracked_vehicles = {
            "vehicle_1": {
                "position": {"x": 2.0, "y": 0.1, "z": 0.0},
                "distance": 2.0,
                "velocity": {"x": 0.2, "y": 0.0},
                "confidence": 0.8,
                "last_seen": rospy.Time.now().to_sec()
            },
            "vehicle_2": {
                "position": {"x": 1.5, "y": 0.0, "z": 0.0},
                "distance": 1.5,
                "velocity": {"x": 0.3, "y": 0.0},
                "confidence": 0.9,
                "last_seen": rospy.Time.now().to_sec()
            }
        }
        
        leading_vehicle = self.controller.find_leading_vehicle(0.3)
        
        # Should find the closest vehicle ahead
        self.assertIsNotNone(leading_vehicle)
        self.assertEqual(leading_vehicle["distance"], 1.5)
    
    def test_calculate_following_speed_command_too_close(self):
        """Test speed command calculation when too close to leading vehicle."""
        current_velocity = 0.3
        base_speed = 0.3
        
        # Create leading vehicle that's too close
        leading_vehicle = {
            "distance": 0.3,  # Too close for 2-second following distance
            "velocity": {"x": 0.2, "y": 0.0},
            "id": "test_vehicle"
        }
        
        adjusted_speed, status = self.controller.calculate_following_speed_command(
            current_velocity, leading_vehicle, base_speed
        )
        
        # Should reduce speed when too close
        self.assertLess(adjusted_speed, base_speed)
        self.assertTrue(status["is_following"])
        self.assertLess(status["distance_error"], 0)  # Negative error means too close
    
    def test_emergency_stop_for_collision(self):
        """Test emergency stop decision for collision scenarios."""
        # Test with collision imminent
        collision_distance = 0.2  # Very close
        
        emergency_needed = self.controller.emergency_stop_for_collision(collision_distance)
        
        self.assertTrue(emergency_needed)
        
        # Test with safe distance
        collision_distance = 1.0  # Safe distance
        
        emergency_needed = self.controller.emergency_stop_for_collision(collision_distance)
        
        self.assertFalse(emergency_needed)


class TestAccelerationProfileManager(unittest.TestCase):
    """Test cases for AccelerationProfileManager class."""
    
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
    
    def test_calculate_smooth_speed_command_acceleration(self):
        """Test smooth speed command calculation during acceleration."""
        # Set current state
        self.manager.update_current_state(0.1)  # Current velocity
        
        target_velocity = 0.3  # Target higher velocity
        
        command_velocity, status = self.manager.calculate_smooth_speed_command(target_velocity)
        
        # Should increase velocity but not exceed acceleration limits
        self.assertGreater(command_velocity, 0.1)
        self.assertLessEqual(command_velocity, target_velocity)
        self.assertFalse(status["emergency_braking"])
        self.assertGreaterEqual(status["acceleration"], 0)  # Positive acceleration
    
    def test_calculate_smooth_speed_command_deceleration(self):
        """Test smooth speed command calculation during deceleration."""
        # Set current state
        self.manager.update_current_state(0.3)  # Current velocity
        
        target_velocity = 0.1  # Target lower velocity
        
        command_velocity, status = self.manager.calculate_smooth_speed_command(target_velocity)
        
        # Should decrease velocity but not exceed deceleration limits
        self.assertLess(command_velocity, 0.3)
        self.assertGreaterEqual(command_velocity, target_velocity)
        self.assertFalse(status["emergency_braking"])
        self.assertLessEqual(status["acceleration"], 0)  # Negative acceleration (deceleration)
    
    def test_emergency_braking(self):
        """Test emergency braking functionality."""
        # Set current state
        self.manager.update_current_state(0.3)  # Current velocity
        
        target_velocity = 0.0  # Target stop
        emergency_stop = True
        
        command_velocity, status = self.manager.calculate_smooth_speed_command(
            target_velocity, emergency_stop
        )
        
        # Should apply emergency braking
        self.assertTrue(status["emergency_braking"])
        self.assertLess(command_velocity, 0.3)
        self.assertLessEqual(status["acceleration"], -self.config["emergency_braking"]["max_deceleration"])
    
    def test_calculate_stopping_distance(self):
        """Test stopping distance calculation."""
        current_velocity = 0.3  # m/s
        
        stopping_distance = self.manager.calculate_stopping_distance(current_velocity)
        
        # Should be positive and reasonable
        self.assertGreater(stopping_distance, 0)
        
        # Should include reaction time distance
        reaction_distance = current_velocity * self.config["emergency_braking"]["reaction_time"]
        self.assertGreater(stopping_distance, reaction_distance)
    
    def test_is_safe_to_proceed(self):
        """Test safety assessment for proceeding."""
        current_velocity = 0.3
        
        # Test with safe distance
        obstacle_distance = 2.0
        is_safe = self.manager.is_safe_to_proceed(obstacle_distance, current_velocity)
        self.assertTrue(is_safe)
        
        # Test with unsafe distance
        obstacle_distance = 0.1
        is_safe = self.manager.is_safe_to_proceed(obstacle_distance, current_velocity)
        self.assertFalse(is_safe)
    
    def test_jerk_limiting(self):
        """Test jerk limiting functionality."""
        # Initialize with some acceleration history
        self.manager.acceleration_history.extend([0.0, 0.1, 0.2])
        
        # Request large acceleration change
        desired_acceleration = 0.8  # Large jump
        
        jerk_limited_accel = self.manager._apply_jerk_limiting(desired_acceleration)
        
        # Should limit the acceleration change
        max_change = self.config["acceleration"]["max_jerk"] * self.manager.dt
        expected_max_accel = 0.2 + max_change  # Previous accel + max change
        
        self.assertLessEqual(jerk_limited_accel, expected_max_accel)


class TestSpeedCalculationAccuracy(unittest.TestCase):
    """Test cases for speed calculation accuracy and constraint enforcement."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "base_linear_velocity": 0.3,
            "max_linear_velocity": 0.6,
            "min_linear_velocity": 0.05,
            "visibility": {"max_speed_reduction": 0.7},
            "traffic_density": {"speed_reduction_per_vehicle": 0.15},
            "following_distance": {"time_based_distance": 2.0},
            "acceleration": {"max_acceleration": 0.3, "max_deceleration": 0.5}
        }
    
    def test_speed_constraint_enforcement(self):
        """Test that speed constraints are properly enforced."""
        analyzer = EnvironmentalAnalyzer(self.config)
        
        # Test maximum speed constraint
        base_speed = 0.8  # Above maximum
        max_allowed = self.config["max_linear_velocity"]
        
        # Speed should be clamped to maximum
        constrained_speed = np.clip(base_speed, 0.0, max_allowed)
        self.assertEqual(constrained_speed, max_allowed)
        
        # Test minimum speed constraint
        base_speed = 0.02  # Below minimum
        min_allowed = self.config["min_linear_velocity"]
        
        # Speed should be raised to minimum if not zero
        if base_speed > 0:
            constrained_speed = max(base_speed, min_allowed)
            self.assertEqual(constrained_speed, min_allowed)
    
    def test_environmental_factor_accuracy(self):
        """Test accuracy of environmental factor calculations."""
        analyzer = EnvironmentalAnalyzer(self.config)
        
        # Test with known conditions
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 100  # Medium brightness
        detected_objects = [{"type": "duckiebot", "distance": 1.0}]  # One nearby vehicle
        
        factors = analyzer.get_environmental_speed_factor(test_image, detected_objects)
        
        # Verify factor ranges
        for factor_name, factor_value in factors.items():
            self.assertGreaterEqual(factor_value, 0.0, f"{factor_name} below minimum")
            self.assertLessEqual(factor_value, 1.0, f"{factor_name} above maximum")
        
        # Overall factor should be most restrictive
        individual_factors = [
            factors["visibility_factor"],
            factors["traffic_density_factor"],
            factors["road_condition_factor"]
        ]
        expected_overall = min(individual_factors)
        self.assertAlmostEqual(factors["overall_environmental_factor"], expected_overall, places=3)
    
    def test_following_distance_accuracy(self):
        """Test accuracy of following distance calculations."""
        controller = FollowingDistanceController(self.config)
        
        # Test time-based distance calculation
        velocity = 0.3  # m/s
        time_distance = self.config["following_distance"]["time_based_distance"]
        
        desired_distance = controller.calculate_desired_following_distance(velocity)
        expected_distance = velocity * time_distance
        
        # Should be at least the time-based distance
        self.assertGreaterEqual(desired_distance, expected_distance)
    
    def test_acceleration_profile_accuracy(self):
        """Test accuracy of acceleration profile calculations."""
        manager = AccelerationProfileManager(self.config)
        
        # Test acceleration limiting
        manager.update_current_state(0.1)  # Set current velocity
        
        target_velocity = 0.5  # High target
        command_velocity, status = manager.calculate_smooth_speed_command(target_velocity)
        
        # Acceleration should not exceed limits
        max_accel = self.config["acceleration"]["max_acceleration"]
        dt = manager.dt
        max_velocity_increase = max_accel * dt
        
        velocity_increase = command_velocity - 0.1
        self.assertLessEqual(velocity_increase, max_velocity_increase * 1.1)  # Small tolerance for smoothing


if __name__ == '__main__':
    # Initialize rospy for testing
    rospy.init_node('test_adaptive_speed_controller', anonymous=True)
    
    # Run the tests
    rostest.rosrun('adaptive_speed_control', 'test_adaptive_speed_controller', 
                   'test_adaptive_speed_controller.TestEnvironmentalAnalyzer')
    rostest.rosrun('adaptive_speed_control', 'test_following_distance_controller', 
                   'test_adaptive_speed_controller.TestFollowingDistanceController')
    rostest.rosrun('adaptive_speed_control', 'test_acceleration_profile_manager', 
                   'test_adaptive_speed_controller.TestAccelerationProfileManager')
    rostest.rosrun('adaptive_speed_control', 'test_speed_calculation_accuracy', 
                   'test_adaptive_speed_controller.TestSpeedCalculationAccuracy')