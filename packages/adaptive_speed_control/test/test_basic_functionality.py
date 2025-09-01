#!/usr/bin/env python3

import unittest
import numpy as np
import sys
import os

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'include'))

# Mock rospy for testing without ROS
class MockRospy:
    def loginfo(self, msg):
        print(f"INFO: {msg}")
    
    def logwarn(self, msg):
        print(f"WARN: {msg}")
    
    class Time:
        @staticmethod
        def now():
            class MockTime:
                def to_sec(self):
                    return 1234567890.0
            return MockTime()

# Mock rospy module
sys.modules['rospy'] = MockRospy()

# Now import our classes
from adaptive_speed_control.environmental_analyzer import EnvironmentalAnalyzer
from adaptive_speed_control.following_distance_controller import FollowingDistanceController
from adaptive_speed_control.acceleration_profile_manager import AccelerationProfileManager


class TestBasicFunctionality(unittest.TestCase):
    """Basic functionality tests that don't require ROS."""
    
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
            },
            "following_distance": {
                "time_based_distance": 2.0,
                "min_distance": 0.5,
                "max_distance": 4.0,
                "speed_reduction_factor": 0.5
            },
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
                "update_rate": 20,
                "smoothing_window": 5,
                "prediction_horizon": 1.0
            }
        }
    
    def test_environmental_analyzer_initialization(self):
        """Test that EnvironmentalAnalyzer initializes correctly."""
        analyzer = EnvironmentalAnalyzer(self.config)
        
        self.assertEqual(analyzer.brightness_threshold, 50)
        self.assertEqual(analyzer.contrast_threshold, 20)
        self.assertEqual(analyzer.detection_distance, 2.0)
        self.assertEqual(len(analyzer.visibility_history), 0)
    
    def test_visibility_analysis_with_good_image(self):
        """Test visibility analysis with a good quality image."""
        analyzer = EnvironmentalAnalyzer(self.config)
        
        # Create a test image with good brightness and contrast
        test_image = np.random.randint(80, 180, (480, 640, 3), dtype=np.uint8)
        
        visibility_factor = analyzer.analyze_visibility(test_image)
        
        # Should return a reasonable visibility factor
        self.assertGreaterEqual(visibility_factor, 0.0)
        self.assertLessEqual(visibility_factor, 1.0)
        self.assertGreater(visibility_factor, 0.3)  # Should be decent for good image
    
    def test_visibility_analysis_with_dark_image(self):
        """Test visibility analysis with a dark image."""
        analyzer = EnvironmentalAnalyzer(self.config)
        
        # Create a very dark test image
        test_image = np.ones((480, 640, 3), dtype=np.uint8) * 20
        
        visibility_factor = analyzer.analyze_visibility(test_image)
        
        # Should return low visibility factor for dark image
        self.assertGreaterEqual(visibility_factor, 0.0)
        self.assertLessEqual(visibility_factor, 1.0)
        self.assertLess(visibility_factor, 0.6)  # Should be low for dark image
    
    def test_traffic_density_no_vehicles(self):
        """Test traffic density analysis with no vehicles."""
        analyzer = EnvironmentalAnalyzer(self.config)
        
        detected_objects = []
        traffic_factor = analyzer.analyze_traffic_density(detected_objects)
        
        # Should return 1.0 for no traffic
        self.assertEqual(traffic_factor, 1.0)
    
    def test_traffic_density_with_vehicles(self):
        """Test traffic density analysis with vehicles."""
        analyzer = EnvironmentalAnalyzer(self.config)
        
        detected_objects = [
            {"type": "duckiebot", "distance": 1.0},
            {"type": "duckiebot", "distance": 1.5},
        ]
        
        traffic_factor = analyzer.analyze_traffic_density(detected_objects)
        
        # Should reduce factor due to nearby vehicles
        self.assertLess(traffic_factor, 1.0)
        self.assertGreaterEqual(traffic_factor, 0.1)
    
    def test_following_distance_controller_initialization(self):
        """Test that FollowingDistanceController initializes correctly."""
        controller = FollowingDistanceController(self.config)
        
        self.assertEqual(controller.time_based_distance, 2.0)
        self.assertEqual(controller.min_distance, 0.5)
        self.assertEqual(controller.max_distance, 4.0)
        self.assertEqual(len(controller.tracked_vehicles), 0)
    
    def test_desired_following_distance_calculation(self):
        """Test desired following distance calculation."""
        controller = FollowingDistanceController(self.config)
        
        current_velocity = 0.3  # m/s
        leading_velocity = 0.2  # m/s
        
        desired_distance = controller.calculate_desired_following_distance(
            current_velocity, leading_velocity
        )
        
        # Should be at least time_based_distance * velocity
        expected_min = current_velocity * 2.0  # time_based_distance
        self.assertGreaterEqual(desired_distance, expected_min)
        
        # Should be within bounds
        self.assertGreaterEqual(desired_distance, 0.5)  # min_distance
        self.assertLessEqual(desired_distance, 4.0)  # max_distance
    
    def test_acceleration_profile_manager_initialization(self):
        """Test that AccelerationProfileManager initializes correctly."""
        manager = AccelerationProfileManager(self.config)
        
        self.assertEqual(manager.max_acceleration, 0.3)
        self.assertEqual(manager.max_deceleration, 0.5)
        self.assertEqual(manager.max_jerk, 0.5)
        self.assertEqual(manager.current_velocity, 0.0)
    
    def test_stopping_distance_calculation(self):
        """Test stopping distance calculation."""
        manager = AccelerationProfileManager(self.config)
        
        current_velocity = 0.3  # m/s
        stopping_distance = manager.calculate_stopping_distance(current_velocity)
        
        # Should be positive
        self.assertGreater(stopping_distance, 0)
        
        # Should include reaction time component
        reaction_distance = current_velocity * 0.1  # reaction_time
        self.assertGreater(stopping_distance, reaction_distance)
    
    def test_emergency_stopping_distance(self):
        """Test emergency stopping distance calculation."""
        manager = AccelerationProfileManager(self.config)
        
        current_velocity = 0.3  # m/s
        emergency_distance = manager.calculate_emergency_stopping_distance(current_velocity)
        normal_distance = manager.calculate_stopping_distance(current_velocity)
        
        # Emergency stopping should be shorter than normal
        self.assertLess(emergency_distance, normal_distance)
        self.assertGreater(emergency_distance, 0)
    
    def test_safety_assessment(self):
        """Test safety assessment for proceeding."""
        manager = AccelerationProfileManager(self.config)
        
        current_velocity = 0.3
        
        # Test with safe distance
        obstacle_distance = 2.0
        is_safe = manager.is_safe_to_proceed(obstacle_distance, current_velocity)
        self.assertTrue(is_safe)
        
        # Test with unsafe distance
        obstacle_distance = 0.1
        is_safe = manager.is_safe_to_proceed(obstacle_distance, current_velocity)
        self.assertFalse(is_safe)
    
    def test_speed_constraint_enforcement(self):
        """Test speed constraint enforcement."""
        # Test maximum speed constraint
        base_speed = 0.8  # Above maximum
        max_allowed = 0.6
        
        constrained_speed = np.clip(base_speed, 0.0, max_allowed)
        self.assertEqual(constrained_speed, max_allowed)
        
        # Test minimum speed constraint
        base_speed = 0.02  # Below minimum
        min_allowed = 0.05
        
        if base_speed > 0:
            constrained_speed = max(base_speed, min_allowed)
            self.assertEqual(constrained_speed, min_allowed)
    
    def test_environmental_factor_ranges(self):
        """Test that environmental factors are within valid ranges."""
        analyzer = EnvironmentalAnalyzer(self.config)
        
        # Test with various image conditions
        test_images = [
            np.ones((480, 640, 3), dtype=np.uint8) * 50,   # Dark
            np.ones((480, 640, 3), dtype=np.uint8) * 120,  # Good
            np.ones((480, 640, 3), dtype=np.uint8) * 200,  # Bright
        ]
        
        for test_image in test_images:
            factors = analyzer.get_environmental_speed_factor(test_image, [])
            
            # All factors should be in valid range [0, 1]
            for factor_name, factor_value in factors.items():
                self.assertGreaterEqual(factor_value, 0.0, 
                                      f"{factor_name} below minimum: {factor_value}")
                self.assertLessEqual(factor_value, 1.0, 
                                   f"{factor_name} above maximum: {factor_value}")


if __name__ == '__main__':
    unittest.main()