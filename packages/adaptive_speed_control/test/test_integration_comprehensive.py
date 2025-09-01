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
        pass
    
    def logwarn(self, msg):
        pass
    
    class Time:
        @staticmethod
        def now():
            class MockTime:
                def to_sec(self):
                    return 1234567890.0
            return MockTime()

sys.modules['rospy'] = MockRospy()

from adaptive_speed_control.environmental_analyzer import EnvironmentalAnalyzer
from adaptive_speed_control.following_distance_controller import FollowingDistanceController
from adaptive_speed_control.acceleration_profile_manager import AccelerationProfileManager


class AdaptiveSpeedControllerIntegration:
    """Mock integration class for comprehensive testing."""
    
    def __init__(self, config):
        self.config = config
        self.environmental_analyzer = EnvironmentalAnalyzer(config)
        self.following_controller = FollowingDistanceController(config)
        self.acceleration_manager = AccelerationProfileManager(config)
        
        # State
        self.current_velocity = 0.0
        self.collision_risk_level = "NONE"
        self.safety_status = "SAFE"
    
    def process_speed_command(self, base_velocity, image, detected_vehicles, 
                            imu_data=None, emergency_stop=False):
        """Process a complete speed command with all systems integrated."""
        
        # Update acceleration manager state
        self.acceleration_manager.update_current_state(self.current_velocity)
        
        # Environmental analysis
        environmental_factors = self.environmental_analyzer.get_environmental_speed_factor(
            image, detected_vehicles, imu_data
        )
        
        # Following distance control
        following_velocity, following_status = self.following_controller.get_following_distance_command(
            self.current_velocity, detected_vehicles, base_velocity
        )
        
        # Safety constraints
        safety_factor = self._apply_safety_constraints()
        
        # Combine adjustments
        target_velocity = self._combine_speed_adjustments(
            base_velocity, environmental_factors, following_velocity, safety_factor
        )
        
        # Apply acceleration profile
        final_velocity, acceleration_profile = self.acceleration_manager.calculate_smooth_speed_command(
            target_velocity, emergency_stop
        )
        
        # Update current velocity for next iteration
        self.current_velocity = final_velocity
        
        return {
            "final_velocity": final_velocity,
            "environmental_factors": environmental_factors,
            "following_status": following_status,
            "acceleration_profile": acceleration_profile,
            "safety_factor": safety_factor,
            "target_velocity": target_velocity
        }
    
    def _apply_safety_constraints(self):
        """Apply safety constraints based on current safety state."""
        safety_factor = 1.0
        
        # Collision risk constraints
        if self.collision_risk_level == "CRITICAL":
            safety_factor = 0.0
        elif self.collision_risk_level == "HIGH":
            safety_factor = 0.3
        elif self.collision_risk_level == "MEDIUM":
            safety_factor = 0.6
        elif self.collision_risk_level == "LOW":
            safety_factor = 0.8
        
        # Safety status constraints
        if self.safety_status == "CRITICAL":
            safety_factor = min(safety_factor, 0.0)
        elif self.safety_status == "DANGER":
            safety_factor = min(safety_factor, 0.3)
        elif self.safety_status == "WARNING":
            safety_factor = min(safety_factor, 0.6)
        
        return safety_factor
    
    def _combine_speed_adjustments(self, base_velocity, environmental_factors, 
                                 following_velocity, safety_factor):
        """Combine all speed adjustments."""
        target_velocity = base_velocity
        
        # Apply environmental factors
        target_velocity *= environmental_factors["overall_environmental_factor"]
        
        # Apply following distance adjustment
        if following_velocity != base_velocity:
            target_velocity = following_velocity
        
        # Apply safety factor (most restrictive)
        target_velocity *= safety_factor
        
        # Ensure within bounds
        target_velocity = np.clip(target_velocity, 0.0, self.config["max_linear_velocity"])
        
        # Apply minimum speed if not zero
        if 0 < target_velocity < self.config["min_linear_velocity"]:
            target_velocity = self.config["min_linear_velocity"]
        
        return target_velocity


class TestComprehensiveIntegration(unittest.TestCase):
    """Comprehensive integration tests for the adaptive speed control system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "base_linear_velocity": 0.3,
            "max_linear_velocity": 0.6,
            "min_linear_velocity": 0.05,
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
                "update_rate": 20
            }
        }
        
        self.controller = AdaptiveSpeedControllerIntegration(self.config)
    
    def test_normal_operation_scenario(self):
        """Test normal operation with good conditions."""
        # Good conditions
        good_image = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
        no_vehicles = []
        base_velocity = 0.3
        
        result = self.controller.process_speed_command(
            base_velocity, good_image, no_vehicles
        )
        
        # Should start accelerating towards base velocity (from rest)
        self.assertGreater(result["final_velocity"], 0.0)
        self.assertLessEqual(result["final_velocity"], base_velocity)
        self.assertFalse(result["acceleration_profile"]["emergency_braking"])
        # Should be accelerating towards target
        self.assertGreater(result["acceleration_profile"]["acceleration"], 0.0)
    
    def test_poor_visibility_scenario(self):
        """Test operation in poor visibility conditions."""
        # Poor visibility
        dark_image = np.ones((480, 640, 3), dtype=np.uint8) * 25
        no_vehicles = []
        base_velocity = 0.3
        
        result = self.controller.process_speed_command(
            base_velocity, dark_image, no_vehicles
        )
        
        # Should reduce speed due to poor visibility
        self.assertLess(result["final_velocity"], base_velocity * 0.8)
        self.assertLess(result["environmental_factors"]["visibility_factor"], 0.6)
    
    def test_vehicle_following_scenario(self):
        """Test vehicle following behavior."""
        # Good visibility with close vehicle
        good_image = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
        close_vehicle = [
            {
                "id": "vehicle_1",
                "position": {"x": 1.0, "y": 0.0, "z": 0.0},
                "distance": 1.0,
                "velocity": {"x": 0.2, "y": 0.0},
                "confidence": 0.8,
                "type": "duckiebot"
            }
        ]
        base_velocity = 0.3
        
        result = self.controller.process_speed_command(
            base_velocity, good_image, close_vehicle
        )
        
        # Should adjust speed for following
        self.assertTrue(result["following_status"]["is_following"])
        self.assertLess(result["following_status"]["following_safety_factor"], 1.0)
    
    def test_emergency_stop_scenario(self):
        """Test emergency stop behavior."""
        # Any conditions with emergency stop
        good_image = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
        no_vehicles = []
        base_velocity = 0.3
        
        # Set emergency conditions
        self.controller.current_velocity = 0.3
        
        result = self.controller.process_speed_command(
            base_velocity, good_image, no_vehicles, emergency_stop=True
        )
        
        # Should activate emergency braking
        self.assertTrue(result["acceleration_profile"]["emergency_braking"])
        self.assertLess(result["final_velocity"], 0.3)
    
    def test_safety_system_override_scenario(self):
        """Test safety system override behavior."""
        # Good conditions but critical safety status
        good_image = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
        no_vehicles = []
        base_velocity = 0.3
        
        # Set critical safety conditions
        self.controller.collision_risk_level = "CRITICAL"
        self.controller.safety_status = "CRITICAL"
        
        result = self.controller.process_speed_command(
            base_velocity, good_image, no_vehicles
        )
        
        # Should override to emergency stop
        self.assertEqual(result["safety_factor"], 0.0)
        self.assertEqual(result["final_velocity"], 0.0)
    
    def test_multiple_constraints_scenario(self):
        """Test behavior with multiple active constraints."""
        # Poor visibility + close vehicle + safety warning
        dark_image = np.ones((480, 640, 3), dtype=np.uint8) * 30
        close_vehicles = [
            {
                "id": "vehicle_1",
                "position": {"x": 0.8, "y": 0.0, "z": 0.0},
                "distance": 0.8,
                "velocity": {"x": 0.1, "y": 0.0},
                "confidence": 0.9,
                "type": "duckiebot"
            },
            {
                "id": "vehicle_2",
                "position": {"x": 1.2, "y": 0.1, "z": 0.0},
                "distance": 1.2,
                "velocity": {"x": 0.15, "y": 0.0},
                "confidence": 0.8,
                "type": "duckiebot"
            }
        ]
        base_velocity = 0.3
        
        # Set warning safety conditions
        self.controller.collision_risk_level = "MEDIUM"
        self.controller.safety_status = "WARNING"
        
        result = self.controller.process_speed_command(
            base_velocity, dark_image, close_vehicles
        )
        
        # Should significantly reduce speed due to multiple constraints
        self.assertLess(result["final_velocity"], base_velocity * 0.5)
        self.assertLess(result["environmental_factors"]["overall_environmental_factor"], 0.8)
        self.assertTrue(result["following_status"]["is_following"])
        self.assertLess(result["safety_factor"], 1.0)
    
    def test_speed_smoothing_over_time(self):
        """Test speed smoothing over multiple control cycles."""
        # Simulate changing conditions over time
        good_image = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
        no_vehicles = []
        base_velocity = 0.3
        
        velocities = []
        
        # Start from rest
        self.controller.current_velocity = 0.0
        
        # Simulate multiple control cycles
        for i in range(20):
            result = self.controller.process_speed_command(
                base_velocity, good_image, no_vehicles
            )
            velocities.append(result["final_velocity"])
        
        # Should show smooth acceleration to target
        self.assertGreater(velocities[-1], velocities[0])  # Should increase
        
        # Check for reasonable smoothness (no huge jumps)
        for i in range(1, len(velocities)):
            velocity_change = abs(velocities[i] - velocities[i-1])
            self.assertLess(velocity_change, 0.1)  # No more than 0.1 m/s change per step
    
    def test_system_requirements_compliance(self):
        """Test compliance with system requirements."""
        # Test various scenarios to ensure requirements are met
        test_scenarios = [
            {
                "name": "Normal operation",
                "image": np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8),
                "vehicles": [],
                "safety_risk": "NONE",
                "safety_status": "SAFE",
                "expected_acceleration": True
            },
            {
                "name": "Poor visibility",
                "image": np.ones((480, 640, 3), dtype=np.uint8) * 25,
                "vehicles": [],
                "safety_risk": "NONE",
                "safety_status": "SAFE",
                "expected_max_speed": 0.2
            },
            {
                "name": "Vehicle following",
                "image": np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8),
                "vehicles": [{"id": "v1", "position": {"x": 1.0, "y": 0.0, "z": 0.0}, 
                           "distance": 1.0, "velocity": {"x": 0.2, "y": 0.0}, 
                           "confidence": 0.8, "type": "duckiebot"}],
                "safety_risk": "NONE",
                "safety_status": "SAFE",
                "expected_following": True
            }
        ]
        
        base_velocity = 0.3
        
        for scenario in test_scenarios:
            # Set safety conditions
            self.controller.collision_risk_level = scenario["safety_risk"]
            self.controller.safety_status = scenario["safety_status"]
            
            result = self.controller.process_speed_command(
                base_velocity, scenario["image"], scenario["vehicles"]
            )
            
            # Check scenario-specific requirements
            if "expected_acceleration" in scenario:
                self.assertGreater(result["acceleration_profile"]["acceleration"], 0.0,
                                 f"Failed acceleration for {scenario['name']}")
            
            if "expected_max_speed" in scenario:
                self.assertLess(result["final_velocity"], scenario["expected_max_speed"],
                               f"Failed max speed for {scenario['name']}")
            
            if "expected_following" in scenario:
                self.assertEqual(result["following_status"]["is_following"], 
                               scenario["expected_following"],
                               f"Failed following requirement for {scenario['name']}")
            
            # General requirements
            self.assertGreaterEqual(result["final_velocity"], 0.0,
                                  f"Negative velocity in {scenario['name']}")
            self.assertLessEqual(result["final_velocity"], self.config["max_linear_velocity"],
                               f"Exceeded max velocity in {scenario['name']}")


if __name__ == '__main__':
    unittest.main()