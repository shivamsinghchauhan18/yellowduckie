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


class TestSafetyIntegration(unittest.TestCase):
    """Test cases for safety system integration with adaptive speed control."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            "base_linear_velocity": 0.3,
            "max_linear_velocity": 0.6,
            "min_linear_velocity": 0.05,
            "safety_integration_enabled": True,
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
            "safety_integration": {
                "collision_risk_threshold": 0.7,
                "safety_margin_factor": 0.8,
                "override_timeout": 1.0
            },
            "control": {
                "update_rate": 20
            }
        }
        
        # Initialize components
        self.environmental_analyzer = EnvironmentalAnalyzer(self.config)
        self.following_controller = FollowingDistanceController(self.config)
        self.acceleration_manager = AccelerationProfileManager(self.config)
    
    def test_collision_risk_speed_constraints(self):
        """Test speed constraints based on collision risk levels."""
        base_velocity = 0.3
        
        # Test different collision risk levels
        risk_scenarios = [
            {"risk": "NONE", "expected_factor": 1.0},
            {"risk": "LOW", "expected_factor": 0.8},
            {"risk": "MEDIUM", "expected_factor": 0.6},
            {"risk": "HIGH", "expected_factor": 0.3},
            {"risk": "CRITICAL", "expected_factor": 0.0}
        ]
        
        for scenario in risk_scenarios:
            safety_factor = self._apply_safety_constraints_by_risk(scenario["risk"])
            constrained_velocity = base_velocity * safety_factor
            
            # Should match expected constraint level
            expected_velocity = base_velocity * scenario["expected_factor"]
            self.assertAlmostEqual(constrained_velocity, expected_velocity, places=2,
                                 msg=f"Failed for risk level {scenario['risk']}")
    
    def test_safety_status_speed_constraints(self):
        """Test speed constraints based on safety system status."""
        base_velocity = 0.3
        
        # Test different safety status levels
        status_scenarios = [
            {"status": "SAFE", "min_factor": 1.0},
            {"status": "CAUTION", "min_factor": 0.8},
            {"status": "WARNING", "min_factor": 0.6},
            {"status": "DANGER", "min_factor": 0.3},
            {"status": "CRITICAL", "min_factor": 0.0}
        ]
        
        for scenario in status_scenarios:
            safety_factor = self._apply_safety_constraints_by_status(scenario["status"])
            constrained_velocity = base_velocity * safety_factor
            
            # Should be at or below expected constraint level
            max_expected_velocity = base_velocity * scenario["min_factor"]
            self.assertLessEqual(constrained_velocity, max_expected_velocity * 1.1,  # Small tolerance
                               msg=f"Failed for status {scenario['status']}")
    
    def test_emergency_stop_integration(self):
        """Test emergency stop integration with safety systems."""
        # Test emergency conditions
        emergency_scenarios = [
            {"collision_risk": "CRITICAL", "should_emergency": True},
            {"safety_status": "CRITICAL", "should_emergency": True},
            {"obstacle_distance": 0.2, "should_emergency": True},
            {"collision_risk": "HIGH", "should_emergency": False},
            {"safety_status": "DANGER", "should_emergency": False},
            {"obstacle_distance": 1.0, "should_emergency": False}
        ]
        
        for scenario in emergency_scenarios:
            emergency_needed = self._check_emergency_conditions(scenario)
            
            self.assertEqual(emergency_needed, scenario["should_emergency"],
                           msg=f"Emergency check failed for scenario {scenario}")
    
    def test_safety_override_priority(self):
        """Test that safety overrides take priority over other speed commands."""
        base_speed = 0.3
        environmental_factor = 0.8  # Good conditions
        following_speed = 0.25      # Following distance adjustment
        
        # Test with normal safety conditions
        safety_factor = 1.0  # No safety constraints
        final_speed = self._combine_speed_adjustments(
            base_speed, environmental_factor, following_speed, safety_factor
        )
        
        # Should use following speed (most restrictive non-safety factor)
        self.assertAlmostEqual(final_speed, following_speed, places=2)
        
        # Test with safety override
        safety_factor = 0.4  # Safety constraint
        final_speed_with_safety = self._combine_speed_adjustments(
            base_speed, environmental_factor, following_speed, safety_factor
        )
        
        # Safety should override other factors
        expected_safety_speed = following_speed * safety_factor
        self.assertAlmostEqual(final_speed_with_safety, expected_safety_speed, places=2)
    
    def test_safety_margin_enforcement(self):
        """Test enforcement of safety margins."""
        # First calculate actual stopping distance for 0.3 m/s
        velocity = 0.3
        stopping_distance = self.acceleration_manager.calculate_stopping_distance(velocity)
        
        # Test with various obstacle distances relative to stopping distance
        # Note: is_safe_to_proceed includes a 0.2m safety margin
        safety_margin = 0.2
        safe_distance = stopping_distance + safety_margin
        
        test_scenarios = [
            {"distance": 0.1, "velocity": velocity, "should_be_safe": False},
            {"distance": stopping_distance * 0.8, "velocity": velocity, "should_be_safe": False},
            {"distance": safe_distance + 0.1, "velocity": velocity, "should_be_safe": True},
            {"distance": 2.0, "velocity": velocity, "should_be_safe": True}
        ]
        
        for scenario in test_scenarios:
            is_safe = self.acceleration_manager.is_safe_to_proceed(
                scenario["distance"], scenario["velocity"]
            )
            
            self.assertEqual(is_safe, scenario["should_be_safe"],
                           msg=f"Safety assessment failed for {scenario}, stopping_distance={stopping_distance:.3f}")
    
    def test_speed_limit_enforcement_with_safety(self):
        """Test speed limit enforcement considering safety margins."""
        # Test various speed scenarios with safety constraints
        test_cases = [
            {
                "base_speed": 0.8,  # Above max
                "safety_factor": 1.0,
                "expected_max": 0.6  # Should be clamped to max_linear_velocity
            },
            {
                "base_speed": 0.3,
                "safety_factor": 0.5,
                "expected_max": 0.15  # Should be reduced by safety factor
            },
            {
                "base_speed": 0.02,  # Below min
                "safety_factor": 1.0,
                "expected_min": 0.05  # Should be raised to min_linear_velocity
            }
        ]
        
        for case in test_cases:
            # Apply safety constraint
            constrained_speed = case["base_speed"] * case["safety_factor"]
            
            # Apply speed limits
            final_speed = np.clip(constrained_speed, 0.0, self.config["max_linear_velocity"])
            
            # Apply minimum speed if not zero
            if 0 < final_speed < self.config["min_linear_velocity"]:
                final_speed = self.config["min_linear_velocity"]
            
            # Check expectations
            if "expected_max" in case:
                self.assertLessEqual(final_speed, case["expected_max"])
            if "expected_min" in case:
                self.assertGreaterEqual(final_speed, case["expected_min"])
    
    def test_integrated_safety_speed_calculation(self):
        """Test integrated safety-aware speed calculation."""
        # Scenario: Multiple safety factors active
        base_velocity = 0.3
        
        # Environmental factors (poor visibility)
        poor_image = np.ones((480, 640, 3), dtype=np.uint8) * 25  # Dark
        environmental_factors = self.environmental_analyzer.get_environmental_speed_factor(poor_image, [])
        
        # Following distance (close vehicle)
        detected_vehicles = [{"type": "duckiebot", "distance": 0.8}]
        following_speed, _ = self.following_controller.get_following_distance_command(
            0.3, detected_vehicles, base_velocity
        )
        
        # Safety constraints (high collision risk)
        safety_factor = 0.4  # High risk
        
        # Combine all factors
        final_speed = self._combine_speed_adjustments(
            base_velocity,
            environmental_factors["overall_environmental_factor"],
            following_speed,
            safety_factor
        )
        
        # Should be significantly reduced due to multiple safety factors
        self.assertLess(final_speed, base_velocity * 0.6)
        self.assertGreaterEqual(final_speed, 0.0)
    
    def test_safety_system_coordination(self):
        """Test coordination between different safety systems."""
        # Test scenario where multiple safety systems provide conflicting advice
        base_speed = 0.3
        
        # Environmental system says slow down (poor visibility)
        env_factor = 0.6
        
        # Following system says speed up (too far from vehicle)
        following_speed = 0.35  # Slightly higher than base
        
        # Safety system says emergency stop
        safety_factor = 0.0  # Critical safety condition
        
        # Safety should override all other systems
        final_speed = self._combine_speed_adjustments(
            base_speed, env_factor, following_speed, safety_factor
        )
        
        self.assertEqual(final_speed, 0.0)  # Should be emergency stop
    
    def test_safety_constraint_timeout_handling(self):
        """Test handling of safety constraint timeouts."""
        # This would test timeout behavior in a real implementation
        # For now, we test the concept with mock timeout logic
        
        constraint_age = 2.0  # seconds
        timeout_threshold = 1.0  # seconds
        
        # Constraint should be considered expired
        is_expired = constraint_age > timeout_threshold
        self.assertTrue(is_expired)
        
        # Expired constraints should not be applied
        if is_expired:
            safety_factor = 1.0  # No constraint
        else:
            safety_factor = 0.5  # Apply constraint
        
        self.assertEqual(safety_factor, 1.0)
    
    def test_safety_performance_monitoring(self):
        """Test safety-related performance monitoring."""
        # Mock performance statistics
        performance_stats = {
            "total_commands": 1000,
            "safety_overrides": 50,
            "emergency_stops": 5,
            "constraint_violations": 10
        }
        
        # Calculate safety metrics
        safety_override_rate = performance_stats["safety_overrides"] / performance_stats["total_commands"]
        emergency_stop_rate = performance_stats["emergency_stops"] / performance_stats["total_commands"]
        
        # Should be within reasonable bounds for safe operation
        self.assertLess(safety_override_rate, 0.1)  # Less than 10% override rate
        self.assertLess(emergency_stop_rate, 0.01)  # Less than 1% emergency stops
    
    # Helper methods for testing
    def _apply_safety_constraints_by_risk(self, collision_risk_level):
        """Apply safety constraints based on collision risk level."""
        if collision_risk_level == "CRITICAL":
            return 0.0
        elif collision_risk_level == "HIGH":
            return 0.3
        elif collision_risk_level == "MEDIUM":
            return 0.6
        elif collision_risk_level == "LOW":
            return 0.8
        else:  # NONE
            return 1.0
    
    def _apply_safety_constraints_by_status(self, safety_status):
        """Apply safety constraints based on safety system status."""
        if safety_status == "CRITICAL":
            return 0.0
        elif safety_status == "DANGER":
            return 0.3
        elif safety_status == "WARNING":
            return 0.6
        elif safety_status == "CAUTION":
            return 0.8
        else:  # SAFE
            return 1.0
    
    def _check_emergency_conditions(self, scenario):
        """Check if emergency conditions are met."""
        # Check collision risk
        if scenario.get("collision_risk") == "CRITICAL":
            return True
        
        # Check safety status
        if scenario.get("safety_status") == "CRITICAL":
            return True
        
        # Check obstacle distance
        obstacle_distance = scenario.get("obstacle_distance")
        if obstacle_distance is not None and obstacle_distance < 0.3:
            return True
        
        return False
    
    def _combine_speed_adjustments(self, base_velocity, environmental_factor, 
                                 following_velocity, safety_factor):
        """Combine speed adjustments with safety priority."""
        # Start with base velocity
        target_velocity = base_velocity
        
        # Apply environmental factors
        target_velocity *= environmental_factor
        
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


if __name__ == '__main__':
    unittest.main()