#!/usr/bin/env python3

import unittest
import numpy as np
import sys
import os
import cv2

# Add the package to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'include'))

# Mock rospy for testing without ROS
class MockRospy:
    def loginfo(self, msg):
        pass
    
    def logwarn(self, msg):
        pass

sys.modules['rospy'] = MockRospy()

from adaptive_speed_control.environmental_analyzer import EnvironmentalAnalyzer


class TestEnvironmentalSpeedAdaptation(unittest.TestCase):
    """Test cases for environmental speed adaptation functionality."""
    
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
    
    def test_visibility_based_speed_reduction_dark_conditions(self):
        """Test speed reduction in dark/low visibility conditions."""
        # Create very dark image
        dark_image = np.ones((480, 640, 3), dtype=np.uint8) * 15  # Very dark
        
        visibility_factor = self.analyzer.analyze_visibility(dark_image)
        
        # Should significantly reduce speed in dark conditions
        self.assertLess(visibility_factor, 0.5)
        
        # Test that speed reduction is applied
        base_speed = 0.3
        adjusted_speed = base_speed * visibility_factor
        speed_reduction = (base_speed - adjusted_speed) / base_speed
        
        self.assertGreater(speed_reduction, 0.2)  # At least 20% reduction
    
    def test_visibility_based_speed_reduction_bright_conditions(self):
        """Test speed behavior in bright/overexposed conditions."""
        # Create overexposed image
        bright_image = np.ones((480, 640, 3), dtype=np.uint8) * 240  # Very bright
        
        visibility_factor = self.analyzer.analyze_visibility(bright_image)
        
        # Should reduce speed in overexposed conditions too
        self.assertLess(visibility_factor, 0.8)
    
    def test_visibility_based_speed_reduction_optimal_conditions(self):
        """Test speed behavior in optimal visibility conditions."""
        # Create image with optimal brightness and good contrast
        optimal_image = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
        
        visibility_factor = self.analyzer.analyze_visibility(optimal_image)
        
        # Should allow near-full speed in optimal conditions
        self.assertGreater(visibility_factor, 0.6)
    
    def test_traffic_density_based_speed_adjustment_single_vehicle(self):
        """Test speed adjustment with single nearby vehicle."""
        detected_objects = [
            {"type": "duckiebot", "distance": 1.0}  # Close vehicle
        ]
        
        traffic_factor = self.analyzer.analyze_traffic_density(detected_objects)
        
        # Should reduce speed with one nearby vehicle
        expected_reduction = self.config["traffic_density"]["speed_reduction_per_vehicle"]
        expected_factor = 1.0 - expected_reduction
        
        self.assertAlmostEqual(traffic_factor, expected_factor, places=2)
    
    def test_traffic_density_based_speed_adjustment_multiple_vehicles(self):
        """Test speed adjustment with multiple nearby vehicles."""
        detected_objects = [
            {"type": "duckiebot", "distance": 0.8},
            {"type": "duckiebot", "distance": 1.2},
            {"type": "car", "distance": 1.5}
        ]
        
        traffic_factor = self.analyzer.analyze_traffic_density(detected_objects)
        
        # Should significantly reduce speed with multiple vehicles
        self.assertLess(traffic_factor, 0.6)  # Significant reduction
    
    def test_traffic_density_distance_filtering(self):
        """Test that only nearby vehicles affect speed."""
        detected_objects = [
            {"type": "duckiebot", "distance": 0.5},  # Close - should affect
            {"type": "duckiebot", "distance": 3.0},  # Far - should not affect
            {"type": "duckiebot", "distance": 5.0}   # Very far - should not affect
        ]
        
        traffic_factor = self.analyzer.analyze_traffic_density(detected_objects)
        
        # Should only be affected by the close vehicle
        expected_reduction = self.config["traffic_density"]["speed_reduction_per_vehicle"]
        expected_factor = 1.0 - expected_reduction
        
        self.assertAlmostEqual(traffic_factor, expected_factor, places=2)
    
    def test_road_condition_assessment_smooth_surface(self):
        """Test road condition assessment with smooth surface."""
        # Create image representing smooth road surface
        smooth_road = np.ones((480, 640, 3), dtype=np.int32) * 100
        # Add minimal texture variation for smooth surface
        smooth_road[240:, :] += np.random.randint(-5, 5, smooth_road[240:, :].shape)
        smooth_road = np.clip(smooth_road, 0, 255).astype(np.uint8)
        
        road_factor = self.analyzer.analyze_road_conditions(smooth_road)
        
        # Should allow higher speeds on smooth roads
        self.assertGreater(road_factor, 0.7)
    
    def test_road_condition_assessment_rough_surface(self):
        """Test road condition assessment with rough surface."""
        # Create image representing rough road surface
        rough_road = np.ones((480, 640, 3), dtype=np.int32) * 100
        # Add significant texture variation for rough surface
        rough_road[240:, :] += np.random.randint(-30, 30, rough_road[240:, :].shape)
        rough_road = np.clip(rough_road, 0, 255).astype(np.uint8)
        
        road_factor = self.analyzer.analyze_road_conditions(rough_road)
        
        # Should reduce speed on rough roads
        self.assertLess(road_factor, 0.9)
    
    def test_imu_based_road_condition_assessment(self):
        """Test road condition assessment using IMU data."""
        # Test with low vibration (good road)
        good_road_imu = {
            "accel_x": 0.1,
            "accel_y": 0.05,
            "accel_z": 9.81
        }
        
        road_factor_good = self.analyzer.analyze_road_conditions(
            np.ones((480, 640, 3), dtype=np.uint8) * 100,
            good_road_imu
        )
        
        # Test with high vibration (poor road)
        poor_road_imu = {
            "accel_x": 1.5,
            "accel_y": 1.2,
            "accel_z": 10.5
        }
        
        road_factor_poor = self.analyzer.analyze_road_conditions(
            np.ones((480, 640, 3), dtype=np.uint8) * 100,
            poor_road_imu
        )
        
        # Good road should allow higher speeds than poor road
        self.assertGreater(road_factor_good, road_factor_poor)
    
    def test_environmental_factor_combination(self):
        """Test combination of multiple environmental factors."""
        # Create challenging conditions
        dark_image = np.ones((480, 640, 3), dtype=np.uint8) * 30  # Dark
        multiple_vehicles = [
            {"type": "duckiebot", "distance": 1.0},
            {"type": "duckiebot", "distance": 1.5}
        ]
        rough_road_imu = {
            "accel_x": 1.0,
            "accel_y": 0.8,
            "accel_z": 10.2
        }
        
        factors = self.analyzer.get_environmental_speed_factor(
            dark_image, multiple_vehicles, rough_road_imu
        )
        
        # Overall factor should be most restrictive
        individual_factors = [
            factors["visibility_factor"],
            factors["traffic_density_factor"],
            factors["road_condition_factor"]
        ]
        
        self.assertEqual(factors["overall_environmental_factor"], min(individual_factors))
        
        # Should significantly reduce speed in challenging conditions
        self.assertLess(factors["overall_environmental_factor"], 0.7)
    
    def test_environmental_adaptation_smoothing(self):
        """Test that environmental analysis includes smoothing."""
        # Test visibility smoothing
        test_images = [
            np.ones((480, 640, 3), dtype=np.uint8) * 50,   # Dark
            np.ones((480, 640, 3), dtype=np.uint8) * 120,  # Good
            np.ones((480, 640, 3), dtype=np.uint8) * 60,   # Dim
        ]
        
        visibility_factors = []
        for image in test_images:
            factor = self.analyzer.analyze_visibility(image)
            visibility_factors.append(factor)
        
        # Later measurements should be influenced by history (smoothing)
        # The exact values depend on the smoothing algorithm, but there should be some continuity
        self.assertTrue(len(self.analyzer.visibility_history) > 0)
    
    def test_environmental_condition_detection_accuracy(self):
        """Test accuracy of environmental condition detection."""
        # Test brightness score calculation
        bright_score = self.analyzer._calculate_brightness_score(120)  # Good brightness
        dark_score = self.analyzer._calculate_brightness_score(20)     # Poor brightness
        
        self.assertGreater(bright_score, dark_score)
        
        # Test contrast score calculation
        high_contrast_score = self.analyzer._calculate_contrast_score(40)  # High contrast
        low_contrast_score = self.analyzer._calculate_contrast_score(5)    # Low contrast
        
        self.assertGreater(high_contrast_score, low_contrast_score)
    
    def test_vehicle_type_recognition(self):
        """Test vehicle type recognition for traffic density."""
        # Test with various object types
        test_objects = [
            {"type": "duckiebot", "distance": 1.0},      # Should be recognized
            {"type": "car", "distance": 1.0},            # Should be recognized
            {"type": "pedestrian", "distance": 1.0},     # Should not be recognized
            {"type": "cone", "distance": 1.0}            # Should not be recognized
        ]
        
        vehicle_count = 0
        for obj in test_objects:
            if self.analyzer._is_vehicle(obj):
                vehicle_count += 1
        
        # Should recognize 2 vehicles (duckiebot and car)
        self.assertEqual(vehicle_count, 2)
    
    def test_speed_response_to_environmental_changes(self):
        """Test speed response to changing environmental conditions."""
        base_speed = 0.3
        
        # Good conditions
        good_image = np.random.randint(100, 150, (480, 640, 3), dtype=np.uint8)
        good_factors = self.analyzer.get_environmental_speed_factor(good_image, [])
        good_speed = base_speed * good_factors["overall_environmental_factor"]
        
        # Poor conditions
        poor_image = np.ones((480, 640, 3), dtype=np.uint8) * 25  # Very dark
        poor_vehicles = [{"type": "duckiebot", "distance": 0.8}]  # Close vehicle
        poor_factors = self.analyzer.get_environmental_speed_factor(poor_image, poor_vehicles)
        poor_speed = base_speed * poor_factors["overall_environmental_factor"]
        
        # Speed should be significantly lower in poor conditions
        speed_reduction = (good_speed - poor_speed) / good_speed
        self.assertGreater(speed_reduction, 0.3)  # At least 30% reduction


if __name__ == '__main__':
    unittest.main()