#!/usr/bin/env python3

import unittest
import numpy as np
import time
import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'include'))

from predictive_perception.sensor_fusion import (
    SensorFusionEngine,
    CameraData,
    IMUData,
    EncoderData,
    FusedPerceptionData,
    OutlierDetector,
    TemporalConsistencyChecker,
    AdaptiveFusionWeights
)


class TestSensorFusion(unittest.TestCase):
    """Test cases for sensor fusion functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fusion_engine = SensorFusionEngine(
            fusion_window=0.1,
            max_data_age=1.0
        )
        
        # Test timestamps
        self.base_time = time.time()
        
        # Test sensor data
        self.test_detections = [
            {'class_name': 'duck', 'confidence': 0.8, 'position': [100, 200]},
            {'class_name': 'duckiebot', 'confidence': 0.9, 'position': [300, 400]}
        ]
    
    def test_sensor_data_creation(self):
        """Test creation of sensor data objects."""
        # Camera data
        camera_data = CameraData(
            timestamp=self.base_time,
            detections=self.test_detections,
            confidence=0.8
        )
        
        self.assertEqual(camera_data.timestamp, self.base_time)
        self.assertEqual(len(camera_data.detections), 2)
        self.assertEqual(camera_data.confidence, 0.8)
        
        # IMU data
        imu_data = IMUData(
            timestamp=self.base_time,
            angular_velocity=[0.1, 0.2, 0.3],
            linear_acceleration=[1.0, 2.0, 9.8],
            confidence=0.9
        )
        
        self.assertEqual(imu_data.timestamp, self.base_time)
        np.testing.assert_array_equal(imu_data.angular_velocity, [0.1, 0.2, 0.3])
        np.testing.assert_array_equal(imu_data.linear_acceleration, [1.0, 2.0, 9.8])
        
        # Encoder data
        encoder_data = EncoderData(
            timestamp=self.base_time,
            left_ticks=100,
            right_ticks=105,
            confidence=0.95
        )
        
        self.assertEqual(encoder_data.left_ticks, 100)
        self.assertEqual(encoder_data.right_ticks, 105)
    
    def test_sensor_data_buffering(self):
        """Test sensor data buffering in fusion engine."""
        # Add camera data
        camera_data = CameraData(
            timestamp=self.base_time,
            detections=self.test_detections,
            confidence=0.8
        )
        self.fusion_engine.add_camera_data(camera_data)
        
        self.assertEqual(len(self.fusion_engine.camera_buffer), 1)
        
        # Add IMU data
        imu_data = IMUData(
            timestamp=self.base_time,
            angular_velocity=[0.0, 0.0, 0.1],
            linear_acceleration=[0.0, 0.0, 9.8],
            confidence=0.9
        )
        self.fusion_engine.add_imu_data(imu_data)
        
        self.assertEqual(len(self.fusion_engine.imu_buffer), 1)
        
        # Add encoder data
        encoder_data = EncoderData(
            timestamp=self.base_time,
            left_ticks=100,
            right_ticks=100,
            confidence=0.95
        )
        self.fusion_engine.add_encoder_data(encoder_data)
        
        self.assertEqual(len(self.fusion_engine.encoder_buffer), 1)
    
    def test_data_synchronization(self):
        """Test sensor data synchronization."""
        # Add synchronized sensor data
        sync_time = self.base_time
        
        camera_data = CameraData(sync_time, self.test_detections, confidence=0.8)
        imu_data = IMUData(sync_time + 0.01, [0, 0, 0.1], [0, 0, 9.8], confidence=0.9)
        encoder_data = EncoderData(sync_time + 0.02, 100, 100, confidence=0.95)
        
        self.fusion_engine.add_camera_data(camera_data)
        self.fusion_engine.add_imu_data(imu_data)
        self.fusion_engine.add_encoder_data(encoder_data)
        
        # Test synchronization
        sync_data = self.fusion_engine._synchronize_sensor_data(sync_time)
        
        self.assertIn('camera', sync_data)
        self.assertIn('imu', sync_data)
        self.assertIn('encoder', sync_data)
    
    def test_ego_motion_estimation(self):
        """Test ego motion estimation from multiple sensors."""
        sync_time = self.base_time
        
        # Create test data with known motion
        imu_data = IMUData(
            sync_time, 
            angular_velocity=[0, 0, 0.5],  # 0.5 rad/s rotation
            linear_acceleration=[1.0, 0, 9.8], 
            confidence=0.9
        )
        
        encoder_data = EncoderData(sync_time, 100, 110, confidence=0.95)
        encoder_data.wheel_velocities = [0.2, 0.3]  # Different wheel speeds
        
        self.fusion_engine.add_imu_data(imu_data)
        self.fusion_engine.add_encoder_data(encoder_data)
        
        # Test ego motion estimation
        sync_data = {'imu': imu_data, 'encoder': encoder_data}
        ego_motion = self.fusion_engine._fuse_ego_motion(sync_data)
        
        self.assertIn('linear_velocity', ego_motion)
        self.assertIn('angular_velocity', ego_motion)
        self.assertIn('confidence', ego_motion)
        
        # Check that motion values are reasonable
        self.assertGreater(ego_motion['confidence'], 0.0)
    
    def test_object_detection_fusion(self):
        """Test object detection fusion with motion compensation."""
        sync_time = self.base_time
        
        camera_data = CameraData(sync_time, self.test_detections, confidence=0.8)
        imu_data = IMUData(sync_time, [0, 0, 0.1], [0, 0, 9.8], confidence=0.9)
        
        self.fusion_engine.add_camera_data(camera_data)
        self.fusion_engine.add_imu_data(imu_data)
        
        sync_data = {'camera': camera_data, 'imu': imu_data}
        fused_objects = self.fusion_engine._fuse_object_detections(sync_data)
        
        # Should have fused objects
        self.assertGreater(len(fused_objects), 0)
        
        # Objects should have enhanced information
        for obj in fused_objects:
            self.assertIn('class_name', obj)
            self.assertIn('confidence', obj)
    
    def test_complete_fusion_workflow(self):
        """Test complete sensor fusion workflow."""
        sync_time = self.base_time
        
        # Add all sensor types
        camera_data = CameraData(sync_time, self.test_detections, confidence=0.8)
        imu_data = IMUData(sync_time, [0, 0, 0.1], [0, 0, 9.8], confidence=0.9)
        encoder_data = EncoderData(sync_time, 100, 105, confidence=0.95)
        
        self.fusion_engine.add_camera_data(camera_data)
        self.fusion_engine.add_imu_data(imu_data)
        self.fusion_engine.add_encoder_data(encoder_data)
        
        # Perform fusion
        fused_data = self.fusion_engine.fuse_sensor_data(sync_time)
        
        # Check fused data structure
        self.assertIsInstance(fused_data, FusedPerceptionData)
        self.assertEqual(fused_data.timestamp, sync_time)
        self.assertIsInstance(fused_data.objects, list)
        self.assertIsInstance(fused_data.confidence_metrics, dict)
        self.assertIsInstance(fused_data.sensor_health, dict)
    
    def test_confidence_metrics_calculation(self):
        """Test confidence metrics calculation."""
        sync_time = self.base_time
        
        # Create sync data with all sensors
        camera_data = CameraData(sync_time, self.test_detections, confidence=0.8)
        imu_data = IMUData(sync_time, [0, 0, 0.1], [0, 0, 9.8], confidence=0.9)
        encoder_data = EncoderData(sync_time, 100, 105, confidence=0.95)
        
        sync_data = {
            'camera': camera_data,
            'imu': imu_data,
            'encoder': encoder_data
        }
        
        metrics = self.fusion_engine._calculate_confidence_metrics(sync_data)
        
        # Check required metrics
        required_metrics = [
            'overall_confidence', 'sensor_agreement', 
            'temporal_consistency', 'data_completeness'
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertGreaterEqual(metrics[metric], 0.0)
            self.assertLessEqual(metrics[metric], 1.0)
        
        # Data completeness should be 1.0 (all sensors present)
        self.assertEqual(metrics['data_completeness'], 1.0)
    
    def test_sensor_health_assessment(self):
        """Test sensor health assessment."""
        sync_time = self.base_time
        
        # Create sensors with different health levels
        healthy_camera = CameraData(sync_time, self.test_detections, confidence=0.9)
        degraded_imu = IMUData(sync_time, [0, 0, 0.1], [0, 0, 9.8], confidence=0.3)
        
        sync_data = {
            'camera': healthy_camera,
            'imu': degraded_imu
        }
        
        health = self.fusion_engine._assess_sensor_health(sync_data)
        
        # Check health assessment
        self.assertIn('camera', health)
        self.assertIn('imu', health)
        
        self.assertEqual(health['camera']['status'], 'healthy')
        self.assertEqual(health['imu']['status'], 'degraded')
    
    def test_old_data_cleanup(self):
        """Test cleanup of old sensor data."""
        # Temporarily disable automatic cleanup by setting max_data_age very high
        original_max_age = self.fusion_engine.max_data_age
        self.fusion_engine.max_data_age = 10.0  # 10 seconds
        
        # Add old data
        old_time = self.base_time - 2.0  # 2 seconds old
        
        old_camera_data = CameraData(old_time, self.test_detections, confidence=0.8)
        self.fusion_engine.add_camera_data(old_camera_data)
        
        # Add recent data
        recent_time = self.base_time
        recent_camera_data = CameraData(recent_time, self.test_detections, confidence=0.8)
        self.fusion_engine.add_camera_data(recent_camera_data)
        
        # Should have both now
        self.assertEqual(len(self.fusion_engine.camera_buffer), 2)
        
        # Restore original max age and cleanup
        self.fusion_engine.max_data_age = original_max_age
        self.fusion_engine._cleanup_old_data()
        
        # Should only have recent data
        self.assertEqual(len(self.fusion_engine.camera_buffer), 1)
        self.assertEqual(self.fusion_engine.camera_buffer[0].timestamp, recent_time)
    
    def test_weighted_fusion(self):
        """Test weighted fusion of motion estimates."""
        estimates = [
            {'linear_velocity': 0.2, 'angular_velocity': 0.1, 'confidence': 0.8},
            {'linear_velocity': 0.3, 'angular_velocity': 0.2, 'confidence': 0.9},
            {'linear_velocity': 0.25, 'angular_velocity': 0.15, 'confidence': 0.7}
        ]
        
        weights = [0.5, 0.3, 0.2]
        
        fused = self.fusion_engine._weighted_fusion(estimates, weights)
        
        # Check fused result structure
        self.assertIn('linear_velocity', fused)
        self.assertIn('angular_velocity', fused)
        self.assertIn('confidence', fused)
        
        # Fused values should be weighted averages
        expected_linear_vel = (0.2 * 0.5 + 0.3 * 0.3 + 0.25 * 0.2)
        self.assertAlmostEqual(fused['linear_velocity'], expected_linear_vel, places=3)


class TestOutlierDetector(unittest.TestCase):
    """Test cases for outlier detection."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.detector = OutlierDetector(threshold=3.0)
    
    def test_outlier_detection_initialization(self):
        """Test outlier detector initialization."""
        self.assertEqual(self.detector.threshold, 3.0)
        self.assertEqual(len(self.detector.history), 0)
    
    def test_outlier_detection_with_insufficient_history(self):
        """Test outlier detection with insufficient history."""
        measurement = {'value': 10.0}
        
        # Should not be outlier with insufficient history
        is_outlier = self.detector.is_outlier(measurement)
        self.assertFalse(is_outlier)
    
    def test_history_update(self):
        """Test measurement history update."""
        measurements = [{'value': i} for i in range(5)]
        
        for measurement in measurements:
            self.detector.update_history(measurement)
        
        self.assertEqual(len(self.detector.history), 5)
        
        # Add more measurements to test max history limit
        for i in range(10):
            self.detector.update_history({'value': i + 10})
        
        # Should not exceed max history
        self.assertLessEqual(len(self.detector.history), self.detector.max_history)


class TestTemporalConsistencyChecker(unittest.TestCase):
    """Test cases for temporal consistency checking."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.checker = TemporalConsistencyChecker()
    
    def test_consistency_checker_initialization(self):
        """Test consistency checker initialization."""
        self.assertEqual(len(self.checker.previous_measurements), 0)
        self.assertEqual(self.checker.consistency_score, 1.0)
    
    def test_consistency_check(self):
        """Test consistency checking."""
        measurement = {'value': 10.0, 'timestamp': time.time()}
        
        # Should be consistent (placeholder implementation)
        is_consistent = self.checker.is_consistent(measurement)
        self.assertTrue(is_consistent)
    
    def test_consistency_score(self):
        """Test consistency score retrieval."""
        score = self.checker.get_consistency_score()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


class TestAdaptiveFusionWeights(unittest.TestCase):
    """Test cases for adaptive fusion weights."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.adaptive_weights = AdaptiveFusionWeights()
    
    def test_adaptive_weights_initialization(self):
        """Test adaptive weights initialization."""
        weights = self.adaptive_weights.get_weights()
        
        # Should have weights for all sensors
        self.assertIn('camera', weights)
        self.assertIn('imu', weights)
        self.assertIn('encoder', weights)
        
        # Weights should sum to approximately 1.0
        total_weight = sum(weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=3)
    
    def test_weight_updates(self):
        """Test weight updates based on performance."""
        # Simulate performance data
        performance = {
            'camera': 0.9,  # High performance
            'imu': 0.5,     # Medium performance
            'encoder': 0.8  # Good performance
        }
        
        initial_weights = self.adaptive_weights.get_weights().copy()
        
        # Update weights
        self.adaptive_weights.update_weights(performance)
        
        updated_weights = self.adaptive_weights.get_weights()
        
        # Weights should still sum to approximately 1.0
        total_weight = sum(updated_weights.values())
        self.assertAlmostEqual(total_weight, 1.0, places=3)
        
        # All weights should be positive
        for weight in updated_weights.values():
            self.assertGreater(weight, 0.0)
    
    def test_performance_history_management(self):
        """Test performance history management."""
        # Add many performance updates
        for i in range(15):
            performance = {
                'camera': 0.8 + i * 0.01,
                'imu': 0.6 + i * 0.01,
                'encoder': 0.7 + i * 0.01
            }
            self.adaptive_weights.update_weights(performance)
        
        # History should be limited
        for sensor_history in self.adaptive_weights.performance_history.values():
            self.assertLessEqual(len(sensor_history), 10)


if __name__ == '__main__':
    unittest.main()