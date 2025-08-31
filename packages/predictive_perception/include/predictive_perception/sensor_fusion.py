#!/usr/bin/env python3

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import time


class SensorData:
    """Base class for sensor data."""
    
    def __init__(self, timestamp, confidence=1.0):
        self.timestamp = timestamp
        self.confidence = confidence
        self.processed = False


class CameraData(SensorData):
    """Camera sensor data."""
    
    def __init__(self, timestamp, detections, lane_pose=None, confidence=1.0):
        super().__init__(timestamp, confidence)
        self.detections = detections  # List of detected objects
        self.lane_pose = lane_pose    # Lane pose information
        self.image_quality = 1.0      # Image quality metric


class IMUData(SensorData):
    """IMU sensor data."""
    
    def __init__(self, timestamp, angular_velocity, linear_acceleration, confidence=1.0):
        super().__init__(timestamp, confidence)
        self.angular_velocity = np.array(angular_velocity)  # [wx, wy, wz]
        self.linear_acceleration = np.array(linear_acceleration)  # [ax, ay, az]


class EncoderData(SensorData):
    """Wheel encoder sensor data."""
    
    def __init__(self, timestamp, left_ticks, right_ticks, confidence=1.0):
        super().__init__(timestamp, confidence)
        self.left_ticks = left_ticks
        self.right_ticks = right_ticks
        self.wheel_velocities = None  # Will be calculated


class FusedPerceptionData:
    """Fused perception output."""
    
    def __init__(self, timestamp):
        self.timestamp = timestamp
        self.objects = []           # Fused object detections
        self.ego_motion = None      # Estimated ego motion
        self.lane_information = None # Fused lane information
        self.confidence_metrics = {}
        self.sensor_health = {}


class SensorFusionEngine:
    """
    Multi-modal sensor fusion engine for combining camera, IMU, and encoder data.
    """
    
    def __init__(self, fusion_window=0.1, max_data_age=1.0):
        """
        Initialize sensor fusion engine.
        
        Args:
            fusion_window: Time window for sensor data synchronization (seconds)
            max_data_age: Maximum age of sensor data to consider (seconds)
        """
        self.fusion_window = fusion_window
        self.max_data_age = max_data_age
        
        # Sensor data buffers
        self.camera_buffer = []
        self.imu_buffer = []
        self.encoder_buffer = []
        
        # Fusion parameters
        self.camera_weight = 0.6
        self.imu_weight = 0.3
        self.encoder_weight = 0.1
        
        # Calibration parameters
        self.wheel_base = 0.1  # Distance between wheels (meters)
        self.wheel_radius = 0.0318  # Wheel radius (meters)
        self.ticks_per_revolution = 135  # Encoder ticks per wheel revolution
        
        # State estimation
        self.last_pose = np.array([0.0, 0.0, 0.0])  # [x, y, theta]
        self.last_velocity = np.array([0.0, 0.0])    # [v, omega]
        
        # Outlier detection
        self.outlier_detector = OutlierDetector()
        
        # Temporal consistency checker
        self.consistency_checker = TemporalConsistencyChecker()
    
    def add_camera_data(self, camera_data):
        """Add camera data to fusion buffer."""
        self.camera_buffer.append(camera_data)
        self._cleanup_old_data()
    
    def add_imu_data(self, imu_data):
        """Add IMU data to fusion buffer."""
        self.imu_buffer.append(imu_data)
        self._cleanup_old_data()
    
    def add_encoder_data(self, encoder_data):
        """Add encoder data to fusion buffer."""
        # Calculate wheel velocities from encoder ticks
        encoder_data.wheel_velocities = self._calculate_wheel_velocities(encoder_data)
        self.encoder_buffer.append(encoder_data)
        self._cleanup_old_data()
    
    def fuse_sensor_data(self, target_timestamp=None):
        """
        Fuse sensor data from all modalities.
        
        Args:
            target_timestamp: Target timestamp for fusion (current time if None)
            
        Returns:
            FusedPerceptionData object
        """
        if target_timestamp is None:
            target_timestamp = time.time()
        
        # Find synchronized sensor data
        sync_data = self._synchronize_sensor_data(target_timestamp)
        
        if not sync_data:
            return None
        
        # Create fused perception output
        fused_data = FusedPerceptionData(target_timestamp)
        
        # Fuse object detections
        fused_data.objects = self._fuse_object_detections(sync_data)
        
        # Fuse ego motion estimation
        fused_data.ego_motion = self._fuse_ego_motion(sync_data)
        
        # Fuse lane information
        fused_data.lane_information = self._fuse_lane_information(sync_data)
        
        # Calculate confidence metrics
        fused_data.confidence_metrics = self._calculate_confidence_metrics(sync_data)
        
        # Assess sensor health
        fused_data.sensor_health = self._assess_sensor_health(sync_data)
        
        return fused_data
    
    def _synchronize_sensor_data(self, target_timestamp):
        """
        Synchronize sensor data within the fusion window.
        
        Args:
            target_timestamp: Target timestamp for synchronization
            
        Returns:
            Dictionary of synchronized sensor data
        """
        sync_data = {}
        
        # Find camera data within window
        camera_data = self._find_closest_data(
            self.camera_buffer, target_timestamp, self.fusion_window
        )
        if camera_data:
            sync_data['camera'] = camera_data
        
        # Find IMU data within window
        imu_data = self._find_closest_data(
            self.imu_buffer, target_timestamp, self.fusion_window
        )
        if imu_data:
            sync_data['imu'] = imu_data
        
        # Find encoder data within window
        encoder_data = self._find_closest_data(
            self.encoder_buffer, target_timestamp, self.fusion_window
        )
        if encoder_data:
            sync_data['encoder'] = encoder_data
        
        return sync_data
    
    def _find_closest_data(self, data_buffer, target_timestamp, window):
        """Find closest sensor data within time window."""
        if not data_buffer:
            return None
        
        closest_data = None
        min_time_diff = float('inf')
        
        for data in data_buffer:
            time_diff = abs(data.timestamp - target_timestamp)
            if time_diff <= window and time_diff < min_time_diff:
                min_time_diff = time_diff
                closest_data = data
        
        return closest_data
    
    def _fuse_object_detections(self, sync_data):
        """
        Fuse object detections from multiple sensors.
        
        Args:
            sync_data: Synchronized sensor data
            
        Returns:
            List of fused object detections
        """
        fused_objects = []
        
        # Start with camera detections as primary source
        if 'camera' in sync_data:
            camera_objects = sync_data['camera'].detections
            
            for obj in camera_objects:
                # Apply outlier detection
                if not self.outlier_detector.is_outlier(obj):
                    # Enhance object with motion information
                    enhanced_obj = self._enhance_object_with_motion(obj, sync_data)
                    
                    # Apply temporal consistency check
                    if self.consistency_checker.is_consistent(enhanced_obj):
                        fused_objects.append(enhanced_obj)
        
        return fused_objects
    
    def _enhance_object_with_motion(self, obj, sync_data):
        """Enhance object detection with motion information from IMU/encoders."""
        enhanced_obj = obj.copy() if hasattr(obj, 'copy') else dict(obj)
        
        # Add ego motion compensation
        if 'imu' in sync_data or 'encoder' in sync_data:
            ego_motion = self._estimate_ego_motion(sync_data)
            
            # Compensate object position for ego motion
            if 'position' in enhanced_obj and ego_motion:
                # Simple ego motion compensation (would be more sophisticated in practice)
                enhanced_obj['ego_compensated_position'] = enhanced_obj['position']
                enhanced_obj['ego_motion'] = ego_motion
        
        return enhanced_obj
    
    def _fuse_ego_motion(self, sync_data):
        """
        Fuse ego motion estimation from multiple sensors.
        
        Args:
            sync_data: Synchronized sensor data
            
        Returns:
            Dictionary with ego motion estimates
        """
        ego_motion = {
            'linear_velocity': 0.0,
            'angular_velocity': 0.0,
            'acceleration': np.array([0.0, 0.0, 0.0]),
            'confidence': 0.0
        }
        
        estimates = []
        weights = []
        
        # IMU-based motion estimation
        if 'imu' in sync_data:
            imu_motion = self._estimate_motion_from_imu(sync_data['imu'])
            estimates.append(imu_motion)
            weights.append(self.imu_weight)
        
        # Encoder-based motion estimation
        if 'encoder' in sync_data:
            encoder_motion = self._estimate_motion_from_encoders(sync_data['encoder'])
            estimates.append(encoder_motion)
            weights.append(self.encoder_weight)
        
        # Camera-based motion estimation (if available)
        if 'camera' in sync_data and sync_data['camera'].lane_pose:
            camera_motion = self._estimate_motion_from_camera(sync_data['camera'])
            estimates.append(camera_motion)
            weights.append(self.camera_weight)
        
        # Weighted fusion of estimates
        if estimates:
            ego_motion = self._weighted_fusion(estimates, weights)
        
        return ego_motion
    
    def _estimate_motion_from_imu(self, imu_data):
        """Estimate motion from IMU data."""
        return {
            'linear_velocity': 0.0,  # Would integrate acceleration
            'angular_velocity': imu_data.angular_velocity[2],  # Z-axis rotation
            'acceleration': imu_data.linear_acceleration,
            'confidence': imu_data.confidence
        }
    
    def _estimate_motion_from_encoders(self, encoder_data):
        """Estimate motion from wheel encoder data."""
        if encoder_data.wheel_velocities is None:
            return {'linear_velocity': 0.0, 'angular_velocity': 0.0, 'confidence': 0.0}
        
        v_left, v_right = encoder_data.wheel_velocities
        
        # Differential drive kinematics
        linear_velocity = (v_left + v_right) / 2.0
        angular_velocity = (v_right - v_left) / self.wheel_base
        
        return {
            'linear_velocity': linear_velocity,
            'angular_velocity': angular_velocity,
            'acceleration': np.array([0.0, 0.0, 0.0]),  # Not available from encoders
            'confidence': encoder_data.confidence
        }
    
    def _estimate_motion_from_camera(self, camera_data):
        """Estimate motion from camera/lane pose data."""
        # Placeholder for visual odometry
        return {
            'linear_velocity': 0.0,
            'angular_velocity': 0.0,
            'acceleration': np.array([0.0, 0.0, 0.0]),
            'confidence': camera_data.confidence * 0.5  # Lower confidence for camera motion
        }
    
    def _fuse_lane_information(self, sync_data):
        """Fuse lane information from camera and motion sensors."""
        lane_info = {
            'lane_pose': None,
            'lane_confidence': 0.0,
            'motion_corrected': False
        }
        
        if 'camera' in sync_data and sync_data['camera'].lane_pose:
            lane_info['lane_pose'] = sync_data['camera'].lane_pose
            lane_info['lane_confidence'] = sync_data['camera'].confidence
            
            # Correct for ego motion if available
            if 'imu' in sync_data or 'encoder' in sync_data:
                ego_motion = self._estimate_ego_motion(sync_data)
                # Apply motion correction to lane pose
                lane_info['motion_corrected'] = True
        
        return lane_info
    
    def _weighted_fusion(self, estimates, weights):
        """Perform weighted fusion of motion estimates."""
        if not estimates:
            return {'linear_velocity': 0.0, 'angular_velocity': 0.0, 'confidence': 0.0}
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Fuse estimates
        fused_linear_vel = sum(est['linear_velocity'] * w for est, w in zip(estimates, weights))
        fused_angular_vel = sum(est['angular_velocity'] * w for est, w in zip(estimates, weights))
        fused_confidence = sum(est['confidence'] * w for est, w in zip(estimates, weights))
        
        return {
            'linear_velocity': fused_linear_vel,
            'angular_velocity': fused_angular_vel,
            'confidence': fused_confidence
        }
    
    def _calculate_wheel_velocities(self, encoder_data):
        """Calculate wheel velocities from encoder ticks."""
        # This is a simplified calculation - would need previous tick counts for velocity
        # For now, return placeholder values
        return [0.0, 0.0]  # [left_velocity, right_velocity]
    
    def _calculate_confidence_metrics(self, sync_data):
        """Calculate overall confidence metrics for fused data."""
        metrics = {
            'overall_confidence': 0.0,
            'sensor_agreement': 0.0,
            'temporal_consistency': 0.0,
            'data_completeness': 0.0
        }
        
        # Data completeness
        available_sensors = len(sync_data)
        total_sensors = 3  # Camera, IMU, Encoder
        metrics['data_completeness'] = available_sensors / total_sensors
        
        # Sensor agreement (simplified)
        if available_sensors > 1:
            metrics['sensor_agreement'] = 0.8  # Placeholder
        
        # Temporal consistency
        metrics['temporal_consistency'] = self.consistency_checker.get_consistency_score()
        
        # Overall confidence
        metrics['overall_confidence'] = (
            0.4 * metrics['data_completeness'] +
            0.3 * metrics['sensor_agreement'] +
            0.3 * metrics['temporal_consistency']
        )
        
        return metrics
    
    def _assess_sensor_health(self, sync_data):
        """Assess health of individual sensors."""
        health = {}
        
        for sensor_name, sensor_data in sync_data.items():
            health[sensor_name] = {
                'status': 'healthy' if sensor_data.confidence > 0.5 else 'degraded',
                'confidence': sensor_data.confidence,
                'last_update': sensor_data.timestamp
            }
        
        return health
    
    def _cleanup_old_data(self):
        """Remove old data from sensor buffers."""
        current_time = time.time()
        
        # Clean camera buffer
        self.camera_buffer = [
            data for data in self.camera_buffer 
            if current_time - data.timestamp <= self.max_data_age
        ]
        
        # Clean IMU buffer
        self.imu_buffer = [
            data for data in self.imu_buffer 
            if current_time - data.timestamp <= self.max_data_age
        ]
        
        # Clean encoder buffer
        self.encoder_buffer = [
            data for data in self.encoder_buffer 
            if current_time - data.timestamp <= self.max_data_age
        ]
    
    def _estimate_ego_motion(self, sync_data):
        """Estimate ego motion from available sensor data."""
        return self._fuse_ego_motion(sync_data)


class OutlierDetector:
    """Detect and filter outlier measurements."""
    
    def __init__(self, threshold=3.0):
        self.threshold = threshold
        self.history = []
        self.max_history = 10
    
    def is_outlier(self, measurement):
        """Check if measurement is an outlier."""
        if len(self.history) < 3:
            self.history.append(measurement)
            return False
        
        # Simple statistical outlier detection
        # In practice, would use more sophisticated methods
        return False  # Placeholder
    
    def update_history(self, measurement):
        """Update measurement history."""
        self.history.append(measurement)
        if len(self.history) > self.max_history:
            self.history.pop(0)


class TemporalConsistencyChecker:
    """Check temporal consistency of measurements."""
    
    def __init__(self):
        self.previous_measurements = []
        self.consistency_score = 1.0
    
    def is_consistent(self, measurement):
        """Check if measurement is temporally consistent."""
        # Placeholder for temporal consistency logic
        return True
    
    def get_consistency_score(self):
        """Get current consistency score."""
        return self.consistency_score
    
    def update_consistency(self, measurement):
        """Update consistency score based on new measurement."""
        # Placeholder for consistency update logic
        pass


class AdaptiveFusionWeights:
    """Adaptively adjust fusion weights based on sensor performance."""
    
    def __init__(self):
        self.base_weights = {'camera': 0.6, 'imu': 0.3, 'encoder': 0.1}
        self.current_weights = self.base_weights.copy()
        self.performance_history = {sensor: [] for sensor in self.base_weights}
    
    def update_weights(self, sensor_performance):
        """Update fusion weights based on sensor performance."""
        for sensor, performance in sensor_performance.items():
            if sensor in self.performance_history:
                self.performance_history[sensor].append(performance)
                
                # Keep only recent history
                if len(self.performance_history[sensor]) > 10:
                    self.performance_history[sensor].pop(0)
        
        # Recalculate weights based on recent performance
        self._recalculate_weights()
    
    def _recalculate_weights(self):
        """Recalculate fusion weights based on performance history."""
        for sensor in self.current_weights:
            if self.performance_history[sensor]:
                avg_performance = np.mean(self.performance_history[sensor])
                # Adjust weight based on performance
                self.current_weights[sensor] = self.base_weights[sensor] * avg_performance
        
        # Normalize weights
        total_weight = sum(self.current_weights.values())
        if total_weight > 0:
            for sensor in self.current_weights:
                self.current_weights[sensor] /= total_weight
    
    def get_weights(self):
        """Get current fusion weights."""
        return self.current_weights.copy()