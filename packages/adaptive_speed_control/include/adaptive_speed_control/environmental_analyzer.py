#!/usr/bin/env python3

import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional
import rospy


class EnvironmentalAnalyzer:
    """
    Analyzes environmental conditions to determine appropriate speed adjustments.
    
    This class processes sensor data to assess visibility, traffic density, and road
    conditions, providing speed adjustment factors for the adaptive speed controller.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the environmental analyzer.
        
        Args:
            config (Dict): Configuration parameters
        """
        self.config = config
        
        # Visibility analysis parameters
        self.brightness_threshold = config.get("visibility", {}).get("brightness_threshold", 50)
        self.contrast_threshold = config.get("visibility", {}).get("contrast_threshold", 20)
        self.min_visibility_threshold = config.get("visibility", {}).get("min_visibility_threshold", 0.3)
        self.max_speed_reduction = config.get("visibility", {}).get("max_speed_reduction", 0.7)
        
        # Traffic density parameters
        self.detection_distance = config.get("traffic_density", {}).get("detection_distance", 2.0)
        self.max_vehicles_threshold = config.get("traffic_density", {}).get("max_vehicles_threshold", 3)
        self.speed_reduction_per_vehicle = config.get("traffic_density", {}).get("speed_reduction_per_vehicle", 0.15)
        
        # Road condition parameters
        self.surface_quality_threshold = config.get("road_conditions", {}).get("surface_quality_threshold", 0.7)
        self.weather_impact_factor = config.get("road_conditions", {}).get("weather_impact_factor", 0.8)
        
        # Analysis history for smoothing
        self.visibility_history = []
        self.traffic_history = []
        self.road_condition_history = []
        self.history_length = 10
        
        rospy.loginfo("Environmental Analyzer initialized")
    
    def analyze_visibility(self, image: np.ndarray) -> float:
        """
        Analyze image visibility conditions.
        
        Args:
            image (np.ndarray): Input camera image
            
        Returns:
            float: Visibility factor (0.0 = poor visibility, 1.0 = excellent visibility)
        """
        if image is None or image.size == 0:
            return 0.5  # Default moderate visibility
        
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Calculate brightness (mean intensity)
            brightness = np.mean(gray)
            
            # Calculate contrast (standard deviation of intensity)
            contrast = np.std(gray)
            
            # Calculate visibility score based on brightness and contrast
            brightness_score = self._calculate_brightness_score(brightness)
            contrast_score = self._calculate_contrast_score(contrast)
            
            # Combine scores (weighted average)
            visibility_score = 0.6 * brightness_score + 0.4 * contrast_score
            
            # Apply smoothing
            self.visibility_history.append(visibility_score)
            if len(self.visibility_history) > self.history_length:
                self.visibility_history.pop(0)
            
            smoothed_visibility = np.mean(self.visibility_history)
            
            # Ensure visibility is within valid range
            visibility_factor = np.clip(smoothed_visibility, 0.0, 1.0)
            
            return visibility_factor
            
        except Exception as e:
            rospy.logwarn(f"Error analyzing visibility: {e}")
            return 0.5  # Default moderate visibility
    
    def _calculate_brightness_score(self, brightness: float) -> float:
        """
        Calculate visibility score based on image brightness.
        
        Args:
            brightness (float): Mean image brightness (0-255)
            
        Returns:
            float: Brightness-based visibility score (0.0-1.0)
        """
        # Optimal brightness range is around 80-180
        if brightness < 30:  # Too dark
            return 0.2
        elif brightness < 50:  # Dark
            return 0.4
        elif brightness < 80:  # Dim
            return 0.7
        elif brightness <= 180:  # Good
            return 1.0
        elif brightness <= 220:  # Bright
            return 0.8
        else:  # Too bright (overexposed)
            return 0.3
    
    def _calculate_contrast_score(self, contrast: float) -> float:
        """
        Calculate visibility score based on image contrast.
        
        Args:
            contrast (float): Image contrast (standard deviation)
            
        Returns:
            float: Contrast-based visibility score (0.0-1.0)
        """
        # Higher contrast generally means better visibility
        if contrast < 10:  # Very low contrast
            return 0.2
        elif contrast < 20:  # Low contrast
            return 0.5
        elif contrast < 40:  # Moderate contrast
            return 0.8
        else:  # High contrast
            return 1.0
    
    def analyze_traffic_density(self, detected_objects: List[Dict]) -> float:
        """
        Analyze traffic density based on detected objects.
        
        Args:
            detected_objects (List[Dict]): List of detected objects with positions
            
        Returns:
            float: Traffic density factor (0.0 = heavy traffic, 1.0 = no traffic)
        """
        if not detected_objects:
            return 1.0  # No traffic detected
        
        try:
            # Count vehicles within detection distance
            nearby_vehicles = 0
            
            for obj in detected_objects:
                # Check if object is a vehicle (duckiebot, car, etc.)
                if self._is_vehicle(obj):
                    distance = obj.get("distance", float('inf'))
                    if distance <= self.detection_distance:
                        nearby_vehicles += 1
            
            # Calculate traffic density factor
            if nearby_vehicles == 0:
                traffic_factor = 1.0
            elif nearby_vehicles >= self.max_vehicles_threshold:
                traffic_factor = 1.0 - (self.max_vehicles_threshold * self.speed_reduction_per_vehicle)
            else:
                traffic_factor = 1.0 - (nearby_vehicles * self.speed_reduction_per_vehicle)
            
            # Apply smoothing
            self.traffic_history.append(traffic_factor)
            if len(self.traffic_history) > self.history_length:
                self.traffic_history.pop(0)
            
            smoothed_traffic = np.mean(self.traffic_history)
            
            # Ensure factor is within valid range
            traffic_density_factor = np.clip(smoothed_traffic, 0.1, 1.0)
            
            return traffic_density_factor
            
        except Exception as e:
            rospy.logwarn(f"Error analyzing traffic density: {e}")
            return 0.8  # Default moderate traffic
    
    def _is_vehicle(self, obj: Dict) -> bool:
        """
        Determine if detected object is a vehicle.
        
        Args:
            obj (Dict): Detected object information
            
        Returns:
            bool: True if object is likely a vehicle
        """
        # Check object type/class
        obj_type = obj.get("type", "").lower()
        vehicle_types = ["duckiebot", "car", "vehicle", "bot", "robot"]
        
        return any(vehicle_type in obj_type for vehicle_type in vehicle_types)
    
    def analyze_road_conditions(self, image: np.ndarray, imu_data: Optional[Dict] = None) -> float:
        """
        Analyze road surface conditions.
        
        Args:
            image (np.ndarray): Camera image for visual road assessment
            imu_data (Optional[Dict]): IMU data for vibration analysis
            
        Returns:
            float: Road condition factor (0.0 = poor conditions, 1.0 = excellent conditions)
        """
        try:
            # Visual road surface analysis
            surface_quality = self._analyze_surface_quality(image)
            
            # IMU-based vibration analysis (if available)
            vibration_quality = 1.0
            if imu_data:
                vibration_quality = self._analyze_vibration(imu_data)
            
            # Combine assessments
            road_condition = 0.7 * surface_quality + 0.3 * vibration_quality
            
            # Apply smoothing
            self.road_condition_history.append(road_condition)
            if len(self.road_condition_history) > self.history_length:
                self.road_condition_history.pop(0)
            
            smoothed_condition = np.mean(self.road_condition_history)
            
            # Ensure factor is within valid range
            road_condition_factor = np.clip(smoothed_condition, 0.1, 1.0)
            
            return road_condition_factor
            
        except Exception as e:
            rospy.logwarn(f"Error analyzing road conditions: {e}")
            return 0.8  # Default good road conditions
    
    def _analyze_surface_quality(self, image: np.ndarray) -> float:
        """
        Analyze road surface quality from camera image.
        
        Args:
            image (np.ndarray): Camera image
            
        Returns:
            float: Surface quality score (0.0-1.0)
        """
        if image is None or image.size == 0:
            return 0.8  # Default good surface
        
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Focus on road area (bottom half of image)
            height = gray.shape[0]
            road_area = gray[height//2:, :]
            
            # Calculate texture metrics
            # High variance indicates rough/damaged surface
            texture_variance = np.var(road_area)
            
            # Edge density indicates surface irregularities
            edges = cv2.Canny(road_area, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            # Calculate surface quality score
            # Lower variance and edge density = better surface
            variance_score = 1.0 - min(texture_variance / 1000.0, 1.0)
            edge_score = 1.0 - min(edge_density * 10.0, 1.0)
            
            surface_quality = 0.6 * variance_score + 0.4 * edge_score
            
            return np.clip(surface_quality, 0.0, 1.0)
            
        except Exception as e:
            rospy.logwarn(f"Error analyzing surface quality: {e}")
            return 0.8
    
    def _analyze_vibration(self, imu_data: Dict) -> float:
        """
        Analyze road conditions based on IMU vibration data.
        
        Args:
            imu_data (Dict): IMU acceleration data
            
        Returns:
            float: Vibration-based road quality score (0.0-1.0)
        """
        try:
            # Extract acceleration data
            accel_x = imu_data.get("accel_x", 0.0)
            accel_y = imu_data.get("accel_y", 0.0)
            accel_z = imu_data.get("accel_z", 9.81)  # Default gravity
            
            # Calculate vibration magnitude (excluding gravity)
            vibration_magnitude = np.sqrt(accel_x**2 + accel_y**2 + (accel_z - 9.81)**2)
            
            # Convert vibration to quality score
            # Lower vibration = better road conditions
            if vibration_magnitude < 0.5:
                return 1.0  # Excellent road
            elif vibration_magnitude < 1.0:
                return 0.8  # Good road
            elif vibration_magnitude < 2.0:
                return 0.6  # Fair road
            else:
                return 0.3  # Poor road
                
        except Exception as e:
            rospy.logwarn(f"Error analyzing vibration: {e}")
            return 0.8
    
    def get_environmental_speed_factor(self, image: np.ndarray, 
                                     detected_objects: List[Dict],
                                     imu_data: Optional[Dict] = None) -> Dict[str, float]:
        """
        Get comprehensive environmental speed adjustment factors.
        
        Args:
            image (np.ndarray): Camera image
            detected_objects (List[Dict]): Detected objects
            imu_data (Optional[Dict]): IMU data
            
        Returns:
            Dict[str, float]: Environmental factors and overall speed factor
        """
        # Analyze individual environmental factors
        visibility_factor = self.analyze_visibility(image)
        traffic_factor = self.analyze_traffic_density(detected_objects)
        road_factor = self.analyze_road_conditions(image, imu_data)
        
        # Calculate overall environmental speed factor
        # Use the most restrictive factor to ensure safety
        overall_factor = min(visibility_factor, traffic_factor, road_factor)
        
        return {
            "visibility_factor": visibility_factor,
            "traffic_density_factor": traffic_factor,
            "road_condition_factor": road_factor,
            "overall_environmental_factor": overall_factor
        }
    
    def reset_history(self):
        """Reset analysis history for fresh start."""
        self.visibility_history.clear()
        self.traffic_history.clear()
        self.road_condition_history.clear()