#!/usr/bin/env python3

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum


class RiskLevel(IntEnum):
    """Risk levels for feasibility assessment"""
    VERY_LOW = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class VehicleInfo:
    """Information about a detected vehicle"""
    x: float  # Position in camera frame (meters)
    y: float  # Position in camera frame (meters)
    velocity_x: float = 0.0  # Estimated velocity (m/s)
    velocity_y: float = 0.0  # Estimated velocity (m/s)
    confidence: float = 1.0  # Detection confidence (0-1)
    timestamp: float = 0.0  # Detection timestamp


@dataclass
class LanePoseInfo:
    """Current lane pose information"""
    d: float  # Lateral deviation from lane center (meters)
    phi: float  # Heading deviation from lane direction (radians)
    confidence: float = 1.0  # Pose confidence (0-1)


@dataclass
class FeasibilityResult:
    """Result of feasibility assessment"""
    overall_score: float  # Overall feasibility score (0-1)
    adjacent_lane_score: float  # Adjacent lane clearance score
    gap_analysis_score: float  # Gap analysis score
    traffic_flow_score: float  # Traffic flow score
    risk_level: RiskLevel  # Overall risk level
    blocking_factors: List[str]  # List of factors preventing lane change
    recommendations: List[str]  # Recommendations for improving feasibility


class LaneChangeFeasibilityAssessor:
    """
    Advanced feasibility assessment for lane changes using enhanced perception data.
    
    This class implements sophisticated algorithms for:
    - Adjacent lane occupancy detection
    - Gap analysis and merge safety calculations
    - Traffic flow analysis for optimal timing
    """
    
    def __init__(self, config: dict = None):
        """Initialize feasibility assessor with configuration"""
        self.config = config or {}
        
        # Default configuration parameters
        self.lane_width = self.config.get('lane_width', 0.6)  # meters
        self.safe_gap_length = self.config.get('safe_gap_length', 2.0)  # meters
        self.safe_gap_time = self.config.get('safe_gap_time', 2.0)  # seconds
        self.lateral_clearance = self.config.get('lateral_clearance', 0.1)  # meters
        self.min_visibility_distance = self.config.get('min_visibility_distance', 3.0)  # meters
        
        # Scoring weights
        self.adjacent_lane_weight = self.config.get('adjacent_lane_weight', 0.4)
        self.gap_analysis_weight = self.config.get('gap_analysis_weight', 0.4)
        self.traffic_flow_weight = self.config.get('traffic_flow_weight', 0.2)
        
        # Risk thresholds
        self.risk_thresholds = {
            RiskLevel.VERY_LOW: 0.9,
            RiskLevel.LOW: 0.7,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.3,
            RiskLevel.CRITICAL: 0.0
        }
    
    def assess_feasibility(self, 
                          direction: int,
                          current_pose: LanePoseInfo,
                          detected_vehicles: List[VehicleInfo],
                          predicted_trajectories: Optional[dict] = None) -> FeasibilityResult:
        """
        Perform comprehensive feasibility assessment for lane change.
        
        Args:
            direction: Lane change direction (-1=left, 1=right)
            current_pose: Current lane pose information
            detected_vehicles: List of detected vehicles
            predicted_trajectories: Optional trajectory predictions
            
        Returns:
            FeasibilityResult with detailed assessment
        """
        # Assess adjacent lane occupancy
        adjacent_score = self._assess_adjacent_lane_occupancy(
            direction, current_pose, detected_vehicles
        )
        
        # Perform gap analysis
        gap_score = self._perform_gap_analysis(
            direction, current_pose, detected_vehicles, predicted_trajectories
        )
        
        # Analyze traffic flow
        traffic_score = self._analyze_traffic_flow(
            direction, detected_vehicles, predicted_trajectories
        )
        
        # Calculate overall score
        overall_score = (
            self.adjacent_lane_weight * adjacent_score +
            self.gap_analysis_weight * gap_score +
            self.traffic_flow_weight * traffic_score
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(overall_score)
        
        # Identify blocking factors
        blocking_factors = self._identify_blocking_factors(
            adjacent_score, gap_score, traffic_score, direction
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            adjacent_score, gap_score, traffic_score, direction
        )
        
        return FeasibilityResult(
            overall_score=overall_score,
            adjacent_lane_score=adjacent_score,
            gap_analysis_score=gap_score,
            traffic_flow_score=traffic_score,
            risk_level=risk_level,
            blocking_factors=blocking_factors,
            recommendations=recommendations
        )
    
    def _assess_adjacent_lane_occupancy(self, 
                                       direction: int,
                                       current_pose: LanePoseInfo,
                                       detected_vehicles: List[VehicleInfo]) -> float:
        """
        Assess occupancy of adjacent lane using enhanced perception.
        
        This method analyzes the target lane for:
        - Direct occupancy by other vehicles
        - Approaching vehicles that would conflict
        - Visibility and detection confidence
        """
        if not detected_vehicles:
            return 1.0  # No vehicles detected, assume clear
        
        # Calculate target lane center position
        target_lane_center = current_pose.d + (direction * self.lane_width)
        
        # Define adjacent lane boundaries
        lane_left_bound = target_lane_center - (self.lane_width / 2)
        lane_right_bound = target_lane_center + (self.lane_width / 2)
        
        occupancy_score = 1.0
        vehicles_in_lane = 0
        closest_vehicle_distance = float('inf')
        
        for vehicle in detected_vehicles:
            # Convert vehicle position to lane-relative coordinates
            vehicle_lateral_pos = self._estimate_vehicle_lateral_position(vehicle, current_pose)
            vehicle_distance = np.sqrt(vehicle.x**2 + vehicle.y**2)
            
            # Check if vehicle is in target lane
            if lane_left_bound <= vehicle_lateral_pos <= lane_right_bound:
                vehicles_in_lane += 1
                closest_vehicle_distance = min(closest_vehicle_distance, vehicle_distance)
                
                # Reduce score based on proximity and confidence
                proximity_factor = max(0.1, min(1.0, vehicle_distance / self.min_visibility_distance))
                confidence_factor = vehicle.confidence
                
                occupancy_score *= proximity_factor * confidence_factor
        
        # Additional penalty for multiple vehicles in lane
        if vehicles_in_lane > 1:
            occupancy_score *= 0.5
        
        # Bonus for good visibility (no vehicles nearby)
        if closest_vehicle_distance > self.min_visibility_distance:
            occupancy_score = min(1.0, occupancy_score * 1.2)
        
        return max(0.0, occupancy_score)
    
    def _perform_gap_analysis(self,
                             direction: int,
                             current_pose: LanePoseInfo,
                             detected_vehicles: List[VehicleInfo],
                             predicted_trajectories: Optional[dict] = None) -> float:
        """
        Perform detailed gap analysis for safe merging.
        
        Analyzes:
        - Available gap size in target lane
        - Gap duration based on vehicle speeds
        - Merge safety margins
        """
        if not detected_vehicles:
            return 1.0  # No vehicles, plenty of gap
        
        # Find vehicles in target lane and sort by longitudinal position
        target_vehicles = self._get_vehicles_in_target_lane(
            direction, current_pose, detected_vehicles
        )
        
        if not target_vehicles:
            return 1.0  # No vehicles in target lane
        
        # Sort vehicles by longitudinal position (x-coordinate)
        target_vehicles.sort(key=lambda v: v.x)
        
        # Find the best gap for merging
        best_gap_score = 0.0
        
        # Check gap before first vehicle (if vehicle is far enough ahead)
        if target_vehicles[0].x > self.safe_gap_length:
            gap_score = self._evaluate_gap(
                gap_start=0.0,
                gap_end=target_vehicles[0].x,
                leading_vehicle=target_vehicles[0],
                following_vehicle=None,
                predicted_trajectories=predicted_trajectories
            )
            best_gap_score = max(best_gap_score, gap_score)
        else:
            # Vehicle is too close, no safe gap before it
            pass
        
        # Check gaps between vehicles
        for i in range(len(target_vehicles) - 1):
            gap_start = target_vehicles[i].x
            gap_end = target_vehicles[i + 1].x
            gap_length = gap_end - gap_start
            
            if gap_length > self.safe_gap_length:
                gap_score = self._evaluate_gap(
                    gap_start=gap_start,
                    gap_end=gap_end,
                    leading_vehicle=target_vehicles[i + 1],
                    following_vehicle=target_vehicles[i],
                    predicted_trajectories=predicted_trajectories
                )
                best_gap_score = max(best_gap_score, gap_score)
        
        # Check gap after last vehicle (only if there's sufficient distance)
        last_vehicle = target_vehicles[-1]
        gap_after_last = self.min_visibility_distance - last_vehicle.x
        if gap_after_last > self.safe_gap_length:
            gap_score = self._evaluate_gap(
                gap_start=last_vehicle.x,
                gap_end=self.min_visibility_distance,
                leading_vehicle=None,
                following_vehicle=last_vehicle,
                predicted_trajectories=predicted_trajectories
            )
            best_gap_score = max(best_gap_score, gap_score)
        
        return best_gap_score
    
    def _analyze_traffic_flow(self,
                             direction: int,
                             detected_vehicles: List[VehicleInfo],
                             predicted_trajectories: Optional[dict] = None) -> float:
        """
        Analyze traffic flow patterns for optimal lane change timing.
        
        Considers:
        - Overall traffic density
        - Vehicle speed patterns
        - Flow consistency
        - Predicted traffic evolution
        """
        if not detected_vehicles:
            return 1.0  # No traffic, optimal conditions
        
        # Calculate traffic density
        visible_area = np.pi * (self.min_visibility_distance ** 2)
        traffic_density = len(detected_vehicles) / visible_area
        
        # Normalize density score (lower density = higher score)
        max_expected_density = 0.05  # vehicles per square meter (more sensitive)
        density_score = max(0.0, 1.0 - (traffic_density / max_expected_density))
        
        # Analyze speed patterns
        speeds = []
        for vehicle in detected_vehicles:
            speed = np.sqrt(vehicle.velocity_x**2 + vehicle.velocity_y**2)
            speeds.append(speed)
        
        if speeds:
            avg_speed = np.mean(speeds)
            speed_variance = np.var(speeds)
            
            # Prefer moderate, consistent speeds
            speed_score = 1.0
            if avg_speed > 0.5:  # Too fast
                speed_score *= 0.7
            elif avg_speed < 0.1:  # Too slow/stopped
                speed_score *= 0.8
            
            # Penalize high speed variance (inconsistent flow)
            if speed_variance > 0.1:
                speed_score *= 0.8
        else:
            speed_score = 1.0
        
        # Analyze flow direction consistency
        flow_score = self._analyze_flow_consistency(detected_vehicles)
        
        # Combine factors
        traffic_flow_score = (density_score * 0.4 + 
                             speed_score * 0.4 + 
                             flow_score * 0.2)
        
        return max(0.0, min(1.0, traffic_flow_score))
    
    def _get_vehicles_in_target_lane(self,
                                    direction: int,
                                    current_pose: LanePoseInfo,
                                    detected_vehicles: List[VehicleInfo]) -> List[VehicleInfo]:
        """Get vehicles that are in the target lane"""
        target_lane_center = current_pose.d + (direction * self.lane_width)
        lane_left_bound = target_lane_center - (self.lane_width / 2)
        lane_right_bound = target_lane_center + (self.lane_width / 2)
        
        target_vehicles = []
        for vehicle in detected_vehicles:
            vehicle_lateral_pos = self._estimate_vehicle_lateral_position(vehicle, current_pose)
            if lane_left_bound <= vehicle_lateral_pos <= lane_right_bound:
                target_vehicles.append(vehicle)
        
        return target_vehicles
    
    def _estimate_vehicle_lateral_position(self,
                                          vehicle: VehicleInfo,
                                          current_pose: LanePoseInfo) -> float:
        """Estimate vehicle's lateral position relative to current lane"""
        # Simplified estimation - in practice would use more sophisticated
        # coordinate transformation based on camera calibration
        # For now, assume vehicle.y is already in lane-relative coordinates
        return vehicle.y
    
    def _evaluate_gap(self,
                     gap_start: float,
                     gap_end: float,
                     leading_vehicle: Optional[VehicleInfo],
                     following_vehicle: Optional[VehicleInfo],
                     predicted_trajectories: Optional[dict] = None) -> float:
        """Evaluate the quality of a specific gap for merging"""
        gap_length = gap_end - gap_start
        
        # Base score from gap length
        length_score = min(1.0, gap_length / self.safe_gap_length)
        
        # Adjust for vehicle velocities
        velocity_score = 1.0
        
        if leading_vehicle and leading_vehicle.velocity_x > 0:
            # Gap is expanding if leading vehicle is moving away
            velocity_score *= 1.2
        elif leading_vehicle and leading_vehicle.velocity_x < -0.1:
            # Gap is shrinking if leading vehicle is approaching
            velocity_score *= 0.6
        
        if following_vehicle and following_vehicle.velocity_x > 0.1:
            # Following vehicle is catching up
            velocity_score *= 0.8
        
        # Time-based gap evaluation
        if predicted_trajectories:
            time_score = self._evaluate_gap_over_time(
                gap_start, gap_end, leading_vehicle, following_vehicle, predicted_trajectories
            )
        else:
            time_score = 1.0
        
        return length_score * velocity_score * time_score
    
    def _evaluate_gap_over_time(self,
                               gap_start: float,
                               gap_end: float,
                               leading_vehicle: Optional[VehicleInfo],
                               following_vehicle: Optional[VehicleInfo],
                               predicted_trajectories: dict) -> float:
        """Evaluate gap quality over predicted time horizon"""
        # Simplified time-based evaluation
        # In practice, would use detailed trajectory predictions
        
        time_horizon = 3.0  # seconds
        time_steps = 10
        dt = time_horizon / time_steps
        
        gap_scores = []
        
        for t in range(time_steps):
            current_time = t * dt
            
            # Predict gap boundaries at this time
            predicted_start = gap_start
            predicted_end = gap_end
            
            if leading_vehicle:
                predicted_end += leading_vehicle.velocity_x * current_time
            
            if following_vehicle:
                predicted_start += following_vehicle.velocity_x * current_time
            
            predicted_gap = predicted_end - predicted_start
            gap_score = max(0.0, min(1.0, predicted_gap / self.safe_gap_length))
            gap_scores.append(gap_score)
        
        # Return minimum gap score over time horizon
        return min(gap_scores) if gap_scores else 0.0
    
    def _analyze_flow_consistency(self, detected_vehicles: List[VehicleInfo]) -> float:
        """Analyze consistency of traffic flow direction"""
        if len(detected_vehicles) < 2:
            return 1.0
        
        # Calculate flow directions
        flow_directions = []
        for vehicle in detected_vehicles:
            if abs(vehicle.velocity_x) > 0.05:  # Minimum speed threshold
                flow_directions.append(np.sign(vehicle.velocity_x))
        
        if not flow_directions:
            return 1.0  # No moving vehicles
        
        # Calculate consistency (how many vehicles move in same direction)
        forward_count = sum(1 for d in flow_directions if d > 0)
        backward_count = sum(1 for d in flow_directions if d < 0)
        total_count = len(flow_directions)
        
        consistency = max(forward_count, backward_count) / total_count
        return consistency
    
    def _determine_risk_level(self, overall_score: float) -> RiskLevel:
        """Determine risk level based on overall feasibility score"""
        for risk_level in [RiskLevel.VERY_LOW, RiskLevel.LOW, RiskLevel.MEDIUM, 
                          RiskLevel.HIGH, RiskLevel.CRITICAL]:
            if overall_score >= self.risk_thresholds[risk_level]:
                return risk_level
        return RiskLevel.CRITICAL
    
    def _identify_blocking_factors(self,
                                  adjacent_score: float,
                                  gap_score: float,
                                  traffic_score: float,
                                  direction: int) -> List[str]:
        """Identify factors that are blocking or hindering the lane change"""
        blocking_factors = []
        
        if adjacent_score < 0.5:
            blocking_factors.append("Adjacent lane occupied by vehicles")
        
        if gap_score < 0.5:
            blocking_factors.append("Insufficient gap for safe merging")
        
        if traffic_score < 0.5:
            blocking_factors.append("Heavy or inconsistent traffic flow")
        
        if adjacent_score < 0.3:
            blocking_factors.append("High risk of collision in target lane")
        
        if gap_score < 0.3:
            blocking_factors.append("No suitable gaps available for merging")
        
        return blocking_factors
    
    def _generate_recommendations(self,
                                 adjacent_score: float,
                                 gap_score: float,
                                 traffic_score: float,
                                 direction: int) -> List[str]:
        """Generate recommendations for improving lane change feasibility"""
        recommendations = []
        
        if adjacent_score < 0.7:
            recommendations.append("Wait for adjacent lane to clear")
            recommendations.append("Monitor approaching vehicles in target lane")
        
        if gap_score < 0.7:
            recommendations.append("Wait for larger gap between vehicles")
            recommendations.append("Consider adjusting speed to align with traffic flow")
        
        if traffic_score < 0.7:
            recommendations.append("Wait for traffic flow to stabilize")
            recommendations.append("Consider lane change during lighter traffic")
        
        if adjacent_score > 0.8 and gap_score > 0.8 and traffic_score > 0.8:
            recommendations.append("Conditions are favorable for lane change")
            recommendations.append("Proceed with caution and proper signaling")
        
        return recommendations