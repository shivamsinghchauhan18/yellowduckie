#!/usr/bin/env python3

import numpy as np
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import IntEnum
import time


class TrafficRuleType(IntEnum):
    """Types of traffic rules"""
    RIGHT_OF_WAY = 0
    LANE_CHANGE_RESTRICTION = 1
    INTERSECTION_RULE = 2
    SPEED_LIMIT = 3
    FOLLOWING_DISTANCE = 4
    SIGNALING_REQUIREMENT = 5


class RoadElementType(IntEnum):
    """Types of road elements"""
    STRAIGHT_ROAD = 0
    INTERSECTION = 1
    MERGE_ZONE = 2
    CONSTRUCTION_ZONE = 3
    SCHOOL_ZONE = 4
    PARKING_AREA = 5


class VehicleState(IntEnum):
    """Vehicle states for traffic rule evaluation"""
    NORMAL_DRIVING = 0
    TURNING_LEFT = 1
    TURNING_RIGHT = 2
    CHANGING_LANES = 3
    STOPPED = 4
    EMERGENCY = 5


@dataclass
class TrafficRule:
    """Definition of a traffic rule"""
    rule_type: TrafficRuleType
    description: str
    applicable_areas: List[RoadElementType]
    priority: int  # Higher number = higher priority
    conditions: Dict  # Conditions for rule activation
    restrictions: Dict  # Restrictions imposed by the rule


@dataclass
class VehicleInfo:
    """Information about a vehicle for traffic rule evaluation"""
    vehicle_id: str
    position: Tuple[float, float]  # (x, y) coordinates
    velocity: Tuple[float, float]  # (vx, vy) velocity
    heading: float  # Heading angle in radians
    state: VehicleState
    lane_id: int
    arrival_time: float  # Time when vehicle arrived at current position
    signaling: bool  # Whether vehicle is signaling


@dataclass
class RoadContext:
    """Context information about the current road situation"""
    current_road_element: RoadElementType
    intersection_id: Optional[str] = None
    lane_count: int = 2
    speed_limit: float = 0.5  # m/s
    construction_active: bool = False
    school_zone_active: bool = False
    traffic_density: float = 0.0  # vehicles per meter


@dataclass
class RuleViolation:
    """Information about a traffic rule violation"""
    rule: TrafficRule
    severity: int  # 1=minor, 2=moderate, 3=major, 4=critical
    description: str
    recommended_action: str


class TrafficRulesEngine:
    """
    Traffic rules engine for lane change decision making.
    
    Implements right-of-way rules, traffic law compliance, and
    intersection-aware lane change restrictions.
    """
    
    def __init__(self):
        """Initialize the traffic rules engine"""
        self.traffic_rules = self._initialize_traffic_rules()
        self.active_rules = set()
        self.rule_violations = []
        
        # Configuration parameters
        self.intersection_clearance_distance = 10.0  # meters
        self.merge_zone_length = 5.0  # meters
        self.following_distance_time = 2.0  # seconds
        self.lane_change_signal_time = 2.0  # seconds
        
        # State tracking
        self.last_evaluation_time = 0.0
        self.vehicle_history = {}  # Track vehicle positions over time
    
    def _initialize_traffic_rules(self) -> Dict[str, TrafficRule]:
        """Initialize the standard traffic rules"""
        rules = {}
        
        # Right-of-way rules
        rules["right_of_way_intersection"] = TrafficRule(
            rule_type=TrafficRuleType.RIGHT_OF_WAY,
            description="Vehicles already in intersection have right-of-way",
            applicable_areas=[RoadElementType.INTERSECTION],
            priority=10,
            conditions={"in_intersection": True},
            restrictions={"must_yield": True, "no_lane_change": True}
        )
        
        rules["right_of_way_straight"] = TrafficRule(
            rule_type=TrafficRuleType.RIGHT_OF_WAY,
            description="Straight-through traffic has right-of-way over turning traffic",
            applicable_areas=[RoadElementType.INTERSECTION],
            priority=8,
            conditions={"vehicle_state": VehicleState.NORMAL_DRIVING},
            restrictions={"turning_must_yield": True}
        )
        
        rules["right_of_way_arrival"] = TrafficRule(
            rule_type=TrafficRuleType.RIGHT_OF_WAY,
            description="First to arrive has right-of-way at intersection",
            applicable_areas=[RoadElementType.INTERSECTION],
            priority=6,
            conditions={"arrival_time_matters": True},
            restrictions={"yield_to_earlier_arrival": True}
        )
        
        # Lane change restrictions
        rules["no_lane_change_intersection"] = TrafficRule(
            rule_type=TrafficRuleType.LANE_CHANGE_RESTRICTION,
            description="No lane changes within intersection approach zone",
            applicable_areas=[RoadElementType.INTERSECTION],
            priority=9,
            conditions={"distance_to_intersection": 10.0},
            restrictions={"lane_change_prohibited": True}
        )
        
        rules["no_lane_change_construction"] = TrafficRule(
            rule_type=TrafficRuleType.LANE_CHANGE_RESTRICTION,
            description="No lane changes in construction zones",
            applicable_areas=[RoadElementType.CONSTRUCTION_ZONE],
            priority=8,
            conditions={"construction_active": True},
            restrictions={"lane_change_prohibited": True}
        )
        
        rules["merge_zone_priority"] = TrafficRule(
            rule_type=TrafficRuleType.LANE_CHANGE_RESTRICTION,
            description="Merging vehicles must yield to through traffic",
            applicable_areas=[RoadElementType.MERGE_ZONE],
            priority=7,
            conditions={"in_merge_zone": True},
            restrictions={"merging_must_yield": True}
        )
        
        # Signaling requirements
        rules["signal_before_lane_change"] = TrafficRule(
            rule_type=TrafficRuleType.SIGNALING_REQUIREMENT,
            description="Must signal for at least 2 seconds before lane change",
            applicable_areas=[RoadElementType.STRAIGHT_ROAD, RoadElementType.MERGE_ZONE],
            priority=5,
            conditions={"lane_change_intended": True},
            restrictions={"signal_duration_min": 2.0}
        )
        
        rules["signal_before_turn"] = TrafficRule(
            rule_type=TrafficRuleType.SIGNALING_REQUIREMENT,
            description="Must signal before turning at intersection",
            applicable_areas=[RoadElementType.INTERSECTION],
            priority=6,
            conditions={"turning": True},
            restrictions={"signal_required": True}
        )
        
        # Following distance rules
        rules["safe_following_distance"] = TrafficRule(
            rule_type=TrafficRuleType.FOLLOWING_DISTANCE,
            description="Maintain safe following distance",
            applicable_areas=[RoadElementType.STRAIGHT_ROAD, RoadElementType.INTERSECTION],
            priority=4,
            conditions={"vehicle_ahead": True},
            restrictions={"min_time_gap": 2.0}
        )
        
        # Speed limit rules
        rules["school_zone_speed"] = TrafficRule(
            rule_type=TrafficRuleType.SPEED_LIMIT,
            description="Reduced speed limit in school zones",
            applicable_areas=[RoadElementType.SCHOOL_ZONE],
            priority=7,
            conditions={"school_zone_active": True},
            restrictions={"max_speed": 0.2}  # 0.2 m/s in school zones
        )
        
        return rules
    
    def evaluate_lane_change_legality(self,
                                     ego_vehicle: VehicleInfo,
                                     other_vehicles: List[VehicleInfo],
                                     road_context: RoadContext,
                                     target_lane: int) -> Tuple[bool, List[RuleViolation]]:
        """
        Evaluate whether a lane change is legal under current traffic rules.
        
        Args:
            ego_vehicle: Information about the vehicle requesting lane change
            other_vehicles: Information about other vehicles in the area
            road_context: Current road context and conditions
            target_lane: Target lane for the lane change
            
        Returns:
            (is_legal, list_of_violations)
        """
        violations = []
        
        # Check intersection restrictions
        intersection_violations = self._check_intersection_restrictions(
            ego_vehicle, road_context
        )
        violations.extend(intersection_violations)
        
        # Check right-of-way rules
        right_of_way_violations = self._check_right_of_way_rules(
            ego_vehicle, other_vehicles, road_context, target_lane
        )
        violations.extend(right_of_way_violations)
        
        # Check signaling requirements
        signaling_violations = self._check_signaling_requirements(
            ego_vehicle, road_context
        )
        violations.extend(signaling_violations)
        
        # Check construction and special zone restrictions
        zone_violations = self._check_zone_restrictions(
            ego_vehicle, road_context
        )
        violations.extend(zone_violations)
        
        # Check following distance requirements
        distance_violations = self._check_following_distance_rules(
            ego_vehicle, other_vehicles, target_lane
        )
        violations.extend(distance_violations)
        
        # Lane change is legal if there are no critical violations
        critical_violations = [v for v in violations if v.severity >= 3]
        is_legal = len(critical_violations) == 0
        
        return is_legal, violations
    
    def _check_intersection_restrictions(self,
                                       ego_vehicle: VehicleInfo,
                                       road_context: RoadContext) -> List[RuleViolation]:
        """Check intersection-related lane change restrictions"""
        violations = []
        
        if road_context.current_road_element == RoadElementType.INTERSECTION:
            # Check if vehicle is too close to intersection for lane change
            distance_to_intersection = self._calculate_distance_to_intersection(
                ego_vehicle.position, road_context
            )
            
            if distance_to_intersection < self.intersection_clearance_distance:
                rule = self.traffic_rules["no_lane_change_intersection"]
                violation = RuleViolation(
                    rule=rule,
                    severity=4,  # Critical
                    description=f"Lane change prohibited within {self.intersection_clearance_distance}m of intersection",
                    recommended_action="Wait until past intersection to change lanes"
                )
                violations.append(violation)
        
        return violations
    
    def _check_right_of_way_rules(self,
                                 ego_vehicle: VehicleInfo,
                                 other_vehicles: List[VehicleInfo],
                                 road_context: RoadContext,
                                 target_lane: int) -> List[RuleViolation]:
        """Check right-of-way rules for lane change"""
        violations = []
        
        # Find vehicles in target lane
        target_lane_vehicles = [v for v in other_vehicles if v.lane_id == target_lane]
        
        for other_vehicle in target_lane_vehicles:
            # Check if other vehicle has right-of-way
            if self._has_right_of_way(other_vehicle, ego_vehicle, road_context):
                # Calculate if lane change would violate right-of-way
                if self._would_violate_right_of_way(ego_vehicle, other_vehicle, target_lane):
                    rule = self.traffic_rules["right_of_way_intersection"]
                    violation = RuleViolation(
                        rule=rule,
                        severity=3,  # Major
                        description=f"Lane change would violate right-of-way of vehicle {other_vehicle.vehicle_id}",
                        recommended_action="Wait for clear gap in target lane"
                    )
                    violations.append(violation)
        
        return violations
    
    def _check_signaling_requirements(self,
                                    ego_vehicle: VehicleInfo,
                                    road_context: RoadContext) -> List[RuleViolation]:
        """Check signaling requirements for lane change"""
        violations = []
        
        # Check if vehicle is signaling (this would be tracked by the signaling system)
        if not ego_vehicle.signaling:
            rule = self.traffic_rules["signal_before_lane_change"]
            violation = RuleViolation(
                rule=rule,
                severity=2,  # Moderate
                description="Must signal before lane change",
                recommended_action="Activate turn signal before attempting lane change"
            )
            violations.append(violation)
        
        return violations
    
    def _check_zone_restrictions(self,
                               ego_vehicle: VehicleInfo,
                               road_context: RoadContext) -> List[RuleViolation]:
        """Check special zone restrictions"""
        violations = []
        
        # Construction zone restrictions
        if (road_context.current_road_element == RoadElementType.CONSTRUCTION_ZONE and
            road_context.construction_active):
            rule = self.traffic_rules["no_lane_change_construction"]
            violation = RuleViolation(
                rule=rule,
                severity=4,  # Critical
                description="Lane changes prohibited in active construction zone",
                recommended_action="Maintain current lane through construction zone"
            )
            violations.append(violation)
        
        # School zone restrictions (additional caution required)
        if (road_context.current_road_element == RoadElementType.SCHOOL_ZONE and
            road_context.school_zone_active):
            # More restrictive lane change requirements in school zones
            if road_context.traffic_density > 0.1:  # High traffic density
                violation = RuleViolation(
                    rule=TrafficRule(
                        TrafficRuleType.LANE_CHANGE_RESTRICTION,
                        "Extra caution required for lane changes in school zones",
                        [RoadElementType.SCHOOL_ZONE], 6, {}, {}
                    ),
                    severity=2,  # Moderate
                    description="Extra caution required for lane changes in active school zone",
                    recommended_action="Ensure extra clearance and reduced speed"
                )
                violations.append(violation)
        
        return violations
    
    def _check_following_distance_rules(self,
                                      ego_vehicle: VehicleInfo,
                                      other_vehicles: List[VehicleInfo],
                                      target_lane: int) -> List[RuleViolation]:
        """Check following distance requirements after lane change"""
        violations = []
        
        # Find vehicles in target lane
        target_lane_vehicles = [v for v in other_vehicles if v.lane_id == target_lane]
        
        for other_vehicle in target_lane_vehicles:
            # Check following distance after potential lane change
            relative_position = self._calculate_relative_position(
                ego_vehicle.position, other_vehicle.position
            )
            
            # If other vehicle is ahead and too close
            if relative_position > 0:  # Other vehicle is ahead
                ego_speed = np.sqrt(ego_vehicle.velocity[0]**2 + ego_vehicle.velocity[1]**2)
                required_distance = ego_speed * self.following_distance_time
                
                if relative_position < required_distance:
                    rule = self.traffic_rules["safe_following_distance"]
                    violation = RuleViolation(
                        rule=rule,
                        severity=3,  # Major
                        description=f"Insufficient following distance after lane change ({relative_position:.1f}m < {required_distance:.1f}m required)",
                        recommended_action="Wait for larger gap or reduce speed before lane change"
                    )
                    violations.append(violation)
        
        return violations
    
    def _has_right_of_way(self,
                         vehicle: VehicleInfo,
                         ego_vehicle: VehicleInfo,
                         road_context: RoadContext) -> bool:
        """Determine if a vehicle has right-of-way over ego vehicle"""
        
        # In intersections, check arrival time
        if road_context.current_road_element == RoadElementType.INTERSECTION:
            # Vehicle that arrived first has right-of-way
            if vehicle.arrival_time < ego_vehicle.arrival_time:
                return True
            
            # Straight-through traffic has right-of-way over turning traffic
            if (vehicle.state == VehicleState.NORMAL_DRIVING and
                ego_vehicle.state in [VehicleState.TURNING_LEFT, VehicleState.TURNING_RIGHT]):
                return True
        
        # In merge zones, through traffic has right-of-way over merging traffic
        if road_context.current_road_element == RoadElementType.MERGE_ZONE:
            # Assume vehicles in lower-numbered lanes have right-of-way
            if vehicle.lane_id < ego_vehicle.lane_id:
                return True
        
        return False
    
    def _would_violate_right_of_way(self,
                                   ego_vehicle: VehicleInfo,
                                   other_vehicle: VehicleInfo,
                                   target_lane: int) -> bool:
        """Check if lane change would violate other vehicle's right-of-way"""
        
        # Calculate time and distance to potential conflict
        ego_speed = np.sqrt(ego_vehicle.velocity[0]**2 + ego_vehicle.velocity[1]**2)
        other_speed = np.sqrt(other_vehicle.velocity[0]**2 + other_vehicle.velocity[1]**2)
        
        # Simplified conflict detection
        relative_distance = self._calculate_relative_position(
            ego_vehicle.position, other_vehicle.position
        )
        
        # If other vehicle is close and moving faster, lane change would violate right-of-way
        if abs(relative_distance) < 3.0 and other_speed > ego_speed:
            return True
        
        # If other vehicle is very close (within 2 meters), always violates right-of-way
        if abs(relative_distance) < 2.0:
            return True
        
        return False
    
    def _calculate_distance_to_intersection(self,
                                          position: Tuple[float, float],
                                          road_context: RoadContext) -> float:
        """Calculate distance to nearest intersection"""
        # Simplified calculation - in practice would use map data
        # For now, assume intersection is at a fixed distance ahead
        return 5.0  # Default 5 meters to intersection
    
    def _calculate_relative_position(self,
                                   pos1: Tuple[float, float],
                                   pos2: Tuple[float, float]) -> float:
        """Calculate relative longitudinal position between two vehicles"""
        # Simplified - assume vehicles are on parallel lanes
        return pos2[0] - pos1[0]  # Difference in x-coordinate
    
    def get_lane_change_recommendations(self,
                                      ego_vehicle: VehicleInfo,
                                      other_vehicles: List[VehicleInfo],
                                      road_context: RoadContext) -> Dict[int, str]:
        """
        Get recommendations for lane changes to different lanes.
        
        Returns:
            Dictionary mapping lane_id to recommendation string
        """
        recommendations = {}
        
        # Check each possible target lane
        available_lanes = range(max(1, road_context.lane_count))
        
        for target_lane in available_lanes:
            if target_lane == ego_vehicle.lane_id:
                continue  # Skip current lane
            
            is_legal, violations = self.evaluate_lane_change_legality(
                ego_vehicle, other_vehicles, road_context, target_lane
            )
            
            if is_legal:
                recommendations[target_lane] = "Lane change permitted"
            else:
                # Provide specific recommendation based on violations
                critical_violations = [v for v in violations if v.severity >= 3]
                if critical_violations:
                    recommendations[target_lane] = critical_violations[0].recommended_action
                else:
                    recommendations[target_lane] = "Lane change not recommended"
        
        return recommendations
    
    def update_vehicle_history(self, vehicles: List[VehicleInfo]):
        """Update vehicle position history for tracking"""
        current_time = time.time()
        
        for vehicle in vehicles:
            if vehicle.vehicle_id not in self.vehicle_history:
                self.vehicle_history[vehicle.vehicle_id] = []
            
            # Add current position to history
            self.vehicle_history[vehicle.vehicle_id].append({
                'time': current_time,
                'position': vehicle.position,
                'velocity': vehicle.velocity,
                'state': vehicle.state
            })
            
            # Keep only recent history (last 10 seconds)
            cutoff_time = current_time - 10.0
            self.vehicle_history[vehicle.vehicle_id] = [
                entry for entry in self.vehicle_history[vehicle.vehicle_id]
                if entry['time'] > cutoff_time
            ]
    
    def get_traffic_rule_summary(self) -> Dict[str, int]:
        """Get summary of active traffic rules"""
        summary = {}
        
        for rule_name, rule in self.traffic_rules.items():
            summary[rule_name] = {
                'type': rule.rule_type.name,
                'priority': rule.priority,
                'applicable_areas': [area.name for area in rule.applicable_areas]
            }
        
        return summary
    
    def validate_intersection_behavior(self,
                                     ego_vehicle: VehicleInfo,
                                     other_vehicles: List[VehicleInfo],
                                     road_context: RoadContext) -> Tuple[bool, List[str]]:
        """
        Validate behavior at intersections according to traffic rules.
        
        Returns:
            (is_compliant, list_of_issues)
        """
        issues = []
        
        if road_context.current_road_element != RoadElementType.INTERSECTION:
            return True, []
        
        # Check signaling requirements for turns
        if ego_vehicle.state in [VehicleState.TURNING_LEFT, VehicleState.TURNING_RIGHT]:
            if not ego_vehicle.signaling:
                issues.append("Must signal before turning at intersection")
        
        # Check right-of-way compliance
        for other_vehicle in other_vehicles:
            if self._has_right_of_way(other_vehicle, ego_vehicle, road_context):
                # Check if ego vehicle is yielding appropriately
                if not self._is_yielding_appropriately(ego_vehicle, other_vehicle):
                    issues.append(f"Must yield right-of-way to vehicle {other_vehicle.vehicle_id}")
        
        # Check speed compliance in intersection
        ego_speed = np.sqrt(ego_vehicle.velocity[0]**2 + ego_vehicle.velocity[1]**2)
        if ego_speed > road_context.speed_limit:
            issues.append(f"Speed {ego_speed:.1f} m/s exceeds limit {road_context.speed_limit:.1f} m/s")
        
        return len(issues) == 0, issues
    
    def _is_yielding_appropriately(self,
                                 ego_vehicle: VehicleInfo,
                                 other_vehicle: VehicleInfo) -> bool:
        """Check if ego vehicle is yielding appropriately to other vehicle"""
        # Simplified yielding check
        relative_distance = self._calculate_relative_position(
            ego_vehicle.position, other_vehicle.position
        )
        
        # If other vehicle is close and ego vehicle is stopped/slow, consider it yielding
        ego_speed = np.sqrt(ego_vehicle.velocity[0]**2 + ego_vehicle.velocity[1]**2)
        
        if abs(relative_distance) < 5.0 and ego_speed < 0.1:
            return True  # Ego vehicle is yielding (stopped/very slow)
        
        return False