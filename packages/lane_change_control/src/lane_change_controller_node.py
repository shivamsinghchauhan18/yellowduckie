#!/usr/bin/env python3

import os
import rospy
import numpy as np
from enum import IntEnum
from threading import Lock

from duckietown.dtros import DTROS, NodeType, TopicType, DTParam, ParamType
from duckietown_msgs.msg import Twist2DStamped, LanePose, VehicleCorners, BoolStamped
from duckietown_msgs.srv import ChangePattern, ChangePatternRequest
from lane_change_control.msg import LaneChangeRequest, LaneChangeStatus
from std_msgs.msg import String, Header
from geometry_msgs.msg import Point32
from typing import Optional


# Import enhanced feasibility assessment and trajectory generation
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'include'))
from lane_change_control.feasibility_assessor import (
    LaneChangeFeasibilityAssessor, VehicleInfo, LanePoseInfo, FeasibilityResult
)
from lane_change_control.trajectory_generator import (
    LaneChangeTrajectoryGenerator, TrajectoryConstraints, TrajectoryPoint, TrajectoryType
)
from lane_change_control.led_signaling_system import (
    LaneChangeSignalingSystem, SignalType, SignalRequest, SignalPriority
)
from lane_change_control.traffic_rules_engine import (
    TrafficRulesEngine, VehicleInfo, RoadContext, VehicleState, RoadElementType
)


class LaneChangeState(IntEnum):
    """Lane change controller states"""
    LANE_FOLLOWING = 0
    ASSESSING_CHANGE = 1
    SIGNALING = 2
    EXECUTING = 3
    VERIFYING = 4
    ABORTING = 5


class LaneChangeDirection(IntEnum):
    """Lane change directions"""
    LEFT = -1
    RIGHT = 1


class LaneChangeControllerNode(DTROS):
    """
    Lane Change Controller Node
    
    Implements safe and efficient lane changing capabilities with state machine-based control.
    Provides lane change feasibility assessment and multi-phase execution.
    
    Publishers:
        ~lane_change_status (LaneChangeStatus): Current status of lane change controller
        ~car_cmd_override (Twist2DStamped): Override commands during lane change execution
        
    Subscribers:
        ~lane_pose (LanePose): Current lane pose estimate
        ~vehicle_detections (VehicleCorners): Detected vehicles for safety assessment
        ~lane_change_request (LaneChangeRequest): External lane change requests
        ~safety_override (BoolStamped): Emergency safety override signal
        
    Services:
        ~request_lane_change: Service to request lane changes programmatically
    """
    
    def __init__(self, node_name):
        super(LaneChangeControllerNode, self).__init__(
            node_name=node_name, 
            node_type=NodeType.CONTROL
        )
        
        # Get vehicle name
        self.veh_name = rospy.get_param("~veh_name", os.environ.get('VEHICLE_NAME', 'duckiebot'))
        
        # Initialize parameters
        self._init_parameters()
        
        # Initialize feasibility assessor
        self.feasibility_assessor = LaneChangeFeasibilityAssessor({
            'lane_width': 0.6,
            'safe_gap_length': 2.0,
            'safe_gap_time': 2.0,
            'lateral_clearance': 0.1,
            'min_visibility_distance': 3.0,
            'adjacent_lane_weight': 0.4,
            'gap_analysis_weight': 0.4,
            'traffic_flow_weight': 0.2
        })
        
        # Initialize trajectory generator
        trajectory_constraints = TrajectoryConstraints(
            max_lateral_acceleration=0.5,
            max_lateral_velocity=0.3,
            max_curvature=2.0,
            comfort_factor=0.8,
            safety_margin=0.1
        )
        self.trajectory_generator = LaneChangeTrajectoryGenerator(trajectory_constraints)
        
        # Initialize enhanced signaling system
        self.signaling_system = LaneChangeSignalingSystem(self.led_service)
        
        # Initialize traffic rules engine
        self.traffic_rules_engine = TrafficRulesEngine()
        
        # State management
        self.state = LaneChangeState.LANE_FOLLOWING
        self.state_lock = Lock()
        self.active_request = None
        self.state_start_time = rospy.Time.now()
        
        # Current sensor data
        self.current_lane_pose = None
        self.detected_vehicles = []
        self.safety_override_active = False
        
        # Lane change execution variables
        self.maneuver_start_time = None
        self.maneuver_progress = 0.0
        self.target_lateral_offset = 0.0
        self.initial_lateral_position = 0.0
        
        # Trajectory execution variables
        self.current_trajectory = None
        self.trajectory_start_time = None
        self.current_trajectory_index = 0
        
        # Feasibility assessment results
        self.last_feasibility_result = None
        
        # Publishers
        self.pub_status = rospy.Publisher(
            "~lane_change_status", 
            LaneChangeStatus, 
            queue_size=1,
            dt_topic_type=TopicType.CONTROL
        )
        
        self.pub_car_cmd_override = rospy.Publisher(
            "~car_cmd_override", 
            Twist2DStamped, 
            queue_size=1,
            dt_topic_type=TopicType.CONTROL
        )
        
        # Subscribers
        self.sub_lane_pose = rospy.Subscriber(
            "~lane_pose", 
            LanePose, 
            self.cb_lane_pose, 
            queue_size=1
        )
        
        self.sub_vehicle_detections = rospy.Subscriber(
            "~vehicle_detections", 
            VehicleCorners, 
            self.cb_vehicle_detections, 
            queue_size=1
        )
        
        self.sub_lane_change_request = rospy.Subscriber(
            "~lane_change_request", 
            LaneChangeRequest, 
            self.cb_lane_change_request, 
            queue_size=1
        )
        
        self.sub_safety_override = rospy.Subscriber(
            "~safety_override", 
            BoolStamped, 
            self.cb_safety_override, 
            queue_size=1
        )
        
        # LED service for signaling
        self.led_service = None
        self._init_led_service()
        
        self.log("Lane Change Controller initialized")
    
    def _init_parameters(self):
        """Initialize configuration parameters"""
        self.params = {}
        
        # Timing parameters
        self.params["signal_duration"] = DTParam(
            "~signal_duration", 
            param_type=ParamType.FLOAT, 
            min_value=1.0, 
            max_value=10.0,
            default=2.0
        )
        
        self.params["max_maneuver_time"] = DTParam(
            "~max_maneuver_time", 
            param_type=ParamType.FLOAT, 
            min_value=3.0, 
            max_value=15.0,
            default=8.0
        )
        
        self.params["assessment_timeout"] = DTParam(
            "~assessment_timeout", 
            param_type=ParamType.FLOAT, 
            min_value=1.0, 
            max_value=5.0,
            default=3.0
        )
        
        # Safety parameters
        self.params["min_feasibility_score"] = DTParam(
            "~min_feasibility_score", 
            param_type=ParamType.FLOAT, 
            min_value=0.0, 
            max_value=1.0,
            default=0.7
        )
        
        self.params["lateral_acceleration_limit"] = DTParam(
            "~lateral_acceleration_limit", 
            param_type=ParamType.FLOAT, 
            min_value=0.1, 
            max_value=1.0,
            default=0.5
        )
        
        self.params["safe_following_distance"] = DTParam(
            "~safe_following_distance", 
            param_type=ParamType.FLOAT, 
            min_value=0.5, 
            max_value=3.0,
            default=1.5
        )
        
        # Lane change geometry
        self.params["lane_width"] = DTParam(
            "~lane_width", 
            param_type=ParamType.FLOAT, 
            min_value=0.3, 
            max_value=1.0,
            default=0.6
        )
        
        self.params["lateral_clearance"] = DTParam(
            "~lateral_clearance", 
            param_type=ParamType.FLOAT, 
            min_value=0.05, 
            max_value=0.3,
            default=0.1
        )
    
    def _init_led_service(self):
        """Initialize LED service for signaling"""
        led_service_name = f"/{self.veh_name}/led_emitter_node/set_pattern"
        try:
            rospy.wait_for_service(led_service_name, timeout=3.0)
            self.led_service = rospy.ServiceProxy(led_service_name, ChangePattern)
            self.log("LED service connected")
        except rospy.ROSException:
            self.log("LED service not available, continuing without LED signaling", "warn")
    
    def cb_lane_pose(self, msg):
        """Callback for lane pose updates"""
        self.current_lane_pose = msg
    
    def cb_vehicle_detections(self, msg):
        """Callback for vehicle detection updates"""
        self.detected_vehicles = msg.corners
    
    def cb_lane_change_request(self, msg):
        """Callback for lane change requests"""
        with self.state_lock:
            if self.state == LaneChangeState.LANE_FOLLOWING:
                self.log(f"Received lane change request: direction={msg.direction}, reason={msg.reason}")
                self.active_request = msg
                self._transition_to_state(LaneChangeState.ASSESSING_CHANGE)
            else:
                self.log(f"Ignoring lane change request - currently in state {self.state}", "warn")
    
    def cb_safety_override(self, msg):
        """Callback for safety override signals"""
        self.safety_override_active = msg.data
        if self.safety_override_active and self.state != LaneChangeState.LANE_FOLLOWING:
            self.log("Safety override activated - aborting lane change", "warn")
            
            # Activate emergency signaling
            self.signaling_system.emergency_signal_override()
            
            self._transition_to_state(LaneChangeState.ABORTING)
    
    def _transition_to_state(self, new_state):
        """Transition to a new state with proper cleanup"""
        old_state = self.state
        self.state = new_state
        self.state_start_time = rospy.Time.now()
        
        self.log(f"State transition: {old_state} -> {new_state}")
        
        # State entry actions
        if new_state == LaneChangeState.ASSESSING_CHANGE:
            self._start_feasibility_assessment()
        elif new_state == LaneChangeState.SIGNALING:
            self._start_signaling()
        elif new_state == LaneChangeState.EXECUTING:
            self._start_execution()
        elif new_state == LaneChangeState.VERIFYING:
            self._start_verification()
        elif new_state == LaneChangeState.ABORTING:
            self._start_abort()
        elif new_state == LaneChangeState.LANE_FOLLOWING:
            self._cleanup_maneuver()
    
    def _start_feasibility_assessment(self):
        """Start feasibility assessment for lane change"""
        if not self.active_request:
            self.log("No active request for feasibility assessment", "error")
            self._transition_to_state(LaneChangeState.LANE_FOLLOWING)
            return
        
        self.log(f"Starting feasibility assessment for {self.active_request.direction} lane change")
        
        # Calculate feasibility score
        feasibility_score = self._calculate_feasibility_score()
        self.active_request.feasibility_score = feasibility_score
        
        self.log(f"Feasibility score: {feasibility_score:.2f}")
    
    def _calculate_feasibility_score(self):
        """Calculate feasibility score using enhanced assessment with traffic rules"""
        if not self.current_lane_pose or not self.active_request:
            return 0.0
        
        # Convert current data to assessor format
        current_pose_info = LanePoseInfo(
            d=self.current_lane_pose.d,
            phi=self.current_lane_pose.phi,
            confidence=1.0  # Assume good confidence for now
        )
        
        # Convert detected vehicles to assessor format
        vehicle_infos = []
        for corner in self.detected_vehicles:
            vehicle_info = VehicleInfo(
                x=corner.x,
                y=corner.y,
                velocity_x=0.0,  # Would need velocity estimation from tracking
                velocity_y=0.0,
                confidence=1.0,
                timestamp=rospy.Time.now().to_sec()
            )
            vehicle_infos.append(vehicle_info)
        
        # Perform enhanced feasibility assessment
        result = self.feasibility_assessor.assess_feasibility(
            direction=self.active_request.direction,
            current_pose=current_pose_info,
            detected_vehicles=vehicle_infos,
            predicted_trajectories=None  # Would integrate with predictive perception
        )
        
        # Apply traffic rules evaluation
        traffic_rules_score = self._evaluate_traffic_rules_compliance()
        
        # Combine feasibility and traffic rules scores
        # Traffic rules have higher priority - if illegal, score is severely reduced
        if traffic_rules_score < 0.3:  # Critical traffic rule violations
            combined_score = traffic_rules_score * 0.5  # Severely penalize
        else:
            # Weight: 60% feasibility, 40% traffic rules compliance
            combined_score = 0.6 * result.overall_score + 0.4 * traffic_rules_score
        
        # Store detailed results for debugging/logging
        self.last_feasibility_result = result
        
        self.log(f"Feasibility assessment: overall={result.overall_score:.2f}, "
                f"adjacent={result.adjacent_lane_score:.2f}, "
                f"gap={result.gap_analysis_score:.2f}, "
                f"traffic={result.traffic_flow_score:.2f}, "
                f"traffic_rules={traffic_rules_score:.2f}, "
                f"combined={combined_score:.2f}")
        
        if result.blocking_factors:
            self.log(f"Blocking factors: {', '.join(result.blocking_factors)}")
        
        return combined_score
    
    def _assess_adjacent_lane_clearance(self):
        """Assess if adjacent lane is clear for lane change"""
        # Simplified assessment - in real implementation would use
        # enhanced perception data and trajectory predictions
        
        if not self.detected_vehicles:
            return 1.0  # No vehicles detected, assume clear
        
        # Check for vehicles in adjacent lane
        # This is a simplified check - real implementation would need
        # more sophisticated spatial reasoning
        vehicle_count = len(self.detected_vehicles)
        if vehicle_count == 0:
            return 1.0
        elif vehicle_count <= 2:
            return 0.8
        else:
            return 0.3
    
    def _assess_vehicle_proximity(self):
        """Assess proximity of other vehicles"""
        if not self.detected_vehicles:
            return 1.0
        
        # Simple proximity check based on number of detected vehicles
        # Real implementation would use actual distances and velocities
        min_distance = float('inf')
        for vehicle in self.detected_vehicles:
            # Simplified distance calculation
            distance = np.sqrt(vehicle.x**2 + vehicle.y**2)
            min_distance = min(min_distance, distance)
        
        if min_distance > self.params["safe_following_distance"].value:
            return 1.0
        else:
            return min_distance / self.params["safe_following_distance"].value
    
    def _assess_current_position(self):
        """Assess if current position is suitable for lane change"""
        if not self.current_lane_pose:
            return 0.5
        
        # Prefer lane changes when well-centered in current lane
        lateral_error = abs(self.current_lane_pose.d)
        if lateral_error < 0.1:
            return 1.0
        elif lateral_error < 0.2:
            return 0.8
        else:
            return 0.5
    
    def _start_signaling(self):
        """Start signaling phase with enhanced LED system"""
        if not self.active_request:
            self._transition_to_state(LaneChangeState.LANE_FOLLOWING)
            return
        
        direction = self.active_request.direction
        self.log(f"Starting signaling for {direction} lane change")
        
        # Determine signal type
        if direction == LaneChangeDirection.LEFT:
            signal_type = SignalType.LANE_CHANGE_LEFT
        else:
            signal_type = SignalType.LANE_CHANGE_RIGHT
        
        # Validate signal timing
        signal_duration = self.params["signal_duration"].value
        is_valid, error_msg = self.signaling_system.validate_signal_timing(signal_type, signal_duration)
        
        if not is_valid:
            self.log(f"Invalid signal timing: {error_msg}", "warn")
            signal_duration = 2.0  # Use minimum valid duration
        
        # Create signal request
        signal_request = SignalRequest(
            signal_type=signal_type,
            duration=signal_duration,
            priority=SignalPriority.HIGH,
            adaptive_brightness=True,
            visibility_verification=True
        )
        
        # Request signaling
        if self.signaling_system.request_signal(signal_request):
            self.log(f"Activated lane change signaling: {signal_type.name}")
        else:
            self.log("Failed to activate lane change signaling", "warn")
    
    def _start_execution(self):
        """Start lane change execution with trajectory generation"""
        if not self.active_request or not self.current_lane_pose:
            self._transition_to_state(LaneChangeState.ABORTING)
            return
        
        self.log("Starting lane change execution")
        self.maneuver_start_time = rospy.Time.now()
        self.trajectory_start_time = rospy.Time.now()
        self.maneuver_progress = 0.0
        self.current_trajectory_index = 0
        
        # Record initial position
        self.initial_lateral_position = self.current_lane_pose.d
        
        # Calculate target lateral offset
        direction = self.active_request.direction
        self.target_lateral_offset = direction * self.params["lane_width"].value
        
        # Generate trajectory for smooth lane change
        start_position = (0.0, self.initial_lateral_position)  # Relative coordinates
        start_heading = self.current_lane_pose.phi
        maneuver_distance = 4.0  # 4 meters longitudinal distance
        velocity = 0.15  # Reduced speed during lane change
        
        try:
            self.current_trajectory = self.trajectory_generator.generate_lane_change_trajectory(
                start_position=start_position,
                start_heading=start_heading,
                target_lateral_offset=self.target_lateral_offset,
                maneuver_distance=maneuver_distance,
                velocity=velocity,
                trajectory_type=TrajectoryType.POLYNOMIAL
            )
            
            # Validate trajectory
            is_valid, violations = self.trajectory_generator.validate_trajectory(self.current_trajectory)
            if not is_valid:
                self.log(f"Generated trajectory is invalid: {violations}", "error")
                self._transition_to_state(LaneChangeState.ABORTING)
                return
            
            # Log trajectory metrics
            metrics = self.trajectory_generator.calculate_trajectory_metrics(self.current_trajectory)
            self.log(f"Generated trajectory: {len(self.current_trajectory)} points, "
                    f"max_curvature={metrics['max_curvature']:.3f}, "
                    f"max_lateral_accel={metrics['max_lateral_acceleration']:.3f}")
            
        except Exception as e:
            self.log(f"Failed to generate trajectory: {e}", "error")
            self._transition_to_state(LaneChangeState.ABORTING)
            return
        
        self.log(f"Target lateral offset: {self.target_lateral_offset:.2f}m, "
                f"trajectory points: {len(self.current_trajectory)}")
    
    def _start_verification(self):
        """Start verification phase"""
        self.log("Starting lane change verification")
        
        # Stop lane change signaling
        if self.active_request:
            if self.active_request.direction == LaneChangeDirection.LEFT:
                signal_type = SignalType.LANE_CHANGE_LEFT
            else:
                signal_type = SignalType.LANE_CHANGE_RIGHT
            
            self.signaling_system.stop_signal(signal_type)
            self.log("Stopped lane change signaling")
    
    def _start_abort(self):
        """Start abort sequence with safe trajectory"""
        self.log("Starting lane change abort sequence")
        
        # Generate abort trajectory
        abort_trajectory = self._generate_abort_trajectory()
        if abort_trajectory:
            self.current_trajectory = abort_trajectory
            self.trajectory_start_time = rospy.Time.now()
            self.log("Using abort trajectory for safe return")
        else:
            # Fallback: stop immediately
            self.current_trajectory = None
            self.log("No abort trajectory available, stopping immediately")
        
        # Stop all lane change signaling and activate hazard lights
        self.signaling_system.stop_signal()  # Stop all signals
        
        # Activate hazard signaling during abort
        hazard_request = SignalRequest(
            signal_type=SignalType.HAZARD,
            duration=5.0,  # 5 seconds of hazard lights
            priority=SignalPriority.HIGH,
            adaptive_brightness=True,
            visibility_verification=False
        )
        
        if self.signaling_system.request_signal(hazard_request):
            self.log("Activated hazard signaling during abort")
        else:
            self.log("Failed to activate hazard signaling", "warn")
    
    def _get_trajectory_point_at_time(self, elapsed_time: float) -> Optional[TrajectoryPoint]:
        """Get the trajectory point at the specified elapsed time"""
        if not self.current_trajectory:
            return None
        
        # Handle time before trajectory start
        if elapsed_time <= 0:
            return self.current_trajectory[0]
        
        # Handle time after trajectory end
        if elapsed_time >= self.current_trajectory[-1].time:
            return self.current_trajectory[-1]
        
        # Find the appropriate trajectory segment
        for i in range(len(self.current_trajectory) - 1):
            if (self.current_trajectory[i].time <= elapsed_time <= 
                self.current_trajectory[i + 1].time):
                
                # Linear interpolation between trajectory points
                t1 = self.current_trajectory[i].time
                t2 = self.current_trajectory[i + 1].time
                p1 = self.current_trajectory[i]
                p2 = self.current_trajectory[i + 1]
                
                if t2 - t1 > 0:
                    alpha = (elapsed_time - t1) / (t2 - t1)
                    
                    # Interpolate position and other properties
                    interpolated_point = TrajectoryPoint(
                        x=p1.x + alpha * (p2.x - p1.x),
                        y=p1.y + alpha * (p2.y - p1.y),
                        heading=p1.heading + alpha * (p2.heading - p1.heading),
                        curvature=p1.curvature + alpha * (p2.curvature - p1.curvature),
                        velocity=p1.velocity + alpha * (p2.velocity - p1.velocity),
                        time=elapsed_time
                    )
                    return interpolated_point
                else:
                    return p1
        
        # Fallback to last point
        return self.current_trajectory[-1]
    
    def _generate_abort_trajectory(self):
        """Generate abort trajectory to return to safe position"""
        if not self.current_lane_pose:
            return None
        
        try:
            # Calculate target position (return to original lane)
            original_lane_center = self.initial_lateral_position
            current_position = (0.0, self.current_lane_pose.d)
            current_heading = self.current_lane_pose.phi
            current_velocity = 0.1  # Slow speed for abort
            
            abort_trajectory = self.trajectory_generator.generate_abort_trajectory(
                current_position=current_position,
                current_heading=current_heading,
                current_velocity=current_velocity,
                target_lane_center=original_lane_center
            )
            
            self.log(f"Generated abort trajectory with {len(abort_trajectory)} points")
            return abort_trajectory
            
        except Exception as e:
            self.log(f"Failed to generate abort trajectory: {e}", "error")
            return None
    
    def _evaluate_traffic_rules_compliance(self):
        """Evaluate traffic rules compliance for lane change"""
        if not self.current_lane_pose or not self.active_request:
            return 0.0
        
        # Create ego vehicle info for traffic rules engine
        ego_vehicle = self._create_ego_vehicle_info()
        
        # Create other vehicles info
        other_vehicles = self._create_other_vehicles_info()
        
        # Create road context
        road_context = self._create_road_context()
        
        # Determine target lane
        current_lane = 1  # Assume lane 1 as default (would be determined from lane pose)
        target_lane = current_lane + self.active_request.direction
        
        # Evaluate lane change legality
        is_legal, violations = self.traffic_rules_engine.evaluate_lane_change_legality(
            ego_vehicle, other_vehicles, road_context, target_lane
        )
        
        # Calculate compliance score based on violations
        if is_legal:
            compliance_score = 1.0
        else:
            # Calculate score based on violation severity
            total_severity = sum(v.severity for v in violations)
            max_possible_severity = len(violations) * 4  # Max severity is 4
            
            if max_possible_severity > 0:
                compliance_score = max(0.0, 1.0 - (total_severity / max_possible_severity))
            else:
                compliance_score = 0.0
        
        # Log violations for debugging
        if violations:
            violation_msgs = [f"{v.rule.description} (severity {v.severity})" for v in violations]
            self.log(f"Traffic rule violations: {'; '.join(violation_msgs)}")
        
        return compliance_score
    
    def _create_ego_vehicle_info(self):
        """Create VehicleInfo for ego vehicle"""
        from lane_change_control.traffic_rules_engine import VehicleInfo, VehicleState
        
        # Estimate velocity from recent pose changes (simplified)
        velocity_x = 0.15  # Default forward velocity
        velocity_y = 0.0
        
        # Determine vehicle state
        if self.state == LaneChangeState.EXECUTING:
            vehicle_state = VehicleState.CHANGING_LANES
        elif self.active_request and self.active_request.direction == LaneChangeDirection.LEFT:
            vehicle_state = VehicleState.TURNING_LEFT
        elif self.active_request and self.active_request.direction == LaneChangeDirection.RIGHT:
            vehicle_state = VehicleState.TURNING_RIGHT
        else:
            vehicle_state = VehicleState.NORMAL_DRIVING
        
        # Check if signaling is active
        signaling_active = self.signaling_system.get_signal_status()['active']
        
        return VehicleInfo(
            vehicle_id="ego",
            position=(0.0, self.current_lane_pose.d),  # Relative coordinates
            velocity=(velocity_x, velocity_y),
            heading=self.current_lane_pose.phi,
            state=vehicle_state,
            lane_id=1,  # Current lane (simplified)
            arrival_time=rospy.Time.now().to_sec(),
            signaling=signaling_active
        )
    
    def _create_other_vehicles_info(self):
        """Create VehicleInfo list for other detected vehicles"""
        from lane_change_control.traffic_rules_engine import VehicleInfo, VehicleState
        
        other_vehicles = []
        for i, corner in enumerate(self.detected_vehicles):
            # Estimate which lane the vehicle is in based on lateral position
            lane_id = 1 if abs(corner.y) < 0.3 else 2  # Simplified lane assignment
            
            vehicle_info = VehicleInfo(
                vehicle_id=f"vehicle_{i}",
                position=(corner.x, corner.y),
                velocity=(0.1, 0.0),  # Estimated velocity
                heading=0.0,  # Unknown heading
                state=VehicleState.NORMAL_DRIVING,
                lane_id=lane_id,
                arrival_time=rospy.Time.now().to_sec() - 1.0,  # Assume arrived earlier
                signaling=False  # Unknown signaling state
            )
            other_vehicles.append(vehicle_info)
        
        return other_vehicles
    
    def _create_road_context(self):
        """Create RoadContext for current situation"""
        from lane_change_control.traffic_rules_engine import RoadContext, RoadElementType
        
        # Determine road element type (simplified - would use map data in practice)
        # For now, assume straight road unless near intersection
        road_element = RoadElementType.STRAIGHT_ROAD
        
        # Check if near intersection (simplified detection)
        if self._is_near_intersection():
            road_element = RoadElementType.INTERSECTION
        
        # Estimate traffic density
        traffic_density = len(self.detected_vehicles) / 10.0  # Vehicles per 10m
        
        return RoadContext(
            current_road_element=road_element,
            intersection_id=None,
            lane_count=2,  # Assume 2-lane road
            speed_limit=0.5,  # 0.5 m/s default
            construction_active=False,
            school_zone_active=False,
            traffic_density=traffic_density
        )
    
    def _is_near_intersection(self):
        """Check if vehicle is near an intersection (simplified)"""
        # In practice, this would use AprilTag detection or map data
        # For now, return False as default
        return False
    
    def _get_intersection_recommendations(self):
        """Get traffic rules recommendations for intersection behavior"""
        if not self.current_lane_pose:
            return []
        
        ego_vehicle = self._create_ego_vehicle_info()
        other_vehicles = self._create_other_vehicles_info()
        road_context = self._create_road_context()
        
        # Get lane change recommendations
        recommendations = self.traffic_rules_engine.get_lane_change_recommendations(
            ego_vehicle, other_vehicles, road_context
        )
        
        # Validate intersection behavior
        is_compliant, issues = self.traffic_rules_engine.validate_intersection_behavior(
            ego_vehicle, other_vehicles, road_context
        )
        
        result = {
            'lane_change_recommendations': recommendations,
            'intersection_compliant': is_compliant,
            'intersection_issues': issues
        }
        
        return result
    
    def _cleanup_maneuver(self):
        """Clean up after maneuver completion or abort"""
        self.active_request = None
        self.maneuver_start_time = None
        self.trajectory_start_time = None
        self.maneuver_progress = 0.0
        self.target_lateral_offset = 0.0
        self.initial_lateral_position = 0.0
        self.current_trajectory = None
        self.current_trajectory_index = 0
        
        # Stop all signaling and return to normal driving
        self.signaling_system.stop_signal()
        self.log("Cleaned up lane change maneuver and reset signaling")
    
    def _update_state_machine(self):
        """Update state machine logic"""
        current_time = rospy.Time.now()
        time_in_state = (current_time - self.state_start_time).to_sec()
        
        with self.state_lock:
            if self.state == LaneChangeState.ASSESSING_CHANGE:
                self._update_assessment_state(time_in_state)
            elif self.state == LaneChangeState.SIGNALING:
                self._update_signaling_state(time_in_state)
            elif self.state == LaneChangeState.EXECUTING:
                self._update_execution_state(time_in_state)
            elif self.state == LaneChangeState.VERIFYING:
                self._update_verification_state(time_in_state)
            elif self.state == LaneChangeState.ABORTING:
                self._update_abort_state(time_in_state)
    
    def _update_assessment_state(self, time_in_state):
        """Update assessment state logic with priority-based decision making"""
        if time_in_state > self.params["assessment_timeout"].value:
            self.log("Assessment timeout - returning to lane following", "warn")
            self._transition_to_state(LaneChangeState.LANE_FOLLOWING)
            return
        
        if not self.active_request:
            self._transition_to_state(LaneChangeState.LANE_FOLLOWING)
            return
        
        # Apply priority-based decision making
        decision_result = self._make_priority_based_decision()
        
        if decision_result['proceed']:
            self.log(f"Priority-based decision: proceeding with lane change "
                    f"(priority: {decision_result['priority']}, "
                    f"score: {self.active_request.feasibility_score:.2f})")
            self._transition_to_state(LaneChangeState.SIGNALING)
        elif time_in_state > 1.0:  # Give some time for assessment
            reason = decision_result.get('reason', 'Unknown')
            self.log(f"Lane change denied by priority system: {reason} "
                    f"(score: {self.active_request.feasibility_score:.2f})")
            self._transition_to_state(LaneChangeState.LANE_FOLLOWING)
    
    def _make_priority_based_decision(self):
        """Make priority-based lane change decision considering traffic rules and urgency"""
        if not self.active_request:
            return {'proceed': False, 'reason': 'No active request'}
        
        # Calculate priority score based on multiple factors
        priority_factors = self._calculate_priority_factors()
        
        # Check traffic rules compliance
        traffic_compliance = self._evaluate_traffic_rules_compliance()
        
        # Get intersection recommendations
        intersection_recommendations = self._get_intersection_recommendations()
        
        # Priority decision matrix
        decision_score = 0.0
        reasons = []
        
        # Factor 1: Urgency of the request (30% weight)
        urgency_weight = 0.3
        urgency_score = min(1.0, self.active_request.urgency / 3.0)  # Normalize to 0-1
        decision_score += urgency_weight * urgency_score
        
        # Factor 2: Traffic rules compliance (40% weight) - CRITICAL
        compliance_weight = 0.4
        if traffic_compliance < 0.3:  # Critical violations
            return {
                'proceed': False, 
                'reason': 'Critical traffic rule violations',
                'priority': 0,
                'traffic_compliance': traffic_compliance
            }
        decision_score += compliance_weight * traffic_compliance
        
        # Factor 3: Safety feasibility (25% weight)
        safety_weight = 0.25
        safety_score = self.active_request.feasibility_score
        decision_score += safety_weight * safety_score
        
        # Factor 4: Intersection compliance (5% weight) - VETO power
        if not intersection_recommendations['intersection_compliant']:
            return {
                'proceed': False,
                'reason': f"Intersection violations: {'; '.join(intersection_recommendations['intersection_issues'])}",
                'priority': 0,
                'traffic_compliance': traffic_compliance
            }
        
        # Additional priority considerations
        
        # Emergency or high-priority situations
        if self.active_request.reason == "emergency_avoidance":
            decision_score += 0.3  # Boost for emergency
            reasons.append("Emergency avoidance priority")
        
        # Obstacle avoidance gets higher priority than optimization
        elif self.active_request.reason == "obstacle_avoidance":
            decision_score += 0.2
            reasons.append("Obstacle avoidance priority")
        
        # Navigation requirements get medium priority
        elif self.active_request.reason == "navigation":
            decision_score += 0.1
            reasons.append("Navigation requirement")
        
        # Efficiency optimization gets lowest priority
        elif self.active_request.reason == "optimization":
            decision_score += 0.05
            reasons.append("Traffic optimization")
        
        # Right-of-way considerations
        if priority_factors['has_right_of_way']:
            decision_score += 0.1
            reasons.append("Has right-of-way")
        
        # Traffic density penalty
        if priority_factors['traffic_density'] > 0.5:  # High traffic
            decision_score -= 0.1
            reasons.append("High traffic density penalty")
        
        # Time-based priority (longer wait = higher priority)
        wait_time = rospy.Time.now().to_sec() - self.state_start_time.to_sec()
        if wait_time > 5.0:  # Waiting more than 5 seconds
            time_bonus = min(0.2, (wait_time - 5.0) / 10.0)  # Up to 0.2 bonus
            decision_score += time_bonus
            reasons.append(f"Wait time bonus ({wait_time:.1f}s)")
        
        # Final decision threshold
        proceed_threshold = 0.7  # Require 70% confidence to proceed
        proceed = decision_score >= proceed_threshold
        
        # Calculate final priority level (1-5 scale)
        priority_level = max(1, min(5, int(decision_score * 5)))
        
        return {
            'proceed': proceed,
            'priority': priority_level,
            'decision_score': decision_score,
            'traffic_compliance': traffic_compliance,
            'reasons': reasons,
            'threshold': proceed_threshold,
            'reason': f"Decision score {decision_score:.2f} {'≥' if proceed else '<'} threshold {proceed_threshold:.2f}"
        }
    
    def _calculate_priority_factors(self):
        """Calculate various priority factors for decision making"""
        factors = {
            'has_right_of_way': False,
            'traffic_density': 0.0,
            'visibility_conditions': 1.0,
            'road_conditions': 1.0
        }
        
        if not self.current_lane_pose:
            return factors
        
        # Create vehicle info for right-of-way calculation
        ego_vehicle = self._create_ego_vehicle_info()
        other_vehicles = self._create_other_vehicles_info()
        road_context = self._create_road_context()
        
        # Check right-of-way status
        factors['has_right_of_way'] = True  # Default assumption
        for other_vehicle in other_vehicles:
            if self.traffic_rules_engine._has_right_of_way(other_vehicle, ego_vehicle, road_context):
                factors['has_right_of_way'] = False
                break
        
        # Calculate traffic density
        factors['traffic_density'] = road_context.traffic_density
        
        # Visibility conditions (simplified - would use camera analysis)
        factors['visibility_conditions'] = 0.8  # Assume good visibility
        
        # Road conditions (simplified - would use sensor data)
        factors['road_conditions'] = 0.9  # Assume good road conditions
        
        return factors
    
    def _update_signaling_state(self, time_in_state):
        """Update signaling state logic with traffic rules compliance"""
        if time_in_state >= self.params["signal_duration"].value:
            # Re-check feasibility and traffic rules compliance before proceeding
            feasibility_score = self._calculate_feasibility_score()
            traffic_compliance = self._evaluate_traffic_rules_compliance()
            
            # Check intersection-specific restrictions
            intersection_recommendations = self._get_intersection_recommendations()
            
            # Ensure minimum signaling time is met (traffic rule requirement)
            min_signal_time = 2.0  # Minimum 2 seconds as per traffic rules
            if time_in_state < min_signal_time:
                self.log(f"Waiting for minimum signal time ({min_signal_time}s)")
                return
            
            # Check if still compliant with traffic rules
            if traffic_compliance < 0.5:  # Major traffic rule violations
                self.log("Traffic rule violations detected during signaling - aborting")
                self._transition_to_state(LaneChangeState.ABORTING)
                return
            
            # Check intersection compliance
            if not intersection_recommendations['intersection_compliant']:
                issues = intersection_recommendations['intersection_issues']
                self.log(f"Intersection compliance issues: {'; '.join(issues)} - aborting")
                self._transition_to_state(LaneChangeState.ABORTING)
                return
            
            # Check overall feasibility
            if feasibility_score >= self.params["min_feasibility_score"].value:
                self.log("Traffic rules compliant - proceeding with lane change")
                self._transition_to_state(LaneChangeState.EXECUTING)
            else:
                self.log("Feasibility changed during signaling - aborting")
                self._transition_to_state(LaneChangeState.ABORTING)
    
    def _update_execution_state(self, time_in_state):
        """Update execution state logic"""
        if time_in_state > self.params["max_maneuver_time"].value:
            self.log("Lane change execution timeout - aborting", "warn")
            self._transition_to_state(LaneChangeState.ABORTING)
            return
        
        # Update progress
        if self.maneuver_start_time:
            elapsed_time = (rospy.Time.now() - self.maneuver_start_time).to_sec()
            expected_duration = self.active_request.estimated_duration if self.active_request else 5.0
            self.maneuver_progress = min(1.0, elapsed_time / expected_duration)
        
        # Check if maneuver is complete
        if self._is_lane_change_complete():
            self._transition_to_state(LaneChangeState.VERIFYING)
        
        # Generate control commands for lane change
        self._generate_lane_change_commands()
    
    def _update_verification_state(self, time_in_state):
        """Update verification state logic"""
        if time_in_state > 2.0:  # Verify for 2 seconds
            if self._verify_lane_change_success():
                self.log("Lane change completed successfully")
                self._transition_to_state(LaneChangeState.LANE_FOLLOWING)
            else:
                self.log("Lane change verification failed - may need correction")
                self._transition_to_state(LaneChangeState.LANE_FOLLOWING)
    
    def _update_abort_state(self, time_in_state):
        """Update abort state logic"""
        if time_in_state > 1.0:  # Allow 1 second for abort actions
            self._transition_to_state(LaneChangeState.LANE_FOLLOWING)
    
    def _is_lane_change_complete(self):
        """Check if lane change maneuver is complete"""
        if not self.current_lane_pose or not self.active_request:
            return False
        
        # Primary check: trajectory completion
        if self.current_trajectory and self.trajectory_start_time:
            elapsed_time = (rospy.Time.now() - self.trajectory_start_time).to_sec()
            trajectory_complete = elapsed_time >= self.current_trajectory[-1].time * 0.9  # 90% completion
            
            if trajectory_complete:
                return True
        
        # Secondary check: geometric completion (fallback)
        current_lateral = self.current_lane_pose.d
        lateral_movement = abs(current_lateral - self.initial_lateral_position)
        target_movement = abs(self.target_lateral_offset) * 0.8  # 80% of target
        
        geometric_complete = lateral_movement >= target_movement
        
        # Also check progress-based completion
        progress_complete = self.maneuver_progress >= 0.9
        
        return geometric_complete or progress_complete
    
    def _verify_lane_change_success(self):
        """Verify that lane change was successful"""
        if not self.current_lane_pose:
            return False
        
        # Check if we're reasonably centered in the new lane
        lateral_error = abs(self.current_lane_pose.d)
        return lateral_error < 0.2  # Within 20cm of lane center
    
    def _generate_lane_change_commands(self):
        """Generate control commands during lane change execution using trajectory"""
        if not self.current_lane_pose or not self.active_request or not self.current_trajectory:
            return
        
        current_time = rospy.Time.now()
        elapsed_time = (current_time - self.trajectory_start_time).to_sec()
        
        # Find the appropriate trajectory point based on elapsed time
        target_point = self._get_trajectory_point_at_time(elapsed_time)
        if not target_point:
            self.log("No valid trajectory point found", "warn")
            return
        
        # Calculate control commands to follow trajectory
        current_lateral = self.current_lane_pose.d
        target_lateral = target_point.y
        
        # Lateral error control
        lateral_error = target_lateral - current_lateral
        
        # Use trajectory curvature for feedforward control
        feedforward_omega = target_point.curvature * target_point.velocity
        
        # Proportional feedback control
        lateral_gain = 3.0
        feedback_omega = lateral_gain * lateral_error
        
        # Combine feedforward and feedback
        omega = feedforward_omega + feedback_omega
        
        # Limit angular velocity
        max_omega = self.params["lateral_acceleration_limit"].value
        omega = np.clip(omega, -max_omega, max_omega)
        
        # Use trajectory velocity
        v = target_point.velocity
        
        # Update progress based on trajectory completion
        if self.current_trajectory:
            progress = min(1.0, elapsed_time / self.current_trajectory[-1].time)
            self.maneuver_progress = progress
        
        # Publish override command
        cmd_msg = Twist2DStamped()
        cmd_msg.header.stamp = current_time
        cmd_msg.v = v
        cmd_msg.omega = omega
        
        self.pub_car_cmd_override.publish(cmd_msg)
        
        # Log trajectory following performance
        if hasattr(self, '_last_log_time'):
            if (current_time - self._last_log_time).to_sec() > 1.0:  # Log every second
                self.log(f"Following trajectory: progress={progress:.2f}, "
                        f"lateral_error={lateral_error:.3f}m, omega={omega:.3f}")
                self._last_log_time = current_time
        else:
            self._last_log_time = current_time
    
    def _publish_status(self):
        """Publish current lane change status with signaling information"""
        status_msg = LaneChangeStatus()
        status_msg.header.stamp = rospy.Time.now()
        status_msg.state = int(self.state)
        
        if self.active_request:
            status_msg.active_request = self.active_request
        
        status_msg.progress = self.maneuver_progress
        
        # Calculate time remaining in current phase
        current_time = rospy.Time.now()
        time_in_state = (current_time - self.state_start_time).to_sec()
        
        if self.state == LaneChangeState.SIGNALING:
            # Get actual signaling time remaining from signaling system
            signal_status = self.signaling_system.get_signal_status()
            if signal_status['active']:
                status_msg.time_remaining = signal_status['signal_duration_remaining']
            else:
                status_msg.time_remaining = max(0, self.params["signal_duration"].value - time_in_state)
        elif self.state == LaneChangeState.EXECUTING:
            expected_duration = self.active_request.estimated_duration if self.active_request else 5.0
            status_msg.time_remaining = max(0, expected_duration - time_in_state)
        else:
            status_msg.time_remaining = 0.0
        
        # Safety status
        status_msg.is_safe_to_proceed = not self.safety_override_active
        
        if self.state == LaneChangeState.ABORTING:
            status_msg.abort_reason = "Safety override or timeout"
        
        self.pub_status.publish(status_msg)
        
        # Update signaling system with current conditions (simplified)
        # In a real implementation, this would use actual sensor data
        ambient_light = 0.5  # Default moderate lighting
        visibility_distance = 3.0  # Default good visibility
        self.signaling_system.update_ambient_conditions(ambient_light, visibility_distance)
    
    def run(self):
        """Main execution loop"""
        rate = rospy.Rate(10)  # 10 Hz
        
        while not rospy.is_shutdown():
            self._update_state_machine()
            self._publish_status()
            rate.sleep()


if __name__ == "__main__":
    node = LaneChangeControllerNode("lane_change_controller_node")
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass