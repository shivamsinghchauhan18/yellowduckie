#!/usr/bin/env python3

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass
from enum import IntEnum


class TrajectoryType(IntEnum):
    """Types of lane change trajectories"""
    POLYNOMIAL = 0
    SPLINE = 1
    CLOTHOID = 2


@dataclass
class TrajectoryPoint:
    """A point along the trajectory"""
    x: float  # Longitudinal position (meters)
    y: float  # Lateral position (meters)
    heading: float  # Heading angle (radians)
    curvature: float  # Path curvature (1/meters)
    velocity: float  # Desired velocity (m/s)
    time: float  # Time from trajectory start (seconds)


@dataclass
class TrajectoryConstraints:
    """Constraints for trajectory generation"""
    max_lateral_acceleration: float = 0.5  # m/s²
    max_lateral_velocity: float = 0.3  # m/s
    max_curvature: float = 2.0  # 1/m
    comfort_factor: float = 0.8  # Reduce limits for comfort (0-1)
    safety_margin: float = 0.1  # Additional safety margin (meters)


class LaneChangeTrajectoryGenerator:
    """
    Generates smooth trajectories for lane change maneuvers.
    
    Supports multiple trajectory types:
    - Polynomial trajectories for simple lane changes
    - Spline trajectories for complex scenarios
    - Clothoid trajectories for optimal curvature profiles
    """
    
    def __init__(self, constraints: TrajectoryConstraints = None):
        """Initialize trajectory generator with constraints"""
        self.constraints = constraints or TrajectoryConstraints()
        
        # Apply comfort factor to constraints
        self.effective_max_lateral_accel = (
            self.constraints.max_lateral_acceleration * 
            self.constraints.comfort_factor
        )
        self.effective_max_lateral_vel = (
            self.constraints.max_lateral_velocity * 
            self.constraints.comfort_factor
        )
        self.effective_max_curvature = (
            self.constraints.max_curvature * 
            self.constraints.comfort_factor
        )
    
    def generate_lane_change_trajectory(self,
                                       start_position: Tuple[float, float],
                                       start_heading: float,
                                       target_lateral_offset: float,
                                       maneuver_distance: float,
                                       velocity: float,
                                       trajectory_type: TrajectoryType = TrajectoryType.POLYNOMIAL) -> List[TrajectoryPoint]:
        """
        Generate a complete lane change trajectory.
        
        Args:
            start_position: (x, y) starting position
            start_heading: Initial heading angle (radians)
            target_lateral_offset: Lateral distance to move (meters, negative for left)
            maneuver_distance: Longitudinal distance over which to perform maneuver (meters)
            velocity: Desired velocity during maneuver (m/s)
            trajectory_type: Type of trajectory to generate
            
        Returns:
            List of trajectory points
        """
        if trajectory_type == TrajectoryType.POLYNOMIAL:
            return self._generate_polynomial_trajectory(
                start_position, start_heading, target_lateral_offset, 
                maneuver_distance, velocity
            )
        elif trajectory_type == TrajectoryType.SPLINE:
            return self._generate_spline_trajectory(
                start_position, start_heading, target_lateral_offset, 
                maneuver_distance, velocity
            )
        elif trajectory_type == TrajectoryType.CLOTHOID:
            return self._generate_clothoid_trajectory(
                start_position, start_heading, target_lateral_offset, 
                maneuver_distance, velocity
            )
        else:
            raise ValueError(f"Unsupported trajectory type: {trajectory_type}")
    
    def _generate_polynomial_trajectory(self,
                                       start_pos: Tuple[float, float],
                                       start_heading: float,
                                       lateral_offset: float,
                                       distance: float,
                                       velocity: float) -> List[TrajectoryPoint]:
        """Generate polynomial trajectory (5th order for smooth acceleration)"""
        
        # Boundary conditions
        x0, y0 = start_pos
        x1 = x0 + distance
        y1 = y0 + lateral_offset
        
        # Initial and final derivatives
        dy_dx_0 = np.tan(start_heading)  # Initial slope
        dy_dx_1 = 0.0  # Final slope (parallel to lane)
        d2y_dx2_0 = 0.0  # Initial curvature
        d2y_dx2_1 = 0.0  # Final curvature
        
        # Solve for 5th order polynomial coefficients
        # y = a0 + a1*x + a2*x^2 + a3*x^3 + a4*x^4 + a5*x^5
        # Relative to start position
        
        # Normalize x to [0, 1] for better numerical stability
        coeffs = self._solve_polynomial_coefficients(
            lateral_offset, dy_dx_0, dy_dx_1, d2y_dx2_0, d2y_dx2_1
        )
        
        # Generate trajectory points
        num_points = max(20, int(distance / 0.1))  # At least 20 points, or every 10cm
        trajectory = []
        
        for i in range(num_points + 1):
            # Normalized parameter [0, 1]
            s = i / num_points
            x = x0 + s * distance
            
            # Calculate lateral position using polynomial
            y = y0 + self._evaluate_polynomial(coeffs, s) * lateral_offset
            
            # Calculate heading from derivative
            dy_ds = self._evaluate_polynomial_derivative(coeffs, s) * lateral_offset
            dy_dx = dy_ds / distance if distance > 0 else 0
            heading = np.arctan(dy_dx)
            
            # Calculate curvature from second derivative
            d2y_ds2 = self._evaluate_polynomial_second_derivative(coeffs, s) * lateral_offset
            d2y_dx2 = d2y_ds2 / (distance ** 2) if distance > 0 else 0
            curvature = d2y_dx2 / ((1 + dy_dx**2) ** 1.5)
            
            # Time calculation
            time = (x - x0) / velocity if velocity > 0 else 0
            
            trajectory.append(TrajectoryPoint(
                x=x, y=y, heading=heading, curvature=curvature,
                velocity=velocity, time=time
            ))
        
        return trajectory
    
    def _solve_polynomial_coefficients(self,
                                      lateral_offset: float,
                                      dy_dx_0: float,
                                      dy_dx_1: float,
                                      d2y_dx2_0: float,
                                      d2y_dx2_1: float) -> np.ndarray:
        """Solve for 5th order polynomial coefficients"""
        
        # For normalized parameter s ∈ [0, 1]
        # Boundary conditions:
        # y(0) = 0, y(1) = 1 (normalized lateral offset)
        # y'(0) = dy_dx_0, y'(1) = dy_dx_1
        # y''(0) = d2y_dx2_0, y''(1) = d2y_dx2_1
        
        # Coefficient matrix for 5th order polynomial
        # [1, 0, 0, 0, 0, 0] [a0]   [0]
        # [1, 1, 1, 1, 1, 1] [a1]   [1]
        # [0, 1, 0, 0, 0, 0] [a2] = [dy_dx_0]
        # [0, 1, 2, 3, 4, 5] [a3]   [dy_dx_1]
        # [0, 0, 2, 0, 0, 0] [a4]   [d2y_dx2_0]
        # [0, 0, 2, 6,12,20] [a5]   [d2y_dx2_1]
        
        A = np.array([
            [1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [0, 1, 0, 0, 0, 0],
            [0, 1, 2, 3, 4, 5],
            [0, 0, 2, 0, 0, 0],
            [0, 0, 2, 6, 12, 20]
        ])
        
        b = np.array([0, 1, dy_dx_0, dy_dx_1, d2y_dx2_0, d2y_dx2_1])
        
        # Solve for coefficients
        coeffs = np.linalg.solve(A, b)
        return coeffs
    
    def _evaluate_polynomial(self, coeffs: np.ndarray, s: float) -> float:
        """Evaluate polynomial at parameter s"""
        return sum(coeffs[i] * (s ** i) for i in range(len(coeffs)))
    
    def _evaluate_polynomial_derivative(self, coeffs: np.ndarray, s: float) -> float:
        """Evaluate polynomial first derivative at parameter s"""
        if len(coeffs) <= 1:
            return 0.0
        return sum(i * coeffs[i] * (s ** (i-1)) for i in range(1, len(coeffs)))
    
    def _evaluate_polynomial_second_derivative(self, coeffs: np.ndarray, s: float) -> float:
        """Evaluate polynomial second derivative at parameter s"""
        if len(coeffs) <= 2:
            return 0.0
        return sum(i * (i-1) * coeffs[i] * (s ** (i-2)) for i in range(2, len(coeffs)))
    
    def _generate_spline_trajectory(self,
                                   start_pos: Tuple[float, float],
                                   start_heading: float,
                                   lateral_offset: float,
                                   distance: float,
                                   velocity: float) -> List[TrajectoryPoint]:
        """Generate spline-based trajectory (simplified implementation)"""
        # For now, use polynomial trajectory as spline implementation
        # In a full implementation, this would use cubic splines with
        # multiple control points for more complex scenarios
        return self._generate_polynomial_trajectory(
            start_pos, start_heading, lateral_offset, distance, velocity
        )
    
    def _generate_clothoid_trajectory(self,
                                     start_pos: Tuple[float, float],
                                     start_heading: float,
                                     lateral_offset: float,
                                     distance: float,
                                     velocity: float) -> List[TrajectoryPoint]:
        """Generate clothoid-based trajectory (simplified implementation)"""
        # Clothoid (Euler spiral) provides linear curvature change
        # For now, use polynomial trajectory as approximation
        # Full implementation would use Fresnel integrals
        return self._generate_polynomial_trajectory(
            start_pos, start_heading, lateral_offset, distance, velocity
        )
    
    def validate_trajectory(self, trajectory: List[TrajectoryPoint]) -> Tuple[bool, List[str]]:
        """
        Validate trajectory against constraints.
        
        Returns:
            (is_valid, list_of_violations)
        """
        violations = []
        
        if not trajectory:
            return False, ["Empty trajectory"]
        
        for i, point in enumerate(trajectory):
            # Check curvature constraint
            if abs(point.curvature) > self.effective_max_curvature:
                violations.append(f"Point {i}: Curvature {point.curvature:.3f} exceeds limit {self.effective_max_curvature:.3f}")
            
            # Check lateral acceleration (v² * κ)
            lateral_accel = point.velocity ** 2 * abs(point.curvature)
            if lateral_accel > self.effective_max_lateral_accel:
                violations.append(f"Point {i}: Lateral acceleration {lateral_accel:.3f} exceeds limit {self.effective_max_lateral_accel:.3f}")
        
        # Check lateral velocity between consecutive points
        for i in range(1, len(trajectory)):
            dt = trajectory[i].time - trajectory[i-1].time
            if dt > 0:
                dy = trajectory[i].y - trajectory[i-1].y
                lateral_vel = abs(dy / dt)
                if lateral_vel > self.effective_max_lateral_vel:
                    violations.append(f"Segment {i-1}-{i}: Lateral velocity {lateral_vel:.3f} exceeds limit {self.effective_max_lateral_vel:.3f}")
        
        return len(violations) == 0, violations
    
    def optimize_trajectory_for_comfort(self, trajectory: List[TrajectoryPoint]) -> List[TrajectoryPoint]:
        """
        Optimize trajectory for passenger comfort by smoothing acceleration profiles.
        """
        if len(trajectory) < 3:
            return trajectory
        
        # Apply smoothing to curvature profile
        smoothed_trajectory = []
        
        for i, point in enumerate(trajectory):
            if i == 0 or i == len(trajectory) - 1:
                # Keep endpoints unchanged
                smoothed_trajectory.append(point)
            else:
                # Apply simple moving average to curvature
                prev_curv = trajectory[i-1].curvature
                curr_curv = trajectory[i].curvature
                next_curv = trajectory[i+1].curvature
                
                smoothed_curvature = (prev_curv + 2*curr_curv + next_curv) / 4
                
                # Create new point with smoothed curvature
                smoothed_point = TrajectoryPoint(
                    x=point.x,
                    y=point.y,
                    heading=point.heading,
                    curvature=smoothed_curvature,
                    velocity=point.velocity,
                    time=point.time
                )
                smoothed_trajectory.append(smoothed_point)
        
        return smoothed_trajectory
    
    def calculate_trajectory_metrics(self, trajectory: List[TrajectoryPoint]) -> dict:
        """Calculate metrics for trajectory analysis"""
        if not trajectory:
            return {}
        
        # Calculate maximum values
        max_curvature = max(abs(p.curvature) for p in trajectory)
        max_lateral_accel = max(p.velocity**2 * abs(p.curvature) for p in trajectory)
        
        # Calculate lateral velocities
        lateral_velocities = []
        for i in range(1, len(trajectory)):
            dt = trajectory[i].time - trajectory[i-1].time
            if dt > 0:
                dy = trajectory[i].y - trajectory[i-1].y
                lateral_velocities.append(abs(dy / dt))
        
        max_lateral_vel = max(lateral_velocities) if lateral_velocities else 0.0
        
        # Calculate total trajectory length
        total_length = 0.0
        for i in range(1, len(trajectory)):
            dx = trajectory[i].x - trajectory[i-1].x
            dy = trajectory[i].y - trajectory[i-1].y
            total_length += np.sqrt(dx**2 + dy**2)
        
        # Calculate smoothness (curvature variation)
        curvature_changes = []
        for i in range(1, len(trajectory)):
            curvature_changes.append(abs(trajectory[i].curvature - trajectory[i-1].curvature))
        
        avg_curvature_change = np.mean(curvature_changes) if curvature_changes else 0.0
        
        return {
            'max_curvature': max_curvature,
            'max_lateral_acceleration': max_lateral_accel,
            'max_lateral_velocity': max_lateral_vel,
            'total_length': total_length,
            'total_time': trajectory[-1].time,
            'average_curvature_change': avg_curvature_change,
            'num_points': len(trajectory)
        }
    
    def generate_abort_trajectory(self,
                                 current_position: Tuple[float, float],
                                 current_heading: float,
                                 current_velocity: float,
                                 target_lane_center: float) -> List[TrajectoryPoint]:
        """
        Generate emergency abort trajectory to return to safe position.
        
        This creates a trajectory that quickly but safely returns the vehicle
        to the original lane or a safe position.
        """
        x0, y0 = current_position
        
        # Calculate lateral offset needed to return to target lane center
        lateral_offset = target_lane_center - y0
        
        # Use shorter distance for abort maneuver
        abort_distance = min(3.0, abs(lateral_offset) * 8)  # Aggressive but safe
        
        # Reduce velocity for safety during abort
        abort_velocity = min(current_velocity, 0.1)
        
        # Generate trajectory with higher curvature limits for emergency
        original_constraints = self.constraints
        
        # Temporarily increase limits for abort (but still safe)
        abort_constraints = TrajectoryConstraints(
            max_lateral_acceleration=original_constraints.max_lateral_acceleration * 1.2,
            max_lateral_velocity=original_constraints.max_lateral_velocity * 1.1,
            max_curvature=original_constraints.max_curvature * 1.1,
            comfort_factor=0.6,  # Reduced comfort for emergency
            safety_margin=original_constraints.safety_margin
        )
        
        self.constraints = abort_constraints
        
        try:
            abort_trajectory = self._generate_polynomial_trajectory(
                current_position, current_heading, lateral_offset, 
                abort_distance, abort_velocity
            )
        finally:
            # Restore original constraints
            self.constraints = original_constraints
        
        return abort_trajectory