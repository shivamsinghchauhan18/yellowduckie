#!/usr/bin/env python3

import numpy as np


class KalmanTracker:
    """
    Kalman filter-based object tracker for 2D motion with constant velocity model.
    State vector: [x, y, vx, vy] where (x,y) is position and (vx,vy) is velocity.
    """
    
    def __init__(self, initial_position, object_id, object_class, confidence=1.0):
        """
        Initialize Kalman tracker with initial position.
        
        Args:
            initial_position: (x, y) tuple of initial position
            object_id: Unique identifier for this object
            object_class: Class of the object (0=duck, 1=duckiebot, 2=other)
            confidence: Initial confidence score
        """
        self.object_id = object_id
        self.object_class = object_class
        self.confidence = confidence
        self.age = 0
        self.time_since_update = 0
        self.hit_streak = 0
        self.hits = 1
        
        # State vector [x, y, vx, vy]
        self.state = np.array([initial_position[0], initial_position[1], 0.0, 0.0])
        
        # State covariance matrix
        self.covariance = np.eye(4)
        self.covariance[2:, 2:] *= 1000.0  # High uncertainty for velocity initially
        
        # Motion model (constant velocity)
        self.F = np.array([
            [1, 0, 1, 0],  # x = x + vx*dt
            [0, 1, 0, 1],  # y = y + vy*dt
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ])
        
        # Observation model (we observe position only)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Process noise covariance
        self.Q = np.eye(4)
        self.Q[2:, 2:] *= 0.01  # Small process noise for velocity
        
        # Measurement noise covariance
        self.R = np.eye(2) * 10.0  # Measurement uncertainty
        
    def predict(self, dt=1.0):
        """
        Predict the next state using the motion model.
        
        Args:
            dt: Time step
        """
        # Update motion model with time step
        F = self.F.copy()
        F[0, 2] = dt
        F[1, 3] = dt
        
        # Process noise scaled by time
        Q = self.Q * dt
        
        # Predict state and covariance
        self.state = F @ self.state
        self.covariance = F @ self.covariance @ F.T + Q
        
        self.age += 1
        self.time_since_update += 1
        
        return self.state[:2]  # Return predicted position
    
    def update(self, measurement, confidence=None):
        """
        Update the tracker with a new measurement.
        
        Args:
            measurement: (x, y) position measurement
            confidence: Confidence score for this measurement
        """
        if confidence is not None:
            # Update confidence with exponential moving average
            self.confidence = 0.7 * self.confidence + 0.3 * confidence
        
        # Kalman filter update
        z = np.array(measurement)
        y = z - self.H @ self.state  # Innovation
        S = self.H @ self.covariance @ self.H.T + self.R  # Innovation covariance
        K = self.covariance @ self.H.T @ np.linalg.inv(S)  # Kalman gain
        
        # Update state and covariance
        self.state = self.state + K @ y
        I_KH = np.eye(4) - K @ self.H
        self.covariance = I_KH @ self.covariance
        
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
    
    def get_position(self):
        """Get current position estimate."""
        return self.state[:2]
    
    def get_velocity(self):
        """Get current velocity estimate."""
        return self.state[2:]
    
    def get_acceleration(self):
        """Get estimated acceleration (simplified as zero for constant velocity model)."""
        return np.array([0.0, 0.0])
    
    def predict_trajectory(self, time_horizon, dt=0.1):
        """
        Predict trajectory over a given time horizon.
        
        Args:
            time_horizon: Time to predict ahead (seconds)
            dt: Time step for prediction
            
        Returns:
            List of (x, y) positions over time
        """
        trajectory = []
        current_state = self.state.copy()
        
        steps = int(time_horizon / dt)
        for _ in range(steps):
            # Predict next position using constant velocity
            current_state[0] += current_state[2] * dt  # x += vx * dt
            current_state[1] += current_state[3] * dt  # y += vy * dt
            trajectory.append((current_state[0], current_state[1]))
        
        return trajectory
    
    def predict_trajectory_with_confidence(self, time_horizon, dt=0.1):
        """
        Predict trajectory with confidence estimation.
        
        Args:
            time_horizon: Time to predict ahead (seconds)
            dt: Time step for prediction
            
        Returns:
            Tuple of (trajectory_points, confidence_scores)
        """
        trajectory = self.predict_trajectory(time_horizon, dt)
        
        # Calculate confidence scores based on tracker confidence and time
        confidence_scores = []
        base_confidence = self.confidence
        decay_factor = 0.95
        
        current_confidence = base_confidence
        for _ in trajectory:
            confidence_scores.append(current_confidence)
            current_confidence *= decay_factor
        
        return trajectory, confidence_scores
    
    def is_valid(self, max_age=30, min_hits=3):
        """
        Check if tracker is still valid based on age and hit count.
        
        Args:
            max_age: Maximum age without updates
            min_hits: Minimum hits required for validity
            
        Returns:
            True if tracker is valid
        """
        return (self.time_since_update <= max_age and 
                (self.hits >= min_hits or self.hit_streak >= 1))
    
    def should_delete(self, max_age=30):
        """
        Check if tracker should be deleted.
        
        Args:
            max_age: Maximum age without updates
            
        Returns:
            True if tracker should be deleted
        """
        return self.time_since_update > max_age