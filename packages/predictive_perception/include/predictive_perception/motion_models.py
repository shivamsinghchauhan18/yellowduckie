#!/usr/bin/env python3

import numpy as np
from abc import ABC, abstractmethod


class MotionModel(ABC):
    """Abstract base class for motion models."""
    
    @abstractmethod
    def predict(self, state, dt):
        """
        Predict next state given current state and time step.
        
        Args:
            state: Current state vector
            dt: Time step
            
        Returns:
            Predicted next state
        """
        pass
    
    @abstractmethod
    def predict_trajectory(self, state, time_horizon, dt):
        """
        Predict trajectory over time horizon.
        
        Args:
            state: Current state vector
            time_horizon: Time to predict ahead
            dt: Time step for prediction
            
        Returns:
            List of predicted states
        """
        pass


class ConstantVelocityModel(MotionModel):
    """Constant velocity motion model."""
    
    def predict(self, state, dt):
        """Predict next state assuming constant velocity."""
        next_state = state.copy()
        next_state[0] += state[2] * dt  # x += vx * dt
        next_state[1] += state[3] * dt  # y += vy * dt
        return next_state
    
    def predict_trajectory(self, state, time_horizon, dt):
        """Predict trajectory assuming constant velocity."""
        trajectory = []
        current_state = state.copy()
        
        steps = int(time_horizon / dt)
        for _ in range(steps):
            current_state = self.predict(current_state, dt)
            trajectory.append(current_state.copy())
        
        return trajectory


class ConstantAccelerationModel(MotionModel):
    """Constant acceleration motion model."""
    
    def __init__(self):
        # State vector: [x, y, vx, vy, ax, ay]
        pass
    
    def predict(self, state, dt):
        """Predict next state assuming constant acceleration."""
        if len(state) < 6:
            # Extend state to include acceleration if not present
            extended_state = np.zeros(6)
            extended_state[:len(state)] = state
            state = extended_state
        
        next_state = state.copy()
        # Position update: x = x + vx*dt + 0.5*ax*dt^2
        next_state[0] += state[2] * dt + 0.5 * state[4] * dt**2
        next_state[1] += state[3] * dt + 0.5 * state[5] * dt**2
        
        # Velocity update: vx = vx + ax*dt
        next_state[2] += state[4] * dt
        next_state[3] += state[5] * dt
        
        # Acceleration remains constant
        return next_state
    
    def predict_trajectory(self, state, time_horizon, dt):
        """Predict trajectory assuming constant acceleration."""
        trajectory = []
        current_state = state.copy()
        
        if len(current_state) < 6:
            # Extend state to include acceleration if not present
            extended_state = np.zeros(6)
            extended_state[:len(current_state)] = current_state
            current_state = extended_state
        
        steps = int(time_horizon / dt)
        for _ in range(steps):
            current_state = self.predict(current_state, dt)
            trajectory.append(current_state.copy())
        
        return trajectory


class InteractingMultipleModel:
    """
    Interacting Multiple Model (IMM) for handling multiple motion hypotheses.
    """
    
    def __init__(self, models, transition_matrix=None, model_probabilities=None):
        """
        Initialize IMM with multiple motion models.
        
        Args:
            models: List of motion model instances
            transition_matrix: Model transition probability matrix
            model_probabilities: Initial model probabilities
        """
        self.models = models
        self.n_models = len(models)
        
        if transition_matrix is None:
            # Default: equal probability of staying in same model or switching
            self.transition_matrix = np.full((self.n_models, self.n_models), 0.1)
            np.fill_diagonal(self.transition_matrix, 0.9)
        else:
            self.transition_matrix = np.array(transition_matrix)
        
        if model_probabilities is None:
            self.model_probabilities = np.ones(self.n_models) / self.n_models
        else:
            self.model_probabilities = np.array(model_probabilities)
        
        # Normalize probabilities
        self.model_probabilities /= np.sum(self.model_probabilities)
    
    def predict(self, state, dt):
        """
        Predict next state using IMM approach.
        
        Args:
            state: Current state vector
            dt: Time step
            
        Returns:
            Weighted prediction from all models
        """
        predictions = []
        max_length = len(state)
        
        for model in self.models:
            pred = model.predict(state, dt)
            # Ensure all predictions have the same length
            if len(pred) > max_length:
                max_length = len(pred)
        
        # Normalize all predictions to the same length
        for model in self.models:
            pred = model.predict(state, dt)
            if len(pred) < max_length:
                # Extend with zeros if needed
                extended_pred = np.zeros(max_length)
                extended_pred[:len(pred)] = pred
                predictions.append(extended_pred)
            else:
                predictions.append(pred[:max_length])
        
        # Weighted average of predictions
        predictions = np.array(predictions)
        weighted_prediction = np.average(predictions, axis=0, weights=self.model_probabilities)
        
        return weighted_prediction
    
    def predict_trajectory(self, state, time_horizon, dt):
        """
        Predict trajectory using IMM approach.
        
        Args:
            state: Current state vector
            time_horizon: Time to predict ahead
            dt: Time step for prediction
            
        Returns:
            List of weighted predictions over time
        """
        trajectory = []
        current_state = state.copy()
        
        steps = int(time_horizon / dt)
        for _ in range(steps):
            current_state = self.predict(current_state, dt)
            trajectory.append(current_state.copy())
        
        return trajectory
    
    def update_model_probabilities(self, likelihood_scores):
        """
        Update model probabilities based on likelihood scores.
        
        Args:
            likelihood_scores: Array of likelihood scores for each model
        """
        if len(likelihood_scores) == self.n_models:
            # Update probabilities using Bayes' rule
            self.model_probabilities *= likelihood_scores
            self.model_probabilities /= np.sum(self.model_probabilities)


class TrajectoryPredictor:
    """
    Advanced trajectory predictor with confidence estimation.
    """
    
    def __init__(self, motion_model=None, confidence_decay=0.95):
        """
        Initialize trajectory predictor.
        
        Args:
            motion_model: Motion model to use for prediction
            confidence_decay: Confidence decay factor over time
        """
        if motion_model is None:
            self.motion_model = ConstantVelocityModel()
        else:
            self.motion_model = motion_model
        
        self.confidence_decay = confidence_decay
    
    def predict_trajectory_with_confidence(self, state, covariance, time_horizon, dt):
        """
        Predict trajectory with confidence estimation.
        
        Args:
            state: Current state vector
            covariance: State covariance matrix
            time_horizon: Time to predict ahead
            dt: Time step for prediction
            
        Returns:
            Tuple of (trajectory_points, confidence_scores)
        """
        trajectory = self.motion_model.predict_trajectory(state, time_horizon, dt)
        
        # Calculate confidence scores based on uncertainty propagation
        confidence_scores = []
        current_confidence = 1.0
        
        for i, predicted_state in enumerate(trajectory):
            # Decay confidence over time
            current_confidence *= self.confidence_decay
            
            # Adjust confidence based on state uncertainty
            if covariance is not None:
                # Use trace of position covariance as uncertainty measure
                position_uncertainty = np.trace(covariance[:2, :2])
                uncertainty_factor = 1.0 / (1.0 + position_uncertainty * 0.01)
                current_confidence *= uncertainty_factor
            
            confidence_scores.append(current_confidence)
        
        return trajectory, confidence_scores
    
    def predict_multiple_hypotheses(self, state, covariance, time_horizon, dt, n_hypotheses=3):
        """
        Predict multiple trajectory hypotheses with different assumptions.
        
        Args:
            state: Current state vector
            covariance: State covariance matrix
            time_horizon: Time to predict ahead
            dt: Time step for prediction
            n_hypotheses: Number of hypotheses to generate
            
        Returns:
            List of (trajectory, confidence) tuples
        """
        hypotheses = []
        
        # Hypothesis 1: Constant velocity
        cv_model = ConstantVelocityModel()
        traj1, conf1 = self.predict_trajectory_with_confidence(
            state, covariance, time_horizon, dt
        )
        hypotheses.append((traj1, conf1, "constant_velocity"))
        
        if n_hypotheses > 1:
            # Hypothesis 2: Deceleration
            decel_state = state.copy()
            if len(decel_state) >= 4:
                # Reduce velocity by 20%
                decel_state[2] *= 0.8
                decel_state[3] *= 0.8
            
            traj2, conf2 = self.predict_trajectory_with_confidence(
                decel_state, covariance, time_horizon, dt
            )
            # Lower confidence for deceleration hypothesis
            conf2 = [c * 0.8 for c in conf2]
            hypotheses.append((traj2, conf2, "deceleration"))
        
        if n_hypotheses > 2:
            # Hypothesis 3: Slight course change
            turn_state = state.copy()
            if len(turn_state) >= 4:
                # Add small perpendicular velocity component
                perp_vel = np.array([-turn_state[3], turn_state[2]]) * 0.1
                turn_state[2] += perp_vel[0]
                turn_state[3] += perp_vel[1]
            
            traj3, conf3 = self.predict_trajectory_with_confidence(
                turn_state, covariance, time_horizon, dt
            )
            # Lower confidence for turning hypothesis
            conf3 = [c * 0.6 for c in conf3]
            hypotheses.append((traj3, conf3, "course_change"))
        
        return hypotheses
    
    def calculate_trajectory_confidence(self, trajectory, reference_trajectory=None):
        """
        Calculate overall confidence for a trajectory.
        
        Args:
            trajectory: List of predicted states
            reference_trajectory: Optional reference for comparison
            
        Returns:
            Overall confidence score
        """
        if not trajectory:
            return 0.0
        
        # Base confidence starts high and decays
        base_confidence = 1.0
        
        # Factor in trajectory smoothness
        smoothness_score = self._calculate_smoothness(trajectory)
        
        # Factor in consistency with reference if available
        consistency_score = 1.0
        if reference_trajectory is not None:
            consistency_score = self._calculate_consistency(trajectory, reference_trajectory)
        
        # Combine factors
        overall_confidence = base_confidence * smoothness_score * consistency_score
        
        return max(0.0, min(1.0, overall_confidence))
    
    def _calculate_smoothness(self, trajectory):
        """Calculate trajectory smoothness score."""
        if len(trajectory) < 3:
            return 1.0
        
        # Calculate acceleration changes (jerk)
        accelerations = []
        for i in range(1, len(trajectory) - 1):
            prev_state = trajectory[i - 1]
            curr_state = trajectory[i]
            next_state = trajectory[i + 1]
            
            # Approximate acceleration
            acc_x = next_state[0] - 2 * curr_state[0] + prev_state[0]
            acc_y = next_state[1] - 2 * curr_state[1] + prev_state[1]
            acc_magnitude = np.sqrt(acc_x**2 + acc_y**2)
            accelerations.append(acc_magnitude)
        
        # Lower acceleration changes indicate smoother trajectory
        avg_acceleration = np.mean(accelerations)
        smoothness = 1.0 / (1.0 + avg_acceleration * 0.1)
        
        return smoothness
    
    def _calculate_consistency(self, trajectory, reference_trajectory):
        """Calculate consistency with reference trajectory."""
        if len(trajectory) != len(reference_trajectory):
            return 0.5
        
        # Calculate average distance between trajectories
        distances = []
        for pred_state, ref_state in zip(trajectory, reference_trajectory):
            dist = np.sqrt((pred_state[0] - ref_state[0])**2 + 
                          (pred_state[1] - ref_state[1])**2)
            distances.append(dist)
        
        avg_distance = np.mean(distances)
        consistency = 1.0 / (1.0 + avg_distance * 0.1)
        
        return consistency