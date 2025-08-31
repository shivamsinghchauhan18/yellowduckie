#!/usr/bin/env python3

import unittest
import numpy as np
import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'include'))

from predictive_perception.motion_models import (
    ConstantVelocityModel, 
    ConstantAccelerationModel,
    InteractingMultipleModel,
    TrajectoryPredictor
)


class TestMotionModels(unittest.TestCase):
    """Test cases for motion models."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.cv_model = ConstantVelocityModel()
        self.ca_model = ConstantAccelerationModel()
        
        # Test state: [x, y, vx, vy]
        self.test_state = np.array([0.0, 0.0, 2.0, 1.0])
        self.dt = 1.0
    
    def test_constant_velocity_prediction(self):
        """Test constant velocity model prediction."""
        predicted_state = self.cv_model.predict(self.test_state, self.dt)
        
        # Expected: x = 0 + 2*1 = 2, y = 0 + 1*1 = 1
        expected_state = np.array([2.0, 1.0, 2.0, 1.0])
        np.testing.assert_array_almost_equal(predicted_state, expected_state)
    
    def test_constant_velocity_trajectory(self):
        """Test constant velocity trajectory prediction."""
        time_horizon = 3.0
        dt = 1.0
        
        trajectory = self.cv_model.predict_trajectory(self.test_state, time_horizon, dt)
        
        # Should have 3 points
        self.assertEqual(len(trajectory), 3)
        
        # Check trajectory points
        expected_points = [
            np.array([2.0, 1.0, 2.0, 1.0]),  # t=1
            np.array([4.0, 2.0, 2.0, 1.0]),  # t=2
            np.array([6.0, 3.0, 2.0, 1.0])   # t=3
        ]
        
        for i, expected in enumerate(expected_points):
            np.testing.assert_array_almost_equal(trajectory[i], expected)
    
    def test_constant_acceleration_prediction(self):
        """Test constant acceleration model prediction."""
        # State with acceleration: [x, y, vx, vy, ax, ay]
        state_with_accel = np.array([0.0, 0.0, 2.0, 1.0, 0.5, 0.2])
        
        predicted_state = self.ca_model.predict(state_with_accel, self.dt)
        
        # Expected: 
        # x = 0 + 2*1 + 0.5*0.5*1^2 = 2.25
        # y = 0 + 1*1 + 0.5*0.2*1^2 = 1.1
        # vx = 2 + 0.5*1 = 2.5
        # vy = 1 + 0.2*1 = 1.2
        expected_state = np.array([2.25, 1.1, 2.5, 1.2, 0.5, 0.2])
        np.testing.assert_array_almost_equal(predicted_state, expected_state)
    
    def test_constant_acceleration_trajectory(self):
        """Test constant acceleration trajectory prediction."""
        state_with_accel = np.array([0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
        time_horizon = 2.0
        dt = 1.0
        
        trajectory = self.ca_model.predict_trajectory(state_with_accel, time_horizon, dt)
        
        # Should have 2 points
        self.assertEqual(len(trajectory), 2)
        
        # First point: x = 0 + 1*1 + 0.5*1*1^2 = 1.5, vx = 1 + 1*1 = 2
        # Second point: x = 1.5 + 2*1 + 0.5*1*1^2 = 4, vx = 2 + 1*1 = 3
        expected_first = np.array([1.5, 0.0, 2.0, 0.0, 1.0, 0.0])
        expected_second = np.array([4.0, 0.0, 3.0, 0.0, 1.0, 0.0])
        
        np.testing.assert_array_almost_equal(trajectory[0], expected_first)
        np.testing.assert_array_almost_equal(trajectory[1], expected_second)
    
    def test_interacting_multiple_model(self):
        """Test Interacting Multiple Model."""
        models = [self.cv_model, self.ca_model]
        imm = InteractingMultipleModel(models)
        
        # Test prediction
        predicted_state = imm.predict(self.test_state, self.dt)
        
        # Should be a weighted average of CV and CA predictions
        # CA model extends to 6 dimensions, so result should be 6
        self.assertGreaterEqual(len(predicted_state), 4)
        self.assertIsInstance(predicted_state, np.ndarray)
    
    def test_imm_model_probabilities(self):
        """Test IMM model probability updates."""
        models = [self.cv_model, self.ca_model]
        imm = InteractingMultipleModel(models)
        
        # Initial probabilities should be equal
        np.testing.assert_array_almost_equal(imm.model_probabilities, [0.5, 0.5])
        
        # Update with likelihood scores favoring first model
        imm.update_model_probabilities([0.8, 0.2])
        
        # First model should have higher probability
        self.assertGreater(imm.model_probabilities[0], imm.model_probabilities[1])
        
        # Probabilities should sum to 1
        self.assertAlmostEqual(np.sum(imm.model_probabilities), 1.0)


class TestTrajectoryPredictor(unittest.TestCase):
    """Test cases for trajectory predictor."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.predictor = TrajectoryPredictor()
        self.test_state = np.array([0.0, 0.0, 2.0, 1.0])
        self.test_covariance = np.eye(4) * 0.1
    
    def test_trajectory_with_confidence(self):
        """Test trajectory prediction with confidence estimation."""
        time_horizon = 3.0
        dt = 1.0
        
        trajectory, confidences = self.predictor.predict_trajectory_with_confidence(
            self.test_state, self.test_covariance, time_horizon, dt
        )
        
        # Should have trajectory points and confidence scores
        self.assertEqual(len(trajectory), 3)
        self.assertEqual(len(confidences), 3)
        
        # Confidence should decay over time
        self.assertGreater(confidences[0], confidences[1])
        self.assertGreater(confidences[1], confidences[2])
        
        # All confidences should be between 0 and 1
        for conf in confidences:
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 1.0)
    
    def test_multiple_hypotheses(self):
        """Test multiple trajectory hypotheses generation."""
        time_horizon = 2.0
        dt = 1.0
        n_hypotheses = 3
        
        hypotheses = self.predictor.predict_multiple_hypotheses(
            self.test_state, self.test_covariance, time_horizon, dt, n_hypotheses
        )
        
        # Should have requested number of hypotheses
        self.assertEqual(len(hypotheses), n_hypotheses)
        
        # Each hypothesis should have trajectory, confidence, and label
        for trajectory, confidences, label in hypotheses:
            self.assertIsInstance(trajectory, list)
            self.assertIsInstance(confidences, list)
            self.assertIsInstance(label, str)
            self.assertEqual(len(trajectory), len(confidences))
    
    def test_trajectory_confidence_calculation(self):
        """Test trajectory confidence calculation."""
        # Create a smooth trajectory
        smooth_trajectory = [
            np.array([0.0, 0.0, 1.0, 0.0]),
            np.array([1.0, 0.0, 1.0, 0.0]),
            np.array([2.0, 0.0, 1.0, 0.0]),
            np.array([3.0, 0.0, 1.0, 0.0])
        ]
        
        confidence = self.predictor.calculate_trajectory_confidence(smooth_trajectory)
        
        # Should have reasonable confidence
        self.assertGreater(confidence, 0.5)
        self.assertLessEqual(confidence, 1.0)
    
    def test_confidence_decay(self):
        """Test confidence decay factor."""
        predictor_fast_decay = TrajectoryPredictor(confidence_decay=0.5)
        predictor_slow_decay = TrajectoryPredictor(confidence_decay=0.9)
        
        time_horizon = 3.0
        dt = 1.0
        
        _, conf_fast = predictor_fast_decay.predict_trajectory_with_confidence(
            self.test_state, self.test_covariance, time_horizon, dt
        )
        
        _, conf_slow = predictor_slow_decay.predict_trajectory_with_confidence(
            self.test_state, self.test_covariance, time_horizon, dt
        )
        
        # Fast decay should result in lower final confidence
        self.assertLess(conf_fast[-1], conf_slow[-1])
    
    def test_empty_trajectory_handling(self):
        """Test handling of empty trajectories."""
        empty_trajectory = []
        
        confidence = self.predictor.calculate_trajectory_confidence(empty_trajectory)
        self.assertEqual(confidence, 0.0)
    
    def test_prediction_accuracy_over_time(self):
        """Test prediction accuracy over different time horizons."""
        short_horizon = 1.0
        long_horizon = 5.0
        dt = 0.5
        
        # Short horizon prediction
        traj_short, conf_short = self.predictor.predict_trajectory_with_confidence(
            self.test_state, self.test_covariance, short_horizon, dt
        )
        
        # Long horizon prediction
        traj_long, conf_long = self.predictor.predict_trajectory_with_confidence(
            self.test_state, self.test_covariance, long_horizon, dt
        )
        
        # Short horizon should have higher average confidence
        avg_conf_short = np.mean(conf_short)
        avg_conf_long = np.mean(conf_long)
        
        self.assertGreater(avg_conf_short, avg_conf_long)
    
    def test_uncertainty_impact_on_confidence(self):
        """Test impact of state uncertainty on confidence."""
        low_uncertainty = np.eye(4) * 0.01
        high_uncertainty = np.eye(4) * 1.0
        
        time_horizon = 2.0
        dt = 1.0
        
        # Prediction with low uncertainty
        _, conf_low = self.predictor.predict_trajectory_with_confidence(
            self.test_state, low_uncertainty, time_horizon, dt
        )
        
        # Prediction with high uncertainty
        _, conf_high = self.predictor.predict_trajectory_with_confidence(
            self.test_state, high_uncertainty, time_horizon, dt
        )
        
        # Low uncertainty should result in higher confidence
        avg_conf_low = np.mean(conf_low)
        avg_conf_high = np.mean(conf_high)
        
        self.assertGreater(avg_conf_low, avg_conf_high)


if __name__ == '__main__':
    unittest.main()