#!/usr/bin/env python3

import unittest
import numpy as np
import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'include'))

from predictive_perception.kalman_tracker import KalmanTracker


class TestKalmanTracker(unittest.TestCase):
    """Test cases for KalmanTracker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.initial_position = (10.0, 20.0)
        self.object_id = 1
        self.object_class = 0  # Duck
        self.confidence = 0.8
        
        self.tracker = KalmanTracker(
            self.initial_position,
            self.object_id,
            self.object_class,
            self.confidence
        )
    
    def test_initialization(self):
        """Test tracker initialization."""
        self.assertEqual(self.tracker.object_id, self.object_id)
        self.assertEqual(self.tracker.object_class, self.object_class)
        self.assertEqual(self.tracker.confidence, self.confidence)
        
        # Check initial state
        np.testing.assert_array_almost_equal(
            self.tracker.get_position(), 
            np.array(self.initial_position)
        )
        np.testing.assert_array_almost_equal(
            self.tracker.get_velocity(), 
            np.array([0.0, 0.0])
        )
    
    def test_prediction(self):
        """Test state prediction."""
        dt = 1.0
        initial_pos = self.tracker.get_position().copy()
        
        # Predict next state
        predicted_pos = self.tracker.predict(dt)
        
        # With zero initial velocity, position should remain the same
        np.testing.assert_array_almost_equal(predicted_pos, initial_pos)
        
        # Age should increase
        self.assertEqual(self.tracker.age, 1)
        self.assertEqual(self.tracker.time_since_update, 1)
    
    def test_update(self):
        """Test measurement update."""
        new_position = (15.0, 25.0)
        new_confidence = 0.9
        
        # Update with new measurement
        self.tracker.update(new_position, new_confidence)
        
        # Position should be updated (not exactly the measurement due to Kalman filtering)
        updated_pos = self.tracker.get_position()
        self.assertTrue(np.allclose(updated_pos, new_position, atol=5.0))
        
        # Confidence should be updated
        self.assertGreater(self.tracker.confidence, self.confidence)
        
        # Time since update should be reset
        self.assertEqual(self.tracker.time_since_update, 0)
        self.assertEqual(self.tracker.hits, 2)
    
    def test_trajectory_prediction(self):
        """Test trajectory prediction."""
        # Set some velocity by updating with displaced measurements
        positions = [(10.0, 20.0), (12.0, 22.0), (14.0, 24.0)]
        
        for pos in positions:
            self.tracker.update(pos)
            self.tracker.predict(1.0)
        
        # Predict trajectory
        time_horizon = 2.0
        dt = 0.5
        trajectory = self.tracker.predict_trajectory(time_horizon, dt)
        
        # Should have 4 points (2.0 / 0.5 = 4)
        self.assertEqual(len(trajectory), 4)
        
        # Trajectory points should be tuples of (x, y)
        for point in trajectory:
            self.assertEqual(len(point), 2)
            self.assertIsInstance(point[0], (int, float))
            self.assertIsInstance(point[1], (int, float))
    
    def test_validity_checks(self):
        """Test tracker validity and deletion criteria."""
        # New tracker should be valid
        self.assertTrue(self.tracker.is_valid(max_age=30, min_hits=1))
        
        # Age the tracker without updates
        for _ in range(35):
            self.tracker.predict(1.0)
        
        # Should not be valid due to age
        self.assertFalse(self.tracker.is_valid(max_age=30, min_hits=1))
        
        # Should be marked for deletion
        self.assertTrue(self.tracker.should_delete(max_age=30))
    
    def test_confidence_update(self):
        """Test confidence score updates."""
        initial_confidence = self.tracker.confidence
        
        # Update with higher confidence
        self.tracker.update((11.0, 21.0), 0.95)
        self.assertGreater(self.tracker.confidence, initial_confidence)
        
        # Update with lower confidence
        low_confidence = 0.3
        self.tracker.update((12.0, 22.0), low_confidence)
        
        # Confidence should be between the previous and new values
        self.assertGreater(self.tracker.confidence, low_confidence)
        self.assertLess(self.tracker.confidence, 0.95)
    
    def test_motion_model(self):
        """Test constant velocity motion model."""
        # Create a sequence of measurements showing constant motion
        dt = 1.0
        positions = [(0.0, 0.0), (2.0, 1.0), (4.0, 2.0), (6.0, 3.0)]
        
        tracker = KalmanTracker(positions[0], 1, 0, 0.8)
        
        for i, pos in enumerate(positions[1:], 1):
            tracker.predict(dt)
            tracker.update(pos)
        
        # After learning the motion, velocity should be approximately (2, 1)
        velocity = tracker.get_velocity()
        np.testing.assert_array_almost_equal(velocity, [2.0, 1.0], decimal=0)
        
        # Predict next position
        predicted_pos = tracker.predict(dt)
        expected_pos = np.array([8.0, 4.0])  # Next position in sequence
        
        # Should be close to expected position
        np.testing.assert_array_almost_equal(predicted_pos, expected_pos, decimal=0)


if __name__ == '__main__':
    unittest.main()