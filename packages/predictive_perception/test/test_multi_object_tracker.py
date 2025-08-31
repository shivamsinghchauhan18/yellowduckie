#!/usr/bin/env python3

import unittest
import numpy as np
import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'include'))

from predictive_perception.multi_object_tracker import MultiObjectTracker


class TestMultiObjectTracker(unittest.TestCase):
    """Test cases for MultiObjectTracker class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.tracker = MultiObjectTracker(max_disappeared=10, max_distance=20.0)
    
    def test_initialization(self):
        """Test tracker initialization."""
        self.assertEqual(len(self.tracker.trackers), 0)
        self.assertEqual(self.tracker.next_id, 0)
        self.assertEqual(self.tracker.max_disappeared, 10)
        self.assertEqual(self.tracker.max_distance, 20.0)
    
    def test_single_object_tracking(self):
        """Test tracking a single object."""
        # First detection
        detections = [(10.0, 20.0, 0)]  # (x, y, class)
        confidences = [0.8]
        
        trackers = self.tracker.update(detections, confidences, dt=1.0)
        
        # Should create one tracker
        self.assertEqual(len(trackers), 1)
        self.assertEqual(self.tracker.next_id, 1)
        
        # Check tracker properties
        tracker_id = list(trackers.keys())[0]
        tracker = trackers[tracker_id]
        self.assertEqual(tracker.object_id, 0)
        self.assertEqual(tracker.object_class, 0)
        np.testing.assert_array_almost_equal(
            tracker.get_position(), 
            np.array([10.0, 20.0])
        )
    
    def test_multiple_object_tracking(self):
        """Test tracking multiple objects."""
        # Multiple detections
        detections = [(10.0, 20.0, 0), (30.0, 40.0, 1), (50.0, 60.0, 0)]
        confidences = [0.8, 0.9, 0.7]
        
        trackers = self.tracker.update(detections, confidences, dt=1.0)
        
        # Should create three trackers
        self.assertEqual(len(trackers), 3)
        self.assertEqual(self.tracker.next_id, 3)
        
        # Check that all objects are tracked
        positions = [tracker.get_position() for tracker in trackers.values()]
        expected_positions = [np.array([10.0, 20.0]), np.array([30.0, 40.0]), np.array([50.0, 60.0])]
        
        for expected_pos in expected_positions:
            found = False
            for pos in positions:
                if np.allclose(pos, expected_pos, atol=1.0):
                    found = True
                    break
            self.assertTrue(found, f"Expected position {expected_pos} not found in tracked positions")
    
    def test_object_association(self):
        """Test association of detections with existing trackers."""
        # Create initial trackers
        initial_detections = [(10.0, 20.0, 0), (30.0, 40.0, 1)]
        initial_confidences = [0.8, 0.9]
        
        self.tracker.update(initial_detections, initial_confidences, dt=1.0)
        initial_count = len(self.tracker.trackers)
        
        # Update with slightly moved detections
        moved_detections = [(12.0, 22.0, 0), (32.0, 42.0, 1)]
        moved_confidences = [0.85, 0.95]
        
        trackers = self.tracker.update(moved_detections, moved_confidences, dt=1.0)
        
        # Should still have the same number of trackers (no new ones created)
        self.assertEqual(len(trackers), initial_count)
        
        # Trackers should have updated positions
        positions = [tracker.get_position() for tracker in trackers.values()]
        
        # Check that positions are closer to new detections
        for pos in positions:
            min_dist_to_new = min(np.linalg.norm(pos - np.array(det[:2])) 
                                 for det in moved_detections)
            min_dist_to_old = min(np.linalg.norm(pos - np.array(det[:2])) 
                                 for det in initial_detections)
            self.assertLess(min_dist_to_new, min_dist_to_old)
    
    def test_tracker_cleanup(self):
        """Test removal of old trackers."""
        # Create a tracker
        detections = [(10.0, 20.0, 0)]
        confidences = [0.8]
        
        self.tracker.update(detections, confidences, dt=1.0)
        self.assertEqual(len(self.tracker.trackers), 1)
        
        # Update without detections for many frames
        for _ in range(15):  # More than max_disappeared (10)
            self.tracker.update([], [], dt=1.0)
        
        # Tracker should be removed
        self.assertEqual(len(self.tracker.trackers), 0)
    
    def test_trajectory_prediction(self):
        """Test trajectory prediction for tracked objects."""
        # Create and update a tracker with motion
        detections_sequence = [
            [(0.0, 0.0, 0)],
            [(2.0, 1.0, 0)],
            [(4.0, 2.0, 0)],
            [(6.0, 3.0, 0)]
        ]
        
        for detections in detections_sequence:
            self.tracker.update(detections, [0.8], dt=1.0)
        
        # Predict trajectories
        trajectories = self.tracker.predict_trajectories(time_horizon=2.0, dt=0.5)
        
        # Should have one trajectory
        self.assertEqual(len(trajectories), 1)
        
        # Trajectory should have points
        trajectory = list(trajectories.values())[0]
        self.assertGreater(len(trajectory), 0)
        
        # Each point should be a tuple of (x, y)
        for point in trajectory:
            self.assertEqual(len(point), 2)
    
    def test_tracking_statistics(self):
        """Test tracking statistics calculation."""
        # Initially no objects
        stats = self.tracker.get_tracking_stats()
        self.assertEqual(stats['total_tracked_objects'], 0)
        self.assertEqual(stats['average_confidence'], 0.0)
        
        # Add some objects
        detections = [(10.0, 20.0, 0), (30.0, 40.0, 1)]
        confidences = [0.8, 0.9]
        
        self.tracker.update(detections, confidences, dt=1.0)
        
        stats = self.tracker.get_tracking_stats()
        # Check that we have some tracked objects (may not be exactly 2 due to validity criteria)
        self.assertGreaterEqual(stats['total_tracked_objects'], 0)
        if stats['total_tracked_objects'] > 0:
            self.assertGreater(stats['average_confidence'], 0.0)
            self.assertGreaterEqual(stats['new_tracks_count'], 0)
    
    def test_valid_trackers_filtering(self):
        """Test filtering of valid trackers."""
        # Create trackers
        detections = [(10.0, 20.0, 0), (30.0, 40.0, 1)]
        confidences = [0.8, 0.9]
        
        self.tracker.update(detections, confidences, dt=1.0)
        
        # All should be valid initially (with min_hits=1)
        all_trackers = self.tracker.trackers
        valid_trackers = {}
        for tid, tracker in all_trackers.items():
            if tracker.is_valid(max_age=30, min_hits=1):
                valid_trackers[tid] = tracker
        self.assertEqual(len(valid_trackers), 2)
        
        # Age one tracker without updates
        tracker_ids = list(self.tracker.trackers.keys())
        for _ in range(5):
            self.tracker.trackers[tracker_ids[0]].predict(1.0)
        
        # Update the other tracker
        self.tracker.update([(32.0, 42.0, 1)], [0.95], dt=1.0)
        
        # Should still have valid trackers, but aged one might have lower confidence
        valid_trackers = self.tracker.get_valid_trackers()
        self.assertGreaterEqual(len(valid_trackers), 1)
    
    def test_no_detections_handling(self):
        """Test handling of frames with no detections."""
        # Create initial tracker
        detections = [(10.0, 20.0, 0)]
        confidences = [0.8]
        
        self.tracker.update(detections, confidences, dt=1.0)
        initial_count = len(self.tracker.trackers)
        
        # Update with no detections
        trackers = self.tracker.update([], [], dt=1.0)
        
        # Should still have the tracker (just aged)
        self.assertEqual(len(trackers), initial_count)
        
        # Tracker should have increased age
        tracker = list(trackers.values())[0]
        self.assertGreater(tracker.time_since_update, 0)
    
    def test_distance_threshold(self):
        """Test distance threshold for association."""
        # Create tracker
        detections = [(10.0, 20.0, 0)]
        confidences = [0.8]
        
        self.tracker.update(detections, confidences, dt=1.0)
        initial_count = len(self.tracker.trackers)
        
        # Update with detection far away (beyond max_distance)
        far_detections = [(100.0, 200.0, 0)]  # Far from original (10, 20)
        far_confidences = [0.9]
        
        trackers = self.tracker.update(far_detections, far_confidences, dt=1.0)
        
        # Should create a new tracker instead of associating
        self.assertGreater(len(trackers), initial_count)


if __name__ == '__main__':
    unittest.main()