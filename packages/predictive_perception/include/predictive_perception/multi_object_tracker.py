#!/usr/bin/env python3

import numpy as np
from .kalman_tracker import KalmanTracker


def linear_sum_assignment(cost_matrix):
    """
    Simple Hungarian algorithm implementation for assignment problem.
    This is a simplified version that works for small matrices.
    """
    cost_matrix = np.array(cost_matrix)
    n_rows, n_cols = cost_matrix.shape
    
    if n_rows == 0 or n_cols == 0:
        return np.array([]), np.array([])
    
    # For small matrices, use brute force approach
    if n_rows <= 3 and n_cols <= 3:
        min_cost = float('inf')
        best_assignment = None
        
        # Generate all possible assignments
        import itertools
        for perm in itertools.permutations(range(n_cols), min(n_rows, n_cols)):
            cost = sum(cost_matrix[i, perm[i]] for i in range(len(perm)))
            if cost < min_cost:
                min_cost = cost
                best_assignment = perm
        
        if best_assignment:
            row_indices = list(range(len(best_assignment)))
            col_indices = list(best_assignment)
            return np.array(row_indices), np.array(col_indices)
    
    # Fallback: greedy assignment for larger matrices
    row_indices = []
    col_indices = []
    used_cols = set()
    
    for i in range(n_rows):
        best_col = None
        best_cost = float('inf')
        
        for j in range(n_cols):
            if j not in used_cols and cost_matrix[i, j] < best_cost:
                best_cost = cost_matrix[i, j]
                best_col = j
        
        if best_col is not None:
            row_indices.append(i)
            col_indices.append(best_col)
            used_cols.add(best_col)
    
    return np.array(row_indices), np.array(col_indices)


class MultiObjectTracker:
    """
    Multi-object tracker using Kalman filters and Hungarian algorithm for data association.
    """
    
    def __init__(self, max_disappeared=30, max_distance=50.0):
        """
        Initialize multi-object tracker.
        
        Args:
            max_disappeared: Maximum frames an object can be missing before deletion
            max_distance: Maximum distance for associating detections with tracks
        """
        self.trackers = {}
        self.next_id = 0
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        
    def update(self, detections, confidences=None, dt=1.0):
        """
        Update tracker with new detections.
        
        Args:
            detections: List of (x, y, class) tuples
            confidences: List of confidence scores for each detection
            dt: Time step since last update
            
        Returns:
            Dictionary of active trackers
        """
        if confidences is None:
            confidences = [1.0] * len(detections)
        
        # Predict all existing trackers
        for tracker in self.trackers.values():
            tracker.predict(dt)
        
        if len(detections) == 0:
            # No detections, just age existing trackers
            self._cleanup_trackers()
            return self.trackers
        
        if len(self.trackers) == 0:
            # No existing trackers, create new ones for all detections
            for i, (detection, confidence) in enumerate(zip(detections, confidences)):
                self._create_tracker(detection, confidence)
            return self.trackers
        
        # Associate detections with existing trackers
        self._associate_detections_to_trackers(detections, confidences)
        
        # Clean up old trackers
        self._cleanup_trackers()
        
        return self.trackers
    
    def _associate_detections_to_trackers(self, detections, confidences):
        """
        Associate detections with existing trackers using Hungarian algorithm.
        """
        # Get predicted positions from all trackers
        tracker_ids = list(self.trackers.keys())
        tracker_positions = [self.trackers[tid].get_position() for tid in tracker_ids]
        
        # Compute cost matrix (distances between detections and predictions)
        cost_matrix = np.zeros((len(detections), len(tracker_positions)))
        
        for i, detection in enumerate(detections):
            det_pos = np.array(detection[:2])  # (x, y)
            for j, track_pos in enumerate(tracker_positions):
                distance = np.linalg.norm(det_pos - track_pos)
                cost_matrix[i, j] = distance
        
        # Solve assignment problem
        if cost_matrix.size > 0:
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Process assignments
            unmatched_detections = set(range(len(detections)))
            unmatched_trackers = set(range(len(tracker_positions)))
            
            for row, col in zip(row_indices, col_indices):
                if cost_matrix[row, col] <= self.max_distance:
                    # Valid assignment
                    tracker_id = tracker_ids[col]
                    detection = detections[row]
                    confidence = confidences[row]
                    
                    # Update tracker with detection
                    self.trackers[tracker_id].update(detection[:2], confidence)
                    
                    # Update object class if needed
                    if len(detection) > 2:
                        self.trackers[tracker_id].object_class = detection[2]
                    
                    unmatched_detections.discard(row)
                    unmatched_trackers.discard(col)
            
            # Create new trackers for unmatched detections
            for det_idx in unmatched_detections:
                detection = detections[det_idx]
                confidence = confidences[det_idx]
                self._create_tracker(detection, confidence)
    
    def _create_tracker(self, detection, confidence):
        """
        Create a new tracker for a detection.
        
        Args:
            detection: (x, y, class) tuple
            confidence: Confidence score
        """
        position = detection[:2]
        object_class = detection[2] if len(detection) > 2 else 2  # Default to 'other'
        
        tracker = KalmanTracker(
            initial_position=position,
            object_id=self.next_id,
            object_class=object_class,
            confidence=confidence
        )
        
        self.trackers[self.next_id] = tracker
        self.next_id += 1
    
    def _cleanup_trackers(self):
        """Remove old or invalid trackers."""
        to_delete = []
        
        for tracker_id, tracker in self.trackers.items():
            if tracker.should_delete(self.max_disappeared):
                to_delete.append(tracker_id)
        
        for tracker_id in to_delete:
            del self.trackers[tracker_id]
    
    def get_valid_trackers(self):
        """
        Get all valid trackers.
        
        Returns:
            Dictionary of valid trackers
        """
        valid_trackers = {}
        for tracker_id, tracker in self.trackers.items():
            if tracker.is_valid(max_age=self.max_disappeared, min_hits=1):
                valid_trackers[tracker_id] = tracker
        return valid_trackers
    
    def predict_trajectories(self, time_horizon=3.0, dt=0.1):
        """
        Predict trajectories for all valid trackers.
        
        Args:
            time_horizon: Time to predict ahead (seconds)
            dt: Time step for prediction
            
        Returns:
            Dictionary mapping tracker_id to trajectory points
        """
        trajectories = {}
        valid_trackers = self.get_valid_trackers()
        
        for tracker_id, tracker in valid_trackers.items():
            trajectory = tracker.predict_trajectory(time_horizon, dt)
            trajectories[tracker_id] = trajectory
        
        return trajectories
    
    def get_tracking_stats(self):
        """
        Get tracking statistics.
        
        Returns:
            Dictionary with tracking statistics
        """
        valid_trackers = self.get_valid_trackers()
        
        if not valid_trackers:
            return {
                'total_tracked_objects': 0,
                'average_confidence': 0.0,
                'lost_tracks_count': 0,
                'new_tracks_count': 0
            }
        
        confidences = [tracker.confidence for tracker in valid_trackers.values()]
        avg_confidence = np.mean(confidences)
        
        # Count new tracks (age <= 5 frames)
        new_tracks = sum(1 for tracker in valid_trackers.values() if tracker.age <= 5)
        
        # Count recently lost tracks (time_since_update > 10)
        lost_tracks = sum(1 for tracker in self.trackers.values() 
                         if tracker.time_since_update > 10)
        
        return {
            'total_tracked_objects': len(valid_trackers),
            'average_confidence': avg_confidence,
            'lost_tracks_count': lost_tracks,
            'new_tracks_count': new_tracks
        }