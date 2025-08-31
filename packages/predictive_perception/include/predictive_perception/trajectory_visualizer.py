#!/usr/bin/env python3

import numpy as np
import cv2
from typing import List, Tuple, Optional


class TrajectoryVisualizer:
    """
    Visualization tools for trajectory prediction debugging and validation.
    """
    
    def __init__(self, image_width=640, image_height=480, scale=1.0):
        """
        Initialize trajectory visualizer.
        
        Args:
            image_width: Width of visualization image
            image_height: Height of visualization image
            scale: Scale factor for coordinate conversion
        """
        self.image_width = image_width
        self.image_height = image_height
        self.scale = scale
        
        # Colors for different elements (BGR format)
        self.colors = {
            'current_position': (0, 0, 255),      # Red
            'predicted_trajectory': (0, 255, 0),   # Green
            'confidence_high': (0, 255, 0),        # Green
            'confidence_medium': (0, 255, 255),    # Yellow
            'confidence_low': (0, 0, 255),         # Red
            'velocity_vector': (255, 0, 0),        # Blue
            'uncertainty_ellipse': (128, 128, 128), # Gray
            'grid': (64, 64, 64),                  # Dark gray
            'text': (255, 255, 255),               # White
            'background': (0, 0, 0)                # Black
        }
    
    def create_visualization(self, tracked_objects, trajectories, confidences=None):
        """
        Create comprehensive trajectory visualization.
        
        Args:
            tracked_objects: Dictionary of tracked objects
            trajectories: Dictionary of predicted trajectories
            confidences: Optional dictionary of confidence scores
            
        Returns:
            Visualization image as numpy array
        """
        # Create blank image
        image = np.full((self.image_height, self.image_width, 3), 
                       self.colors['background'], dtype=np.uint8)
        
        # Draw grid
        self._draw_grid(image)
        
        # Draw trajectories and objects
        for obj_id, trajectory in trajectories.items():
            if obj_id in tracked_objects:
                obj = tracked_objects[obj_id]
                
                # Get confidence scores if available
                obj_confidences = confidences.get(obj_id, None) if confidences else None
                
                # Draw trajectory
                self._draw_trajectory(image, trajectory, obj_confidences)
                
                # Draw current position and velocity
                self._draw_object_state(image, obj)
                
                # Draw object info
                self._draw_object_info(image, obj, obj_id)
        
        return image
    
    def _draw_grid(self, image):
        """Draw coordinate grid on image."""
        grid_spacing = 50
        
        # Vertical lines
        for x in range(0, self.image_width, grid_spacing):
            cv2.line(image, (x, 0), (x, self.image_height), 
                    self.colors['grid'], 1)
        
        # Horizontal lines
        for y in range(0, self.image_height, grid_spacing):
            cv2.line(image, (0, y), (self.image_width, y), 
                    self.colors['grid'], 1)
        
        # Draw axes
        center_x = self.image_width // 2
        center_y = self.image_height // 2
        
        # X-axis
        cv2.line(image, (0, center_y), (self.image_width, center_y), 
                self.colors['text'], 2)
        # Y-axis
        cv2.line(image, (center_x, 0), (center_x, self.image_height), 
                self.colors['text'], 2)
    
    def _world_to_image(self, world_x, world_y):
        """Convert world coordinates to image coordinates."""
        # Center the coordinate system and apply scale
        img_x = int(self.image_width // 2 + world_x * self.scale)
        img_y = int(self.image_height // 2 - world_y * self.scale)  # Flip Y
        
        # Clamp to image bounds
        img_x = max(0, min(self.image_width - 1, img_x))
        img_y = max(0, min(self.image_height - 1, img_y))
        
        return img_x, img_y
    
    def _draw_trajectory(self, image, trajectory, confidences=None):
        """Draw predicted trajectory with confidence visualization."""
        if len(trajectory) < 2:
            return
        
        # Convert trajectory points to image coordinates
        points = []
        for state in trajectory:
            img_x, img_y = self._world_to_image(state[0], state[1])
            points.append((img_x, img_y))
        
        # Draw trajectory line with varying thickness based on confidence
        for i in range(len(points) - 1):
            start_point = points[i]
            end_point = points[i + 1]
            
            # Determine color and thickness based on confidence
            if confidences and i < len(confidences):
                confidence = confidences[i]
                color = self._get_confidence_color(confidence)
                thickness = max(1, int(confidence * 3))
            else:
                color = self.colors['predicted_trajectory']
                thickness = 2
            
            cv2.line(image, start_point, end_point, color, thickness)
        
        # Draw trajectory points
        for i, point in enumerate(points):
            if confidences and i < len(confidences):
                confidence = confidences[i]
                color = self._get_confidence_color(confidence)
                radius = max(2, int(confidence * 4))
            else:
                color = self.colors['predicted_trajectory']
                radius = 3
            
            cv2.circle(image, point, radius, color, -1)
    
    def _draw_object_state(self, image, obj):
        """Draw current object position and velocity vector."""
        # Get object position and velocity
        pos = obj.get_position()
        vel = obj.get_velocity()
        
        # Convert to image coordinates
        img_x, img_y = self._world_to_image(pos[0], pos[1])
        
        # Draw current position
        cv2.circle(image, (img_x, img_y), 8, self.colors['current_position'], -1)
        cv2.circle(image, (img_x, img_y), 10, self.colors['text'], 2)
        
        # Draw velocity vector
        vel_magnitude = np.sqrt(vel[0]**2 + vel[1]**2)
        if vel_magnitude > 0.1:  # Only draw if significant velocity
            # Scale velocity for visualization
            vel_scale = 20.0
            vel_end_x = pos[0] + vel[0] * vel_scale
            vel_end_y = pos[1] + vel[1] * vel_scale
            
            vel_end_img_x, vel_end_img_y = self._world_to_image(vel_end_x, vel_end_y)
            
            # Draw arrow
            cv2.arrowedLine(image, (img_x, img_y), (vel_end_img_x, vel_end_img_y),
                           self.colors['velocity_vector'], 3, tipLength=0.3)
    
    def _draw_object_info(self, image, obj, obj_id):
        """Draw object information text."""
        pos = obj.get_position()
        vel = obj.get_velocity()
        
        img_x, img_y = self._world_to_image(pos[0], pos[1])
        
        # Prepare info text
        class_names = {0: "Duck", 1: "Duckiebot", 2: "Other"}
        class_name = class_names.get(obj.object_class, "Unknown")
        
        info_lines = [
            f"ID: {obj_id}",
            f"Class: {class_name}",
            f"Conf: {obj.confidence:.2f}",
            f"Pos: ({pos[0]:.1f}, {pos[1]:.1f})",
            f"Vel: ({vel[0]:.1f}, {vel[1]:.1f})"
        ]
        
        # Draw text background
        text_x = img_x + 15
        text_y = img_y - 10
        
        for i, line in enumerate(info_lines):
            y_offset = text_y + i * 15
            
            # Ensure text stays within image bounds
            if y_offset > 0 and y_offset < self.image_height and text_x < self.image_width - 100:
                cv2.putText(image, line, (text_x, y_offset),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, self.colors['text'], 1)
    
    def _get_confidence_color(self, confidence):
        """Get color based on confidence level."""
        if confidence > 0.7:
            return self.colors['confidence_high']
        elif confidence > 0.4:
            return self.colors['confidence_medium']
        else:
            return self.colors['confidence_low']
    
    def draw_uncertainty_ellipse(self, image, position, covariance, confidence=0.95):
        """
        Draw uncertainty ellipse for position estimate.
        
        Args:
            image: Image to draw on
            position: (x, y) position
            covariance: 2x2 covariance matrix for position
            confidence: Confidence level for ellipse (0.95 = 95%)
        """
        if covariance.shape != (2, 2):
            return
        
        # Convert position to image coordinates
        img_x, img_y = self._world_to_image(position[0], position[1])
        
        # Calculate eigenvalues and eigenvectors
        eigenvals, eigenvecs = np.linalg.eigh(covariance)
        
        # Calculate ellipse parameters
        # Chi-squared value for desired confidence level
        chi2_val = 5.991 if confidence == 0.95 else 2.296  # 95% or 68%
        
        # Semi-axes lengths
        a = np.sqrt(chi2_val * eigenvals[1]) * self.scale
        b = np.sqrt(chi2_val * eigenvals[0]) * self.scale
        
        # Rotation angle
        angle = np.degrees(np.arctan2(eigenvecs[1, 1], eigenvecs[0, 1]))
        
        # Draw ellipse
        cv2.ellipse(image, (img_x, img_y), (int(a), int(b)), angle, 
                   0, 360, self.colors['uncertainty_ellipse'], 2)
    
    def create_trajectory_comparison(self, predicted_trajectories, actual_trajectory=None):
        """
        Create visualization comparing multiple trajectory predictions.
        
        Args:
            predicted_trajectories: List of (trajectory, confidence, label) tuples
            actual_trajectory: Optional actual trajectory for comparison
            
        Returns:
            Comparison visualization image
        """
        image = np.full((self.image_height, self.image_width, 3), 
                       self.colors['background'], dtype=np.uint8)
        
        self._draw_grid(image)
        
        # Colors for different predictions
        prediction_colors = [
            (0, 255, 0),    # Green
            (0, 255, 255),  # Yellow
            (255, 0, 255),  # Magenta
            (255, 165, 0),  # Orange
            (0, 165, 255)   # Orange-red
        ]
        
        # Draw predicted trajectories
        for i, (trajectory, confidences, label) in enumerate(predicted_trajectories):
            color = prediction_colors[i % len(prediction_colors)]
            
            # Convert trajectory to image coordinates
            points = []
            for state in trajectory:
                img_x, img_y = self._world_to_image(state[0], state[1])
                points.append((img_x, img_y))
            
            # Draw trajectory
            for j in range(len(points) - 1):
                thickness = max(1, int(confidences[j] * 3)) if confidences else 2
                cv2.line(image, points[j], points[j + 1], color, thickness)
            
            # Draw start point
            if points:
                cv2.circle(image, points[0], 6, color, -1)
                
                # Add label
                cv2.putText(image, label, (points[0][0] + 10, points[0][1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Draw actual trajectory if provided
        if actual_trajectory:
            points = []
            for state in actual_trajectory:
                img_x, img_y = self._world_to_image(state[0], state[1])
                points.append((img_x, img_y))
            
            # Draw actual trajectory in white with dashed line effect
            for i in range(len(points) - 1):
                if i % 2 == 0:  # Dashed effect
                    cv2.line(image, points[i], points[i + 1], 
                            self.colors['text'], 3)
        
        return image
    
    def save_visualization(self, image, filename):
        """Save visualization image to file."""
        cv2.imwrite(filename, image)
    
    def create_animation_frames(self, trajectory_history, output_dir="trajectory_animation"):
        """
        Create animation frames showing trajectory evolution over time.
        
        Args:
            trajectory_history: List of (tracked_objects, trajectories, confidences) tuples
            output_dir: Directory to save animation frames
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        for frame_idx, (objects, trajectories, confidences) in enumerate(trajectory_history):
            image = self.create_visualization(objects, trajectories, confidences)
            
            # Add frame number
            cv2.putText(image, f"Frame: {frame_idx}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, self.colors['text'], 2)
            
            # Save frame
            filename = os.path.join(output_dir, f"frame_{frame_idx:04d}.png")
            self.save_visualization(image, filename)