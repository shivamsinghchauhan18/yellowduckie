#!/usr/bin/env python3

import unittest
import numpy as np
import cv2
import sys
import os

# Add the package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'include'))

from predictive_perception.scene_analyzer import (
    SceneAnalyzer, 
    SceneClassifier, 
    ScenarioType, 
    EnvironmentalCondition
)


class MockTrackedObject:
    """Mock tracked object for testing."""
    
    def __init__(self, position, velocity=(0, 0), object_class=0):
        self.position = np.array(position)
        self.velocity = np.array(velocity)
        self.object_class = object_class
    
    def get_position(self):
        return self.position
    
    def get_velocity(self):
        return self.velocity


class TestSceneAnalyzer(unittest.TestCase):
    """Test cases for SceneAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = SceneAnalyzer()
        
        # Create test images
        self.bright_image = np.full((480, 640, 3), 200, dtype=np.uint8)
        self.dark_image = np.full((480, 640, 3), 30, dtype=np.uint8)
        self.normal_image = np.full((480, 640, 3), 128, dtype=np.uint8)
        
        # Add some texture to normal image
        noise = np.random.randint(-20, 20, (480, 640, 3))
        self.normal_image = np.clip(self.normal_image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    def test_lighting_assessment(self):
        """Test lighting condition assessment."""
        # Test bright image
        bright_condition = self.analyzer._assess_lighting(self.bright_image)
        self.assertEqual(bright_condition, EnvironmentalCondition.GOOD_VISIBILITY)
        
        # Test dark image
        dark_condition = self.analyzer._assess_lighting(self.dark_image)
        self.assertEqual(dark_condition, EnvironmentalCondition.LOW_LIGHT)
    
    def test_traffic_density_analysis(self):
        """Test traffic density analysis."""
        # Test with no objects
        empty_objects = {}
        density = self.analyzer._analyze_traffic_density(empty_objects)
        
        self.assertEqual(density['total_objects'], 0)
        self.assertEqual(density['density_level'], 'none')
        
        # Test with few objects
        few_objects = {
            1: MockTrackedObject((10, 20)),
            2: MockTrackedObject((30, 40))
        }
        density = self.analyzer._analyze_traffic_density(few_objects)
        
        self.assertEqual(density['total_objects'], 2)
        self.assertEqual(density['density_level'], 'low')
        
        # Test with many objects
        many_objects = {}
        for i in range(6):
            many_objects[i] = MockTrackedObject((i * 10, i * 15))
        
        density = self.analyzer._analyze_traffic_density(many_objects)
        
        self.assertEqual(density['total_objects'], 6)
        self.assertEqual(density['density_level'], 'high')
    
    def test_visibility_assessment(self):
        """Test visibility score calculation."""
        # Test with normal image (should have reasonable visibility)
        visibility = self.analyzer._assess_visibility(self.normal_image)
        self.assertGreater(visibility, 0.0)
        self.assertLessEqual(visibility, 1.0)
        
        # Test with uniform image (should have low visibility due to lack of features)
        uniform_image = np.full((480, 640, 3), 128, dtype=np.uint8)
        uniform_visibility = self.analyzer._assess_visibility(uniform_image)
        
        # Normal image should have better visibility than uniform image
        self.assertGreater(visibility, uniform_visibility)
    
    def test_scenario_classification(self):
        """Test traffic scenario classification."""
        # Test lane following scenario (no objects, no tags)
        scenario = self.analyzer._classify_traffic_scenario({}, None, [])
        self.assertEqual(scenario, ScenarioType.LANE_FOLLOWING)
        
        # Test obstacle avoidance scenario (objects in path)
        objects_in_path = {
            1: MockTrackedObject((10, 50)),  # Close to center, ahead
            2: MockTrackedObject((200, 300))  # Far away
        }
        scenario = self.analyzer._classify_traffic_scenario(objects_in_path, None, [])
        self.assertEqual(scenario, ScenarioType.OBSTACLE_AVOIDANCE)
    
    def test_environmental_conditions_assessment(self):
        """Test environmental conditions assessment."""
        # Test with normal image and few objects
        objects = {1: MockTrackedObject((10, 20))}
        conditions = self.analyzer._assess_environmental_conditions(self.normal_image, objects)
        
        # Should return a list of conditions
        self.assertIsInstance(conditions, list)
        self.assertGreater(len(conditions), 0)
        
        # All conditions should be EnvironmentalCondition enum values
        for condition in conditions:
            self.assertIsInstance(condition, EnvironmentalCondition)
    
    def test_scene_analysis_integration(self):
        """Test complete scene analysis integration."""
        # Create test scenario
        tracked_objects = {
            1: MockTrackedObject((25, 30)),
            2: MockTrackedObject((100, 150))
        }
        
        # Perform analysis
        analysis = self.analyzer.analyze_scene(
            self.normal_image, 
            tracked_objects, 
            lane_pose=None, 
            apriltags=[]
        )
        
        # Check analysis structure
        required_keys = [
            'scenario_type', 'environmental_conditions', 'traffic_density',
            'visibility_score', 'object_count', 'confidence', 'timestamp'
        ]
        
        for key in required_keys:
            self.assertIn(key, analysis)
        
        # Check data types
        self.assertIsInstance(analysis['scenario_type'], ScenarioType)
        self.assertIsInstance(analysis['environmental_conditions'], list)
        self.assertIsInstance(analysis['traffic_density'], dict)
        self.assertIsInstance(analysis['visibility_score'], (int, float))
        self.assertIsInstance(analysis['object_count'], int)
        self.assertIsInstance(analysis['confidence'], (int, float))
        
        # Check value ranges
        self.assertGreaterEqual(analysis['visibility_score'], 0.0)
        self.assertLessEqual(analysis['visibility_score'], 1.0)
        self.assertGreaterEqual(analysis['confidence'], 0.0)
        self.assertLessEqual(analysis['confidence'], 1.0)
        self.assertEqual(analysis['object_count'], len(tracked_objects))
    
    def test_history_tracking(self):
        """Test analysis history tracking."""
        # Perform multiple analyses
        for i in range(5):
            objects = {j: MockTrackedObject((j * 10, j * 20)) for j in range(i + 1)}
            self.analyzer.analyze_scene(self.normal_image, objects, None, [])
        
        # Check history length
        self.assertEqual(len(self.analyzer.scenario_history), 5)
        self.assertEqual(len(self.analyzer.condition_history), 5)
        
        # Get scene summary
        summary = self.analyzer.get_scene_summary()
        
        self.assertIn('dominant_scenario', summary)
        self.assertIn('dominant_conditions', summary)
        self.assertIn('scenario_stability', summary)
        self.assertIn('analysis_frames', summary)
    
    def test_confidence_calculation(self):
        """Test confidence calculation with temporal consistency."""
        # Perform consistent analyses (same scenario)
        for _ in range(5):
            objects = {1: MockTrackedObject((25, 30))}  # Same object position
            self.analyzer.analyze_scene(self.normal_image, objects, None, [])
        
        confidence = self.analyzer._calculate_analysis_confidence()
        
        # Should have high confidence due to consistency
        self.assertGreater(confidence, 0.5)
        
        # Now add inconsistent analysis
        objects = {}  # No objects - different scenario
        self.analyzer.analyze_scene(self.normal_image, objects, None, [])
        
        new_confidence = self.analyzer._calculate_analysis_confidence()
        
        # Confidence should decrease due to inconsistency
        self.assertLess(new_confidence, confidence)


class TestSceneClassifier(unittest.TestCase):
    """Test cases for SceneClassifier class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.classifier = SceneClassifier()
        
        # Create test images with different characteristics
        self.simple_image = np.full((480, 640, 3), 128, dtype=np.uint8)
        
        # Complex image with many edges
        self.complex_image = np.zeros((480, 640, 3), dtype=np.uint8)
        for i in range(0, 640, 20):
            cv2.line(self.complex_image, (i, 0), (i, 480), (255, 255, 255), 2)
        for i in range(0, 480, 20):
            cv2.line(self.complex_image, (0, i), (640, i), (255, 255, 255), 2)
    
    def test_feature_extraction(self):
        """Test feature extraction methods."""
        # Test histogram features
        hist_features = self.classifier._extract_histogram_features(self.simple_image)
        self.assertEqual(len(hist_features), 150)  # 50 bins * 3 channels
        
        # Test edge features
        edge_features = self.classifier._extract_edge_features(self.simple_image)
        self.assertEqual(len(edge_features), 9)  # 1 density + 8 orientation bins
        
        # Test texture features
        texture_features = self.classifier._extract_texture_features(self.simple_image)
        self.assertEqual(len(texture_features), 2)  # Mean and std
    
    def test_scene_classification(self):
        """Test scene type classification."""
        # Test with simple image
        scene_type, confidence = self.classifier.classify_scene_type(self.simple_image)
        
        self.assertIsInstance(scene_type, ScenarioType)
        self.assertIsInstance(confidence, (int, float))
        self.assertGreaterEqual(confidence, 0.0)
        self.assertLessEqual(confidence, 1.0)
        
        # Test with complex image
        complex_scene_type, complex_confidence = self.classifier.classify_scene_type(self.complex_image)
        
        # Complex image should be classified differently than simple image
        # (though exact classification depends on the rule-based logic)
        self.assertIsInstance(complex_scene_type, ScenarioType)
    
    def test_feature_consistency(self):
        """Test feature extraction consistency."""
        # Extract features multiple times from same image
        features1 = self.classifier._extract_features(self.simple_image)
        features2 = self.classifier._extract_features(self.simple_image)
        
        # Features should be identical
        for key in features1:
            np.testing.assert_array_almost_equal(features1[key], features2[key])
    
    def test_edge_density_calculation(self):
        """Test edge density calculation for different image types."""
        # Simple image should have low edge density
        simple_features = self.classifier._extract_edge_features(self.simple_image)
        simple_edge_density = simple_features[0]
        
        # Complex image should have high edge density
        complex_features = self.classifier._extract_edge_features(self.complex_image)
        complex_edge_density = complex_features[0]
        
        # Complex image should have higher edge density
        self.assertGreater(complex_edge_density, simple_edge_density)


if __name__ == '__main__':
    unittest.main()