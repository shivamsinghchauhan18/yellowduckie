"""Predictive Perception Package public API."""

# Import main modules to make them available
from .multi_object_tracker import MultiObjectTracker
from .kalman_tracker import KalmanTracker
# Backward-compatible alias: expose SensorFusionEngine as SensorFusion
from .sensor_fusion import SensorFusionEngine as SensorFusion
from .scene_analyzer import SceneAnalyzer
from .motion_models import ConstantVelocityModel, ConstantAccelerationModel
from .trajectory_visualizer import TrajectoryVisualizer

__all__ = [
    'MultiObjectTracker',
    'KalmanTracker',
    'SensorFusion',
    'SceneAnalyzer',
    'ConstantVelocityModel',
    'ConstantAccelerationModel',
    'TrajectoryVisualizer',
]