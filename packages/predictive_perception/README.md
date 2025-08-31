# Enhanced Perception System for Duckietown

This package implements an advanced perception system for the Duckietown Safe Navigation project, providing state-of-the-art object tracking, trajectory prediction, scene understanding, and sensor fusion capabilities.

## Overview

The Enhanced Perception System consists of four main components:

1. **Predictive Perception Manager** - Multi-object tracking with Kalman filters and trajectory prediction
2. **Object Trajectory Prediction** - Advanced motion models and confidence estimation
3. **Scene Understanding Module** - High-level scene interpretation and environmental assessment
4. **Enhanced Object Detection Integration** - Improved detection with filtering and noise reduction
5. **Perception Data Fusion** - Multi-modal sensor fusion combining camera, IMU, and encoder data

## Features

### Predictive Perception Manager
- Multi-object tracking using Kalman filters
- Unique ID assignment and trajectory prediction
- Confidence estimation for tracking accuracy
- Real-time performance optimization

### Object Trajectory Prediction
- Multiple motion models (constant velocity, constant acceleration, IMM)
- Confidence estimation for trajectory predictions
- Multiple hypothesis generation
- Trajectory visualization tools

### Scene Understanding Module
- Traffic scenario classification (intersection, lane following, obstacles)
- Environmental condition assessment (lighting, visibility, traffic density)
- Temporal consistency checking
- Scene context analysis for navigation decisions

### Enhanced Object Detection
- Advanced filtering and noise reduction
- Bounding box tracking and smoothing
- Region of interest (ROI) processing
- Priority-based detection ranking

### Perception Data Fusion
- Multi-modal sensor fusion (camera, IMU, encoders)
- Temporal synchronization and consistency checks
- Outlier detection and filtering
- Adaptive fusion weights based on sensor performance

## Installation

1. Build the package:
```bash
catkin build predictive_perception
```

2. Source the workspace:
```bash
source devel/setup.bash
```

## Usage

### Launch Individual Components

**Predictive Perception Manager:**
```bash
roslaunch predictive_perception predictive_perception_manager_node.launch veh:=duckiebot
```

**Scene Understanding Module:**
```bash
roslaunch predictive_perception scene_understanding_module_node.launch veh:=duckiebot
```

**Perception Data Fusion:**
```bash
roslaunch predictive_perception perception_data_fusion_node.launch veh:=duckiebot
```

**Enhanced Object Detection:**
```bash
roslaunch object_detection enhanced_object_detection_node.launch veh:=duckiebot
```

### Launch Complete System

Launch the entire enhanced perception system:
```bash
roslaunch predictive_perception enhanced_perception_system.launch veh:=duckiebot
```

## Configuration

Configuration files are located in the `config/` directory for each component:

- `predictive_perception_manager_node/default.yaml`
- `scene_understanding_module_node/default.yaml`
- `perception_data_fusion_node/default.yaml`
- `enhanced_object_detection_node/default.yaml` (in object_detection package)

### Key Parameters

**Predictive Perception Manager:**
- `max_disappeared`: Maximum frames an object can be missing (default: 30)
- `max_distance`: Maximum distance for associating detections (default: 50.0)
- `prediction_horizon`: Time horizon for trajectory prediction (default: 3.0s)

**Scene Understanding:**
- `publish_rate`: Rate for scene analysis (default: 5.0 Hz)
- `brightness_threshold_low`: Low brightness threshold (default: 50)
- `traffic_density_threshold`: Objects for high traffic (default: 3)

**Data Fusion:**
- `fusion_rate`: Sensor fusion processing rate (default: 20.0 Hz)
- `fusion_window`: Time window for synchronization (default: 0.1s)
- `camera_weight`: Weight for camera data (default: 0.6)

## ROS Topics

### Published Topics

**Predictive Perception Manager:**
- `~predicted_trajectories` (PredictedTrajectoryArray): Predicted object trajectories
- `~tracking_confidence` (TrackingConfidence): Tracking statistics and confidence
- `~tracked_objects` (TrackedObject): Currently tracked objects
- `~debug_image/compressed` (CompressedImage): Debug visualization

**Scene Understanding:**
- `~scene_analysis` (String): Comprehensive scene analysis
- `~scenario_type` (String): Current traffic scenario
- `~environmental_conditions` (String): Environmental conditions

**Data Fusion:**
- `~fused_perception` (String): Fused perception output
- `~confidence_metrics` (String): Fusion confidence metrics
- `~sensor_health` (String): Sensor health status
- `~ego_motion` (Twist2DStamped): Estimated ego motion

### Subscribed Topics

**Input Sources:**
- `/{veh}/camera_node/image/compressed`: Camera images
- `/{veh}/imu_node/data`: IMU data
- `/{veh}/lane_filter_node/lane_pose`: Lane pose estimates
- `/{veh}/apriltag_detector_node/detections`: AprilTag detections

## Testing

Run the comprehensive test suite:

```bash
# Test Kalman tracker
python3 test/test_kalman_tracker.py

# Test multi-object tracker
python3 test/test_multi_object_tracker.py

# Test trajectory prediction
python3 test/test_trajectory_prediction.py

# Test scene understanding
python3 test/test_scene_understanding.py

# Test sensor fusion
python3 test/test_sensor_fusion.py

# Test enhanced object detection
python3 test/test_enhanced_object_detection.py
```

## Architecture

```
Enhanced Perception System
├── Predictive Perception Manager
│   ├── Kalman Tracker
│   ├── Multi-Object Tracker
│   └── Trajectory Predictor
├── Scene Understanding Module
│   ├── Scene Analyzer
│   ├── Scene Classifier
│   └── Environmental Assessment
├── Enhanced Object Detection
│   ├── Detection Enhancement
│   ├── Filtering & Noise Reduction
│   └── Temporal Smoothing
└── Perception Data Fusion
    ├── Sensor Fusion Engine
    ├── Outlier Detection
    └── Temporal Consistency
```

## Performance

The system is optimized for real-time performance on Jetson Nano hardware:

- **Tracking**: Up to 5 objects simultaneously at 10 Hz
- **Prediction**: 3-second trajectory horizon with 0.1s resolution
- **Scene Analysis**: 5 Hz scene understanding
- **Fusion**: 20 Hz multi-modal sensor fusion

## Safety Features

- **Fail-safe Operation**: Graceful degradation when sensors fail
- **Confidence Monitoring**: Continuous confidence assessment
- **Outlier Detection**: Automatic filtering of erroneous measurements
- **Temporal Consistency**: Validation of measurement sequences

## Integration

The Enhanced Perception System integrates seamlessly with:

- **Safety System**: Provides object trajectories for collision prediction
- **Lane Control**: Supplies enhanced lane pose estimates
- **Navigation**: Delivers scene context for decision making
- **Visualization**: Offers comprehensive debug information

## Requirements Compliance

This implementation satisfies the following requirements from the specification:

- **Requirement 3.1**: Track position, velocity, and predicted trajectory for at least 3 seconds
- **Requirement 3.2**: Predict collision risk and adjust accordingly
- **Requirement 3.3**: Maintain tracking of up to 5 objects simultaneously
- **Requirement 3.4**: Increase safety margins when confidence drops below 70%
- **Requirement 3.5**: Update predictions at least 10 Hz

## Future Enhancements

Potential improvements for future versions:

1. **Deep Learning Integration**: Neural network-based trajectory prediction
2. **Multi-Camera Fusion**: Stereo vision and 360-degree perception
3. **Semantic Segmentation**: Pixel-level scene understanding
4. **Behavior Prediction**: Intent recognition for other vehicles
5. **Map Integration**: Incorporation of prior map knowledge

## Troubleshooting

**Common Issues:**

1. **Low Tracking Performance**: Check camera calibration and lighting conditions
2. **Fusion Errors**: Verify sensor synchronization and timing
3. **Memory Usage**: Adjust buffer sizes and history lengths
4. **CPU Load**: Reduce processing rates or enable adaptive scaling

**Debug Tools:**

- Enable debug visualization: `enable_visualization: true`
- Monitor confidence metrics: `rostopic echo /duckiebot/perception_data_fusion_node/confidence_metrics`
- Check sensor health: `rostopic echo /duckiebot/perception_data_fusion_node/sensor_health`

## Contributing

When contributing to this package:

1. Follow the existing code structure and naming conventions
2. Add comprehensive unit tests for new functionality
3. Update configuration files and documentation
4. Ensure real-time performance requirements are met
5. Test integration with existing Duckietown systems

## License

This package is part of the Duckietown Safe Navigation project and follows the same licensing terms.