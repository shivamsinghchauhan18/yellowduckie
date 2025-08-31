# Camera Mock Package

This package provides a mock camera node for testing Duckietown systems without physical camera hardware.

## Features

- **Synthetic Images**: Generates moving objects (circles, rectangles) that simulate ducks and duckiebots
- **Static Test Pattern**: Provides a checkerboard pattern with colored objects for calibration
- **Random Noise**: Generates random noise images for stress testing
- **Configurable Parameters**: Adjustable frame rate, image size, and mock type

## Usage

### Launch with default settings:
```bash
roslaunch camera_mock camera_mock_node.launch
```

### Launch with custom parameters:
```bash
roslaunch camera_mock camera_mock_node.launch mock_type:=static publish_rate:=15.0
```

## Parameters

- `publish_rate`: Camera frame rate in Hz (default: 10.0)
- `image_width`: Image width in pixels (default: 640)
- `image_height`: Image height in pixels (default: 480)
- `mock_type`: Type of mock image (default: "synthetic")
  - `synthetic`: Moving objects that simulate real scenarios
  - `static`: Static test pattern with colored objects
  - `noise`: Random noise for testing robustness

## Published Topics

- `~image/compressed`: Compressed camera images
- `~camera_info`: Camera calibration information

## Integration

The camera mock is automatically included in the main launch file when `camera_mock:=true` (default).
This allows the anti_instagram_node and other camera-dependent nodes to receive image data for processing.