#!/usr/bin/env python3

try:
    import rospy
except ImportError:
    # Mock rospy for testing
    class MockRospy:
        def logwarn(self, msg):
            print(f"WARN: {msg}")
    rospy = MockRospy()

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import IntEnum
import threading
import time


class SignalType(IntEnum):
    """Types of LED signals"""
    NONE = 0
    LEFT_TURN = 1
    RIGHT_TURN = 2
    HAZARD = 3
    LANE_CHANGE_LEFT = 4
    LANE_CHANGE_RIGHT = 5
    EMERGENCY_STOP = 6
    CAUTION = 7


class SignalPriority(IntEnum):
    """Signal priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    EMERGENCY = 3


@dataclass
class SignalPattern:
    """LED signal pattern definition"""
    name: str
    on_duration: float  # seconds
    off_duration: float  # seconds
    repeat_count: int  # -1 for infinite
    brightness: float  # 0.0 to 1.0
    color: str  # LED color pattern name
    priority: SignalPriority


@dataclass
class SignalRequest:
    """Request for LED signaling"""
    signal_type: SignalType
    duration: float  # Total duration in seconds, -1 for indefinite
    priority: SignalPriority
    adaptive_brightness: bool = True
    visibility_verification: bool = True


class LaneChangeSignalingSystem:
    """
    Enhanced LED signaling system for lane changes with adaptive brightness,
    timing-based activation/deactivation, and visibility verification.
    """
    
    def __init__(self, led_service_proxy=None):
        """Initialize the signaling system"""
        self.led_service = led_service_proxy
        
        # Signal patterns configuration
        self.signal_patterns = self._initialize_signal_patterns()
        
        # Current signaling state
        self.current_signal = SignalType.NONE
        self.current_pattern = None
        self.signal_start_time = None
        self.signal_duration = 0.0
        self.signal_thread = None
        self.signal_active = False
        self.signal_lock = threading.Lock()
        
        # Adaptive brightness parameters
        self.base_brightness = 1.0
        self.ambient_light_level = 0.5  # 0.0 = dark, 1.0 = bright
        self.min_brightness = 0.3
        self.max_brightness = 1.0
        
        # Visibility verification
        self.visibility_distance_threshold = 2.0  # meters
        self.visibility_confirmed = True
        
        # Signal timing requirements (from traffic regulations)
        self.min_signal_duration = 2.0  # seconds
        self.max_signal_duration = 30.0  # seconds
        self.signal_frequency = 1.0  # Hz (1 flash per second)
        
        # Priority queue for signal requests
        self.signal_queue = []
        self.queue_lock = threading.Lock()
    
    def _initialize_signal_patterns(self) -> Dict[SignalType, SignalPattern]:
        """Initialize predefined signal patterns"""
        patterns = {}
        
        # Standard turn signals
        patterns[SignalType.LEFT_TURN] = SignalPattern(
            name="CAR_SIGNAL_LEFT",
            on_duration=0.5,
            off_duration=0.5,
            repeat_count=-1,
            brightness=1.0,
            color="YELLOW",
            priority=SignalPriority.NORMAL
        )
        
        patterns[SignalType.RIGHT_TURN] = SignalPattern(
            name="CAR_SIGNAL_RIGHT",
            on_duration=0.5,
            off_duration=0.5,
            repeat_count=-1,
            brightness=1.0,
            color="YELLOW",
            priority=SignalPriority.NORMAL
        )
        
        # Lane change signals (slightly different timing for distinction)
        patterns[SignalType.LANE_CHANGE_LEFT] = SignalPattern(
            name="CAR_SIGNAL_LEFT",
            on_duration=0.4,
            off_duration=0.6,
            repeat_count=-1,
            brightness=1.0,
            color="YELLOW",
            priority=SignalPriority.HIGH
        )
        
        patterns[SignalType.LANE_CHANGE_RIGHT] = SignalPattern(
            name="CAR_SIGNAL_RIGHT",
            on_duration=0.4,
            off_duration=0.6,
            repeat_count=-1,
            brightness=1.0,
            color="YELLOW",
            priority=SignalPriority.HIGH
        )
        
        # Hazard and emergency signals
        patterns[SignalType.HAZARD] = SignalPattern(
            name="CAR_SIGNAL_HAZARD",
            on_duration=0.5,
            off_duration=0.5,
            repeat_count=-1,
            brightness=1.0,
            color="RED",
            priority=SignalPriority.HIGH
        )
        
        patterns[SignalType.EMERGENCY_STOP] = SignalPattern(
            name="CAR_EMERGENCY_STOP",
            on_duration=0.2,
            off_duration=0.2,
            repeat_count=-1,
            brightness=1.0,
            color="RED",
            priority=SignalPriority.EMERGENCY
        )
        
        patterns[SignalType.CAUTION] = SignalPattern(
            name="CAR_CAUTION",
            on_duration=1.0,
            off_duration=1.0,
            repeat_count=-1,
            brightness=0.8,
            color="ORANGE",
            priority=SignalPriority.NORMAL
        )
        
        return patterns
    
    def request_signal(self, request: SignalRequest) -> bool:
        """
        Request a signal to be displayed.
        
        Args:
            request: Signal request with type, duration, and priority
            
        Returns:
            True if request was accepted, False otherwise
        """
        with self.queue_lock:
            # Check if we can accept this request
            if not self._can_accept_request(request):
                return False
            
            # Add to queue or replace current signal if higher priority
            if self._should_preempt_current_signal(request):
                self._stop_current_signal()
                self._start_signal(request)
            else:
                self.signal_queue.append(request)
                self.signal_queue.sort(key=lambda r: r.priority.value, reverse=True)
            
            return True
    
    def _can_accept_request(self, request: SignalRequest) -> bool:
        """Check if a signal request can be accepted"""
        # Validate signal type
        if request.signal_type not in self.signal_patterns:
            return False
        
        # Validate duration
        if request.duration > 0:
            if (request.duration < self.min_signal_duration or 
                request.duration > self.max_signal_duration):
                return False
        
        return True
    
    def _should_preempt_current_signal(self, request: SignalRequest) -> bool:
        """Check if new request should preempt current signal"""
        if not self.signal_active:
            return True
        
        # Emergency signals always preempt
        if request.priority == SignalPriority.EMERGENCY:
            return True
        
        # Higher priority preempts lower priority
        current_pattern = self.signal_patterns.get(self.current_signal)
        if current_pattern and request.priority.value > current_pattern.priority.value:
            return True
        
        return False
    
    def _start_signal(self, request: SignalRequest):
        """Start displaying a signal"""
        with self.signal_lock:
            self.current_signal = request.signal_type
            self.current_pattern = self.signal_patterns[request.signal_type]
            self.signal_start_time = time.time()
            self.signal_duration = request.duration
            self.signal_active = True
            
            # Calculate adaptive brightness if requested
            if request.adaptive_brightness:
                self.current_pattern.brightness = self._calculate_adaptive_brightness()
            
            # Start signaling thread
            self.signal_thread = threading.Thread(target=self._signal_worker)
            self.signal_thread.daemon = True
            self.signal_thread.start()
    
    def _signal_worker(self):
        """Worker thread for signal timing and LED control"""
        pattern = self.current_pattern
        if not pattern:
            return
        
        cycle_count = 0
        start_time = time.time()
        
        while self.signal_active:
            current_time = time.time()
            
            # Check if signal duration has expired
            if (self.signal_duration > 0 and 
                current_time - start_time >= self.signal_duration):
                break
            
            # Check if we've reached repeat limit
            if (pattern.repeat_count > 0 and 
                cycle_count >= pattern.repeat_count):
                break
            
            # LED ON phase
            if self.signal_active:
                self._set_led_pattern(pattern.name, pattern.brightness)
                time.sleep(pattern.on_duration)
            
            # LED OFF phase
            if self.signal_active:
                self._set_led_pattern("CAR_DRIVING", 0.5)  # Default driving pattern
                time.sleep(pattern.off_duration)
            
            cycle_count += 1
        
        # Signal completed, clean up
        self._stop_current_signal()
        self._process_signal_queue()
    
    def _set_led_pattern(self, pattern_name: str, brightness: float):
        """Set LED pattern through service call"""
        if not self.led_service:
            return
        
        try:
            # Import here to avoid ROS dependency issues in tests
            from duckietown_msgs.srv import ChangePatternRequest
            from std_msgs.msg import String
            
            request = ChangePatternRequest(String(pattern_name))
            self.led_service(request)
            
        except Exception as e:
            rospy.logwarn(f"Failed to set LED pattern {pattern_name}: {e}")
    
    def _calculate_adaptive_brightness(self) -> float:
        """Calculate adaptive brightness based on ambient conditions"""
        # Base brightness adjustment based on ambient light
        if self.ambient_light_level < 0.3:  # Dark conditions
            brightness_factor = 0.7  # Reduce brightness to avoid glare
        elif self.ambient_light_level > 0.7:  # Bright conditions
            brightness_factor = 1.0  # Full brightness for visibility
        else:  # Normal conditions
            brightness_factor = 0.85
        
        # Apply visibility distance factor
        if hasattr(self, 'visibility_distance'):
            if self.visibility_distance < self.visibility_distance_threshold:
                brightness_factor *= 1.2  # Increase brightness if visibility is poor
        
        # Clamp to valid range
        adaptive_brightness = self.base_brightness * brightness_factor
        return max(self.min_brightness, min(self.max_brightness, adaptive_brightness))
    
    def _stop_current_signal(self):
        """Stop the current signal"""
        with self.signal_lock:
            self.signal_active = False
            self.current_signal = SignalType.NONE
            self.current_pattern = None
            self.signal_start_time = None
            
            # Reset to default driving pattern
            self._set_led_pattern("CAR_DRIVING", 0.8)
    
    def _process_signal_queue(self):
        """Process the next signal in the queue"""
        with self.queue_lock:
            if self.signal_queue:
                next_request = self.signal_queue.pop(0)
                self._start_signal(next_request)
    
    def stop_signal(self, signal_type: SignalType = None) -> bool:
        """
        Stop a specific signal or all signals.
        
        Args:
            signal_type: Specific signal to stop, or None to stop all
            
        Returns:
            True if signal was stopped, False otherwise
        """
        with self.signal_lock:
            if signal_type is None or self.current_signal == signal_type:
                self._stop_current_signal()
                return True
            
            # Remove from queue if present
            with self.queue_lock:
                original_length = len(self.signal_queue)
                self.signal_queue = [req for req in self.signal_queue 
                                   if req.signal_type != signal_type]
                return len(self.signal_queue) < original_length
    
    def update_ambient_conditions(self, light_level: float, visibility_distance: float = None):
        """
        Update ambient conditions for adaptive brightness.
        
        Args:
            light_level: Ambient light level (0.0 = dark, 1.0 = bright)
            visibility_distance: Visibility distance in meters
        """
        self.ambient_light_level = max(0.0, min(1.0, light_level))
        
        if visibility_distance is not None:
            self.visibility_distance = visibility_distance
            self.visibility_confirmed = visibility_distance >= self.visibility_distance_threshold
    
    def verify_signal_visibility(self, observer_distance: float) -> bool:
        """
        Verify that signals are visible from specified distance.
        
        Args:
            observer_distance: Distance from which visibility is checked
            
        Returns:
            True if signals should be visible, False otherwise
        """
        if not self.signal_active:
            return True  # No signal to verify
        
        # Check if current brightness is sufficient for the distance
        required_brightness = self._calculate_required_brightness(observer_distance)
        current_brightness = self.current_pattern.brightness if self.current_pattern else 0.0
        
        return current_brightness >= required_brightness
    
    def _calculate_required_brightness(self, distance: float) -> float:
        """Calculate required brightness for visibility at given distance"""
        # Simple model: brightness requirement increases with distance
        base_requirement = 0.3
        distance_factor = min(1.0, distance / 5.0)  # Normalize to 5 meters
        ambient_factor = 1.0 - self.ambient_light_level * 0.3  # Darker = higher requirement
        
        required = base_requirement + distance_factor * 0.4 + ambient_factor * 0.3
        return max(0.3, min(1.0, required))
    
    def get_signal_status(self) -> Dict:
        """Get current signaling status"""
        with self.signal_lock:
            status = {
                'active': self.signal_active,
                'current_signal': self.current_signal.name if self.current_signal != SignalType.NONE else None,
                'signal_duration_remaining': 0.0,
                'brightness': self.current_pattern.brightness if self.current_pattern else 0.0,
                'queue_length': len(self.signal_queue),
                'visibility_confirmed': self.visibility_confirmed
            }
            
            if self.signal_active and self.signal_start_time and self.signal_duration > 0:
                elapsed = time.time() - self.signal_start_time
                status['signal_duration_remaining'] = max(0.0, self.signal_duration - elapsed)
            
            return status
    
    def validate_signal_timing(self, signal_type: SignalType, duration: float) -> Tuple[bool, str]:
        """
        Validate signal timing against regulations and safety requirements.
        
        Args:
            signal_type: Type of signal to validate
            duration: Requested duration
            
        Returns:
            (is_valid, error_message)
        """
        if signal_type not in self.signal_patterns:
            return False, f"Unknown signal type: {signal_type}"
        
        if duration <= 0:
            return True, ""  # Indefinite duration is allowed
        
        if duration < self.min_signal_duration:
            return False, f"Duration {duration}s is below minimum {self.min_signal_duration}s"
        
        if duration > self.max_signal_duration:
            return False, f"Duration {duration}s exceeds maximum {self.max_signal_duration}s"
        
        # Check signal-specific requirements
        pattern = self.signal_patterns[signal_type]
        
        if signal_type in [SignalType.LANE_CHANGE_LEFT, SignalType.LANE_CHANGE_RIGHT]:
            if duration < 2.0:
                return False, "Lane change signals must be active for at least 2 seconds"
        
        if signal_type == SignalType.EMERGENCY_STOP:
            if duration > 0 and duration < 5.0:
                return False, "Emergency stop signals should be active for at least 5 seconds"
        
        return True, ""
    
    def create_lane_change_signal_sequence(self, direction: int, total_duration: float) -> List[SignalRequest]:
        """
        Create a sequence of signals for a complete lane change maneuver.
        
        Args:
            direction: -1 for left, 1 for right
            total_duration: Total duration of lane change maneuver
            
        Returns:
            List of signal requests in sequence
        """
        sequence = []
        
        # Determine signal type
        signal_type = SignalType.LANE_CHANGE_LEFT if direction < 0 else SignalType.LANE_CHANGE_RIGHT
        
        # Pre-maneuver signaling (2 seconds minimum)
        pre_signal_duration = max(2.0, total_duration * 0.25)
        sequence.append(SignalRequest(
            signal_type=signal_type,
            duration=pre_signal_duration,
            priority=SignalPriority.HIGH,
            adaptive_brightness=True,
            visibility_verification=True
        ))
        
        # During maneuver (continue signaling)
        maneuver_duration = total_duration * 0.75
        sequence.append(SignalRequest(
            signal_type=signal_type,
            duration=maneuver_duration,
            priority=SignalPriority.HIGH,
            adaptive_brightness=True,
            visibility_verification=True
        ))
        
        return sequence
    
    def emergency_signal_override(self):
        """Activate emergency signaling, overriding all other signals"""
        emergency_request = SignalRequest(
            signal_type=SignalType.EMERGENCY_STOP,
            duration=-1,  # Indefinite
            priority=SignalPriority.EMERGENCY,
            adaptive_brightness=True,
            visibility_verification=False  # Emergency doesn't wait for verification
        )
        
        self.request_signal(emergency_request)
    
    def shutdown(self):
        """Shutdown the signaling system"""
        self.stop_signal()  # Stop all signals
        
        # Wait for signal thread to finish
        if self.signal_thread and self.signal_thread.is_alive():
            self.signal_thread.join(timeout=1.0)
        
        # Reset to default driving pattern
        self._set_led_pattern("CAR_DRIVING", 0.8)