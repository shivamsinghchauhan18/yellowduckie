#!/usr/bin/env python3

import rospy
import yaml
from threading import Lock
from duckietown.dtros import DTROS, NodeType, DTParam, ParamType
from std_msgs.msg import String
from dynamic_reconfigure.server import Server
from dynamic_reconfigure.msg import Config


class DynamicParameterManager(DTROS):
    """
    Dynamic Parameter Manager for Safety System
    
    Manages dynamic updates of safety system parameters with validation and safety checks.
    """
    
    def __init__(self, node_name):
        super(DynamicParameterManager, self).__init__(
            node_name=node_name,
            node_type=NodeType.CONTROL
        )
        
        # Parameter storage
        self.current_parameters = {}
        self.parameter_history = []
        self.parameter_lock = Lock()
        
        # Safety constraints for parameter updates
        self.parameter_constraints = {
            'emergency_stop_system': {
                'max_response_time': {'min': 0.01, 'max': 0.2, 'safety_critical': True},
                'min_distance_threshold': {'min': 0.1, 'max': 1.0, 'safety_critical': True}
            },
            'collision_detection_manager': {
                'min_safe_distance': {'min': 0.1, 'max': 1.0, 'safety_critical': True},
                'update_rate': {'min': 5.0, 'max': 50.0, 'safety_critical': False}
            }
        }
        
        # Publishers
        self.pub_parameter_updates = rospy.Publisher(
            "~parameter_updates",
            String,
            queue_size=10
        )
        
        self.pub_parameter_validation = rospy.Publisher(
            "~parameter_validation",
            String,
            queue_size=10
        )
        
        # Subscribers
        self.sub_parameter_requests = rospy.Subscriber(
            "~parameter_requests",
            String,
            self.cb_parameter_request,
            queue_size=10
        )
        
        # Load initial parameters
        self.load_initial_parameters()
        
        self.log("Dynamic Parameter Manager initialized")
    
    def load_initial_parameters(self):
        """Load initial parameters from configuration"""
        try:
            # Load default configuration
            config_path = rospy.get_param('~config_file', 'default.yaml')
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            with self.parameter_lock:
                self.current_parameters = config
                self.log(f"Loaded initial parameters from {config_path}")
                
        except Exception as e:
            self.log(f"Failed to load initial parameters: {e}", "error")
            # Use safe defaults
            self.current_parameters = self.get_safe_default_parameters()
    
    def get_safe_default_parameters(self):
        """Get safe default parameters"""
        return {
            'emergency_stop_system': {
                'max_response_time': 0.1,
                'min_distance_threshold': 0.3,
                'enable_predictive_stop': True,
                'emergency_timeout': 5.0
            },
            'collision_detection_manager': {
                'min_safe_distance': 0.3,
                'critical_distance': 0.15,
                'time_horizon': 3.0,
                'confidence_threshold': 0.7,
                'update_rate': 20.0
            }
        }
    
    def cb_parameter_request(self, msg):
        """
        Handle parameter update requests
        
        Args:
            msg (String): Parameter update request
        """
        try:
            # Parse parameter request (format: SECTION.PARAMETER:VALUE)
            parts = msg.data.split(':')
            if len(parts) != 2:
                self.log(f"Invalid parameter request format: {msg.data}", "warn")
                return
            
            param_path, value_str = parts
            
            # Parse parameter path
            path_parts = param_path.split('.')
            if len(path_parts) != 2:
                self.log(f"Invalid parameter path: {param_path}", "warn")
                return
            
            section, parameter = path_parts
            
            # Convert value to appropriate type
            value = self.parse_parameter_value(value_str)
            
            # Validate and update parameter
            if self.validate_parameter_update(section, parameter, value):
                self.update_parameter(section, parameter, value)
            else:
                self.log(f"Parameter update validation failed: {param_path}={value}", "warn")
                
        except Exception as e:
            self.log(f"Error processing parameter request: {e}", "error")
    
    def parse_parameter_value(self, value_str):
        """
        Parse parameter value from string
        
        Args:
            value_str (str): String representation of value
            
        Returns:
            Parsed value with appropriate type
        """
        # Try to parse as different types
        value_str = value_str.strip()
        
        # Boolean
        if value_str.lower() in ['true', 'false']:
            return value_str.lower() == 'true'
        
        # Float
        try:
            if '.' in value_str:
                return float(value_str)
        except ValueError:
            pass
        
        # Integer
        try:
            return int(value_str)
        except ValueError:
            pass
        
        # String (default)
        return value_str
    
    def validate_parameter_update(self, section, parameter, value):
        """
        Validate parameter update against safety constraints
        
        Args:
            section (str): Parameter section
            parameter (str): Parameter name
            value: New parameter value
            
        Returns:
            bool: True if update is valid and safe
        """
        # Check if section and parameter exist in constraints
        if section not in self.parameter_constraints:
            self.log(f"Unknown parameter section: {section}", "warn")
            return False
        
        section_constraints = self.parameter_constraints[section]
        if parameter not in section_constraints:
            self.log(f"Unknown parameter: {section}.{parameter}", "warn")
            return False
        
        constraints = section_constraints[parameter]
        
        # Type validation
        current_value = self.get_current_parameter(section, parameter)
        if current_value is not None and type(value) != type(current_value):
            self.log(f"Parameter type mismatch: {section}.{parameter}", "error")
            return False
        
        # Range validation for numeric values
        if isinstance(value, (int, float)):
            min_val = constraints.get('min')
            max_val = constraints.get('max')
            
            if min_val is not None and value < min_val:
                self.log(f"Parameter value below minimum: {section}.{parameter}={value} < {min_val}", "error")
                return False
            
            if max_val is not None and value > max_val:
                self.log(f"Parameter value above maximum: {section}.{parameter}={value} > {max_val}", "error")
                return False
        
        # Safety-critical parameter checks
        if constraints.get('safety_critical', False):
            if not self.validate_safety_critical_update(section, parameter, value):
                return False
        
        return True
    
    def validate_safety_critical_update(self, section, parameter, value):
        """
        Additional validation for safety-critical parameters
        
        Args:
            section (str): Parameter section
            parameter (str): Parameter name
            value: New parameter value
            
        Returns:
            bool: True if update is safe
        """
        # Emergency response time must be very fast
        if section == 'emergency_stop_system' and parameter == 'max_response_time':
            if value > 0.15:
                self.log(f"Emergency response time too slow for safety: {value}s", "error")
                return False
        
        # Minimum distances must maintain safety margins
        if parameter in ['min_distance_threshold', 'min_safe_distance']:
            if value < 0.15:
                self.log(f"Safety distance too small: {value}m", "error")
                return False
        
        return True
    
    def update_parameter(self, section, parameter, value):
        """
        Update parameter value
        
        Args:
            section (str): Parameter section
            parameter (str): Parameter name
            value: New parameter value
        """
        with self.parameter_lock:
            # Store old value for history
            old_value = self.get_current_parameter(section, parameter)
            
            # Update parameter
            if section not in self.current_parameters:
                self.current_parameters[section] = {}
            
            self.current_parameters[section][parameter] = value
            
            # Record in history
            update_record = {
                'timestamp': rospy.Time.now(),
                'section': section,
                'parameter': parameter,
                'old_value': old_value,
                'new_value': value
            }
            self.parameter_history.append(update_record)
            
            # Publish update notification
            update_msg = String()
            update_msg.data = f"UPDATED:{section}.{parameter}:{old_value}->{value}"
            self.pub_parameter_updates.publish(update_msg)
            
            self.log(f"Parameter updated: {section}.{parameter} = {value}")
    
    def get_current_parameter(self, section, parameter):
        """
        Get current parameter value
        
        Args:
            section (str): Parameter section
            parameter (str): Parameter name
            
        Returns:
            Current parameter value or None if not found
        """
        with self.parameter_lock:
            return self.current_parameters.get(section, {}).get(parameter)
    
    def get_all_parameters(self):
        """
        Get all current parameters
        
        Returns:
            dict: All current parameters
        """
        with self.parameter_lock:
            return self.current_parameters.copy()
    
    def save_parameters_to_file(self, filename):
        """
        Save current parameters to file
        
        Args:
            filename (str): Output filename
        """
        try:
            with self.parameter_lock:
                with open(filename, 'w') as file:
                    yaml.dump(self.current_parameters, file, default_flow_style=False)
            
            self.log(f"Parameters saved to {filename}")
            
        except Exception as e:
            self.log(f"Failed to save parameters: {e}", "error")
    
    def load_parameters_from_file(self, filename):
        """
        Load parameters from file with validation
        
        Args:
            filename (str): Input filename
            
        Returns:
            bool: True if loaded successfully
        """
        try:
            with open(filename, 'r') as file:
                new_parameters = yaml.safe_load(file)
            
            # Validate all parameters before applying
            validation_errors = []
            for section, section_params in new_parameters.items():
                for parameter, value in section_params.items():
                    if not self.validate_parameter_update(section, parameter, value):
                        validation_errors.append(f"{section}.{parameter}")
            
            if validation_errors:
                self.log(f"Parameter validation failed for: {validation_errors}", "error")
                return False
            
            # Apply all parameters
            with self.parameter_lock:
                self.current_parameters = new_parameters
            
            self.log(f"Parameters loaded from {filename}")
            return True
            
        except Exception as e:
            self.log(f"Failed to load parameters: {e}", "error")
            return False
    
    def get_parameter_history(self, limit=None):
        """
        Get parameter update history
        
        Args:
            limit (int): Maximum number of records to return
            
        Returns:
            list: Parameter update history
        """
        with self.parameter_lock:
            if limit:
                return self.parameter_history[-limit:]
            else:
                return self.parameter_history.copy()


if __name__ == "__main__":
    # Create and run the dynamic parameter manager
    param_manager = DynamicParameterManager("dynamic_parameter_manager")
    rospy.spin()