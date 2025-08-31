#!/usr/bin/env python3

import yaml
import sys
import os
import argparse
from typing import Dict, Any, List, Tuple


class SafetyConfigValidator:
    """
    Safety Configuration Validator
    
    Validates safety system configuration files for correctness and safety compliance.
    """
    
    def __init__(self):
        """Initialize the validator with safety requirements"""
        # Define safety requirements and constraints
        self.safety_requirements = {
            'emergency_stop_system': {
                'max_response_time': {'min': 0.01, 'max': 0.5, 'type': float},
                'min_distance_threshold': {'min': 0.1, 'max': 2.0, 'type': float},
                'enable_predictive_stop': {'type': bool},
                'emergency_timeout': {'min': 1.0, 'max': 60.0, 'type': float}
            },
            'collision_detection_manager': {
                'min_safe_distance': {'min': 0.1, 'max': 2.0, 'type': float},
                'critical_distance': {'min': 0.05, 'max': 1.0, 'type': float},
                'time_horizon': {'min': 1.0, 'max': 10.0, 'type': float},
                'confidence_threshold': {'min': 0.1, 'max': 1.0, 'type': float},
                'update_rate': {'min': 1.0, 'max': 100.0, 'type': float}
            },
            'safety_fusion_manager': {
                'fusion_update_rate': {'min': 1.0, 'max': 50.0, 'type': float},
                'health_check_timeout': {'min': 0.1, 'max': 10.0, 'type': float},
                'safety_margin_buffer': {'min': 0.0, 'max': 1.0, 'type': float},
                'decision_confidence_threshold': {'min': 0.1, 'max': 1.0, 'type': float}
            },
            'safety_command_arbiter': {
                'max_linear_velocity': {'min': 0.1, 'max': 2.0, 'type': float},
                'max_angular_velocity': {'min': 0.1, 'max': 5.0, 'type': float},
                'safety_override_priority': {'min': 1, 'max': 10, 'type': int},
                'command_timeout': {'min': 0.1, 'max': 2.0, 'type': float},
                'enable_safety_limits': {'type': bool}
            }
        }
        
        # Define cross-parameter constraints
        self.cross_constraints = [
            ('collision_detection_manager.critical_distance', 
             'collision_detection_manager.min_safe_distance', 
             'less_than'),
            ('emergency_stop_system.min_distance_threshold',
             'collision_detection_manager.min_safe_distance',
             'less_than_or_equal'),
            ('emergency_stop_system.max_response_time',
             0.2,  # Hard safety limit
             'less_than')
        ]
        
        self.validation_errors = []
        self.validation_warnings = []
    
    def validate_config_file(self, config_path: str) -> Tuple[bool, List[str], List[str]]:
        """
        Validate a safety configuration file
        
        Args:
            config_path (str): Path to configuration file
            
        Returns:
            Tuple[bool, List[str], List[str]]: (is_valid, errors, warnings)
        """
        self.validation_errors = []
        self.validation_warnings = []
        
        try:
            # Load configuration file
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
            
            # Validate structure and parameters
            self._validate_structure(config)
            self._validate_parameters(config)
            self._validate_cross_constraints(config)
            self._validate_safety_compliance(config)
            
            is_valid = len(self.validation_errors) == 0
            
            return is_valid, self.validation_errors, self.validation_warnings
            
        except FileNotFoundError:
            self.validation_errors.append(f"Configuration file not found: {config_path}")
            return False, self.validation_errors, self.validation_warnings
        
        except yaml.YAMLError as e:
            self.validation_errors.append(f"YAML parsing error: {e}")
            return False, self.validation_errors, self.validation_warnings
        
        except Exception as e:
            self.validation_errors.append(f"Unexpected error: {e}")
            return False, self.validation_errors, self.validation_warnings
    
    def _validate_structure(self, config: Dict[str, Any]):
        """Validate configuration structure"""
        required_sections = list(self.safety_requirements.keys())
        
        for section in required_sections:
            if section not in config:
                self.validation_errors.append(f"Missing required section: {section}")
            else:
                # Check required parameters in each section
                section_requirements = self.safety_requirements[section]
                for param in section_requirements:
                    if param not in config[section]:
                        self.validation_errors.append(
                            f"Missing required parameter: {section}.{param}"
                        )
    
    def _validate_parameters(self, config: Dict[str, Any]):
        """Validate individual parameters"""
        for section_name, section_config in config.items():
            if section_name in self.safety_requirements:
                section_requirements = self.safety_requirements[section_name]
                
                for param_name, param_value in section_config.items():
                    if param_name in section_requirements:
                        requirements = section_requirements[param_name]
                        self._validate_parameter(
                            f"{section_name}.{param_name}", 
                            param_value, 
                            requirements
                        )
    
    def _validate_parameter(self, param_path: str, value: Any, requirements: Dict[str, Any]):
        """Validate a single parameter"""
        # Type validation
        expected_type = requirements.get('type')
        if expected_type and not isinstance(value, expected_type):
            self.validation_errors.append(
                f"Parameter {param_path} has incorrect type. "
                f"Expected {expected_type.__name__}, got {type(value).__name__}"
            )
            return
        
        # Range validation for numeric types
        if isinstance(value, (int, float)):
            min_val = requirements.get('min')
            max_val = requirements.get('max')
            
            if min_val is not None and value < min_val:
                self.validation_errors.append(
                    f"Parameter {param_path} value {value} is below minimum {min_val}"
                )
            
            if max_val is not None and value > max_val:
                self.validation_errors.append(
                    f"Parameter {param_path} value {value} is above maximum {max_val}"
                )
    
    def _validate_cross_constraints(self, config: Dict[str, Any]):
        """Validate cross-parameter constraints"""
        for constraint in self.cross_constraints:
            param1_path, param2_path, constraint_type = constraint
            
            param1_value = self._get_nested_value(config, param1_path)
            
            if isinstance(param2_path, str):
                param2_value = self._get_nested_value(config, param2_path)
            else:
                param2_value = param2_path  # It's a literal value
            
            if param1_value is None or param2_value is None:
                continue  # Skip if parameters don't exist
            
            if constraint_type == 'less_than' and param1_value >= param2_value:
                self.validation_errors.append(
                    f"Constraint violation: {param1_path} ({param1_value}) "
                    f"must be less than {param2_path} ({param2_value})"
                )
            elif constraint_type == 'less_than_or_equal' and param1_value > param2_value:
                self.validation_errors.append(
                    f"Constraint violation: {param1_path} ({param1_value}) "
                    f"must be less than or equal to {param2_path} ({param2_value})"
                )
    
    def _validate_safety_compliance(self, config: Dict[str, Any]):
        """Validate safety compliance requirements"""
        # Check emergency response time is within safety limits
        emergency_config = config.get('emergency_stop_system', {})
        response_time = emergency_config.get('max_response_time')
        
        if response_time and response_time > 0.15:
            self.validation_warnings.append(
                f"Emergency response time {response_time}s exceeds recommended 150ms limit"
            )
        
        # Check minimum safe distances are reasonable
        collision_config = config.get('collision_detection_manager', {})
        min_safe_distance = collision_config.get('min_safe_distance')
        
        if min_safe_distance and min_safe_distance < 0.2:
            self.validation_warnings.append(
                f"Minimum safe distance {min_safe_distance}m may be too small for safe operation"
            )
        
        # Check velocity limits are reasonable
        arbiter_config = config.get('safety_command_arbiter', {})
        max_linear_vel = arbiter_config.get('max_linear_velocity')
        
        if max_linear_vel and max_linear_vel > 1.0:
            self.validation_warnings.append(
                f"Maximum linear velocity {max_linear_vel} m/s may be too high for safe operation"
            )
    
    def _get_nested_value(self, config: Dict[str, Any], path: str) -> Any:
        """Get nested configuration value using dot notation"""
        keys = path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None
    
    def generate_validation_report(self, config_path: str) -> str:
        """Generate a detailed validation report"""
        is_valid, errors, warnings = self.validate_config_file(config_path)
        
        report = f"Safety Configuration Validation Report\n"
        report += f"Configuration File: {config_path}\n"
        report += f"Validation Status: {'PASSED' if is_valid else 'FAILED'}\n"
        report += f"Errors: {len(errors)}\n"
        report += f"Warnings: {len(warnings)}\n\n"
        
        if errors:
            report += "ERRORS:\n"
            for i, error in enumerate(errors, 1):
                report += f"  {i}. {error}\n"
            report += "\n"
        
        if warnings:
            report += "WARNINGS:\n"
            for i, warning in enumerate(warnings, 1):
                report += f"  {i}. {warning}\n"
            report += "\n"
        
        if is_valid:
            report += "Configuration is valid and safe for deployment.\n"
        else:
            report += "Configuration has errors and should not be deployed.\n"
        
        return report


def main():
    """Main function for command-line usage"""
    parser = argparse.ArgumentParser(description='Validate safety system configuration files')
    parser.add_argument('config_file', help='Path to configuration file to validate')
    parser.add_argument('--report', '-r', action='store_true', 
                       help='Generate detailed validation report')
    parser.add_argument('--output', '-o', help='Output file for validation report')
    
    args = parser.parse_args()
    
    validator = SafetyConfigValidator()
    
    if args.report:
        report = validator.generate_validation_report(args.config_file)
        
        if args.output:
            with open(args.output, 'w') as f:
                f.write(report)
            print(f"Validation report written to {args.output}")
        else:
            print(report)
    else:
        is_valid, errors, warnings = validator.validate_config_file(args.config_file)
        
        if is_valid:
            print("✓ Configuration is valid")
            if warnings:
                print(f"⚠ {len(warnings)} warnings found")
                for warning in warnings:
                    print(f"  - {warning}")
            sys.exit(0)
        else:
            print("✗ Configuration validation failed")
            print(f"Found {len(errors)} errors:")
            for error in errors:
                print(f"  - {error}")
            if warnings:
                print(f"Found {len(warnings)} warnings:")
                for warning in warnings:
                    print(f"  - {warning}")
            sys.exit(1)


if __name__ == '__main__':
    main()