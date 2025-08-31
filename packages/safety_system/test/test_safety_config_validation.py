#!/usr/bin/env python3

import unittest
import tempfile
import os
import yaml
from packages.safety_system.src.safety_config_validator import SafetyConfigValidator


class TestSafetyConfigValidation(unittest.TestCase):
    """
    Unit tests for Safety Configuration Validation
    
    Tests configuration validation, parameter boundary checks, and safety compliance.
    """
    
    def setUp(self):
        """Set up test environment"""
        self.validator = SafetyConfigValidator()
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test environment"""
        # Clean up temporary files
        for file in os.listdir(self.temp_dir):
            os.remove(os.path.join(self.temp_dir, file))
        os.rmdir(self.temp_dir)
    
    def create_test_config(self, config_data):
        """Create a temporary configuration file for testing"""
        config_path = os.path.join(self.temp_dir, 'test_config.yaml')
        with open(config_path, 'w') as file:
            yaml.dump(config_data, file)
        return config_path
    
    def test_valid_configuration(self):
        """Test validation of a valid configuration"""
        valid_config = {
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
            },
            'safety_fusion_manager': {
                'fusion_update_rate': 10.0,
                'health_check_timeout': 2.0,
                'safety_margin_buffer': 0.1,
                'decision_confidence_threshold': 0.8
            },
            'safety_command_arbiter': {
                'max_linear_velocity': 0.5,
                'max_angular_velocity': 2.0,
                'safety_override_priority': 10,
                'command_timeout': 0.5,
                'enable_safety_limits': True
            }
        }
        
        config_path = self.create_test_config(valid_config)
        is_valid, errors, warnings = self.validator.validate_config_file(config_path)
        
        self.assertTrue(is_valid, f"Valid configuration failed validation: {errors}")
        self.assertEqual(len(errors), 0, f"Unexpected errors: {errors}")
    
    def test_missing_section(self):
        """Test validation with missing required section"""
        incomplete_config = {
            'emergency_stop_system': {
                'max_response_time': 0.1,
                'min_distance_threshold': 0.3,
                'enable_predictive_stop': True,
                'emergency_timeout': 5.0
            }
            # Missing other required sections
        }
        
        config_path = self.create_test_config(incomplete_config)
        is_valid, errors, warnings = self.validator.validate_config_file(config_path)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        
        # Check that missing sections are reported
        error_text = ' '.join(errors)
        self.assertIn('collision_detection_manager', error_text)
        self.assertIn('safety_fusion_manager', error_text)
        self.assertIn('safety_command_arbiter', error_text)
    
    def test_missing_parameter(self):
        """Test validation with missing required parameter"""
        config_with_missing_param = {
            'emergency_stop_system': {
                'max_response_time': 0.1,
                # Missing min_distance_threshold
                'enable_predictive_stop': True,
                'emergency_timeout': 5.0
            },
            'collision_detection_manager': {
                'min_safe_distance': 0.3,
                'critical_distance': 0.15,
                'time_horizon': 3.0,
                'confidence_threshold': 0.7,
                'update_rate': 20.0
            },
            'safety_fusion_manager': {
                'fusion_update_rate': 10.0,
                'health_check_timeout': 2.0,
                'safety_margin_buffer': 0.1,
                'decision_confidence_threshold': 0.8
            },
            'safety_command_arbiter': {
                'max_linear_velocity': 0.5,
                'max_angular_velocity': 2.0,
                'safety_override_priority': 10,
                'command_timeout': 0.5,
                'enable_safety_limits': True
            }
        }
        
        config_path = self.create_test_config(config_with_missing_param)
        is_valid, errors, warnings = self.validator.validate_config_file(config_path)
        
        self.assertFalse(is_valid)
        self.assertIn('min_distance_threshold', ' '.join(errors))
    
    def test_parameter_type_validation(self):
        """Test parameter type validation"""
        config_with_wrong_types = {
            'emergency_stop_system': {
                'max_response_time': "0.1",  # Should be float, not string
                'min_distance_threshold': 0.3,
                'enable_predictive_stop': "true",  # Should be bool, not string
                'emergency_timeout': 5.0
            },
            'collision_detection_manager': {
                'min_safe_distance': 0.3,
                'critical_distance': 0.15,
                'time_horizon': 3.0,
                'confidence_threshold': 0.7,
                'update_rate': 20.0
            },
            'safety_fusion_manager': {
                'fusion_update_rate': 10.0,
                'health_check_timeout': 2.0,
                'safety_margin_buffer': 0.1,
                'decision_confidence_threshold': 0.8
            },
            'safety_command_arbiter': {
                'max_linear_velocity': 0.5,
                'max_angular_velocity': 2.0,
                'safety_override_priority': 10,
                'command_timeout': 0.5,
                'enable_safety_limits': True
            }
        }
        
        config_path = self.create_test_config(config_with_wrong_types)
        is_valid, errors, warnings = self.validator.validate_config_file(config_path)
        
        self.assertFalse(is_valid)
        
        # Check for type errors
        error_text = ' '.join(errors)
        self.assertIn('incorrect type', error_text)
    
    def test_parameter_range_validation(self):
        """Test parameter range validation"""
        config_with_out_of_range = {
            'emergency_stop_system': {
                'max_response_time': 1.0,  # Too high (max 0.5)
                'min_distance_threshold': -0.1,  # Too low (min 0.1)
                'enable_predictive_stop': True,
                'emergency_timeout': 5.0
            },
            'collision_detection_manager': {
                'min_safe_distance': 0.3,
                'critical_distance': 0.15,
                'time_horizon': 3.0,
                'confidence_threshold': 1.5,  # Too high (max 1.0)
                'update_rate': 20.0
            },
            'safety_fusion_manager': {
                'fusion_update_rate': 10.0,
                'health_check_timeout': 2.0,
                'safety_margin_buffer': 0.1,
                'decision_confidence_threshold': 0.8
            },
            'safety_command_arbiter': {
                'max_linear_velocity': 0.5,
                'max_angular_velocity': 2.0,
                'safety_override_priority': 10,
                'command_timeout': 0.5,
                'enable_safety_limits': True
            }
        }
        
        config_path = self.create_test_config(config_with_out_of_range)
        is_valid, errors, warnings = self.validator.validate_config_file(config_path)
        
        self.assertFalse(is_valid)
        
        # Check for range errors
        error_text = ' '.join(errors)
        self.assertIn('above maximum', error_text)
        self.assertIn('below minimum', error_text)
    
    def test_cross_parameter_constraints(self):
        """Test cross-parameter constraint validation"""
        config_with_constraint_violation = {
            'emergency_stop_system': {
                'max_response_time': 0.1,
                'min_distance_threshold': 0.5,  # Should be <= collision min_safe_distance
                'enable_predictive_stop': True,
                'emergency_timeout': 5.0
            },
            'collision_detection_manager': {
                'min_safe_distance': 0.3,  # Less than emergency min_distance_threshold
                'critical_distance': 0.4,  # Should be < min_safe_distance
                'time_horizon': 3.0,
                'confidence_threshold': 0.7,
                'update_rate': 20.0
            },
            'safety_fusion_manager': {
                'fusion_update_rate': 10.0,
                'health_check_timeout': 2.0,
                'safety_margin_buffer': 0.1,
                'decision_confidence_threshold': 0.8
            },
            'safety_command_arbiter': {
                'max_linear_velocity': 0.5,
                'max_angular_velocity': 2.0,
                'safety_override_priority': 10,
                'command_timeout': 0.5,
                'enable_safety_limits': True
            }
        }
        
        config_path = self.create_test_config(config_with_constraint_violation)
        is_valid, errors, warnings = self.validator.validate_config_file(config_path)
        
        self.assertFalse(is_valid)
        
        # Check for constraint violation errors
        error_text = ' '.join(errors)
        self.assertIn('Constraint violation', error_text)
    
    def test_safety_compliance_warnings(self):
        """Test safety compliance warning generation"""
        config_with_safety_concerns = {
            'emergency_stop_system': {
                'max_response_time': 0.18,  # Should generate warning (> 150ms)
                'min_distance_threshold': 0.3,
                'enable_predictive_stop': True,
                'emergency_timeout': 5.0
            },
            'collision_detection_manager': {
                'min_safe_distance': 0.15,  # Should generate warning (< 200mm)
                'critical_distance': 0.1,
                'time_horizon': 3.0,
                'confidence_threshold': 0.7,
                'update_rate': 20.0
            },
            'safety_fusion_manager': {
                'fusion_update_rate': 10.0,
                'health_check_timeout': 2.0,
                'safety_margin_buffer': 0.1,
                'decision_confidence_threshold': 0.8
            },
            'safety_command_arbiter': {
                'max_linear_velocity': 1.2,  # Should generate warning (> 1.0 m/s)
                'max_angular_velocity': 2.0,
                'safety_override_priority': 10,
                'command_timeout': 0.5,
                'enable_safety_limits': True
            }
        }
        
        config_path = self.create_test_config(config_with_safety_concerns)
        is_valid, errors, warnings = self.validator.validate_config_file(config_path)
        
        # Should be valid but have warnings
        self.assertTrue(is_valid)
        self.assertGreater(len(warnings), 0)
        
        # Check for specific warnings
        warning_text = ' '.join(warnings)
        self.assertIn('response time', warning_text)
        self.assertIn('safe distance', warning_text)
        self.assertIn('velocity', warning_text)
    
    def test_malformed_yaml(self):
        """Test handling of malformed YAML files"""
        malformed_yaml = "invalid: yaml: content: ["
        
        config_path = os.path.join(self.temp_dir, 'malformed.yaml')
        with open(config_path, 'w') as file:
            file.write(malformed_yaml)
        
        is_valid, errors, warnings = self.validator.validate_config_file(config_path)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertIn('YAML parsing error', ' '.join(errors))
    
    def test_nonexistent_file(self):
        """Test handling of nonexistent configuration files"""
        nonexistent_path = os.path.join(self.temp_dir, 'nonexistent.yaml')
        
        is_valid, errors, warnings = self.validator.validate_config_file(nonexistent_path)
        
        self.assertFalse(is_valid)
        self.assertGreater(len(errors), 0)
        self.assertIn('not found', ' '.join(errors))
    
    def test_validation_report_generation(self):
        """Test validation report generation"""
        valid_config = {
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
            },
            'safety_fusion_manager': {
                'fusion_update_rate': 10.0,
                'health_check_timeout': 2.0,
                'safety_margin_buffer': 0.1,
                'decision_confidence_threshold': 0.8
            },
            'safety_command_arbiter': {
                'max_linear_velocity': 0.5,
                'max_angular_velocity': 2.0,
                'safety_override_priority': 10,
                'command_timeout': 0.5,
                'enable_safety_limits': True
            }
        }
        
        config_path = self.create_test_config(valid_config)
        report = self.validator.generate_validation_report(config_path)
        
        self.assertIsInstance(report, str)
        self.assertIn('Validation Status: PASSED', report)
        self.assertIn('Configuration is valid', report)


if __name__ == '__main__':
    unittest.main()