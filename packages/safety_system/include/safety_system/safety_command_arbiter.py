# Safety Command Arbiter Core Logic
# Provides reusable command arbitration functionality

import numpy as np
from threading import Lock
import rospy


class SafetyCommandArbiter:
    """
    Core command arbitration logic
    
    Validates and arbitrates between different command sources with safety constraints.
    """
    
    def __init__(self, max_linear_vel=0.5, max_angular_vel=2.0):
        """
        Initialize safety command arbiter
        
        Args:
            max_linear_vel (float): Maximum linear velocity in m/s
            max_angular_vel (float): Maximum angular velocity in rad/s
        """
        self.max_linear_vel = max_linear_vel
        self.max_angular_vel = max_angular_vel
        
        self.command_sources = {}
        self.safety_constraints = {}
        self.arbitration_stats = {
            'total_commands': 0,
            'violations': 0,
            'overrides': 0
        }
        
        self.state_lock = Lock()
    
    def register_command_source(self, source_name, priority):
        """
        Register a command source with priority
        
        Args:
            source_name (str): Name of command source
            priority (int): Priority level (higher = more important)
        """
        with self.state_lock:
            self.command_sources[source_name] = {
                'priority': priority,
                'last_command': None,
                'last_timestamp': None
            }
    
    def update_command(self, source_name, command, timestamp=None):
        """
        Update command from a specific source
        
        Args:
            source_name (str): Name of command source
            command (dict): Command data with 'v' and 'omega'
            timestamp: Command timestamp
        """
        with self.state_lock:
            if source_name not in self.command_sources:
                return False
            
            if timestamp is None:
                timestamp = rospy.Time.now()
            
            self.command_sources[source_name]['last_command'] = command
            self.command_sources[source_name]['last_timestamp'] = timestamp
            
            return True
    
    def update_safety_constraints(self, constraints):
        """
        Update safety constraints
        
        Args:
            constraints (dict): Safety constraint parameters
        """
        with self.state_lock:
            self.safety_constraints = constraints
    
    def arbitrate_commands(self, command_timeout=0.5):
        """
        Arbitrate between available commands
        
        Args:
            command_timeout (float): Command validity timeout in seconds
            
        Returns:
            dict: Final arbitrated command or None
        """
        with self.state_lock:
            current_time = rospy.Time.now()
            
            # Get valid commands
            valid_commands = []
            for source_name, source_info in self.command_sources.items():
                cmd = source_info['last_command']
                timestamp = source_info['last_timestamp']
                
                if cmd is not None and timestamp is not None:
                    age = (current_time - timestamp).to_sec()
                    if age <= command_timeout:
                        valid_commands.append({
                            'source': source_name,
                            'command': cmd,
                            'priority': source_info['priority'],
                            'age': age
                        })
            
            if not valid_commands:
                return None
            
            # Sort by priority (highest first)
            valid_commands.sort(key=lambda x: x['priority'], reverse=True)
            
            # Select highest priority command
            selected_cmd = valid_commands[0]['command']
            
            # Apply safety constraints
            final_command = self._apply_safety_constraints(selected_cmd)
            
            # Update statistics
            self.arbitration_stats['total_commands'] += 1
            
            return final_command
    
    def _apply_safety_constraints(self, command):
        """
        Apply safety constraints to command
        
        Args:
            command (dict): Input command
            
        Returns:
            dict: Constrained command
        """
        constrained_cmd = {
            'v': command.get('v', 0.0),
            'omega': command.get('omega', 0.0)
        }
        
        violations = []
        
        # Apply velocity limits
        if abs(constrained_cmd['v']) > self.max_linear_vel:
            violations.append('linear_velocity_limit')
            constrained_cmd['v'] = np.sign(constrained_cmd['v']) * self.max_linear_vel
        
        if abs(constrained_cmd['omega']) > self.max_angular_vel:
            violations.append('angular_velocity_limit')
            constrained_cmd['omega'] = np.sign(constrained_cmd['omega']) * self.max_angular_vel
        
        # Apply safety-based constraints
        safety_factor = self._get_safety_factor()
        if safety_factor < 1.0:
            constrained_cmd['v'] *= safety_factor
            constrained_cmd['omega'] *= safety_factor
            violations.append('safety_constraint_applied')
        
        # Update violation statistics
        if violations:
            self.arbitration_stats['violations'] += len(violations)
        
        return constrained_cmd
    
    def _get_safety_factor(self):
        """
        Get safety constraint factor based on current safety status
        
        Returns:
            float: Safety factor (0.0 to 1.0)
        """
        safety_level = self.safety_constraints.get('safety_level', 0)
        
        if safety_level >= 4:  # CRITICAL
            return 0.0
        elif safety_level >= 3:  # DANGER
            return 0.3
        elif safety_level >= 2:  # WARNING
            return 0.6
        elif safety_level >= 1:  # CAUTION
            return 0.8
        else:
            return 1.0
    
    def get_arbitration_statistics(self):
        """
        Get command arbitration statistics
        
        Returns:
            dict: Arbitration statistics
        """
        with self.state_lock:
            return self.arbitration_stats.copy()
    
    def reset_statistics(self):
        """
        Reset arbitration statistics
        """
        with self.state_lock:
            self.arbitration_stats = {
                'total_commands': 0,
                'violations': 0,
                'overrides': 0
            }