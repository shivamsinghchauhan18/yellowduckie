# Safety Fusion Manager Core Logic
# Provides reusable safety data fusion functionality

from threading import Lock
from collections import deque
import rospy


class SafetyFusionManager:
    """
    Core safety fusion logic
    
    Coordinates multiple safety systems and provides unified safety decisions.
    """
    
    def __init__(self, fusion_rate=10.0, health_timeout=2.0):
        """
        Initialize safety fusion manager
        
        Args:
            fusion_rate (float): Safety fusion update rate in Hz
            health_timeout (float): Health check timeout in seconds
        """
        self.fusion_rate = fusion_rate
        self.health_timeout = health_timeout
        
        self.safety_inputs = {}
        self.system_health = {}
        self.safety_history = deque(maxlen=100)
        
        self.state_lock = Lock()
        
        # Safety level constants
        self.SAFETY_SAFE = 0
        self.SAFETY_CAUTION = 1
        self.SAFETY_WARNING = 2
        self.SAFETY_DANGER = 3
        self.SAFETY_CRITICAL = 4
    
    def update_safety_input(self, source, data):
        """
        Update safety input from a specific source
        
        Args:
            source (str): Source identifier
            data (dict): Safety data
        """
        with self.state_lock:
            self.safety_inputs[source] = {
                'data': data,
                'timestamp': rospy.Time.now()
            }
    
    def update_system_health(self, system, health_status):
        """
        Update system health status
        
        Args:
            system (str): System identifier
            health_status (dict): Health status information
        """
        with self.state_lock:
            self.system_health[system] = {
                'status': health_status,
                'timestamp': rospy.Time.now()
            }
    
    def fuse_safety_data(self):
        """
        Fuse safety data from all sources
        
        Returns:
            dict: Fused safety assessment
        """
        with self.state_lock:
            current_time = rospy.Time.now()
            
            # Check for stale data
            valid_inputs = {}
            for source, input_data in self.safety_inputs.items():
                age = (current_time - input_data['timestamp']).to_sec()
                if age <= self.health_timeout:
                    valid_inputs[source] = input_data['data']
            
            # Determine overall safety level
            overall_safety = self._calculate_overall_safety(valid_inputs)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(overall_safety)
            
            # Store in history
            safety_assessment = {
                'safety_level': overall_safety,
                'recommendations': recommendations,
                'valid_inputs': len(valid_inputs),
                'timestamp': current_time
            }
            
            self.safety_history.append(safety_assessment)
            
            return safety_assessment
    
    def _calculate_overall_safety(self, valid_inputs):
        """
        Calculate overall safety level from valid inputs
        
        Args:
            valid_inputs (dict): Valid safety inputs
            
        Returns:
            int: Overall safety level
        """
        max_safety_level = self.SAFETY_SAFE
        
        # Check emergency status
        emergency_input = valid_inputs.get('emergency_system')
        if emergency_input and emergency_input.get('active', False):
            max_safety_level = max(max_safety_level, self.SAFETY_CRITICAL)
        
        # Check collision risk
        collision_input = valid_inputs.get('collision_detection')
        if collision_input:
            risk_level = collision_input.get('risk_level', 0)
            if risk_level >= 4:  # CRITICAL
                max_safety_level = max(max_safety_level, self.SAFETY_CRITICAL)
            elif risk_level >= 3:  # HIGH
                max_safety_level = max(max_safety_level, self.SAFETY_DANGER)
            elif risk_level >= 2:  # MEDIUM
                max_safety_level = max(max_safety_level, self.SAFETY_WARNING)
            elif risk_level >= 1:  # LOW
                max_safety_level = max(max_safety_level, self.SAFETY_CAUTION)
        
        # Check system health
        for system, health_info in self.system_health.items():
            status = health_info['status'].get('status', 'UNKNOWN')
            if status == 'CRITICAL_FAILURE':
                max_safety_level = max(max_safety_level, self.SAFETY_CRITICAL)
            elif status == 'FAILURE':
                max_safety_level = max(max_safety_level, self.SAFETY_DANGER)
            elif status == 'DEGRADED':
                max_safety_level = max(max_safety_level, self.SAFETY_WARNING)
        
        return max_safety_level
    
    def _generate_recommendations(self, safety_level):
        """
        Generate safety recommendations based on safety level
        
        Args:
            safety_level (int): Current safety level
            
        Returns:
            list: Safety recommendations
        """
        recommendations = []
        
        if safety_level >= self.SAFETY_CRITICAL:
            recommendations.extend([
                'IMMEDIATE_STOP',
                'ACTIVATE_EMERGENCY_PROTOCOLS',
                'REQUEST_HUMAN_INTERVENTION'
            ])
        elif safety_level >= self.SAFETY_DANGER:
            recommendations.extend([
                'REDUCE_SPEED_SIGNIFICANTLY',
                'INCREASE_SAFETY_MARGINS',
                'ENHANCED_MONITORING'
            ])
        elif safety_level >= self.SAFETY_WARNING:
            recommendations.extend([
                'REDUCE_SPEED_MODERATELY',
                'INCREASE_VIGILANCE'
            ])
        elif safety_level >= self.SAFETY_CAUTION:
            recommendations.append('MAINTAIN_AWARENESS')
        
        return recommendations
    
    def get_safety_status(self):
        """
        Get current safety status
        
        Returns:
            dict: Current safety status
        """
        with self.state_lock:
            if self.safety_history:
                return self.safety_history[-1]
            else:
                return {
                    'safety_level': self.SAFETY_SAFE,
                    'recommendations': [],
                    'valid_inputs': 0,
                    'timestamp': rospy.Time.now()
                }
    
    def get_safety_statistics(self):
        """
        Get safety system statistics
        
        Returns:
            dict: Safety statistics
        """
        with self.state_lock:
            return {
                'active_inputs': len(self.safety_inputs),
                'healthy_systems': sum(1 for h in self.system_health.values() 
                                     if h['status'].get('status') == 'HEALTHY'),
                'total_systems': len(self.system_health),
                'history_length': len(self.safety_history)
            }