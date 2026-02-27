# models/ace_core/ace_config.py

import yaml
from pathlib import Path

class ACEConfig:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.ace_integration_path = self.project_root / "ace_integration"
        
        # Load service configurations
        self.load_configs()
        
        # Endpoints
        self.a2f_endpoint = f"http://{self.a2f_config['host']}:{self.a2f_config['service']['rpc_port']}"
        self.animation_endpoint = f"http://{self.animation_config['host']}:{self.animation_config['port']}"
        
        # ACE Service endpoints
        self.ace_controller_endpoint = "http://ace-controller:8080"
        self.a2f_endpoint = "http://a2f-service:52000"
        self.animation_graph_endpoint = "http://ace-controller:50051"
        
        # Consciousness Core Integration settings
        self.consciousness_params = {
            'attention_threshold': 0.7,
            'emotional_coherence': 0.8,
            'memory_retention': 0.9
        }
        
        # Animation parameters
        self.animation_params = {
            'blend_shape_mapping': {
                'happy': 'emotion_happy',
                'sad': 'emotion_sad',
                'angry': 'emotion_angry',
                'surprised': 'emotion_surprised'
            },
            'emotion_intensity_scale': 1.0
        }
        
    def load_configs(self):
        """Load ACE service configurations"""
        try:
            # Load A2F config
            with open(self.ace_integration_path / "a2f_config.yaml") as f:
                self.a2f_config = yaml.safe_load(f)
                
            # Load animation config
            with open(self.ace_integration_path / "ac_a2f_config.yaml") as f:
                self.animation_config = yaml.safe_load(f)
        except Exception as e:
            print(f"Failed to load ACE configurations: {e}")
            raise

    def get_service_endpoint(self, service_name):
        """Get endpoint configuration for a specific service"""
        if service_name == "a2f":
            return (
                self.a2f_config["host"],
                self.a2f_config["service"]["rpc_port"]
            )
        elif service_name == "animation":
            return (
                self.animation_config["pipeline"]["stages"][1]["config"]["host"],
                self.animation_config["pipeline"]["stages"][1]["config"]["port"]
            )
        return None