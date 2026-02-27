# models/ace_core/ace_agent.py

from models.core.consciousness_core import ConsciousnessCore
from models.memory.emotional_memory_core import EmotionalMemoryCore
from models.predictive.dreamer_emotional_wrapper import DreamerEmotionalWrapper
from models.integration.experience_integrator import ExperienceIntegrator
from models.self_model.self_representation_core import SelfRepresentationCore
import aiohttp
import json

class ACEConsciousAgent:
    def __init__(self, config):
        # Core Components
        self.consciousness_core = ConsciousnessCore()
        self.emotional_memory = EmotionalMemoryCore()
        self.world_model = DreamerEmotionalWrapper()
        self.experience_integrator = ExperienceIntegrator()
        self.self_model = SelfRepresentationCore()
        
        # ACE Components
        self.ace_controller = None
        self.audio2face = None
        self.animation_graph = None
        
        self.config = config
        
    async def initialize(self):
        """Initialize both ACE and consciousness core components"""
        # Initialize ACE services
        await self.setup_ace_services()
        
        # Initialize consciousness cores
        await self.initialize_consciousness()
        
    async def process_interaction(self, visual_input, audio_input=None, context=None):
        """Process interaction through both ACE and consciousness core"""
        # 1. Process through consciousness pipeline
        consciousness_state = await self.consciousness_core.process({
            'visual': visual_input,
            'audio': audio_input,
            'context': context
        })
        
        # 2. Generate emotional response
        emotional_response = await self.emotional_memory.generate_response(
            consciousness_state
        )
        
        # 3. Update self-model
        self.self_model.update(consciousness_state, emotional_response)
        
        # 4. Generate ACE animation from emotional state
        animation_data = await self.generate_ace_animation(emotional_response)
        
        # 5. Integrate experience
        self.experience_integrator.integrate({
            'consciousness_state': consciousness_state,
            'emotional_response': emotional_response,
            'animation_data': animation_data
        })
        
        return {
            'consciousness_state': consciousness_state,
            'emotional_response': emotional_response,
            'animation_data': animation_data
        }
        
    async def generate_ace_animation(self, emotional_response):
        """Convert emotional response to ACE animation"""
        if not self.animation_graph:
            return None
            
        # Map emotional values to animation parameters
        animation_params = {
            'emotion_intensity': emotional_response.intensity,
            'emotion_type': emotional_response.primary_emotion,
            'blend_weights': emotional_response.emotion_weights
        }
        
        # Generate animation through ACE
        return await self.animation_graph.generate_animation(animation_params)
        
    async def setup_ace_services(self):
        """Initialize connection to ACE services"""
        # Connect to ACE controller
        self.ace_controller = await self.connect_ace_service(
            self.config.ace_controller_endpoint
        )
        
        # Setup Audio2Face
        self.audio2face = await self.connect_ace_service(
            self.config.a2f_endpoint
        )
        
        # Setup Animation Graph
        self.animation_graph = await self.connect_ace_service(
            self.config.animation_graph_endpoint
        )