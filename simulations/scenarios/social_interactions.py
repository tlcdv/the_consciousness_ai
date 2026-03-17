# Refining `social_interactions.py`

# File: /simulations/scenarios/social_interactions.py
"""
Social Interactions Module

Simulates complex social scenarios to teach agents empathy, negotiation, and collaboration.
Includes predefined interaction scripts and dynamic multimodal inputs.
"""

import random
import logging

class SocialInteraction:
    def __init__(self, interaction_id, participants, scenario, success_criteria):
        """
        Initialize a social interaction.
        Args:
            interaction_id (str): Unique identifier for the interaction.
            participants (list): list of participant IDs (agents or humans).
            scenario (str): Description of the social scenario.
            success_criteria (callable): A function to evaluate interaction success.
        """
        self.interaction_id = interaction_id
        self.participants = participants
        self.scenario = scenario
        self.success_criteria = success_criteria
        self.completed = False

    def evaluate_interaction(self, interaction_state):
        """
        Evaluate the success of the social interaction.
        Args:
            interaction_state (dict): Current state of the interaction.
        Returns:
            bool: True if the interaction is successful, False otherwise.
        """
        try:
            self.completed = self.success_criteria(interaction_state)
            return self.completed
        except Exception as e:
            logging.error(f"Error in interaction {self.interaction_id}: {e}")
            return False


class SocialInteractionManager:
    def __init__(self):
        """
        Manage a collection of social interactions.
        """
        self.interactions = []

    def add_interaction(self, interaction):
        """
        Add a social interaction to the manager.
        Args:
            interaction (SocialInteraction): The interaction to add.
        """
        self.interactions.append(interaction)

    def evaluate_interactions(self, interaction_state):
        """
        Evaluate all social interactions based on the interaction state.
        Args:
            interaction_state (dict): Current state of all interactions.
        """
        for interaction in self.interactions:
            interaction.evaluate_interaction(interaction_state)

# Example Interaction Definitions
def negotiation_success(interaction_state):
    """
    Success criteria: Participants reach an agreement.
    Args:
        interaction_state (dict): Current state of the interaction.
    Returns:
        bool: True if an agreement is reached, False otherwise.
    """
    return interaction_state.get("agreement_reached", False)


def empathy_test_success(interaction_state):
    """
    Success criteria: Agent shows appropriate empathy.
    Args:
        interaction_state (dict): Current state of the interaction.
    Returns:
        bool: True if empathy is demonstrated, False otherwise.
    """
    return interaction_state.get("empathy_displayed", False)


# Example Usage
if __name__ == "__main__":
    interaction_manager = SocialInteractionManager()

    # Define interactions
    interaction1 = SocialInteraction(
        interaction_id="interaction_1",
        participants=["agent_1", "human_1"],
        scenario="Negotiate resource allocation.",
        success_criteria=negotiation_success
    )

    interaction2 = SocialInteraction(
        interaction_id="interaction_2",
        participants=["agent_2", "human_2"],
        scenario="Comfort a distressed participant.",
        success_criteria=empathy_test_success
    )

    # Add interactions to the manager
    interaction_manager.add_interaction(interaction1)
    interaction_manager.add_interaction(interaction2)

    # Simulate interaction states
    interaction_state = {
        "agreement_reached": True,
        "empathy_displayed": True
    }

    # Evaluate interactions
    interaction_manager.evaluate_interactions(interaction_state)

    # Check interaction statuses
    for interaction in interaction_manager.interactions:
        print(f"Interaction {interaction.interaction_id} completed: {interaction.completed}")
