# Implementing and refining `simple_tasks.py`

# File: /simulations/scenarios/simple_tasks.py
"""
Simple Tasks Module

Provides a framework for basic tasks in VR simulations to help agents 
develop fundamental skills like navigation, object manipulation, and 
reaction to stimuli.
"""

import random
import logging

class SimpleTask:
    def __init__(self, task_id, description, success_criteria):
        """
        Initialize a simple task.
        Args:
            task_id (str): Unique identifier for the task.
            description (str): Description of the task.
            success_criteria (callable): A function to evaluate task success.
        """
        self.task_id = task_id
        self.description = description
        self.success_criteria = success_criteria
        self.completed = False

    def check_completion(self, agent_state):
        """
        Check if the task is completed based on agent state.
        Args:
            agent_state (dict): The current state of the agent.
        Returns:
            bool: True if the task is completed, False otherwise.
        """
        try:
            self.completed = self.success_criteria(agent_state)
            return self.completed
        except Exception as e:
            logging.error(f"Error in task {self.task_id}: {e}")
            return False


class SimpleTaskManager:
    def __init__(self):
        """
        Manage a collection of simple tasks.
        """
        self.tasks = []

    def add_task(self, task):
        """
        Add a task to the manager.
        Args:
            task (SimpleTask): The task to add.
        """
        self.tasks.append(task)

    def get_incomplete_tasks(self):
        """
        Retrieve all tasks that are not yet completed.
        Returns:
            list: list of incomplete tasks.
        """
        return [task for task in self.tasks if not task.completed]

    def evaluate_tasks(self, agent_state):
        """
        Evaluate all tasks based on the agent state.
        Args:
            agent_state (dict): The current state of the agent.
        """
        for task in self.tasks:
            task.check_completion(agent_state)


# Example Task Definitions
def reach_waypoint(agent_state):
    """
    Success criteria: Agent reaches a specific waypoint.
    Args:
        agent_state (dict): The current state of the agent.
    Returns:
        bool: True if the agent is at the waypoint, False otherwise.
    """
    waypoint = agent_state.get("waypoint", None)
    position = agent_state.get("position", None)
    return position == waypoint


def pick_object(agent_state):
    """
    Success criteria: Agent picks up an object.
    Args:
        agent_state (dict): The current state of the agent.
    Returns:
        bool: True if the agent has picked up the object, False otherwise.
    """
    return agent_state.get("holding_object", False)


# Example Usage
if __name__ == "__main__":
    task_manager = SimpleTaskManager()

    # Define tasks
    task1 = SimpleTask(
        task_id="task_1",
        description="Reach the designated waypoint.",
        success_criteria=reach_waypoint
    )

    task2 = SimpleTask(
        task_id="task_2",
        description="Pick up the target object.",
        success_criteria=pick_object
    )

    # Add tasks to the manager
    task_manager.add_task(task1)
    task_manager.add_task(task2)

    # Simulate an agent state
    agent_state = {
        "position": [5, 5],
        "waypoint": [5, 5],
        "holding_object": True
    }

    # Evaluate tasks
    task_manager.evaluate_tasks(agent_state)

    # Check task statuses
    incomplete_tasks = task_manager.get_incomplete_tasks()
    if incomplete_tasks:
        print(f"Incomplete Tasks: {[task.description for task in incomplete_tasks]}")
    else:
        print("All tasks completed!")
