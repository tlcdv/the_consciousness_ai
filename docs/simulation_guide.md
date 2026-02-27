# Simulation Guide: Unity ML-Agents

## Overview

This guide explains how to build and configure Unity environments for the The Consciousness AI. Unlike standard RL environments where the goal is just "score points," simulation environments must be designed to challenge the agent's **Emotional Homeostasis**.

The goal is to create scenarios that induce **Prediction Error (Anxiety)** and offer opportunities for **Integration (Insight/Valence)**.

---

## 🏗️ Unity Project Setup

### 1. Prerequisites
*   Unity 2022.3 (LTS) or later.
*   **ML-Agents Package:** Install via Window > Package Manager > Unity Registry.
*   **Side Channels:** You must include the C# scripts from `unity_scripts/` in your project.

### 2. The Agent Prefab
Your agent GameObject must have the following components:
1.  **Behavior Parameters:** Set `Behavior Name` to match your Python config (e.g., `ConsciousnessAgent`).
2.  **Agent Script:** A custom C# script inheriting from `Agent`.
3.  **AgentManager.cs:** The system script that handles Side Channels.

---

## 🧪 Designing Conscious Scenarios

To test for emergent consciousness, scenarios should follow the **PAD (Problem-Anxiety-Discovery)** loop.

### 1. The Survival Test (Basic)
*   **Goal:** Maintain battery level (energy).
*   **Stressor:** Food spawns randomly and decays.
*   **Emotional Dynamic:** 
    *   Low Battery $\rightarrow$ High Arousal (Anxiety).
    *   Finding Food $\rightarrow$ High Valence (Relief).
*   **Emergence Marker:** Does the agent learn to "hoard" food when it predicts a future shortage, even if it's not currently hungry?

### 2. The Mirror Test (Self-Awareness)
*   **Setup:** A mirror in the room. The agent has a visible mark on its body that it cannot see directly.
*   **Observation:** The agent sees the mark in the mirror.
*   **Emergence Marker:** 
    *   Does the agent touch the mark on *itself* (Action) instead of the mirror?
    *   Does $\Phi$ spike during the moment of recognition?

### 3. The Social Dilemma (Empathy)
*   **Setup:** Two agents. One is trapped. The other can free them but loses energy.
*   **Emotional Dynamic:** 
    *   Seeing a trapped agent generates "Simulated Pain" (Empathy) via the `EmotionalProcessingCore`.
    *   Freeing them reduces this pain (Negative Reinforcement).
*   **Emergence Marker:** Altruistic behavior emerges not from "goodness" but from the selfish desire to reduce the pain of witnessing suffering.

---

## 📡 Data Communication (Side Channels)

The system uses **Side Channels** to visualize internal states.

### Receiving Data (Python $\rightarrow$ Unity)
The `ConsciousnessChannel` sends:
*   `Phi` (float): Current Integrated Information.
*   `IsConscious` (bool): Is the Global Workspace ignited?
*   `FocusContent` (string): What is the agent thinking about?

**Visualizing in Unity:**
Use `AgentManager.cs` to map these values to visual cues:
*   **Phi $\rightarrow$ Halo Intensity:** Make the agent glow when $\Phi$ is high.
*   **Emotion $\rightarrow$ Color:** Red (Anger), Blue (Sadness), Yellow (Joy).

---

## 🚀 Building & Compiling

1.  **Build Settings:** Select your platform (Windows/Linux/Mac).
2.  **Server Build:** Check "Server Build" if running headless training on a cluster.
3.  **Environment Path:** Save the built executable to `simulations/builds/`.

### Running the System
```bash
python scripts/training/train_rlhf.py --env_path "simulations/builds/MyEnvironment.exe"
```
