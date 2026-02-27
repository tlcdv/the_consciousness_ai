# NVIDIA ACE Integration Guide

# ACE Integration with VideoLLaMA3

## Overview

The system project integrates NVIDIA's Avatar Cloud Engine (ACE) to enable realistic avatar animations driven by the AI's emotional and conscious states.

## Components

- **Audio2Face (A2F)**: Real-time facial animation from audio
- **ACE Controller**: Core animation and interaction management
- **Animation Graph**: Emotional state to animation mapping

## Memory Optimization

The integration includes memory optimization features:

- Dynamic frame buffer management
- Resolution optimization
- Memory metric tracking
- ACE animation state optimization

## Configuration

See [ace_integration/a2f_config.yaml](ace_integration/a2f_config.yaml) and [ace_integration/ac_a2f_config.yaml](ace_integration/ac_a2f_config.yaml) for service configuration.

Memory and ACE settings can be configured in `configs/ace_integration.yaml`:

```yaml
memory_optimization:
  max_buffer_size: 32 # Maximum frames to buffer
  cleanup_threshold: 0.8 # When to trigger cleanup
```
