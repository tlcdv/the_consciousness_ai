# Installation & Hardware Guide

## 🖥️ Hardware Requirements

The system is a high-performance research framework. It runs "Deep Learning heavy" workloads, fusing Vision Transformers (Qwen2-VL) with Reinforcement Learning in real-time.

### **Recommended Workstation**
*   **GPU:** NVIDIA RTX 4090 (24GB VRAM).
    *   *Why?* The Vision Model (Qwen2-VL-7B) takes ~6GB in 4-bit mode. The remaining 18GB is critical for the Replay Buffer, World Model (Dreamer), and batch processing during training.
*   **CPU:** AMD Ryzen 9 / Intel Core i9 (16+ cores).
*   **RAM:** 64GB DDR5 (Minimum 32GB).
*   **Storage:** 2TB NVMe SSD (Datasets and Replay Logs grow fast).

### **Minimum Requirements**
*   **GPU:** NVIDIA RTX 3060 (12GB VRAM).
    *   *Note:* You must use 4-bit quantization and reduce batch sizes.
*   **RAM:** 32GB.

---

## 🔧 Software Setup

### 1. Environment
We recommend **Conda** to manage the complex CUDA dependencies.

```bash
# 1. Create Environment
conda create -n acm_lab python=3.10
conda activate acm_lab

# 2. Install PyTorch with CUDA 12.1 (Compatible with RTX 40-series)
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 3. Clone & Install Dependencies
git clone https://github.com/tlcdv/the_consciousness_ai.git
cd the_consciousness_ai
pip install -r requirements.txt
```

### 2. External Libraries
*   **Flash Attention 2:** Highly recommended for Qwen2-VL speed.
    ```bash
    pip install flash-attn --no-build-isolation
    ```
*   **PyGame:** For the lightweight simulation environment.
    ```bash
    pip install pygame gymnasium
    ```

---

## 🚀 Verifying the Setup

Run the diagnostic script to ensure your RTX 4090 is being utilized correctly:

```bash
python scripts/utils/verify_gpu.py
```
*(Note: Create this script if it doesn't exist)*
