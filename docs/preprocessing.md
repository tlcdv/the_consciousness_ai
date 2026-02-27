# Dataset Preprocessing Guide

This document provides instructions for downloading, preprocessing, and organizing datasets required for the The Consciousness AI project. It assumes that `docs/datasets.md` (a currently planned document) provides the primary list and sources for these datasets.

---

## 1. Downloading Datasets

The datasets used in this project are stored externally to ensure efficient management of large files. Follow these steps to download them:

### Emotion Recognition Datasets

#### **GoEmotions**

1. Visit the [GoEmotions GitHub Repository](https://github.com/google-research/google-research/tree/master/goemotions).
2. Clone the repository or download the dataset directly:
   ```bash
   git clone https://github.com/google-research/google-research.git
   ```
3. Extract the `dataset/` folder from the repository and place it in the `data/emotions/` directory:
   ```bash
   mv google-research/goemotions/data /path/to/your/repo/data/emotions/goemotions
   ```

#### **MELD**

1. Download the dataset from the [MELD Dataset GitHub](https://github.com/declare-lab/MELD):
   ```bash
   wget https://github.com/declare-lab/MELD/raw/master/data/MELD.Raw.zip
   ```
2. Unzip the file:
   ```bash
   unzip MELD.Raw.zip -d /path/to/your/repo/data/emotions/meld
   ```

#### **HEU Emotion**

1. Refer to the [HEU Emotion Dataset page](https://arxiv.org/abs/2007.12519) for access.
2. Follow the instructions to request access or download directly, if available.
3. Place the dataset files in the `data/emotions/heu_emotion/` directory.

---

### Simulation and Interaction Datasets

#### **INTERACTION Dataset**

1. Visit the [INTERACTION Dataset Website](https://interaction-dataset.com/).
2. Register and download the dataset.
3. Place the CSV files in the `data/simulations/interaction_data/` directory.

#### **UE-HRI Dataset**

1. Access the dataset through [UE-HRI GitHub](https://github.com/mjyc/awesome-hri-datasets).
2. Download and extract the dataset to the `data/simulations/ue_hri_data/` directory.

---

## 2. Preprocessing Steps

### Text-Based Emotion Datasets (GoEmotions, MELD)

1. Ensure CSV files are clean and include the following columns:
   - **Text**: The input text.
   - **Label**: The emotion category.
2. Use the preprocessing script (`scripts/utils/preprocess_emotions.py`) to clean and normalize the data:
   ```bash
   python scripts/utils/preprocess_emotions.py --input /path/to/raw/data --output /path/to/processed/data
   ```

### Audio-Visual Emotion Datasets (HEU Emotion)

1. Convert audio files to a uniform format (e.g., WAV, 16 kHz sampling rate) using a tool like FFmpeg:
   ```bash
   ffmpeg -i input.mp4 -ar 16000 output.wav
   ```
2. Ensure facial images are resized and aligned for visual analysis.
3. Use the preprocessing script (`scripts/utils/preprocess_audio_visual.py`) for automated cleaning:
   ```bash
   python scripts/utils/preprocess_audio_visual.py --input /path/to/raw/data --output /path/to/processed/data
   ```

### Simulation Interaction Datasets

1. Normalize interaction logs to include consistent fields like:
   - **Participant ID**
   - **Interaction Type**
   - **Outcome**
2. Use the preprocessing script (`scripts/utils/preprocess_simulations.py`):
   ```bash
   python scripts/utils/preprocess_simulations.py --input /path/to/raw/data --output /path/to/processed/data
   ```

### Reinforcement Learning Datasets

1. Format interaction logs to include:
   - Emotional responses
   - Reward signals
   - State transitions
2. Use preprocessing script:
   ```bash
   python scripts/utils/preprocess_rl_data.py
   ```

---

## 3. Organizing Preprocessed Data

After preprocessing, organize datasets into the following structure:

```
/data
├── emotions
│   ├── goemotions
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   ├── meld
│   │   ├── train.csv
│   │   ├── val.csv
│   │   └── test.csv
│   └── heu_emotion
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── simulations
│   ├── interaction_data
│   │   ├── scenario_1.csv
│   │   └── scenario_2.csv
│   └── ue_hri_data
│       ├── session_1.csv
│       └── session_2.csv
```

---

## Notes

- Ensure all dataset licenses are adhered to.
- Document any custom preprocessing scripts used.
- Validate preprocessed datasets using appropriate testing scripts in `/tests/`.
