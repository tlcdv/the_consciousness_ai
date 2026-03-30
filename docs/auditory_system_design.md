# Biologically Inspired Auditory Processing System

## 1. Motivation

The project models consciousness emergence through biologically grounded perception. The visual pipeline (DINOv2 retinotopic encoder, topographic maps, inverse effectiveness fusion) mirrors the structure of the mammalian visual system from retina to primary visual cortex. However, audio processing was limited to a Faster Whisper transcription stub, which has no biological correspondence. Human auditory perception does not start with text. It starts at the cochlea.

This document describes the design of a cochlear inspired auditory pipeline that replaces Whisper transcription with a biophysically grounded processing chain: gammatone filterbank, inner hair cell model, tonotopic encoder, acoustic affect extraction, and workspace integration. The system is open source (MIT/BSD dependencies only), runs on average hardware, and produces features aligned with human auditory perception.

## 2. Biological Basis

### 2.1 Cochlear Frequency Decomposition

The cochlea is a fluid filled spiral structure in the inner ear. Sound pressure waves enter at the base and propagate along the basilar membrane. Each location on the membrane resonates at a characteristic frequency: high frequencies near the base (~20 kHz), low frequencies at the apex (~20 Hz). This creates a frequency to place mapping called **tonotopy**.

The bandwidth of each cochlear filter increases with center frequency, following the **Equivalent Rectangular Bandwidth (ERB)** scale (Glasberg & Moore 1990):

```
ERB(f) = 24.7 * (4.37 * f/1000 + 1)
```

The **gammatone filter** (Patterson et al. 1992) is the standard linear approximation of basilar membrane filtering. It captures the impulse response shape (a gamma distribution envelope modulating a sinusoidal carrier) and the frequency dependent bandwidth.

**Implementation**: `models/audio/gammatone_filterbank.py` uses 64 4th order gammatone filters on the ERB scale, computed via `scipy.signal.gammatone` and stored as a frozen `nn.Conv1d` weight buffer. All parameters are frozen, paralleling DINOv2 in the visual pathway. The cochlea's physical structure does not change during learning.

### 2.2 Inner Hair Cell Transduction

Inner hair cells (IHCs) sit on the basilar membrane and convert mechanical vibration into neural signals via mechanoelectric transduction. The output has two components:

1. **Envelope** (rate code): The slowly varying amplitude of each frequency band. Obtained by half wave rectification (IHCs respond asymmetrically) and low pass filtering (~50 Hz cutoff). Drives auditory nerve fiber firing rate. Critical for loudness perception and amplitude modulation detection.

2. **Temporal Fine Structure (TFS)**: The fast oscillatory component that preserves phase information. Critical for pitch, sound localization (ITD), and timbre perception.

Reference: Joris et al. 2004 ("Neural coding in auditory nerve and cochlear nucleus").

**Implementation**: `models/audio/hair_cell_model.py` applies `F.relu()` (half wave rectification) followed by temporal smoothing via average pooling. TFS is the residual: input minus smoothed envelope.

### 2.3 Tonotopic Organization

Tonotopy is preserved from cochlea through the auditory nerve, brainstem nuclei (cochlear nucleus, superior olivary complex, inferior colliculus), medial geniculate body, to primary auditory cortex (A1). In A1, neurons are arranged along isofrequency contours: adjacent cortical locations respond to adjacent frequencies (Romani et al. 1982, Merzenich & Reid 1974).

This is the auditory analog of retinotopy in the visual system. Just as DINOv2 patch tokens at grid position (i,j) correspond to the 14x14 pixel region at (i*14, j*14) in the input image, tonotopic encoder outputs at position k correspond to the k-th ERB frequency band.

**Implementation**: `models/audio/tonotopic_encoder.py` is a trainable 1D conv stack (analogous to `RetinotopicConvStack`) that processes the hair cell output and produces a compact feature map `[B, 64, 16]` where the 16 positions correspond to 16 frequency bands. For tectum integration, this reshapes to `[B, 64, 16, 16]` where frequency maps to the elevation axis and azimuth is handled by spatial audio placement.

### 2.4 Sound Localization

Binaural sound localization in mammals relies on two primary cues computed in the brainstem (Grothe et al. 2010):

- **ITD** (Interaural Time Difference): Computed by the medial superior olive (MSO) via coincidence detection neurons. The MSO receives inputs from both ears and fires maximally when signals arrive simultaneously, corresponding to a particular azimuth. Dominant below ~1500 Hz where phase locking occurs.

- **ILD** (Interaural Level Difference): Computed by the lateral superior olive (LSO) via excitatory inhibitory comparison. The head shadows high frequency sounds, creating a level difference between ears that maps to azimuth. Dominant above ~1500 Hz.

Elevation is estimated from spectral cues created by pinna filtering, which is harder to model and typically provided by the environment.

**Implementation**: `models/audio/spatial_audio.py` computes azimuth from ITD (cross correlation peak) and ILD (energy ratio) for stereo input. Mono input defaults to center. Environment metadata can override.

### 2.5 Auditory Emotion and Affect

Emotional responses to sound are driven by acoustic features, not semantic content. Studies show consistent cross cultural mappings between acoustic properties and perceived emotion (Juslin & Laukka 2003, Eerola & Vuoskoski 2013):

| Acoustic Feature | High Value | Low Value |
|---|---|---|
| **Spectral centroid** (brightness) | Alert, tense, angry | Calm, sad, tender |
| **Loudness variability** | Exciting, dynamic | Calm, predictable |
| **Roughness** (15-300 Hz AM) | Threatening, tense, dissonant | Pleasant, consonant |
| **Pitch contour slope** | Questioning (rising), urgent | Declarative (falling), calm |
| **Spectral flux** | Novel, surprising, active | Familiar, stable |
| **Harmonic to noise ratio** | Tonal, pleasant (high) | Breathy, rough, screams (low) |

The auditory startle reflex (Davis 1984) is one of the fastest neural pathways for threat detection: inferior colliculus to amygdala in ~15ms, much faster than cortical processing. This justifies adding audio to the THREAT_MODULES set in the affective modulator.

**Paralinguistic vocalizations** (laughter, crying, screaming, growling, sighing) are detected from the same acoustic features. Schuller et al. 2013 (ComParE challenge) and Eyben et al. 2010 (openSMILE) showed that spectral and prosodic features suffice for paralinguistic classification without speech recognition.

**Implementation**: `models/audio/audio_affect_extractor.py` extracts 6 features, maps them to PAD (Pleasure, Arousal, Dominance) via a trainable MLP, and classifies 7 paralinguistic categories.

### 2.6 Multisensory Integration in the Superior Colliculus

The superior colliculus (tectum) contains aligned spatial maps for vision, audition, and somatosensation (Stein & Meredith 1993). Audio is represented as a spatial map where frequency runs along one axis (isofrequency contours) and azimuth runs along the orthogonal axis (Merzenich & Reid 1974). The existing `TopographicMap` class already implements inverse effectiveness fusion for visual audio interaction:

- When both visual and audio signals at a grid cell are weak, proportional enhancement is large.
- When both are strong, enhancement is modest.
- This follows from the sigmoid response function of SC neurons.

The audio channel in the tectum handles **spatial orienting** (where is the sound?). Content analysis (what is the sound?) goes through the auditory cortex pathway, which here corresponds to the `AuditorySpecialist` competing in the Global Workspace.

## 3. Architecture

### 3.1 Pipeline

```
Raw waveform [B,1,T] (16 kHz mono)
  -> GammatoneFilterbank (64 ERB bands, frozen)
  -> [B, 64, T_frames] cochleagram
  -> HairCellModel (envelope + TFS extraction)
  -> [B, 128, T_frames]
  -> TonotopicEncoder (trainable 1D conv stack)
  -> [B, 64, 16] tonotopic features
  -> AuditorySpecialist:
      a) SpatialAudioComputer -> [B, 64, 2] for tectum IE fusion
      b) AudioAffectExtractor -> PAD deltas + paralinguistic class
      c) Linear projection -> [B, 256] workspace content, scalar bid
```

### 3.2 Integration Points

1. **Tectum (spatial)**: `get_spatial_for_tectum()` returns `[B, feature_dim, 2]` for `TopographicMap._place_audio_on_grid()` and inverse effectiveness fusion with vision and somatosensation.

2. **Global Workspace (content)**: Audio is oscillator #2 in the 5 module Kuramoto binding system. The specialist submits a content tensor and scalar salience bid. The bid is modulated by the affective modulator (audio is now in THREAT_MODULES).

3. **Reentrant feedback**: `receive_broadcast()` computes prediction error between broadcast content and last audio content. High PE increases bid (audio was not attended), low PE settles.

4. **Emotion (affect)**: `get_affect_output()` provides PAD deltas and paralinguistic class to the two stage appraisal system. Stage 1 (reflex): spectral flux drives arousal, roughness drives negative valence. Stage 2 (appraisal): integrated through the phenomenological mapper.

5. **Environment (synthesis)**: `AudioMixin` generates FM/ADSR synthesized audio per step based on environment state. Each environment type has specialized audio mappings (proximity tones, collision bursts, reward jingles, warning roughness).

### 3.3 Graceful Degradation

When `--enable-audio` is not set (default), no auditory specialist is instantiated. The training loop falls back to zero tensors for audio spatial and zero bids, exactly matching the previous behavior. All 465+ existing tests continue to pass unmodified.

## 4. Design Decisions

### 4.1 Gammatone over Mel Spectrogram

Mel spectrograms are an engineering approximation motivated by perceptual frequency scaling. Gammatone filters model the actual impulse response of basilar membrane filters (Patterson et al. 1992). They preserve temporal fine structure, have biologically grounded bandwidth (ERB), and produce a natural tonotopic map. For a project modeling consciousness emergence from biological principles, the mechanistic correspondence matters.

### 4.2 Frozen Filterbank, Trainable Downstream

Parallels DINOv2 (frozen) in the visual pipeline. The cochlea's physical structure does not change during learning in mammals. Cortical processing (auditory cortex, represented by the tonotopic encoder and affect extractor) adapts. This provides a strong inductive bias: the filterbank encodes physics (frequency decomposition), downstream modules learn to use those features for the task.

### 4.3 Separate Specialist, Not Folded Into Tectum

Biologically, the superior colliculus handles spatial audio (where), auditory cortex handles content (what). These are parallel pathways. The tectum gets spatial features for grid placement and IE fusion. The auditory specialist separately submits content to the workspace. This mirrors the dual "what/where" auditory streams in the brain (Rauschecker & Tian 2000).

### 4.4 64 Bands Pooled to 16

64 bands provide ~1 ERB per band, matching human cochlear frequency resolution. Pooled to 16 for tectum grid compatibility (grid_size=16). The full 64 band representation is used internally for affect extraction where roughness and pitch analysis need fine frequency resolution.

### 4.5 Audio in THREAT_MODULES

The auditory startle reflex is mediated by a subcortical pathway (cochlear nucleus to reticular formation to motor neurons, with amygdala modulation) that operates in ~15ms. This is faster than any cortical processing. Adding audio to THREAT_MODULES means that negative emotional valence boosts the audio bid, enabling the agent to prioritize sudden loud or rough sounds for conscious access. This is biologically correct and functionally useful for threat detection.

## 5. References

- Belin, P., Fecteau, S., & Bedard, C. (2004). Thinking the voice: neural correlates of voice perception. Trends in Cognitive Sciences, 8(3), 129-135.
- Davis, M. (1984). The mammalian startle response. In Neural Mechanisms of Startle Behavior (pp. 287-351). Plenum Press.
- Eerola, T., & Vuoskoski, J. K. (2013). A review of music and emotion studies: approaches, emotion models, and stimuli. Music Perception, 30(3), 307-340.
- Eyben, F., Wollmer, M., & Schuller, B. (2010). openSMILE: the Munich versatile and fast open-source audio feature extractor. Proceedings of ACM Multimedia, 1459-1462.
- Glasberg, B. R., & Moore, B. C. (1990). Derivation of auditory filter shapes from notched-noise data. Hearing Research, 47(1-2), 103-138.
- Grothe, B., Pecka, M., & McAlpine, D. (2010). Mechanisms of sound localization in mammals. Physiological Reviews, 90(3), 983-1012.
- Joris, P. X., Schreiner, C. E., & Rees, A. (2004). Neural processing of amplitude-modulated sounds. Physiological Reviews, 84(2), 541-577.
- Juslin, P. N., & Laukka, P. (2003). Communication of emotions in vocal expression and music performance: different channels, same code? Psychological Bulletin, 129(5), 770-814.
- Koch, M. (1999). The neurobiology of startle. Progress in Neurobiology, 59(2), 107-128.
- McAdams, S., & Giordano, B. L. (2009). The perception of musical timbre. In Oxford Handbook of Music Psychology. Oxford University Press.
- Merzenich, M. M., & Reid, M. D. (1974). Representation of the cochlea within the inferior colliculus of the cat. Brain Research, 77(3), 397-415.
- Ohshiro, T., Angelaki, D. E., & DeAngelis, G. C. (2011). A normalization model of multisensory integration. Nature Neuroscience, 14(6), 775-782.
- Patterson, R. D., Robinson, K., Holdsworth, J., McKeown, D., Zhang, C., & Allerhand, M. (1992). Complex sounds and auditory images. In Auditory Physiology and Perception (pp. 429-446). Pergamon.
- Rauschecker, J. P., & Tian, B. (2000). Mechanisms and streams for processing of "what" and "where" in auditory cortex. PNAS, 97(22), 11800-11806.
- Romani, G. L., Williamson, S. J., & Kaufman, L. (1982). Tonotopic organization of the human auditory cortex. Science, 216(4552), 1339-1340.
- Schuller, B., Steidl, S., Batliner, A., et al. (2013). The INTERSPEECH 2013 computational paralinguistics challenge: social signals, conflict, emotion, autism. Proceedings of INTERSPEECH, 148-152.
- Stein, B. E., & Meredith, M. A. (1993). The Merging of the Senses. MIT Press.
- Vassilakis, P. N. (2005). Auditory roughness as a means of musical expression. Selected Reports in Ethnomusicology, 12, 119-144.
