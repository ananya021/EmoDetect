# EmoDetect: Emotion Recognition Using Facial and Speech Analysis
EmoDetect is a multimodal emotion recognition system that analyzes and classifies emotions based on facial expressions and speech. This project leverages advanced artificial intelligence techniques to provide robust and real-time emotion detection capabilities.

# Features
1. Speech-Based Emotion Recognition: Utilizes advanced speech feature extraction techniques like MFCCs, pitch variations, and spectral contrast for identifying emotions from audio.
2. Facial Emotion Recognition: Employs Convolutional Neural Networks (CNNs) to analyze facial expressions using grayscale image datasets.
3. Multimodal Integration: Combines speech and facial data to improve recognition accuracy and reliability.
Real-Time Processing: Enables live emotion recognition with optimized deep learning architectures.

# Datasets
1. TESS Dataset (Speech):
Contains recordings of 7 emotions: happiness, sadness, anger, fear, disgust, surprise, and neutral.
High-quality WAV files sampled at 16 kHz.
2. FER-2013 Dataset (Facial):
Includes 48x48 grayscale images covering 7 emotion classes.
A total of 35,887 labeled images.

# Tech Stack
1. **Programming Language:** Python
2. **Deep Learning Frameworks:** TensorFlow, Keras
3. **Libraries for Audio and Image Processing:**
    Librosa (audio feature extraction)
    OpenCV (image processing)
4. **Machine Learning Models:**
   Long Short-Term Memory (LSTM) networks for speech emotion analysis.
    Convolutional Neural Networks (CNNs) for facial emotion analysis.
    System Architecture
5. **Input:** Speech audio and facial images.
6. **Feature Extraction:** Extracts key features from speech (e.g., MFCCs) and facial data (hierarchical features using CNNs).
6. **Model Processing:**
    LSTM for speech data.
    CNN for facial data.
7. **Emotion Classification:** Combines results from both modalities for a final prediction.
8. **Output:** Real-time emotion classification with high accuracy.

# Results
1. Speech Analysis Model: Effective in identifying emotional tones from high-quality audio.
2. Facial Analysis Model: Successfully classifies emotions based on grayscale images of facial expressions.

# Applications
1. Mental Health Monitoring: Detect emotions for therapeutic insights.
2. Customer Service: Improve user interaction by analyzing emotions in real-time.
3. Surveillance and Security: Detect stress or fear in sensitive environments.
4. Human-Computer Interaction: Enhance user experience by adapting systems to emotional states.

# Future Scope
1. Expand datasets to include more diverse and naturalistic emotions.
2. Integrate additional modalities like physiological signals.
3. Optimize for deployment on mobile and IoT devices for broader accessibility.
