import librosa
import sounddevice as sd
import numpy as np
from keras.models import load_model
import queue

# Load the speech emotion recognition model
speech_emotion_model = load_model(r'C:\Users\aasth\Downloads\Face-recognition\speech_emotion_model.h5')
speech_emotion_labels = ['Angry', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Audio recording parameters
sample_rate = 22050  # Sample rate for audio recording
duration = 5  # Duration of recording in seconds
buffer_queue = queue.Queue()

def record_audio(duration, sample_rate):
    """Record audio for a given duration and sample rate."""
    print("Recording...")
    audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until the recording is finished
    print("Recording complete.")
    return audio.flatten()

def extract_features(audio, sample_rate):
    """Extract features from audio for emotion prediction."""
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs = np.mean(mfccs.T, axis=0)
    return np.expand_dims(mfccs, axis=0)

def predict_speech_emotion(audio, sample_rate):
    """Predict emotion from audio."""
    features = extract_features(audio, sample_rate)
    prediction = speech_emotion_model.predict(features)
    emotion = speech_emotion_labels[np.argmax(prediction)]
    return emotion

# Webcam capture and speech emotion detection
cap = cv2.VideoCapture(0)

while True:
    # Capture video frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale for emotion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    # Process each face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Identify person
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        # Detect facial emotion
        roi_gray = gray[top:bottom, left:right]
        try:
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = emotion_classifier.predict(roi)[0]
                face_emotion = emotion_labels[prediction.argmax()]

                # Draw rectangle and labels
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
                cv2.putText(frame, f"{name}, {face_emotion}", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        except cv2.error:
            pass

    # Record and predict speech emotion every few frames
    if buffer_queue.qsize() == 0:
        audio = record_audio(duration, sample_rate)
        speech_emotion = predict_speech_emotion(audio, sample_rate)
        buffer_queue.put(speech_emotion)

    # Overlay speech emotion on frame
    if not buffer_queue.empty():
        speech_emotion = buffer_queue.get()
        cv2.putText(frame, f"Speech Emotion: {speech_emotion}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # Display the frame
    cv2.imshow('Multimodal Emotion Detection', frame)

    # Exit loop on 'q' press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()