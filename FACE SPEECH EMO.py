import cv2
import numpy as np
import librosa
import face_recognition
import tensorflow as tf
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import sounddevice as sd
import soundfile as sf
import tempfile
import os
import sys
from datetime import datetime
import threading
import queue
import keyboard
import time

# Initialize face classifier and emotion detection models
face_classifier = cv2.CascadeClassifier(r'C:\Users\aasth\OneDrive\Desktop\Face-recognition\Face-recognition\haarcascade_frontalface_default.xml')
face_emotion_classifier = load_model(r'C:\Users\aasth\OneDrive\Desktop\Face-recognition\Face-recognition\model.h5')
speech_emotion_model = tf.keras.models.load_model(r'C:\Users\aasth\OneDrive\Desktop\Face-recognition\speech_emotion_model.h5')

# Print model information
print("\nSpeech Emotion Model Summary:")
speech_emotion_model.summary()
print("\nModel input shape:", speech_emotion_model.input_shape)

# Labels for emotion detection
face_emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
speech_emotion_labels = ['Neutral', 'Happy', 'Sad', 'Angry', 'Fear', 'Disgust', 'Surprise']

# Global variables
running = True
recording = False
speech_emotion_result = "No speech analyzed yet"
audio_queue = queue.Queue()

# Known faces setup
known_face_encodings = []
known_face_names = ["Aastha", "Ananya", "Aryan"]

# Preload face encodings
known_images = [
    (r"C:\Users\aasth\OneDrive\Desktop\Face-recognition\Face-recognition\images\Aastha.jpg", "Aastha"),
    (r"C:\Users\aasth\OneDrive\Desktop\Face-recognition\Face-recognition\images\Hariyali.jpg", "Ananya"),
    (r"C:\Users\aasth\OneDrive\Desktop\Face-recognition\Face-recognition\images\batman.jpg", "Aryan")
]

for image_path, name in known_images:
    image = face_recognition.load_image_file(image_path)
    encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(encoding)

def preprocess_audio(file_path):
    """Preprocess audio for emotion detection."""
    try:
        # Load the audio file
        y, sr = librosa.load(file_path, sr=16000)
        print(f"Loaded audio shape: {y.shape}")
        
        # Extract MFCC features with 40 coefficients
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        print(f"Initial MFCC shape: {mfcc.shape}")
        
        # Transpose to get (timestep, features) shape
        mfcc = mfcc.T
        
        # Ensure exactly 500 timesteps
        if mfcc.shape[0] > 500:
            mfcc = mfcc[:500, :]
        else:
            # Pad with zeros if less than 500 timesteps
            padding = np.zeros((500 - mfcc.shape[0], 40))
            mfcc = np.vstack((mfcc, padding))
        
        # Add batch dimension
        mfcc = np.expand_dims(mfcc, axis=0)  # Shape becomes (1, 500, 40)
        
        print(f"Final processed shape: {mfcc.shape}")
        return mfcc
    except Exception as e:
        print(f"Error preprocessing audio: {str(e)}")
        return None

def record_audio(duration=3, sample_rate=16000):
    """Record audio from microphone."""
    global recording
    if recording:
        return None
    
    recording = True
    print("\nRecording started... Speak now!")
    
    try:
        # Record audio
        audio_data = sd.rec(int(duration * sample_rate),
                          samplerate=sample_rate,
                          channels=1,
                          blocking=True)
        
        # Create temporary file
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, f'speech_{datetime.now().strftime("%Y%m%d_%H%M%S")}.wav')
        
        # Save recording
        sf.write(temp_path, audio_data, sample_rate)
        print(f"Recording finished and saved!")
        
        return temp_path
    
    except Exception as e:
        print(f"Error during recording: {str(e)}")
        return None
    finally:
        recording = False

def process_audio_queue():
    """Process audio files in the queue."""
    global speech_emotion_result
    while running:
        try:
            if not audio_queue.empty():
                audio_path = audio_queue.get_nowait()
                if audio_path:
                    result = detect_speech_emotion(audio_path)
                    speech_emotion_result = result
                    try:
                        os.remove(audio_path)
                    except:
                        pass
        except queue.Empty:
            pass
        time.sleep(0.1)

# def detect_speech_emotion(audio_path):
#     """Detect emotion from speech audio."""
#     try:
#         # Get preprocessed audio features
#         processed_audio = preprocess_audio(audio_path)
#         if processed_audio is None:
#             return "Error processing audio"
        
#         # Ensure the shape is correct (1, 500, 40)
#         if processed_audio.shape != (1, 500, 40):
#             print(f"Unexpected shape: {processed_audio.shape}")
#             return "Error: incorrect input shape"
        
#         print("Input shape before prediction:", processed_audio.shape)
        
#         # Make prediction
#         predictions = speech_emotion_model.predict(processed_audio, verbose=0)[0]
#         emotion = speech_emotion_labels[np.argmax(predictions)]
#         confidence = float(np.max(predictions))
#         return f"{emotion} ({confidence:.2%})"
#     except Exception as e:
#         print(f"Error detecting speech emotion: {str(e)}")
#         return "Error detecting emotion"
def detect_speech_emotion(audio_path):
    try:
        processed_audio = preprocess_audio(audio_path)
        if processed_audio is None:
            return "Error processing audio"
        
        print("Input shape before prediction:", processed_audio.shape)
        
        predictions = speech_emotion_model.predict(processed_audio, verbose=0)[0]
        print("Raw predictions:", predictions)
        print("Prediction shape:", predictions.shape)
        
        max_index = np.argmax(predictions)
        print("Predicted index:", max_index)
        print("Number of labels:", len(speech_emotion_labels))
        print("Labels:", speech_emotion_labels)
        
        if max_index < 0 or max_index >= len(speech_emotion_labels):
            return f"Error: Predicted index {max_index} out of range (0-{len(speech_emotion_labels)-1})"
        
        emotion = speech_emotion_labels[max_index]
        confidence = float(np.max(predictions))
        return f"{emotion} ({confidence:.2%})"
    except Exception as e:
        print(f"Error detecting speech emotion: {str(e)}")
        import traceback
        traceback.print_exc()
        return "Error detecting emotion"

def handle_keyboard():
    """Handle keyboard inputs."""
    global running
    while running:
        try:
            if keyboard.is_pressed('s') and not recording:
                audio_path = record_audio()
                if audio_path:
                    audio_queue.put(audio_path)
            elif keyboard.is_pressed('q'):
                running = False
                break
        except Exception as e:
            print(f"Keyboard handler error: {str(e)}")
        time.sleep(0.1)

# def main():
#     global running, speech_emotion_result
    
#     # Start keyboard handler thread
#     keyboard_thread = threading.Thread(target=handle_keyboard)
#     keyboard_thread.daemon = True
#     keyboard_thread.start()
    
#     # Start audio processing thread
#     audio_thread = threading.Thread(target=process_audio_queue)
#     audio_thread.daemon = True
#     audio_thread.start()
    
#     # Initialize webcam
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         print("Error: Could not open webcam")
#         running = False
#         return

#     print("\nControls:")
#     print("Press 's' to start speech recording")
#     print("Press 'q' to quit")

#     try:
#         while running:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
#             # Detect faces
#             face_locations = face_recognition.face_locations(frame)
#             face_encodings = face_recognition.face_encodings(frame, face_locations)
            
#             for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#                 matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#                 name = "Unknown"
#                 if True in matches:
#                     first_match_index = matches.index(True)
#                     name = known_face_names[first_match_index]

#                 # Face Emotion Detection
#                 roi_gray = gray[top:bottom, left:right]
#                 try:
#                     roi_gray = cv2.resize(roi_gray, (48, 48))
#                     roi = roi_gray.astype('float') / 255.0
#                     roi = img_to_array(roi)
#                     roi = np.expand_dims(roi, axis=0)
#                     predictions = face_emotion_classifier.predict(roi, verbose=0)[0]
#                     face_emotion = face_emotion_labels[np.argmax(predictions)]
                    
#                     # Display face name and emotion
#                     cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
#                     cv2.putText(frame, f"{name} - {face_emotion}", (left, top - 10),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#                 except cv2.error:
#                     pass

#             # Display speech emotion result
#             cv2.putText(frame, f"Speech Emotion: {speech_emotion_result}", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
#             # Display recording status
#             if recording:
#                 cv2.putText(frame, "Recording...", (10, 60),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
#             cv2.imshow('Face and Emotion Recognition', frame)
            
#             # Check for window close button
#             if cv2.getWindowProperty('Face and Emotion Recognition', cv2.WND_PROP_VISIBLE) < 1:
#                 break
            
#             cv2.waitKey(1)

#     except Exception as e:
#         print(f"Error in main loop: {str(e)}")
    
#     finally:
#         running = False
#         cap.release()
#         cv2.destroyAllWindows()
#         # Wait for threads to finish
#         keyboard_thread.join(timeout=1)
#         audio_thread.join(timeout=1)
#         print("\nProgram terminated.")
def main():
    global running, speech_emotion_result, speech_emotion_model

    # Load the speech emotion model
    try:
        speech_emotion_model = tf.keras.models.load_model(r'C:\Users\aasth\OneDrive\Desktop\Face-recognition\speech_emotion_model.h5')
        print("\nSpeech Emotion Model Summary:")
        speech_emotion_model.summary()
        print("\nModel input shape:", speech_emotion_model.input_shape)
        print("Model output shape:", speech_emotion_model.output_shape)
        print("Number of emotion labels:", len(speech_emotion_labels))
        if speech_emotion_model.output_shape[-1] != len(speech_emotion_labels):
            print("Warning: Mismatch between model output and number of labels")
    except Exception as e:
        print(f"Error loading speech emotion model: {str(e)}")
        return

    # Start keyboard handler thread
    keyboard_thread = threading.Thread(target=handle_keyboard)
    keyboard_thread.daemon = True
    keyboard_thread.start()
    
    # Start audio processing thread
    audio_thread = threading.Thread(target=process_audio_queue)
    audio_thread.daemon = True
    audio_thread.start()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        running = False
        return

    print("\nControls:")
    print("Press 's' to start speech recording")
    print("Press 'q' to quit")

    try:
        while running:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            face_locations = face_recognition.face_locations(frame)
            face_encodings = face_recognition.face_encodings(frame, face_locations)
            
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                if True in matches:
                    first_match_index = matches.index(True)
                    name = known_face_names[first_match_index]

                # Face Emotion Detection
                roi_gray = gray[top:bottom, left:right]
                try:
                    roi_gray = cv2.resize(roi_gray, (48, 48))
                    roi = roi_gray.astype('float') / 255.0
                    roi = img_to_array(roi)
                    roi = np.expand_dims(roi, axis=0)
                    predictions = face_emotion_classifier.predict(roi, verbose=0)[0]
                    face_emotion = face_emotion_labels[np.argmax(predictions)]
                    
                    # Display face name and emotion
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
                    cv2.putText(frame, f"{name} - {face_emotion}", (left, top - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                except cv2.error:
                    pass

            # Display speech emotion result
            cv2.putText(frame, f"Speech Emotion: {speech_emotion_result}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display recording status
            if recording:
                cv2.putText(frame, "Recording...", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Face and Emotion Recognition', frame)
            
            # Check for window close button
            if cv2.getWindowProperty('Face and Emotion Recognition', cv2.WND_PROP_VISIBLE) < 1:
                break
            
            cv2.waitKey(1)

    except Exception as e:
        print(f"Error in main loop: {str(e)}")
    
    finally:
        running = False
        cap.release()
        cv2.destroyAllWindows()
        # Wait for threads to finish
        keyboard_thread.join(timeout=1)
        audio_thread.join(timeout=1)
        print("\nProgram terminated.")


if __name__ == "__main__":
    main()