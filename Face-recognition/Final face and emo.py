# import cv2
# import face_recognition

# # Load known face encodings and names
# known_face_encodings = []
# known_face_names = []

# # Load known faces and their names here
# known_person1_image = face_recognition.load_image_file(r"C:\Users\aasth\Downloads\Face-recognition\Face-recognition\images\Aastha.jpg")
# known_person2_image = face_recognition.load_image_file(r"C:\Users\aasth\Downloads\Face-recognition\Face-recognition\images\Hariyali.jpg")
# known_person3_image = face_recognition.load_image_file(r"C:\Users\aasth\Downloads\Face-recognition\Face-recognition\images\nalla.jpg")

# known_person1_encoding = face_recognition.face_encodings(known_person1_image)[0]
# known_person2_encoding = face_recognition.face_encodings(known_person2_image)[0]
# known_person3_encoding = face_recognition.face_encodings(known_person3_image)[0]

# known_face_encodings.append(known_person1_encoding)
# known_face_encodings.append(known_person2_encoding)
# known_face_encodings.append(known_person3_encoding)

# known_face_names.append("Aastha")
# known_face_names.append("Ananya")
# known_face_names.append("Suraj")

# # Initialize webcam
# video_capture = cv2.VideoCapture(0)

# while True:
#     # Capture frame-by-frame
#     ret, frame = video_capture.read()

#     # Find all face locations in the current frame
#     face_locations = face_recognition.face_locations(frame)
#     face_encodings = face_recognition.face_encodings(frame, face_locations)

#     # Loop through each face found in the frame
#     for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
#         # Check if the face matches any known faces
#         matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
#         name = "Unknown"

#         if True in matches:
#             first_match_index = matches.index(True)
#             name = known_face_names[first_match_index]

#         # Draw a box around the face and label with the name
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
#         cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

#     # Display the resulting frame
#     cv2.imshow("Video", frame)

#     # Break the loop when the 'q' key is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release the webcam and close OpenCV windows
# video_capture.release()
# cv2.destroyAllWindows()

import cv2
import numpy as np
import face_recognition
from keras.models import load_model
from keras.preprocessing.image import img_to_array

# Initialize face classifier and emotion detection model
face_classifier = cv2.CascadeClassifier(r'C:\Users\aasth\Downloads\Face-recognition\Face-recognition\haarcascade_frontalface_default.xml')
emotion_classifier = load_model(r'C:\Users\aasth\Downloads\Face-recognition\Face-recognition\model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load known face encodings and names
known_face_encodings = []
known_face_names = []

# Load known faces and their names
known_person1_image = face_recognition.load_image_file(r"C:\Users\aasth\Downloads\Face-recognition\Face-recognition\images\Aastha.jpg")
known_person2_image = face_recognition.load_image_file(r"C:\Users\aasth\Downloads\Face-recognition\Face-recognition\images\Hariyali.jpg")
known_person3_image = face_recognition.load_image_file(r"C:\Users\aasth\Downloads\Face-recognition\Face-recognition\images\nalla.jpg")
known_person4_image = face_recognition.load_image_file(r"C:\Users\aasth\Downloads\Face-recognition\Face-recognition\images\papa.jpg")

# Generate face encodings
known_person1_encoding = face_recognition.face_encodings(known_person1_image)[0]
known_person2_encoding = face_recognition.face_encodings(known_person2_image)[0]
known_person3_encoding = face_recognition.face_encodings(known_person3_image)[0]
known_person4_encoding = face_recognition.face_encodings(known_person4_image)[0]

# Add encodings to list
known_face_encodings.extend([
    known_person1_encoding,
    known_person2_encoding,
    known_person3_encoding,
    known_person4_encoding
])

# Add names to list
known_face_names.extend([
    "Aastha",
    "Ananya",
    "Suraj",
    "Papa"
])

# Initialize webcam
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
        
    # Convert to grayscale for emotion detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces using face_recognition (for identity recognition)
    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    
    # Process each detected face
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Identity Recognition
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        
        # Emotion Detection
        roi_gray = gray[top:bottom, left:right]
        try:
            roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
            
            if np.sum([roi_gray]) != 0:
                roi = roi_gray.astype('float') / 255.0
                roi = img_to_array(roi)
                roi = np.expand_dims(roi, axis=0)
                prediction = emotion_classifier.predict(roi)[0]
                emotion = emotion_labels[prediction.argmax()]
                
                # Draw rectangle and labels
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 255), 2)
                
                # Display name
                cv2.putText(frame, name, (left, top - 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
                # Display emotion
                cv2.putText(frame, emotion, (left, top - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
        except cv2.error:
            pass
    
    # Display the resulting frame
    cv2.imshow('Face Recognition and Emotion Detection', frame)
    
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()