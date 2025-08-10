import cv2
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model('models/asl_model.h5')
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']  # Adjust per dataset

# Input video
video_path = "input_video.mp4"
cap = cv2.VideoCapture(video_path)

# Output video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('processed_video.avi', fourcc, 20.0, (640, 480))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (64, 64)) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction[0])

    cv2.putText(frame, f'Predicted: {labels[predicted_class]}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    out.write(frame)

cap.release()
out.release()
