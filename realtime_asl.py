import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model
model = tf.keras.models.load_model('models/asl_model.h5')

# Labels (ensure same order as training)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory(
    'dataset', target_size=(64, 64),
    batch_size=32, class_mode='categorical', subset='training'
)
labels = list(train_generator.class_indices.keys())

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (64, 64))
    img = np.expand_dims(img, axis=0) / 255.0

    prediction = model.predict(img)
    predicted_class = np.argmax(prediction[0])

    cv2.putText(frame, f'Predicted: {labels[predicted_class]}', (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('ASL Recognition', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
