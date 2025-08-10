import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load model
model = load_model('models/asl_model.h5')

# Load validation data
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
validation_generator = datagen.flow_from_directory(
    'dataset', target_size=(64, 64),
    batch_size=32, class_mode='categorical', subset='validation'
)

# Predictions
val_steps = validation_generator.samples // validation_generator.batch_size
predictions = model.predict(validation_generator, steps=val_steps)
true_labels = validation_generator.classes
predicted_labels = np.argmax(predictions, axis=1)

class_labels = list(validation_generator.class_indices.keys())

# Confusion matrix
conf_matrix = confusion_matrix(true_labels[:len(predicted_labels)], predicted_labels)
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=class_labels)
disp.plot(cmap=plt.cm.Blues, xticks_rotation=45)
plt.title("Confusion Matrix")
plt.show()

# Classification report
print(classification_report(true_labels[:len(predicted_labels)], predicted_labels, target_names=class_labels))
