🖐 American Sign Language (ASL) Recognition using Deep Learning
📌 About the Project
This project is a real-time American Sign Language recognition system built using Convolutional Neural Networks (CNN) and OpenCV.
It is capable of:

Training on a custom ASL dataset.

Recognizing hand gestures in real-time using a webcam.

Processing pre-recorded videos to detect signs.

Evaluating model performance using classification reports, confusion matrices, and accuracy/loss plots.

This tool can assist in bridging the communication gap between the hearing and hearing-impaired communities, and can be extended to support other sign languages or gesture-based control systems.

🚀 Features
📷 Real-Time Detection – Recognizes ASL gestures using your webcam.

🎥 Video File Processing – Annotates pre-recorded videos with predictions.

🧠 Custom CNN Architecture – Optimized for image classification.

🔄 Data Augmentation – Improves accuracy and generalization.

📊 Evaluation Tools – Generates confusion matrix and classification report.

💾 Model Saving & Loading – Trained models can be reused without retraining.

🛠 Tech Stack
Python 3

TensorFlow / Keras – Deep learning model building & training

OpenCV – Image & video capture/processing

Matplotlib – Data visualization

Scikit-learn – Model evaluation metrics

Google Colab – Training environment & Drive integration

📂 Dataset Structure
The dataset should be arranged as follows:

css
Copy
Edit
asl_dataset/
│── A/
│   ├── img1.jpg
│   ├── img2.jpg
│── B/
│   ├── img1.jpg
│── ...
│── Z/
Each folder name represents the gesture label.

Images are resized to 64x64 pixels during preprocessing.

🧠 Model Architecture
The CNN consists of:

Input Layer: 64×64×3 RGB images

3 Convolutional Layers with ReLU activation

MaxPooling after each convolutional block

Fully Connected Dense Layer (128 neurons)

Dropout Layer (0.5 rate to prevent overfitting)

Softmax Output Layer for multi-class classification


📊 Example Output
Confusion Matrix


Accuracy and Loss Curves


📈 Results
Training Accuracy: 83% (depends on dataset)

Validation Accuracy: ~84%

Test Accuracy: ~86%

📜 License
This project is licensed under the MIT License – feel free to use and modify it.

🤝 Acknowledgements
ASL Dataset source:

TensorFlow & Keras for deep learning framework

OpenCV for computer vision tasks