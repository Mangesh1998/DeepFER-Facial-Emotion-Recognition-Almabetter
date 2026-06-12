# DeepFER: Facial Emotion Recognition Using Deep Learning

DeepFER is a robust and efficient system designed to detect and classify human emotions from facial expressions in real-time. This project leverages Convolutional Neural Networks (CNNs), Transfer Learning (using pre-trained ResNet50V2 and EfficientNetV2B0 architectures), and advanced computer vision algorithms to achieve high accuracy and latency-optimized performance.

The system classifies facial expressions into **7 distinct emotion categories**:
*   😠 **Angry**
*   🤢 **Disgust**
*   😨 **Fear**
*   😊 **Happy**
*   😐 **Neutral**
*   😢 **Sad**
*   😮 **Surprise**

---

## 🚀 Key Features & Optimizations

*   **Custom CNN & Transfer Learning**: Implements a custom 4-block CNN built from scratch alongside fine-tuned `ResNet50V2` and `EfficientNetV2B0` transfer models.
*   **Dual-Detector Face Crop Pipeline**: Utilizes a fast OpenCV **Haar Cascade** for face detection, with a head-focused **YOLOv8** (`yolov8n.pt`) crop fallback to ensure detection robustness.
*   **⚡ CLAHE Lighting Normalization**: Integrated adaptive histogram equalization (CLAHE) to equalize uneven illumination and shadows on face crops, enhancing emotion classification stability under different lighting conditions.
*   **⚖️ Prior Probability Scaling**: Mitigates dataset class imbalances (such as the high ratio of Happy/Neutral samples vs. Disgust/Fear) by recalibrating predictions using training class weights.
*   **🔄 Temporal Smoothing**: Features a frame-rolling prediction queue (`deque`) to smooth output labels over time, preventing classification flickering in real-time webcam video feeds.
*   **🖥️ Streamlit Web Dashboard**: A user-friendly web interface allowing both static image uploads and live webcam snapshot classifications with interactive probability bar charts.

---

## 📁 Repository Structure

*   `models.ipynb`: Training pipeline including data loading, augmentation generators, model definition, training loops, and evaluation metrics (accuracy, precision, recall, F1-score).
*   `realtime_emotion_recognition.ipynb`: Script to run the emotion recognition pipeline on live webcam feeds, video files, or test images using temporal smoothing.
*   `streamlit_dashboard.py`: Interactive web dashboard application for uploading photos and running webcam snapshot inference.
*   `models/best_model.h5`: The saved trained weights file loaded by the web app and webcam script.
*   `yolov8n.pt`: YOLOv8 face detector weights used as a fallback detector.
*   `Technical_Documentation_ABSM_Project5.docx`: Stakeholder report and technical specifications guide.

---

## 🛠️ Installation & Setup

1.  **Clone the Repository**:
    ```bash
    git clone https://github.com/Mangesh1998/DeepFER-Facial-Emotion-Recognition-Almabetter.git
    cd DeepFER-Facial-Emotion-Recognition-Almabetter
    ```

2.  **Install Dependencies**:
    Ensure you have Python 3.10+ installed. Install the required libraries:
    ```bash
    pip install tensorflow opencv-python ultralytics streamlit pandas numpy scikit-learn pillow
    ```
    *Note: If you encounter binary compatibility warnings with pandas/scikit-learn, downgrade NumPy to 1.x:*
    ```bash
    pip install "numpy<2.0.0"
    ```

3.  **Download the Dataset**:
    Keep your dataset in an `images` folder at the root of the project with the following structure:
    ```
    images/
    ├── train/
    │   ├── angry/
    │   ├── disgust/
    │   └── ... (7 emotions)
    └── validation/
        ├── angry/
        ├── ... (7 emotions)
        
    ```

---

## 💻 How to Run

### 1. Training the Models
Open the Jupyter notebook `models.ipynb` and run all cells. It will load the images from the `images/` directory, apply data augmentations, train the three model tracks (Custom CNN, ResNet50V2, and EfficientNetV2B0), output validation classification reports, and save the best-performing model as `models/best_model.h5`.

### 2. Streamlit Web Dashboard
Launch the web interface locally by running:
```bash
python -m streamlit run streamlit_dashboard.py
```
This will start a local server at `http://localhost:8501`. Here you can upload files, toggle CLAHE/Prior-scaling optimizations in the sidebar, and test webcam snapshots.

### 3. Real-Time Webcam Inference
Open `realtime_emotion_recognition.ipynb` and run the webcam cells. It will activate your webcam, detect your face using Haar Cascade/YOLO, apply temporal smoothing, and draw classification bounding boxes and probability bars. Press `q` to exit.
