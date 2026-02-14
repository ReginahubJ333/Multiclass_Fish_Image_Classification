🐟 Multiclass Fish Image Classification
📌 Project Overview

This project focuses on building a deep learning model to classify fish images into 11 different categories. A custom Convolutional Neural Network (CNN) was developed as a baseline model and compared with a transfer learning approach using EfficientNetB0.

The best-performing model was deployed as a Streamlit web application for real-time predictions.

🎯 Problem Statement

To develop an accurate multiclass image classification model capable of identifying fish species from images and deploy it as an interactive web application for end users.

🧠 Approach
1️⃣ Data Preprocessing

Images resized to 224×224
Data augmentation applied (rotation, zoom, flipping)
Pixel normalization performed
Class imbalance handled using class weights

2️⃣ Model Training

Built a CNN model from scratch
Implemented transfer learning using EfficientNetB0 (pretrained on ImageNet)
Used early stopping to prevent overfitting
Compared performance across models

3️⃣ Model Evaluation

Accuracy
Precision
Recall
F1-score
Confusion Matrix
Validation vs Training accuracy plots

4️⃣ Deployment

Saved best-performing model (.h5)
Built Streamlit web application
Enabled real-time image upload and prediction

📂 Dataset

11 Fish Categories
Training Images: 6225
Validation Images: 1092
Test Images: 3187
Structured into train/, val/, and test/ folders

📊 Model Performance
Model	Validation Accuracy	Test Accuracy
CNN (Baseline)	80.6%	87.67%
EfficientNetB0	~99%	99.46%
Key Insight:

Transfer learning significantly improved performance by leveraging pretrained ImageNet feature representations.

⚠ Limitations

Minority class showed slightly lower precision due to data imbalance.
Performance may vary with low-quality or blurry images.
Larger dataset could further improve robustness.

🚀 How to Run the Application
1️⃣ Clone the repository
git clone https://github.com/YOUR_USERNAME/Multiclass_Fish_Image_Classification.git
cd Multiclass_Fish_Image_Classification

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run Streamlit app
streamlit run app.py

4️⃣ Open browser

Go to:

http://localhost:8501

Upload a fish image and view prediction.

🛠️ Technologies Used

Python
TensorFlow / Keras
EfficientNetB0 (Transfer Learning)
Streamlit
NumPy
Matplotlib
Scikit-learn
Seaborn

📈 Project Highlights

✔ Implemented CNN from scratch
✔ Applied transfer learning
✔ Handled class imbalance
✔ Achieved 99.46% test accuracy
✔ Built interactive deployment app
✔ Followed clean coding standards

📷 Application Demo

(![App Screenshot](app_screenshot.png))

👩‍💻 Author

Regina
Deep Learning & Data Science Enthusiast
