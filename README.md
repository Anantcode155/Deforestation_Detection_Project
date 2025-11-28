# Deforestation_Detection_Project
This project focuses on detecting deforestation from satellite imagery using Convolutional Neural Networks (CNNs) techniques. The main goal is to support sustainability efforts by identifying regions where forest cover has been lost over time.

The system takes satellite images of land areas and classifies them into two categories:

🌳 Forest

🏜️ Deforested

By analyzing these images, the model can help environmental organizations, researchers, and policymakers monitor forest health and track changes in vegetation over time.

Commad to execute the project:
>>python main.py


🌿 Deforestation Detection using Deep Learning

🧠 Week 1 – Design Phase Summary
Problem Statement

Deforestation poses a major threat to biodiversity, climate stability, and ecosystem balance. Manual monitoring using satellite or aerial images is time-consuming, error-prone, and lacks scalability.
To address this issue, the project aims to develop an AI-powered Deforestation Detection System capable of automatically identifying deforested regions from images using deep learning models.

💡 Solution Approach

An intelligent image classification model was planned using Convolutional Neural Networks (CNN) to distinguish between Forest (Tree Cover) and Deforested (No Trees) areas.
The system design includes data preprocessing, model training using Teachable Machine, and deployment using TensorFlow for backend prediction.

🗂️ Dataset Information

Dataset Name: Deforestation Detection Dataset

Source: Custom satellite imagery and open datasets (via Teachable Machine)

Description: Contains forested and non-forested images for binary classification.

Purpose: To automate forest cover analysis and aid environmental monitoring.

🧱 Design Activities

Identified dataset categories: Forest (Trees) and Deforested (No Trees).

Planned preprocessing: resizing, normalization, and augmentation.

Selected TensorFlow 2.20 for model integration and Python 3.13 as the environment.

Designed model export in SavedModel format for backend use.

Outcome:
Week 1 successfully completed the problem definition, data collection, and system design, establishing the base for model implementation in the next phase.

💻 Week 2 – Implementation Phase Summary
⚙️ Implementation Overview

During Week 2, the designed deforestation detection model was implemented using TensorFlow and Teachable Machine. The trained model was exported and integrated into a Python backend for image-based predictions.

🧩 Implementation Steps

Trained model using Teachable Machine with two classes: Deforested and Forest.

Exported model in TensorFlow SavedModel format and loaded using tf.keras.layers.TFSMLayer.

Preprocessed test images (resizing to 224×224, normalization).

Implemented Python script for prediction and class mapping to descriptive labels.

Verified correct predictions for both forested and deforested test images.

📊 Results

Achieved accurate binary classification between Deforested and Forest regions.

Model demonstrated reliable real-time predictions on sample images.

Improved code to display class names and confidence scores.

🧾 Files Added

predict.py – Backend prediction script using TensorFlow.

model/ – Contains the exported SavedModel and labels.txt.

test_images/ – Folder for input testing images.

Outcome:
Week 2 completed the model integration and testing phase, achieving correct classification results and a working backend setup for deforestation detection.

🌍 Future Enhancements (Planned)

Develop a frontend web interface for users to upload satellite or drone images.

Display predictions dynamically with confidence scores and visual feedback.

Add map-based visualization of deforested regions.

Integrate REST API for seamless communication between frontend and backend.
