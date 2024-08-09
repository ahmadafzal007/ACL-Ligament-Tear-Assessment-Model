# ACL Ligament Tear Assessment

![Demo Video](https://img.shields.io/badge/Streamlit-Powered-red)
![Deep Learning](https://img.shields.io/badge/DeepLearning-TensorFlow-orange)

This repository contains the code for a Streamlit-based application that utilizes deep learning models to assess ACL (Anterior Cruciate Ligament) tears and determine the grade of the tear based on MRI images using Convolutional Neural Networks (CNN).

## 📹 Demo Video

Watch the demo of our application on YouTube:

[![Watch the video](https://img.youtube.com/vi/bdXmeM4_DGw/0.jpg)](https://www.youtube.com/watch?v=bdXmeM4_DGw)

## 📝 Project Overview

Anterior Cruciate Ligament (ACL) tears are one of the most common injuries among athletes. Accurately diagnosing and grading the severity of the tear is crucial for effective treatment. This project leverages deep learning, specifically Convolutional Neural Networks (CNN), to assist medical professionals in the assessment of ACL tears by analyzing MRI scans.

### Key Features:
- **Upload MRI Images**: Users can upload MRI images of the knee for analysis.
- **Deep Learning Models**: The application uses two different CNN models trained on separate datasets to predict the presence and grade of an ACL tear.
- **Real-time Analysis**: Results from both models are provided in real-time, with predictions and grades displayed in an easy-to-understand format.
- **MRI Scan Analysis**: The CNN models are trained to accurately analyze MRI scans and provide insights into ACL tear detection.
- **User-friendly Interface**: Built using Streamlit, the application is intuitive and accessible for users without deep technical expertise.

## 📊 Datasets Used

The following datasets were used to train the CNN models in this project:

1. **kneeMRIdataset**  
   - **Dataset Link**: [kneeMRIdataset on Kaggle](https://www.kaggle.com/datasets/sohaibanwaar1203/kneemridataset)
   - **Description**: This dataset contains MRI images of knees, which are labeled for ACL tear presence and severity.

2. **MRNet v1 dataset**  
   - **Dataset Link**: [MRNet v1 dataset on Kaggle](https://www.kaggle.com/code/basel99/automated-interpretation-of-mri-images/input)
   - **Description**: The MRNet dataset contains knee MRI scans that are labeled for various abnormalities, including ACL tears.

## 🚀 Getting Started

### Prerequisites

- Python 3.x
- Streamlit
- TensorFlow
- OpenCV
- Numpy

### Installation

Clone the repository:

```bash
git clone git@github.com:ahmadafzal007/ACL-Ligament-Tear-Assessment-Model.git
cd ACL-Ligament-Tear-Assessment
